from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import IDENTIFIERS, make_model_pipeline
from train_detector import attack_scores, chronological_splits, load_split
from train_models import classifier_factories


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate a two-stage attack classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    summary = json.loads((args.data_dir / "data_summary.json").read_text(encoding="utf-8"))
    if summary["target"] != "attack_type":
        raise ValueError("Prepare the multiclass attack_type splits before training the cascade.")
    device, to_cuda, factories = classifier_factories(args.cuda, summary["seed"])
    train = load_split(args.data_dir, "train")
    validation = load_split(args.data_dir, "validation")
    later_train, later_validation, later_test = chronological_splits(train, validation)
    backend = "_cuda" if args.cuda else ""
    output = args.data_dir / f"cascade{backend}"
    output.mkdir(parents=True, exist_ok=True)
    validation_results = []
    evaluation_results = []
    for split, fit_split, validation_split in (
        ("original", train, validation),
        ("chronological", later_train, later_validation),
    ):
        detector_dir = args.data_dir / f"detector{backend}_{split}"
        gates = load_gates(detector_dir, validation_split)
        chosen_model, classes, tuning_rows = tune_specialist(
            fit_split, validation_split, gates[0.05], factories, to_cuda
        )
        validation_results.extend(dict(row, split=split) for row in tuning_rows)
        evaluation = load_split(args.data_dir, "test") if split == "original" else later_test
        comparison = evaluate_split(fit_split, evaluation, chosen_model, classes, gates, factories, to_cuda)
        evaluation_results.extend(dict(row, split=split, device=device) for row in comparison)
        with (output / f"{split}_specialist.pkl").open("wb") as stream:
            pickle.dump({"model": chosen_model, "classes": classes}, stream)

    pd.DataFrame(validation_results).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(evaluation_results).to_csv(output / "test_metrics.csv", index=False)
    print(pd.DataFrame(evaluation_results).to_string(index=False))


def load_gates(directory: Path, validation):
    settings = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    features, attack_types = validation
    attacks = attack_types.ne("benign").to_numpy()
    gates = {}
    for entry in settings["selected"]:
        with (directory / f"{entry['model']}.pkl").open("rb") as stream:
            detector = pickle.load(stream)
        threshold = entry["threshold"]
        flagged = attack_scores(detector, features) >= threshold
        observed = (np.count_nonzero(flagged & attacks), np.count_nonzero(flagged & ~attacks))
        recorded = (entry["attacks_detected"], entry["benign_flagged"])
        if observed != recorded:
            raise ValueError(f"Threshold results differ from the recorded validation run in {directory}.")
        gates[entry["budget"]] = (detector, threshold)
    return gates


def tune_specialist(train, validation, gate, factories, to_cuda):
    train_features, train_types = train
    is_attack = train_types.ne("benign")
    attack_rows = train_features.loc[is_attack]
    attack_types = train_types.loc[is_attack]
    classes = np.array(sorted(attack_types.unique()))
    features, actual = validation
    detector, threshold = gate
    flagged = attack_scores(detector, features) >= threshold
    candidates = {
        **{f"logistic_c_{strength:g}": factories["logistic_regression"]().set_params(C=strength)
           for strength in (0.1, 1.0, 10.0)},
        **{f"forest_depth_{depth}": factories["random_forest"]().set_params(max_depth=depth)
           for depth in (4, 8, 16)},
    }
    models = {}
    rows = []
    for name, classifier in candidates.items():
        specialist = fit_classifier(attack_rows, attack_types, classes, classifier, to_cuda)
        predictions = cascade_predictions(features, flagged, specialist, classes)
        metrics = attribution_metrics(actual, predictions, classes)
        models[name] = specialist
        rows.append({"candidate": name, "attack_training_rows": len(attack_rows), **metrics})
    selected = max(rows, key=lambda row: (row["attack_type_macro_f1"], row["attacks_typed_correctly"]))
    for row in rows:
        row["selected"] = row is selected
    return models[selected["candidate"]], classes, rows


def fit_classifier(features, attack_types, classes, classifier, to_cuda):
    encoded = np.searchsorted(classes, attack_types).astype(np.int32)
    weights = compute_sample_weight("balanced", encoded)
    model = make_model_pipeline(classifier, to_cuda)
    return model.fit(features.drop(columns=IDENTIFIERS), encoded, classifier__sample_weight=weights)


def cascade_predictions(features, flagged, specialist, classes):
    predictions = np.full(len(features), "benign", dtype=object)
    if np.any(flagged):
        predictions[flagged] = classes[np.asarray(specialist.predict(features.iloc[flagged]), dtype=int).ravel()]
    return predictions


def evaluate_split(train, evaluation, specialist, attack_classes, gates, factories, to_cuda):
    train_features, train_types = train
    classes = np.array(sorted(train_types.unique()))
    forest = fit_classifier(train_features, train_types, classes, factories["random_forest"](), to_cuda)
    features, actual = evaluation
    def predict_forest(batch):
        return classes[np.asarray(forest.predict(batch), dtype=int).ravel()]

    rows = [{
        "model": "multiclass_forest", "validation_budget": np.nan,
        **attribution_metrics(actual, predict_forest(features), attack_classes),
        "single_record_median_ms": measure_single_record_latency(predict_forest, features),
    }]
    for budget, gate in sorted(gates.items()):
        def predict_cascade(batch):
            detector, threshold = gate
            flagged = attack_scores(detector, batch) >= threshold
            return cascade_predictions(batch, flagged, specialist, attack_classes)

        rows.append({
            "model": "two_stage_cascade", "validation_budget": budget,
            **attribution_metrics(actual, predict_cascade(features), attack_classes),
            "single_record_median_ms": measure_single_record_latency(predict_cascade, features),
        })
    return rows


def measure_single_record_latency(predict, features: pd.DataFrame) -> float:
    samples = features.sample(n=min(100, len(features)), random_state=42)
    predict(samples.iloc[:1])
    timings = []
    for index in range(len(samples)):
        record = samples.iloc[[index]]
        start = perf_counter()
        predict(record)
        timings.append((perf_counter() - start) * 1000)
    return median(timings)


def attribution_metrics(actual, predictions, attack_classes):
    is_attack = actual.ne("benign").to_numpy()
    flagged = predictions != "benign"
    correct = predictions == actual.to_numpy()
    attack_rows = int(is_attack.sum())
    benign_rows = len(actual) - attack_rows
    return {
        "attack_rows": attack_rows,
        "attacks_flagged": int((is_attack & flagged).sum()),
        "attacks_typed_correctly": int((is_attack & correct).sum()),
        "benign_rows": benign_rows,
        "benign_flagged": int((~is_attack & flagged).sum()),
        "false_alarm_rate": float((~is_attack & flagged).sum() / benign_rows),
        "attack_type_macro_f1": f1_score(actual, predictions, labels=attack_classes, average="macro", zero_division=0),
    }


if __name__ == "__main__":
    main()
