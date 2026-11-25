from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

from preprocessing import IDENTIFIERS, make_model_pipeline, traffic_features
from train_models import classifier_factories, load_features


FALSE_ALARM_BUDGETS = (0.01, 0.05)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a binary attack detector.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cuda", action="store_true", help="Use native cuML estimators on CUDA.")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--predict-test", action="store_true", help="Score the original holdout.")
    evaluation.add_argument("--chronological", action="store_true", help="Stress-test on later development rows.")
    args = parser.parse_args()

    data_dir = args.data_dir
    summary = json.loads((data_dir / "data_summary.json").read_text(encoding="utf-8"))
    if summary["target"] != "attack_type":
        raise ValueError("Use the multiclass splits to retain row alignment with the original experiment.")
    device, to_cuda, factories = classifier_factories(args.cuda, summary["seed"])
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    if args.chronological:
        train, validation, later = chronological_splits(train, validation)
    models, validation_results = fit_detectors(train, validation, factories, to_cuda)
    selected = select_detectors(validation_results)

    output_name = "detector_cuda" if args.cuda else "detector"
    split_name = "chronological" if args.chronological else "original"
    output = data_dir / f"{output_name}_{split_name}"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_results).to_csv(output / "validation_metrics.csv", index=False)
    for name in {result["model"] for result in selected}:
        with (output / f"{name}.pkl").open("wb") as stream:
            pickle.dump(models[name], stream)

    settings = {"device": device, "split_seed": summary["seed"], "split": split_name, "selected": selected}
    if args.chronological:
        settings["validation_starts"] = pd.to_datetime(validation[0]["timestamp"], utc=True).min().isoformat()
        settings["test_starts"] = pd.to_datetime(later[0]["timestamp"], utc=True).min().isoformat()
        export_test_results(output, later, selected)
    if args.predict_test:
        export_test_results(output, load_split(data_dir, "test"), selected)
    (output / "run.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs: {output.resolve()}")


def fit_detectors(train, validation, factories, to_cuda):
    train_features, train_types = train
    validation_features, validation_types = validation
    train_labels = train_types.ne("benign").to_numpy(dtype=np.int32)
    validation_labels = validation_types.ne("benign").to_numpy()
    weights = compute_sample_weight("balanced", train_labels)

    models = {}
    validation_results = []
    for name, classifier, feature_transform in detector_candidates(factories):
        model = make_model_pipeline(classifier, to_cuda, feature_transform)
        fit_features = train_features if feature_transform else train_features.drop(columns=IDENTIFIERS)
        model.fit(fit_features, train_labels, classifier__sample_weight=weights)
        models[name] = model
        scores = attack_scores(model, validation_features)
        for budget in FALSE_ALARM_BUDGETS:
            threshold = select_threshold(validation_labels, scores, budget)
            result = detection_metrics(validation_labels, scores, threshold)
            assert result["false_alarm_rate"] <= budget
            validation_results.append({"model": name, "budget": budget, **result})

    return models, validation_results


def select_detectors(validation_results: list[dict]) -> list[dict]:
    return [
        max(
            (result for result in validation_results if result["budget"] == budget),
            key=lambda result: (result["recall"], -result["false_alarm_rate"], result["average_precision"]),
        )
        for budget in FALSE_ALARM_BUDGETS
    ]
def load_split(directory: Path, name: str) -> tuple[pd.DataFrame, pd.Series]:
    features = load_features(directory / f"X_{name}.csv")
    attack_types = pd.read_csv(directory / f"y_{name}.csv")["attack_type"]
    if len(features) != len(attack_types) or attack_types.isna().any():
        raise ValueError(f"Invalid attack-type labels in the {name} split.")
    if attack_types.eq("benign").nunique() != 2:
        raise ValueError(f"The {name} split needs both benign and attack rows.")
    return features, attack_types


def chronological_splits(train, validation):
    features = pd.concat([train[0], validation[0]], ignore_index=True)
    attack_types = pd.concat([train[1], validation[1]], ignore_index=True)
    timestamps = pd.to_datetime(features["timestamp"], errors="raise", utc=True)
    if timestamps.isna().any():
        raise ValueError("Chronological evaluation requires a timestamp on every row.")
    ordered = timestamps.sort_values()
    validation_start = ordered.iloc[int(len(ordered) * 0.70)]
    test_start = ordered.iloc[int(len(ordered) * 0.85)]
    masks = [
        timestamps < validation_start,
        timestamps.ge(validation_start) & timestamps.lt(test_start),
        timestamps >= test_start,
    ]
    splits = [(features.loc[mask], attack_types.loc[mask]) for mask in masks]
    if any(labels.eq("benign").nunique() != 2 for _, labels in splits):
        raise ValueError("Each chronological split needs both benign and attack rows.")
    return splits


def detector_candidates(factories):
    yield "forest_existing_features", factories["random_forest"](), None
    yield "logistic_traffic_features", factories["logistic_regression"](), traffic_features
    for depth in (6, 12):
        forest = factories["random_forest"]().set_params(max_depth=depth)
        yield f"forest_traffic_features_depth_{depth}", forest, traffic_features


def attack_scores(model, features: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(features))[:, 1]


def select_threshold(labels: np.ndarray, scores: np.ndarray, budget: float) -> float:
    false_alarm, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    feasible = false_alarm <= budget
    threshold = thresholds[feasible][recall[feasible].argmax()]
    return float(threshold) if np.isfinite(threshold) else float(np.nextafter(1.0, np.inf))


def detection_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    flagged = scores >= threshold
    caught = int(np.count_nonzero(flagged & labels))
    false_alarms = int(np.count_nonzero(flagged & ~labels))
    attacks = int(labels.sum())
    benign = len(labels) - attacks
    return {
        "threshold": threshold,
        "average_precision": float(average_precision_score(labels, scores)),
        "recall": caught / attacks,
        "precision": caught / max(caught + false_alarms, 1),
        "false_alarm_rate": false_alarms / benign,
        "attacks_detected": caught,
        "attack_rows": attacks,
        "benign_flagged": false_alarms,
        "benign_rows": benign,
    }


def export_test_results(output: Path, split, selected: list[dict]) -> None:
    features, attack_types = split
    labels = attack_types.ne("benign").to_numpy()
    predictions = pd.DataFrame({"row_index": features.index, "attack_type": attack_types})
    results = []
    for result in selected:
        name, budget, threshold = (result[field] for field in ("model", "budget", "threshold"))
        with (output / f"{name}.pkl").open("rb") as stream:
            model = pickle.load(stream)
        scores = attack_scores(model, features)
        predictions[f"score_{budget:.0%}"] = scores
        predictions[f"flagged_{budget:.0%}"] = scores >= threshold
        metrics = detection_metrics(labels, scores, threshold)
        results.append({"model": name, "validation_budget": budget, **metrics})
        print(f"{budget:.0%} validation budget: {metrics['attacks_detected']}/{metrics['attack_rows']} attacks, "
              f"{metrics['benign_flagged']}/{metrics['benign_rows']} benign flagged", flush=True)
    pd.DataFrame(results).to_csv(output / "test_metrics.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
