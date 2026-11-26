from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from preprocessing import IDENTIFIERS
from train_cascade import cascade_predictions
from train_detector import (
    attack_scores,
    chronological_splits,
    detection_metrics,
    fit_binary_candidates,
    load_split,
    select_detectors,
)
from train_models import classifier_factories, make_xgboost, verify_xgboost_cuda


FEATURE_EXCLUSIONS = {
    "without_ids": IDENTIFIERS,
    "without_ids_user_agent_url": [*IDENTIFIERS, "user_agent", "url"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare random forest and XGBoost as binary attack gates.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir
    summary = json.loads((data_dir / "data_summary.json").read_text(encoding="utf-8"))
    if summary["target"] != "attack_type":
        raise ValueError("Prepare the multiclass attack_type splits before comparing gates.")
    device, to_cuda, factories = classifier_factories(args.cuda, summary["seed"])
    train = load_split(data_dir, "train")
    validation = load_split(data_dir, "validation")
    earlier, middle, later = chronological_splits(train, validation)
    candidates = comparison_candidates(factories["random_forest"], summary["seed"], args.cuda)
    backend = "_cuda" if args.cuda else ""
    output = data_dir / f"boosting_comparison{backend}"
    output.mkdir(parents=True, exist_ok=True)
    evaluations = []
    all_validation = []
    for split, fit_split, validation_split in (
        ("original", train, validation),
        ("chronological", earlier, middle),
    ):
        models, validation_rows, selected = tune_gates(fit_split, validation_split, candidates, to_cuda)
        all_validation.extend(dict(row, split=split) for row in validation_rows)
        evaluation = load_split(data_dir, "test") if split == "original" else later
        specialist_path = data_dir / f"cascade{backend}" / f"{split}_specialist.pkl"
        with specialist_path.open("rb") as stream:
            specialist = pickle.load(stream)
        results, predictions = evaluate_gates(evaluation, selected, models, specialist)
        evaluations.extend(dict(row, split=split, device=device) for row in results)
        predictions.to_csv(output / f"{split}_predictions.csv", index=False)

    pd.DataFrame(all_validation).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame(evaluations).to_csv(output / "test_metrics.csv", index=False)
    print(f"Outputs: {output.resolve()}", flush=True)


def comparison_candidates(forest_factory, seed: int, cuda: bool):
    return {
        "random_forest": (
            (4, 8, 16),
            lambda depth: forest_factory().set_params(max_depth=depth),
        ),
        "xgboost": (
            (2, 4, 6),
            lambda depth: make_xgboost(depth, seed, cuda),
        ),
    }


def tune_gates(train, validation, candidates, to_cuda):
    models = {}
    validation_rows = []
    selected = []
    for feature_set, excluded in FEATURE_EXCLUSIONS.items():
        for family, (depths, factory) in candidates.items():
            family_rows = []
            variants = (
                (f"{family}_{feature_set}_depth_{depth}", factory(depth), None, excluded)
                for depth in depths
            )
            for name, model, operating_points in fit_binary_candidates(train, validation, variants, to_cuda):
                classifier = model.named_steps["classifier"]
                if family == "xgboost" and to_cuda is not None:
                    verify_xgboost_cuda(classifier)
                models[name] = model
                family_rows.extend(dict(
                    point, family=family, feature_set=feature_set, max_depth=classifier.max_depth
                ) for point in operating_points)
            winners = select_detectors(family_rows)
            for row in family_rows:
                row["selected"] = row in winners
            selected.extend(winners)
            validation_rows.extend(family_rows)
    return models, validation_rows, selected


def evaluate_gates(evaluation, selected, models, specialist):
    features, attack_types = evaluation
    labels = attack_types.ne("benign").to_numpy()
    actual = attack_types.to_numpy()
    predictions = pd.DataFrame({"row_index": features.index, "attack_type": attack_types})
    results = []
    for setting in selected:
        scores = attack_scores(models[setting["model"]], features)
        threshold = setting["threshold"]
        flagged = scores >= threshold
        predicted = cascade_predictions(features, flagged, specialist["model"], specialist["classes"])
        suffix = f"{setting['model']}_{setting['budget']:.0%}"
        predictions[f"score_{suffix}"] = scores
        predictions[f"prediction_{suffix}"] = predicted
        results.append({
            "model": setting["model"], "family": setting["family"], "feature_set": setting["feature_set"],
            "max_depth": setting["max_depth"], "validation_budget": setting["budget"],
            **detection_metrics(labels, scores, threshold),
            "attacks_typed_correctly": int(np.count_nonzero(labels & (predicted == actual))),
        })
    return results, predictions


if __name__ == "__main__":
    main()
