from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from prepare_data import TARGET_COLUMNS
from preprocessing import IDENTIFIERS, make_model_pipeline


def classifier_factories(cuda: bool, seed: int):
    if cuda:
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier as CudaForest
        from cuml.linear_model import LogisticRegression as CudaLogisticRegression
        from cuml.svm import SVC as CudaSVC

        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"].decode()
        print(f"CUDA device: {device}", flush=True)
        return device, cp.asarray, {
            "logistic_regression": lambda: CudaLogisticRegression(
                max_iter=1000, output_type="numpy"
            ),
            "svm": lambda: CudaSVC(
                class_weight="balanced", decision_function_shape="ovo", output_type="numpy"
            ),
            "random_forest": lambda: CudaForest(
                n_estimators=100, max_depth=16, random_state=seed, output_type="numpy"
            ),
        }

    return "CPU", None, {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "svm": lambda: SVC(class_weight="balanced"),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=16, random_state=seed),
    }


def load_features(path: Path) -> pd.DataFrame:
    features = pd.read_csv(path)
    if TARGET_COLUMNS.intersection(features.columns):
        raise ValueError(f"Unexpected label columns in {path}.")
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multiclass classifiers and export predictions.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--cuda", action="store_true", help="Use native cuML estimators on CUDA.")
    parser.add_argument("--predict-test", action="store_true", help="Export test predictions without reading test labels.")
    args = parser.parse_args()

    data_dir = args.data_dir
    output_name = "training_cuda" if args.cuda else "training"
    output_dir = data_dir / (f"{output_name}_with_test" if args.predict_test else output_name)
    summary = json.loads((data_dir / "data_summary.json").read_text(encoding="utf-8"))
    if summary["target"] != "attack_type":
        raise ValueError("Prepare the multiclass attack_type splits before training.")
    seed = summary["seed"]
    X_train = load_features(data_dir / "X_train.csv")
    X_validation = load_features(data_dir / "X_validation.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")["attack_type"]
    y_validation = pd.read_csv(data_dir / "y_validation.csv")["attack_type"]
    if len(X_train) != len(y_train) or len(X_validation) != len(y_validation):
        raise ValueError("Feature and target row counts must match within each split.")
    validation_classes = set(y_validation)
    if validation_classes != set(y_train):
        raise ValueError("Train and validation splits must contain the same attack types.")
    classes = np.array(sorted(validation_classes))
    train_labels = np.searchsorted(classes, y_train).astype(np.int32)
    balanced_weights = compute_sample_weight("balanced", train_labels)
    device, to_cuda, factories = classifier_factories(args.cuda, seed)

    input_files = ["X_train.csv", "y_train.csv", "X_validation.csv", "y_validation.csv"]
    if args.predict_test:
        X_test = load_features(data_dir / "X_test.csv")
        input_files.append("X_test.csv")

    validation_predictions = pd.DataFrame({"row_index": X_validation.index, "attack_type": y_validation})
    test_predictions = pd.DataFrame({"row_index": X_test.index}) if args.predict_test else None
    model_runs = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [("most_frequent", [], DummyClassifier(strategy="most_frequent"), None)]
    for suffix, excluded in (("without_ids", IDENTIFIERS), ("with_ids", [])):
        for name, factory in factories.items():
            weights = None if name == "svm" else balanced_weights
            experiments.append((f"{name}_{suffix}", excluded, factory(), weights))

    for name, excluded, classifier, weights in experiments:
        X_fit = X_train.drop(columns=excluded)
        is_baseline = isinstance(classifier, DummyClassifier)
        if is_baseline:
            model = classifier.fit(X_fit, train_labels)
        else:
            model = make_model_pipeline(classifier, to_cuda)
            fit_params = {} if weights is None else {"classifier__sample_weight": weights}
            model.fit(X_fit, train_labels, **fit_params)

        predictions = classes[np.asarray(model.predict(X_validation), dtype=int).ravel()]
        validation_predictions[name] = predictions
        if test_predictions is not None:
            test_predictions[name] = classes[np.asarray(model.predict(X_test), dtype=int).ravel()]
        scores = {
            "macro_f1": f1_score(y_validation, predictions, average="macro", zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y_validation, predictions),
            "accuracy": accuracy_score(y_validation, predictions),
        }
        artifact = f"{name}.pkl"
        with (output_dir / artifact).open("wb") as stream:
            pickle.dump(model, stream)
        model_runs[name] = {
            "artifact": artifact,
            "backend": "scikit-learn CPU" if is_baseline or not args.cuda else "cuML CUDA",
            "feature_columns": list(X_fit.columns),
            "balanced_sample_weights": weights is not None,
            "parameters": classifier.get_params(),
            "validation": scores,
        }
        print(f"{name:36s} macro-F1 {scores['macro_f1']:.4f}  balanced accuracy {scores['balanced_accuracy']:.4f}", flush=True)

    validation_predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    if test_predictions is not None:
        test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    settings = {
        "target": summary["target"],
        "split_seed": seed,
        "split_fractions": summary["split_fractions"],
        "classes": classes.tolist(),
        "device": device,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy")},
        "input_sha256": {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in input_files},
        "models": model_runs,
    }
    if args.cuda:
        settings["packages"].update({name: version(name) for name in ("cuml-cu13", "cupy-cuda13x")})
    (output_dir / "run.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
