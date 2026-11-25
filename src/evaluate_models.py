from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def measure_latency(model, features, exported, classes, seed: int) -> dict[str, float]:
    def timed_predict(batch):
        start = perf_counter()
        prediction = np.asarray(model.predict(batch), dtype=int).ravel()
        return (perf_counter() - start) * 1000, classes[prediction]

    _, restored = timed_predict(features)
    np.testing.assert_array_equal(restored, exported)
    batches = [timed_predict(features)[0] for _ in range(5)]
    positions = np.random.default_rng(seed).choice(len(features), size=min(100, len(features)), replace=False)
    timed_predict(features.iloc[positions[:1]])
    singles = [timed_predict(features.iloc[[position]])[0] for position in positions]
    return {
        "batch_ms_per_record": median(batches) / len(features),
        "single_record_median_ms": median(singles),
        "single_record_p95_ms": float(np.percentile(singles, 95)),
    }


def save_matrix(matrix, support, classes, name, output_dir):
    normalized = matrix / support[:, None]
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    display = ConfusionMatrixDisplay(normalized, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, im_kw={"vmin": 0, "vmax": 1}
    )
    for cell, count in zip(display.text_.flat, matrix.flat):
        cell.set_text(str(count) if count else "")
    display.im_.colorbar.set_label("Fraction of actual class")
    ax.set_title(f"Held-out test · {name.replace('_', ' ')}")
    fig.savefig(output_dir / f"confusion_{name}.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(matrix, index=pd.Index(classes, name="actual"), columns=classes).to_csv(
        output_dir / f"confusion_{name}.csv"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a training run on the held-out test split.")
    parser.add_argument("run_dir", type=Path, help="Folder containing run.json and test_predictions.csv.")
    run_dir = parser.parse_args().run_dir
    settings = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    feature_file = run_dir.parent / "X_test.csv"
    if hashlib.sha256(feature_file.read_bytes()).hexdigest() != settings["input_sha256"]["X_test.csv"]:
        raise ValueError("Test features differ from the split used to generate predictions.")
    features = pd.read_csv(feature_file)
    truth = pd.read_csv(run_dir.parent / "y_test.csv")[settings["target"]]
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    np.testing.assert_array_equal(predictions["row_index"], np.arange(len(features)))
    classes = np.array(settings["classes"])
    if len(truth) != len(features) or set(truth) != set(classes):
        raise ValueError("The test labels do not match the expected rows and classes.")
    benign = np.flatnonzero(classes == "benign").item()

    output_dir = run_dir / "evaluation"
    output_dir.mkdir(exist_ok=True)
    model_results = []
    class_results = []
    for name, info in settings["models"].items():
        predicted = predictions[name]
        matrix = confusion_matrix(truth, predicted, labels=classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=classes, zero_division=0
        )
        true_positives = matrix.diagonal()
        false_alarm = (matrix.sum(axis=0) - true_positives) / (len(truth) - support)
        per_class = pd.DataFrame({
            "precision": precision, "recall": recall, "f1": f1,
            "false_alarm_rate": false_alarm, "support": support,
        })
        class_results.append(per_class.assign(model=name, attack_type=classes))
        result = {
            "model": name,
            "backend": info["backend"],
            "validation_macro_f1": info["validation"]["macro_f1"],
            "accuracy": true_positives.sum() / len(truth),
            "benign_false_alarm_rate": 1 - recall[benign],
            "benign_flagged_as_attack": support[benign] - true_positives[benign],
            "attacks_detected": len(truth) - support[benign] - (matrix[:, benign].sum() - true_positives[benign]),
            "attacks_classified_correctly": true_positives.sum() - true_positives[benign],
        }
        for metric in ("precision", "recall", "f1", "false_alarm_rate"):
            result[f"macro_{metric}"] = per_class[metric].mean()
            result[f"weighted_{metric}"] = np.average(per_class[metric], weights=support)
        with (run_dir / info["artifact"]).open("rb") as stream:
            model = pickle.load(stream)
        result.update(measure_latency(model, features, predicted, classes, settings["split_seed"]))
        model_results.append(result)
        save_matrix(matrix, support, classes, name, output_dir)
        print(f"{name:36s} test macro-F1 {result['macro_f1']:.4f}", flush=True)

    pd.DataFrame(model_results).to_csv(output_dir / "test_metrics.csv", index=False)
    pd.concat(class_results, ignore_index=True).to_csv(output_dir / "per_class_metrics.csv", index=False)
    print(f"Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
