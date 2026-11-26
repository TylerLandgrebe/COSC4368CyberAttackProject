import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import time
import pickle
import json
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

DATA_DIR = Path("data/processed")
TRAINING_DIR = DATA_DIR / "training"
RESULTS_DIR = Path("results")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")["attack_type"]

    X_val = pd.read_csv(DATA_DIR / "X_validation.csv")
    y_val = pd.read_csv(DATA_DIR / "y_validation.csv")["attack_type"]

    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv")["attack_type"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model_name, y_true, y_pred):
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    }

    print(f"\nResults for {model_name}")
    print("-" * 40)

    for metric, value in metrics.items():
        if metric != "model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred, zero_division=0
    ))

    return metrics

def plot_confusion_matrix(model_name, y_true, y_pred):
    labels = sorted(y_true.unique())
    filename = model_name.lower().replace(" ", "_")

    for normalize in [None, "true"]:
        fig, ax = plt.subplots(figsize=(12, 10))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=labels,
            normalize=normalize,
            cmap="Blues",
            xticks_rotation=45,
            values_format=".2f" if normalize else "d",
            ax=ax
        )

        suffix = "normalized" if normalize else "raw"

        ax.set_title(f"{model_name} - {suffix.title()} Confusion Matrix")

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIR / f"{filename}_{suffix}_cm.png",
            dpi=300
        )

        plt.close(fig)


def save_results(results):
    df = pd.DataFrame(results)

    output_path = RESULTS_DIR / "model_metrics.csv"

    df.to_csv(output_path, index=False, float_format="%.6f")

    print(f"\nResults saved to {output_path}")

    return df

def plot_model_comparison(df):
    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy"
    ]

    ax = df.set_index("model")[metrics].plot(
        kind="bar",
        figsize=(20, 9),
        width=0.8
    )

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.xticks(rotation=35, ha="right")
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v * 100:.1f}%" for v in container.datavalues],
            padding=3,
            fontsize=7
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "model_comparison.png",
        dpi=300
    )

    plt.close()

    print("Model comparison chart saved.")

def measure_inference_latency(model, X, repeats=10):
    times = []

    model.predict(X)

    for _ in range(repeats):
        start = time.perf_counter()

        model.predict(X)

        end = time.perf_counter()

        times.append(end - start)

    avg_time = sum(times) / len(times)

    total_ms = avg_time * 1000
    latency_ms = total_ms / len(X)

    return latency_ms

def plot_latency_comparison(df):
    fig, ax = plt.subplots(figsize=(14, 7))

    bars = ax.bar(
        df["model"],
        df["inference_latency_ms"]
    )

    ax.bar_label(bars, fmt="%.6f", padding=3)

    ax.set_title("Inference Latency Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Milliseconds per Sample")
    ax.set_ylim(bottom=0)
    ax.margins(y=0.15)

    plt.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "latency_comparison.png",
        dpi=300
    )

    plt.close(fig)

def plot_per_class_recall(predictions):
    y_true = predictions["attack_type"]
    labels = sorted(y_true.unique())

    model_columns = predictions.columns.drop(
        ["row_index", "attack_type"]
    )

    recall_results = {}

    for model_name in model_columns:
        y_pred = predictions[model_name]

        recall_results[model_name] = recall_score(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0
        )

    recall_df = pd.DataFrame(
        recall_results,
        index=labels
    )

    recall_df.to_csv(
        RESULTS_DIR / "per_class_recall.csv"
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(
        recall_df.values,
        cmap="Blues",
        vmin=0,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(model_columns)))
    ax.set_xticklabels(
        model_columns,
        rotation=35,
        ha="right"
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(model_columns)):
            value = recall_df.iloc[i, j]

            ax.text(
                j, i,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value > 0.5 else "black"
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recall")
    cbar.ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    ax.set_title("Validation Recall by Attack Type")
    ax.set_xlabel("Model")
    ax.set_ylabel("Attack Type")

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "per_class_recall.png",
        dpi=300
    )

    plt.close(fig)

    print("Per-class recall heatmap saved.")

def analyze_attack_detection(predictions):
    y_true = predictions["attack_type"]

    model_columns = predictions.columns.drop(
        ["row_index", "attack_type"]
    )

    actual_attack = y_true != "benign"
    actual_benign = y_true == "benign"

    results = []

    for model_name in model_columns:
        y_pred = predictions[model_name]
        predicted_attack = y_pred != "benign"

        tp = (actual_attack & predicted_attack).sum()
        fp = (actual_benign & predicted_attack).sum()
        tn = (actual_benign & ~predicted_attack).sum()
        fn = (actual_attack & ~predicted_attack).sum()

        false_alarm_rate = fp / (fp + tn)
        attack_detection_rate = tp / (tp + fn)

        results.append({
            "model": model_name,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "false_alarm_rate": false_alarm_rate,
            "attack_detection_rate": attack_detection_rate
        })

    df = pd.DataFrame(results)

    df.to_csv(
        RESULTS_DIR / "attack_detection_metrics.csv",
        index=False,
        float_format="%.6f"
    )

    ax = df.set_index("model")[
        ["attack_detection_rate", "false_alarm_rate"]
    ].plot(
        kind="bar",
        figsize=(14, 7),
        color=["seagreen", "indianred"]
    )

    ax.set_title("Attack Detection vs. False Alarm Rate")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.1)

    ax.yaxis.set_major_formatter(
        PercentFormatter(1.0)
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                f"{v:.1%}" for v in container.datavalues
            ],
            padding=3,
            fontsize=9
        )

    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "attack_detection_comparison.png",
        dpi=300
    )

    plt.close()

    print("\nAttack Detection Results:")
    print(df.to_string(index=False))

    return df

def plot_detector_comparison(multiclass_df):
    detector_dir = DATA_DIR / "detector_original"

    with open(detector_dir / "run.json", "r") as f:
        settings = json.load(f)

    detector_df = pd.DataFrame(settings["selected"])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Multiclass classifiers
    ax.scatter(
        multiclass_df["false_alarm_rate"],
        multiclass_df["attack_detection_rate"],
        color="gray",
        s=80,
        label="Multiclass Classifiers"
    )

    for _, row in multiclass_df.iterrows():
        ax.annotate(
            row["model"],
            (
                row["false_alarm_rate"],
                row["attack_detection_rate"]
            ),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8
        )

    # Selected binary detectors
    ax.scatter(
        detector_df["false_alarm_rate"],
        detector_df["recall"],
        color="seagreen",
        s=150,
        marker="D",
        label="Binary Detector"
    )

    for _, row in detector_df.iterrows():
        ax.annotate(
            f"Binary RF ({row['budget']:.0%} budget)",
            (
                row["false_alarm_rate"],
                row["recall"]
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=10
        )

    ax.axvspan(
        0, 0.05,
        color="green",
        alpha=0.08,
        label="Low False-Alarm Region"
    )

    ax.set_title(
        "Multiclass vs. Binary Detection - Validation"
    )

    ax.set_xlabel("False-Alarm Rate")
    ax.set_ylabel("Attack Detection Rate")

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.05)

    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "detector_comparison.png",
        dpi=300
    )

    plt.close(fig)

    detector_df.to_csv(
        RESULTS_DIR / "binary_detector_metrics.csv",
        index=False
    )

    print("Binary detector comparison saved.")
    
def main():
    X_val = pd.read_csv(DATA_DIR / "X_validation.csv")
    y_val = pd.read_csv(
        DATA_DIR / "y_validation.csv"
    )["attack_type"]

    predictions = pd.read_csv(
        TRAINING_DIR / "validation_predictions.csv"
    )

    with open(TRAINING_DIR / "run.json", "r") as f:
        settings = json.load(f)

    classes = np.array(settings["classes"])

    # Verify that the predictions match our validation data
    np.testing.assert_array_equal(
        predictions["row_index"], np.arange(len(X_val))
    )

    np.testing.assert_array_equal(
        predictions["attack_type"], y_val
    )

    results = []

    for model_name, info in settings["models"].items():

        # Load trained model
        with open(TRAINING_DIR / info["artifact"], "rb") as f:
            model = pickle.load(f)

        # Get exported predictions
        y_pred = predictions[model_name]

        # Verify that the loaded model produces the same predictions
        encoded_predictions = np.asarray(
            model.predict(X_val), dtype=int
        ).ravel()

        np.testing.assert_array_equal(
            classes[encoded_predictions], y_pred
        )

        # Calculate evaluation metrics
        metrics = evaluate_model(model_name, y_val, y_pred)

        # Measure inference latency
        latency = measure_inference_latency(model, X_val)

        metrics["inference_latency_ms"] = latency

        print(f"\nInference latency: {latency:.6f} ms/sample")

        results.append(metrics)

        # Generate confusion matrices
        plot_confusion_matrix(model_name, y_val, y_pred)

    # Save all model results
    df = save_results(results)

    # Generate comparison graphs
    plot_model_comparison(df)
    plot_latency_comparison(df)

    plot_per_class_recall(predictions)
    analyze_attack_detection(predictions)
    detection_df = analyze_attack_detection(predictions)

    plot_detector_comparison(detection_df)

if __name__ == "__main__":
    main()
