import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import time

from sklearn.dummy import DummyClassifier
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
        figsize=(12, 7),
        width=0.8
    )

    ax.set_title("Model Performance Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    plt.xticks(rotation=0)
    plt.legend(loc="upper right")
    plt.grid(axis="y", alpha=0.3)

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v * 100:.1f}%" for v in container.datavalues],
            padding=3,
            fontsize=10
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

    # Warm-up prediction
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
    fig, ax = plt.subplots(figsize=(10, 6))

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
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "latency_comparison.png",
        dpi=300
    )

    plt.close(fig)

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    results = []

    baseline = DummyClassifier(strategy="most_frequent")

    baseline.fit(X_train, y_train)

    y_pred = baseline.predict(X_val)

    metrics = evaluate_model(
        "Majority Class Baseline", y_val, y_pred
    )

    latency = measure_inference_latency(baseline, X_val)

    metrics["inference_latency_ms"] = latency

    print(f"\nInference latency: {latency:.6f} ms/sample")

    results.append(metrics)

    evaluate_model("Majority Class Baseline", y_val, y_pred)

    plot_confusion_matrix("Majority Class Baseline", y_val, y_pred)

    df = save_results(results)

    plot_model_comparison(df)

    plot_latency_comparison(df)


if __name__ == "__main__":
    main()