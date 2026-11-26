import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_validation_vs_test(df, output_dir):
    metrics = df.set_index("model")[
        ["validation_macro_f1", "macro_f1"]
    ]

    metrics.columns = ["Validation", "Test"]

    ax = metrics.plot(
        kind="bar",
        figsize=(14, 7),
        color=["steelblue", "seagreen"]
    )

    ax.set_title("Validation vs. Test Macro F1")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_ylim(0, metrics.max().max() * 1.25)

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.2%}" for v in container.datavalues],
            padding=3,
            fontsize=9
        )

    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "validation_vs_test.png",
        dpi=300
    )

    plt.close()

def plot_attack_classification(df, output_dir):
    metrics = df.set_index("model")[
        ["attacks_detected", "attacks_classified_correctly"]
    ]

    metrics.columns = [
        "Attacks Detected",
        "Correct Attack Type"
    ]

    ax = metrics.plot(
        kind="bar",
        figsize=(16, 8),
        color=["steelblue", "seagreen"]
    )

    ax.set_title("Attack Detection vs. Exact Classification")
    ax.set_xlabel("Model")
    ax.set_ylabel("Number of Attacks")
    ax.set_ylim(0, 70)

    ax.axhline(
        60,
        color="red",
        linestyle="--",
        label="Total Actual Attacks"
    )

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=9)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "attack_classification.png",
        dpi=300
    )

    plt.close()

def plot_detection_tradeoff(df, output_dir):
    metrics = df.set_index("model")[
        ["attacks_detected", "benign_false_alarm_rate"]
    ].copy()

    metrics["attacks_detected"] /= 60

    metrics.columns = [
        "Attack Detection Rate",
        "False Alarm Rate"
    ]

    ax = metrics.plot(
        kind="bar",
        figsize=(14, 7),
        color=["seagreen", "indianred"]
    )

    ax.set_title("Test Attack Detection vs. False Alarm Rate")
    ax.set_xlabel("Model")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.15)

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.1%}" for v in container.datavalues],
            padding=3,
            fontsize=9
        )

    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "detection_tradeoff.png",
        dpi=300
    )

    plt.close()

def plot_inference_latency(df, output_dir):
    metrics = df.set_index("model")[
        [
            "batch_ms_per_record",
            "single_record_median_ms",
            "single_record_p95_ms"
        ]
    ]

    metrics.columns = [
        "Batch Latency",
        "Single Record Median",
        "Single Record P95"
    ]

    ax = metrics.plot(
        kind="bar",
        figsize=(14, 7),
        color=["steelblue", "seagreen", "darkorange"]
    )

    ax.set_title("Test Inference Latency Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (ms, logarithmic scale)")
    ax.set_yscale("log")

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                f"{v:.5f}" if v < 0.01 else f"{v:.2f}"
                for v in container.datavalues
            ],
            padding=3,
            fontsize=8
        )

    ax.legend(loc="lower right")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "inference_latency.png",
        dpi=300
    )

    plt.close()

def plot_test_performance(df, output_dir):
    metrics = df.set_index("model")[
        [
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "weighted_f1"
        ]
    ]

    metrics.columns = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ]

    ax = metrics.plot(
        kind="bar",
        figsize=(18, 8)
    )

    ax.set_title("Overall Test Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)

    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[f"{v:.1%}" for v in container.datavalues],
            padding=3,
            fontsize=7
        )

    ax.legend(loc="upper center", ncol=5)

    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_dir / "test_performance.png",
        dpi=300
    )

    plt.close()

def plot_detector_comparison(df, detector_df, output_dir):
    total_attacks = detector_df["attack_rows"].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Multiclass models
    ax.scatter(
        df["benign_false_alarm_rate"],
        df["attacks_detected"] / total_attacks,
        color="steelblue",
        s=100,
        label="Multiclass Models"
    )

    for _, row in df.iterrows():
        ax.annotate(
            row["model"],
            (
                row["benign_false_alarm_rate"],
                row["attacks_detected"] / total_attacks
            ),
            xytext=(8, 5),
            textcoords="offset points",
            fontsize=9
        )

    # Binary detector
    ax.scatter(
        detector_df["false_alarm_rate"],
        detector_df["recall"],
        color="seagreen",
        marker="D",
        s=150,
        label="Binary Detector"
    )

    for _, row in detector_df.iterrows():
        ax.annotate(
            f"Binary RF ({row['validation_budget']:.0%} budget)",
            (
                row["false_alarm_rate"],
                row["recall"]
            ),
            xytext=(10, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold"
        )

    ax.axvspan(
        0, 0.05,
        color="green",
        alpha=0.08,
        label="Low False-Alarm Region"
    )

    ax.set_title("Multiclass vs. Binary Detector - Test Performance")
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Attack Detection Rate")

    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.1)

    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "detector_comparison.png",
        dpi=300
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "evaluation_dir",
        type=Path
    )

    parser.add_argument(
    "--detector-dir",
    type=Path,
    default=None
)
    
    args = parser.parse_args()

    df = pd.read_csv(
        args.evaluation_dir / "test_metrics.csv"
    )

    output_dir = args.evaluation_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_validation_vs_test(df, output_dir)
    plot_attack_classification(df, output_dir)
    plot_detection_tradeoff(df, output_dir)
    plot_inference_latency(df, output_dir)
    plot_test_performance(df, output_dir)

    if args.detector_dir is not None:
        detector_df = pd.read_csv(
            args.detector_dir / "test_metrics.csv"
    )

        plot_detector_comparison(
         df, detector_df, output_dir
        )

    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()