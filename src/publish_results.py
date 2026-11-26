from __future__ import annotations

import argparse
from pathlib import Path
from shutil import copyfile

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_NAMES = {
    "most_frequent": "Always benign",
    "logistic_regression": "Logistic regression",
    "svm": "RBF SVM",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
}
SPLIT_NAMES = {"original": "Original holdout", "chronological": "Later development rows"}
BLUE, TEAL, GRAY = "#285B93", "#087F75", "#64748B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish aggregate CUDA metrics and presentation figures.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final"))
    parser.add_argument("--include-boosting", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    multiclass = args.data_dir / "training_cuda_with_test" / "evaluation"
    cascade = args.data_dir / "cascade_cuda"
    for source, name in (
        (multiclass / "test_metrics.csv", "multiclass_test_metrics.csv"),
        (multiclass / "per_class_metrics.csv", "multiclass_per_class_metrics.csv"),
        (multiclass / "confusion_random_forest_without_ids.png", "random_forest_confusion.png"),
        (cascade / "test_metrics.csv", "cascade_test_metrics.csv"),
        (cascade / "validation_metrics.csv", "cascade_validation_metrics.csv"),
    ):
        copyfile(source, args.output_dir / name)

    operating_points = pd.concat([
        pd.read_csv(args.data_dir / f"detector_cuda_{split}" / "test_metrics.csv").assign(split=split)
        for split in SPLIT_NAMES
    ], ignore_index=True)
    operating_points.to_csv(args.output_dir / "binary_operating_points.csv", index=False)
    multiclass_metrics = pd.read_csv(multiclass / "test_metrics.csv")
    plot_multiclass(multiclass_metrics, args.output_dir)
    plot_per_class_recall(pd.read_csv(multiclass / "per_class_metrics.csv"), args.output_dir)
    if "xgboost_without_ids" in set(multiclass_metrics["model"]):
        copyfile(multiclass.parent / "xgboost_validation_metrics.csv", args.output_dir / "multiclass_xgboost_validation.csv")
        copyfile(multiclass / "confusion_xgboost_without_ids.png", args.output_dir / "xgboost_confusion.png")
    plot_operating_points(operating_points, args.output_dir)
    plot_attribution(pd.read_csv(cascade / "test_metrics.csv"), args.output_dir)
    if args.include_boosting:
        boosting = args.data_dir / "boosting_comparison_cuda"
        copyfile(boosting / "test_metrics.csv", args.output_dir / "boosting_test_metrics.csv")
        copyfile(boosting / "validation_metrics.csv", args.output_dir / "boosting_validation_metrics.csv")
        plot_boosting(pd.read_csv(boosting / "test_metrics.csv"), args.output_dir)
    print(f"Outputs: {args.output_dir.resolve()}")


def model_label(name: str) -> str:
    if name == "most_frequent":
        return MODEL_NAMES[name]
    model, identifiers, _ = name.rsplit("_", 2)
    return f"{MODEL_NAMES[model]} · {identifiers} IDs"


def plot_multiclass(metrics: pd.DataFrame, output: Path) -> None:
    ordered = metrics.sort_values("macro_f1")
    labels = ordered["model"].map(model_label)
    colors = [TEAL if model.startswith("xgboost") else BLUE if model.startswith("random_forest") else GRAY
              for model in ordered["model"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.7), layout="constrained")
    bars = ax.barh(labels, ordered["macro_f1"], color=colors)
    ax.bar_label(bars, fmt="%.3f", padding=5)
    ax.set(xlim=(0, 1.18 * ordered["macro_f1"].max()), xlabel="Macro-F1",
           title="Multiclass performance on the held-out test split")
    save_plot(fig, output / "multiclass_macro_f1.png")

    ordered = metrics.sort_values("single_record_median_ms")
    fig, ax = plt.subplots(figsize=(8.6, 4.7), layout="constrained")
    bars = ax.barh(ordered["model"].map(model_label), ordered["single_record_median_ms"], color=BLUE)
    ax.bar_label(bars, fmt="%.2f", padding=5)
    ax.set(xscale="log", xlim=(ordered["single_record_median_ms"].min() / 3,
                               ordered["single_record_median_ms"].max() * 2),
           xlabel="Median milliseconds per record (log scale)",
           title="Single-record inference latency on the RTX 3060 workstation")
    save_plot(fig, output / "single_record_latency.png")


def plot_operating_points(metrics: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), layout="constrained")
    for split, color in zip(SPLIT_NAMES, (BLUE, TEAL)):
        points = metrics.loc[metrics["split"].eq(split)]
        ax.scatter(points["false_alarm_rate"], points["recall"], color=color, label=SPLIT_NAMES[split])
        for point in points.itertuples():
            ax.annotate(f"{point.attacks_detected}/{point.attack_rows}",
                        (point.false_alarm_rate, point.recall), xytext=(8, 0),
                        textcoords="offset points", va="center", color=color)
    ax.set(xlim=(0, max(0.065, 1.3 * metrics["false_alarm_rate"].max())), ylim=(0, 1.0),
           xlabel="Observed benign false-alarm rate",
           ylabel="Attack recall", title="Binary detection at two validation-selected thresholds")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", frameon=False)
    save_plot(fig, output / "binary_false_alarm_tradeoff.png")


def plot_attribution(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.9), sharey=True, layout="constrained")
    labels = ["Multiclass\nforest", "Cascade\n1% budget", "Cascade\n5% budget"]
    positions = np.arange(len(labels))
    upper = max(0.5, 1.2 * (metrics["attacks_typed_correctly"] / metrics["attack_rows"]).max())
    for ax, (split, title) in zip(axes, SPLIT_NAMES.items()):
        rows = metrics.loc[metrics["split"].eq(split)]
        bars = ax.bar(positions, rows["attacks_typed_correctly"] / rows["attack_rows"],
                      color=[GRAY, TEAL, BLUE], width=0.58)
        ax.bar_label(bars, labels=[f"{row.attacks_typed_correctly}/{row.attack_rows}"
                                  for row in rows.itertuples()], padding=5)
        ax.set(xticks=positions, xticklabels=labels, ylim=(0, upper), title=title)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Share of all attacks typed correctly")
    fig.suptitle("End-to-end attack attribution")
    fig.savefig(output / "cascade_attribution.png", dpi=160)
    plt.close(fig)


def plot_per_class_recall(metrics: pd.DataFrame, output: Path) -> None:
    recall = metrics.pivot(index="attack_type", columns="model", values="recall")
    recall = recall.reindex(columns=metrics["model"].unique())
    support = metrics.drop_duplicates("attack_type").set_index("attack_type")["support"]
    fig, ax = plt.subplots(figsize=(11.5, 6.6), layout="constrained")
    image = ax.imshow(recall, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(recall.columns)), [model_label(name) for name in recall.columns],
                  rotation=32, ha="right")
    ax.set_yticks(range(len(recall)), [f"{label} (n={support[label]})" for label in recall.index])
    for row, values in enumerate(recall.to_numpy()):
        for column, value in enumerate(values):
            ax.text(column, row, f"{value:.0%}", ha="center", va="center",
                    color="white" if value > 0.5 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Recall", format=PercentFormatter(1))
    ax.set_title("Held-out test recall by attack type")
    fig.savefig(output / "multiclass_recall_heatmap.png", dpi=160)
    plt.close(fig)


def plot_boosting(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), sharex=True, sharey=True, layout="constrained")
    for ax, (split, title) in zip(axes, SPLIT_NAMES.items()):
        rows = metrics.loc[metrics["split"].eq(split)]
        for family, color in (("random_forest", BLUE), ("xgboost", TEAL)):
            name = "Random forest" if family == "random_forest" else "XGBoost"
            for feature_set, marker in (
                ("without_ids", "o"), ("without_ids_user_agent_url", "s")
            ):
                points = rows.loc[rows["family"].eq(family) & rows["feature_set"].eq(feature_set)]
                points = points.sort_values("validation_budget")
                feature_label = "retains URL + UA" if feature_set == "without_ids" else "drops URL + UA"
                ax.scatter(points["false_alarm_rate"], points["recall"], marker=marker,
                           color=color, label=f"{name} · {feature_label}")
                if family == "xgboost" and feature_set == "without_ids":
                    for point in points.itertuples():
                        ax.annotate(
                            f"{point.validation_budget:.0%} budget, depth {point.max_depth}",
                            (point.false_alarm_rate, point.recall), xytext=(5, 7),
                            textcoords="offset points", fontsize=8, color=TEAL,
                        )
        ax.set(title=title, xlabel="Observed benign false-alarm rate", ylim=(0, 1),
               xlim=(0, max(0.065, 1.1 * metrics["false_alarm_rate"].max())))
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Attack recall")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Binary gates at 1% and 5% validation false-alarm budgets")
    fig.savefig(output / "boosting_comparison.png", dpi=160)
    plt.close(fig)


def save_plot(fig, path: Path) -> None:
    ax = fig.axes[0]
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
