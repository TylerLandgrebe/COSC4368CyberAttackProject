from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMNS = {"label", "attack_type"}
SPLIT_NAMES = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the source CSV and create reproducible, stratified dataset "
            "splits for cybersecurity classification."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the downloaded source CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for split CSVs and data_summary.json (default: data/processed).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_COLUMNS),
        default="attack_type",
        help="Prediction target. Defaults to the proposal's multi-class attack_type task.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    return parser.parse_args()


def _validate_sizes(train_size: float, validation_size: float, test_size: float) -> None:
    sizes = (train_size, validation_size, test_size)
    if any(size <= 0 or size >= 1 for size in sizes):
        raise ValueError("Each split size must be greater than 0 and less than 1.")
    if abs(sum(sizes) - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test sizes must sum to 1.0.")


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset CSV not found: {path}\n"
            "Download it from the Kaggle link in data/README.md and pass its path with --input."
        )

    frame = pd.read_csv(path, skipinitialspace=True)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.columns.duplicated().any():
        repeated = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"Column names are duplicated after trimming whitespace: {repeated}")
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")

    text_columns = frame.select_dtypes(include=["object", "string"]).columns
    for column in text_columns:
        frame[column] = frame[column].map(
            lambda value: pd.NA if isinstance(value, str) and not value.strip() else value
        )
    return frame


def _stratified_splits(
    frame: pd.DataFrame,
    target: str,
    train_size: float,
    validation_size: float,
    test_size: float,
    seed: int,
) -> dict[str, pd.DataFrame]:
    _validate_sizes(train_size, validation_size, test_size)
    class_counts = frame[target].value_counts(dropna=False)
    if len(class_counts) < 2:
        raise ValueError(f"Target '{target}' has fewer than two classes; classification needs at least two.")

    try:
        train_validation, test = train_test_split(
            frame,
            test_size=test_size,
            random_state=seed,
            stratify=frame[target],
        )
        validation_fraction_of_remainder = validation_size / (train_size + validation_size)
        train, validation = train_test_split(
            train_validation,
            test_size=validation_fraction_of_remainder,
            random_state=seed,
            stratify=train_validation[target],
        )
    except ValueError as error:
        smallest_class = int(class_counts.min())
        raise ValueError(
            "Could not make stratified splits. Each class needs enough examples to appear "
            f"in all three splits; the smallest class has {smallest_class} row(s). "
            "Review the target distribution or choose a different split plan."
        ) from error

    return {"train": train, "validation": validation, "test": test}


def _class_distribution(values: pd.Series) -> dict[str, dict[str, float | int]]:
    counts = values.value_counts(dropna=False, sort=False)
    total = len(values)
    return {
        str(label): {"count": int(count), "fraction": float(count / total)}
        for label, count in counts.items()
    }


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source = args.input.expanduser().resolve()
    frame = _load_csv(source)
    if args.target not in frame.columns:
        raise ValueError(
            f"Target column '{args.target}' is missing. Available columns: {list(frame.columns)}"
        )

    source_rows = len(frame)
    duplicate_rows = int(frame.duplicated().sum())
    missing_before = frame.isna().sum()
    missing_target_rows = int(frame[args.target].isna().sum())
    if missing_target_rows:
        frame = frame.loc[frame[args.target].notna()].copy()
    if frame.empty:
        raise ValueError(f"No rows remain after excluding rows with missing '{args.target}'.")

    splits = _stratified_splits(
        frame,
        args.target,
        args.train_size,
        args.validation_size,
        args.test_size,
        args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_columns = [
        column for column in frame.columns if column not in TARGET_COLUMNS
    ]
    for split_name in SPLIT_NAMES:
        split = splits[split_name]
        X = split[feature_columns]
        y = split[[args.target]]
        X.to_csv(args.output_dir / f"X_{split_name}.csv", index=False)
        y.to_csv(args.output_dir / f"y_{split_name}.csv", index=False)

    split_summary: dict[str, Any] = {}
    for split_name, split in splits.items():
        split_summary[split_name] = {
            "rows": len(split),
            "target_distribution": _class_distribution(split[args.target]),
            "feature_missing_values": {
                column: int(count)
                for column, count in split[feature_columns].isna().sum().items()
                if count
            },
        }

    feature_profile: dict[str, Any] = {}
    for column in feature_columns:
        values = frame[column]
        unique_count = int(values.nunique(dropna=True))
        feature_profile[column] = {
            "dtype": str(values.dtype),
            "unique_non_null_values": unique_count,
            "missing_values": int(values.isna().sum()),
            "high_cardinality": bool(unique_count > 100 or unique_count / max(len(values), 1) > 0.20),
        }

    summary: dict[str, Any] = {
        "source_file": source.name,
        "target": args.target,
        "seed": args.seed,
        "split_fractions": {
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
        },
        "rows": {
            "source": source_rows,
            "excluded_for_missing_target": missing_target_rows,
            "used": len(frame),
            "exact_duplicate_rows_in_source": duplicate_rows,
        },
        "target_distribution_before_split": _class_distribution(frame[args.target]),
        "feature_columns": feature_columns,
        "feature_profile": feature_profile,
        "excluded_target_columns": sorted(TARGET_COLUMNS.intersection(frame.columns)),
        "column_dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "missing_values_before_split": {
            column: int(count) for column, count in missing_before.items() if count
        },
        "splits": split_summary,
    }
    with (args.output_dir / "data_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    return summary


def main() -> None:
    args = parse_args()
    try:
        summary = prepare_dataset(args)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise SystemExit(f"Data preparation failed: {error}") from error

    print(f"Prepared {summary['rows']['used']:,} rows for target '{summary['target']}'.")
    print(f"Feature columns: {len(summary['feature_columns'])}")
    for name, split in summary["splits"].items():
        print(f"{name.capitalize():10s} {split['rows']:>7,} rows")
    print(f"Outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
