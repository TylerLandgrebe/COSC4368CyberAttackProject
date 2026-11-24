"""Leakage-safe feature preprocessing helpers for scikit-learn model pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor() -> ColumnTransformer:
    """Create a transformer to be fit on X_train only, inside a model Pipeline.

    Numeric columns receive median imputation and standardization. Categorical columns
    receive most-frequent imputation and one-hot encoding. Rare categories are grouped
    to keep high-cardinality fields manageable; unseen validation/test categories are
    ignored. This helper intentionally does not decide whether identifiers such as IPs,
    URLs, or timestamps should be features; inspect the data summary and evaluate those
    choices as part of the experiment.
    """
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            # Preserve missingness as its own category (notably, many URLs are absent).
            ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
            (
                "encode",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=2,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, make_column_selector(dtype_include="number")),
            ("categorical", categorical, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
