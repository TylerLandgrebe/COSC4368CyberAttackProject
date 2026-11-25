from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


IDENTIFIERS = ["timestamp", "src_ip", "dst_ip"]


def make_model_pipeline(classifier, to_cuda=None, feature_transform=None) -> Pipeline:
    preprocessor = make_preprocessor()
    steps = [("preprocess", preprocessor)]
    if feature_transform is not None:
        steps.insert(0, ("features", FunctionTransformer(feature_transform)))
    if to_cuda is not None:
        preprocessor.sparse_threshold = 0
        steps.append(("to_cuda", FunctionTransformer(to_cuda, kw_args={"dtype": np.float32})))
    steps.append(("classifier", classifier))
    return Pipeline(steps)


def traffic_features(features: pd.DataFrame) -> pd.DataFrame:
    url = features["url"].fillna("")
    return features.drop(columns=[*IDENTIFIERS, "src_port", "url"]).assign(
        dst_port=features["dst_port"].fillna(-1).astype(str),
        url_present=url.ne("").astype(int),
        url_length=url.str.len(),
        url_slashes=url.str.count("/"),
        log_byte_ratio=np.log1p(features["bytes_sent"]) - np.log1p(features["bytes_received"]),
    )


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
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
