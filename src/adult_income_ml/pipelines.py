"""Sklearn preprocessing pipelines."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from adult_income_ml.features import get_feature_columns
from adult_income_ml.utils import load_config


def build_preprocessor(
    cfg: dict | None = None,
    drop_sensitive: bool = False,
) -> ColumnTransformer:
    cfg = cfg or load_config()
    feat = get_feature_columns(cfg)
    numeric = list(feat["numeric"])
    categorical = list(feat["categorical"])
    if drop_sensitive:
        drop = set(feat["drop_for_extension"])
        numeric = [c for c in numeric if c not in drop]
        categorical = [c for c in categorical if c not in drop]

    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    estimator,
    cfg: dict | None = None,
    drop_sensitive: bool = False,
) -> Pipeline:
    from sklearn.pipeline import Pipeline as SkPipeline

    return SkPipeline(
        [
            ("preprocessor", build_preprocessor(cfg, drop_sensitive=drop_sensitive)),
            ("classifier", estimator),
        ]
    )
