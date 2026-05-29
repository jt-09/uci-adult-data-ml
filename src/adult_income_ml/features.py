"""Feature typing and column metadata."""

from __future__ import annotations

import pandas as pd

from adult_income_ml.utils import load_config


def get_feature_columns(cfg: dict | None = None) -> dict[str, list[str]]:
    cfg = cfg or load_config()
    cols = cfg["columns"]
    return {
        "numeric": list(cols["numeric"]),
        "categorical": list(cols["categorical"]),
        "sensitive": list(cols["sensitive"]),
        "drop_for_extension": list(cols["drop_for_extension"]),
    }


def get_X_y(
    df: pd.DataFrame,
    cfg: dict | None = None,
    drop_sensitive: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    cfg = cfg or load_config()
    feat = get_feature_columns(cfg)
    target = cfg["dataset"]["target_column"]
    feature_cols = feat["numeric"] + feat["categorical"]
    if drop_sensitive:
        feature_cols = [c for c in feature_cols if c not in feat["drop_for_extension"]]
    X = df[feature_cols].copy()
    y = df[target].copy()
    return X, y


def build_feature_dictionary(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    feat = get_feature_columns(cfg)
    rows = []
    for c in feat["numeric"]:
        rows.append({"feature": c, "type": "numeric", "sensitive": c in feat["sensitive"]})
    for c in feat["categorical"]:
        rows.append({"feature": c, "type": "categorical", "sensitive": c in feat["sensitive"]})
    return pd.DataFrame(rows)
