"""Stratified train/test split with persisted indices."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from adult_income_ml.features import get_X_y
from adult_income_ml.utils import ensure_dir, get_paths, load_config, save_json


def split_train_test(
    df: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    cfg = cfg or load_config()
    X, y = get_X_y(df, cfg)
    sp = cfg["split"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=sp["test_size"],
        random_state=sp["random_state"],
        stratify=y if sp.get("stratify", True) else None,
    )
    return X_train, X_test, y_train, y_test


def save_split_indices(
    train_idx: pd.Index,
    test_idx: pd.Index,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    ensure_dir(paths["splits"])
    joblib.dump(train_idx, paths["splits"] / "train_idx.pkl")
    joblib.dump(test_idx, paths["splits"] / "test_idx.pkl")
    summary = {
        "train_size": len(train_idx),
        "test_size": len(test_idx),
        "train_positive_rate": None,
        "test_positive_rate": None,
    }
    save_json(summary, paths["splits"] / "split_summary.json")
    return summary


def load_split_indices(cfg: dict | None = None) -> tuple[pd.Index, pd.Index]:
    paths = get_paths(cfg or load_config())
    train_idx = joblib.load(paths["splits"] / "train_idx.pkl")
    test_idx = joblib.load(paths["splits"] / "test_idx.pkl")
    return train_idx, test_idx


def get_split_summary_table(
    df: pd.DataFrame,
    train_idx: pd.Index,
    test_idx: pd.Index,
    cfg: dict | None = None,
) -> pd.DataFrame:
    cfg = cfg or load_config()
    target = cfg["dataset"]["target_column"]
    rows = []
    for name, idx in [("train", train_idx), ("test", test_idx)]:
        sub = df.loc[idx, target]
        rows.append(
            {
                "split": name,
                "n": len(idx),
                "positive_rate": float(sub.mean()),
                "negative_rate": float(1 - sub.mean()),
            }
        )
    return pd.DataFrame(rows)
