"""Data cleaning for Adult Income dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from adult_income_ml.utils import load_config


def clean_dataframe(df: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Clean raw Adult data; return cleaned frame and cleaning decision log."""
    cfg = cfg or load_config()
    ds = cfg["dataset"]
    decisions: list[dict] = []
    out = df.copy()

    # Strip whitespace on object columns
    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype(str).str.strip()

    # Missing token (np.nan for sklearn SimpleImputer compatibility; pd.NA breaks object columns)
    token = ds["missing_token"]
    out = out.replace(token, np.nan)
    decisions.append(
        {
            "step": "missing_token",
            "token": token,
            "description": f"Replaced '{token}' with NA",
        }
    )

    if ds.get("drop_duplicates", True):
        n_dup = out.duplicated().sum()
        out = out.drop_duplicates()
        decisions.append({"step": "drop_duplicates", "removed": int(n_dup)})

    # Target encoding: <=50K -> 0, >50K -> 1
    target = ds["target_column"]
    pos = ds["positive_label"]
    neg = ds["negative_label"]
    out[target] = out[target].replace({neg: 0, pos: 1, f"{pos}.": 1, f"{neg}.": 0})
    out[target] = pd.to_numeric(out[target], errors="coerce").astype("Int64")
    decisions.append(
        {
            "step": "target_encoding",
            "mapping": {neg: 0, pos: 1},
            "column": target,
        }
    )

    # Drop rows with missing target
    out = out.dropna(subset=[target])
    out[target] = out[target].astype(int)

    return out, decisions


def get_cleaning_summary_table(decisions: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(decisions)
