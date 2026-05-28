"""Fetch and load UCI Adult dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from adult_income_ml.utils import console, ensure_dir, get_paths, load_config, tag_artifact

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def fetch_adult(force: bool = False) -> Path:
    """Download Adult dataset via ucimlrepo and save raw CSV. Returns path to raw file."""
    cfg = load_config()
    paths = get_paths(cfg)
    ensure_dir(paths["raw"])
    out = paths["raw"] / cfg["dataset"]["raw_filename"]

    if out.exists() and not force:
        console.print(f"[green]Raw data exists:[/green] {out}")
        return out

    console.print("Fetching UCI Adult dataset (id=2)...")
    from ucimlrepo import fetch_ucirepo

    adult = fetch_ucirepo(id=cfg["dataset"]["uci_id"])
    X = adult.data.features
    y = adult.data.targets
    df = pd.concat([X, y], axis=1)
    if list(df.columns) != ADULT_COLUMNS and len(df.columns) == len(ADULT_COLUMNS):
        df.columns = ADULT_COLUMNS
    df.to_csv(out, index=False)
    tag_artifact(out, ["DATA-001"])
    console.print(f"[green]Saved raw data:[/green] {out} ({len(df)} rows)")
    return out


def load_raw(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    path = paths["raw"] / cfg["dataset"]["raw_filename"]
    if not path.exists():
        fetch_adult()
    return pd.read_csv(path)


def load_clean(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    path = paths["processed"] / cfg["dataset"]["clean_filename"]
    if not path.exists():
        raise FileNotFoundError(f"Clean dataset not found: {path}. Run scripts/02_build_dataset.py")
    return pd.read_csv(path)
