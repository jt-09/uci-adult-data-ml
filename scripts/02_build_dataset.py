#!/usr/bin/env python
"""DATA-002: Build cleaned dataset and train/test split indices."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from adult_income_ml.cleaning import clean_dataframe, get_cleaning_summary_table
from adult_income_ml.data import load_raw
from adult_income_ml.reporting import export_table
from adult_income_ml.splitting import get_split_summary_table, save_split_indices, split_train_test
from adult_income_ml.utils import console, ensure_dir, get_paths, load_config, set_seed, tag_artifact


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    ensure_dir(paths["processed"])

    df = load_raw(cfg)
    clean, decisions = clean_dataframe(df, cfg)
    clean = clean.reset_index(drop=True)
    out = paths["processed"] / cfg["dataset"]["clean_filename"]
    clean.to_csv(out, index=False)
    tag_artifact(out, ["DATA-002"])

    X_train, X_test, y_train, y_test = split_train_test(clean, cfg)
    train_idx = X_train.index
    test_idx = X_test.index
    save_split_indices(train_idx, test_idx, cfg)

    export_table(get_cleaning_summary_table(decisions), paths["tables"] / "table_03_cleaning_decisions.csv")
    export_table(get_split_summary_table(clean, train_idx, test_idx, cfg), paths["tables"] / "table_04_split_summary.csv")

    console.print(f"[bold green]DATA-002 complete:[/bold green] {out} ({len(clean)} rows)")


if __name__ == "__main__":
    main()
