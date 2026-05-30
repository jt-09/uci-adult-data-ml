#!/usr/bin/env python
"""EDA-001 to EDA-003: Exploratory data analysis figures and tables."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.features import build_feature_dictionary, get_feature_columns
from adult_income_ml.plotting import (
    fig_correlation_heatmap,
    fig_income_by_group,
    fig_missingness,
    fig_numeric_by_target,
    fig_target_distribution,
)
from adult_income_ml.reporting import export_table
from adult_income_ml.utils import console, get_paths, load_config, set_seed, tag_artifact


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    df = load_clean(cfg)
    target = cfg["dataset"]["target_column"]
    feat = get_feature_columns(cfg)

    # Table 1: dataset summary
    summary = pd.DataFrame(
        {
            "metric": ["rows", "columns", "target_positive_rate"],
            "value": [len(df), len(df.columns), float(df[target].mean())],
        }
    )
    export_table(summary, paths["tables"] / "table_01_dataset_summary.csv")
    export_table(build_feature_dictionary(cfg), paths["tables"] / "table_02_feature_dictionary.csv")

    fig_missingness(df, paths["figures"] / "fig_01_missingness.png", cfg)
    tag_artifact(paths["figures"] / "fig_01_missingness.png", ["EDA-001"])

    fig_target_distribution(df[target], paths["figures"] / "fig_02_target_distribution.png", cfg)
    tag_artifact(paths["figures"] / "fig_02_target_distribution.png", ["EDA-002"])

    for i, col in enumerate(cfg["columns"]["sensitive"]):
        fig_income_by_group(df, col, target, paths["figures"] / f"fig_03_sensitive_{col}.png", cfg)
    tag_artifact(paths["figures"] / "fig_03_sensitive_sex.png", ["EDA-003"])

    fig_numeric_by_target(df, feat["numeric"], target, paths["figures"] / "fig_04_numeric_by_target.png", cfg)
    fig_income_by_group(df, "education", target, paths["figures"] / "fig_05_categorical_rates.png", cfg)
    fig_correlation_heatmap(df, feat["numeric"], paths["figures"] / "fig_06_correlation.png", cfg)

    console.print("[bold green]EDA-001 to EDA-003 complete[/bold green]")


if __name__ == "__main__":
    main()
