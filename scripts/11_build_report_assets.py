#!/usr/bin/env python
"""RPT-001: Sync artifacts and validate traceability matrix."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.reporting import copy_to_reports, validate_traceability
from adult_income_ml.utils import console, get_paths, load_config

EVIDENCE_MAP = {
    "RQ-001": "reports/report_outline.md",
    "DATA-001": "data/raw/adult_raw.csv",
    "DATA-002": "data/processed/adult_clean.csv",
    "EDA-001": "reports/figures/fig_01_missingness.png",
    "EDA-002": "reports/figures/fig_02_target_distribution.png",
    "EDA-003": "reports/figures/fig_03_sensitive_sex.png",
    "FEAT-001": "src/adult_income_ml/pipelines.py",
    "SPLIT-001": "src/adult_income_ml/splitting.py",
    "MODEL-001": "results/models/tuned_logistic_regression.joblib",
    "EVAL-001": "results/metrics/final_test_metrics.json",
    "INT-001": "reports/figures/fig_13_permutation_importance.png",
    "INT-002": "results/shap/shap_summary.png",
    "FAIR-001": "reports/tables/table_13_fairness_sex.csv",
    "CAL-001": "reports/figures/fig_17_calibration.png",
    "ERR-001": "reports/tables/table_16_representative_errors.csv",
    "EXT-001": "reports/tables/table_17_extension.csv",
    "RPT-001": "reports/figures/",
}


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    root = paths["root"]

    # Copy artifacts from results/ to reports/ when not already present
    mappings = [
        (paths["shap"] / "shap_summary.png", "fig_14_shap_summary.png", "figures"),
        (paths["metrics"] / "final_test_metrics.json", "final_test_metrics.json", "tables"),
    ]
    for src, name, sub in mappings:
        if src.exists():
            copy_to_reports(src, name, sub)

    matrix = paths["reports"] / "traceability_matrix.md"
    validation = validate_traceability(matrix)
    if not validation.empty:
        missing = validation[~validation["complete"]]
        if len(missing) > 0:
            console.print(f"[yellow]{len(missing)} requirements missing evidence[/yellow]")
            for _, row in missing.iterrows():
                expected = EVIDENCE_MAP.get(row["id"], "")
                full = root / expected if expected else None
                exists = full.exists() if full else False
                console.print(f"  {row['id']}: {'OK' if exists else 'MISSING'} ({expected})")
        else:
            console.print("[bold green]All traceability rows have evidence[/bold green]")

    console.print("[bold green]RPT-001 report assets build complete[/bold green]")


if __name__ == "__main__":
    main()
