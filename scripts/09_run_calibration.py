#!/usr/bin/env python
"""CAL-001: Calibration analysis."""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.calibration import calibration_summary_table
from adult_income_ml.data import load_clean
from adult_income_ml.features import get_X_y
from adult_income_ml.plotting import fig_calibration_curve
from adult_income_ml.reporting import export_table
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import console, get_paths, load_config, set_seed, tag_artifact
from sklearn.metrics import brier_score_loss


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    model = joblib.load(paths["models"] / "final_model.joblib")
    df = load_clean(cfg)
    _, test_idx = load_split_indices(cfg)
    test = df.loc[test_idx]
    X_test, y_test = get_X_y(test, cfg)
    y_proba = model.predict_proba(X_test)[:, 1]

    brier = brier_score_loss(y_test, y_proba)
    cal_df = calibration_summary_table(y_test, y_proba)
    export_table(cal_df, paths["tables"] / "table_15_calibration.csv")
    export_table(pd.DataFrame([{"brier_score": brier}]), paths["tables"] / "table_15_brier.csv")

    fig_calibration_curve(y_test, y_proba, paths["figures"] / "fig_17_calibration.png", cfg)
    tag_artifact(paths["figures"] / "fig_17_calibration.png", ["CAL-001"])

    console.print(f"[bold green]CAL-001 complete. Brier={brier:.4f}[/bold green]")


if __name__ == "__main__":
    main()
