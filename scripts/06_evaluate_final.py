#!/usr/bin/env python
"""EVAL-001: Final test-set evaluation."""

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.evaluation import build_comparison_table, compute_metrics, confusion_matrix_df
from adult_income_ml.features import get_X_y
from adult_income_ml.plotting import fig_confusion_matrix, fig_model_comparison, fig_roc_pr
from adult_income_ml.reporting import export_table
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import console, get_paths, load_config, load_json, save_json, set_seed, tag_artifact


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)

    model_path = paths["models"] / "final_model.joblib"
    if not model_path.exists():
        console.print("[red]Run 05_tune_models.py first[/red]")
        sys.exit(1)

    model = joblib.load(model_path)
    df = load_clean(cfg)
    train_idx, test_idx = load_split_indices(cfg)
    test = df.loc[test_idx]
    X_test, y_test = get_X_y(test, cfg)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.values, y_pred, y_proba)
    save_json(metrics, paths["metrics"] / "final_test_metrics.json")
    tag_artifact(paths["metrics"] / "final_test_metrics.json", ["EVAL-001"])

    cm = confusion_matrix_df(y_test, y_pred)
    export_table(cm.reset_index(), paths["tables"] / "table_12_final_test_metrics.csv")
    export_table(
        __import__("pandas").DataFrame([metrics]),
        paths["tables"] / "table_12_final_metrics_detail.csv",
    )

    fig_confusion_matrix(
        __import__("numpy").array(cm),
        paths["figures"] / "fig_11_confusion_matrix.png",
        cfg,
    )
    fig_roc_pr(y_test, y_proba, paths["figures"] / "fig_12_roc.png", paths["figures"] / "fig_12_pr.png", cfg)

    # Cross-model comparison if tuning results exist
    cv_path = paths["cv_results"] / "tuning_results.json"
    if cv_path.exists():
        cv = load_json(cv_path)
        rows = [{"model": k, "f1_macro_cv": v["best_score"]} for k, v in cv.items()]
        comp = __import__("pandas").DataFrame(rows)
        export_table(comp, paths["tables"] / "table_11_cross_model_comparison.csv")
        fig_model_comparison(comp, "f1_macro_cv", paths["figures"] / "fig_10_cross_model_comparison.png", cfg)

    np.savez(paths["predictions"] / "test_predictions.npz", y_true=y_test, y_pred=y_pred, y_proba=y_proba)
    console.print("[bold green]EVAL-001 complete[/bold green]")


if __name__ == "__main__":
    main()
