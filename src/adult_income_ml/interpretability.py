"""Permutation importance and SHAP analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from adult_income_ml.utils import ensure_dir, get_n_jobs, load_config


def run_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    cfg = load_config()
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=42, n_jobs=get_n_jobs(cfg), scoring="f1_macro"
    )
    names = X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])]
    if hasattr(model, "named_steps"):
        # Transformed feature names not trivial; use indices
        names = [f"feature_{i}" for i in range(len(result.importances_mean))]
    return pd.DataFrame(
        {
            "feature": names[: len(result.importances_mean)],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)


def run_shap_analysis(
    model,
    X_sample: pd.DataFrame,
    out_dir: Path,
    max_samples: int = 500,
) -> Path | None:
    ensure_dir(out_dir)
    cfg = load_config()
    X = X_sample.head(max_samples)
    try:
        if hasattr(model, "named_steps"):
            pre = model.named_steps["preprocessor"]
            clf = model.named_steps["classifier"]
            X_t = pre.transform(X)
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_t)
            else:
                explainer = shap.LinearExplainer(clf, X_t)
                shap_values = explainer.shap_values(X_t)
        else:
            explainer = shap.Explainer(model.predict_proba, X)
            shap_values = explainer(X)
    except Exception:
        return None

    summary_path = out_dir / "shap_summary.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_t if "X_t" in dir() else X, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
    plt.close()
    return summary_path


def get_representative_errors(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_each: int = 5,
) -> pd.DataFrame:
    work = df.copy()
    work["y_true"] = y_true
    work["y_pred"] = y_pred
    fp = work[(work["y_true"] == 0) & (work["y_pred"] == 1)].head(n_each)
    fn = work[(work["y_true"] == 1) & (work["y_pred"] == 0)].head(n_each)
    fp = fp.assign(error_type="false_positive")
    fn = fn.assign(error_type="false_negative")
    return pd.concat([fp, fn], ignore_index=True)
