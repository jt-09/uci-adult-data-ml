"""Permutation importance and SHAP analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from adult_income_ml.utils import ensure_dir, get_n_jobs, load_config

logger = logging.getLogger(__name__)


def get_transformed_feature_names(model) -> list[str]:
    """Return post-preprocessing feature names from a fitted sklearn Pipeline."""
    if not hasattr(model, "named_steps"):
        return []
    pre = model.named_steps["preprocessor"]
    if hasattr(pre, "get_feature_names_out"):
        return list(pre.get_feature_names_out())
    return []


def prettify_feature_name(name: str) -> str:
    """Map sklearn ColumnTransformer names to readable labels."""
    if name.startswith("num__"):
        return name.removeprefix("num__")
    if name.startswith("cat__"):
        rest = name.removeprefix("cat__")
        if "_" in rest:
            col, value = rest.split("_", 1)
            return f"{col}: {value}"
        return rest
    return name


def prettify_feature_names(names: list[str]) -> list[str]:
    return [prettify_feature_name(n) for n in names]


def _positive_class_shap_values(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    return shap_values


def run_permutation_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
) -> pd.DataFrame:
    cfg = load_config()
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=get_n_jobs(cfg),
        scoring="f1_macro",
    )
    if hasattr(X, "columns"):
        names = X.columns.tolist()
    else:
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
            raw_names = get_transformed_feature_names(model)
            pretty_names = prettify_feature_names(raw_names) if raw_names else None
            if pretty_names and len(pretty_names) == X_t.shape[1]:
                X_plot = pd.DataFrame(X_t, columns=pretty_names)
            else:
                X_plot = pd.DataFrame(X_t, columns=[f"feature_{i}" for i in range(X_t.shape[1])])
            if hasattr(clf, "feature_importances_"):
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer.shap_values(X_t)
            else:
                explainer = shap.LinearExplainer(clf, X_t)
                shap_values = explainer.shap_values(X_t)
            shap_pos = _positive_class_shap_values(shap_values)
        else:
            explainer = shap.Explainer(model.predict_proba, X)
            shap_values = explainer(X)
            X_plot = X
            shap_pos = shap_values.values if hasattr(shap_values, "values") else shap_values
    except Exception as exc:
        logger.warning("SHAP analysis failed: %s", exc)
        return None

    summary_path = out_dir / "shap_summary.png"
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_pos, X_plot, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
    plt.close()

    bar_path = out_dir / "shap_summary_bar.png"
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_pos, X_plot, plot_type="bar", show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
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
