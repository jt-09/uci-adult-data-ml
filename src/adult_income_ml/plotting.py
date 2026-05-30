"""Figure generation for EDA and evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from adult_income_ml.utils import ensure_dir, load_config


def _style(cfg: dict | None = None):
    cfg = cfg or load_config()
    try:
        plt.style.use(cfg["plotting"]["style"])
    except OSError:
        plt.style.use("ggplot")
    sns.set_palette("husl")


def save_fig(fig, path: Path, cfg: dict | None = None):
    cfg = cfg or load_config()
    ensure_dir(path.parent)
    fig.savefig(path, dpi=cfg["plotting"]["dpi"], bbox_inches="tight")
    plt.close(fig)


def fig_missingness(df: pd.DataFrame, path: Path, cfg: dict | None = None):
    _style(cfg)
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        miss = pd.Series({"none": 0})
    fig, ax = plt.subplots(figsize=(10, 6))
    miss.plot(kind="barh", ax=ax)
    ax.set_title("Missing values by column")
    ax.set_xlabel("Count")
    save_fig(fig, path, cfg)


def fig_target_distribution(y: pd.Series, path: Path, cfg: dict | None = None):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(8, 5))
    y.value_counts().sort_index().plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_xticklabels(["<=50K (0)", ">50K (1)"], rotation=0)
    ax.set_title("Target distribution")
    ax.set_ylabel("Count")
    save_fig(fig, path, cfg)


def fig_income_by_group(df: pd.DataFrame, col: str, target: str, path: Path, cfg: dict | None = None):
    _style(cfg)
    rate = df.groupby(col, dropna=False)[target].mean().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6))
    rate.plot(kind="barh", ax=ax)
    ax.set_title(f"High-income rate by {col}")
    ax.set_xlabel("Proportion >50K")
    save_fig(fig, path, cfg)


def fig_numeric_by_target(df: pd.DataFrame, numeric_cols: list[str], target: str, path: Path, cfg: dict | None = None):
    _style(cfg)
    n = len(numeric_cols)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols[:6]):
        for label in df[target].unique():
            subset = df.loc[df[target] == label, col].dropna()
            axes[i].hist(subset, alpha=0.5, bins=30, label=str(label))
        axes[i].set_title(col)
        axes[i].legend()
    fig.tight_layout()
    save_fig(fig, path, cfg)


def fig_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], path: Path, cfg: dict | None = None):
    _style(cfg)
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Numeric feature correlation")
    save_fig(fig, path, cfg)


def fig_confusion_matrix(cm: np.ndarray, path: Path, cfg: dict | None = None):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    save_fig(fig, path, cfg)


def fig_roc_pr(y_true, y_proba, path_roc: Path, path_pr: Path, cfg: dict | None = None):
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

    _style(cfg)
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC curve")
    save_fig(fig, path_roc, cfg)
    fig, ax = plt.subplots(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("Precision-Recall curve")
    save_fig(fig, path_pr, cfg)


def fig_model_comparison(df: pd.DataFrame, metric: str, path: Path, cfg: dict | None = None):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(10, 6))
    df.plot(x="model", y=metric, kind="bar", ax=ax, legend=False)
    ax.set_title(f"Model comparison: {metric}")
    ax.set_ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    save_fig(fig, path, cfg)


def fig_calibration_curve(y_true, y_proba, path: Path, cfg: dict | None = None):
    from sklearn.calibration import calibration_curve

    _style(cfg)
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()
    save_fig(fig, path, cfg)


def fig_subgroup_metrics(df: pd.DataFrame, path: Path, cfg: dict | None = None):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.2
    for i, col in enumerate(["fpr", "fnr", "precision", "recall"]):
        if col in df.columns:
            ax.bar(x + i * w, df[col], width=w, label=col)
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(df["group"], rotation=45, ha="right")
    ax.legend()
    ax.set_title("Subgroup metrics")
    save_fig(fig, path, cfg)


def fig_training_curve(history: dict, path: Path, cfg: dict | None = None):
    _style(cfg)
    fig, ax = plt.subplots(figsize=(8, 5))
    if "loss" in history:
        ax.plot(history["loss"], label="loss")
    ax.set_xlabel("Epoch")
    ax.set_title("Training curve")
    ax.legend()
    save_fig(fig, path, cfg)
