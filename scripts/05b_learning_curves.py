#!/usr/bin/env python
"""MODEL-001 to MODEL-004: Export per-model learning curves (Figures 7-8)."""

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.evaluation import run_learning_curve
from adult_income_ml.features import get_X_y
from adult_income_ml.plotting import fig_learning_curve
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import (
    console,
    ensure_dir,
    get_paths,
    load_config,
    set_seed,
    tag_artifact,
)

DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    lc_cfg = cfg.get("learning_curves", {})
    models = lc_cfg.get("models", ["logistic_regression", "random_forest", "lightgbm"])
    train_sizes = lc_cfg.get("train_sizes", [0.1, 0.25, 0.5, 0.75, 1.0])
    cv_folds = cfg["cv"]["n_splits"]
    scoring = cfg["cv"]["scoring"]

    df = load_clean(cfg)
    train_idx, _ = load_split_indices(cfg)
    train = df.loc[train_idx]
    X, y = get_X_y(train, cfg)
    ensure_dir(paths["figures"])

    for model_name in models:
        model_path = paths["models"] / f"tuned_{model_name}.joblib"
        if not model_path.exists():
            console.print(f"[yellow]Skip learning curve for {model_name}: model not found[/yellow]")
            continue
        pipe = joblib.load(model_path)
        console.print(f"Learning curve: {model_name} ...")
        sizes, train_scores, val_scores = run_learning_curve(
            pipe, X, y, train_sizes=train_sizes, cv=cv_folds, scoring=scoring
        )
        label = DISPLAY_NAMES.get(model_name, model_name)
        if model_name == "logistic_regression":
            out = paths["figures"] / "fig_07_logistic_learning_curve.png"
            req = ["MODEL-001"]
        else:
            out = paths["figures"] / f"fig_08_{model_name}_learning_curve.png"
            req = ["MODEL-003", "MODEL-004"]
        fig_learning_curve(
            sizes,
            train_scores,
            val_scores,
            title=f"Learning curve: {label}",
            path=out,
            cfg=cfg,
        )
        tag_artifact(out, req)
        console.print(f"  saved {out.name}")

    console.print("[bold green]Learning curves complete[/bold green]")


if __name__ == "__main__":
    main()
