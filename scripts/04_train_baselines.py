#!/usr/bin/env python
"""Train dummy and quick baseline models."""

import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.evaluation import run_cross_validation
from adult_income_ml.features import get_X_y
from adult_income_ml.models import get_model
from adult_income_ml.pipelines import build_model_pipeline
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import console, ensure_dir, get_paths, load_config, save_json, set_seed


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    ensure_dir(paths["metrics"])

    df = load_clean(cfg)
    train_idx, _ = load_split_indices(cfg)
    train = df.loc[train_idx]
    X, y = get_X_y(train, cfg)

    results = {}
    for name in ["dummy", "logistic_regression"]:
        est = get_model(name, {})
        pipe = build_model_pipeline(est, cfg)
        cv = run_cross_validation(
            pipe, X, y, cv=cfg["cv"]["n_splits"], scoring=cfg["cv"]["scoring"]
        )
        pipe.fit(X, y)
        results[name] = cv
        joblib.dump(pipe, paths["models"] / f"baseline_{name}.joblib")

    save_json(results, paths["metrics"] / "baseline_cv.json")
    console.print("[bold green]Baselines trained[/bold green]")


if __name__ == "__main__":
    main()
