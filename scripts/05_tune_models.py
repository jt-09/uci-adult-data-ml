#!/usr/bin/env python
"""MODEL-001 to MODEL-004: Hyperparameter tuning with RandomizedSearchCV."""

import sys
from pathlib import Path

import joblib
from sklearn.model_selection import RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.features import get_X_y
from adult_income_ml.models import get_model, get_param_grid
from adult_income_ml.pipelines import build_model_pipeline
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import (
    console,
    ensure_dir,
    get_n_jobs,
    get_paths,
    load_config,
    load_model_spaces,
    save_json,
    set_seed,
)

MODELS = [
    ("logistic_regression", "MODEL-001"),
    ("decision_tree", "MODEL-002"),
    ("random_forest", "MODEL-003"),
    ("hist_gradient_boosting", "MODEL-004"),
    ("xgboost", "MODEL-004"),
    ("lightgbm", "MODEL-004"),
]


def main():
    set_seed()
    cfg = load_config()
    spaces = load_model_spaces()
    paths = get_paths(cfg)
    ensure_dir(paths["cv_results"])
    ensure_dir(paths["models"])

    df = load_clean(cfg)
    train_idx, _ = load_split_indices(cfg)
    train = df.loc[train_idx]
    X, y = get_X_y(train, cfg)

    tuning = spaces["tuning"]
    all_cv = {}
    best_score = -1.0
    best_name = None
    best_est = None

    for model_name, req_id in MODELS:
        try:
            grid = get_param_grid(model_name)
            if not grid:
                continue
            est = get_model(model_name, {})
            pipe = build_model_pipeline(est, cfg)
            param_grid = {f"classifier__{k}": v for k, v in grid.items()}
            search = RandomizedSearchCV(
                pipe,
                param_distributions=param_grid,
                n_iter=min(tuning["n_iter"], 10),
                cv=tuning["cv_folds"],
                scoring=tuning["scoring"],
                random_state=tuning["random_state"],
                n_jobs=get_n_jobs(cfg),
            )
            search.fit(X, y)
            all_cv[model_name] = {
                "best_score": float(search.best_score_),
                "best_params": search.best_params_,
                "requirement": req_id,
            }
            joblib.dump(search.best_estimator_, paths["models"] / f"tuned_{model_name}.joblib")
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_name = model_name
                best_est = search.best_estimator_
            console.print(f"{model_name}: CV {search.best_score_:.4f}")
        except Exception as e:
            console.print(f"[yellow]Skip {model_name}: {e}[/yellow]")

    save_json(all_cv, paths["cv_results"] / "tuning_results.json")
    if best_est is not None:
        joblib.dump(best_est, paths["models"] / "final_model.joblib")
        save_json(
            {"model": best_name, "cv_f1_macro": best_score}, paths["metrics"] / "best_model.json"
        )
    console.print(f"[bold green]Tuning complete. Best: {best_name} ({best_score:.4f})[/bold green]")


if __name__ == "__main__":
    main()
