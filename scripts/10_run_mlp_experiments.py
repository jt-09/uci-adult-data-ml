#!/usr/bin/env python
"""MODEL-005: MLP sklearn and PyTorch embedding experiments."""

import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.evaluation import compute_metrics, run_cross_validation
from adult_income_ml.features import get_feature_columns, get_X_y
from adult_income_ml.models import get_model, get_param_grid
from adult_income_ml.mlp_torch import predict_proba_torch, train_embedding_mlp
from adult_income_ml.pipelines import build_model_pipeline
from adult_income_ml.plotting import fig_training_curve
from adult_income_ml.reporting import export_table
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import console, get_n_jobs, get_paths, load_config, load_model_spaces, save_json, set_seed, tag_artifact


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    df = load_clean(cfg)
    train_idx, test_idx = load_split_indices(cfg)
    train = df.loc[train_idx]
    test = df.loc[test_idx]
    X_train, y_train = get_X_y(train, cfg)
    X_test, y_test = get_X_y(test, cfg)
    feat = get_feature_columns(cfg)

    # Sklearn MLP
    grid = get_param_grid("mlp_sklearn")
    est = get_model("mlp_sklearn", {})
    pipe = build_model_pipeline(est, cfg)
    param_grid = {f"classifier__{k}": v for k, v in grid.items()}
    spaces = load_model_spaces()["tuning"]
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_grid,
        n_iter=5,
        cv=spaces["cv_folds"],
        scoring=spaces["scoring"],
        random_state=42,
        n_jobs=get_n_jobs(cfg),
    )
    search.fit(X_train, y_train)
    joblib.dump(search.best_estimator_, paths["models"] / "tuned_mlp_sklearn.joblib")
    sk_metrics = compute_metrics(
        y_test.values,
        search.predict(X_test),
        search.predict_proba(X_test)[:, 1],
    )

    # PyTorch embedding MLP
    torch_cfg = load_model_spaces()["models"]["mlp_torch"]
    model_torch, history, encoders = train_embedding_mlp(
        X_train,
        y_train,
        feat["numeric"],
        feat["categorical"],
        embedding_dim=torch_cfg["embedding_dim"][0],
        hidden_dims=torch_cfg["hidden_dims"][0],
        dropout=torch_cfg["dropout"][0],
        lr=torch_cfg["lr"][0],
        epochs=torch_cfg["epochs"][0],
        batch_size=torch_cfg["batch_size"][0],
    )
    proba_torch = predict_proba_torch(
        model_torch, X_test, feat["numeric"], feat["categorical"], encoders
    )
    torch_metrics = compute_metrics(y_test.values, (proba_torch >= 0.5).astype(int), proba_torch)
    save_json(history, paths["metrics"] / "mlp_torch_history.json")
    fig_training_curve(history, paths["figures"] / "fig_09_mlp_training_curve.png", cfg)
    tag_artifact(paths["figures"] / "fig_09_mlp_training_curve.png", ["MODEL-005"])

    results = pd.DataFrame(
        [{"model": "mlp_sklearn", **sk_metrics}, {"model": "mlp_torch_embedding", **torch_metrics}]
    )
    export_table(results, paths["tables"] / "table_11_mlp_results.csv")
    save_json({"sklearn": sk_metrics, "torch": torch_metrics}, paths["metrics"] / "mlp_results.json")

    console.print("[bold green]MODEL-005 MLP experiments complete[/bold green]")


if __name__ == "__main__":
    main()
