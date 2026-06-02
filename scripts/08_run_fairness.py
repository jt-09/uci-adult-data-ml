#!/usr/bin/env python
"""FAIR-001 to FAIR-003, EXT-001: Fairness and extension analysis."""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.evaluation import compute_metrics
from adult_income_ml.fairness import intersectional_metrics, subgroup_metrics
from adult_income_ml.features import get_X_y
from adult_income_ml.models import get_model
from adult_income_ml.pipelines import build_model_pipeline
from adult_income_ml.plotting import fig_subgroup_metrics
from adult_income_ml.reporting import export_table
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import console, get_paths, load_config, save_json, set_seed, tag_artifact


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    model = joblib.load(paths["models"] / "final_model.joblib")
    df = load_clean(cfg)
    _, test_idx = load_split_indices(cfg)
    test = df.loc[test_idx]
    X_test, y_test = get_X_y(test, cfg)
    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    for attr, tid, tnum in [("sex", "FAIR-001", "13"), ("race", "FAIR-002", "14")]:
        sub = subgroup_metrics(test, y_test, y_pred, attr)
        export_table(sub, paths["tables"] / f"table_{tnum}_fairness_{attr}.csv")
        if not sub.empty:
            fig_subgroup_metrics(sub, paths["figures"] / f"fig_15_fairness_{attr}.png", cfg)
            tag_artifact(paths["tables"] / f"table_{tnum}_fairness_{attr}.csv", [tid])

    inter = intersectional_metrics(test, y_test, y_pred, cfg["columns"]["sensitive"])
    if not inter.empty:
        export_table(inter, paths["tables"] / "table_fairness_intersectional.csv")
        tag_artifact(paths["tables"] / "table_fairness_intersectional.csv", ["FAIR-003"])

    # EXT-001: model without sensitive attributes
    train_idx, _ = load_split_indices(cfg)
    train = df.loc[train_idx]
    X_tr, y_tr = get_X_y(train, cfg, drop_sensitive=True)
    X_te, y_te = get_X_y(test, cfg, drop_sensitive=True)
    ext_pipe = build_model_pipeline(get_model("logistic_regression", {}), cfg, drop_sensitive=True)
    ext_pipe.fit(X_tr, y_tr)
    y_pred_ext = ext_pipe.predict(X_te)
    y_proba_ext = ext_pipe.predict_proba(X_te)[:, 1]
    ext_metrics = compute_metrics(y_te.values, y_pred_ext, y_proba_ext)
    full_metrics = compute_metrics(y_test.values, y_pred.values, model.predict_proba(X_test)[:, 1])
    ext_table = pd.DataFrame([{"variant": "full", **full_metrics}, {"variant": "no_sensitive", **ext_metrics}])
    export_table(ext_table, paths["tables"] / "table_17_extension.csv")
    save_json(ext_metrics, paths["fairness"] / "extension_metrics.json")
    tag_artifact(paths["tables"] / "table_17_extension.csv", ["EXT-001", "EXT-002"])

    console.print("[bold green]Fairness analysis complete[/bold green]")


if __name__ == "__main__":
    main()
