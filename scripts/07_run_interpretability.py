#!/usr/bin/env python
"""INT-001, INT-002, ERR-001: Interpretability and error analysis."""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import load_clean
from adult_income_ml.features import get_X_y
from adult_income_ml.interpretability import (
    get_representative_errors,
    run_permutation_importance,
    run_shap_analysis,
)
from adult_income_ml.reporting import export_table
from adult_income_ml.splitting import load_split_indices
from adult_income_ml.utils import console, get_paths, load_config, set_seed, tag_artifact


def main():
    set_seed()
    cfg = load_config()
    paths = get_paths(cfg)
    model = joblib.load(paths["models"] / "final_model.joblib")
    df = load_clean(cfg)
    _, test_idx = load_split_indices(cfg)
    test = df.loc[test_idx]
    X_test, y_test = get_X_y(test, cfg)
    y_pred = model.predict(X_test)

    imp = run_permutation_importance(model, X_test, y_test, n_repeats=5)
    export_table(imp.head(20), paths["tables"] / "table_perm_importance.csv")
    fig, ax = plt.subplots(figsize=(10, 6))
    imp.head(15).plot(x="feature", y="importance_mean", kind="barh", ax=ax)
    fig.savefig(
        paths["figures"] / "fig_13_permutation_importance.png",
        dpi=cfg["plotting"]["dpi"],
        bbox_inches="tight",
    )
    plt.close(fig)
    tag_artifact(paths["figures"] / "fig_13_permutation_importance.png", ["INT-001"])

    shap_path = run_shap_analysis(model, X_test, paths["shap"])
    if shap_path:
        tag_artifact(shap_path, ["INT-002"])

    errors = get_representative_errors(test, y_test.values, y_pred)
    export_table(errors, paths["tables"] / "table_16_representative_errors.csv")
    tag_artifact(paths["tables"] / "table_16_representative_errors.csv", ["ERR-001"])

    console.print("[bold green]Interpretability complete[/bold green]")


if __name__ == "__main__":
    main()
