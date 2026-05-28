#!/usr/bin/env python
"""Ensure all project directories exist."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.utils import ensure_dir, get_paths, load_config

DIRS = [
    "configs",
    "data/raw",
    "data/interim",
    "data/processed",
    "data/external",
    "notebooks",
    "scripts",
    "reports/figures",
    "reports/tables",
    "results/metrics",
    "results/models",
    "results/predictions",
    "results/cv_results",
    "results/shap",
    "results/fairness",
    "results/calibration",
    "results/splits",
    "tests",
    "docs/model_cards",
    "docs/cursor_skills",
]


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    for d in DIRS:
        ensure_dir(ROOT / d)
    for p in paths.values():
        if isinstance(p, Path) and p != paths["root"]:
            ensure_dir(p)
    print("Directories ready.")


if __name__ == "__main__":
    main()
