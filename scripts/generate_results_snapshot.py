#!/usr/bin/env python
"""Generate RESULTS.md snapshot from committed report tables."""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.utils import get_paths, load_config


def main():
    cfg = load_config()
    paths = get_paths(cfg)
    tables = paths["tables"]
    metrics = pd.read_csv(tables / "table_12_final_metrics_detail.csv")
    row = metrics.iloc[0]
    sex = pd.read_csv(tables / "table_13_fairness_sex.csv")
    race = pd.read_csv(tables / "table_14_fairness_race.csv")

    female = sex.loc[sex["group"] == "Female", "recall"].iloc[0]
    male = sex.loc[sex["group"] == "Male", "recall"].iloc[0]
    black = race.loc[race["group"] == "Black", "recall"].iloc[0]
    white = race.loc[race["group"] == "White", "recall"].iloc[0]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    content = f"""# Results Snapshot

Pinned metrics from the UCI Adult income classification study. Generated: {now}.

Reproduce the full pipeline with `make all`. See [docs/reproduction.md](docs/reproduction.md).

## Selected model

- **Model:** LightGBM (5-fold CV macro F1 on train split)
- **CV macro F1 (train only):** 0.815
- **Selection rule:** highest CV macro F1; test set used once for final evaluation

## Held-out test metrics

| Metric | Value |
|--------|-------|
| Accuracy | {row['accuracy']:.3f} |
| Balanced accuracy | {row['balanced_accuracy']:.3f} |
| Precision (macro) | {row['precision_macro']:.3f} |
| Recall (macro) | {row['recall_macro']:.3f} |
| **F1 (macro)** | **{row['f1_macro']:.3f}** |
| ROC-AUC | {row['roc_auc']:.3f} |
| PR-AUC | {row['pr_auc']:.3f} |
| Brier score | {row['brier_score']:.3f} |

## Fairness highlights (test set)

| Group | Recall |
|-------|--------|
| Female | {female:.3f} |
| Male | {male:.3f} |
| Black | {black:.3f} |
| White | {white:.3f} |

Full tables: [table_13](reports/tables/table_13_fairness_sex.csv), [table_14](reports/tables/table_14_fairness_race.csv).

## Artifacts

- Academic report: [reports/report.md](reports/report.md)
- PDF: [reports/overleaf/main.pdf](reports/overleaf/main.pdf)
- Developer log (Quarto): https://jt-09.github.io/uci-adult-data-ml/
- Figure index: [reports/artifact_inventory.md](reports/artifact_inventory.md)
- Traceability: [reports/traceability_matrix.md](reports/traceability_matrix.md)

## Citation

See [CITATION.cff](CITATION.cff) and the UCI Adult dataset.
"""
    out = paths["root"] / "RESULTS.md"
    out.write_text(content, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
