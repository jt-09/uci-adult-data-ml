# Results Snapshot

Pinned metrics from the UCI Adult income classification study. Generated: 2026-07-07 15:11 UTC.

Reproduce the full pipeline with `make all`. See [docs/reproduction.md](docs/reproduction.md).

## Selected model

- **Model:** LightGBM (5-fold CV macro F1 on train split)
- **CV macro F1 (train only):** 0.815
- **Selection rule:** highest CV macro F1; test set used once for final evaluation

## Held-out test metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.874 |
| Balanced accuracy | 0.802 |
| Precision (macro) | 0.839 |
| Recall (macro) | 0.802 |
| **F1 (macro)** | **0.818** |
| ROC-AUC | 0.930 |
| PR-AUC | 0.836 |
| Brier score | 0.087 |

## Fairness highlights (test set)

| Group | Recall |
|-------|--------|
| Female | 0.546 |
| Male | 0.685 |
| Black | 0.496 |
| White | 0.670 |

Full tables: [table_13](reports/tables/table_13_fairness_sex.csv), [table_14](reports/tables/table_14_fairness_race.csv).

## Artifacts

- Academic report: [reports/report.md](reports/report.md)
- PDF: [reports/overleaf/main.pdf](reports/overleaf/main.pdf)
- Developer log (Quarto): https://jt-09.github.io/uci-adult-data-ml/
- Figure index: [reports/artifact_inventory.md](reports/artifact_inventory.md)
- Traceability: [reports/traceability_matrix.md](reports/traceability_matrix.md)

## Citation

See [CITATION.cff](CITATION.cff) and the UCI Adult dataset.
