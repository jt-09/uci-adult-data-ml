# Architecture

High-level view of how code, configs, and artifacts connect.

## Data flow

```
UCI (ucimlrepo)
    -> 01_fetch -> data/raw/adult_raw.csv
    -> 02_build -> data/processed/adult_clean.csv
                 -> results/splits/train_idx.pkl, test_idx.pkl

train_idx + features
    -> 04_baselines / 05_tune -> results/cv_results/tuning_results.json
                               -> results/models/final_model.joblib

final_model + test_idx
    -> 06_evaluate -> results/metrics/final_test_metrics.json
                   -> reports/figures (eval plots)
    -> 07_interpret -> permutation + SHAP
    -> 08_fairness -> subgroup tables and plots
    -> 09_calibrate -> calibration plot and Brier score
    -> 10_mlp -> neural baseline metrics

all artifacts -> 11_report -> reports/tables, reports/figures sync
```

## Package layout (`src/adult_income_ml/`)

| Module | Role |
|--------|------|
| `data.py` | Fetch and load raw/clean CSV |
| `cleaning.py` | Missing tokens, dedup, target encoding |
| `features.py` | Column role definitions |
| `splitting.py` | Stratified train/test split |
| `pipelines.py` | sklearn `ColumnTransformer` + estimator pipelines |
| `models.py` | Estimator factory |
| `evaluation.py` | Metrics and CV helpers |
| `plotting.py` | Shared matplotlib helpers |
| `interpretability.py` | Permutation importance, SHAP |
| `fairness.py` | Subgroup rates |
| `calibration.py` | Reliability curve, Brier |
| `mlp_torch.py` | PyTorch embedding MLP |
| `reporting.py` | Copy/tag artifacts for `reports/` |
| `utils.py` | Config load, paths, seeds |

## Configuration

| File | Purpose |
|------|---------|
| `configs/project_config.yaml` | Seed, paths, column lists, sensitive attributes |
| `configs/model_spaces.yaml` | Hyperparameter distributions per model |
| `configs/report_config.yaml` | Figure/table naming for the report |

Single source of truth for column names and paths. Scripts call `load_config()` from `utils.py`.

## Leakage controls

1. Train/test split before any preprocessor fit (`02_build_dataset.py` persists indices).
2. Preprocessing inside sklearn `Pipeline` or fit on train slice only.
3. `RandomizedSearchCV` on training data only; macro F1 for selection.
4. Script `06` is the only step that scores the held-out test set for the final model.

See [`experiment_protocol.md`](experiment_protocol.md) and [`decision_log.md`](decision_log.md).

## Results directory

| Subdir | Contents |
|--------|----------|
| `results/splits/` | Pickled train/test indices, split summary JSON |
| `results/cv_results/` | Tuning results per model |
| `results/models/` | `final_model.joblib` |
| `results/metrics/` | Final test metrics JSON |
| `results/shap/` | SHAP summary image (pre-copy) |
| `results/fairness/` | Subgroup metric exports |
| `results/calibration/` | Calibration bins |

All under `results/` are gitignored. Empty subdirs use `.gitkeep` files in git; `00_setup_dirs.py` also creates them on first run.

## Reports directory

| Subdir | Contents |
|--------|----------|
| `reports/figures/` | PNG figures + optional `.meta.json` sidecars |
| `reports/tables/` | CSV tables for the write-up |
| `reports/overleaf/` | LaTeX source and compiled PDF |

Open `reports/report.md` in an editor with Markdown preview; image paths are relative to that file.

## Traceability

Scripts tag outputs with requirement IDs (e.g. `DATA-001`, `EVAL-001`) via `tag_artifact()` or `.meta.json` sidecars. The matrix in `reports/traceability_matrix.md` links IDs to files.
