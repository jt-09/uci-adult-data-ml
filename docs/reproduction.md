# Reproducing results

This page describes how to regenerate data, models, figures, and tables from a fresh clone.

## Prerequisites

- Python 3.10 or newer
- Network access for `scripts/01_fetch_data.py` (downloads UCI Adult via `ucimlrepo`)
- Roughly 2 GB free disk space after installs (PyTorch, SHAP, boosting libraries)

## One-command run

```bash
make all
```

`make all` runs `setup`, then scripts `01` through `11`. On Windows without Make, run the same scripts manually (listed in the README).

## What is and is not in git

| Path | In git? | Notes |
|------|---------|-------|
| `data/` | No | Fetched and built locally |
| `results/` | No | Models, metrics, split indices |
| `reports/figures/*.png` | Yes | Sample outputs for the showcase; overwrite with `make eda` etc. |
| `reports/tables/*.csv` | Yes | Table exports from the pipeline |
| `reports/report.md` | Yes | Narrative report |

After clone, you can view the committed report and figures immediately. To verify you can rebuild everything, run the full pipeline.

## Script order and outputs

| Step | Script | Main outputs |
|------|--------|--------------|
| 0 | `00_setup_dirs.py` | Directory tree |
| 1 | `01_fetch_data.py` | `data/raw/adult_raw.csv` |
| 2 | `02_build_dataset.py` | `data/processed/adult_clean.csv`, `results/splits/*` |
| 3 | `03_run_eda.py` | `fig_01` - `fig_06` |
| 4 | `04_train_baselines.py` | Baseline CV summaries |
| 5 | `05_tune_models.py` | `results/models/final_model.joblib`, tuning JSON |
| 6 | `06_evaluate_final.py` | Test metrics, `fig_10` - `fig_12` |
| 7 | `07_run_interpretability.py` | `fig_13`, `results/shap/shap_summary.png` |
| 8 | `08_run_fairness.py` | Fairness tables, `fig_15_*` |
| 9 | `09_run_calibration.py` | `fig_17`, calibration tables |
| 10 | `10_run_mlp_experiments.py` | `fig_09`, MLP metrics |
| 11 | `11_build_report_assets.py` | Copies SHAP to `fig_14`, syncs tables |
| 12 | `12_verify_report_numbers.py` | `reports/verification_report.md` |

## Figure reference

| Figure | Produced by |
|--------|-------------|
| `fig_01_missingness.png` | `03_run_eda.py` |
| `fig_02_target_distribution.png` | `03_run_eda.py` |
| `fig_03_sensitive_sex.png`, `fig_03_sensitive_race.png` | `03_run_eda.py` |
| `fig_04_numeric_by_target.png` | `03_run_eda.py` |
| `fig_05_categorical_rates.png` | `03_run_eda.py` |
| `fig_06_correlation.png` | `03_run_eda.py` |
| `fig_09_mlp_training_curve.png` | `10_run_mlp_experiments.py` |
| `fig_10_cross_model_comparison.png` | `06_evaluate_final.py` |
| `fig_11_confusion_matrix.png` | `06_evaluate_final.py` |
| `fig_12_roc.png`, `fig_12_pr.png` | `06_evaluate_final.py` |
| `fig_13_permutation_importance.png` | `07_run_interpretability.py` |
| `fig_14_shap_summary.png` | `07` then `11` (copy from `results/shap/`) |
| `fig_15_fairness_sex.png`, `fig_15_fairness_race.png` | `08_run_fairness.py` |
| `fig_17_calibration.png` | `09_run_calibration.py` |

Figures 7, 8, and 16 from the original plan were not implemented; per-model learning curves are summarized in tables and JSON instead.

## Configuration

- Seed: `configs/project_config.yaml` (`project.seed: 42`)
- Override via `.env`: `RANDOM_SEED`, `N_JOBS`
- Model search spaces: `configs/model_spaces.yaml`
- Tuning uses `RandomizedSearchCV` with up to 10 iterations per model in `05_tune_models.py` (config lists 20; script caps at 10 for runtime)

## Runtime expectations

| Stage | Rough time (CPU) |
|-------|------------------|
| Fetch + build | Under 1 minute |
| EDA | 1-2 minutes |
| Tune (`05`) | 30-60 minutes |
| Evaluate through report | 10-20 minutes |

GPU is optional; PyTorch MLP runs on CPU by default.

## Verification

```bash
python scripts/12_verify_report_numbers.py
```

Reads tables and metric JSON files and writes `reports/verification_report.md`. Run only after steps 1-11 complete.

## Common issues

**`FileNotFoundError` for split indices or `final_model.joblib`**

Run `02_build_dataset.py` and `05_tune_models.py` before later scripts.

**SHAP figure missing**

`07_run_interpretability.py` may skip SHAP on failure. Check console output; ensure `shap` installed and final model exists.

**XGBoost or LightGBM skipped during tuning**

Install errors are caught per model; check pip install and rerun `05`.

**Paths with spaces**

Quote the repo path in shell commands (common on OneDrive).

**Make not found on Windows**

Use the manual script list in the README or install Make via Chocolatey/WSL.

## Tests

```bash
pytest tests/ -v
```

Tests cover cleaning, features, pipelines, and metrics on small synthetic data. They do not run the full tuning pipeline.
