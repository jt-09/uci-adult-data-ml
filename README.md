# Predicting High Income from Census Attributes

**A supervised machine-learning analysis of the UCI Adult / Census Income dataset**

Binary classification, interpretability, and fairness study on whether income exceeds $50K/year can be predicted from demographic, education, employment, and financial attributes.

## Research question

To what extent can income above $50K/year be predicted from census attributes, and how do model family, preprocessing, calibration, interpretability, and subgroup fairness diagnostics affect the strength and reliability of that prediction?

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .
```

On macOS/Linux, activate with `source .venv/bin/activate`.

> **Note:** This workspace path may contain spaces (OneDrive). Quote paths in commands when needed.

## Pipeline

| Command | Script | Requirement IDs |
|---------|--------|-----------------|
| `make dirs` | `00_setup_dirs.py` | — |
| `make fetch` | `01_fetch_data.py` | DATA-001 |
| `make build` | `02_build_dataset.py` | DATA-002 |
| `make eda` | `03_run_eda.py` | EDA-001–003 |
| `make train` | `04_train_baselines.py` | MODEL-* baselines |
| `make tune` | `05_tune_models.py` | MODEL-001–004 |
| `make evaluate` | `06_evaluate_final.py` | EVAL-001 |
| `make interpret` | `07_run_interpretability.py` | INT-001/002 |
| `make fairness` | `08_run_fairness.py` | FAIR-001–003 |
| `make calibrate` | `09_run_calibration.py` | CAL-001 |
| `make mlp` | `10_run_mlp_experiments.py` | MODEL-005 |
| `make report` | `11_build_report_assets.py` | RPT-001 |
| `make test` | pytest | — |
| `make all` | Full pipeline | — |

## Project layout

- `src/adult_income_ml/` — core library
- `scripts/` — numbered pipeline scripts
- `configs/` — YAML configuration
- `notebooks/` — narrative analysis
- `reports/` — report markdown, figures, tables (open `reports/report.md` for preview; images use `./figures/` relative to that file)
- `results/` — metrics, models, predictions (gitignored)
- `.cursor/skills/` — Cursor agent skills
- `.cursor/agents/` — specialized subagents

See [`AGENTS.md`](AGENTS.md) for Cursor orchestration and [`reports/traceability_matrix.md`](reports/traceability_matrix.md) for requirement-to-artifact mapping.

## Ethics

Sensitive attributes (`sex`, `race`) are audited for fairness and optionally removed in extension experiments. Results describe patterns in a historical survey dataset and must not be used for individual-level decisions without further validation.

## License

Academic / educational use. UCI Adult dataset: see [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/adult).
