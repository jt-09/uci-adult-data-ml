# Predicting High Income from Census Attributes

[![CI](https://github.com/jaythakkar/uci-adult-data-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/jaythakkar/uci-adult-data-ml/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end tabular ML project on the [UCI Adult (Census Income)](https://archive.ics.uci.edu/ml/datasets/adult) dataset: data cleaning, exploratory analysis, model comparison, held-out evaluation, interpretability, subgroup fairness checks, and calibration.

**Question:** Can we predict whether annual income exceeds $50K from census-style features, and what do model choice, preprocessing, and fairness diagnostics tell us about that prediction?

## Results at a glance

| Item | Value |
|------|-------|
| Selected model | LightGBM (tuned on train CV) |
| Test macro F1 | 0.818 |
| Test ROC-AUC | 0.930 |
| Test Brier score | 0.087 |
| Positive class rate | 23.9% (imbalanced) |

Full write-up: [`reports/report.md`](reports/report.md). PDF: [`reports/overleaf/main.pdf`](reports/overleaf/main.pdf).

Sample figures ship in `reports/figures/` (regenerate with the pipeline below if needed).

## Quick start

**Requirements:** Python 3.10+, network access for the UCI fetch.

```powershell
git clone https://github.com/jaythakkar/uci-adult-data-ml.git
cd uci-adult-data-ml

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install -e .

python scripts/00_setup_dirs.py
python scripts/01_fetch_data.py
python scripts/02_build_dataset.py
```

On macOS/Linux use `source .venv/bin/activate`. Quote paths that contain spaces.

Optional env overrides (see [`.env.example`](.env.example)): `RANDOM_SEED`, `N_JOBS`.

## Reproduce the full study

With [GNU Make](https://www.gnu.org/software/make/) installed:

```bash
make all
```

Without Make (same steps, Windows-friendly):

```bash
python scripts/00_setup_dirs.py
python scripts/01_fetch_data.py
python scripts/02_build_dataset.py
python scripts/03_run_eda.py
python scripts/04_train_baselines.py
python scripts/05_tune_models.py
python scripts/06_evaluate_final.py
python scripts/07_run_interpretability.py
python scripts/08_run_fairness.py
python scripts/09_run_calibration.py
python scripts/10_run_mlp_experiments.py
python scripts/11_build_report_assets.py
```

Tuning step (`05`) trains several model families with randomized search and can take 30-60 minutes on CPU. Figures land in `reports/figures/`; models and metrics in `results/` (gitignored).

Check report numbers after a run:

```bash
python scripts/12_verify_report_numbers.py
```

See [`docs/reproduction.md`](docs/reproduction.md) for figure-to-script mapping, runtime notes, and troubleshooting.

## Pipeline commands

| Command | Script | Notes |
|---------|--------|-------|
| `make setup` | install + `00_setup_dirs` | Editable install |
| `make fetch` | `01_fetch_data.py` | Downloads UCI Adult |
| `make build` | `02_build_dataset.py` | Cleaning + split indices |
| `make eda` | `03_run_eda.py` | Figures 1-6 |
| `make train` | `04_train_baselines.py` | Baseline fits |
| `make tune` | `05_tune_models.py` | CV search, saves final model |
| `make evaluate` | `06_evaluate_final.py` | Test metrics, figures 10-12 |
| `make interpret` | `07_run_interpretability.py` | Permutation + SHAP |
| `make fairness` | `08_run_fairness.py` | Subgroup metrics |
| `make calibrate` | `09_run_calibration.py` | Calibration curve |
| `make mlp` | `10_run_mlp_experiments.py` | sklearn + PyTorch MLP |
| `make report` | `11_build_report_assets.py` | Sync assets to `reports/` |
| `make verify` | `12_verify_report_numbers.py` | Cross-check tables vs report |
| `make test` | pytest | Unit tests |
| `make all` | Full pipeline | setup through report |

**Rule:** never tune or select models on the held-out test set. Selection uses 5-fold CV macro F1 on the training split only.

## Repository layout

```
src/adult_income_ml/   Core library (data, pipelines, models, evaluation)
scripts/               Numbered pipeline entry points (00-12)
configs/               Paths, columns, model search spaces
notebooks/             Optional notebooks (run after build/tune)
reports/               Report markdown, figures, tables, Overleaf source
results/               Generated models and metrics (gitignored)
docs/                  Protocol, reproduction guide, model cards
tests/                 pytest suite
```

Requirement-to-artifact mapping: [`reports/traceability_matrix.md`](reports/traceability_matrix.md).

## Notebooks

| Notebook | Run after |
|----------|-----------|
| `01_data_audit.ipynb` | `make build` |
| `02_eda_feature_selection.ipynb` | `make eda` |
| `03_model_experiments.ipynb` | `make tune` |
| `04_interpretability_fairness.ipynb` | `make fairness` |
| `05_report_figures.ipynb` | `make report` |

Notebooks load artifacts from `data/` and `results/`; they are not required to reproduce figures (scripts are).

## Ethics and limitations

`sex` and `race` are treated as sensitive attributes for auditing only. The dataset reflects a specific historical U.S. survey population. Outputs describe associations in this data; they are not suitable for individual hiring or credit decisions without further validation. See [`reports/limitations.md`](reports/limitations.md).

## Development

```bash
make test
make lint
```

Contributing notes: [`CONTRIBUTING.md`](CONTRIBUTING.md). Cursor agent layout: [`AGENTS.md`](AGENTS.md).

## Citation

If you use this repo, cite the UCI dataset and this project. See [`CITATION.cff`](CITATION.cff).

## License

MIT License. See [LICENSE](LICENSE). UCI Adult dataset terms apply to the underlying data.
