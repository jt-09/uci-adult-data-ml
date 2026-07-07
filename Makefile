.PHONY: setup fetch build eda train tune learning-curves evaluate interpret fairness calibrate mlp report verify test lint demo all dirs

PYTHON ?= python
PIP ?= pip

setup: dirs
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dirs:
	$(PYTHON) scripts/00_setup_dirs.py

fetch:
	$(PYTHON) scripts/01_fetch_data.py

build:
	$(PYTHON) scripts/02_build_dataset.py

eda:
	$(PYTHON) scripts/03_run_eda.py

train:
	$(PYTHON) scripts/04_train_baselines.py

tune:
	$(PYTHON) scripts/05_tune_models.py
	$(PYTHON) scripts/05b_learning_curves.py

learning-curves:
	$(PYTHON) scripts/05b_learning_curves.py

evaluate:
	$(PYTHON) scripts/06_evaluate_final.py

interpret:
	$(PYTHON) scripts/07_run_interpretability.py

fairness:
	$(PYTHON) scripts/08_run_fairness.py

calibrate:
	$(PYTHON) scripts/09_run_calibration.py

mlp:
	$(PYTHON) scripts/10_run_mlp_experiments.py

report:
	$(PYTHON) scripts/11_build_report_assets.py

results:
	$(PYTHON) scripts/generate_results_snapshot.py

verify:
	$(PYTHON) scripts/12_verify_report_numbers.py

test:
	pytest tests/ -v

lint:
	ruff check src scripts tests
	black --check src scripts tests

demo:
	$(PYTHON) demo/app.py

all: setup fetch build eda train tune evaluate interpret fairness calibrate mlp report
