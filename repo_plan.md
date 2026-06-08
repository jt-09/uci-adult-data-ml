# Repository plan (internal)

Planning notes from initial project setup. Public docs live in README and `docs/`.

## Title

Predicting High Income from Census Attributes: classification, interpretability, and fairness on UCI Adult.

## Directory layout
adult-income-ml-study/
│
├── README.md
├── pyproject.toml
├── requirements.txt
├── Makefile
├── .gitignore
├── .env.example
│
├── configs/
│   ├── project_config.yaml
│   ├── model_spaces.yaml
│   └── report_config.yaml
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── interim/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── external/
│       └── .gitkeep
│
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_eda_feature_selection.ipynb
│   ├── 03_model_experiments.ipynb
│   ├── 04_interpretability_fairness.ipynb
│   └── 05_report_figures.ipynb
│
├── src/
│   └── adult_income_ml/
│       ├── __init__.py
│       ├── data.py
│       ├── cleaning.py
│       ├── features.py
│       ├── splitting.py
│       ├── pipelines.py
│       ├── models.py
│       ├── evaluation.py
│       ├── interpretability.py
│       ├── fairness.py
│       ├── calibration.py
│       ├── plotting.py
│       ├── reporting.py
│       └── utils.py
│
├── scripts/
│   ├── 00_setup_dirs.py
│   ├── 01_fetch_data.py
│   ├── 02_build_dataset.py
│   ├── 03_run_eda.py
│   ├── 04_train_baselines.py
│   ├── 05_tune_models.py
│   ├── 06_evaluate_final.py
│   ├── 07_run_interpretability.py
│   ├── 08_run_fairness.py
│   ├── 09_run_calibration.py
│   ├── 10_run_mlp_experiments.py
│   └── 11_build_report_assets.py
│
├── reports/
│   ├── report.md
│   ├── report_outline.md
│   ├── traceability_matrix.md
│   ├── findings_log.md
│   ├── limitations.md
│   ├── references.bib
│   ├── figures/
│   │   └── .gitkeep
│   └── tables/
│       └── .gitkeep
│
├── results/
│   ├── metrics/
│   ├── models/
│   ├── predictions/
│   ├── cv_results/
│   ├── shap/
│   ├── fairness/
│   └── calibration/
│
├── tests/
│   ├── test_data.py
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_pipelines.py
│   └── test_evaluation.py
│
└── docs/
    ├── experiment_protocol.md
    ├── model_cards/
    ├── data_dictionary.md
    ├── decision_log.md
    └── cursor_skills/
        ├── data_audit_skill.md
        ├── sklearn_pipeline_skill.md
        ├── interpretability_skill.md
        ├── fairness_skill.md
        ├── mlp_skill.md
        └── report_writer_skill.md
Research framing
Main research question

To what extent can income above $50K/year be predicted from demographic, education, employment, and financial attributes in the Adult / Census Income dataset, and how do model family, preprocessing choices, calibration, interpretability, and subgroup fairness diagnostics affect the strength and reliability of that prediction?

Supporting objectives
Build a leakage-aware and reproducible binary classification dataset.
Compare classical models and neural MLP models under the same split and metric protocol.
Evaluate whether stronger predictive performance comes at the cost of interpretability or subgroup imbalance.
Use SHAP, permutation importance, coefficients, calibration, and subgroup confusion matrices to explain model behavior.
Produce a figure/table-heavy report with numbered sections and an appendix.
Planned report structure

Section outline implemented in `reports/report.md`:

1 Introduction
2 Research Focus
3 Dataset and Task Validity
4 Exploratory Data Analysis and Feature Selection
5 Target Balance and Sensitive Attribute Audit
6 Modelling Approach
7 Logistic Regression
8 Decision Tree
9 Random Forest
10 Gradient Boosting
11 MLP Classifier
12 Cross-Model Comparison
13 Final Model Evaluation
14 Interpretability and Error Analysis
15 Fairness and Subgroup Diagnostics
16 Calibration Analysis
17 Extension: Removing Sensitive Attributes / Proxy Analysis
18 Reflection on Learning
19 Conclusion
20 References
A Appendix

This mirrors a standard applied ML report layout (intro through conclusion plus appendix).

Traceability system

Every output should map back to a requirement, figure, table, or report claim.

Requirement IDs
ID	Requirement	Evidence artifact
RQ-001	Define research question and objectives	reports/report_outline.md
DATA-001	Fetch Adult dataset reproducibly	scripts/01_fetch_data.py, data/raw/
DATA-002	Create cleaned dataset	scripts/02_build_dataset.py, data/processed/adult_clean.csv
EDA-001	Audit missing values and unknown categories	Figure 1, Table 1
EDA-002	Audit target balance	Figure 2, Table 2
EDA-003	Audit sensitive attributes	Figure 3, Table 3
FEAT-001	Define preprocessing pipeline	src/adult_income_ml/pipelines.py
SPLIT-001	Use stratified train/test split	src/adult_income_ml/splitting.py
MODEL-001	Train logistic regression baseline	Section 7
MODEL-002	Train decision tree	Section 8
MODEL-003	Train random forest	Section 9
MODEL-004	Train gradient boosting	Section 10
MODEL-005	Train MLP	Section 11
EVAL-001	Compare accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC	Tables 6-11
INT-001	Run permutation importance	Figure 12
INT-002	Run SHAP analysis	Figures 13-14
FAIR-001	Evaluate by sex	Table 13, Figure 15
FAIR-002	Evaluate by race	Table 14, Figure 16
FAIR-003	Evaluate by intersectional groups if sample size allows	Appendix
CAL-001	Calibration curves and Brier score	Figure 17, Table 15
ERR-001	Representative false positives and false negatives	Table 16
EXT-001	Compare with sensitive attributes removed	Section 17
EXT-002	Proxy analysis after removing sex/race	Section 17
RPT-001	Generate final report assets	reports/figures/, reports/tables/
Model plan

Use these model families:

Model	Why included
Dummy classifier	Shows majority-class baseline
Logistic regression	Interpretable linear baseline
Decision tree	Transparent threshold model
Random forest	Nonlinear ensemble baseline
HistGradientBoosting / XGBoost / LightGBM	Strong tabular model
MLP with one-hot encoding	Neural baseline
MLP with categorical embeddings	More advanced neural tabular experiment

Main selection metric:

Mean 5-fold cross-validated macro F1 on the training split

Also report:

Accuracy
Balanced accuracy
Precision
Recall
Macro F1
Weighted F1
ROC-AUC
PR-AUC
Brier score
Calibration curve
Confusion matrix
Subgroup false positive rate
Subgroup false negative rate
Subgroup recall
Subgroup precision
Figure and table plan
Figures
Missingness and unknown-category audit
Target distribution
Income rate by feature groups
Numeric feature distributions by target
Categorical feature target rates
Correlation / association heatmap
Logistic regression tuning results
Decision tree tuning results
Random forest tuning results
Gradient boosting tuning results
MLP training curves
Cross-model comparison
Final confusion matrix
ROC curve and precision-recall curve
Permutation importance
SHAP summary plot
SHAP dependence plots for top features
Subgroup performance by sex
Subgroup performance by race
Calibration curve
Confidence distribution for correct vs incorrect predictions
Extension comparison: full features vs sensitive attributes removed
Tables
Dataset summary
Feature dictionary
Cleaning decisions
Sensitive/proxy feature audit
Train/test split summary
Model search spaces
Logistic regression results
Decision tree results
Random forest results
Gradient boosting results
MLP results
Cross-model comparison
Final test-set metrics
Subgroup metrics by sex
Subgroup metrics by race
Representative errors
Calibration metrics
Extension results
Environment setup

Use this as the actual terminal setup:

python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

Suggested requirements.txt:

pandas
numpy
scikit-learn
scipy
matplotlib
seaborn
shap
ucimlrepo
jupyter
ipykernel
pyyaml
joblib
pytest
black
ruff
mypy
rich
tabulate
markdown

Optional if you want XGBoost/LightGBM/CatBoost:

xgboost
lightgbm
catboost

Optional for PyTorch MLP embeddings:

torch
torchmetrics