# Traceability Matrix

| ID | Requirement | Evidence |
|----|-------------|----------|
| RQ-001 | Define research question and objectives | reports/report_outline.md |
| DATA-001 | Fetch Adult dataset reproducibly | data/raw/adult_raw.csv |
| DATA-002 | Create cleaned dataset | data/processed/adult_clean.csv |
| EDA-001 | Audit missing values and unknown categories | reports/figures/fig_01_missingness.png, reports/tables/table_01_dataset_summary.csv |
| EDA-002 | Audit target balance | reports/figures/fig_02_target_distribution.png |
| EDA-003 | Audit sensitive attributes | reports/figures/fig_03_sensitive_sex.png |
| FEAT-001 | Define preprocessing pipeline | src/adult_income_ml/pipelines.py |
| SPLIT-001 | Use stratified train/test split | results/splits/train_idx.pkl |
| MODEL-001 | Train logistic regression | results/models/tuned_logistic_regression.joblib, Section 7 |
| MODEL-002 | Train decision tree | results/models/tuned_decision_tree.joblib, Section 8 |
| MODEL-003 | Train random forest | results/models/tuned_random_forest.joblib, Section 9 |
| MODEL-004 | Train gradient boosting | results/models/tuned_xgboost.joblib, Section 10 |
| MODEL-005 | Train MLP | reports/tables/table_11_mlp_results.csv, Section 11 |
| EVAL-001 | Compare metrics on test set | results/metrics/final_test_metrics.json, Section 13 |
| INT-001 | Run permutation importance | reports/figures/fig_13_permutation_importance.png |
| INT-002 | Run SHAP analysis | reports/figures/fig_14_shap_summary.png, Section 14 |
| FAIR-001 | Evaluate by sex | reports/tables/table_13_fairness_sex.csv |
| FAIR-002 | Evaluate by race | reports/tables/table_14_fairness_race.csv |
| FAIR-003 | Intersectional groups | reports/tables/table_fairness_intersectional.csv |
| CAL-001 | Calibration curves and Brier | reports/figures/fig_17_calibration.png |
| ERR-001 | Representative FP/FN | reports/tables/table_16_representative_errors.csv |
| EXT-001 | Sensitive attributes removed | reports/tables/table_17_extension.csv |
| EXT-002 | Proxy analysis | reports/tables/table_17_extension.csv |
| RPT-001 | Generate report assets | reports/figures/ |
