# Artifact Inventory (RPT-001)

| Report # | Filename | Requirement ID | Section |
|----------|----------|----------------|---------|
| Figure 1 | `figures/fig_01_missingness.png` | EDA-001 | 4 |
| Figure 2 | `figures/fig_02_target_distribution.png` | EDA-002 | 5 |
| Figure 3a | `figures/fig_03_sensitive_sex.png` | EDA-003 | 5 |
| Figure 3b | `figures/fig_03_sensitive_race.png` | EDA-003 | 5 |
| Figure 4 | `figures/fig_04_numeric_by_target.png` | EDA-001 | 4 |
| Figure 5 | `figures/fig_05_categorical_rates.png` | EDA-001 | 4 |
| Figure 6 | `figures/fig_06_correlation.png` | EDA-001 | 4 |
| Figure 9 | `figures/fig_09_mlp_training_curve.png` | MODEL-005 | 11 |
| Figure 10 | `figures/fig_10_cross_model_comparison.png` | EVAL-001 | 12 |
| Figure 11 | `figures/fig_11_confusion_matrix.png` | EVAL-001 | 13 |
| Figure 12a | `figures/fig_12_roc.png` | EVAL-001 | 13 |
| Figure 12b | `figures/fig_12_pr.png` | EVAL-001 | 13 |
| Figure 13 | `figures/fig_13_permutation_importance.png` | INT-001 | 14 |
| Figure 14 | `figures/fig_14_shap_summary.png` | INT-002 | 14 |
| Figure 15a | `figures/fig_15_fairness_sex.png` | FAIR-001 | 15 |
| Figure 15b | `figures/fig_15_fairness_race.png` | FAIR-002 | 15 |
| Figure 17 | `figures/fig_17_calibration.png` | CAL-001 | 16 |
| Table 1 | `tables/table_01_dataset_summary.csv` | EDA-001 | 3 |
| Table 2 | `tables/table_02_feature_dictionary.csv` | EDA-001 | 4 |
| Table 3 | `tables/table_03_cleaning_decisions.csv` | DATA-002 | 3 |
| Table 4 | `tables/table_04_split_summary.csv` | SPLIT-001 | 6 |
| Table 11 | `tables/table_11_cross_model_comparison.csv` | EVAL-001 | 12 |
| Table 11b | `tables/table_11_mlp_results.csv` | MODEL-005 | 11 |
| Table 12 | `tables/table_12_final_metrics_detail.csv` | EVAL-001 | 13 |
| Table 13 | `tables/table_13_fairness_sex.csv` | FAIR-001 | 15 |
| Table 14 | `tables/table_14_fairness_race.csv` | FAIR-002 | 15 |
| Table 15 | `tables/table_15_brier.csv` | CAL-001 | 16 |
| Table 16 | `tables/table_16_representative_errors.csv` | ERR-001 | 14 |
| Table 17 | `tables/table_17_extension.csv` | EXT-001, EXT-002 | 17 |
| Intersectional | `tables/table_fairness_intersectional.csv` | FAIR-003 | Appendix |

**Note:** Per-model tuning curve figures (repo plan Figures 7–10) were not generated; CV scores and hyperparameters are reported via Table 11 and `results/cv_results/tuning_results.json`.

**Artifact gate:** PASS (all required files present as of inventory build).
