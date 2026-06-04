# Findings Log

| Date | Finding | Evidence |
|------|---------|----------|
| 2026-06-02 | 48,813 rows after cleaning; 23.9% positive class | table_01, Section 3 |
| 2026-06-02 | Stratified split: 39,050 train / 9,763 test; balanced prevalence | table_04, Section 6 |
| 2026-06-02 | LightGBM best CV macro F1 (0.815); selected as final model | tuning_results.json, Section 10 |
| 2026-06-02 | Test macro F1 0.818, ROC-AUC 0.930 for final LightGBM | final_test_metrics.json, Section 13 |
| 2026-06-02 | Logistic regression CV macro F1 0.780; weakest linear baseline | Section 7 |
| 2026-06-02 | sklearn MLP test macro F1 0.791; embedding MLP 0.559 | table_11_mlp_results, Section 11 |
| 2026-06-02 | Female FNR 0.454 vs Male 0.315 on test set | table_13, Section 15 |
| 2026-06-02 | Black recall 0.496 vs White 0.670 | table_14, Section 15 |
| 2026-06-02 | Brier score 0.087; reasonable calibration | table_15_brier, Section 16 |
| 2026-06-02 | Removing sex/race drops macro F1 from 0.818 to 0.783 | table_17_extension, Section 17 |
| 2026-06-02 | Permutation/SHAP on encoded features post-pipeline | fig_13, fig_14, Section 14 |
| 2026-06-02 | FP errors: managerial profiles below $50K threshold | table_16, Section 14 |
