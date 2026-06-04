---
name: interpretability
description: >-
  Runs permutation importance, SHAP analysis, and representative error tables.
  Use for INT-001, INT-002, ERR-001, interpretability.py, script 07.
---

# Interpretability Skill

## Workflow

1. Load final model from `results/models/final_model.joblib`.
2. Run permutation importance → `results/shap/` or `reports/tables/table_16_errors.csv`.
3. SHAP: TreeExplainer for tree models; LinearExplainer or skip on failure.
4. Export representative FP/FN rows (ERR-001).

## Outputs

- Figure 12: permutation importance
- Figures 13-14: SHAP summary and dependence
- Table 16: representative errors
