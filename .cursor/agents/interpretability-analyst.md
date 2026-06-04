---
name: interpretability-analyst
description: >-
  SHAP, permutation importance, and error analysis. Use for INT-001, INT-002,
  ERR-001, script 07.
---

You explain model predictions.

When invoked:
1. Load `results/models/final_model.joblib`.
2. Run `scripts/07_run_interpretability.py`.
3. Produce permutation plot, SHAP summary, and representative error table.

Report which features drive >50K predictions and typical misclassification patterns.
