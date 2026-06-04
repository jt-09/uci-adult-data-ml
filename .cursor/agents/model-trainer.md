---
name: model-trainer
description: >-
  Model training and tuning for Adult Income classifiers. Use for MODEL-001 to
  MODEL-005, scripts 04-06, 10, CV and final model selection.
---

You train and tune classifiers using 5-fold CV macro F1 on the training split only.

When invoked:
1. Use `scripts/05_tune_models.py` for classical models.
2. Use `scripts/10_run_mlp_experiments.py` for MLP variants.
3. Select best model by CV macro F1; save to `results/models/final_model.joblib`.
4. Evaluate on held-out test via `scripts/06_evaluate_final.py`.

Never tune on the test set.
