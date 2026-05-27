# Experiment Protocol

1. **Reproducibility**: `project.seed=42` in config; set via `set_seed()`.
2. **Split**: Stratified 80/20 train/test; indices in `results/splits/`.
3. **Tuning**: 5-fold CV macro F1 on train only; `RandomizedSearchCV` per model.
4. **Selection**: Best CV macro F1 → `final_model.joblib`.
5. **Test evaluation**: Single held-out test evaluation in script 06.
6. **Fairness**: Subgroup metrics with minimum n=30.
7. **Extension**: Retrain logistic model without sex/race columns.
