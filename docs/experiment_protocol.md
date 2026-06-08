# Experiment protocol

Rules followed across scripts `01` to `12`.

1. **Reproducibility:** default seed 42 in `configs/project_config.yaml`; set in code via `set_seed()`. Override with `RANDOM_SEED` in `.env` if needed.

2. **Split:** stratified 80/20 train/test. Indices saved under `results/splits/` during `02_build_dataset.py`.

3. **Tuning:** 5-fold cross-validation on the training split only. Primary score: macro F1. Search: `RandomizedSearchCV` per model family (`configs/model_spaces.yaml`).

4. **Selection:** model with best CV macro F1 is saved as `results/models/final_model.joblib`. The test set is not used for this step.

5. **Test evaluation:** script `06_evaluate_final.py` runs one held-out evaluation for the selected model.

6. **Fairness:** subgroup metrics computed on test predictions. Groups with fewer than 30 rows are flagged or omitted per `fairness.py`.

7. **Extension:** logistic regression retrained without `sex` and `race` on the same splits for comparison (EXT-001).

8. **Artifacts:** figures and tables copied or tagged into `reports/` with requirement IDs. See `reports/traceability_matrix.md`.
