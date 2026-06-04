---
name: sklearn-pipeline
description: >-
  Builds leakage-safe sklearn ColumnTransformer pipelines and stratified splits.
  Use for FEAT-001, SPLIT-001, preprocessing, pipelines.py, splitting.py.
---

# Sklearn Pipeline Skill

## Workflow

1. Split with `splitting.split_train_test` — stratified 80/20, persist indices to `results/splits/`.
2. Build preprocessor via `pipelines.build_preprocessor()` — fit ONLY on training data inside `Pipeline.fit`.
3. For extension (EXT-001), use `drop_sensitive=True`.

## Rules

- Never scale or impute before the train/test split.
- Never tune hyperparameters on the test set.
- Persist fitted pipelines with `joblib` to `results/models/`.
