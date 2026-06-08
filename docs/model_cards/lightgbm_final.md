# Model card: LightGBM (final classifier)

## Model

Gradient-boosted decision trees (`lightgbm.LGBMClassifier`) inside a sklearn `Pipeline` with numeric standardization and categorical one-hot encoding.

## Intended use

Research and education on tabular income classification with the UCI Adult dataset. Not for individual employment or credit decisions.

## Training data

- Source: UCI Adult, cleaned to 48,813 rows
- Split: stratified 80% train (39,050 rows), 20% test (9,763 rows)
- Features: 14 inputs (see `docs/data_dictionary.md`)
- Target: binary income above $50K

## Selection

Chosen by highest 5-fold CV macro F1 on the training split only (script `05_tune_models.py`). Test set used once in script `06`.

## Performance (held-out test)

| Metric | Value |
|--------|-------|
| Macro F1 | 0.818 |
| ROC-AUC | 0.930 |
| Brier score | 0.087 |
| Balanced accuracy | 0.802 |

## Limitations

- Trained on 1990s census-style data; poor external validity to other populations or eras.
- Subgroup recall gaps by sex and race (see fairness tables in `reports/tables/`).
- Encoded feature names in importance plots do not map directly to raw column labels without inspecting the fitted preprocessor.

## Ethical notes

Sensitive attributes are included in training. Removing sex and race in the extension experiment lowers macro F1 by about 0.035; proxy features remain informative.
