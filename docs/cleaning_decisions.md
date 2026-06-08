# Cleaning decisions

Summary of how raw UCI Adult rows become `data/processed/adult_clean.csv`. Implementation: `src/adult_income_ml/cleaning.py`, script `02_build_dataset.py`.

## Missing values

- The raw file uses `?` for unknown categorical values.
- These are converted to pandas `NA` before imputation in the modelling pipeline.

## Duplicates

- Exact duplicate rows are dropped (`drop_duplicates: true` in config).
- 29 duplicates removed in the reference run (48,842 raw rows to 48,813 clean).

## Target encoding

| Raw label | Encoded |
|-----------|---------|
| `<=50K` | 0 |
| `>50K` | 1 |

Positive class (income above $50K) rate after cleaning: about 23.9%.

## Features retained

Fourteen input columns plus target: six numeric, eight categorical. Full list in `configs/project_config.yaml` and `docs/data_dictionary.md`.

## Sensitive attributes

`sex` and `race` stay in the primary modelling pipeline. They are used for fairness auditing (script `08`) and can be dropped in the extension experiment (script logic in fairness/extension modules).

## What we do not do

- No row filtering by occupation or geography beyond deduplication.
- No manual removal of rare categories before one-hot encoding (unknown levels at test time use `handle_unknown="ignore"` in the pipeline).
