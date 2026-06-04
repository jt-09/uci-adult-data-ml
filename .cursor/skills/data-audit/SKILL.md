---
name: data-audit
description: >-
  Audits UCI Adult dataset fetch, cleaning, missing values, target balance, and
  sensitive attributes. Use for DATA-001, DATA-002, EDA-001, EDA-002, EDA-003,
  Figures 1-3, Tables 1-3.
---

# Data Audit Skill

## Workflow

1. Run `scripts/01_fetch_data.py` → `data/raw/adult_raw.csv` (DATA-001).
2. Run `scripts/02_build_dataset.py` → `data/processed/adult_clean.csv` (DATA-002).
3. Run `scripts/03_run_eda.py` for EDA figures/tables.

## Rules

- Do NOT fit models during audit scripts.
- Tag outputs with requirement IDs via `tag_artifact()`.
- Use `configs/project_config.yaml` for column names and missing token `?`.

## Outputs

| ID | Artifact |
|----|----------|
| EDA-001 | `reports/figures/fig_01_missingness.png`, `reports/tables/table_01_dataset_summary.csv` |
| EDA-002 | `reports/figures/fig_02_target_distribution.png` |
| EDA-003 | `reports/figures/fig_03_sensitive_audit.png` |
