---
name: fairness
description: >-
  Computes subgroup FPR, FNR, precision, recall by sex and race and intersectional
  groups. Use for FAIR-001, FAIR-002, FAIR-003, EXT-001, EXT-002, script 08.
---

# Fairness Skill

## Workflow

1. Evaluate final model predictions on test set with subgroup splits.
2. `fairness.subgroup_metrics` for `sex` and `race`.
3. `fairness.intersectional_metrics` when `n >= min_subgroup_size` (30).
4. Extension: compare full vs `drop_sensitive` pipeline (Section 17).

## Outputs

- Tables 13-14, Figures 15-16
- Extension table in `reports/tables/table_17_extension.csv`
