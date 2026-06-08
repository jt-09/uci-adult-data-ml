---
name: report-writer
description: >-
  Syncs figures and tables to reports/, updates traceability matrix, validates
  RPT-001. Use for report.md, traceability_matrix.md, script 11.
---

# Report Writer Skill

## Workflow

1. Run `scripts/11_build_report_assets.py`.
2. Copy artifacts from `results/` to `reports/figures/` and `reports/tables/`.
3. Update `reports/traceability_matrix.md` evidence column.
4. Fill `reports/report.md` placeholders with figure/table references.

## Rules

- Figure numbering: `fig_01_` through `fig_18_` per report_config.yaml.
- Every requirement ID must have a linked file path in the matrix.
