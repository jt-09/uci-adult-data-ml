# Agent orchestration

Cursor agent layout for this repo. Human docs: README and `docs/`.

## Pipeline order

```
00_setup_dirs -> 01_fetch -> 02_build -> 03_eda -> 04_baselines -> 05_tune ->
06_evaluate -> 07_interpret -> 08_fairness -> 09_calibrate -> 10_mlp -> 11_report
```

Run via `make all` (includes `setup`). Optional: `make verify` for script 12.

## Subagent mapping

| Scripts | Subagent |
|---------|----------|
| 01-02 | `data-pipeline` |
| 03 | `eda-analyst` |
| 04-06, 10 | `model-trainer` |
| 07 | `interpretability-analyst` |
| 08 | `fairness-analyst` |
| 11 | `report-builder` |

## Skills

See `.cursor/skills/` and `docs/cursor_skills/README.md`.

## Critical rule

**Never tune or select models using the held-out test set.**
