# Contributing

Thanks for looking at this project. It started as a coursework-style ML study and is shared as a reference pipeline.

## Before you change modelling code

1. Do not use the held-out test set for tuning or model selection.
2. Keep preprocessing fit inside training data only (see `docs/experiment_protocol.md`).
3. Run tests after edits: `pytest tests/ -v` or `make test`.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Code style

- `ruff` and `black` with line length 100 (see `pyproject.toml`).
- `make lint` before opening a PR.

## Pull requests

- Describe which pipeline scripts you ran and whether figures/metrics changed.
- If metrics change, note it in `docs/decision_log.md` with a short rationale.
- Avoid committing `data/` or `results/` artifacts (gitignored).

## Reporting issues

Include Python version, OS, and the script step that failed. For tuning timeouts, note CPU/RAM available.
