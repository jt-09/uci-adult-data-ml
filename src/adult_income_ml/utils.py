"""Configuration, paths, logging, and I/O helpers."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from rich.console import Console

console = Console()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_project_root() -> Path:
    return PROJECT_ROOT


def load_yaml(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str = "project_config.yaml") -> dict[str, Any]:
    return load_yaml(get_project_root() / "configs" / name)


def load_model_spaces() -> dict[str, Any]:
    return load_yaml(get_project_root() / "configs" / "model_spaces.yaml")


def load_report_config() -> dict[str, Any]:
    return load_yaml(get_project_root() / "configs" / "report_config.yaml")


def get_paths(cfg: dict[str, Any] | None = None) -> dict[str, Path]:
    if cfg is None:
        cfg = load_config()
    root = get_project_root()
    p = cfg["paths"]
    return {
        "root": root,
        "raw": root / p["raw_dir"],
        "interim": root / p["interim_dir"],
        "processed": root / p["processed_dir"],
        "results": root / p["results_dir"],
        "reports": root / p["reports_dir"],
        "figures": root / p["reports_dir"] / "figures",
        "tables": root / p["reports_dir"] / "tables",
        "metrics": root / p["results_dir"] / "metrics",
        "models": root / p["results_dir"] / "models",
        "predictions": root / p["results_dir"] / "predictions",
        "cv_results": root / p["results_dir"] / "cv_results",
        "shap": root / p["results_dir"] / "shap",
        "fairness": root / p["results_dir"] / "fairness",
        "calibration": root / p["results_dir"] / "calibration",
        "splits": root / p["results_dir"] / "splits",
    }


def set_seed(seed: int | None = None) -> int:
    cfg = load_config()
    s = seed if seed is not None else int(cfg["project"]["seed"])
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import torch

        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass
    return s


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def tag_artifact(path: Path, requirement_ids: list[str]) -> Path:
    """Write sidecar JSON linking artifact to requirement IDs."""
    meta = path.with_suffix(path.suffix + ".meta.json")
    save_json({"path": str(path), "requirements": requirement_ids}, meta)
    return path


def get_n_jobs(cfg: dict[str, Any] | None = None) -> int:
    if cfg is None:
        cfg = load_config()
    n = cfg["project"].get("n_jobs", -1)
    env = os.environ.get("N_JOBS")
    if env is not None:
        return int(env)
    return int(n)
