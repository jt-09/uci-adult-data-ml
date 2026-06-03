"""Report asset management and traceability."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd

from adult_income_ml.utils import ensure_dir, get_paths, load_config


def copy_to_reports(src: Path, dest_name: str, subdir: str = "figures") -> Path:
    paths = get_paths()
    dest_dir = paths["reports"] / subdir
    ensure_dir(dest_dir)
    dest = dest_dir / dest_name
    if src.exists() and src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def update_traceability_evidence(matrix_path: Path, req_id: str, evidence: str) -> None:
    if not matrix_path.exists():
        return
    text = matrix_path.read_text(encoding="utf-8")
    pattern = rf"(\| {re.escape(req_id)} \|[^\|]+\|)[^\|]*(\|)"
    replacement = rf"\1 {evidence} \2"
    new_text = re.sub(pattern, replacement, text, count=1)
    matrix_path.write_text(new_text, encoding="utf-8")


def validate_traceability(matrix_path: Path) -> pd.DataFrame:
    rows = []
    if not matrix_path.exists():
        return pd.DataFrame()
    for line in matrix_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and not line.startswith("| ID") and not line.startswith("|--"):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3 and parts[0] and parts[0] != "ID":
                rows.append(
                    {
                        "id": parts[0],
                        "requirement": parts[1],
                        "evidence": parts[2] if len(parts) > 2 else "",
                        "complete": bool(parts[2] and parts[2] not in ("—", "-", "TBD", "")),
                    }
                )
    return pd.DataFrame(rows)


def export_table(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)
    return path
