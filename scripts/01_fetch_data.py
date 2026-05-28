#!/usr/bin/env python
"""DATA-001: Fetch UCI Adult dataset."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adult_income_ml.data import fetch_adult
from adult_income_ml.utils import console, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed()
    path = fetch_adult(force=args.force)
    console.print(f"[bold green]DATA-001 complete:[/bold green] {path}")


if __name__ == "__main__":
    main()
