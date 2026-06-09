#!/usr/bin/env bash
set -euo pipefail
SRC="../reports/figures"
DEST="figures"
mkdir -p "$DEST"
find "$SRC" -maxdepth 1 -name '*.png' -exec cp -f {} "$DEST/" \;
echo "Synced $(ls -1 "$DEST"/*.png | wc -l) figures into book/figures/"
