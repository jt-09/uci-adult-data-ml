#!/usr/bin/env bash
set -euo pipefail
SRC="../reports/figures"
DEST="figures"
ASSETS="../reports/overleaf"
mkdir -p "$DEST" assets
find "$SRC" -maxdepth 1 -name '*.png' -exec cp -f {} "$DEST/" \;
if [ -f "$ASSETS/main.pdf" ]; then
  cp -f "$ASSETS/main.pdf" assets/main.pdf
fi
echo "Synced $(ls -1 "$DEST"/*.png 2>/dev/null | wc -l) figures into book/figures/"
if [ -f assets/main.pdf ]; then
  echo "Synced main.pdf into book/assets/"
fi
