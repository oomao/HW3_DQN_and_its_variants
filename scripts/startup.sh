#!/usr/bin/env bash
# dev:start — pull、讀 handover、列出 openspec active changes
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[startup] git pull --ff-only"
git pull --ff-only || echo "[startup] (no upstream or nothing to pull)"

HANDOVER="docs/HANDOVER.md"
if [ -f "$HANDOVER" ]; then
  echo; echo "===== $HANDOVER ====="; cat "$HANDOVER"; echo "====================="
fi

if command -v openspec >/dev/null 2>&1; then
  echo; echo "[startup] active openspec changes:"
  ls openspec/changes 2>/dev/null | grep -v '^archive$' || echo "(none)"
fi

echo; echo "[startup] Suggested next actions:"
echo "  1. PYTHONPATH=src python -m hw3.train_naive"
echo "  2. PYTHONPATH=src python -m hw3.train_variants"
echo "  3. PYTHONPATH=src python -m hw3.train_lightning"
