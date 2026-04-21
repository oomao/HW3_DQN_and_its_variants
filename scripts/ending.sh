#!/usr/bin/env bash
# dev:ending — validate changes、寫 handover、可選 push
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if command -v openspec >/dev/null 2>&1; then
  for d in openspec/changes/*/; do
    name="$(basename "$d")"
    [ "$name" = "archive" ] && continue
    echo "[ending] validating $name"
    openspec validate "$name" || true
  done
fi

HANDOVER="docs/HANDOVER.md"
mkdir -p docs
{
  echo "# Handover"
  echo
  echo "_Last updated: $(date -u +%Y-%m-%dT%H:%M:%SZ)_"
  echo
  echo "## Current branch"
  echo "- $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '(no git)')"
  echo
  echo "## Active openspec changes"
  ls openspec/changes 2>/dev/null | grep -v '^archive$' | sed 's/^/- /' || echo "- (none)"
  echo
  echo "## Next actions"
  echo "- Run scripts/startup.sh to resume."
  echo "- See README.md for reproducing figures."
} > "$HANDOVER"

if [ -n "${PUSH:-}" ]; then
  git add -A
  git commit -m "dev:ending checkpoint" || echo "(nothing to commit)"
  git push
else
  echo "[ending] set PUSH=1 to commit & push automatically"
fi
