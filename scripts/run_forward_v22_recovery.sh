#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${QLAB_AUDIT_PYTHON:-/home/hr/quant-alpha-audit/.venv/bin/python}"

exec "$python_bin" "$repo_root/audit/scripts/stage_v22_forward.py" \
  --mode recovery \
  --repo "$repo_root" \
  "$@"
