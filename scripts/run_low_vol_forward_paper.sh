#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_ROOT="${AUDIT_ROOT:-$HOME/quant-alpha-audit-v2}"
AUDIT_V1_ROOT="${AUDIT_V1_ROOT:-$HOME/quant-alpha-audit}"
PYTHON_BIN="${AUDIT_PYTHON:-$AUDIT_V1_ROOT/.venv/bin/python}"

export AUDIT_ROOT AUDIT_V1_ROOT
cd "$REPO_ROOT"
exec "$PYTHON_BIN" audit/scripts/stage_v2_forward.py "$@"
