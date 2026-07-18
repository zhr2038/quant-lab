#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT="${LOCAL_AUDIT_ROOT:-/home/hr/quant-alpha-audit}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${AUDIT_ROOT}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "audit virtualenv missing: ${PYTHON_BIN}" >&2
  exit 2
fi

export LOCAL_AUDIT_ROOT="${AUDIT_ROOT}"
exec "${PYTHON_BIN}" "${REPO_ROOT}/audit/scripts/audit_runner.py" "$@"
