#!/usr/bin/env bash
set -euo pipefail

AUDIT_ROOT="${AUDIT_V221_ROOT:-/var/lib/quant-lab/forward_v221}"
REPO_PATH="${QUANT_LAB_FORWARD_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${QUANT_LAB_FORWARD_PYTHON:-python}"
MARKET_BAR_PATH="${QUANT_LAB_FORWARD_MARKET_BAR_PATH:-/var/lib/quant-lab/lake/silver/market_bar/data.parquet}"
LOCK_PATH="${AUDIT_ROOT}/state/forward_v221.lock"

mkdir -p "$(dirname "${LOCK_PATH}")"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
  echo "forward_v221_concurrent_run_blocked" >&2
  exit 75
fi

exec "${PYTHON_BIN}" "${REPO_PATH}/audit/scripts/stage_v221_forward.py" \
  --mode realtime \
  --root "${AUDIT_ROOT}" \
  --repo "${REPO_PATH}" \
  --market-bar "${MARKET_BAR_PATH}" \
  --resume \
  "$@"
