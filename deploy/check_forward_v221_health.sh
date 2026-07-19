#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${QUANT_LAB_FORWARD_REPO:-/opt/quant-lab-forward-v221}"
AUDIT_ROOT="${AUDIT_V221_ROOT:-/var/lib/quant-lab/forward_v221}"
PYTHON_BIN="${QUANT_LAB_FORWARD_PYTHON:-/opt/quant-lab/.venv/bin/python}"
MARKET_BAR_PATH="${QUANT_LAB_FORWARD_MARKET_BAR_PATH:-/var/lib/quant-lab/lake/silver/market_bar/data.parquet}"
SYSTEMCTL_BIN="${QUANT_LAB_FORWARD_SYSTEMCTL:-systemctl}"
SERVICE_UNIT="${QUANT_LAB_FORWARD_SERVICE_UNIT:-/etc/systemd/system/quant-lab-forward-v221.service}"
TIMER_UNIT="${QUANT_LAB_FORWARD_TIMER_UNIT:-/etc/systemd/system/quant-lab-forward-v221.timer}"

export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${REPO_PATH}"

timer_installed=false
timer_enabled=false
timer_active=false
[[ -f "${TIMER_UNIT}" ]] && timer_installed=true
[[ "$(${SYSTEMCTL_BIN} is-enabled quant-lab-forward-v221.timer 2>/dev/null || true)" == "enabled" ]] && timer_enabled=true
[[ "$(${SYSTEMCTL_BIN} is-active quant-lab-forward-v221.timer 2>/dev/null || true)" == "active" ]] && timer_active=true
service_result="$(${SYSTEMCTL_BIN} show quant-lab-forward-v221.service --property=Result --value 2>/dev/null || true)"
[[ -n "${service_result}" ]] || service_result="never-run"
next_trigger="$(${SYSTEMCTL_BIN} list-timers quant-lab-forward-v221.timer --no-legend --no-pager 2>/dev/null | awk '{$1=$1; print $1" "$2" "$3" "$4}' || true)"
[[ -n "${next_trigger}" ]] || next_trigger="UNKNOWN"
disk_free_bytes="$(stat -f --format='%a*%S' "${AUDIT_ROOT}" | awk -F'*' '{print $1*$2}')"
market_staleness="$(${PYTHON_BIN} - "${MARKET_BAR_PATH}" <<'PY'
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
import polars as pl
p = Path(sys.argv[1])
schema = pl.scan_parquet(p).collect_schema()
lf = pl.scan_parquet(p)
if "venue" in schema:
    lf = lf.filter(pl.col("venue") == "okx")
if "market_type" in schema:
    lf = lf.filter(pl.col("market_type") == "SPOT")
if "timeframe" in schema:
    lf = lf.filter(pl.col("timeframe") == "1H")
if "is_closed" in schema:
    lf = lf.filter(pl.col("is_closed"))
last = lf.filter(pl.col("symbol") == "BTC-USDT").select(pl.col("ts").max()).collect().item()
cutoff = last + timedelta(hours=1)
print(max(0.0, (datetime.now(UTC) - cutoff).total_seconds()))
PY
)"
checked_at="$(date --utc --iso-8601=seconds)"

exec "${PYTHON_BIN}" "${REPO_PATH}/audit/scripts/stage_v221_deployment.py" health \
  --root "${AUDIT_ROOT}" \
  --repo "${REPO_PATH}" \
  --installed-service "${SERVICE_UNIT}" \
  --installed-timer "${TIMER_UNIT}" \
  --timer-installed "${timer_installed}" \
  --timer-enabled "${timer_enabled}" \
  --timer-active "${timer_active}" \
  --service-last-result "${service_result}" \
  --next-trigger "${next_trigger}" \
  --disk-free-bytes "${disk_free_bytes}" \
  --market-data-staleness-seconds "${market_staleness}" \
  --checked-at "${checked_at}"
