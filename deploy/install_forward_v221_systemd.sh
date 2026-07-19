#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="${QUANT_LAB_FORWARD_REPO:-/opt/quant-lab-forward-v221}"
AUDIT_ROOT="${AUDIT_V221_ROOT:-/var/lib/quant-lab/forward_v221}"
PYTHON_BIN="${QUANT_LAB_FORWARD_PYTHON:-/opt/quant-lab/.venv/bin/python}"
MARKET_BAR_PATH="${QUANT_LAB_FORWARD_MARKET_BAR_PATH:-/var/lib/quant-lab/lake/silver/market_bar/data.parquet}"
LOCK_SOURCE=""
SYSTEMCTL_BIN="${QUANT_LAB_FORWARD_SYSTEMCTL:-systemctl}"
TEST_MODE="${QUANT_LAB_FORWARD_INSTALL_TEST_MODE:-0}"
DESTDIR="${QUANT_LAB_FORWARD_DESTDIR:-}"

# The isolated checkout may be owned by the non-root service user. Scope Git's
# safe-directory exception to this process tree instead of mutating global config.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=safe.directory
export GIT_CONFIG_VALUE_0="${REPO_PATH}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) REPO_PATH="$2"; shift 2 ;;
    --root) AUDIT_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --market-bar) MARKET_BAR_PATH="$2"; shift 2 ;;
    --parameter-lock) LOCK_SOURCE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 64 ;;
  esac
done

export GIT_CONFIG_VALUE_0="${REPO_PATH}"

if [[ "${TEST_MODE}" != "1" && "${EUID}" -ne 0 ]]; then
  echo "install_forward_v221_systemd.sh must run as root" >&2
  exit 77
fi
[[ -n "${LOCK_SOURCE}" && -f "${LOCK_SOURCE}" ]] || { echo "--parameter-lock is required" >&2; exit 66; }
[[ -d "${REPO_PATH}/.git" || -f "${REPO_PATH}/.git" ]] || { echo "quant-lab Git checkout is missing" >&2; exit 66; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Python runtime is missing" >&2; exit 66; }
[[ -f "${MARKET_BAR_PATH}" ]] || { echo "market_bar is missing" >&2; exit 66; }

head_sha="$(git -C "${REPO_PATH}" rev-parse HEAD)"
[[ -z "$(git -C "${REPO_PATH}" status --porcelain)" ]] || { echo "working tree is dirty" >&2; exit 65; }
locked_sha="$(${PYTHON_BIN} - "${LOCK_SOURCE}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["strategy_code_commit"])
PY
)"
[[ "${head_sha}" == "${locked_sha}" ]] || { echo "Git HEAD does not match parameter lock" >&2; exit 65; }

PYTHONPATH="${REPO_PATH}" "${PYTHON_BIN}" - "${REPO_PATH}" "${LOCK_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

from audit.auditlib.forward_v221 import runtime_identity

repo = Path(sys.argv[1]).resolve()
lock = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
identity = runtime_identity(
    repo=repo,
    lock=lock,
    installed_service=repo / "deploy/systemd/quant-lab-forward-v221.service",
    installed_timer=repo / "deploy/systemd/quant-lab-forward-v221.timer",
)
if not identity["ok"]:
    raise SystemExit("forward identity preflight failed: " + ",".join(identity["errors"]))
PY

if [[ "${TEST_MODE}" != "1" ]]; then
  getent passwd quantlab >/dev/null
  getent group quant-research >/dev/null
fi

ETC_SYSTEMD="${DESTDIR}/etc/systemd/system"
ETC_QUANT="${DESTDIR}/etc/quant-lab"
LOG_ROOT="${DESTDIR}/var/log/quant-lab-forward-v221"
STATE_ROOT="${DESTDIR}${AUDIT_ROOT}"
mkdir -p "${ETC_SYSTEMD}" "${ETC_QUANT}" "${LOG_ROOT}" "${STATE_ROOT}/artifacts" "${STATE_ROOT}/manifests" "${STATE_ROOT}/state"
install -m 0644 "${REPO_PATH}/deploy/systemd/quant-lab-forward-v221.service" "${ETC_SYSTEMD}/quant-lab-forward-v221.service"
install -m 0644 "${REPO_PATH}/deploy/systemd/quant-lab-forward-v221.timer" "${ETC_SYSTEMD}/quant-lab-forward-v221.timer"
installed_lock="${STATE_ROOT}/manifests/parameter_lock_v221.json"
if [[ -f "${installed_lock}" ]] && ! cmp -s "${LOCK_SOURCE}" "${installed_lock}"; then
  echo "installed parameter lock differs; preserve existing Forward evidence" >&2
  exit 65
fi
install -m 0440 "${LOCK_SOURCE}" "${installed_lock}"
cat >"${ETC_QUANT}/forward-v221.env.partial" <<EOF
AUDIT_V221_ROOT=${AUDIT_ROOT}
QUANT_LAB_FORWARD_REPO=${REPO_PATH}
QUANT_LAB_FORWARD_PYTHON=${PYTHON_BIN}
QUANT_LAB_FORWARD_MARKET_BAR_PATH=${MARKET_BAR_PATH}
EOF
chmod 0640 "${ETC_QUANT}/forward-v221.env.partial"
mv "${ETC_QUANT}/forward-v221.env.partial" "${ETC_QUANT}/forward-v221.env"

if [[ "${TEST_MODE}" == "1" ]]; then
  echo "FORWARD_V221_TEST_INSTALL_OK"
  exit 0
fi

chown -R quantlab:quant-research "${AUDIT_ROOT}" /var/log/quant-lab-forward-v221
chown root:quant-research /etc/quant-lab/forward-v221.env
chown root:quant-research "${AUDIT_ROOT}/manifests/parameter_lock_v221.json"
chmod 0750 "${AUDIT_ROOT}" /var/log/quant-lab-forward-v221
chmod 0640 /etc/quant-lab/forward-v221.env
chmod 0440 "${AUDIT_ROOT}/manifests/parameter_lock_v221.json"
${SYSTEMCTL_BIN} daemon-reload
${SYSTEMCTL_BIN} enable --now quant-lab-forward-v221.timer
[[ "$(${SYSTEMCTL_BIN} is-enabled quant-lab-forward-v221.timer)" == "enabled" ]]
[[ "$(${SYSTEMCTL_BIN} is-active quant-lab-forward-v221.timer)" == "active" ]]
timer_enabled_at="$(date --utc --iso-8601=seconds)"
if [[ -f "${AUDIT_ROOT}/manifests/systemd_deployment_v221.json" ]]; then
  timer_enabled_at="$(${PYTHON_BIN} - "${AUDIT_ROOT}/manifests/systemd_deployment_v221.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["timer_enabled_at"])
PY
)"
fi
next_trigger="$(${SYSTEMCTL_BIN} list-timers quant-lab-forward-v221.timer --no-legend --no-pager | awk '{$1=$1; print $1" "$2" "$3" "$4}')"
[[ -n "${next_trigger}" ]] || { echo "next timer trigger is unavailable" >&2; exit 69; }

runuser -u quantlab -- env \
  AUDIT_V221_ROOT="${AUDIT_ROOT}" \
  QUANT_LAB_FORWARD_REPO="${REPO_PATH}" \
  QUANT_LAB_FORWARD_PYTHON="${PYTHON_BIN}" \
  QUANT_LAB_FORWARD_MARKET_BAR_PATH="${MARKET_BAR_PATH}" \
  "${REPO_PATH}/scripts/run_forward_v221_realtime.sh" --dry-run --allow-no-cutoff

"${PYTHON_BIN}" "${REPO_PATH}/audit/scripts/stage_v221_deployment.py" manifest \
  --root "${AUDIT_ROOT}" \
  --repo "${REPO_PATH}" \
  --node "$(hostname)" \
  --node-always-on true \
  --timer-installed true \
  --timer-enabled true \
  --timer-active true \
  --timer-enabled-at "${timer_enabled_at}" \
  --next-trigger "${next_trigger}" \
  --installed-at "${timer_enabled_at}"

"${REPO_PATH}/deploy/check_forward_v221_health.sh"
cutoff_created_at="$(date --utc --iso-8601=seconds)"
"${PYTHON_BIN}" "${REPO_PATH}/audit/scripts/stage_v221_deployment.py" cutoff \
  --root "${AUDIT_ROOT}" \
  --repo "${REPO_PATH}" \
  --created-at "${cutoff_created_at}"
chmod 0440 "${AUDIT_ROOT}/manifests/forward_v221_cutoff.json" "${AUDIT_ROOT}/manifests/systemd_deployment_v221.json"
chown root:quant-research \
  "${AUDIT_ROOT}/manifests/parameter_lock_v221.json" \
  "${AUDIT_ROOT}/manifests/forward_v221_cutoff.json" \
  "${AUDIT_ROOT}/manifests/systemd_deployment_v221.json"
"${REPO_PATH}/deploy/check_forward_v221_health.sh"

echo "FORWARD_V221_READY"
echo "next_timer_trigger=${next_trigger}"
${SYSTEMCTL_BIN} list-timers quant-lab-forward-v221.timer --no-pager
