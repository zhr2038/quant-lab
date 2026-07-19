#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${QLAB_AUDIT_PYTHON:-/home/hr/quant-alpha-audit/.venv/bin/python}"
audit_root="${AUDIT_V221_ROOT:-/home/hr/quant-alpha-audit-v2.2.1}"
v22_bundle="${AUDIT_V22_BUNDLE:-/mnt/c/Users/HR/Downloads/alpha_audit_v22_bundle_20260719_053423.zip}"
stage="all"

if [[ "${1:-}" == "--stage" ]]; then
  stage="${2:?missing stage name}"
  shift 2
fi

run_stage() {
  local wanted="$1"
  shift
  if [[ "${stage}" == "all" || "${stage}" == "${wanted}" ]]; then
    "$@"
  fi
}

run_stage consistency "${python_bin}" "${repo_root}/audit/scripts/stage_v221_consistency.py" \
  --bundle "${v22_bundle}" --root "${audit_root}" --repo "${repo_root}"
run_stage init "${python_bin}" "${repo_root}/audit/scripts/stage_v221_init.py" \
  --root "${audit_root}" --repo "${repo_root}"
run_stage forward "${repo_root}/scripts/run_forward_v221_realtime.sh" \
  --root "${audit_root}" --resume "$@"
run_stage test "${python_bin}" "${repo_root}/audit/scripts/stage_v221_test.py" \
  --root "${audit_root}"
run_stage report "${python_bin}" "${repo_root}/audit/scripts/stage_v221_report.py" \
  --root "${audit_root}"
run_stage bundle "${python_bin}" "${repo_root}/audit/scripts/stage_v221_bundle.py" \
  --root "${audit_root}" --repo "${repo_root}"
