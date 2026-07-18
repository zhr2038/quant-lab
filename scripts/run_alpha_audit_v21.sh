#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_V21_ROOT="${AUDIT_V21_ROOT:-$HOME/quant-alpha-audit-v2.1}"
AUDIT_V2_DIR="${AUDIT_V2_DIR:-$AUDIT_V21_ROOT/v2}"
AUDIT_V1_DIR="${AUDIT_V1_DIR:-$HOME/quant-alpha-audit-v2/v1}"
AUDIT_V1_ROOT="${AUDIT_V1_ROOT:-$HOME/quant-alpha-audit}"
AUDIT_V2_BUNDLE="${AUDIT_V2_BUNDLE:-/mnt/c/Users/HR/Downloads/alpha_audit_v2_bundle_20260718_212042.zip}"
PYTHON_BIN="${AUDIT_PYTHON:-$AUDIT_V1_ROOT/.venv/bin/python}"
STAGE="all"
AS_OF=""
RESUME=0

usage() {
  cat <<'EOF'
Usage: ./scripts/run_alpha_audit_v21.sh [--stage NAME] [--as-of UTC] [--resume]

Stages: init, forward, analysis, test, report, bundle, all.
The command is research-only: no deployment, service restart, exchange private
API, production state write, order submission, Alpha enablement or branch merge.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="${2:?--stage requires a value}"; shift 2 ;;
    --as-of) AS_OF="${2:?--as-of requires a value}"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

export AUDIT_V21_ROOT AUDIT_V2_DIR AUDIT_V1_DIR AUDIT_V1_ROOT AUDIT_V2_BUNDLE
cd "$REPO_ROOT"

run_init() {
  "$PYTHON_BIN" audit/scripts/stage_v21_init.py \
    --root "$AUDIT_V21_ROOT" --v2-dir "$AUDIT_V2_DIR" --repo "$REPO_ROOT"
}

run_forward() {
  local args=(--root "$AUDIT_V21_ROOT" --v1-root "$AUDIT_V1_ROOT")
  [[ -n "$AS_OF" ]] && args+=(--as-of "$AS_OF")
  [[ "$RESUME" -eq 1 ]] && args+=(--resume)
  "$PYTHON_BIN" audit/scripts/stage_v21_forward.py "${args[@]}"
}

run_analysis() {
  "$PYTHON_BIN" audit/scripts/stage_v21_analysis.py \
    --root "$AUDIT_V21_ROOT" --v2-dir "$AUDIT_V2_DIR" \
    --v1-dir "$AUDIT_V1_DIR" --v1-root "$AUDIT_V1_ROOT"
}

run_test() {
  "$PYTHON_BIN" audit/scripts/stage_v21_test.py \
    --root "$AUDIT_V21_ROOT" --repo "$REPO_ROOT" --python "$PYTHON_BIN"
}

run_report() {
  "$PYTHON_BIN" audit/scripts/stage_v21_report.py \
    --root "$AUDIT_V21_ROOT" --repo "$REPO_ROOT"
}

run_bundle() {
  "$PYTHON_BIN" audit/scripts/stage_v21_bundle.py \
    --root "$AUDIT_V21_ROOT" --repo "$REPO_ROOT"
}

case "$STAGE" in
  init) run_init ;;
  forward) run_forward ;;
  analysis) run_analysis ;;
  test) run_test ;;
  report) run_report ;;
  bundle) run_bundle ;;
  all)
    run_init
    RESUME=1 run_forward
    run_analysis
    run_test
    run_report
    ;;
  *) echo "Unknown stage: $STAGE" >&2; usage >&2; exit 2 ;;
esac
