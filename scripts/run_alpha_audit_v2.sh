#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT_ROOT="${AUDIT_ROOT:-$HOME/quant-alpha-audit-v2}"
AUDIT_V1_ROOT="${AUDIT_V1_ROOT:-$HOME/quant-alpha-audit}"
AUDIT_V1_DIR="${AUDIT_V1_DIR:-$AUDIT_ROOT/v1}"
AUDIT_V1_BUNDLE="${AUDIT_V1_BUNDLE:-/mnt/c/Users/HR/Downloads/alpha_audit_bundle_20260718_162903.zip}"
QLAB_REPO_PATH="${QLAB_REPO_PATH:-$REPO_ROOT}"
PYTHON_BIN="${AUDIT_PYTHON:-$AUDIT_V1_ROOT/.venv/bin/python}"
STAGE="all"
RESUME=0
AS_OF=""

usage() {
  cat <<'EOF'
Usage: ./scripts/run_alpha_audit_v2.sh [--stage NAME] [--resume] [--as-of TIMESTAMP]

Stages:
  consistency  Validate the immutable v1 bundle and conclusions
  analysis     Run the locked 24h, layered low-vol, funding and statistics work
  cost         Rebuild cost-coverage evidence from the read-only local snapshot
  v5-snapshot  Capture the production V5 runtime over read-only SSH (requires SSHPASS)
  forward      Update the single frozen low-vol forward paper hypothesis
  report       Regenerate decisions, Markdown reports and the static dashboard
  test         Run Audit v2 plus the full quant-lab test suite and lint
  bundle       Validate and package final evidence (run after commits/browser QA)
  all          Run consistency, analysis, cost, v5-snapshot, forward, report, test

The runner never deploys, restarts V5, writes production state, or submits orders.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="${2:?--stage requires a value}"
      shift 2
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --as-of)
      AS_OF="${2:?--as-of requires a timestamp}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export AUDIT_ROOT AUDIT_V1_ROOT AUDIT_V1_DIR AUDIT_V1_BUNDLE QLAB_REPO_PATH
cd "$REPO_ROOT"

run_consistency() {
  "$PYTHON_BIN" audit/scripts/stage_v2_consistency.py
}

run_analysis() {
  "$PYTHON_BIN" audit/scripts/stage_v2_analysis.py
}

run_cost() {
  "$PYTHON_BIN" audit/scripts/stage_v2_cost.py
}

run_v5_snapshot() {
  if [[ -z "${SSHPASS:-}" ]]; then
    echo "SSHPASS is required for the read-only V5 runtime snapshot" >&2
    exit 2
  fi
  "$PYTHON_BIN" audit/scripts/stage_v2_v5_snapshot.py
}

run_forward() {
  local args=(--start-after "2026-07-17T23:00:00+00:00")
  if [[ -n "$AS_OF" ]]; then
    args+=(--as-of "$AS_OF")
  fi
  if [[ "$RESUME" -eq 1 ]]; then
    args+=(--resume)
  fi
  "$PYTHON_BIN" audit/scripts/stage_v2_forward.py "${args[@]}"
}

run_report() {
  "$PYTHON_BIN" audit/scripts/stage_v2_report.py
}

run_test() {
  "$PYTHON_BIN" -m pytest -q audit/tests \
    --junitxml="$AUDIT_ROOT/artifacts/audit_v2_tests.xml"
  "$PYTHON_BIN" -m pytest -q \
    --junitxml="$AUDIT_ROOT/artifacts/pytest_all_v2.xml"
  "$PYTHON_BIN" -m ruff check .
  "$PYTHON_BIN" -m compileall -q audit src
  git diff --check
}

run_bundle() {
  "$PYTHON_BIN" audit/scripts/stage_v2_bundle.py
}

case "$STAGE" in
  consistency) run_consistency ;;
  analysis) run_analysis ;;
  cost) run_cost ;;
  v5-snapshot) run_v5_snapshot ;;
  forward) run_forward ;;
  report) run_report ;;
  test) run_test ;;
  bundle) run_bundle ;;
  all)
    run_consistency
    run_analysis
    run_cost
    run_v5_snapshot
    run_forward
    run_report
    run_test
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    usage >&2
    exit 2
    ;;
esac
