#!/usr/bin/env bash
set -euo pipefail

QLAB_BIN="${QLAB_BIN:-/opt/quant-lab/.venv/bin/qlab}"
LAKE_ROOT="${QUANT_LAB_LAKE_ROOT:-/var/lib/quant-lab/lake}"

HOT_DATASETS=(
  "bronze/okx_public_ws"
  "silver/trade_print"
  "silver/orderbook_snapshot"
)

V5_TELEMETRY_DATASETS=(
  "bronze/strategy_telemetry/v5/raw_file_index"
  "silver/v5_quant_lab_usage"
  "silver/v5_quant_lab_request"
  "silver/v5_quant_lab_compliance"
  "silver/v5_quant_lab_cost_usage"
  "silver/v5_quant_lab_fallback"
  "silver/v5_decision_audit"
  "silver/v5_run_summary"
  "silver/v5_state_snapshot"
  "silver/v5_config_audit"
  "silver/v5_issue"
  "silver/v5_probe_diagnostic"
  "silver/v5_paper_strategy_run"
  "silver/v5_paper_strategy_daily"
  "silver/v5_paper_slippage_coverage"
  "silver/v5_equity_point"
  "silver/v5_router_decision"
  "silver/v5_high_score_blocked_target"
  "silver/v5_high_score_blocked_outcome"
  "silver/v5_skipped_candidate_outcome"
  "silver/v5_candidate_event"
  "silver/v5_candidate_label"
  "silver/v5_shadow_outcome"
  "silver/v5_trade_event"
)

compact_dataset() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"

  "${QLAB_BIN}" compact-lake-dataset \
    --lake-root "${LAKE_ROOT}" \
    --dataset "${dataset}" \
    --target-rows-per-file "${target_rows}" \
    --max-source-files-per-batch "${batch_files}"
}

for dataset in "${HOT_DATASETS[@]}"; do
  compact_dataset "${dataset}" 500000 10000
done

for dataset in "${V5_TELEMETRY_DATASETS[@]}"; do
  compact_dataset "${dataset}" 250000 5000
done

"${QLAB_BIN}" lake-health --lake-root "${LAKE_ROOT}"
