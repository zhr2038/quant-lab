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

OPS_DATASETS=(
  "bronze/api_request_metrics"
  "gold/job_run_history"
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

parquet_file_count() {
  local dataset="$1"
  local dataset_path="${LAKE_ROOT}/${dataset}"
  if [[ ! -d "${dataset_path}" ]]; then
    echo 0
    return
  fi
  find "${dataset_path}" -type f -name '*.parquet' | wc -l
}

compact_if_file_count_at_least() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"
  local min_files="$4"
  local file_count

  file_count="$(parquet_file_count "${dataset}")"
  if (( file_count < min_files )); then
    echo "SKIP_COMPACT dataset=${dataset} parquet_files=${file_count} min_files=${min_files}"
    return
  fi

  compact_dataset "${dataset}" "${target_rows}" "${batch_files}"
}

for dataset in "${HOT_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 500000 50 500
done

for dataset in "${V5_TELEMETRY_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 5000 100
done

for dataset in "${OPS_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 5000 100
done

"${QLAB_BIN}" lake-health --lake-root "${LAKE_ROOT}"
