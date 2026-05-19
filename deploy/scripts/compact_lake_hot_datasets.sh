#!/usr/bin/env bash
set -euo pipefail

QLAB_BIN="${QLAB_BIN:-/opt/quant-lab/.venv/bin/qlab}"
LAKE_ROOT="${QUANT_LAB_LAKE_ROOT:-/var/lib/quant-lab/lake}"
COMPACT_DATASET_TIMEOUT_SECONDS="${COMPACT_DATASET_TIMEOUT_SECONDS:-180}"
COMPACT_RUN_BUDGET_SECONDS="${COMPACT_RUN_BUDGET_SECONDS:-1500}"
COMPACT_STARTED_AT="$(date +%s)"

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
  local status

  echo "START_COMPACT dataset=${dataset} timeout_seconds=${COMPACT_DATASET_TIMEOUT_SECONDS}"
  set +e
  timeout --kill-after=30s "${COMPACT_DATASET_TIMEOUT_SECONDS}s" \
    "${QLAB_BIN}" compact-lake-dataset \
    --lake-root "${LAKE_ROOT}" \
    --dataset "${dataset}" \
    --target-rows-per-file "${target_rows}" \
    --max-source-files-per-batch "${batch_files}"
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "WARN_COMPACT_FAILED dataset=${dataset} status=${status}"
    return 0
  fi
  echo "FINISH_COMPACT dataset=${dataset}"
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
  local elapsed

  file_count="$(parquet_file_count "${dataset}")"
  if (( file_count < min_files )); then
    echo "SKIP_COMPACT dataset=${dataset} parquet_files=${file_count} min_files=${min_files}"
    return
  fi
  elapsed="$(( $(date +%s) - COMPACT_STARTED_AT ))"
  if (( elapsed >= COMPACT_RUN_BUDGET_SECONDS )); then
    echo "SKIP_COMPACT_BUDGET dataset=${dataset} elapsed_seconds=${elapsed}"
    return
  fi

  compact_dataset "${dataset}" "${target_rows}" "${batch_files}"
}

cleanup_internal_compaction_dirs() {
  find "${LAKE_ROOT}" -type d \( -name '__*_backup_*' -o -name '__*_compact_*' \) \
    -prune -print -exec rm -rf {} +
}

compact_if_file_count_at_least "bronze/okx_public_ws" 500000 50 120
compact_if_file_count_at_least "silver/trade_print" 500000 50 40

# Order book snapshots are denser than raw websocket and trade-print files.
# Keep the source batch smaller so scheduled compaction does not exceed qyun2 memory.
compact_if_file_count_at_least "silver/orderbook_snapshot" 250000 10 80

for dataset in "${V5_TELEMETRY_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 100 10
done

for dataset in "${OPS_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 100 20
done

cleanup_internal_compaction_dirs

timeout --kill-after=30s 120s "${QLAB_BIN}" lake-health --lake-root "${LAKE_ROOT}" \
  || echo "WARN_LAKE_HEALTH_FAILED_OR_TIMED_OUT"
