#!/usr/bin/env bash
set -euo pipefail

QLAB_BIN="${QLAB_BIN:-/opt/quant-lab/.venv/bin/qlab}"
LAKE_ROOT="${QUANT_LAB_LAKE_ROOT:-/var/lib/quant-lab/lake}"
COMPACT_DATASET_TIMEOUT_SECONDS="${COMPACT_DATASET_TIMEOUT_SECONDS:-180}"
COMPACT_RUN_BUDGET_SECONDS="${COMPACT_RUN_BUDGET_SECONDS:-1500}"
COMPACT_RAW_OKX_WS="${COMPACT_RAW_OKX_WS:-0}"
COMPACT_HOT_WS_PARTITION_REPAIR="${COMPACT_HOT_WS_PARTITION_REPAIR:-0}"
COMPACT_DIRECT_MAX_SOURCE_FILES="${COMPACT_DIRECT_MAX_SOURCE_FILES:-8}"
COMPACT_DIRECT_MIN_SOURCE_FILES="${COMPACT_DIRECT_MIN_SOURCE_FILES:-64}"
COMPACT_MAX_SOURCE_BATCH_BYTES="${COMPACT_MAX_SOURCE_BATCH_BYTES:-134217728}"
COMPACT_CONSOLIDATE_EXISTING_COMPACT_OUTPUTS="${COMPACT_CONSOLIDATE_EXISTING_COMPACT_OUTPUTS:-0}"
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
  "silver/v5_order_lifecycle"
  "silver/v5_roundtrip"
  "silver/v5_open_position"
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
    --max-source-files-per-batch "${batch_files}" \
    --max-source-batch-bytes "${COMPACT_MAX_SOURCE_BATCH_BYTES}" \
    --compact-output
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "WARN_COMPACT_FAILED dataset=${dataset} status=${status}"
    return 0
  fi
  echo "FINISH_COMPACT dataset=${dataset}"
}

compact_dataset_direct_only() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"
  local include_existing="${4:-0}"
  local status
  local include_args=()

  if [[ "${include_existing}" == "1" ]]; then
    include_args=(--include-existing-compact-files)
  fi

  echo "START_DIRECT_COMPACT dataset=${dataset} timeout_seconds=${COMPACT_DATASET_TIMEOUT_SECONDS}"
  set +e
  timeout --kill-after=30s "${COMPACT_DATASET_TIMEOUT_SECONDS}s" \
    "${QLAB_BIN}" compact-lake-dataset \
    --lake-root "${LAKE_ROOT}" \
    --dataset "${dataset}" \
    --target-rows-per-file "${target_rows}" \
    --max-source-files-per-batch "${batch_files}" \
    --max-source-batch-bytes "${COMPACT_MAX_SOURCE_BATCH_BYTES}" \
    --direct-only \
    "${include_args[@]}" \
    --compact-output
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "WARN_DIRECT_COMPACT_FAILED dataset=${dataset} status=${status}"
    return 0
  fi
  echo "FINISH_DIRECT_COMPACT dataset=${dataset}"
}

build_market_data_rollups() {
  local status

  echo "START_MARKET_DATA_ROLLUPS timeout_seconds=${COMPACT_DATASET_TIMEOUT_SECONDS}"
  set +e
  timeout --kill-after=30s "${COMPACT_DATASET_TIMEOUT_SECONDS}s" \
    "${QLAB_BIN}" build-market-data-rollups \
    --lake-root "${LAKE_ROOT}" \
    --apply \
    --compact-output
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "WARN_MARKET_DATA_ROLLUPS_FAILED status=${status}"
    return 0
  fi
  echo "FINISH_MARKET_DATA_ROLLUPS"
}

repair_dataset_partitions() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"
  local dataset_path="${LAKE_ROOT}/${dataset}"
  local status

  if [[ ! -d "${dataset_path}" ]]; then
    echo "SKIP_REPAIR_PARTITIONS dataset=${dataset} reason=missing"
    return
  fi
  if ! find "${dataset_path}" -type d \( -name '*=__null__' -o -name '*=__empty__' \) \
      -print -quit | grep -q .; then
    echo "SKIP_REPAIR_PARTITIONS dataset=${dataset} reason=no_bad_partitions"
    return
  fi

  echo "START_REPAIR_PARTITIONS dataset=${dataset} timeout_seconds=${COMPACT_DATASET_TIMEOUT_SECONDS}"
  set +e
  timeout --kill-after=30s "${COMPACT_DATASET_TIMEOUT_SECONDS}s" \
    "${QLAB_BIN}" repair-lake-partitions \
    --lake-root "${LAKE_ROOT}" \
    --dataset "${dataset}" \
    --target-rows-per-file "${target_rows}" \
    --max-source-files-per-batch "${batch_files}" \
    --max-source-batch-bytes "${COMPACT_MAX_SOURCE_BATCH_BYTES}" \
    --compact-output
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "WARN_REPAIR_PARTITIONS_FAILED dataset=${dataset} status=${status}"
    return 0
  fi
  echo "FINISH_REPAIR_PARTITIONS dataset=${dataset}"
}

parquet_file_count() {
  local dataset="$1"
  local dataset_path="${LAKE_ROOT}/${dataset}"
  if [[ ! -d "${dataset_path}" ]]; then
    echo 0
    return
  fi
  visible_parquet_files "${dataset_path}" | wc -l
}

visible_parquet_files() {
  local root_path="$1"
  find "${root_path}" \
    \( -type d \( -name '__*' -o -name '.*' \) -prune \) -o \
    \( -type f -name '*.parquet' ! -name '.*' ! -name '*.tmp.parquet' -print \)
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

direct_source_parquet_file_count() {
  local dataset="$1"
  local dataset_path="${LAKE_ROOT}/${dataset}"
  if [[ ! -d "${dataset_path}" ]]; then
    echo 0
    return
  fi
  find "${dataset_path}" -maxdepth 1 \
    -type f -name '*.parquet' ! -name '.*' ! -name '*.tmp.parquet' \
    ! -name 'compact_*' ! -name 'data.parquet' | wc -l
}

compact_direct_if_file_count_at_least() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"
  local min_files="$4"
  local file_count
  local elapsed

  file_count="$(direct_source_parquet_file_count "${dataset}")"
  if (( file_count < min_files )); then
    echo "SKIP_DIRECT_COMPACT dataset=${dataset} direct_source_files=${file_count} min_files=${min_files}"
    return
  fi
  elapsed="$(( $(date +%s) - COMPACT_STARTED_AT ))"
  if (( elapsed >= COMPACT_RUN_BUDGET_SECONDS )); then
    echo "SKIP_DIRECT_COMPACT_BUDGET dataset=${dataset} elapsed_seconds=${elapsed}"
    return
  fi

  compact_dataset_direct_only "${dataset}" "${target_rows}" "${batch_files}" 0
}

compact_hot_ws_dataset() {
  local dataset="$1"
  local target_rows="$2"
  local partition_batch_files="$3"
  local direct_min_files="$4"
  local partition_min_files="$5"

  if [[ "${COMPACT_HOT_WS_PARTITION_REPAIR}" == "1" ]]; then
    repair_dataset_partitions "${dataset}" "${target_rows}" 100
    compact_leaf_partitions_if_file_count_at_least \
      "${dataset}" "${target_rows}" "${partition_batch_files}" "${partition_min_files}" "${direct_min_files}"
    return
  fi

  echo "SKIP_HOT_WS_PARTITION_REPAIR dataset=${dataset} opt_in=COMPACT_HOT_WS_PARTITION_REPAIR"
  compact_direct_if_file_count_at_least \
    "${dataset}" "${target_rows}" "${COMPACT_DIRECT_MAX_SOURCE_FILES}" "${direct_min_files}"
}

compact_leaf_partitions_if_file_count_at_least() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"
  local min_files="$4"
  local direct_min_files="${5:-${COMPACT_DIRECT_MIN_SOURCE_FILES}}"
  local dataset_path="${LAKE_ROOT}/${dataset}"
  local elapsed
  local effective_min_files

  if [[ ! -d "${dataset_path}" ]]; then
    echo "SKIP_LEAF_COMPACT dataset=${dataset} reason=missing"
    return
  fi

  visible_parquet_files "${dataset_path}" | sed 's#/[^/]*$##' \
    | sort | uniq -c | sort -nr \
    | while read -r file_count leaf_path; do
        effective_min_files="${min_files}"
        if [[ "${leaf_path}" == "${dataset_path}" ]]; then
          effective_min_files="${direct_min_files}"
        fi
        if (( file_count < effective_min_files )); then
          if [[ "${leaf_path}" == "${dataset_path}" ]]; then
            echo "SKIP_DIRECT_COMPACT dataset=${dataset} parquet_files=${file_count} min_files=${effective_min_files}"
          fi
          continue
        fi
        elapsed="$(( $(date +%s) - COMPACT_STARTED_AT ))"
        if (( elapsed >= COMPACT_RUN_BUDGET_SECONDS )); then
          echo "SKIP_LEAF_COMPACT_BUDGET dataset=${dataset} elapsed_seconds=${elapsed}"
          return
        fi
        if [[ "${leaf_path}" == "${dataset_path}" ]]; then
          compact_dataset_direct_only \
            "${leaf_path#${LAKE_ROOT}/}" \
            "${target_rows}" \
            "${COMPACT_DIRECT_MAX_SOURCE_FILES}" \
            "${COMPACT_CONSOLIDATE_EXISTING_COMPACT_OUTPUTS}"
        else
          compact_dataset "${leaf_path#${LAKE_ROOT}/}" "${target_rows}" "${batch_files}"
        fi
      done
}

cleanup_internal_compaction_dirs() {
  find "${LAKE_ROOT}" -type d \( -name '__*_backup_*' -o -name '__*_compact_*' -o -name '__*_repair_*' \) \
    -prune -print -exec rm -rf {} +
  find "${LAKE_ROOT}" -type d -name '__*_write_*' -mmin +60 \
    -prune -print -exec rm -rf {} +
  find "${LAKE_ROOT}" -type d -name '._tmp' -empty -mmin +60 \
    -print -delete
}

if [[ "${COMPACT_RAW_OKX_WS}" == "1" ]]; then
  compact_hot_ws_dataset "bronze/okx_public_ws" 500000 100 64 20
else
  echo "SKIP_COMPACT_RAW_OKX_WS dataset=bronze/okx_public_ws opt_in=COMPACT_RAW_OKX_WS"
fi
compact_hot_ws_dataset "silver/trade_print" 500000 100 20 20

# Order book snapshots are denser than raw websocket and trade-print files.
# Compact only direct append files by default while the long-running collector is active.
compact_hot_ws_dataset "silver/orderbook_snapshot" 500000 100 64 10

for dataset in "${V5_TELEMETRY_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 100 10
done

for dataset in "${OPS_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 100 20
done

build_market_data_rollups

cleanup_internal_compaction_dirs

timeout --kill-after=30s 120s "${QLAB_BIN}" lake-health --lake-root "${LAKE_ROOT}" --compact-output \
  || echo "WARN_LAKE_HEALTH_FAILED_OR_TIMED_OUT"
