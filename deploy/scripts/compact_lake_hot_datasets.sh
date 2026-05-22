#!/usr/bin/env bash
set -euo pipefail

QLAB_BIN="${QLAB_BIN:-/opt/quant-lab/.venv/bin/qlab}"
LAKE_ROOT="${QUANT_LAB_LAKE_ROOT:-/var/lib/quant-lab/lake}"
COMPACT_DATASET_TIMEOUT_SECONDS="${COMPACT_DATASET_TIMEOUT_SECONDS:-180}"
COMPACT_RUN_BUDGET_SECONDS="${COMPACT_RUN_BUDGET_SECONDS:-1500}"
COMPACT_RAW_OKX_WS="${COMPACT_RAW_OKX_WS:-0}"
COMPACT_DIRECT_MAX_SOURCE_FILES="${COMPACT_DIRECT_MAX_SOURCE_FILES:-8}"
COMPACT_MAX_SOURCE_BATCH_BYTES="${COMPACT_MAX_SOURCE_BATCH_BYTES:-134217728}"
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
    --max-source-batch-bytes "${COMPACT_MAX_SOURCE_BATCH_BYTES}"
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
  local status

  echo "START_DIRECT_COMPACT dataset=${dataset} timeout_seconds=${COMPACT_DATASET_TIMEOUT_SECONDS}"
  set +e
  timeout --kill-after=30s "${COMPACT_DATASET_TIMEOUT_SECONDS}s" \
    "${QLAB_BIN}" compact-lake-dataset \
    --lake-root "${LAKE_ROOT}" \
    --dataset "${dataset}" \
    --target-rows-per-file "${target_rows}" \
    --max-source-files-per-batch "${batch_files}" \
    --max-source-batch-bytes "${COMPACT_MAX_SOURCE_BATCH_BYTES}" \
    --direct-only
  status="$?"
  set -e
  if (( status != 0 )); then
    echo "WARN_DIRECT_COMPACT_FAILED dataset=${dataset} status=${status}"
    return 0
  fi
  echo "FINISH_DIRECT_COMPACT dataset=${dataset}"
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
    --max-source-batch-bytes "${COMPACT_MAX_SOURCE_BATCH_BYTES}"
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

compact_leaf_partitions_if_file_count_at_least() {
  local dataset="$1"
  local target_rows="$2"
  local batch_files="$3"
  local min_files="$4"
  local dataset_path="${LAKE_ROOT}/${dataset}"
  local elapsed

  if [[ ! -d "${dataset_path}" ]]; then
    echo "SKIP_LEAF_COMPACT dataset=${dataset} reason=missing"
    return
  fi

  find "${dataset_path}" -type f -name '*.parquet' -printf '%h\n' \
    | sort | uniq -c | sort -nr \
    | while read -r file_count leaf_path; do
        if (( file_count < min_files )); then
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
            "${COMPACT_DIRECT_MAX_SOURCE_FILES}"
        else
          compact_dataset "${leaf_path#${LAKE_ROOT}/}" "${target_rows}" "${batch_files}"
        fi
      done
}

cleanup_internal_compaction_dirs() {
  find "${LAKE_ROOT}" -type d \( -name '__*_backup_*' -o -name '__*_compact_*' -o -name '__*_repair_*' \) \
    -prune -print -exec rm -rf {} +
}

if [[ "${COMPACT_RAW_OKX_WS}" == "1" ]]; then
  # Raw websocket bronze is partitioned by day/channel/inst_id. Compact leaf
  # partitions instead of the dataset root; root compaction can multiply files by
  # writing one output per partition per source batch.
  repair_dataset_partitions "bronze/okx_public_ws" 500000 100
  compact_leaf_partitions_if_file_count_at_least "bronze/okx_public_ws" 500000 100 20
else
  echo "SKIP_COMPACT_RAW_OKX_WS dataset=bronze/okx_public_ws opt_in=COMPACT_RAW_OKX_WS"
fi
repair_dataset_partitions "silver/trade_print" 500000 100
compact_leaf_partitions_if_file_count_at_least "silver/trade_print" 500000 100 20

# Order book snapshots are denser than raw websocket and trade-print files.
# Compact leaf partitions to avoid multiplying partition files across batches.
repair_dataset_partitions "silver/orderbook_snapshot" 500000 100
compact_leaf_partitions_if_file_count_at_least "silver/orderbook_snapshot" 500000 100 10

for dataset in "${V5_TELEMETRY_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 100 10
done

for dataset in "${OPS_DATASETS[@]}"; do
  compact_if_file_count_at_least "${dataset}" 250000 100 20
done

cleanup_internal_compaction_dirs

timeout --kill-after=30s 120s "${QLAB_BIN}" lake-health --lake-root "${LAKE_ROOT}" --compact-output \
  || echo "WARN_LAKE_HEALTH_FAILED_OR_TIMED_OUT"
