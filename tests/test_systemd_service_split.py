from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"


SCRIPTS = ROOT / "deploy" / "scripts"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


def _script(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_v5_health_analysis_stays_lightweight():
    unit = _unit("quant-lab-v5-daily-analysis.service")
    sync_unit = _unit("quant-lab-v5-telemetry-sync.service")
    sync_timer = _unit("quant-lab-v5-telemetry-sync.timer")

    assert "analyze-v5-telemetry" in unit
    assert "--skip-candidate-gold" in unit
    assert "--compact-output" in unit
    assert "SKIP_V5_DAILY_ANALYSIS_LOCK_BUSY" in unit
    assert "flock -E 75 -w 30 /var/lock/quant-lab-heavy.lock" in unit
    assert "flock -E 75 -w 5" in unit
    assert "/var/lock/quant-lab-v5-daily-analysis.lock" in unit
    assert "TimeoutStartSec=10min" in unit
    assert "MemoryHigh=2560M" in unit
    assert "MemoryMax=3G" in unit
    assert "/var/lock/quant-lab-v5-research.lock" not in unit
    assert "/usr/bin/timeout 8m" in unit
    assert "build-v5-candidate-labels" not in unit
    assert "build-strategy-evidence" not in unit
    assert "build-alpha-discovery-board" not in unit
    assert "--remote-max-files 1" in sync_unit
    assert "--max-scan-bundles 1" in sync_unit
    assert "--skip-analysis-after-sync" in sync_unit
    assert "--run-analysis-after-sync" not in sync_unit
    assert "--compact-output" in sync_unit
    assert "flock -E 75 -w 3000 /var/lock/quant-lab-heavy.lock" in sync_unit
    assert "flock -E 75 -w 60 /var/lock/quant-lab-v5-telemetry-sync.lock" in sync_unit
    assert "/usr/bin/timeout 15m" in sync_unit
    assert "TimeoutStartSec=70min" in sync_unit
    assert "MemoryHigh=2G" in sync_unit
    assert "MemoryMax=3G" in sync_unit
    assert "QUANT_LAB_V5_SYNC_REMOTE_MAX_FILES=1" in sync_unit
    assert "QUANT_LAB_V5_SYNC_MAX_SCAN_BUNDLES=1" in sync_unit
    assert "OnUnitInactiveSec=10min" in sync_timer
    assert "OnUnitActiveSec=10min" not in sync_timer


def test_api_service_uses_async_metrics_flush():
    unit = _unit("quant-lab-api.service")

    assert "QUANT_LAB_API_METRICS_ASYNC_FLUSH=1" in unit
    assert "QUANT_LAB_API_METRICS_FLUSH_ROWS=1000" in unit
    assert "QUANT_LAB_API_METRICS_FLUSH_SECONDS=300" in unit
    assert "QUANT_LAB_API_METRICS_FLUSH_JOIN_SECONDS=0.25" in unit
    assert "QUANT_LAB_API_METRICS_RESPONSE_CACHE_SECONDS=12" in unit
    assert "QUANT_LAB_API_METRICS_PRODUCTION_CLIENT_HOSTS=43.156.105.125" in unit


def test_okx_rest_backfill_runs_every_15_minutes_to_reduce_stale_market_bar_window():
    timer = _unit("quant-lab-okx-rest-backfill.timer")
    service = _unit("quant-lab-okx-rest-backfill.service")

    assert "every 15 minutes" in timer
    assert "OnActiveSec=15min" in timer
    assert "OnUnitActiveSec=15min" in timer
    assert "OnUnitActiveSec=1h" not in timer
    assert "okx-fetch-candles" in service
    assert "--history" not in service


def test_all_quant_lab_jobs_run_as_service_user_except_root_only_helpers():
    for unit_path in SYSTEMD.glob("*.service"):
        unit = unit_path.read_text(encoding="utf-8")
        if "ExecStart=" not in unit:
            continue
        assert "User=quantlab" in unit, unit_path.name
        assert "Group=quantlab" in unit, unit_path.name


def test_oneshot_services_do_not_use_ignored_runtime_max_sec():
    for unit_path in SYSTEMD.glob("*.service"):
        unit = unit_path.read_text(encoding="utf-8")
        if "Type=oneshot" not in unit:
            continue
        assert "RuntimeMaxSec=" not in unit, unit_path.name


def test_cost_calibration_starts_readonly_private_backfill_when_configured():
    unit = _unit("quant-lab-cost-calibration.service")

    assert "PermissionsStartOnly=true" in unit
    assert "ExecStartPre=/bin/bash -lc" in unit
    assert "[ -f /etc/quant-lab/okx_readonly.env ]" in unit
    assert "/usr/bin/systemctl start quant-lab-okx-readonly-backfill.service" in unit
    assert "SKIP_OKX_READONLY_BACKFILL_ENV_MISSING" in unit
    assert "flock -E 75 -w 600" in unit
    assert "TimeoutStartSec=30min" in unit


def test_nas_redacted_archive_is_pull_only_and_checksum_verified():
    script = (
        ROOT / "deploy" / "nas_archive" / "pull_qyun2_redacted_v5.sh"
    ).read_text(encoding="utf-8")

    assert "archive/v5/bundles" in script
    assert "archive_restricted" not in script
    assert "--remove-source-files" not in script
    assert "cmp --silent" in script
    assert ".archive_manifest.sha256" in script
    assert "ARCHIVE_COMPLETE_HISTORY_RETAINED" in script
    assert "retention_removed" not in script
    assert "/volume2/quant-lab/archive/current/qyun2" in script


def test_nas_high_frequency_archive_requires_verified_copy_before_source_prune():
    pull_script = (
        ROOT / "deploy" / "nas_archive" / "pull_qyun2_high_frequency.sh"
    ).read_text(encoding="utf-8")
    prune_script = (
        ROOT
        / "deploy"
        / "nas_archive"
        / "prune_verified_high_frequency_archive.py"
    ).read_text(encoding="utf-8")
    sudoers = (
        ROOT
        / "deploy"
        / "nas_archive"
        / "quant-research-high-frequency-prune.sudoers"
    ).read_text(encoding="utf-8")

    assert "lake/archive/high_frequency" in pull_script
    assert "cmp --silent" in pull_script
    assert ".archive_manifest.sha256" in pull_script
    assert pull_script.count("IPQoS=none") == 2
    assert "QUANT_ARCHIVE_TRANSFER_STREAMS:-2" in pull_script
    assert "QUANT_ARCHIVE_TRANSFER_ATTEMPT_TIMEOUT_SECONDS:-180" in pull_script
    assert "QUANT_ARCHIVE_TRANSFER_MAX_ATTEMPTS:-20" in pull_script
    assert '--files-from="$file_list"' in pull_script
    assert "--partial-dir=.rsync-partial" in pull_script
    assert 'rm -rf -- "$stage/.rsync-partial"' in pull_script
    assert 'rm -rf -- "$stage"' not in pull_script
    assert 'wait "$transfer_pid"' in pull_script
    assert "retry high-frequency archive transfer" in pull_script
    assert "high-frequency archive transfer attempts exhausted" in pull_script
    assert "-exec stat -c '%s' {} +" in pull_script
    assert 'du -sb "$stage"' not in pull_script
    assert "prune_verified_high_frequency_archive.py" in pull_script
    assert "--apply" in pull_script
    assert '"reason":"heavy_lock_timeout"' in pull_script
    assert "source_prune_deferred_heavy_lock" in pull_script
    assert 'exit "$prune_exit_code"' in pull_script
    assert "rm -rf -- \"$source\"" not in pull_script
    assert "ALLOWED_DATASETS" in prune_script
    assert "manifest_sha256_mismatch" in prune_script
    assert "date_not_before_today" in prune_script
    assert "shutil.rmtree" in prune_script
    assert "(quantlab) NOPASSWD" in sudoers
    assert "prune_verified_high_frequency_archive.py" in sudoers


def test_lake_permission_repair_script_targets_service_user():
    script = _script("repair_lake_permissions.sh")

    assert 'LAKE_ROOT="${LAKE_ROOT:-/var/lib/quant-lab/lake}"' in script
    assert "QUANT_LAB_BASE_DIR=" in script
    assert "EXPORTS_DIR=" in script
    assert 'QUANT_LAB_USER="${QUANT_LAB_USER:-quantlab}"' in script
    assert 'QUANT_LAB_GROUP="${QUANT_LAB_GROUP:-quantlab}"' in script
    assert "install -d" in script
    assert "chown -R" in script
    assert "chmod u+rwX,g+rwX,o+rX,g+s" in script
    assert "chmod u+rw,g+rw,o+r" in script


def test_deploy_permission_repair_script_targets_deploy_user():
    script = _script("repair_deploy_permissions.sh")

    assert 'APP_ROOT="${APP_ROOT:-/opt/quant-lab}"' in script
    assert 'DEPLOY_USER="${DEPLOY_USER:-ubuntu}"' in script
    assert 'SERVICE_GROUP="${SERVICE_GROUP:-quantlab}"' in script
    assert "START_REPAIR_DEPLOY_PERMISSIONS" in script
    assert 'chown -R "${DEPLOY_USER}:${SERVICE_GROUP}" "${APP_ROOT}"' in script
    assert "chmod u=rwx,g=rx,o=,g+s" in script
    assert "chmod u=rwX,g=rX,o=" in script
    assert "g+rw" not in script
    assert "FINISH_REPAIR_DEPLOY_PERMISSIONS" in script


def test_scheduled_compaction_covers_hot_ws_datasets():
    unit = _unit("quant-lab-lake-compaction.service")
    timer = _unit("quant-lab-lake-compaction.timer")
    script = _script("compact_lake_hot_datasets.sh")

    assert "compact_lake_hot_datasets.sh" in unit
    assert "compact-lake-dataset" in script
    assert "build-market-data-rollups" in script
    assert "START_MARKET_DATA_ROLLUPS" in script
    assert "WARN_MARKET_DATA_ROLLUPS_FAILED" in script
    assert "MARKET_ROLLUP_LOOKBACK_HOURS" in script
    assert "MARKET_ROLLUP_TIMEOUT_SECONDS" in script
    assert "MARKET_ROLLUP_POLARS_MAX_THREADS" in script
    assert "MARKET_ROLLUP_ARCHIVE_OLD_OKX_WS" in script
    assert "MARKET_ROLLUP_ARCHIVE_HOT_HOURS" in script
    assert "--archive-old-okx-public-ws" in script
    assert "--archive-hot-hours" in script
    assert script.index('compact_hot_ws_dataset "silver/orderbook_snapshot"') < script.rindex(
        "\nbuild_market_data_rollups"
    )
    assert "--lookback-hours" in script
    assert "repair-lake-partitions" in script
    assert "START_REPAIR_PARTITIONS" in script
    assert "WARN_REPAIR_PARTITIONS_FAILED" in script
    assert "COMPACT_DATASET_TIMEOUT_SECONDS" in script
    assert "COMPACT_RUN_BUDGET_SECONDS" in script
    assert "COMPACT_DIRECT_MAX_SOURCE_FILES" in script
    assert "COMPACT_DIRECT_MIN_SOURCE_FILES" in script
    assert "COMPACT_MAX_SOURCE_BATCH_BYTES" in script
    assert "COMPACT_SMALL_FILE_MAX_BYTES" in script
    assert "COMPACT_SMALL_FILE_MAINTENANCE" in script
    assert "START_SMALL_FILE_MAINTENANCE" in script
    assert "WARN_SMALL_FILE_MAINTENANCE_FAILED" in script
    assert "lake-small-file-maintenance" in script
    assert "--max-source-files-per-group" in script
    assert "--priority-dataset" in script
    assert "COMPACT_SMALL_FILE_MAINTENANCE_TARGET_ROWS" in script
    assert "COMPACT_SMALL_FILE_MAINTENANCE_DATASETS" in script
    assert (
        'COMPACT_CONSOLIDATE_EXISTING_COMPACT_OUTPUTS="'
        '${COMPACT_CONSOLIDATE_EXISTING_COMPACT_OUTPUTS:-0}"'
    ) in script
    assert "WARN_COMPACT_FAILED" in script
    assert "SKIP_COMPACT_BUDGET" in script
    assert "WARN_LAKE_HEALTH_FAILED_OR_TIMED_OUT" in script
    assert "lake-health --lake-root" in script
    assert "--compact-output" in script
    assert script.count("--compact-output") >= 4
    assert "COMPACT_RAW_OKX_WS" in script
    assert "COMPACT_HOT_WS_PARTITION_REPAIR" in script
    assert "SKIP_COMPACT_RAW_OKX_WS" in script
    assert "COMPACT_RAW_OKX_WS=1" in unit
    assert "COMPACT_HOT_WS_PARTITION_REPAIR=1" in unit
    assert "COMPACT_DATASET_TIMEOUT_SECONDS=300" in unit
    assert "COMPACT_RUN_BUDGET_SECONDS=1800" in unit
    assert "COMPACT_DIRECT_MAX_SOURCE_FILES=64" in unit
    assert "COMPACT_DIRECT_MIN_SOURCE_FILES=16" in unit
    assert "COMPACT_MAX_SOURCE_BATCH_BYTES=134217728" in unit
    assert "COMPACT_CONSOLIDATE_EXISTING_COMPACT_OUTPUTS=0" in unit
    assert "MARKET_ROLLUP_LOOKBACK_HOURS=24" in unit
    assert "MARKET_ROLLUP_TIMEOUT_SECONDS=600" in unit
    assert "MARKET_ROLLUP_POLARS_MAX_THREADS=2" in unit
    assert "MARKET_ROLLUP_ARCHIVE_OLD_OKX_WS=1" in unit
    assert "MARKET_ROLLUP_ARCHIVE_HOT_HOURS=24" in unit
    assert "COMPACT_SMALL_FILE_MAINTENANCE=1" in unit
    assert "COMPACT_SMALL_FILE_MAINTENANCE_TIMEOUT_SECONDS=300" in unit
    assert "COMPACT_SMALL_FILE_MAINTENANCE_MAX_GROUPS=6" in unit
    assert "COMPACT_SMALL_FILE_MAINTENANCE_MAX_SOURCE_FILES_PER_GROUP=64" in unit
    assert "COMPACT_SMALL_FILE_MAINTENANCE_TARGET_ROWS=500000" in unit
    assert "COMPACT_SMALL_FILE_MAINTENANCE_DATASETS=silver/v5_quant_lab_request" in unit
    assert "Nice=10" in unit
    assert "IOSchedulingClass=best-effort" in unit
    assert "IOSchedulingPriority=7" in unit
    assert "CPUQuota=80%" in unit
    assert "MemoryHigh=3G" in unit
    assert "MemoryMax=4G" in unit
    assert "flock -E 75 -w 600 /var/lock/quant-lab-heavy.lock" in unit
    assert "--max-source-batch-bytes" in script
    assert "--direct-only" in script
    assert "visible_parquet_files" in script
    assert "-name '__*'" in script
    assert "-name '.*'" in script
    assert "! -name '*.tmp.parquet'" in script
    assert "START_DIRECT_COMPACT" in script
    assert "file_count_before" in script
    assert "file_count_after" in script
    assert "small_file_count_before" in script
    assert '"${COMPACT_DIRECT_MAX_SOURCE_FILES}"' in script
    assert "WARN_DIRECT_COMPACT_FAILED" in script
    assert "SKIP_DIRECT_COMPACT" in script
    assert "direct_source_parquet_file_count" in script
    assert "compact_hot_ws_dataset" in script
    assert "SKIP_HOT_WS_PARTITION_REPAIR" in script
    assert "compact_leaf_partitions_if_file_count_at_least" in script
    assert "SKIP_LEAF_COMPACT_BUDGET" in script
    assert '"bronze/okx_public_ws"' in script
    assert '"silver/trade_print"' in script
    assert '"silver/orderbook_snapshot"' in script
    assert 'compact_hot_ws_dataset "bronze/okx_public_ws" 500000 100 64 20' in script
    assert 'compact_hot_ws_dataset "silver/trade_print" 500000 100 20 20' in script
    assert 'compact_hot_ws_dataset "silver/orderbook_snapshot" 500000 100 64 10' in script
    assert 'compact_if_file_count_at_least "${dataset}" 250000 100 10' in script
    assert 'compact_if_file_count_at_least "${dataset}" 250000 100 20' in script
    assert "cleanup_internal_compaction_dirs" in script
    assert "__*_backup_*" in script
    assert "__*_write_*" in script
    assert "__*_repair_*" in script
    assert "-name '._tmp' -empty -mmin +60" in script
    assert '"bronze/strategy_telemetry/v5/raw_file_index"' in script
    assert '"silver/v5_quant_lab_usage"' in script
    assert '"silver/v5_candidate_event"' in script
    assert '"silver/v5_order_lifecycle"' in script
    assert '"silver/v5_roundtrip"' in script
    assert '"silver/v5_open_position"' in script
    assert '"gold/job_run_history"' in script
    assert '"bronze/api_request_metrics"' in script
    assert "OnUnitActiveSec=1h" in timer


def test_okx_ws_service_uses_unpartitioned_bounded_batches():
    unit = _unit("quant-lab-okx-ws.service")

    assert "QUANT_LAB_WS_APPEND_TARGET_ROWS=500000" in unit
    assert "QUANT_LAB_WS_APPEND_PARTITIONED=0" in unit
    assert "QUANT_LAB_APPEND_AUTO_COMPACT_FILES=0" in unit
    assert "/usr/bin/timeout 2h" in unit
    assert "--flush-interval-seconds 60" in unit
    assert "--flush-max-messages 10000" in unit
    assert "Restart=always" in unit
    assert "SuccessExitStatus=124 130 143" in unit
    assert "RuntimeMaxSec=2h" not in unit
    symbols = unit.split("--symbols ", 1)[1].split()[0].split(",")
    assert symbols == ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
    assert "--channels tickers,trades,books5" in unit
