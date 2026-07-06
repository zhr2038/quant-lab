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

    assert "analyze-v5-telemetry" in unit
    assert "--skip-candidate-gold" in unit
    assert "--compact-output" in unit
    assert "SKIP_V5_DAILY_ANALYSIS_LOCK_BUSY" in unit
    assert "flock -E 75 -w 5" in unit
    assert "/var/lock/quant-lab-v5-daily-analysis.lock" in unit
    assert "/var/lock/quant-lab-v5-research.lock" not in unit
    assert "/usr/bin/timeout 8m" in unit
    assert "build-v5-candidate-labels" not in unit
    assert "build-strategy-evidence" not in unit
    assert "build-alpha-discovery-board" not in unit
    assert "--remote-max-files 1" in sync_unit
    assert "--max-scan-bundles 1" in sync_unit
    assert "--run-analysis-after-sync" in sync_unit
    assert "--skip-analysis-after-sync" not in sync_unit
    assert "--compact-output" in sync_unit
    assert "/var/lock/quant-lab-heavy.lock" in sync_unit
    assert "flock -E 75 -w 300" in sync_unit
    assert "/usr/bin/timeout 20m" in sync_unit
    assert "TimeoutStartSec=35min" in sync_unit
    assert "MemoryHigh=5G" in sync_unit
    assert "MemoryMax=6G" in sync_unit
    assert "QUANT_LAB_V5_SYNC_REMOTE_MAX_FILES=1" in sync_unit
    assert "QUANT_LAB_V5_SYNC_MAX_SCAN_BUNDLES=1" in sync_unit


def test_api_service_uses_async_metrics_flush():
    unit = _unit("quant-lab-api.service")

    assert "QUANT_LAB_API_METRICS_ASYNC_FLUSH=1" in unit
    assert "QUANT_LAB_API_METRICS_FLUSH_ROWS=1000" in unit
    assert "QUANT_LAB_API_METRICS_FLUSH_SECONDS=300" in unit
    assert "QUANT_LAB_API_METRICS_FLUSH_JOIN_SECONDS=0.25" in unit
    assert "QUANT_LAB_API_METRICS_RESPONSE_CACHE_SECONDS=12" in unit
    assert "QUANT_LAB_API_METRICS_PRODUCTION_CLIENT_HOSTS=43.156.105.125" in unit
    assert "QUANT_LAB_WEB_ON_DEMAND_EXPORT=true" in unit
    assert "QUANT_LAB_WEB_EXPORT_BACKGROUND=true" in unit
    assert "QUANT_LAB_WEB_EXPORT_BACKGROUND_TRIGGER=request_file" in unit
    assert "QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS=1800" in unit
    assert "QUANT_LAB_BIGSCREEN_STALE_GRACE_SECONDS=0" in unit
    assert "QUANT_LAB_WEB_V2_SMOKE_REQUIRE_STATUS=1" in unit


def test_web_service_disables_bigscreen_stale_grace_by_default():
    unit = _unit("quant-lab-web.service")

    assert "QUANT_LAB_BIGSCREEN_STALE_GRACE_SECONDS=0" in unit


def test_web_v2_smoke_timer_checks_api_contracts_with_production_token():
    unit = _unit("quant-lab-web-v2-smoke.service")
    timer = _unit("quant-lab-web-v2-smoke.timer")

    assert "User=quantlab" in unit
    assert "Group=quantlab" in unit
    assert "EnvironmentFile=/etc/quant-lab/quant_lab_api.env" in unit
    assert "qlab web-v2-smoke" in unit
    assert "--base-url http://127.0.0.1:8027" in unit
    assert "--request-attempts 3" in unit
    assert "PermissionsStartOnly=true" in unit
    assert "ExecStartPre=/usr/bin/install -d -o quantlab -g quantlab" in unit
    assert "--output-json /var/lib/quant-lab/ops/web_v2_smoke/latest.json" in unit
    assert "--allow-live-cost-trust" not in unit
    assert "--allow-live-permission" not in unit
    assert "OnUnitActiveSec=10min" in timer
    assert "RandomizedDelaySec=60s" in timer


def test_okx_rest_backfill_runs_every_15_minutes_to_reduce_stale_market_bar_window():
    timer = _unit("quant-lab-okx-rest-backfill.timer")
    service = _unit("quant-lab-okx-rest-backfill.service")

    assert "every 15 minutes" in timer
    assert "OnActiveSec=15min" in timer
    assert "OnUnitActiveSec=15min" in timer
    assert "OnUnitActiveSec=1h" not in timer
    assert "okx-fetch-candles" in service
    assert "--history" not in service


def test_expanded_universe_backfill_stays_within_bigscreen_freshness_window():
    timer = _unit("quant-lab-okx-expanded-universe-backfill.timer")
    service = _unit("quant-lab-okx-expanded-universe-backfill.service")

    assert "every hour" in timer
    assert "OnUnitActiveSec=1h" in timer
    assert "OnUnitActiveSec=6h" not in timer
    assert "okx-backfill-expanded-universe" in service
    assert "build-expanded-universe-shadow" in service


def test_daily_export_uses_recent_api_metrics_window():
    unit = _unit("quant-lab-daily-export.service")

    assert "QUANT_LAB_API_METRICS_EXPORT_WINDOW_MINUTES=30" in unit
    assert "QUANT_LAB_API_METRICS_PRODUCTION_CLIENT_HOSTS=43.156.105.125" in unit
    assert "QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES=12" in unit
    assert "QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES=1000" in unit
    assert "MemoryHigh=5G" in unit
    assert "MemoryMax=6G" in unit
    assert "/usr/bin/timeout 30m" in unit
    assert "TimeoutStartSec=75min" in unit
    assert "ExecStartPre=/usr/bin/systemctl start quant-lab-cost-calibration.service" in unit


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


def test_scheduled_lock_contention_is_reported_as_successful_skip():
    expected_skip_markers = {
        "quant-lab-cost-calibration.service": "SKIP_COST_CALIBRATION_LOCK_BUSY",
        "quant-lab-daily-export.service": "SKIP_DAILY_EXPORT_LOCK_BUSY",
        "quant-lab-entry-quality-history.service": "SKIP_ENTRY_QUALITY_HISTORY_LOCK_BUSY",
        "quant-lab-lake-compaction.service": "SKIP_LAKE_COMPACTION_LOCK_BUSY",
        "quant-lab-storage-retention.service": "SKIP_STORAGE_RETENTION_LOCK_BUSY",
        "quant-lab-v5-daily-analysis.service": "SKIP_V5_DAILY_ANALYSIS_LOCK_BUSY",
        "quant-lab-v5-regime-router.service": "SKIP_V5_REGIME_ROUTER_LOCK_BUSY",
        "quant-lab-v5-research-refresh.service": "SKIP_V5_RESEARCH_REFRESH_LOCK_BUSY",
        "quant-lab-v5-telemetry-sync.service": "SKIP_V5_TELEMETRY_SYNC_LOCK_BUSY",
    }

    for unit_name, marker in expected_skip_markers.items():
        unit = _unit(unit_name)

        assert "flock -E 75" in unit, unit_name
        assert marker in unit, unit_name
        assert 'if [ "$${code}" = "75" ]' in unit, unit_name
        assert "exit 0" in unit, unit_name
        assert 'exit "$${code}"' in unit, unit_name


def test_cost_calibration_starts_readonly_private_backfill_when_configured():
    unit = _unit("quant-lab-cost-calibration.service")

    assert "PermissionsStartOnly=true" in unit
    assert "ExecStartPre=/bin/bash -lc" in unit
    assert "[ -f /etc/quant-lab/okx_readonly.env ]" in unit
    assert "/usr/bin/systemctl start quant-lab-okx-readonly-backfill.service" in unit
    assert "SKIP_OKX_READONLY_BACKFILL_ENV_MISSING" in unit
    assert "flock -E 75 -w 600" in unit
    assert "TimeoutStartSec=30min" in unit


def test_storage_retention_does_not_create_root_owned_lake_files():
    unit = _unit("quant-lab-storage-retention.service")
    timer = _unit("quant-lab-storage-retention.timer")

    assert "User=quantlab" in unit
    assert "Group=quantlab" in unit
    assert "PermissionsStartOnly=true" in unit
    assert "prune-storage-retention --base-dir /var/lib/quant-lab" in unit
    assert "--keep-restricted-archive-days 7" in unit
    assert "--keep-high-frequency-archive-days 3" in unit
    assert "journalctl --vacuum-size=200M" in unit
    assert "OnCalendar=*-*-* 09:20:00" in timer


def test_lake_permission_repair_script_targets_service_user():
    script = _script("repair_lake_permissions.sh")

    assert "LAKE_ROOT=\"${LAKE_ROOT:-/var/lib/quant-lab/lake}\"" in script
    assert "QUANT_LAB_BASE_DIR=" in script
    assert "EXPORTS_DIR=" in script
    assert "QUANT_LAB_USER=\"${QUANT_LAB_USER:-quantlab}\"" in script
    assert "QUANT_LAB_GROUP=\"${QUANT_LAB_GROUP:-quantlab}\"" in script
    assert "install -d" in script
    assert "chown -R" in script
    assert "chmod u+rwX,g+rwX,o+rX,g+s" in script
    assert "chmod u+rw,g+rw,o+r" in script


def test_deploy_permission_repair_script_targets_deploy_user():
    script = _script("repair_deploy_permissions.sh")

    assert "APP_ROOT=\"${APP_ROOT:-/opt/quant-lab}\"" in script
    assert "DEPLOY_USER=\"${DEPLOY_USER:-ubuntu}\"" in script
    assert "SERVICE_GROUP=\"${SERVICE_GROUP:-quantlab}\"" in script
    assert "START_REPAIR_DEPLOY_PERMISSIONS" in script
    assert "chown -R \"${DEPLOY_USER}:${SERVICE_GROUP}\" \"${APP_ROOT}\"" in script
    assert "chmod u=rwx,g=rx,o=,g+s" in script
    assert "chmod u=rwX,g=rX,o=" in script
    assert "g+rw" not in script
    assert "FINISH_REPAIR_DEPLOY_PERMISSIONS" in script


def test_candidate_research_refresh_is_separate_from_alpha_evidence():
    alpha_unit = _unit("quant-lab-alpha-evidence.service")
    refresh_unit = _unit("quant-lab-v5-research-refresh.service")
    regime_unit = _unit("quant-lab-v5-regime-router.service")

    assert "build-alpha-evidence" in alpha_unit
    assert "build-v5-candidate-labels" not in alpha_unit
    assert "build-strategy-evidence" not in alpha_unit
    assert "build-alpha-discovery-board" not in alpha_unit

    assert "build-v5-candidate-labels" in refresh_unit
    assert "--mode incremental --lookback-days 8" in refresh_unit
    assert "build-strategy-evidence" in refresh_unit
    assert "--skip-historical-outcomes" in refresh_unit
    assert "build-factor-factory" in refresh_unit
    assert "--horizon-bars 4,8,24,72" in refresh_unit
    assert "build-alpha-discovery-board" in refresh_unit
    assert "--skip-legacy-outcome-counts" in refresh_unit
    assert "build-paper-strategy-tracking" in refresh_unit
    assert "build-research-portfolio-status" in refresh_unit
    assert "refresh-research-diagnostics" in refresh_unit
    assert "refresh-web-derived-snapshots" in refresh_unit
    assert "build-sol-protect-paper-loss-attribution" not in refresh_unit
    assert "build-btc-probe-exit-policy-review" not in refresh_unit
    assert "build-bnb-swing-exit-policy-review" not in refresh_unit
    assert "build-entry-quality" in refresh_unit
    assert "build-regime-router" not in refresh_unit
    assert "flock -E 75 -w 600 /var/lock/quant-lab-heavy.lock" in refresh_unit
    assert "flock -E 75 -w 30 /var/lock/quant-lab-v5-research.lock" in refresh_unit
    assert "/usr/bin/timeout 20m" in refresh_unit
    assert "TimeoutStartSec=25min" in refresh_unit

    assert "build-regime-router" in regime_unit
    assert "/var/lock/quant-lab-v5-regime-router.lock" in regime_unit
    assert "/var/lock/quant-lab-v5-research.lock" in regime_unit
    assert "/usr/bin/timeout 10m" in regime_unit
    assert "TimeoutStartSec=12min" in regime_unit
    assert "MemoryMax=1G" in regime_unit


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
    assert (
        'compact_hot_ws_dataset "silver/orderbook_snapshot" 500000 100 64 10'
        in script
    )
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


def test_candidate_research_refresh_runs_before_daily_export_window():
    refresh_timer = _unit("quant-lab-v5-research-refresh.timer")
    regime_timer = _unit("quant-lab-v5-regime-router.timer")
    export_timer = _unit("quant-lab-daily-export.timer")

    assert "OnCalendar=*-*-* 00:05:00" in refresh_timer
    assert "OnActiveSec=10min" in refresh_timer
    assert "OnUnitActiveSec=1h" in refresh_timer
    assert "OnUnitActiveSec=2h" not in refresh_timer
    assert "OnCalendar=*-*-* 00:12:00" in regime_timer
    assert "OnUnitActiveSec=30min" in regime_timer
    assert "OnCalendar=*-*-* 00:20:00" in export_timer


def test_entry_quality_history_refresh_is_scheduled_separately():
    unit = _unit("quant-lab-entry-quality-history.service")
    timer = _unit("quant-lab-entry-quality-history.timer")

    assert "build-entry-quality-history" in unit
    assert "--mode recent_30d" in unit
    assert "--cost-mode conservative" in unit
    assert "date -u -d" in unit
    assert "$${END_DATE}" in unit
    assert "$${START_DATE}" in unit
    assert "/var/lock/quant-lab-entry-quality-history.lock" in unit
    assert "TimeoutStartSec=20min" in unit
    assert "MemoryMax=3G" in unit
    assert "OnCalendar=*-*-* 01:35:00" in timer
    assert "OnCalendar=*-*-* 13:35:00" in timer
    assert "Persistent=true" in timer


def test_daily_export_template_refreshes_v5_before_packaging():
    unit = _unit("quant-lab-daily-export.service")

    assert "export-daily" in unit
    assert "--no-refresh-risk-permission" in unit
    assert "--pre-export-v5-refresh" in unit
    assert "--allow-stale-v5" not in unit
    assert "--no-pre-export-v5-refresh" not in unit
    assert "TimeoutStartSec=75min" in unit
    assert "/var/lock/quant-lab-heavy.lock" in unit
    assert "/var/lock/quant-lab-v5-telemetry-sync.lock" in unit
    assert "SKIP_DAILY_EXPORT_LOCK_BUSY" in unit
    assert "ExecStartPre=/usr/bin/systemctl start quant-lab-v5-telemetry-sync.service" in unit
    assert "ExecStartPre=/usr/bin/systemctl start quant-lab-cost-calibration.service" in unit
    assert (
        "ExecStartPre=/opt/quant-lab/.venv/bin/qlab publish-risk-permission "
        "--lake-root /var/lib/quant-lab/lake --strategy v5 --version 5.0.0"
    ) in unit
    assert "EnvironmentFile=-/etc/quant-lab/quant_lab_api.env" in unit


def test_web_export_request_refreshes_costs_before_packaging():
    unit = _unit("quant-lab-web-export-request.service")

    assert "TimeoutStartSec=65min" in unit
    assert "ExecStartPre=/usr/bin/systemctl start quant-lab-v5-telemetry-sync.service" in unit
    assert "ExecStartPre=/usr/bin/systemctl start quant-lab-cost-calibration.service" in unit
    assert "QUANT_LAB_EXPORT_GITHUB_CI_STATUS=1" in unit


def test_web_export_relies_on_systemd_memory_limit_for_snapshot_packaging():
    unit = _unit("quant-lab-web.service")

    assert "QUANT_LAB_WEB_EXPORT_MEMORY_LIMIT_MB=0" in unit
    assert "QUANT_LAB_WEB_EXPORT_BACKGROUND_TRIGGER=request_file" in unit
    assert "MemoryMax=5G" in unit
    assert "KillSignal=SIGINT" in unit
    assert "KillMode=mixed" in unit
    assert "TimeoutStopSec=20s" in unit


def test_web_export_request_worker_is_scheduled_outside_dashboard_cgroup():
    service = _unit("quant-lab-web-export-request.service")
    path = _unit("quant-lab-web-export-request.path")

    assert "run-web-export-request" in service
    assert "--request-path /var/lib/quant-lab/exports/.quant_lab_web_export_request.json" in service
    assert "PermissionsStartOnly=true" in service
    assert (
        "ExecStartPre=/usr/bin/systemctl start quant-lab-v5-telemetry-sync.service"
        in service
    )
    assert "flock -E 75 -w 600 /var/lock/quant-lab-heavy.lock" in service
    assert "flock -E 75 -w 600 /var/lock/quant-lab-v5-telemetry-sync.lock" in service
    assert "SKIP_WEB_EXPORT_LOCK_BUSY" in service
    assert "QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES=12" in service
    assert "QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES=1000" in service
    assert "QUANT_LAB_API_METRICS_PRODUCTION_CLIENT_HOSTS=43.156.105.125" in service
    assert "QUANT_LAB_EXPORT_GITHUB_CI_STATUS=1" in service
    assert "QUANT_LAB_GITHUB_REPO=zhr2038/quant-lab" in service
    assert "V5_GITHUB_REPO=zhr2038/V5-prod" in service
    assert "MemoryMax=6G" in service
    request_path = "/var/lib/quant-lab/exports/.quant_lab_web_export_request.json"
    assert f"PathExists={request_path}" in path
    assert f"PathChanged={request_path}" in path
    assert f"PathModified={request_path}" in path
    assert "Unit=quant-lab-web-export-request.service" in path


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
    for symbol in [
        "BTC-USDT",
        "ETH-USDT",
        "SOL-USDT",
        "BNB-USDT",
        "ADA-USDT",
        "ASTER-USDT",
        "BASED-USDT",
        "CHZ-USDT",
        "DASH-USDT",
        "FIL-USDT",
        "GRASS-USDT",
        "HYPE-USDT",
        "ICP-USDT",
        "IP-USDT",
        "JTO-USDT",
        "LINK-USDT",
        "LIT-USDT",
        "LPT-USDT",
        "LTC-USDT",
        "MON-USDT",
        "NEAR-USDT",
        "OKB-USDT",
        "ONDO-USDT",
        "PAXG-USDT",
        "PROS-USDT",
        "SUI-USDT",
        "TON-USDT",
        "TRUMP-USDT",
        "TRX-USDT",
        "WLD-USDT",
        "XAUT-USDT",
        "XRP-USDT",
        "ZEC-USDT",
    ]:
        assert symbol in unit
    assert "--channels tickers,trades,books5" in unit


def test_manual_okx_ws_defaults_match_production_batching():
    cli = (ROOT / "src" / "quant_lab" / "cli.py").read_text(encoding="utf-8")
    readers = (ROOT / "src" / "quant_lab" / "web" / "readers.py").read_text(encoding="utf-8")

    assert "] = 60.0" in cli
    assert "] = 10_000" in cli
    assert "--flush-interval-seconds 60 --flush-max-messages 10000" in readers
