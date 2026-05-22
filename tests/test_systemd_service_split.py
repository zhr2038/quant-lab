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
    assert "build-v5-candidate-labels" not in unit
    assert "build-strategy-evidence" not in unit
    assert "build-alpha-discovery-board" not in unit
    assert "--remote-max-files 1" in sync_unit
    assert "--max-scan-bundles 1" in sync_unit
    assert "--skip-analysis-after-sync" in sync_unit
    assert "--compact-output" in sync_unit
    assert "QUANT_LAB_V5_SYNC_REMOTE_MAX_FILES=1" in sync_unit
    assert "QUANT_LAB_V5_SYNC_MAX_SCAN_BUNDLES=1" in sync_unit


def test_api_service_uses_async_metrics_flush():
    unit = _unit("quant-lab-api.service")

    assert "QUANT_LAB_API_METRICS_ASYNC_FLUSH=1" in unit


def test_candidate_research_refresh_is_separate_from_alpha_evidence():
    alpha_unit = _unit("quant-lab-alpha-evidence.service")
    refresh_unit = _unit("quant-lab-v5-research-refresh.service")

    assert "build-alpha-evidence" in alpha_unit
    assert "build-v5-candidate-labels" not in alpha_unit
    assert "build-strategy-evidence" not in alpha_unit
    assert "build-alpha-discovery-board" not in alpha_unit

    assert "build-v5-candidate-labels" in refresh_unit
    assert "--mode incremental --lookback-days 8" in refresh_unit
    assert "build-strategy-evidence" in refresh_unit
    assert "--skip-historical-outcomes" in refresh_unit
    assert "build-alpha-discovery-board" in refresh_unit
    assert "--skip-legacy-outcome-counts" in refresh_unit
    assert "build-paper-strategy-tracking" in refresh_unit
    assert "build-entry-quality" in refresh_unit


def test_scheduled_compaction_covers_hot_ws_datasets():
    unit = _unit("quant-lab-lake-compaction.service")
    timer = _unit("quant-lab-lake-compaction.timer")
    script = _script("compact_lake_hot_datasets.sh")

    assert "compact_lake_hot_datasets.sh" in unit
    assert "compact-lake-dataset" in script
    assert "repair-lake-partitions" in script
    assert "START_REPAIR_PARTITIONS" in script
    assert "WARN_REPAIR_PARTITIONS_FAILED" in script
    assert "COMPACT_DATASET_TIMEOUT_SECONDS" in script
    assert "COMPACT_RUN_BUDGET_SECONDS" in script
    assert "COMPACT_DIRECT_MAX_SOURCE_FILES" in script
    assert "COMPACT_MAX_SOURCE_BATCH_BYTES" in script
    assert "WARN_COMPACT_FAILED" in script
    assert "SKIP_COMPACT_BUDGET" in script
    assert "WARN_LAKE_HEALTH_FAILED_OR_TIMED_OUT" in script
    assert "lake-health --lake-root" in script
    assert "--compact-output" in script
    assert script.count("--compact-output") >= 4
    assert "COMPACT_RAW_OKX_WS" in script
    assert "SKIP_COMPACT_RAW_OKX_WS" in script
    assert "COMPACT_RAW_OKX_WS=1" in unit
    assert "COMPACT_DATASET_TIMEOUT_SECONDS=300" in unit
    assert "COMPACT_DIRECT_MAX_SOURCE_FILES=8" in unit
    assert "COMPACT_MAX_SOURCE_BATCH_BYTES=134217728" in unit
    assert "--max-source-batch-bytes" in script
    assert "--direct-only" in script
    assert "START_DIRECT_COMPACT" in script
    assert '"${COMPACT_DIRECT_MAX_SOURCE_FILES}"' in script
    assert "WARN_DIRECT_COMPACT_FAILED" in script
    assert "compact_leaf_partitions_if_file_count_at_least" in script
    assert "SKIP_LEAF_COMPACT_BUDGET" in script
    assert '"bronze/okx_public_ws"' in script
    assert '"silver/trade_print"' in script
    assert '"silver/orderbook_snapshot"' in script
    assert (
        'repair_dataset_partitions "bronze/okx_public_ws" '
        "500000 100"
    ) in script
    assert (
        'compact_leaf_partitions_if_file_count_at_least "bronze/okx_public_ws" '
        "500000 100 20"
    ) in script
    assert 'repair_dataset_partitions "silver/trade_print" 500000 100' in script
    assert (
        'compact_leaf_partitions_if_file_count_at_least "silver/trade_print" '
        "500000 100 20"
    ) in script
    assert 'repair_dataset_partitions "silver/orderbook_snapshot" 500000 100' in script
    assert (
        'compact_leaf_partitions_if_file_count_at_least "silver/orderbook_snapshot" '
        "500000 100 10"
    ) in script
    assert 'compact_if_file_count_at_least "${dataset}" 250000 100 10' in script
    assert 'compact_if_file_count_at_least "${dataset}" 250000 100 20' in script
    assert "cleanup_internal_compaction_dirs" in script
    assert "__*_backup_*" in script
    assert "__*_repair_*" in script
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
    export_timer = _unit("quant-lab-daily-export.timer")

    assert "OnCalendar=*-*-* 00:05:00" in refresh_timer
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


def test_daily_export_template_is_packaging_only():
    unit = _unit("quant-lab-daily-export.service")

    assert "export-daily" in unit
    assert "--no-refresh-risk-permission" in unit
    assert "--no-pre-export-v5-refresh" in unit


def test_web_export_memory_limit_allows_snapshot_packaging():
    unit = _unit("quant-lab-web.service")

    assert "QUANT_LAB_WEB_EXPORT_MEMORY_LIMIT_MB=3072" in unit
    assert "MemoryMax=5G" in unit


def test_okx_ws_service_uses_unpartitioned_large_batches():
    unit = _unit("quant-lab-okx-ws.service")

    assert "QUANT_LAB_WS_APPEND_TARGET_ROWS=500000" in unit
    assert "QUANT_LAB_WS_APPEND_PARTITIONED=0" in unit
    assert "--flush-interval-seconds 600" in unit
    assert "--flush-max-messages 50000" in unit
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

    assert "] = 600.0" in cli
    assert "] = 50_000" in cli
    assert "--flush-interval-seconds 600 --flush-max-messages 50000" in readers
