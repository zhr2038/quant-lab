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

    assert "analyze-v5-telemetry" in unit
    assert "--skip-candidate-gold" in unit
    assert "build-v5-candidate-labels" not in unit
    assert "build-strategy-evidence" not in unit
    assert "build-alpha-discovery-board" not in unit


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


def test_scheduled_compaction_covers_hot_ws_datasets():
    unit = _unit("quant-lab-lake-compaction.service")
    timer = _unit("quant-lab-lake-compaction.timer")
    script = _script("compact_lake_hot_datasets.sh")

    assert "compact_lake_hot_datasets.sh" in unit
    assert "compact-lake-dataset" in script
    assert '"bronze/okx_public_ws"' in script
    assert '"silver/trade_print"' in script
    assert '"silver/orderbook_snapshot"' in script
    assert "compact_if_file_count_at_least \"${dataset}\" 500000 10000 500" in script
    assert "compact_if_file_count_at_least \"${dataset}\" 250000 5000 100" in script
    assert '"bronze/strategy_telemetry/v5/raw_file_index"' in script
    assert '"silver/v5_quant_lab_usage"' in script
    assert '"silver/v5_candidate_event"' in script
    assert "OnUnitActiveSec=2h" in timer


def test_daily_export_template_is_packaging_only():
    unit = _unit("quant-lab-daily-export.service")

    assert "export-daily" in unit
    assert "--no-refresh-risk-permission" in unit
    assert "--no-pre-export-v5-refresh" in unit


def test_web_export_memory_limit_allows_snapshot_packaging():
    unit = _unit("quant-lab-web.service")

    assert "QUANT_LAB_WEB_EXPORT_MEMORY_LIMIT_MB=3072" in unit
    assert "MemoryMax=5G" in unit
