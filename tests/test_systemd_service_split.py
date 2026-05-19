from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYSTEMD = ROOT / "deploy" / "systemd"


def _unit(name: str) -> str:
    return (SYSTEMD / name).read_text(encoding="utf-8")


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

    assert "compact-lake-dataset" in unit
    assert "--dataset bronze/okx_public_ws" in unit
    assert "--dataset silver/trade_print" in unit
    assert "--dataset silver/orderbook_snapshot" in unit
    assert "--target-rows-per-file 500000" in unit
    assert "--max-source-files-per-batch 10000" in unit
    assert "bronze/strategy_telemetry/v5/raw_file_index" in unit
    assert "silver/v5_quant_lab_usage" in unit
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
