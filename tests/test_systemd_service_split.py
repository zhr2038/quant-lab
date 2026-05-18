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
    assert "build-strategy-evidence" in refresh_unit
    assert "build-alpha-discovery-board" in refresh_unit


def test_scheduled_compaction_avoids_hot_ws_datasets():
    unit = _unit("quant-lab-lake-compaction.service")

    assert "compact-lake-dataset" in unit
    assert "--dataset okx_public_ws" not in unit
    assert "--dataset bronze/okx_public_ws" not in unit
    assert "--dataset trade_print" not in unit
    assert "--dataset silver/trade_print" not in unit
    assert "--dataset orderbook_snapshot" not in unit
    assert "--dataset silver/orderbook_snapshot" not in unit
    assert "bronze/strategy_telemetry/v5/raw_file_index" in unit
    assert "silver/v5_quant_lab_usage" in unit


def test_daily_export_template_is_packaging_only():
    unit = _unit("quant-lab-daily-export.service")

    assert "export-daily" in unit
    assert "--no-refresh-risk-permission" in unit
    assert "--no-pre-export-v5-refresh" in unit
