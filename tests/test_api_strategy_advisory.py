import zipfile
from datetime import UTC, datetime

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.data.lake import write_parquet_dataset


def test_strategy_opportunity_advisory_endpoint_reads_gold(tmp_path, monkeypatch):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_ts": datetime(2026, 5, 20, tzinfo=UTC),
                    "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
                    "symbol": "SOL-USDT",
                    "decision": "PAPER_READY",
                    "recommended_mode": "paper",
                    "horizon_hours": 24,
                    "sample_count": 72,
                    "complete_sample_count": 40,
                    "avg_net_bps": 22.5,
                    "p25_net_bps": -18.0,
                    "win_rate": 0.62,
                    "cost_source_mix": '{"public_spread_proxy":72}',
                    "live_block_reasons": '["cost_source_not_actual_or_mixed"]',
                    "max_paper_notional_usdt": 1000.0,
                    "max_live_notional_usdt": 500.0,
                },
                {
                    "as_of_ts": datetime(2026, 5, 20, tzinfo=UTC),
                    "strategy_candidate": "v5.multi_position_k3",
                    "symbol": "BNB/USDT",
                    "decision": "KILL",
                    "recommended_mode": "paper",
                    "horizon_hours": 8,
                    "sample_count": 20,
                    "complete_sample_count": 18,
                    "avg_net_bps": -44.0,
                    "p25_net_bps": -91.0,
                    "win_rate": 0.22,
                    "cost_source_mix": '{"mixed_actual_proxy":18}',
                    "live_block_reasons": "non_positive_after_cost_edge",
                    "max_paper_notional_usdt": 0.0,
                    "max_live_notional_usdt": 250.0,
                },
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    response = TestClient(app).get("/v1/strategy-opportunity-advisory")

    assert response.status_code == 200
    rows = response.json()
    paper = next(row for row in rows if row["decision"] == "PAPER_READY")
    killed = next(row for row in rows if row["decision"] == "KILL")
    assert paper["strategy_id"] == "SOL_USDT_V5_SOL_PROTECT_ALPHA6_LOW_EXCEPTION"
    assert paper["max_live_notional_usdt"] == 0.0
    assert killed["symbol"] == "BNB-USDT"
    assert killed["recommended_mode"] == "none"
    assert killed["max_live_notional_usdt"] == 0.0


def test_strategy_opportunity_advisory_aliases_and_latest_report_fallback(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    exports = tmp_path / "exports"
    exports.mkdir()
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    csv_text = (
        "as_of_ts,symbol,strategy_candidate,decision,recommended_mode,horizon_hours,"
        "sample_count,complete_sample_count,avg_net_bps,p25_net_bps,win_rate,"
        "cost_source_mix,live_block_reasons,max_paper_notional_usdt,"
        "max_live_notional_usdt\n"
        "2026-05-20T08:00:00Z,ETH-USDT,v5.f4_volume_expansion_entry,"
        "KEEP_SHADOW,shadow,12,14,12,9.5,-20.0,0.58,"
        "\"{\"\"public_spread_proxy\"\":14}\",\"[\"\"needs_paper_observation\"\"]\","
        "500,0\n"
    )
    with zipfile.ZipFile(exports / "quant_lab_expert_pack_2026-05-20_080000.zip", "w") as archive:
        archive.writestr("reports/strategy_opportunity_advisory.csv", csv_text)

    client = TestClient(app)
    dashed = client.get("/v1/strategy-opportunity-advisory")
    underscored = client.get("/v1/strategy_opportunity_advisory")
    report_alias = client.get("/v1/reports/strategy-opportunity-advisory")

    assert dashed.status_code == 200
    assert underscored.status_code == 200
    assert report_alias.status_code == 200
    assert dashed.json() == underscored.json() == report_alias.json()
    assert dashed.json()[0]["strategy_candidate"] == "v5.f4_volume_expansion_entry"
