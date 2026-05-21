import zipfile
from datetime import UTC, datetime

import polars as pl
from fastapi.testclient import TestClient

from quant_lab.api.main import app
from quant_lab.data.lake import write_parquet_dataset

V5_ADVISORY_FIELDS = {
    "as_of_ts",
    "generated_at",
    "expires_at",
    "contract_version",
    "schema_version",
    "strategy_id",
    "strategy_candidate",
    "symbol",
    "decision",
    "recommended_mode",
    "horizon_hours",
    "sample_count",
    "complete_sample_count",
    "avg_net_bps",
    "p25_net_bps",
    "win_rate",
    "cost_source_mix",
    "live_block_reasons",
    "max_paper_notional_usdt",
    "max_live_notional_usdt",
}


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
    assert paper["as_of_ts"].startswith("2026-05-20T00:00:00")
    assert paper["generated_at"]
    assert paper["expires_at"]
    assert paper["contract_version"] == "v5_quant_lab_contract.v0.1"
    assert paper["schema_version"] == "strategy_opportunity_advisory.v0.1"
    assert paper["max_live_notional_usdt"] == 0.0
    assert killed["symbol"] == "BNB-USDT"
    assert killed["recommended_mode"] == "none"
    assert killed["max_live_notional_usdt"] == 0.0


def test_strategy_opportunity_advisory_response_is_v5_parseable(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_ts": datetime(2026, 5, 20, tzinfo=UTC),
                    "strategy_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "decision": "LIVE_SMALL_READY",
                    "recommended_mode": "live_small",
                    "horizon_hours": 24,
                    "sample_count": 80,
                    "complete_sample_count": 64,
                    "avg_net_bps": 31.5,
                    "p25_net_bps": -8.0,
                    "win_rate": 0.68,
                    "cost_source_mix": '{"mixed_actual_proxy":64}',
                    "live_block_reasons": "[]",
                    "max_paper_notional_usdt": 1000.0,
                    "max_live_notional_usdt": 250.0,
                }
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    response = TestClient(app).get("/v1/strategy-opportunity-advisory")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert V5_ADVISORY_FIELDS <= set(payload[0])
    assert payload[0]["strategy_id"] == "SOL_F4_VOLUME_EXPANSION_PAPER_V1"
    assert payload[0]["decision"] == "LIVE_SMALL_READY"
    assert payload[0]["max_live_notional_usdt"] == 250.0
    assert isinstance(payload[0]["live_block_reasons"], list)
    assert payload[0]["generated_at"]
    assert payload[0]["expires_at"]


def test_strategy_opportunity_advisory_dedupes_legacy_schema_rows(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    as_of = datetime(2026, 5, 20, tzinfo=UTC)
    base = {
        "as_of_ts": as_of,
        "generated_at": as_of,
        "expires_at": datetime(2026, 5, 20, 3, tzinfo=UTC),
        "contract_version": "v5_quant_lab_contract.v0.1",
        "strategy_id": "ETH_ENTRY_QUALITY",
        "strategy_candidate": "v5.pullback_reversal_shadow_eth",
        "symbol": "ETH-USDT",
        "decision": "KEEP_SHADOW",
        "recommended_mode": "shadow",
        "horizon_hours": None,
        "sample_count": 10,
        "complete_sample_count": 10,
        "avg_net_bps": 5.0,
        "p25_net_bps": -12.0,
        "win_rate": 0.55,
        "cost_source_mix": '{"entry_quality_research":10}',
        "live_block_reasons": '["shadow_only"]',
        "max_paper_notional_usdt": 0.0,
        "max_live_notional_usdt": 0.0,
    }
    write_parquet_dataset(
        pl.DataFrame(
            [
                {**base, "schema_version": "entry_quality.v0.1"},
                {
                    **base,
                    "schema_version": "strategy_opportunity_advisory.v0.1",
                    "sample_count": 12,
                },
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    response = TestClient(app).get("/v1/strategy-opportunity-advisory")

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "strategy_opportunity_advisory.v0.1"
    assert rows[0]["sample_count"] == 12


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
    assert dashed.json()[0]["contract_version"] == "v5_quant_lab_contract.v0.1"
    assert dashed.json()[0]["expires_at"]
