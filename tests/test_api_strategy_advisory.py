import zipfile
from datetime import UTC, datetime

import polars as pl
from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
from quant_lab.api.main import app
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import write_parquet_dataset

V5_ADVISORY_FIELDS = {
    "as_of_ts",
    "generated_at",
    "expires_at",
    "contract_version",
    "schema_version",
    "quant_lab_git_commit",
    "source_version",
    "would_block_if_enabled",
    "would_enter",
    "no_sample_reason",
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
                    "contract_version": "v5_quant_lab_contract.v0.1",
                    "quant_lab_git_commit": "not_observable",
                    "source_version": "not_observable",
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
                    "contract_version": "v5_quant_lab_contract.v0.1",
                    "quant_lab_git_commit": "not_observable",
                    "source_version": "not_observable",
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
    assert paper["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
    assert paper["schema_version"] == "strategy_opportunity_advisory.v0.1"
    assert paper["quant_lab_git_commit"] not in {"", None, "not_observable"}
    assert paper["source_version"].startswith("strategy_opportunity_advisory:")
    assert paper["source_version"] != "not_observable"
    assert paper["would_enter"] is True
    assert paper["would_block_if_enabled"] is False
    assert paper["no_sample_reason"] is None
    assert paper["max_live_notional_usdt"] == 0.0
    assert killed["symbol"] == "BNB-USDT"
    assert killed["recommended_mode"] == "none"
    assert killed["would_block_if_enabled"] is True
    assert killed["would_enter"] is False
    assert killed["no_sample_reason"] == "killed_candidate"
    assert killed["max_live_notional_usdt"] == 0.0


def test_strategy_opportunity_advisory_endpoint_applies_portfolio_final_overlay(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    as_of = datetime(2026, 5, 24, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _api_advisory_row("v5.f3_dominant_entry", "ETH-USDT", as_of),
                _api_advisory_row("v5.af.failed_candidate", "NEAR-USDT", as_of),
                _api_advisory_row("v5.af.paused_candidate", "WLD-USDT", as_of),
                _api_advisory_row("v5.core.momentum", "BTC-USDT", as_of),
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                _portfolio_status_row(
                    research_id="v5.af.paused_candidate",
                    strategy_candidate="v5.af.paused_candidate",
                    status="KILL",
                    as_of_date="2026-05-23",
                ),
                _portfolio_status_row(
                    research_id="ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                    strategy_candidate="v5.f3_dominant_entry",
                    status="DOWNGRADED_FROM_PAPER",
                ),
                _portfolio_status_row(
                    research_id="v5.af.failed_candidate",
                    strategy_candidate="v5.af.failed_candidate",
                    status="KILL",
                ),
                _portfolio_status_row(
                    research_id="v5.af.paused_candidate",
                    strategy_candidate="v5.af.paused_candidate",
                    status="PAUSED",
                ),
                _portfolio_status_row(
                    research_id="v5.core.momentum",
                    strategy_candidate="v5.core.momentum",
                    status="BASELINE_ONLY",
                ),
            ]
        ),
        lake / "gold" / "research_portfolio_status",
    )

    rows = TestClient(app).get("/v1/strategy-opportunity-advisory").json()

    by_candidate = {row["strategy_candidate"]: row for row in rows}
    eth = by_candidate["v5.f3_dominant_entry"]
    failed = by_candidate["v5.af.failed_candidate"]
    paused = by_candidate["v5.af.paused_candidate"]
    baseline = by_candidate["v5.core.momentum"]
    assert eth["recommended_mode"] == "shadow"
    assert eth["decision"] == "KEEP_SHADOW"
    assert "downgraded_from_paper" in eth["live_block_reasons"]
    assert failed["decision"] == "KILL"
    assert failed["recommended_mode"] == "none"
    assert "research_portfolio_kill" in failed["live_block_reasons"]
    assert paused["recommended_mode"] == "research"
    assert "research_paused" in paused["live_block_reasons"]
    assert baseline["recommended_mode"] == "research"
    assert "baseline_only" in baseline["live_block_reasons"]
    assert all(row["max_live_notional_usdt"] == 0.0 for row in rows)


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
    assert payload[0]["max_live_notional_usdt"] == 0.0
    assert isinstance(payload[0]["live_block_reasons"], list)
    assert payload[0]["generated_at"]
    assert payload[0]["expires_at"]


def test_strategy_opportunity_advisory_uses_lazy_scan_not_full_dataset_read(
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
                    "decision": "PAPER_READY",
                    "recommended_mode": "paper",
                    "horizon_hours": 24,
                    "sample_count": 80,
                    "complete_sample_count": 64,
                    "avg_net_bps": 31.5,
                    "p25_net_bps": -8.0,
                    "win_rate": 0.68,
                    "cost_source_mix": '{"public_spread_proxy":64}',
                    "live_block_reasons": "[]",
                    "max_paper_notional_usdt": 1000.0,
                    "max_live_notional_usdt": 250.0,
                }
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    def fail_full_read(*_args, **_kwargs):
        raise AssertionError("strategy advisory API should not eager-read full dataset")

    monkeypatch.setattr("quant_lab.api.main.read_parquet_dataset", fail_full_read)

    response = TestClient(app).get("/v1/strategy-opportunity-advisory")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["strategy_candidate"] == "v5.f4_volume_expansion_entry"
    assert payload[0]["max_live_notional_usdt"] == 0.0


def test_strategy_opportunity_advisory_caches_unchanged_source_signature(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    api_main._STRATEGY_OPPORTUNITY_ADVISORY_CACHE.clear()
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_ts": datetime(2026, 5, 20, tzinfo=UTC),
                    "strategy_id": "SOL_F4_VOLUME_EXPANSION_PAPER_V1",
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "decision": "PAPER_READY",
                    "recommended_mode": "paper",
                    "horizon_hours": 24,
                    "sample_count": 80,
                    "cost_source_mix": '{"public_spread_proxy":80}',
                    "max_live_notional_usdt": 0.0,
                }
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )
    client = TestClient(app)

    first = client.get("/v1/strategy-opportunity-advisory")

    def fail_gold_rows(*_args, **_kwargs):
        raise AssertionError("unchanged advisory source should be served from cache")

    monkeypatch.setattr(api_main, "_strategy_opportunity_advisory_gold_rows", fail_gold_rows)
    second = client.get("/v1/strategy-opportunity-advisory")
    api_main._STRATEGY_OPPORTUNITY_ADVISORY_CACHE.clear()

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json() == first.json()


def test_strategy_opportunity_advisory_keeps_older_strategy_rows_when_entry_quality_is_newer(
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
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "decision": "PAPER_READY",
                    "recommended_mode": "paper",
                    "horizon_hours": 24,
                    "sample_count": 40,
                    "complete_sample_count": 30,
                    "avg_net_bps": 20.0,
                    "p25_net_bps": -10.0,
                    "win_rate": 0.6,
                    "cost_source_mix": '{"public_spread_proxy":40}',
                    "max_paper_notional_usdt": 500.0,
                    "max_live_notional_usdt": 0.0,
                },
                {
                    "as_of_ts": datetime(2026, 5, 21, tzinfo=UTC),
                    "strategy_candidate": "v5.entry_quality_missed_low_audit",
                    "symbol": "ALL",
                    "decision": "RESEARCH_ONLY",
                    "recommended_mode": "research",
                    "horizon_hours": None,
                    "sample_count": 7,
                    "complete_sample_count": 7,
                    "avg_net_bps": -50.0,
                    "p25_net_bps": None,
                    "win_rate": None,
                    "cost_source_mix": '{"entry_quality_research":7}',
                    "would_block_if_enabled": False,
                    "would_enter": False,
                    "no_sample_reason": "audit_only",
                    "max_paper_notional_usdt": 0.0,
                    "max_live_notional_usdt": 0.0,
                },
            ]
        ),
        lake / "gold" / "strategy_opportunity_advisory",
    )

    rows = TestClient(app).get("/v1/strategy-opportunity-advisory").json()

    candidates = {row["strategy_candidate"] for row in rows}
    assert "v5.f4_volume_expansion_entry" in candidates
    assert "v5.entry_quality_missed_low_audit" in candidates


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
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
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


def test_strategy_opportunity_advisory_git_commit_lookup_is_cached(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    rows = []
    for index in range(25):
        rows.append(
            {
                "as_of_ts": datetime(2026, 5, 20, tzinfo=UTC),
                "strategy_candidate": f"v5.fast_advisory_{index}",
                "symbol": "SOL-USDT",
                "decision": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "horizon_hours": 24,
                "sample_count": 10,
                "cost_source_mix": '{"public_spread_proxy":10}',
                "max_live_notional_usdt": 0.0,
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "strategy_opportunity_advisory")

    calls = 0

    class Result:
        stdout = "abc123\n"

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return Result()

    api_main._git_commit.cache_clear()
    monkeypatch.setattr(api_main.subprocess, "run", fake_run)

    response = TestClient(app).get("/v1/strategy-opportunity-advisory")

    api_main._git_commit.cache_clear()
    assert response.status_code == 200
    assert len(response.json()) == 25
    assert calls == 1


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
    assert dashed.json()[0]["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
    assert dashed.json()[0]["source_version"]
    assert dashed.json()[0]["would_enter"] is False
    assert dashed.json()[0]["would_block_if_enabled"] is False
    assert dashed.json()[0]["expires_at"]


def _api_advisory_row(
    strategy_candidate: str,
    symbol: str,
    as_of: datetime,
) -> dict:
    return {
        "as_of_ts": as_of,
        "generated_at": as_of,
        "expires_at": as_of,
        "strategy_candidate": strategy_candidate,
        "symbol": symbol,
        "decision": "PAPER_READY",
        "recommended_mode": "paper",
        "horizon_hours": 24,
        "sample_count": 72,
        "complete_sample_count": 40,
        "avg_net_bps": 25.0,
        "p25_net_bps": -15.0,
        "win_rate": 0.61,
        "cost_source_mix": '{"mixed_actual_proxy":72}',
        "live_block_reasons": '["paper_candidate"]',
        "max_paper_notional_usdt": 1000.0,
        "max_live_notional_usdt": 999.0,
    }


def _portfolio_status_row(
    *,
    research_id: str,
    strategy_candidate: str,
    status: str,
    as_of_date: str = "2026-05-24",
) -> dict:
    return {
        "schema_version": "research_portfolio_status.v0.1",
        "as_of_date": as_of_date,
        "research_id": research_id,
        "module": "test",
        "strategy_candidate": strategy_candidate,
        "status": status,
        "action": status,
        "reason": "test_portfolio_overlay",
        "created_at": datetime.fromisoformat(f"{as_of_date}T12:00:00+00:00"),
        "source": "test",
    }
