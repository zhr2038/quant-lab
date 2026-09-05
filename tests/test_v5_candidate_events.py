from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_lab.data.lake import read_parquet_dataset, write_market_bars
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
from tests.v5_bundle_fixture import make_tar


def test_ingest_candidate_snapshot_preserves_facts_without_restarting_research(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "summaries/candidate_snapshot.csv": (
                "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,current_position,"
                "current_weight,target_weight_raw,target_weight_after_risk,final_score,rank,"
                "f1_mom_5d,f2_mom_20d,f3_vol_adj_ret,f4_volume_expansion,"
                "f5_rsi_trend_confirm,alpha6_score,alpha6_side,ml_score,"
                "mean_reversion_score,expected_edge_bps,required_edge_bps,cost_bps,"
                "cost_source,eligible_before_filters,final_decision,block_reason,"
                "strategy_candidate\n"
                "cand_bnb_001,run_001,2026-05-10T01:00:00Z,BNB/USDT,Normal,LOW,0,"
                "0,0.2,0.15,0.91,1,0.1,0.2,0.3,0.4,0.5,0.8,buy,0.6,"
                "0.2,60,45,5,public_spread_proxy,true,OPEN_LONG,,Alpha6Factor\n"
            ),
            "summaries/window_summary.json": '{"run_count": 1}',
        },
    )
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 10, 1, tzinfo=UTC)
    write_market_bars(
        lake,
        [
            {
                "venue": "okx",
                "symbol": "BNB-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=idx),
                "open": 100.0 + idx * 0.1,
                "high": 100.2 + idx * 0.1,
                "low": 99.8 + idx * 0.1,
                "close": 100.0 + idx * 0.1,
                "volume": 10.0,
                "quote_volume": 1000.0,
                "source": "test_fixture",
                "ingest_ts": start + timedelta(hours=idx, minutes=1),
            }
            for idx in range(0, 123)
        ],
    )

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.validation.valid is True
    events = read_parquet_dataset(lake / "silver/v5_candidate_event")
    labels = read_parquet_dataset(lake / "gold/v5_candidate_label")
    assert labels.is_empty()
    assert not (lake / "gold/v5_candidate_quality_daily").exists()
    assert events.height == 1
    assert events["candidate_id"][0] == "cand_bnb_001"
    assert events["symbol"][0] == "BNB-USDT"
    assert events["cost_source"][0] == "public_spread_proxy"
    assert events["cost_bps"][0] == "5"


def test_ingest_candidate_snapshot_preserves_many_cost_sources_and_candidates(tmp_path):
    header = (
        "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,current_position,"
        "current_weight,target_weight_raw,target_weight_after_risk,final_score,rank,"
        "expected_edge_bps,required_edge_bps,cost_bps,selected_total_cost_bps,"
        "cost_source,cost_model_version,cost_gate_verified,would_block_by_cost,"
        "final_decision,block_reason,strategy_candidate\n"
    )
    candidates = [
        "f3_dominant_entry",
        "btc_leadership_alpha6_low_blocked",
        "portfolio_trend_following",
        "f4_volume_expansion_entry",
    ]
    lines = [header]
    start = datetime(2026, 5, 10, 1, tzinfo=UTC)
    for index in range(112):
        ts = (start + timedelta(minutes=index)).isoformat().replace("+00:00", "Z")
        symbol = ["BNB/USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"][index % 4]
        source = "local_estimate" if index % 2 == 0 else "cost_not_requested_no_order"
        candidate = candidates[index % len(candidates)]
        lines.append(
            f",run_112,{ts},{symbol},Normal,LOW,0,0,0.2,0.15,0.91,{index},"
            f"60,45,5,5.5,{source},cost-v1,true,false,KEEP_SHADOW,,{candidate}\n"
        )
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {
            "summaries/candidate_snapshot.csv": "".join(lines),
            "summaries/window_summary.json": '{"run_count": 1}',
        },
    )
    lake = tmp_path / "lake"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    events = read_parquet_dataset(lake / "silver/v5_candidate_event")
    assert result.silver_rows["v5_candidate_event"] == 112
    assert events.height == 112
    assert events["candidate_id"].n_unique() == 112
    assert set(events["cost_source"].drop_nulls().to_list()) == {
        "local_estimate",
        "cost_not_requested_no_order",
    }
    assert set(events["strategy_candidate"].drop_nulls().to_list()) == set(candidates)
    assert "selected_total_cost_bps" in events.columns
    assert "cost_model_version" in events.columns
    assert "cost_gate_verified" in events.columns
    assert "would_block_by_cost" in events.columns
