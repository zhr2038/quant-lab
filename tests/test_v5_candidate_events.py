from __future__ import annotations

from datetime import UTC, datetime, timedelta

from quant_lab.data.lake import read_parquet_dataset, write_market_bars
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
from tests.v5_bundle_fixture import make_tar


def test_ingest_candidate_snapshot_writes_silver_events_and_gold_labels(tmp_path):
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
            for idx in range(0, 121)
        ],
    )

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    assert result.validation.valid is True
    events = read_parquet_dataset(lake / "silver/v5_candidate_event")
    labels = read_parquet_dataset(lake / "gold/v5_candidate_label")
    quality = read_parquet_dataset(lake / "gold/v5_candidate_quality_daily")
    assert events.height == 1
    assert events["candidate_id"][0] == "cand_bnb_001"
    assert events["symbol"][0] == "BNB-USDT"
    assert set(labels["horizon_hours"].to_list()) == {4, 8, 12, 24, 48, 72, 120}
    assert labels.filter(labels["horizon_hours"] == 4)["gross_bps"][0] > 0
    assert labels.filter(labels["horizon_hours"] == 4)["net_bps_after_cost"][0] > 0
    assert quality["candidate_event_rows"][0] == 1
    assert quality["feature_completeness"][0] == 1.0
    assert quality["label_completeness"][0] == 1.0
    assert quality["cost_source_coverage"][0] == 1.0
