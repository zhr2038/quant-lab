import hashlib
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.research.candidate_labels import (
    build_and_publish_candidate_labels,
    build_candidate_quality,
)
from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
from tests.v5_bundle_fixture import make_v5_bundle_fixture


def test_candidate_snapshot_ingest_builds_events_labels_quality_and_summary(tmp_path):
    lake = tmp_path / "lake"
    _write_btc_bars(lake)
    bundle = make_v5_bundle_fixture(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz"
    )

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", tmp_path / "redacted")

    events = read_parquet_dataset(lake / "silver/v5_candidate_event")
    labels = read_parquet_dataset(lake / "gold/v5_candidate_label")
    quality = read_parquet_dataset(lake / "gold/v5_candidate_quality_daily")
    summary = read_parquet_dataset(lake / "gold/v5_candidate_outcome_summary")

    expected_id = "cand_" + hashlib.sha256(
        b"run_001|2026-05-10T01:00:00Z|BTC-USDT|v5.btc_leadership_probe_strict|0"
    ).hexdigest()[:24]

    assert result.silver_rows["v5_candidate_event"] == 1
    assert events.height == 1
    assert events["candidate_id"][0] == expected_id
    assert events["symbol"][0] == "BTC-USDT"
    assert events["strategy_candidate"][0] == "v5.btc_leadership_probe_strict"

    assert labels.height == 7
    assert set(labels["horizon_hours"].to_list()) == {4, 8, 12, 24, 48, 72, 120}
    assert set(labels["label_status"].to_list()) == {"complete"}
    h4 = labels.filter(labels["horizon_hours"] == 4).to_dicts()[0]
    assert h4["candidate_id"] == expected_id
    assert h4["block_reason"] == "risk_gate"
    assert h4["strategy_candidate"] == "v5.btc_leadership_probe_strict"
    assert h4["regime_state"] == "trend"
    assert h4["risk_level"] == "LOW"
    assert h4["btc_trend_state"] == "uptrend"
    assert h4["broad_market_positive_count"] == 3
    assert h4["funding_state"] == "neutral"
    assert h4["volatility_bucket"] == "medium"
    assert h4["win"] is True
    assert h4["gross_bps"] == pytest.approx(((106.0 / 102.0) - 1.0) * 10_000.0)
    assert h4["net_bps_after_cost"] == pytest.approx(h4["gross_bps"] - 3.5)
    assert h4["mfe_bps"] > 0
    assert h4["mae_bps"] < 0

    assert quality.height == 1
    q = quality.to_dicts()[0]
    assert q["candidate_event_rows"] == 1
    assert q["run_symbol_min_rows"] == 1
    assert q["feature_completeness"] == pytest.approx(1.0)
    assert q["label_completeness"] == pytest.approx(1.0)
    assert q["cost_source_coverage"] == pytest.approx(1.0)
    assert "quant_lab" in q["cost_source_quality_counts"]

    assert summary.height == 7
    s4 = summary.filter(summary["horizon_hours"] == 4).to_dicts()[0]
    assert s4["block_reason"] == "risk_gate"
    assert s4["strategy_candidate"] == "v5.btc_leadership_probe_strict"
    assert s4["symbol"] == "BTC-USDT"
    assert s4["complete_sample_count"] == 1


def test_candidate_label_incremental_mode_skips_old_raw_events(tmp_path):
    lake = tmp_path / "lake"
    _write_btc_bars(lake)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _candidate_event("old", datetime(2026, 5, 1, 1, tzinfo=UTC)),
                _candidate_event("recent", datetime(2026, 5, 10, 1, tzinfo=UTC)),
            ]
        ),
        lake / "silver" / "v5_candidate_event",
    )

    result = build_and_publish_candidate_labels(
        lake,
        as_of_date="2026-05-10",
        mode="incremental",
        lookback_days=2,
    )
    labels = read_parquet_dataset(lake / "gold" / "v5_candidate_label")

    assert result.mode == "incremental"
    assert result.candidate_event_rows == 1
    assert set(labels["candidate_id"].to_list()) == {"recent"}


def test_candidate_quality_ignores_pending_future_labels_and_pre_snapshot_runs():
    created_at = datetime(2026, 5, 20, tzinfo=UTC)
    events = pl.DataFrame(
        [
            _candidate_event("recent", datetime(2026, 5, 20, 1, tzinfo=UTC))
            | {"run_id": "20260520_01"}
        ]
    )
    labels = pl.DataFrame(
        [
            {
                "candidate_id": "recent",
                "run_id": "20260520_01",
                "label_status": "complete",
                "label_reason": "ok",
            },
            {
                "candidate_id": "recent",
                "run_id": "20260520_01",
                "label_status": "partial",
                "label_reason": "future_bar_unavailable",
            },
        ]
    )
    run_summary = pl.DataFrame(
        [
            {
                "run_id": "20260519_23",
                "source_path_inside_bundle": "raw/recent_runs/old/summary.json",
            },
            {
                "run_id": "20260520_01",
                "source_path_inside_bundle": "raw/recent_runs/current/summary.json",
            },
        ]
    )

    quality = build_candidate_quality(
        events,
        labels,
        run_summary,
        as_of_date=created_at.date(),
        created_at=created_at,
    )
    row = quality.to_dicts()[0]

    assert row["runs_without_candidate_event"] == 0
    assert row["future_pending_label_rows"] == 1
    assert row["eligible_label_rows"] == 1
    assert row["label_completeness"] == pytest.approx(1.0)
    assert row["raw_label_completeness"] == pytest.approx(1 / 7)
    assert "candidate_label_completeness_below_80pct" not in row["warnings_json"]
    assert "runs_without_candidate_event" not in row["warnings_json"]


def test_candidate_quality_excludes_no_signal_context_from_feature_completeness():
    created_at = datetime(2026, 5, 20, tzinfo=UTC)
    complete_event = _candidate_event("complete", created_at) | {
        "eligible_before_filters": "true",
        "final_score": 0.8,
        "rank": 1,
        "f1_mom_5d": 0.1,
        "f2_mom_20d": 0.2,
        "f3_vol_adj_ret": 0.3,
        "f4_volume_expansion": 0.4,
        "f5_rsi_trend_confirm": 0.5,
        "alpha6_score": 0.6,
        "ml_score": 0.7,
        "mean_reversion_score": 0.0,
        "expected_edge_bps": 50.0,
        "required_edge_bps": 35.0,
    }
    no_signal_context = _candidate_event("no_signal", created_at) | {
        "eligible_before_filters": "false",
        "final_decision": "no_order",
        "block_reason": "",
        "final_score": None,
        "f1_mom_5d": None,
        "f2_mom_20d": None,
        "f3_vol_adj_ret": None,
        "f4_volume_expansion": None,
        "f5_rsi_trend_confirm": None,
        "alpha6_score": None,
        "ml_score": None,
        "mean_reversion_score": None,
        "expected_edge_bps": 0.0,
        "required_edge_bps": 35.0,
    }

    quality = build_candidate_quality(
        pl.DataFrame([complete_event, no_signal_context]),
        pl.DataFrame(),
        pl.DataFrame(),
        as_of_date=created_at.date(),
        created_at=created_at,
    )
    row = quality.to_dicts()[0]

    assert row["candidate_event_rows"] == 2
    assert row["feature_denominator_rows"] == 1
    assert row["no_signal_context_rows"] == 1
    assert row["feature_completeness"] == pytest.approx(1.0)
    assert row["required_feature_completeness"] == pytest.approx(1.0)
    assert "candidate_feature_completeness_below_80pct" not in row["warnings_json"]


def test_candidate_quality_warns_only_when_required_features_are_missing():
    created_at = datetime(2026, 5, 20, tzinfo=UTC)
    score_only_complete = _candidate_event("score_only_complete", created_at) | {
        "eligible_before_filters": "true",
        "final_score": 0.8,
        "f1_mom_5d": None,
        "f2_mom_20d": None,
        "f3_vol_adj_ret": None,
        "f4_volume_expansion": None,
        "f5_rsi_trend_confirm": None,
        "alpha6_score": None,
        "ml_score": 0.8,
        "mean_reversion_score": None,
        "expected_edge_bps": 50.0,
        "required_edge_bps": 35.0,
    }
    missing_required = _candidate_event("missing_required", created_at) | {
        "eligible_before_filters": "true",
        "final_score": None,
        "expected_edge_bps": None,
        "required_edge_bps": None,
    }

    score_only_quality = build_candidate_quality(
        pl.DataFrame([score_only_complete]),
        pl.DataFrame(),
        pl.DataFrame(),
        as_of_date=created_at.date(),
        created_at=created_at,
    ).to_dicts()[0]
    missing_required_quality = build_candidate_quality(
        pl.DataFrame([missing_required]),
        pl.DataFrame(),
        pl.DataFrame(),
        as_of_date=created_at.date(),
        created_at=created_at,
    ).to_dicts()[0]

    assert score_only_quality["feature_completeness"] < 0.8
    assert score_only_quality["required_feature_completeness"] == pytest.approx(1.0)
    assert "candidate_required_feature_completeness_below_80pct" not in score_only_quality[
        "warnings_json"
    ]
    assert missing_required_quality["required_feature_completeness"] == pytest.approx(0.0)
    assert "candidate_required_feature_completeness_below_80pct" in missing_required_quality[
        "warnings_json"
    ]


def _candidate_event(candidate_id: str, ts: datetime) -> dict[str, object]:
    return {
        "strategy": "v5",
        "candidate_id": candidate_id,
        "run_id": "run_001",
        "ts_utc": ts,
        "symbol": "BTC-USDT",
        "strategy_candidate": "v5.btc_leadership_probe_strict",
        "block_reason": "risk_gate",
        "final_decision": "blocked",
        "cost_bps": 3.5,
        "cost_source": "mixed_actual_proxy",
        "regime_state": "TRENDING",
        "risk_level": "normal",
        "btc_trend_state": "uptrend",
        "broad_market_positive_count": 3,
        "funding_state": "neutral",
        "volatility_bucket": "medium",
        "bundle_ts": ts,
        "ingest_ts": ts,
    }


def _write_btc_bars(lake) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = []
    for index in range(130):
        close = 100.0 + index
        rows.append(
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=index),
                "open": close - 0.5,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 100.0,
                "quote_volume": close * 100.0,
                "source": "test",
                "ingest_ts": start + timedelta(hours=index, minutes=1),
                "is_closed": True,
            }
        )
    write_market_bars(lake, rows)
