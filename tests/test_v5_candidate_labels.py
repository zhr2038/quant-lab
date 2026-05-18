import hashlib
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.research.candidate_labels import build_and_publish_candidate_labels
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
