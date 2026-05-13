from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.contracts.models import AlphaResearchSpec
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.research.evidence import build_alpha_dataset, build_alpha_evidence
from quant_lab.research.publish import build_and_publish_alpha_evidence


def test_build_alpha_dataset_joins_features_to_delayed_labels(tmp_path):
    lake = tmp_path / "lake"
    _write_research_fixture(lake)
    spec = _spec(min_samples=10)

    result = build_alpha_dataset(lake, spec)

    assert result.frame.height > 0
    row = result.frame.sort(["symbol", "feature_ts"]).to_dicts()[0]
    assert row["decision_ts"] == row["feature_ts"] + timedelta(hours=1)
    assert row["label_ts"] == row["feature_ts"] + timedelta(hours=3)
    assert "alpha_score" in result.frame.columns


def test_build_alpha_evidence_computes_metrics_without_paper_live_ready(tmp_path):
    lake = tmp_path / "lake"
    _write_research_fixture(lake)

    result = build_alpha_evidence(lake, _spec(min_samples=10))

    assert result.evidence is not None
    assert result.evidence.coverage == pytest.approx(1.0)
    assert result.evidence.ic_mean > 0
    assert result.evidence.paper_days == 0
    assert result.evidence.paper_slippage_coverage == 0.0
    assert result.evidence.evidence_status == "ok"


def test_insufficient_samples_keeps_metrics_and_marks_status(tmp_path):
    lake = tmp_path / "lake"
    _write_research_fixture(lake)

    result = build_alpha_evidence(lake, _spec(min_samples=10_000))

    assert result.evidence is not None
    assert result.status == "insufficient_samples"
    assert result.evidence.evidence_status == "insufficient_samples"
    assert result.evidence.ic_mean > 0


def test_build_alpha_evidence_missing_feature_value_does_not_crash(tmp_path):
    lake = tmp_path / "lake"
    _write_market_fixture(lake)

    result = build_alpha_evidence(lake, _spec(min_samples=10))

    assert result.evidence is None
    assert result.status == "insufficient_data"
    assert "feature_value missing or no rows matched alpha spec" in result.warnings


def test_publish_alpha_evidence_writes_evidence_and_gate(tmp_path):
    lake = tmp_path / "lake"
    _write_research_fixture(lake)

    result = build_and_publish_alpha_evidence(lake, _spec(min_samples=10))

    evidence = read_parquet_dataset(lake / "gold" / "alpha_evidence")
    gates = read_parquet_dataset(lake / "gold" / "gate_decision")
    assert result.alpha_evidence_rows == 1
    assert evidence.height == 1
    assert gates.height == 1
    assert gates["alpha_id"][0] == "v5.core.momentum"


def _spec(min_samples: int) -> AlphaResearchSpec:
    return AlphaResearchSpec(
        alpha_id="v5.core.momentum",
        version="v0.1",
        feature_set="core",
        feature_version="v0.1",
        feature_names=["close_return_24"],
        timeframe="1H",
        label_horizon_bars=2,
        decision_delay_bars=1,
        universe_id="okx-major-spot",
        min_samples=min_samples,
    )


def _write_research_fixture(lake) -> None:
    _write_market_fixture(lake)
    _write_feature_fixture(lake)
    _write_cost_fixture(lake)


def _write_market_fixture(lake) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rates = {
        "BTC-USDT": 0.001,
        "ETH-USDT": 0.002,
        "SOL-USDT": 0.003,
        "BNB-USDT": 0.004,
    }
    rows = []
    for symbol, rate in rates.items():
        for index in range(36):
            close = 100.0 * ((1.0 + rate) ** index)
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": start + timedelta(hours=index),
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 10.0,
                    "quote_volume": close * 10.0,
                    "source": "test",
                    "ingest_ts": start + timedelta(hours=index, minutes=1),
                }
            )
    write_market_bars(lake, rows)


def _write_feature_fixture(lake) -> None:
    start = datetime(2026, 5, 10, tzinfo=UTC)
    ranks = {"BTC-USDT": 1.0, "ETH-USDT": 2.0, "SOL-USDT": 3.0, "BNB-USDT": 4.0}
    rows = []
    for symbol, value in ranks.items():
        for index in range(30):
            ts = start + timedelta(hours=index)
            rows.append(
                {
                    "feature_set": "core",
                    "feature_name": "close_return_24",
                    "feature_version": "v0.1",
                    "symbol": symbol,
                    "timeframe": "1H",
                    "ts": ts,
                    "value": value,
                    "lookback_bars": 24,
                    "input_dataset_version": "market_bar:test",
                    "input_hash": "sha256:test",
                    "code_version": "test",
                    "created_at": ts + timedelta(minutes=2),
                    "source": "test",
                    "is_valid": True,
                    "invalid_reason": None,
                }
            )
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "feature_value")


def _write_cost_fixture(lake) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": symbol,
                    "regime": "public_proxy",
                    "event_type": "spread_proxy",
                    "notional_bucket": "all",
                    "sample_count": 30,
                    "fee_bps_p50": 0.0,
                    "fee_bps_p75": 0.0,
                    "fee_bps_p90": 0.0,
                    "slippage_bps_p50": 0.0,
                    "slippage_bps_p75": 0.0,
                    "slippage_bps_p90": 0.0,
                    "spread_bps_p50": 1.0,
                    "spread_bps_p75": 1.0,
                    "spread_bps_p90": 1.0,
                    "total_cost_bps_p50": 1.0,
                    "total_cost_bps_p75": 1.0,
                    "total_cost_bps_p90": 1.0,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "source": "public_spread_proxy",
                    "cost_model_version": "costs-test",
                    "created_at": "2026-05-10T00:00:00Z",
                }
                for symbol in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )
