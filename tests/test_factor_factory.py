from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.factors.factory import build_and_publish_factor_factory, factor_factory_health
from quant_lab.features.publish import publish_features


def test_factor_factory_builds_definitions_values_evidence_and_candidates(tmp_path):
    lake = tmp_path / "lake"
    _write_bars(lake, symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"], count=180)
    _write_costs(lake)

    publish_features(lake, symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"])
    result = build_and_publish_factor_factory(
        lake,
        as_of_date="2026-05-20",
        feature_set="core",
        feature_version="v0.1",
        factor_version="v0.1",
        timeframe="1H",
        horizon_bars=(4, 8),
        decision_delay_bars=1,
        min_samples=20,
        top_quantile=0.25,
        dry_run=False,
    )

    definitions = read_parquet_dataset(lake / "gold" / "factor_definition")
    values = read_parquet_dataset(lake / "gold" / "factor_value")
    evidence = read_parquet_dataset(lake / "gold" / "factor_evidence")
    candidates = read_parquet_dataset(lake / "gold" / "factor_candidate")
    correlations = read_parquet_dataset(lake / "gold" / "factor_correlation_daily")

    assert result.factor_count > 0
    assert result.live_order_effect == "none_read_only_research"
    assert definitions.height > 0
    assert values.height > 0
    assert evidence.height > 0
    assert candidates.height > 0
    assert correlations.height >= 0
    assert "rank_ic_mean" in evidence.columns
    assert "candidate_state" in candidates.columns


def test_factor_factory_health_reports_missing_datasets(tmp_path):
    health = factor_factory_health(tmp_path / "lake")

    assert health.definition_rows == 0
    assert health.value_rows == 0
    assert health.evidence_rows == 0
    assert health.candidate_rows == 0
    assert health.live_order_effect == "none_read_only_research"
    assert "factor_definition_missing_or_empty" in health.warnings


def test_factor_values_are_cross_sectionally_normalized(tmp_path):
    lake = tmp_path / "lake"
    _write_bars(lake, symbols=["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"], count=120)
    _write_costs(lake)

    publish_features(lake)
    build_and_publish_factor_factory(
        lake,
        as_of_date="2026-05-20",
        horizon_bars=(4,),
        min_samples=20,
    )
    values = read_parquet_dataset(lake / "gold" / "factor_value")
    target_ts = values["ts"].drop_nulls().max()
    slice_df = values.filter(
        (pl.col("factor_id") == "core.close_return_24") & (pl.col("ts") == target_ts)
    )

    assert slice_df.height == 4
    assert slice_df.filter(pl.col("normalized_value").is_not_null()).height >= 3
    assert slice_df.filter(pl.col("rank_value").is_not_null()).height >= 3


def _write_bars(
    lake,
    *,
    symbols: list[str],
    count: int,
    start: datetime = datetime(2026, 5, 10, tzinfo=UTC),
) -> None:
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        drift = 0.05 + symbol_index * 0.02
        for index in range(count):
            close = 100.0 + index * drift + symbol_index * 10.0
            wave = (index % 7) * 0.1
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": start + timedelta(hours=index),
                    "open": close - 0.1,
                    "high": close + 1.0 + wave,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 10.0 + index + symbol_index * 3.0,
                    "quote_volume": close * (10.0 + index + symbol_index * 3.0),
                    "source": "test",
                    "ingest_ts": start + timedelta(hours=index, minutes=1),
                    "is_closed": True,
                }
            )
    write_market_bars(lake, rows)


def _write_costs(lake) -> None:
    rows = []
    for symbol in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]:
        rows.append(
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
                "cost_source": "public_spread_proxy",
                "cost_model_version": "costs-test",
                "created_at": datetime(2026, 5, 10, tzinfo=UTC),
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "cost_bucket_daily")
