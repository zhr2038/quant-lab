import json
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.factors.composite_factory import (
    build_factor_factory_v2_reports,
    build_factor_strategy_bridge_candidates,
)
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


def test_factor_factory_v2_dedupes_and_builds_review_outputs():
    candidates = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-10",
                "factor_id": "core.close_return_24",
                "factor_family": "momentum",
                "best_horizon_bars": 24,
                "best_score": 12.0,
                "best_rank_ic_mean": 0.05,
                "best_rank_ic_tstat": 3.2,
                "best_long_short_mean_bps": 18.0,
                "candidate_state": "PAPER_READY",
            },
            {
                "as_of_date": "2026-06-10",
                "factor_id": "auto.single.close_return_24",
                "factor_family": "momentum",
                "best_horizon_bars": 24,
                "best_score": 10.0,
                "best_rank_ic_mean": 0.04,
                "best_rank_ic_tstat": 2.4,
                "best_long_short_mean_bps": 16.0,
                "candidate_state": "PAPER_READY",
            },
            {
                "as_of_date": "2026-06-10",
                "factor_id": "core.volume_zscore_24",
                "factor_family": "volume",
                "best_horizon_bars": 8,
                "best_score": 3.0,
                "best_rank_ic_mean": 0.01,
                "best_rank_ic_tstat": 0.7,
                "best_long_short_mean_bps": 4.0,
                "candidate_state": "KEEP_SHADOW",
            },
        ]
    )
    evidence = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-10",
                "factor_id": "core.close_return_24",
                "horizon_bars": 24,
                "valid_sample_count": 180,
                "rank_ic_mean": 0.05,
                "rank_ic_tstat": 3.2,
                "long_short_mean_bps": 18.0,
                "win_rate": 0.62,
                "score": 12.0,
                "regime_state": "TREND_UP",
            },
            {
                "as_of_date": "2026-06-10",
                "factor_id": "core.close_return_24",
                "horizon_bars": 24,
                "valid_sample_count": 90,
                "rank_ic_mean": -0.01,
                "rank_ic_tstat": -0.3,
                "long_short_mean_bps": -2.0,
                "win_rate": 0.48,
                "score": -1.0,
                "regime_state": "RISK_OFF",
            },
        ]
    )
    correlations = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-10",
                "factor_id_left": "core.close_return_24",
                "factor_id_right": "auto.single.close_return_24",
                "sample_count": 240,
                "correlation": 0.94,
            }
        ]
    )

    reports = build_factor_factory_v2_reports(
        candidates=candidates,
        evidence=evidence,
        correlations=correlations,
    )

    dedupe = reports["factor_dedupe_decision"].to_dicts()
    dedupe_by_factor = {row["factor_id"]: row for row in dedupe}
    assert dedupe_by_factor["core.close_return_24"]["dedupe_decision"] == "keep_leader"
    assert (
        dedupe_by_factor["auto.single.close_return_24"]["dedupe_decision"]
        == "redundant_suppressed"
    )
    queue = reports["factor_paper_review_queue"].to_dicts()
    assert any(row["recommendation"] == "FACTOR_PAPER_REVIEW" for row in queue)
    assert any(row["recommendation"] == "HOLD_REVIEW_REDUNDANT" for row in queue)
    assert reports["composite_factor_candidates"].height >= 1
    assert reports["factor_regime_effectiveness"].height >= 1
    assert reports["factor_strategy_bridge_candidates"].height >= 1
    assert (
        reports["factor_strategy_bridge_candidates"]["live_order_effect"][0]
        == "none_read_only_research"
    )


def test_factor_bridge_routes_forward_pass_without_regime_stability_block():
    paper_queue = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-13",
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "factor_family": "mean_reversion",
                "candidate_state": "PAPER_READY",
                "best_horizon_bars": 4,
                "best_rank_ic_mean": 0.04,
                "best_rank_ic_tstat": 2.1,
                "best_long_short_mean_bps": 12.5,
                "sample_count": 168,
                "oos_score": 1.4,
                "regime_stability_score": None,
                "correlation_cluster_id": "cluster_001",
                "recommendation": "FACTOR_PAPER_REVIEW",
                "live_order_effect": "none_read_only_research",
            }
        ]
    )
    forward_validation = pl.DataFrame(
        [
            {
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "symbol": "SOL-USDT",
                "regime": "TREND_UP",
                "horizon": "4h",
                "recommendation": "FORWARD_VALIDATION_PASS",
            }
        ]
    )

    bridge = build_factor_strategy_bridge_candidates(
        paper_queue=paper_queue,
        factor_forward_validation=forward_validation,
    )
    row = bridge.to_dicts()[0]
    reasons = json.loads(row["blocking_reasons"])

    assert row["eligible_for_alpha_factory"] == "strategy_review_pending"
    assert row["recommended_action"] == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"
    assert row["symbol"] == "SOL-USDT"
    assert row["regime"] == "TREND_UP"
    assert row["horizon"] == "4h"
    assert "needs_strategy_formulation" in reasons
    assert "needs_paper_tracking" in reasons
    assert "needs_cost_validation" in reasons
    assert "forward_validation_not_passed" not in reasons
    assert "regime_stability_not_positive_or_missing" not in reasons
    assert row["live_order_effect"] == "none_read_only_research"


def test_factor_bridge_adds_review_row_for_forward_pass_outside_paper_queue():
    forward_validation = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-14",
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "factor_family": "risk_adjusted_reversal",
                "candidate_state": "KEEP_SHADOW",
                "symbol": "SOL-USDT",
                "regime": "TREND_UP",
                "horizon_hours": 8,
                "sample_count": 115,
                "rank_ic": 0.296236,
                "cost_adjusted_score": 114.120904,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "live_order_effect": "none_read_only_research",
            }
        ]
    )

    bridge = build_factor_strategy_bridge_candidates(
        paper_queue=pl.DataFrame(),
        factor_forward_validation=forward_validation,
    )
    row = bridge.to_dicts()[0]
    reasons = json.loads(row["blocking_reasons"])

    assert row["factor_id"] == "core.mean_reversion_vol_adjusted_4"
    assert row["symbol"] == "SOL-USDT"
    assert row["regime"] == "TREND_UP"
    assert row["horizon"] == "8h"
    assert row["horizon_hours"] == "8"
    assert row["eligible_for_alpha_factory"] == "strategy_review_pending"
    assert row["recommended_action"] == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"
    assert "not_in_factor_paper_review_queue" in reasons
    assert "needs_strategy_formulation" in reasons
    assert "needs_paper_tracking" in reasons
    assert "needs_cost_validation" in reasons
    assert "forward_validation_not_passed" not in reasons
    assert row["live_order_effect"] == "none_read_only_research"


def test_factor_bridge_aggregates_forward_pass_context_for_strategy_review():
    forward_validation = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-15",
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "factor_family": "risk_adjusted_reversal",
                "candidate_state": "PAPER_READY",
                "symbol": "SOL-USDT",
                "regime": "TREND_UP",
                "horizon_hours": 4,
                "sample_count": 126,
                "rank_ic": 0.10079,
                "cost_adjusted_score": 15.358829,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "live_order_effect": "none_read_only_research",
            },
            {
                "as_of_date": "2026-06-15",
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "factor_family": "risk_adjusted_reversal",
                "candidate_state": "PAPER_READY",
                "symbol": "SOL-USDT",
                "regime": "TREND_UP",
                "horizon_hours": 8,
                "sample_count": 122,
                "rank_ic": 0.276514,
                "cost_adjusted_score": 108.972432,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "live_order_effect": "none_read_only_research",
            },
        ]
    )

    bridge = build_factor_strategy_bridge_candidates(
        paper_queue=pl.DataFrame(),
        factor_forward_validation=forward_validation,
    )
    row = bridge.to_dicts()[0]

    assert row["factor_id"] == "core.mean_reversion_vol_adjusted_4"
    assert row["symbol"] == "SOL-USDT"
    assert row["regime"] == "TREND_UP"
    assert row["horizon"] == "4h-8h"
    assert row["horizon_hours"] == "4,8"
    assert row["forward_sample_count"] == 122
    assert row["forward_cost_adjusted_score"] == 108.972432
    assert row["recommended_action"] == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"


def test_factor_bridge_adds_fast_microstructure_pass_features_to_strategy_review():
    fast_forward = pl.DataFrame(
        [
            {
                "generated_at": "2026-06-16T00:00:00Z",
                "feature_name": "orderbook_imbalance_1m",
                "symbol": "BNB-USDT",
                "regime": "ALL_REGIMES",
                "horizon_hours": 8,
                "sample_count": 144,
                "rank_ic": 0.21,
                "long_short_bps": 32.5,
                "p25_net_bps": -12.0,
                "hit_rate": 0.58,
                "recommendation": "FORWARD_VALIDATION_PASS",
                "live_order_effect": "read_only_no_live_order",
            }
        ]
    )

    bridge = build_factor_strategy_bridge_candidates(
        paper_queue=pl.DataFrame(),
        factor_forward_validation=pl.DataFrame(),
        fast_microstructure_forward_test=fast_forward,
    )
    row = bridge.to_dicts()[0]
    reasons = json.loads(row["blocking_reasons"])

    assert row["factor_id"] == "fast_microstructure.orderbook_imbalance_1m"
    assert row["factor_family"] == "fast_microstructure"
    assert row["symbol"] == "BNB-USDT"
    assert row["horizon"] == "8h"
    assert row["eligible_for_alpha_factory"] == "strategy_review_pending"
    assert row["recommended_action"] == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"
    assert reasons == [
        "needs_strategy_formulation",
        "needs_paper_tracking",
        "needs_cost_validation",
    ]


def test_factor_bridge_prioritizes_strategy_review_rows():
    paper_queue = pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-15",
                "factor_id": "core.display_only",
                "factor_family": "momentum",
                "candidate_state": "PAPER_READY",
                "best_long_short_mean_bps": 12.0,
                "sample_count": 150,
                "oos_score": 1.0,
                "regime_stability_score": 1.0,
                "correlation_cluster_id": "cluster_001",
                "recommendation": "FACTOR_PAPER_REVIEW",
                "live_order_effect": "none_read_only_research",
            },
            {
                "as_of_date": "2026-06-15",
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "factor_family": "risk_adjusted_reversal",
                "candidate_state": "PAPER_READY",
                "best_long_short_mean_bps": 12.0,
                "sample_count": 150,
                "oos_score": 1.0,
                "regime_stability_score": 1.0,
                "correlation_cluster_id": "cluster_002",
                "recommendation": "FACTOR_PAPER_REVIEW",
                "live_order_effect": "none_read_only_research",
            },
        ]
    )
    forward_validation = pl.DataFrame(
        [
            {
                "factor_id": "core.mean_reversion_vol_adjusted_4",
                "symbol": "SOL-USDT",
                "regime": "TREND_UP",
                "horizon_hours": 8,
                "sample_count": 122,
                "rank_ic": 0.27,
                "cost_adjusted_score": 108.0,
                "recommendation": "FORWARD_VALIDATION_PASS",
            }
        ]
    )

    bridge = build_factor_strategy_bridge_candidates(
        paper_queue=paper_queue,
        factor_forward_validation=forward_validation,
    )

    assert bridge["recommended_action"][0] == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"
    assert bridge["factor_id"][0] == "core.mean_reversion_vol_adjusted_4"


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
