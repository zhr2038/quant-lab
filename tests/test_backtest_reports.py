from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.backtest.label_backtest import build_label_backtest_summary
from quant_lab.backtest.reports import build_backtest_report_bundle


def _bars(symbol: str, start: datetime, closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "symbol": symbol,
                "ts": start + timedelta(hours=index),
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": 100.0,
            }
            for index, close in enumerate(closes)
        ]
    )


def test_label_backtest_summarizes_bnb_alpha6_conflict() -> None:
    frame = pl.DataFrame(
        [
            {
                "strategy_id": "BNB_STRONG_ALPHA6_BYPASS_SHADOW_V1",
                "symbol": "BNB/USDT",
                "regime_state": "TREND_UP",
                "ts_utc": "2026-06-01T00:00:00Z",
                "future_4h_net_bps": 80.0,
                "future_8h_net_bps": 120.0,
                "cost_model": "conservative_p75:mixed_actual_proxy",
            },
            {
                "strategy_id": "BNB_STRONG_ALPHA6_BYPASS_SHADOW_V1",
                "symbol": "BNB/USDT",
                "regime_state": "TREND_UP",
                "ts_utc": "2026-06-01T01:00:00Z",
                "future_4h_net_bps": -20.0,
                "future_8h_net_bps": 60.0,
                "cost_model": "conservative_p75:mixed_actual_proxy",
            },
        ]
    )

    summary = build_label_backtest_summary({"bnb_strong_alpha6_bypass_shadow": frame})

    h4 = summary.filter(
        (pl.col("strategy_id") == "BNB_STRONG_ALPHA6_BYPASS_BACKTEST")
        & (pl.col("symbol") == "BNB-USDT")
        & (pl.col("horizon_hours") == 4)
    )
    assert h4.height == 1
    assert h4["sample_count"][0] == 2
    assert h4["complete_sample_count"][0] == 2
    assert h4["avg_net_bps"][0] == 30.0
    assert h4["data_leakage_check"][0] == "pass_visible_at_decision_time"


def test_label_backtest_keeps_advisory_pending_horizon_rows() -> None:
    summary = build_label_backtest_summary(
        {
            "strategy_opportunity_advisory": pl.DataFrame(
                [
                    {
                        "strategy_candidate": "v5.bottom_zone_probe_paper",
                        "symbol": "BNB-USDT",
                        "horizon_hours": 24,
                        "recommended_mode": "paper",
                        "generated_at": "2026-06-01T00:00:00Z",
                    }
                ]
            )
        }
    )

    row = summary.to_dicts()[0]
    assert row["strategy_id"] == "v5.bottom_zone_probe_paper"
    assert row["horizon_hours"] == 24
    assert row["sample_count"] == 1
    assert row["complete_sample_count"] == 0
    assert row["recommendation"] == "RESEARCH_ONLY_PENDING_LABELS"


def test_backtest_bundle_outputs_bottom_zone_and_promotion() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    market = _bars("BNB-USDT", start, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    bottom = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "ts_utc": start.isoformat().replace("+00:00", "Z"),
                "close": 100.0,
                "bottom_zone_state": "BOTTOM_PROBE_ALLOWED",
                "would_probe_paper": True,
                "support_low_24h": 99.0,
                "vwap_24h": 100.5,
                "avg_spread_bps_15m": 4.0,
            }
        ]
    )
    cost = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "total_cost_bps_p75": 10.0,
                "source": "mixed_actual_proxy",
            }
        ]
    )
    labels = pl.DataFrame(
        [
            {
                "strategy_id": "BOTTOM_ZONE_PROBE_BACKTEST",
                "symbol": "BNB-USDT",
                "ts_utc": start.isoformat().replace("+00:00", "Z"),
                "future_4h_net_bps": 90.0,
            }
        ]
    )

    bundle = build_backtest_report_bundle(
        {
            "market_bar": market,
            "bottom_zone_reversal_shadow": bottom,
            "cost_bucket_daily": cost,
            "bottom_zone_reversal_shadow_labels": labels,
        }
    )

    assert "BOTTOM_ZONE_PROBE_BACKTEST" in bundle.bottom_zone_backtest["strategy_id"].to_list()
    row = bundle.bottom_zone_backtest.to_dicts()[0]
    assert row["future_4h_net_bps"] == pytest.approx(390.0)
    assert row["live_order_effect"] == "read_only_no_live_order"
    assert "Read-only" in bundle.bottom_zone_summary_md
    assert "BOTTOM_ZONE_PROBE_BACKTEST" in bundle.regime_breakdown["strategy_id"].to_list()
    assert bundle.regime_breakdown["live_order_effect"].to_list()[0] == "read_only_no_live_order"
    assert bundle.promotion_decision.height >= 1 or bundle.label_summary.height == 0


def test_backtest_label_summary_uses_stable_first_batch_backtest_ids() -> None:
    summary = build_label_backtest_summary(
        {
            "final_score_vs_alpha6_conflict": pl.DataFrame(
                [
                    {
                        "symbol": "BNB-USDT",
                        "ts_utc": "2026-06-01T00:00:00Z",
                        "future_4h_net_bps": 70.0,
                    }
                ]
            ),
            "risk_on_multi_buy_shadow": pl.DataFrame(
                [
                    {
                        "symbol": "MULTI",
                        "current_regime": "ALT_IMPULSE",
                        "ts_utc": "2026-06-01T00:00:00Z",
                        "future_4h_net_bps": 40.0,
                    }
                ]
            ),
        }
    )

    strategy_ids = set(summary["strategy_id"].to_list())
    assert "FINAL_SCORE_ALPHA6_CONFLICT_BACKTEST" in strategy_ids
    assert "RISK_ON_MULTI_BUY_BACKTEST" in strategy_ids
