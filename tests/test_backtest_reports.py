from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.backtest.label_backtest import build_label_backtest_summary
from quant_lab.backtest.reports import (
    build_backtest_report_bundle,
    build_backtest_vs_paper_consistency,
    build_research_promotion_decision,
)


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


def test_label_backtest_dedupes_duplicate_bnb_bypass_bundle_rows() -> None:
    frame = pl.DataFrame(
        [
            {
                "run_id": "20260607_01",
                "strategy_id": "BNB_STRONG_ALPHA6_BYPASS_SHADOW_V1",
                "symbol": "BNB/USDT",
                "ts_utc": "2026-06-07T01:00:00Z",
                "generated_at": "2026-06-07T02:00:00Z",
                "future_4h_net_bps": 100.0,
            },
            {
                "run_id": "20260607_01",
                "strategy_id": "BNB_STRONG_ALPHA6_BYPASS_SHADOW_V1",
                "symbol": "BNB/USDT",
                "ts_utc": "2026-06-07T01:00:00Z",
                "generated_at": "2026-06-07T03:00:00Z",
                "future_4h_net_bps": 120.0,
            },
        ]
    )

    summary = build_label_backtest_summary({"bnb_strong_alpha6_bypass_shadow": frame})

    row = summary.filter(pl.col("strategy_id") == "BNB_STRONG_ALPHA6_BYPASS_BACKTEST").to_dicts()[0]
    assert row["sample_count"] == 1
    assert row["complete_sample_count"] == 1
    assert row["avg_net_bps"] == 120.0
    assert row["dedupe_before_rows"] == 2
    assert row["dedupe_after_rows"] == 1
    assert row["duplicate_rate"] == 0.5


def test_label_backtest_dedupes_duplicate_final_score_conflict_rows() -> None:
    frame = pl.DataFrame(
        [
            {
                "run_id": "20260607_01",
                "symbol": "BNB/USDT",
                "ts_utc": "2026-06-07T01:00:00Z",
                "generated_at": "2026-06-07T02:00:00Z",
                "future_4h_net_bps": 40.0,
            },
            {
                "run_id": "20260607_01",
                "symbol": "BNB/USDT",
                "ts_utc": "2026-06-07T01:00:00Z",
                "generated_at": "2026-06-07T03:00:00Z",
                "future_4h_net_bps": 80.0,
            },
            {
                "run_id": "20260607_02",
                "symbol": "BNB/USDT",
                "ts_utc": "2026-06-07T02:00:00Z",
                "generated_at": "2026-06-07T03:00:00Z",
                "future_4h_net_bps": -20.0,
            },
        ]
    )

    summary = build_label_backtest_summary({"final_score_vs_alpha6_conflict": frame})

    row = summary.filter(
        pl.col("strategy_id") == "FINAL_SCORE_ALPHA6_CONFLICT_BACKTEST"
    ).to_dicts()[0]
    assert row["sample_count"] == 2
    assert row["complete_sample_count"] == 2
    assert row["avg_net_bps"] == 30.0
    assert row["dedupe_before_rows"] == 3
    assert row["dedupe_after_rows"] == 2
    assert row["duplicate_rate"] == pytest.approx(1 / 3)


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


def test_backtest_positive_paper_negative_is_quarantined() -> None:
    labels = pl.DataFrame(
        [
            {
                "run_id": "20260607_01",
                "strategy_id": "BNB_STRONG_ALPHA6_BYPASS_SHADOW_V1",
                "symbol": "BNB-USDT",
                "ts_utc": "2026-06-07T01:00:00Z",
                "future_4h_net_bps": 110.0,
            }
            for index in range(60)
        ]
    ).with_row_index("idx").with_columns(
        (
            pl.lit("20260607_")
            + pl.col("idx").cast(pl.Utf8).str.zfill(2)
        ).alias("run_id")
    ).drop("idx")
    bnb_paper = pl.DataFrame(
        [
            {
                "paper_date": "2026-06-07",
                "strategy_id": "BNB_RISK_ON_BUY_PAPER_V1",
                "symbol": "BNB-USDT",
                "entry_count": 25,
                "paper_days_to_date": 14,
                "avg_paper_pnl_bps_4h": -35.0,
                "avg_paper_pnl_bps": -35.0,
            }
        ]
    )

    bundle = build_backtest_report_bundle(
        {
            "bnb_strong_alpha6_bypass_shadow": labels,
            "bnb_paper_strategy_daily": bnb_paper,
        }
    )

    consistency = bundle.backtest_vs_paper_consistency.filter(
        pl.col("recommendation") == "QUARANTINE_BACKTEST_PAPER_CONFLICT"
    )
    assert consistency.height >= 1
    promotion = bundle.promotion_decision.filter(
        pl.col("strategy_id") == "BNB_STRONG_ALPHA6_BYPASS_BACKTEST"
    )
    assert "PAPER" not in set(promotion["recommended_stage"].to_list())
    assert "QUARANTINE" in set(promotion["recommended_stage"].to_list())
    assert "QUARANTINE_BACKTEST_PAPER_CONFLICT" in ";".join(
        promotion["decision_reasons"].to_list()
    )


def test_bnb_f3_backtest_positive_uses_bnb_paper_proxy_for_quarantine() -> None:
    label_summary = pl.DataFrame(
        [
            {
                "strategy_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                "symbol": "BNB-USDT",
                "horizon_hours": 24,
                "sample_count": 35,
                "complete_sample_count": 35,
                "avg_net_bps": 95.0,
                "p25_net_bps": 40.0,
                "win_rate": 0.74,
            }
        ]
    )
    bnb_paper = pl.DataFrame(
        [
            {
                "paper_date": "2026-06-07",
                "strategy_id": "BNB_RISK_ON_BUY_PAPER_V1",
                "symbol": "BNB-USDT",
                "entry_count": 18,
                "paper_days_to_date": 7,
                "avg_paper_pnl_bps_24h": -42.0,
                "avg_paper_pnl_bps": -42.0,
            }
        ]
    )

    consistency = build_backtest_vs_paper_consistency(
        label_summary=label_summary,
        bnb_paper_daily=bnb_paper,
    )
    conflict = consistency.filter(
        (pl.col("symbol") == "BNB-USDT")
        & (pl.col("recommendation") == "QUARANTINE_BACKTEST_PAPER_CONFLICT")
    )
    assert conflict.height == 1
    row = conflict.to_dicts()[0]
    assert row["paper_strategy_id"] == "BNB_RISK_ON_BUY_PAPER_V1"
    assert row["paper_avg_net_bps"] == -42.0


def test_backtest_vs_paper_reads_horizon_json_for_bnb_conflict() -> None:
    label_summary = pl.DataFrame(
        [
            {
                "strategy_id": "FINAL_SCORE_ALPHA6_CONFLICT_BACKTEST",
                "symbol": "BNB-USDT",
                "horizon_hours": 24,
                "sample_count": 40,
                "complete_sample_count": 40,
                "avg_net_bps": 120.0,
                "p25_net_bps": 35.0,
                "win_rate": 0.7,
            }
        ]
    )
    bnb_paper = pl.DataFrame(
        [
            {
                "paper_date": "2026-06-11",
                "strategy_id": "BNB_RISK_ON_BUY_PAPER_V1",
                "symbol": "BNB-USDT",
                "entry_count": 12,
                "paper_days_to_date": 3,
                "avg_paper_pnl_bps_by_horizon": '{"4h": -8.0, "24h": -55.5}',
            }
        ]
    )

    consistency = build_backtest_vs_paper_consistency(
        label_summary=label_summary,
        bnb_paper_daily=bnb_paper,
    )

    row = consistency.to_dicts()[0]
    assert row["strategy_id"] == "FINAL_SCORE_ALPHA6_CONFLICT_BACKTEST"
    assert row["paper_avg_net_bps"] == -55.5
    assert row["recommendation"] == "QUARANTINE_BACKTEST_PAPER_CONFLICT"


def test_duplicate_label_rate_blocks_paper_promotion() -> None:
    labels = pl.DataFrame(
        [
            {
                "run_id": "20260607_01",
                "strategy_id": "BNB_STRONG_ALPHA6_BYPASS_SHADOW_V1",
                "symbol": "BNB-USDT",
                "ts_utc": "2026-06-07T01:00:00Z",
                "generated_at": f"2026-06-07T01:{index:02d}:00Z",
                "future_4h_net_bps": 120.0,
            }
            for index in range(10)
        ]
    )

    bundle = build_backtest_report_bundle({"bnb_strong_alpha6_bypass_shadow": labels})

    summary = bundle.label_summary.filter(
        pl.col("strategy_id") == "BNB_STRONG_ALPHA6_BYPASS_BACKTEST"
    ).to_dicts()[0]
    assert summary["duplicate_rate"] == 0.9
    assert summary["recommendation"] == "QUARANTINE_DUPLICATE_LABELS"
    promotion = bundle.promotion_decision.filter(
        pl.col("strategy_id") == "BNB_STRONG_ALPHA6_BYPASS_BACKTEST"
    ).to_dicts()[0]
    assert promotion["recommended_stage"] == "QUARANTINE"
    assert "label_duplicate_rate_gt_5pct" in promotion["decision_reasons"]


def test_research_promotion_applies_entry_kill_rules() -> None:
    summary = pl.DataFrame(
        [
            {
                "strategy_id": "BNB_RISK_ON_BUY_PAPER_V1",
                "symbol": "BNB-USDT",
                "regime": "ALL",
                "horizon_hours": 24,
                "dedupe_before_rows": 100,
                "dedupe_after_rows": 100,
                "duplicate_rate": 0.0,
                "sample_count": 120,
                "complete_sample_count": 120,
                "avg_net_bps": 50.0,
                "median_net_bps": 50.0,
                "p25_net_bps": 10.0,
                "p10_net_bps": -20.0,
                "win_rate": 0.6,
                "max_loss_bps": -80.0,
                "cost_model": "mixed_actual_proxy",
                "data_leakage_check": "pass_visible_at_decision_time",
                "recommendation": "PAPER_CANDIDATE_REVIEW",
            },
            {
                "strategy_id": "RISK_ON_MULTI_BUY_BACKTEST",
                "symbol": "MULTI",
                "regime": "ALT_IMPULSE",
                "horizon_hours": 24,
                "dedupe_before_rows": 120,
                "dedupe_after_rows": 120,
                "duplicate_rate": 0.0,
                "sample_count": 120,
                "complete_sample_count": 120,
                "avg_net_bps": -20.0,
                "median_net_bps": -20.0,
                "p25_net_bps": -60.0,
                "p10_net_bps": -100.0,
                "win_rate": 0.4,
                "max_loss_bps": -140.0,
                "cost_model": "mixed_actual_proxy",
                "data_leakage_check": "pass_visible_at_decision_time",
                "recommendation": "KILL_OR_KEEP_RESEARCH",
            },
        ]
    )

    promotion = build_research_promotion_decision(label_summary=summary)

    stages = {row["strategy_id"]: row for row in promotion.to_dicts()}
    assert stages["BNB_RISK_ON_BUY_PAPER_V1"]["recommended_stage"] == "KILL_AS_ENTRY"
    assert (
        "forced_rule_bnb_risk_on_buy_kill_as_entry"
        in stages["BNB_RISK_ON_BUY_PAPER_V1"]["decision_reasons"]
    )
    assert stages["RISK_ON_MULTI_BUY_BACKTEST"]["recommended_stage"] == "KILL_AS_LIVE_ENTRY"
    assert (
        "forced_rule_risk_on_multi_buy_negative"
        in stages["RISK_ON_MULTI_BUY_BACKTEST"]["decision_reasons"]
    )


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


def test_backtest_label_summary_covers_hype_wld_expanded_universe_ready_labels() -> None:
    summary = build_label_backtest_summary(
        {
            "expanded_universe_candidate_maturity": pl.DataFrame(
                [
                    {
                        "symbol": "HYPE-USDT",
                        "expanded_universe_maturity_state": "PAPER_READY",
                        "generated_at": "2026-06-01T00:00:00Z",
                    },
                    {
                        "symbol": "WLD-USDT",
                        "expanded_universe_maturity_state": "PAPER_READY",
                        "generated_at": "2026-06-01T00:00:00Z",
                    },
                ]
            ),
            "expanded_universe_candidate_label": pl.DataFrame(
                [
                    {
                        "symbol": "HYPE-USDT",
                        "decision_ts": "2026-06-01T01:00:00Z",
                        "future_4h_net_bps": 75.0,
                    },
                    {
                        "symbol": "WLD-USDT",
                        "decision_ts": "2026-06-01T01:00:00Z",
                        "future_4h_net_bps": 55.0,
                    },
                ]
            ),
        }
    )

    strategy_ids = set(summary["strategy_id"].to_list())
    assert "HYPE_EXPANDED_UNIVERSE_BACKTEST" in strategy_ids
    assert "WLD_EXPANDED_UNIVERSE_BACKTEST" in strategy_ids
    hype = summary.filter(
        (pl.col("strategy_id") == "HYPE_EXPANDED_UNIVERSE_BACKTEST")
        & (pl.col("horizon_hours") == 4)
    ).to_dicts()[0]
    assert hype["complete_sample_count"] == 1
    assert hype["avg_net_bps"] == 75.0
