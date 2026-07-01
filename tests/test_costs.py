from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.costs.model import (
    DEFAULT_FALLBACK_COST_BPS,
    CostBucket,
    build_cost_bootstrap_readiness,
    estimate_cost_bps,
    estimate_cost_from_cost_bucket_daily_rows,
    evaluate_live_universe_cost_coverage,
)
from quant_lab.costs.probe import build_cost_probe_fill_bill_match


def test_cost_model_uses_exact_matching_bucket():
    estimate = estimate_cost_bps(
        "BTCUSDT",
        "normal",
        10_000,
        [
            CostBucket(
                bucket_id="btc-normal",
                symbol="BTCUSDT",
                regime="normal",
                min_notional_usdt=0,
                max_notional_usdt=50_000,
                cost_bps=4.5,
            )
        ],
    )

    assert estimate.cost_bps == 4.5
    assert estimate.symbol == "BTC-USDT"
    assert estimate.normalized_symbol == "BTC-USDT"
    assert estimate.bucket_id == "btc-normal"
    assert estimate.fallback_level == "NONE"


def test_cost_model_fallback_is_explicit_when_no_bucket_matches():
    estimate = estimate_cost_bps("ETHUSDT", "volatile", 10_000, [])

    assert estimate.cost_bps == DEFAULT_FALLBACK_COST_BPS
    assert estimate.bucket_id is None
    assert estimate.fallback_level == "DEFAULT_FALLBACK"


def test_cost_probe_fill_bill_match_infers_okx_account_bills():
    generated_at = datetime(2026, 6, 24, 9, 0, tzinfo=UTC)
    match = build_cost_probe_fill_bill_match(
        _probe_order_events(),
        _probe_roundtrip_events(),
        _probe_private_fills(),
        pl.DataFrame(
            [
                {
                    "bill_id": "bill-entry",
                    "inst_id": "ETH-USDT",
                    "ccy": "ETH",
                    "fee": -0.000002,
                    "px": 2500.0,
                    "ts": "2026-06-24T08:00:02Z",
                },
                {
                    "bill_id": "bill-entry-principal",
                    "inst_id": "ETH-USDT",
                    "ccy": "USDT",
                    "fee": 0.0,
                    "amount": -5.0,
                    "px": 2500.0,
                    "ts": "2026-06-24T08:00:02Z",
                },
                {
                    "bill_id": "bill-exit",
                    "inst_id": "ETH-USDT",
                    "ccy": "USDT",
                    "fee": -0.004,
                    "ts": "2026-06-24T08:00:10Z",
                },
            ]
        ),
        generated_at=generated_at,
    )

    row = match.to_dicts()[0]
    assert row["generated_at"] == "2026-06-24T09:00:00Z"
    assert row["symbol"] == "ETH-USDT"
    assert row["authorization_id"] == "auth-eth-1"
    assert row["roundtrip_id"] == "rt-eth-1"
    assert row["entry_order_id"] == "entry-order-1"
    assert row["exit_order_id"] == "exit-order-1"
    assert row["entry_trade_id"] == "entry-trade-1"
    assert row["exit_trade_id"] == "exit-trade-1"
    assert row["entry_bill_id"] == "bill-entry"
    assert row["exit_bill_id"] == "bill-exit"
    assert row["entry_fee_from_fill"] == "0.005"
    assert row["entry_fee_from_bill"] == "0.005"
    assert row["exit_fee_from_fill"] == "0.004"
    assert row["exit_fee_from_bill"] == "0.004"
    assert row["fee_diff_usdt"] == "0"
    assert row["bill_match_status"] == "PASS"


def test_cost_probe_fill_bill_match_reports_missing_bills():
    match = build_cost_probe_fill_bill_match(
        _probe_order_events(),
        _probe_roundtrip_events(),
        _probe_private_fills(),
        pl.DataFrame(),
        generated_at=datetime(2026, 6, 24, 9, 0, tzinfo=UTC),
    )

    row = match.to_dicts()[0]
    assert row["entry_fee_from_fill"] == "0.005"
    assert row["exit_fee_from_fill"] == "0.004"
    assert row["entry_bill_id"] == ""
    assert row["exit_bill_id"] == ""
    assert row["bill_match_status"] == "BILL_NOT_OBSERVED"


def test_cost_probe_fill_bill_match_reconstructs_okx_spot_ledger_amounts():
    match = build_cost_probe_fill_bill_match(
        _probe_order_events(),
        _probe_roundtrip_events(),
        _probe_private_fills(),
        pl.DataFrame(
            [
                {
                    "bill_id": "bill-entry-base",
                    "ccy": "ETH",
                    "amount": 0.002998,
                    "ts": "2026-06-24T08:00:01Z",
                },
                {
                    "bill_id": "bill-entry-quote",
                    "ccy": "USDT",
                    "amount": -7.5,
                    "ts": "2026-06-24T08:00:01Z",
                },
                {
                    "bill_id": "bill-exit-base",
                    "ccy": "ETH",
                    "amount": -0.003,
                    "ts": "2026-06-24T08:00:09Z",
                },
                {
                    "bill_id": "bill-exit-quote",
                    "ccy": "USDT",
                    "amount": 7.493,
                    "ts": "2026-06-24T08:00:09Z",
                },
            ]
        ),
        generated_at=datetime(2026, 6, 24, 9, 0, tzinfo=UTC),
    )

    row = match.to_dicts()[0]
    assert set(row["entry_bill_id"].split(";")) == {"bill-entry-base", "bill-entry-quote"}
    assert set(row["exit_bill_id"].split(";")) == {"bill-exit-base", "bill-exit-quote"}
    assert row["entry_fee_from_bill"] == "0.005"
    assert row["exit_fee_from_bill"] == "0.004"
    assert row["fee_diff_usdt"] == "0"
    assert row["bill_match_status"] == "PASS"


def test_bootstrap_readiness_consumes_cost_probe_fill_bill_match_pass():
    now = datetime(2026, 6, 24, 9, 0, tzinfo=UTC)

    readiness = build_cost_bootstrap_readiness(
        pl.DataFrame(
            [
                {
                    **_coverage_cost_row("ETH-USDT", "bootstrap_cost_probe", now),
                    "sample_count": 2,
                    "cost_probe_fill_count": 2,
                }
            ]
        ),
        v5_cost_probe_order_events=_probe_order_events(),
        v5_cost_probe_roundtrip_events=_probe_roundtrip_events(),
        okx_private_readonly_fills=_probe_private_fills(),
        okx_private_readonly_bills=pl.DataFrame(
            [
                {
                    "bill_id": "bill-entry",
                    "inst_id": "ETH-USDT",
                    "ccy": "ETH",
                    "fee": -0.000002,
                    "px": 2500.0,
                    "ts": "2026-06-24T08:00:02Z",
                },
                {
                    "bill_id": "bill-entry-principal",
                    "inst_id": "ETH-USDT",
                    "ccy": "USDT",
                    "fee": 0.0,
                    "amount": -5.0,
                    "px": 2500.0,
                    "ts": "2026-06-24T08:00:02Z",
                },
                {
                    "bill_id": "bill-exit",
                    "inst_id": "ETH-USDT",
                    "ccy": "USDT",
                    "fee": -0.004,
                    "ts": "2026-06-24T08:00:10Z",
                },
            ]
        ),
        live_symbols=["ETH-USDT"],
        generated_at=now,
    )

    eth = readiness.to_dicts()[0]
    assert eth["bootstrap_state"] == "BOOTSTRAP_PROBE_AVAILABLE"
    assert eth["bill_match_status"] == "PASS"
    assert eth["bill_matched_count"] == 2
    assert eth["fee_match_status"] == "fill_bill_fee_match"
    assert eth["fee_match_diff_usdt"] == "0"
    assert eth["trusted_for_live"] is False
    assert eth["actual_or_mixed_trusted_coverage_live_universe"] == 0.0
    assert "BOOTSTRAP_COMPLETE_BILL_MATCHED" in eth["next_action"]
    assert "resolve bill_match" not in eth["next_action"]


def _probe_order_events() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "event_ts": "2026-06-24T08:00:01Z",
                "symbol": "ETH-USDT",
                "leg": "entry",
                "order_status": "filled",
                "order_id": "entry-order-1",
                "exchange_order_id": "entry-order-1",
                "trade_id": "entry-trade-1",
                "filled_qty": "0.003",
                "avg_px": "2500",
                "fee_usdt": "0.005",
            },
            {
                "event_ts": "2026-06-24T08:00:09Z",
                "symbol": "ETH-USDT",
                "leg": "exit",
                "order_status": "filled",
                "order_id": "exit-order-1",
                "exchange_order_id": "exit-order-1",
                "trade_id": "exit-trade-1",
                "filled_qty": "0.003",
                "avg_px": "2499",
                "fee_usdt": "0.004",
            },
        ]
    )


def _probe_roundtrip_events() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "event_ts": "2026-06-24T08:00:12Z",
                "symbol": "ETH-USDT",
                "roundtrip_id": "rt-eth-1",
                "roundtrip_status": "closed",
                "authorization_id": "auth-eth-1",
                "entry_order_id": "entry-order-1",
                "exit_order_id": "exit-order-1",
                "execution_completed": True,
                "completed": True,
                "flat_verified": True,
                "exchange_flat_verified": True,
                "local_flat_verified": True,
                "reconcile_ok": True,
                "cost_evidence_complete": True,
                "eligible_for_cost_model": True,
                "no_order_submitted": False,
            }
        ]
    )


def _probe_private_fills() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "inst_id": "ETH-USDT",
                "order_id": "entry-order-1",
                "trade_id": "entry-trade-1",
                "side": "buy",
                "fill_price": 2500.0,
                "fill_size": 0.003,
                "fee": -0.005,
                "fee_currency": "USDT",
                "ts": "2026-06-24T08:00:01Z",
            },
            {
                "inst_id": "ETH-USDT",
                "order_id": "exit-order-1",
                "trade_id": "exit-trade-1",
                "side": "sell",
                "fill_price": 2499.0,
                "fill_size": 0.003,
                "fee": -0.004,
                "fee_currency": "USDT",
                "ts": "2026-06-24T08:00:09Z",
            },
        ]
    )


def test_live_universe_cost_coverage_rejects_stale_direct_even_with_proxy():
    now = datetime(2026, 6, 15, tzinfo=UTC)
    stale = now - timedelta(days=7)

    evaluation = evaluate_live_universe_cost_coverage(
        pl.DataFrame(
            [
                _coverage_cost_row("BTC-USDT", "actual_fills", now),
                _coverage_cost_row("SOL-USDT", "mixed_actual_proxy", stale),
                _coverage_cost_row("SOL-USDT", "public_spread_proxy", now),
            ]
        ),
        live_symbols=["BTC-USDT", "SOL-USDT"],
        generated_at=now,
    )

    sol = evaluation["detail_by_symbol"]["SOL-USDT"]
    assert sol["stale_actual_or_mixed"] is True
    assert sol["latest_actual_or_mixed_created_at"] == stale.isoformat().replace("+00:00", "Z")
    assert sol["latest_actual_or_mixed_age_sec"] == 7 * 24 * 60 * 60
    assert sol["actual_or_mixed_direct"] is False
    assert sol["anchored_mixed_proxy_candidate"] is True
    assert sol["mixed_proxy_eligible"] is True
    assert sol["actual_or_mixed_covered"] is False
    assert sol["cost_evidence_tier"] == "anchored_proxy_candidate_not_counted"
    assert sol["fee_fresh"] is False
    assert sol["spread_fresh"] is True
    assert sol["slippage_fresh"] is False
    assert sol["coverage_reason"] == "stale_actual_or_mixed_with_anchored_proxy_not_counted"
    assert evaluation["direct_symbols"] == ["BTC-USDT"]
    assert evaluation["mixed_proxy_symbols"] == ["SOL-USDT"]
    assert evaluation["coverage_rate"] == 0.5


def test_live_universe_cost_coverage_uses_generated_at_for_stale_window():
    generated_at = datetime(2026, 6, 15, tzinfo=UTC)
    fresh_at_generation = generated_at - timedelta(hours=35)

    evaluation = evaluate_live_universe_cost_coverage(
        pl.DataFrame(
            [
                _coverage_cost_row("BTC-USDT", "actual_fills", fresh_at_generation),
            ]
        ),
        live_symbols=["BTC-USDT"],
        generated_at=generated_at,
    )

    btc = evaluation["detail_by_symbol"]["BTC-USDT"]
    assert btc["stale_actual_or_mixed"] is False
    assert btc["actual_or_mixed_direct"] is True
    assert btc["actual_or_mixed_covered"] is True
    assert btc["cost_evidence_tier"] == "strict_direct_actual_or_mixed"
    assert btc["fee_fresh"] is True
    assert btc["slippage_fresh"] is True
    assert evaluation["direct_symbols"] == ["BTC-USDT"]
    assert evaluation["coverage_rate"] == 1.0


def test_live_universe_cost_coverage_rejects_stale_direct_without_fresh_anchor():
    now = datetime(2026, 6, 15, tzinfo=UTC)
    stale = now - timedelta(days=7)

    evaluation = evaluate_live_universe_cost_coverage(
        pl.DataFrame(
            [
                _coverage_cost_row("BTC-USDT", "actual_fills", stale),
                _coverage_cost_row("BTC-USDT", "public_spread_proxy", now),
            ]
        ),
        live_symbols=["BTC-USDT"],
        generated_at=now,
    )

    btc = evaluation["detail_by_symbol"]["BTC-USDT"]
    assert btc["stale_actual_or_mixed"] is True
    assert btc["actual_or_mixed_direct"] is False
    assert btc["anchored_mixed_proxy_candidate"] is False
    assert btc["mixed_proxy_eligible"] is False
    assert btc["actual_or_mixed_covered"] is False
    assert btc["coverage_reason"] == "stale_actual_or_mixed_no_fresh_live_anchor"
    assert evaluation["direct_symbols"] == []
    assert evaluation["coverage_rate"] == 0.0


def test_cost_bootstrap_readiness_keeps_proxy_and_probe_out_of_trusted_live():
    now = datetime(2026, 6, 15, tzinfo=UTC)

    readiness = build_cost_bootstrap_readiness(
        pl.DataFrame([_coverage_cost_row("SOL-USDT", "public_spread_proxy", now)]),
        v5_order_lifecycle=pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "order_state": "FILLED",
                    "fill_count": 1,
                    "avg_fill_px": 70000.0,
                    "filled_qty": 0.0001,
                    "notional_usdt": 7.0,
                    "fee_bps": 1.0,
                    "arrival_slippage_bps": 0.5,
                    "arrival_spread_bps": 0.2,
                    "execution_purpose": "cost_probe",
                    "eligible_for_cost_model": True,
                    "eligible_for_alpha_pnl": False,
                    "last_fill_ts": now.isoformat().replace("+00:00", "Z"),
                }
            ]
        ),
        live_symbols=["BTC-USDT", "SOL-USDT"],
        generated_at=now,
    )

    rows = {row["symbol"]: row for row in readiness.to_dicts()}
    assert rows["SOL-USDT"]["bootstrap_state"] == "PUBLIC_PROXY_ONLY"
    assert rows["SOL-USDT"]["actual_or_mixed_bootstrap_covered"] is False
    assert rows["SOL-USDT"]["trusted_for_live"] is False
    assert rows["BTC-USDT"]["bootstrap_state"] == "BOOTSTRAP_PROBE_AVAILABLE"
    assert rows["BTC-USDT"]["actual_or_mixed_bootstrap_covered"] is True
    assert rows["BTC-USDT"]["actual_or_mixed_trusted_covered"] is False
    assert rows["BTC-USDT"]["trusted_for_live"] is False
    assert rows["BTC-USDT"]["live_order_effect"] == "read_only_no_live_order"
    assert rows["BTC-USDT"]["actual_or_mixed_bootstrap_coverage_live_universe"] == 0.5
    assert rows["BTC-USDT"]["actual_or_mixed_trusted_coverage_live_universe"] == 0.0


def test_cost_probe_only_bucket_does_not_count_as_live_universe_cost_coverage():
    now = datetime(2026, 6, 15, tzinfo=UTC)

    evaluation = evaluate_live_universe_cost_coverage(
        pl.DataFrame([_coverage_cost_row("BTC-USDT", "bootstrap_cost_probe", now)]),
        live_symbols=["BTC-USDT"],
        generated_at=now,
    )

    btc = evaluation["detail_by_symbol"]["BTC-USDT"]
    assert btc["latest_source"] == "bootstrap_cost_probe"
    assert btc["cost_evidence_tier"] == "bootstrap_cost_probe_not_counted"
    assert btc["actual_or_mixed_direct"] is False
    assert btc["actual_or_mixed_covered"] is False
    assert btc["eligible_for_live_cost_coverage"] is False
    assert btc["coverage_reason"] == "bootstrap_cost_probe_not_live_coverage"
    assert evaluation["direct_symbols"] == []
    assert evaluation["coverage_rate"] == 0.0


def test_live_universe_cost_coverage_keeps_latest_bootstrap_detail_over_old_proxy():
    now = datetime(2026, 6, 15, tzinfo=UTC)
    old_proxy = now - timedelta(days=1)

    evaluation = evaluate_live_universe_cost_coverage(
        pl.DataFrame(
            [
                _coverage_cost_row("SOL-USDT", "actual_fills", now),
                _coverage_cost_row("BNB-USDT", "public_spread_proxy", old_proxy),
                _coverage_cost_row("BNB-USDT", "bootstrap_cost_probe", now),
            ]
        ),
        live_symbols=["BNB-USDT", "SOL-USDT"],
        generated_at=now,
    )

    bnb = evaluation["detail_by_symbol"]["BNB-USDT"]
    assert bnb["latest_source"] == "bootstrap_cost_probe"
    assert bnb["effective_cost_source"] == "bootstrap_cost_probe"
    assert bnb["sample_origin_mix"] == "cost_probe_only"
    assert bnb["sample_count"] == 4
    assert bnb["cost_probe_fill_count"] == 4
    assert bnb["proxy_sample_count"] == 0
    assert bnb["cost_evidence_tier"] == "bootstrap_cost_probe_not_counted"
    assert bnb["coverage_reason"] == "bootstrap_cost_probe_not_live_coverage"
    assert bnb["actual_or_mixed_covered"] is False
    assert evaluation["direct_symbols"] == ["SOL-USDT"]
    assert evaluation["mixed_proxy_symbols"] == []
    assert evaluation["coverage_rate"] == 0.5


def test_cost_probe_only_bucket_counts_as_bootstrap_not_trusted_live():
    now = datetime(2026, 6, 15, tzinfo=UTC)

    readiness = build_cost_bootstrap_readiness(
        pl.DataFrame(
            [
                {
                    **_coverage_cost_row("BTC-USDT", "bootstrap_cost_probe", now),
                    "sample_count": 30,
                    "cost_probe_fill_count": 30,
                }
            ]
        ),
        live_symbols=["BTC-USDT"],
        generated_at=now,
    )

    btc = readiness.to_dicts()[0]
    assert btc["bootstrap_state"] == "BOOTSTRAP_PROBE_AVAILABLE"
    assert btc["cost_evidence_tier"] == "bootstrap_cost_probe"
    assert btc["sample_count"] == 30
    assert btc["cost_probe_fill_count"] == 30
    assert btc["live_cost_sample_count"] == 0
    assert btc["trusted_sample_count"] == 0
    assert btc["actual_or_mixed_bootstrap_covered"] is True
    assert btc["actual_or_mixed_trusted_covered"] is False
    assert btc["trusted_for_live"] is False
    assert btc["actual_or_mixed_bootstrap_coverage_live_universe"] == 1.0
    assert btc["actual_or_mixed_trusted_coverage_live_universe"] == 0.0


def test_bootstrap_readiness_prefers_fresh_probe_over_stale_actual_row():
    now = datetime(2026, 6, 21, 13, 30, tzinfo=UTC)
    stale_actual = _coverage_cost_row("BTC-USDT", "mixed_actual_proxy", now - timedelta(days=3))
    bootstrap_probe = {
        **_coverage_cost_row("BTC-USDT", "bootstrap_cost_probe", now - timedelta(minutes=15)),
        "sample_count": 2,
        "cost_probe_fill_count": 2,
    }

    readiness = build_cost_bootstrap_readiness(
        pl.DataFrame([stale_actual, bootstrap_probe]),
        v5_order_lifecycle=pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "order_state": "FILLED",
                    "fill_count": 1,
                    "avg_fill_px": 70000.0,
                    "filled_qty": 0.0001,
                    "notional_usdt": 7.0,
                    "fee_bps": 1.0,
                    "arrival_slippage_bps": 0.5,
                    "arrival_spread_bps": 0.2,
                    "execution_purpose": "strategy_live",
                    "last_fill_ts": (now - timedelta(days=10)).isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                }
            ]
        ),
        v5_cost_probe_roundtrip_events=pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "roundtrip_status": "closed",
                    "event_ts": (now - timedelta(minutes=10)).isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                    "entry_filled_qty": "0.0001",
                    "exit_filled_qty": "0.000099",
                    "execution_completed": True,
                    "bill_match_status": "bill_not_observed",
                    "fee_match_status": "fill_fee_observed",
                    "fee_match_diff_usdt": "",
                }
            ]
        ),
        live_symbols=["BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"],
        generated_at=now,
    )

    rows = {row["symbol"]: row for row in readiness.to_dicts()}
    btc = rows["BTC-USDT"]
    assert btc["bootstrap_state"] == "BOOTSTRAP_PROBE_AVAILABLE"
    assert btc["latest_cost_source"] == "bootstrap_cost_probe"
    assert btc["cost_probe_fill_count"] == 2
    assert btc["actual_fill_count"] == 0
    assert btc["strategy_live_fill_count"] == 0
    assert btc["latest_probe_ts"] == (now - timedelta(minutes=10)).isoformat().replace(
        "+00:00",
        "Z",
    )
    assert btc["latest_probe_fill_ts"] == btc["latest_probe_ts"]
    assert btc["fill_match_status"] == "entry_exit_fill_observed"
    assert btc["bill_match_status"] == "bill_not_observed"
    assert btc["fee_match_status"] == "fill_fee_observed"
    assert btc["matched_bill_count"] == 0
    assert btc["actual_or_mixed_bootstrap_covered"] is True
    assert btc["trusted_for_live"] is False
    assert btc["actual_or_mixed_bootstrap_coverage_live_universe"] == 0.25
    assert btc["actual_or_mixed_trusted_coverage_live_universe"] == 0.0


def test_cost_probe_mixed_samples_do_not_satisfy_trusted_live():
    now = datetime(2026, 6, 15, tzinfo=UTC)
    row = {
        **_coverage_cost_row("BTC-USDT", "mixed_actual_proxy", now),
        "sample_count": 30,
        "actual_fill_count": 0,
        "mixed_fill_count": 1,
        "cost_probe_fill_count": 29,
        "strategy_live_fill_count": 1,
        "private_fill_count": 0,
        "sample_origin_mix": "cost_probe+strategy_live",
        "eligible_for_live_cost_coverage": True,
    }

    readiness = build_cost_bootstrap_readiness(
        pl.DataFrame([row]),
        live_symbols=["BTC-USDT"],
        min_trusted_sample_count=30,
        generated_at=now,
    )

    btc = readiness.to_dicts()[0]
    assert btc["bootstrap_state"] == "MIXED_ACTUAL_PROXY_AVAILABLE"
    assert btc["sample_count"] == 30
    assert btc["cost_probe_fill_count"] == 29
    assert btc["live_cost_sample_count"] == 1
    assert btc["trusted_sample_count"] == 1
    assert btc["actual_or_mixed_bootstrap_covered"] is True
    assert btc["actual_or_mixed_trusted_covered"] is False
    assert btc["trusted_for_live"] is False
    assert btc["actual_or_mixed_bootstrap_coverage_live_universe"] == 1.0
    assert btc["actual_or_mixed_trusted_coverage_live_universe"] == 0.0


def test_thirty_live_samples_without_cost_probe_satisfy_trusted_live():
    now = datetime(2026, 6, 15, tzinfo=UTC)
    row = {
        **_coverage_cost_row("BTC-USDT", "actual_fills", now),
        "sample_count": 30,
        "actual_fill_count": 30,
        "mixed_fill_count": 0,
        "cost_probe_fill_count": 0,
        "strategy_live_fill_count": 30,
        "private_fill_count": 0,
        "sample_origin_mix": "strategy_live",
        "eligible_for_live_cost_coverage": True,
    }

    readiness = build_cost_bootstrap_readiness(
        pl.DataFrame([row]),
        live_symbols=["BTC-USDT"],
        min_trusted_sample_count=30,
        generated_at=now,
    )

    btc = readiness.to_dicts()[0]
    assert btc["bootstrap_state"] == "ACTUAL_FILLS_TRUSTED"
    assert btc["sample_count"] == 30
    assert btc["live_cost_sample_count"] == 30
    assert btc["trusted_sample_count"] == 30
    assert btc["actual_or_mixed_trusted_covered"] is True
    assert btc["trusted_for_live"] is True
    assert btc["actual_or_mixed_trusted_coverage_live_universe"] == 1.0


def test_cost_estimate_live_trust_uses_live_samples_not_probe_samples():
    row = {
        **_estimate_cost_row(
            source="mixed_actual_proxy",
            sample_count=30,
            fallback_level="COST_PROBE_INCLUDED",
        ),
        "actual_fill_count": 0,
        "mixed_fill_count": 1,
        "cost_probe_fill_count": 29,
        "strategy_live_fill_count": 1,
        "private_fill_count": 0,
        "sample_origin_mix": "cost_probe+strategy_live",
        "eligible_for_live_cost_coverage": True,
    }

    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p75",
        rows=[row],
    )

    assert estimate.sample_count == 30
    assert estimate.live_cost_sample_count == 1
    assert estimate.trusted_live_sample_count == 1
    assert estimate.cost_quality == "small_sample"
    assert estimate.cost_trusted_for_live is False
    assert estimate.cost_trusted_for_live_canary is False
    assert "sample_count_lt_30" in estimate.cost_trust_block_reasons


def test_cost_estimate_live_trust_accepts_thirty_live_samples_without_probe():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                **_estimate_cost_row(
                    source="actual_fills",
                    sample_count=30,
                    fallback_level="NONE",
                ),
                "actual_fill_count": 30,
                "mixed_fill_count": 0,
                "cost_probe_fill_count": 0,
                "strategy_live_fill_count": 30,
                "private_fill_count": 0,
                "sample_origin_mix": "strategy_live",
                "eligible_for_live_cost_coverage": True,
            }
        ],
    )

    assert estimate.live_cost_sample_count == 30
    assert estimate.trusted_live_sample_count == 30
    assert estimate.cost_trusted_for_live is True
    assert estimate.cost_trusted_for_live_canary is True


def test_cost_model_can_fallback_to_symbol_bucket():
    estimate = estimate_cost_bps(
        "BTCUSDT",
        "volatile",
        10_000,
        [
            {
                "bucket_id": "btc-any-regime",
                "symbol": "BTCUSDT",
                "regime": None,
                "min_notional_usdt": 0,
                "max_notional_usdt": None,
                "cost_bps": 6.0,
            }
        ],
    )

    assert estimate.cost_bps == 6.0
    assert estimate.bucket_id == "btc-any-regime"
    assert estimate.fallback_level == "REGIME_FALLBACK"


def test_cost_bucket_daily_estimate_uses_requested_quantile():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p90",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BTC-USDT",
                "regime": "normal",
                "event_type": "trade",
                "notional_bucket": "1k-10k",
                "sample_count": 42,
                "fee_bps_p50": 1.0,
                "fee_bps_p75": 1.5,
                "fee_bps_p90": 2.0,
                "slippage_bps_p50": 2.0,
                "slippage_bps_p75": 3.0,
                "slippage_bps_p90": 4.0,
                "spread_bps_p50": 0.5,
                "spread_bps_p75": 0.75,
                "spread_bps_p90": 1.0,
                "total_cost_bps_p50": 3.5,
                "total_cost_bps_p75": 5.25,
                "total_cost_bps_p90": 7.0,
                "fallback_level": "actual_okx_fills_and_bills",
                "source": "actual_okx_fills_and_bills",
                "cost_model_version": "costs-2026-05-10",
            }
        ],
    )

    assert estimate.quantile == "p90"
    assert estimate.requested_quantile == "p90"
    assert estimate.fee_bps == 2.0
    assert estimate.slippage_bps == 4.0
    assert estimate.spread_bps == 1.0
    assert estimate.total_cost_bps == 7.0
    assert estimate.total_cost_bps_p50 == 3.5
    assert estimate.total_cost_bps_p75 == 5.25
    assert estimate.total_cost_bps_p90 == 7.0
    assert estimate.cost_bps == 7.0
    assert estimate.fallback_level == "NONE"
    assert estimate.fallback_reason == "NONE"
    assert estimate.cost_source == "actual_okx_fills_and_bills"
    assert estimate.degraded_cost_model is False
    assert estimate.sample_count == 42
    assert estimate.sample_size == 42
    assert estimate.cost_model_version == "costs-2026-05-10"
    assert estimate.fee_source == "actual_fills_bills"
    assert estimate.spread_source == "fresh_public_orderbook_p75"
    assert estimate.slippage_source == "v5_order_lifecycle_arrival_mid"
    assert estimate.uncertainty_buffer_bps == 0.0
    assert estimate.one_way_all_in_cost_bps == 7.0
    assert estimate.roundtrip_all_in_cost_bps == 14.0
    assert estimate.cost_quality == "actual"
    assert estimate.cost_trusted_for_paper is True
    assert estimate.cost_trusted_for_live is True


def test_cost_bucket_daily_estimate_uses_global_fallback_when_no_bucket_matches():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="DOGE-USDT",
        regime="volatile",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "GLOBAL",
                "regime": "global",
                "event_type": "trade",
                "notional_bucket": "all",
                "sample_count": 8,
                "fee_bps_p50": 1.0,
                "fee_bps_p75": 2.0,
                "fee_bps_p90": 3.0,
                "slippage_bps_p50": 1.0,
                "slippage_bps_p75": 2.0,
                "slippage_bps_p90": 3.0,
                "spread_bps_p50": 1.0,
                "spread_bps_p75": 2.0,
                "spread_bps_p90": 3.0,
                "total_cost_bps_p50": 3.0,
                "total_cost_bps_p75": 6.0,
                "total_cost_bps_p90": 9.0,
                "fallback_level": "public_spread_proxy",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.total_cost_bps == 6.0
    assert estimate.fallback_level == "GLOBAL_BUCKET_FALLBACK;public_spread_proxy"
    assert estimate.source == "public_spread_proxy"


def test_cost_bucket_daily_estimate_preserves_proxy_fallback_on_exact_match():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BTC-USDT",
                "regime": "public_proxy",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 8,
                "fee_bps_p75": 0.0,
                "slippage_bps_p75": 0.0,
                "spread_bps_p75": 2.0,
                "total_cost_bps_p75": 2.0,
                "fallback_level": "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.total_cost_bps == 2.0
    assert estimate.fallback_level == "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY"
    assert estimate.source == "public_spread_proxy"
    assert estimate.fee_bps == 10.0
    assert estimate.fee_source == "config_fee_bps"
    assert estimate.slippage_bps == 2.0
    assert estimate.slippage_source == "config_slippage_bps"
    assert estimate.spread_source == "fresh_public_orderbook_p75"
    assert estimate.uncertainty_buffer_bps == 5.0
    assert estimate.one_way_all_in_cost_bps == 19.0
    assert estimate.roundtrip_all_in_cost_bps == 38.0
    assert estimate.cost_quality == "public_proxy_only"
    assert estimate.cost_trusted_for_paper is True
    assert estimate.cost_trusted_for_live is False


def test_cost_bucket_daily_estimate_returns_explicit_global_default_without_rows():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p50",
        rows=[],
    )

    assert estimate.total_cost_bps == DEFAULT_FALLBACK_COST_BPS
    assert estimate.fallback_level == "GLOBAL_DEFAULT"
    assert estimate.source == "global_default"
    assert estimate.sample_count == 0
    assert estimate.fee_bps == 10.0
    assert estimate.spread_bps == 5.0
    assert estimate.slippage_bps == 5.0
    assert estimate.uncertainty_buffer_bps == 5.0
    assert estimate.one_way_all_in_cost_bps == 25.0
    assert estimate.roundtrip_all_in_cost_bps == 50.0
    assert estimate.cost_quality == "global_default"
    assert estimate.cost_trusted_for_paper is False
    assert estimate.cost_trusted_for_live is False


def test_cost_bucket_daily_estimate_normalizes_strategy_symbol_to_cost_bucket():
    rows = [
        {
            "day": "2026-05-10",
            "symbol": "BNB-USDT",
            "regime": "public_proxy",
            "event_type": "spread_proxy",
            "notional_bucket": "all",
            "sample_count": 8,
            "fee_bps_p50": 0.0,
            "fee_bps_p75": 0.0,
            "fee_bps_p90": 0.0,
            "slippage_bps_p50": 0.0,
            "slippage_bps_p75": 0.0,
            "slippage_bps_p90": 0.0,
            "spread_bps_p50": 1.0,
            "spread_bps_p75": 2.0,
            "spread_bps_p90": 3.0,
            "total_cost_bps_p50": 1.0,
            "total_cost_bps_p75": 2.0,
            "total_cost_bps_p90": 3.0,
            "fallback_level": "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY",
            "source": "public_spread_proxy",
            "cost_model_version": "costs-2026-05-10",
            "created_at": "2026-05-10T01:00:00Z",
        }
    ]

    slash = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB/USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=rows,
    )
    hyphen = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB-USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=rows,
    )

    assert slash.symbol == "BNB-USDT"
    assert slash.normalized_symbol == "BNB-USDT"
    assert slash.total_cost_bps == 2.0
    assert slash.source == "public_spread_proxy"
    assert slash.cost_source == "public_spread_proxy"
    assert slash.sample_size == 8
    assert slash.as_of_ts is not None
    assert hyphen.model_dump() == slash.model_dump()


def test_cost_bucket_daily_estimate_can_fallback_to_symbol_bucket_across_regime():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB/USDT",
        regime="normal",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BNB-USDT",
                "regime": "public_proxy",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 8,
                "total_cost_bps_p75": 2.0,
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.symbol == "BNB-USDT"
    assert estimate.total_cost_bps == 2.0
    assert estimate.source == "public_spread_proxy"
    assert estimate.fallback_level == "REGIME_FALLBACK;PUBLIC_SPREAD_PROXY"
    assert estimate.requested_regime == "normal"
    assert estimate.matched_regime == "public_proxy"
    assert estimate.cost_source == "public_spread_proxy"
    assert estimate.selected_total_cost_bps == 2.0
    assert estimate.fallback_reason in {"no_matching_regime", "cost_bucket_stale"}
    assert estimate.degraded_cost_model is True
    assert estimate.one_way_all_in_cost_bps > estimate.selected_total_cost_bps
    assert estimate.roundtrip_all_in_cost_bps == estimate.one_way_all_in_cost_bps * 2.0
    assert estimate.cost_trusted_for_paper is True
    assert estimate.cost_trusted_for_live is False


def test_cost_bucket_daily_estimate_unknown_symbol_uses_global_default():
    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="UNKNOWN/USDT",
        regime="public_proxy",
        notional_usdt=5_000,
        quantile="p75",
        rows=[
            {
                "day": "2026-05-10",
                "symbol": "BNB-USDT",
                "regime": "public_proxy",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 8,
                "total_cost_bps_p75": 2.0,
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
            }
        ],
    )

    assert estimate.source == "global_default"
    assert estimate.cost_source == "global_default"
    assert estimate.fallback_level == "GLOBAL_DEFAULT"
    assert estimate.fallback_reason == "symbol_missing"
    assert estimate.degraded_reason == "global_default_cost"
    assert estimate.degraded_cost_model is True
    assert estimate.total_cost_bps == DEFAULT_FALLBACK_COST_BPS


def test_cost_bucket_daily_estimate_prefers_actual_fills_over_public_proxy():
    rows = [
        {
            "day": "2026-05-10",
            "symbol": "BNB-USDT",
            "regime": "realized",
            "event_type": "actual_fill",
            "notional_bucket": "all",
            "sample_count": 3,
            "total_cost_bps_p75": 4.0,
            "fallback_level": "SLIPPAGE_UNKNOWN",
            "source": "actual_fills",
        },
        {
            "day": "2026-05-10",
            "symbol": "BNB-USDT",
            "regime": "realized",
            "event_type": "spread_proxy",
            "notional_bucket": "all",
            "sample_count": 300,
            "total_cost_bps_p75": 1.0,
            "fallback_level": "PUBLIC_SPREAD_PROXY",
            "source": "public_spread_proxy",
        },
    ]

    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB/USDT",
        regime="realized",
        notional_usdt=5_000,
        quantile="p75",
        rows=rows,
    )

    assert estimate.total_cost_bps == 4.0
    assert estimate.source == "actual_fills"
    assert estimate.sample_size == 3


def test_cost_bucket_daily_estimate_prefers_cross_regime_mixed_actual_over_public_proxy():
    rows = [
        {
            "day": "2026-05-14",
            "symbol": "BNB-USDT",
            "regime": "realized",
            "event_type": "actual_fill",
            "notional_bucket": "all",
            "sample_count": 4,
            "fee_bps_p75": 1.0,
            "spread_bps_p75": 1.5,
            "total_cost_bps_p75": 2.5,
            "fallback_level": "SLIPPAGE_UNKNOWN;SPREAD_PROXY",
            "source": "mixed_actual_proxy",
        },
        {
            "day": "2026-05-14",
            "symbol": "BNB-USDT",
            "regime": "public_proxy",
            "event_type": "spread_proxy",
            "notional_bucket": "all",
            "sample_count": 1000,
            "spread_bps_p75": 1.49,
            "total_cost_bps_p75": 1.49,
            "fallback_level": "FEE_MISSING;SLIPPAGE_UNKNOWN;PUBLIC_SPREAD_PROXY",
            "source": "public_spread_proxy",
        },
    ]

    estimate = estimate_cost_from_cost_bucket_daily_rows(
        symbol="BNB-USDT",
        regime="Trending",
        notional_usdt=1_000,
        quantile="p75",
        rows=rows,
    )

    assert estimate.source == "mixed_actual_proxy"
    assert estimate.matched_regime == "realized"
    assert estimate.total_cost_bps == 2.5


def _coverage_cost_row(symbol: str, source: str, created_at: datetime) -> dict[str, object]:
    is_bootstrap_probe = source == "bootstrap_cost_probe"
    return {
        "day": created_at.date().isoformat(),
        "symbol": symbol,
        "regime": "realized" if source != "public_spread_proxy" else "public_proxy",
        "event_type": "actual_fill" if source != "public_spread_proxy" else "spread_proxy",
        "notional_bucket": "all",
        "sample_count": 4 if source != "public_spread_proxy" else 100,
        "fee_bps_p75": 10.0 if source != "public_spread_proxy" else 0.0,
        "spread_bps_p75": 1.0,
        "slippage_bps_p75": 0.5 if source != "public_spread_proxy" else 0.0,
        "total_cost_bps_p75": 11.5 if source != "public_spread_proxy" else 1.0,
        "source": source,
        "cost_source": source,
        "actual_fill_count": 4 if source == "actual_fills" else 0,
        "mixed_fill_count": 4 if source == "mixed_actual_proxy" else 0,
        "proxy_sample_count": 100 if source == "public_spread_proxy" else 0,
        "cost_probe_fill_count": 4 if is_bootstrap_probe else 0,
        "strategy_live_fill_count": 0,
        "private_fill_count": 0,
        "sample_origin_mix": (
            "cost_probe_only"
            if is_bootstrap_probe
            else "public_proxy"
            if source == "public_spread_proxy"
            else "strategy_live"
        ),
        "eligible_for_live_cost_coverage": (
            source in {"actual_fills", "mixed_actual_proxy", "actual_okx_fills_fee_missing"}
        ),
        "created_at": created_at.isoformat(),
    }


def _estimate_cost_row(
    *,
    source: str,
    sample_count: int,
    fallback_level: str,
) -> dict[str, object]:
    return {
        "day": "2026-05-10",
        "symbol": "BTC-USDT",
        "regime": "normal",
        "event_type": "actual_fill",
        "notional_bucket": "1k-10k",
        "sample_count": sample_count,
        "fee_bps_p50": 1.0,
        "fee_bps_p75": 1.5,
        "fee_bps_p90": 2.0,
        "slippage_bps_p50": 2.0,
        "slippage_bps_p75": 3.0,
        "slippage_bps_p90": 4.0,
        "spread_bps_p50": 0.5,
        "spread_bps_p75": 0.75,
        "spread_bps_p90": 1.0,
        "total_cost_bps_p50": 3.5,
        "total_cost_bps_p75": 5.25,
        "total_cost_bps_p90": 7.0,
        "fallback_level": fallback_level,
        "source": source,
        "cost_source": source,
        "cost_model_version": "costs-2026-05-10",
        "created_at": datetime.now(UTC).isoformat(),
    }
