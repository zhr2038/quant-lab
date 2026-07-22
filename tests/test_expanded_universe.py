import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import (
    read_parquet_dataset,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.expanded_universe import (
    STRATEGY_EVIDENCE_UPSERT_KEYS,
    build_and_publish_expanded_crypto_universe_shadow,
    build_expanded_universe_candidate_maturity,
    build_expanded_universe_watchlist,
    build_symbol_quality_score,
)
from quant_lab.web import readers


def test_expanded_universe_filters_and_recommends_replacements(tmp_path):
    lake_root = tmp_path / "lake"
    _write_market(lake_root)
    _write_orderbooks(lake_root)
    _write_strategy_evidence(lake_root)

    result = build_and_publish_expanded_crypto_universe_shadow(
        lake_root,
        as_of_date="2026-05-21",
        min_quote_volume_24h=100_000,
        max_spread_bps=20,
        min_coverage_bars=4,
    )

    assert result.quality_rows >= 5
    quality = read_parquet_dataset(lake_root / "gold" / "symbol_quality_score")
    rows = {row["symbol"]: row for row in quality.to_dicts()}

    assert rows["XRP-USDT"]["recommendation"] == "candidate_for_expanded_paper_universe"
    assert not any(
        str(row["recommendation"]).startswith("candidate_replace_")
        for row in quality.to_dicts()
    )
    assert "high_risk_meme" in json.loads(rows["DOGE-USDT"]["blocking_reasons"])
    assert "low_quote_volume" in json.loads(rows["LOW-USDT"]["blocking_reasons"])

    shadow = read_parquet_dataset(lake_root / "gold" / "expanded_crypto_universe_shadow")
    assert "XRP-USDT" in set(shadow["symbol"].to_list())
    assert all(row["min_shadow_days_required"] == 7 for row in shadow.to_dicts())

    recommendations = read_parquet_dataset(
        lake_root / "gold" / "expanded_crypto_recommendations"
    )
    latest = recommendations.to_dicts()[0]
    assert latest["schema_version"] == "expanded_crypto_recommendations.v0.1"
    assert latest["min_stable_output_days"] == 7
    assert "XRP-USDT" in latest["top_symbols_json"]
    assert json.loads(latest["candidate_replace_eth_json"]) == []
    assert json.loads(latest["candidate_replace_bnb_json"]) == []


def test_symbol_quality_uses_btc_correlation_and_spread_filters():
    market = _market_frame(["BTC-USDT", "ALT-USDT"], hours=6, quote_volume=1_000_000)
    orderbook = _orderbook_frame({"BTC-USDT": 10.0, "ALT-USDT": 50.0})
    evidence = pl.DataFrame(
        [
            _evidence_row("ALT-USDT", "v5.f4_volume_expansion_entry", 24, 20.0, 0.6),
        ]
    )

    quality = build_symbol_quality_score(
        market_bars=market,
        orderbook_snapshots=orderbook,
        strategy_evidence=evidence,
        pullback_by_symbol=pl.DataFrame(),
        late_entry_by_symbol=pl.DataFrame(),
        as_of_date=datetime(2026, 5, 21, tzinfo=UTC).date(),
        min_quote_volume_24h=100_000,
        max_spread_bps=20,
        min_coverage_bars=4,
    )

    alt = [row for row in quality.to_dicts() if row["symbol"] == "ALT-USDT"][0]
    assert "high_spread" in json.loads(alt["blocking_reasons"])
    assert alt["recommendation"] == "reject_high_spread"
    assert alt["btc_correlation"] is not None


def test_symbol_quality_uses_rest_candidate_spread_when_orderbook_missing():
    market = _market_frame(["BTC-USDT", "XRP-USDT"], hours=6, quote_volume=1_000)
    spot_candidates = pl.DataFrame(
        [
            {
                "generated_at": datetime(2026, 5, 21, tzinfo=UTC),
                "rank": 1,
                "symbol": "XRP-USDT",
                "quote_volume_24h": 2_000_000.0,
                "spread_bps": 4.0,
            }
        ]
    )

    quality = build_symbol_quality_score(
        market_bars=market,
        orderbook_snapshots=pl.DataFrame(),
        strategy_evidence=pl.DataFrame(),
        pullback_by_symbol=pl.DataFrame(),
        late_entry_by_symbol=pl.DataFrame(),
        spot_universe_candidates=spot_candidates,
        as_of_date=datetime(2026, 5, 21, tzinfo=UTC).date(),
        min_quote_volume_24h=100_000,
        max_spread_bps=20,
        min_coverage_bars=4,
    )

    xrp = [row for row in quality.to_dicts() if row["symbol"] == "XRP-USDT"][0]
    assert xrp["avg_spread_bps"] == 4.0
    assert xrp["quote_volume_24h"] == 2_000_000.0
    assert "spread_not_observed" not in json.loads(xrp["blocking_reasons"])


def test_web_strategy_page_reads_expanded_universe(tmp_path):
    lake_root = tmp_path / "lake"
    _write_market(lake_root)
    _write_orderbooks(lake_root)
    _write_strategy_evidence(lake_root)
    build_and_publish_expanded_crypto_universe_shadow(
        lake_root,
        as_of_date="2026-05-21",
        min_quote_volume_24h=100_000,
        max_spread_bps=20,
        min_coverage_bars=4,
    )

    summary = readers.alpha_gate_summary(lake_root)

    assert not summary["expanded_crypto_universe_shadow"].is_empty()
    assert not summary["symbol_quality_score"].is_empty()
    assert not summary["expanded_crypto_recommendations"].is_empty()
    assert "expanded_universe_candidate" in summary
    assert "expanded_universe_watchlist" in summary
    assert "expanded_universe_candidate_maturity" in summary


def test_expanded_universe_automation_builds_events_labels_and_promotion(tmp_path):
    lake_root = tmp_path / "lake"
    seed_symbols = [
        "TRX-USDT",
        "HYPE-USDT",
        "SUI-USDT",
        "XAUT-USDT",
        "PAXG-USDT",
        "ZEC-USDT",
    ]
    write_parquet_dataset(
        _market_frame(["BTC-USDT", *seed_symbols], hours=96, quote_volume=2_000_000),
        lake_root / "silver" / "market_bar",
    )
    write_parquet_dataset(
        _orderbook_frame({symbol: 5.0 for symbol in ["BTC-USDT", *seed_symbols]}),
        lake_root / "silver" / "orderbook_snapshot",
    )
    strategy_evidence_root = lake_root / "gold" / "strategy_evidence"
    strategy_evidence_root.mkdir(parents=True)
    candidate_generation_sidecar = (
        strategy_evidence_root / "_v5_candidate_evidence_generation.json"
    )
    sidecar_bytes = b'{"generation_id":"v5-candidate-generation"}\n'
    candidate_generation_sidecar.write_bytes(sidecar_bytes)

    result = build_and_publish_expanded_crypto_universe_shadow(
        lake_root,
        as_of_date="2026-05-24",
        min_quote_volume_24h=100_000,
        max_spread_bps=20,
        min_coverage_bars=24,
    )

    assert result.candidate_rows >= len(seed_symbols)
    assert result.event_rows >= len(seed_symbols)
    assert result.label_rows >= len(seed_symbols) * 6
    candidates = read_parquet_dataset(lake_root / "gold" / "expanded_universe_candidate")
    assert set(seed_symbols).issubset(set(candidates["symbol"].to_list()))

    events = read_parquet_dataset(lake_root / "gold" / "expanded_universe_candidate_event")
    assert set(seed_symbols).issubset(set(events["symbol"].to_list()))
    assert set(events["universe_type"].to_list()) == {"expanded_paper"}

    labels = read_parquet_dataset(lake_root / "gold" / "expanded_universe_candidate_label")
    assert {4, 8, 12, 24, 48, 72}.issubset(set(labels["horizon_hours"].to_list()))

    evidence = read_parquet_dataset(lake_root / "gold" / "strategy_evidence")
    expanded = evidence.filter(pl.col("universe_type") == "expanded_paper")
    assert not expanded.is_empty()
    assert candidate_generation_sidecar.read_bytes() == sidecar_bytes

    queue = read_parquet_dataset(lake_root / "gold" / "expanded_universe_promotion_queue")
    assert not queue.is_empty()
    assert "LIVE_SMALL_CANDIDATE" not in set(queue["promotion_state"].to_list())
    assert queue["max_live_notional_usdt"].max() == 0.0
    watchlist = read_parquet_dataset(lake_root / "gold" / "expanded_universe_watchlist")
    assert {
        "quality_watchlist",
        "outcome_watchlist",
        "reject_list",
    }.issubset(set(watchlist["watchlist_type"].to_list()))
    maturity = read_parquet_dataset(lake_root / "gold" / "expanded_universe_candidate_maturity")
    assert not maturity.is_empty()

    build_and_publish_alpha_discovery_board(lake_root, as_of_date="2026-05-24")
    board = read_parquet_dataset(lake_root / "gold" / "alpha_discovery_board")
    assert "expanded_paper" in set(board["universe_type"].drop_nulls().to_list())


def test_expanded_evidence_upsert_keeps_candidate_managed_scope(tmp_path):
    dataset = tmp_path / "lake" / "gold" / "strategy_evidence"
    shared_identity = {
        "strategy": "v5",
        "as_of_date": "2026-05-24",
        "strategy_candidate": "Alpha6Factor",
        "symbol": "BTC-USDT",
        "regime_state": "expanded_universe",
        "horizon_hours": 4,
    }
    managed = {
        **shared_identity,
        "source": "research.strategy_evidence.v0.1",
        "evidence_version": "strategy-evidence-v0.1",
        "decision": "PAPER_READY",
    }
    expanded = {
        **shared_identity,
        "source": "expanded_crypto_universe.v1",
        "evidence_version": "expanded-crypto-universe-v1",
        "decision": "KEEP_SHADOW",
    }
    write_parquet_dataset(pl.DataFrame([managed]), dataset)
    sidecar = dataset / "_v5_candidate_evidence_generation.json"
    sidecar_bytes = b'{"generation_id":"v5-candidate-generation"}\n'
    sidecar.write_bytes(sidecar_bytes)

    rows = upsert_parquet_dataset(
        pl.DataFrame([expanded]),
        dataset,
        key_columns=STRATEGY_EVIDENCE_UPSERT_KEYS,
        preserve_files=(sidecar.name,),
    )

    assert rows == 2
    assert sidecar.read_bytes() == sidecar_bytes
    evidence = read_parquet_dataset(dataset)
    assert set(evidence["source"].to_list()) == {
        "research.strategy_evidence.v0.1",
        "expanded_crypto_universe.v1",
    }


def test_expanded_universe_maturity_rules_and_watchlist():
    generated_at = datetime(2026, 5, 24, tzinfo=UTC)
    evidence = pl.DataFrame(
        [
            _expanded_evidence_row("NEAR-USDT", "Alpha6Factor", 4, 12, 12, 40.0, 0.60, -20.0),
            _expanded_evidence_row("NEAR-USDT", "Alpha6Factor", 8, 12, 12, 25.0, 0.58, -30.0),
            _expanded_evidence_row("WLD-USDT", "Alpha6Factor", 4, 8, 8, 80.0, 0.75, -10.0),
            _expanded_evidence_row("OKB-USDT", "Alpha6Factor", 4, 35, 35, 65.0, 0.62, -25.0),
        ]
    )
    maturity = build_expanded_universe_candidate_maturity(
        evidence,
        as_of_date=generated_at.date(),
        generated_at=generated_at,
    )
    rows = {
        (row["symbol"], row["strategy_candidate"]): row for row in maturity.to_dicts()
    }
    assert rows[("NEAR-USDT", "Alpha6Factor")]["maturity_state"] == "KEEP_SHADOW"
    assert rows[("WLD-USDT", "Alpha6Factor")]["maturity_state"] == "RESEARCH"
    assert rows[("OKB-USDT", "Alpha6Factor")]["maturity_state"] == "PAPER_READY"

    quality = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "generated_at": generated_at,
                "schema_version": "expanded_crypto_universe_shadow.v0.1",
                "symbol": "HYPE-USDT",
                "quote_volume_24h": 1_000_000.0,
                "avg_spread_bps": 5.0,
                "min_notional_ok": True,
                "data_coverage": 1.0,
                "avg_24h_net_bps": None,
                "avg_48h_net_bps": None,
                "win_rate_24h": None,
                "win_rate_48h": None,
                "f3_dominant_negative_score": 0.0,
                "f4_confirmed_win_rate": None,
                "f5_confirmed_win_rate": None,
                "pullback_shadow_avg_24h": None,
                "late_chase_loss_rate": 0.0,
                "negative_expectancy_bps": 0.0,
                "btc_correlation": 0.1,
                "quality_score": 75.0,
                "recommendation": "candidate_for_expanded_paper_universe",
                "blocking_reasons": "[]",
                "source": "fixture",
            }
        ]
    )
    watchlist = build_expanded_universe_watchlist(
        quality,
        maturity,
        as_of_date=generated_at.date(),
        generated_at=generated_at,
    )
    assert "TRX-USDT" in set(watchlist["symbol"].to_list())
    assert "NEAR-USDT" in set(watchlist["symbol"].to_list())
    rows = {
        (row["watchlist_type"], row["symbol"]): row
        for row in watchlist.to_dicts()
    }
    quality_symbols = _watchlist_symbols(watchlist, "quality_watchlist")
    outcome_symbols = _watchlist_symbols(watchlist, "outcome_watchlist")
    reject_symbols = _watchlist_symbols(watchlist, "reject_list")
    assert quality_symbols == {
        "TRX-USDT",
        "XAUT-USDT",
    }
    assert outcome_symbols == {
        "NEAR-USDT",
        "WLD-USDT",
        "OKB-USDT",
    }
    assert reject_symbols == {
        "HYPE-USDT",
        "SUI-USDT",
        "ZEC-USDT",
        "FIL-USDT",
    }
    assert (
        rows[("reject_list", "HYPE-USDT")]["recommendation"]
        == "reject_low_priority_current_weak"
    )
    assert not any(
        str(row["recommendation"]).startswith("candidate_replace_")
        for row in watchlist.to_dicts()
    )


def _watchlist_symbols(watchlist: pl.DataFrame, watchlist_type: str) -> set[str]:
    return set(
        row["symbol"]
        for row in watchlist.filter(pl.col("watchlist_type") == watchlist_type).to_dicts()
    )


def _write_market(lake_root: Path) -> None:
    write_parquet_dataset(
        _market_frame(
            ["BTC-USDT", "ETH-USDT", "BNB-USDT", "XRP-USDT", "ADA-USDT", "DOGE-USDT"],
            hours=8,
            quote_volume=1_000_000,
        ).vstack(_market_frame(["LOW-USDT"], hours=8, quote_volume=10_000, start_price=0.005)),
        lake_root / "silver" / "market_bar",
    )


def _market_frame(
    symbols: list[str],
    *,
    hours: int,
    quote_volume: float,
    start_price: float = 100.0,
) -> pl.DataFrame:
    start = datetime(2026, 5, 20, tzinfo=UTC)
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        base_price = start_price + symbol_index * 10
        for hour in range(hours):
            ts = start + timedelta(hours=hour)
            close = base_price * (1 + hour * (0.001 + symbol_index * 0.0002))
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": ts,
                    "open": close * 0.999,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": quote_volume / close,
                    "quote_volume": quote_volume,
                    "source": "fixture",
                    "ingest_ts": ts,
                    "is_closed": True,
                }
            )
    return pl.DataFrame(rows)


def _write_orderbooks(lake_root: Path) -> None:
    write_parquet_dataset(
        _orderbook_frame(
            {
                "BTC-USDT": 5.0,
                "ETH-USDT": 6.0,
                "BNB-USDT": 6.0,
                "XRP-USDT": 4.0,
                "ADA-USDT": 8.0,
                "DOGE-USDT": 4.0,
                "LOW-USDT": 4.0,
            }
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )


def _orderbook_frame(spreads_bps: dict[str, float]) -> pl.DataFrame:
    ts = datetime(2026, 5, 20, 7, tzinfo=UTC)
    rows = []
    for index, (symbol, spread_bps) in enumerate(spreads_bps.items()):
        mid = 100.0 + index * 10
        half = spread_bps / 20_000
        bid = mid * (1 - half)
        ask = mid * (1 + half)
        rows.append(
            {
                "venue": "okx",
                "symbol": symbol,
                "channel": "books5",
                "ts": ts,
                "asks_json": json.dumps([[str(ask), "1"]]),
                "bids_json": json.dumps([[str(bid), "1"]]),
                "source": "fixture",
                "ingest_ts": ts,
                "raw_json": "{}",
            }
        )
    return pl.DataFrame(rows)


def _write_strategy_evidence(lake_root: Path) -> None:
    rows = [
        _evidence_row("XRP-USDT", "v5.f4_volume_expansion_entry", 24, 90.0, 0.72),
        _evidence_row("XRP-USDT", "v5.f4_volume_expansion_entry", 48, 120.0, 0.70),
        _evidence_row("ADA-USDT", "v5.f5_confirmed_entry", 24, 40.0, 0.58),
        _evidence_row("ETH-USDT", "v5.f3_dominant_entry", 24, -80.0, 0.35),
        _evidence_row("BNB-USDT", "v5.f3_dominant_entry", 24, -60.0, 0.40),
    ]
    write_parquet_dataset(
        pl.DataFrame(rows),
        lake_root / "gold" / "strategy_evidence",
    )


def _evidence_row(
    symbol: str,
    candidate: str,
    horizon: int,
    avg_net_bps: float,
    win_rate: float,
) -> dict[str, object]:
    return {
        "strategy": "v5",
        "evidence_version": "test",
        "as_of_date": "2026-05-21",
        "strategy_candidate": candidate,
        "candidate_name": candidate,
        "symbol": symbol,
        "regime_state": "all",
        "horizon_hours": horizon,
        "sample_count": 20,
        "complete_sample_count": 20,
        "avg_net_bps": avg_net_bps,
        "median_net_bps": avg_net_bps,
        "p25_net_bps": avg_net_bps - 20,
        "win_rate": win_rate,
        "cost_source_mix": json.dumps({"public_spread_proxy": 20}),
        "decision": "KEEP_SHADOW",
        "decision_reasons": "fixture",
        "start_ts": datetime(2026, 5, 1, tzinfo=UTC),
        "end_ts": datetime(2026, 5, 21, tzinfo=UTC),
        "created_at": datetime(2026, 5, 21, tzinfo=UTC),
        "source": "fixture",
    }


def _expanded_evidence_row(
    symbol: str,
    candidate: str,
    horizon: int,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float,
    win_rate: float,
    p25_net_bps: float,
) -> dict[str, object]:
    return {
        "strategy": "v5",
        "evidence_version": "test",
        "as_of_date": "2026-05-24",
        "strategy_candidate": candidate,
        "candidate_name": candidate,
        "source_type": "expanded_universe_candidate_label",
        "symbol": symbol,
        "universe_type": "expanded_paper",
        "replacement_target_candidate": "",
        "expansion_state": "RESEARCH",
        "regime_state": "expanded_universe",
        "horizon_hours": horizon,
        "sample_count": sample_count,
        "complete_sample_count": complete_sample_count,
        "avg_net_bps": avg_net_bps,
        "median_net_bps": avg_net_bps,
        "p25_net_bps": p25_net_bps,
        "win_rate": win_rate,
        "cost_source_mix": json.dumps({"public_spread_proxy": complete_sample_count}),
        "decision": "KEEP_SHADOW",
        "decision_reasons": "fixture",
        "start_ts": datetime(2026, 5, 1, tzinfo=UTC),
        "end_ts": datetime(2026, 5, 24, tzinfo=UTC),
        "created_at": datetime(2026, 5, 24, tzinfo=UTC),
        "source": "fixture",
    }
