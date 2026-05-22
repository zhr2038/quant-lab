import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.expanded_universe import (
    build_and_publish_expanded_crypto_universe_shadow,
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

    assert rows["XRP-USDT"]["recommendation"] in {
        "candidate_replace_eth",
        "candidate_replace_bnb",
    }
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

    queue = read_parquet_dataset(lake_root / "gold" / "expanded_universe_promotion_queue")
    assert not queue.is_empty()
    assert "LIVE_SMALL_CANDIDATE" not in set(queue["promotion_state"].to_list())
    assert queue["max_live_notional_usdt"].max() == 0.0

    build_and_publish_alpha_discovery_board(lake_root, as_of_date="2026-05-24")
    board = read_parquet_dataset(lake_root / "gold" / "alpha_discovery_board")
    assert "expanded_paper" in set(board["universe_type"].drop_nulls().to_list())


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
