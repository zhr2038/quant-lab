from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.alpha_factory import (
    ALPHA_FACTORY_CANDIDATES,
    alpha_factory_decision,
    build_and_publish_alpha_factory,
    build_default_template_registry,
)
from quant_lab.research.portfolio import build_and_publish_research_portfolio_status
from quant_lab.research.second_stage_alpha_factory import (
    SECOND_STAGE_CANDIDATES,
    build_and_publish_second_stage_alpha_factory,
)


def test_second_stage_alpha_factory_publishes_into_research_chain(tmp_path):
    lake = tmp_path / "lake"
    _write_market(lake)
    _write_expanded_labels(lake)
    _write_exit_policy_inputs(lake)

    result = build_and_publish_second_stage_alpha_factory(
        lake,
        as_of_date="2026-05-24",
        lookback_days=30,
    )

    assert result.sample_rows > 0
    assert result.summary_rows > 0
    assert "LIVE_SMALL_READY" not in result.decision_counts

    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    summary = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    sample_candidates = set(samples["strategy_candidate"].to_list())
    summary_candidates = set(summary["strategy_candidate"].to_list())

    assert SECOND_STAGE_CANDIDATES.issubset(sample_candidates)
    assert SECOND_STAGE_CANDIDATES.issubset(summary_candidates)
    assert "LIVE_SMALL_READY" not in set(summary["decision"].drop_nulls().to_list())

    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-24")
    build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-24")

    board = read_parquet_dataset(lake / "gold" / "alpha_discovery_board")
    portfolio = read_parquet_dataset(lake / "gold" / "research_portfolio_status")
    assert SECOND_STAGE_CANDIDATES & set(board["strategy_candidate"].to_list())
    assert SECOND_STAGE_CANDIDATES & set(portfolio["strategy_candidate"].to_list())


def test_alpha_factory_outputs_candidates_results_and_queue_without_live(tmp_path):
    lake = tmp_path / "lake"
    _write_market(lake)
    _write_expanded_labels(lake)
    _write_exit_policy_inputs(lake)
    _write_alt_impulse_evidence(lake)

    result = build_and_publish_alpha_factory(
        lake,
        as_of_date="2026-05-24",
        lookback_days=30,
        max_candidates=200,
    )

    assert result.candidate_rows > 0
    assert result.candidate_rows <= 200
    assert result.result_rows == result.candidate_rows
    assert result.promotion_rows == result.candidate_rows
    assert result.template_registry_rows == 5
    assert "LIVE_SMALL_READY" not in result.decision_counts

    registry = read_parquet_dataset(lake / "gold" / "alpha_factory_template_registry")
    candidates = read_parquet_dataset(lake / "gold" / "alpha_factory_candidate")
    results = read_parquet_dataset(lake / "gold" / "alpha_factory_result")
    promotion = read_parquet_dataset(lake / "gold" / "alpha_factory_promotion_queue")
    strategy_evidence = read_parquet_dataset(lake / "gold" / "strategy_evidence")

    assert registry.height == 5
    assert set(registry["safety_mode"].to_list()) == {"paper_shadow_only"}
    assert "expanded_relative_strength_v1" in set(registry["template_id"].to_list())
    expanded_space = registry.filter(
        pl.col("template_id") == "expanded_relative_strength_v1"
    )["parameter_space_json"][0]
    assert "lookback_hours" in expanded_space
    assert "max_live_notional_usdt" in expanded_space
    assert set(results["strategy_candidate"].to_list()).issubset(ALPHA_FACTORY_CANDIDATES)
    assert "v5.alt_impulse_shadow" in set(results["strategy_candidate"].to_list())
    assert set(candidates["max_live_notional_usdt"].to_list()) == {0.0}
    assert set(results["max_live_notional_usdt"].to_list()) == {0.0}
    assert set(promotion["max_live_notional_usdt"].to_list()) == {0.0}
    assert "LIVE_SMALL_READY" not in set(results["decision"].drop_nulls().to_list())
    assert "LIVE_SMALL_READY" not in set(promotion["promotion_state"].drop_nulls().to_list())
    assert "v5.alt_impulse_shadow" in set(strategy_evidence["strategy_candidate"].to_list())


def test_alpha_factory_reads_template_registry_enabled_flags(tmp_path):
    lake = tmp_path / "lake"
    _write_market(lake)
    _write_expanded_labels(lake)
    _write_exit_policy_inputs(lake)
    _write_alt_impulse_evidence(lake)

    registry = build_default_template_registry(datetime(2026, 5, 24, tzinfo=UTC))
    registry = registry.with_columns(
        pl.when(pl.col("template_id") == "futures_hedge_shadow_v1")
        .then(pl.lit(False))
        .otherwise(pl.col("enabled"))
        .alias("enabled")
    )
    write_parquet_dataset(registry, lake / "gold" / "alpha_factory_template_registry")

    build_and_publish_alpha_factory(lake, as_of_date="2026-05-24")

    results = read_parquet_dataset(lake / "gold" / "alpha_factory_result")
    assert "futures_hedge_shadow" not in set(results["template_name"].to_list())
    assert "v5.futures_risk_off_hedge_shadow" not in set(
        results["strategy_candidate"].to_list()
    )


def test_alpha_factory_decision_ladder_is_shadow_paper_only():
    assert alpha_factory_decision(
        sample_count=5,
        complete_sample_count=2,
        avg_net_bps=100.0,
        p25_net_bps=10.0,
        win_rate=0.9,
    )[0] == "RESEARCH"
    assert alpha_factory_decision(
        sample_count=12,
        complete_sample_count=12,
        avg_net_bps=10.0,
        p25_net_bps=-80.0,
        win_rate=0.5,
    )[0] == "KEEP_SHADOW"
    assert alpha_factory_decision(
        sample_count=35,
        complete_sample_count=35,
        avg_net_bps=20.0,
        p25_net_bps=-20.0,
        win_rate=0.6,
    )[0] == "PAPER_READY"
    assert alpha_factory_decision(
        sample_count=35,
        complete_sample_count=35,
        avg_net_bps=-10.0,
        p25_net_bps=-100.0,
        win_rate=0.4,
    )[0] == "KILL"


def test_daily_export_includes_alpha_factory_reports_and_advisory_is_not_live(tmp_path):
    lake = tmp_path / "lake"
    out_dir = tmp_path / "exports"
    _write_market(lake)
    _write_expanded_labels(lake)
    _write_exit_policy_inputs(lake)
    _write_alt_impulse_evidence(lake)
    build_and_publish_alpha_factory(lake, as_of_date="2026-05-24")
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-24")
    build_and_publish_research_portfolio_status(lake, as_of_date="2026-05-24")

    result = export_daily_pack(
        export_date="2026-05-24",
        lake_root=lake,
        out_dir=out_dir,
        pre_export_v5_refresh=False,
        refresh_risk_permission=False,
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/second_stage_alpha_factory_summary.csv" in names
        assert "reports/second_stage_alpha_factory_samples.csv" in names
        assert "reports/alpha_factory_template_registry.csv" in names
        assert "reports/alpha_factory_candidates.csv" in names
        assert "reports/alpha_factory_results.csv" in names
        assert "reports/alpha_factory_promotion_queue.csv" in names
        assert "reports/alpha_factory_daily.md" in names
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode(
                        "utf-8"
                    )
                )
            )
        )

    second_stage = [
        row for row in advisory if row["strategy_candidate"] in SECOND_STAGE_CANDIDATES
    ]
    assert second_stage
    assert all(float(row["max_live_notional_usdt"] or 0.0) == 0.0 for row in second_stage)
    assert "LIVE_SMALL_READY" not in {row["decision"] for row in second_stage}


def _write_market(lake) -> None:
    start = datetime(2026, 5, 21, tzinfo=UTC)
    rows = []
    for symbol, base, drift in [
        ("BTC-USDT", 100_000.0, -25.0),
        ("ETH-USDT", 3_000.0, 3.5),
        ("SOL-USDT", 170.0, 0.8),
        ("BNB-USDT", 650.0, 0.2),
        ("NEAR-USDT", 5.0, 0.05),
        ("WLD-USDT", 2.0, 0.02),
        ("OKB-USDT", 50.0, 0.15),
    ]:
        for hour in range(80):
            ts = start + timedelta(hours=hour)
            close = base + drift * hour
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": ts,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0 + hour,
                    "quote_volume": close * (1000.0 + hour),
                    "source": "test",
                    "ingest_ts": ts,
                    "is_closed": True,
                }
            )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "market_bar")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-24",
                    "symbol": symbol,
                    "source": "public_spread_proxy",
                    "total_cost_bps_p75": 4.0,
                    "created_at": datetime(2026, 5, 24, tzinfo=UTC),
                }
                for symbol in ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
            ]
        ),
        lake / "gold" / "cost_bucket_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-24",
                    "current_regime": "TREND_UP",
                    "created_at": datetime(2026, 5, 24, tzinfo=UTC),
                }
            ]
        ),
        lake / "gold" / "market_regime_daily",
    )


def _write_expanded_labels(lake) -> None:
    start = datetime(2026, 5, 23, tzinfo=UTC)
    label_rows = []
    quality_rows = []
    for rank, symbol in enumerate(["NEAR-USDT", "WLD-USDT", "OKB-USDT"], start=1):
        quality_rows.append(
            {
                "as_of_date": "2026-05-24",
                "generated_at": datetime(2026, 5, 24, tzinfo=UTC),
                "symbol": symbol,
                "symbol_quality_score": 90.0 - rank,
                "quality_score": 90.0 - rank,
            }
        )
        for index in range(4):
            for horizon in [4, 8, 12, 24, 48]:
                ts = start + timedelta(hours=index)
                net = 80.0 - rank * 5.0 + horizon * 0.1
                label_rows.append(
                    {
                        "candidate_id": f"{symbol}-{index}",
                        "ts_utc": ts,
                        "decision_ts": ts,
                        "label_ts": ts + timedelta(hours=horizon),
                        "generated_at": datetime(2026, 5, 24, tzinfo=UTC),
                        "symbol": symbol,
                        "universe_type": "expanded_paper",
                        "strategy_candidate": "Alpha6Factor",
                        "horizon_hours": horizon,
                        "entry_close": 1.0,
                        "label_close": 1.01,
                        "gross_bps": net + 4.0,
                        "net_bps_after_cost": net,
                        "win": True,
                        "mfe_bps": net + 10.0,
                        "mae_bps": -10.0,
                        "label_status": "complete",
                        "cost_bps": 4.0,
                        "cost_source": "public_spread_proxy",
                        "source": "test",
                    }
                )
    write_parquet_dataset(
        pl.DataFrame(quality_rows),
        lake / "gold" / "expanded_universe_quality",
    )
    write_parquet_dataset(
        pl.DataFrame(label_rows),
        lake / "gold" / "expanded_universe_candidate_label",
    )


def _write_exit_policy_inputs(lake) -> None:
    created = datetime(2026, 5, 24, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "entry_ts": datetime(2026, 5, 23, 1, tzinfo=UTC),
                    "run_id": "btc-probe-1",
                    "entry_px": 100_000.0,
                    "exit_px": 99_500.0,
                    "actual_exit_net_bps": -37.7,
                    "would_hold_4h_net_bps": -5.0,
                    "would_hold_8h_net_bps": 10.0,
                    "would_hold_12h_net_bps": 20.0,
                    "would_hold_24h_net_bps": 45.95,
                    "would_hold_48h_net_bps": 30.0,
                    "mae_bps": -80.0,
                    "mfe_bps": 70.0,
                    "exit_reason": "probe_stop_loss",
                    "created_at": created,
                }
            ]
        ),
        lake / "gold" / "btc_probe_exit_policy_review",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-24",
                    "strategy_id": "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                    "proposal_id": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                    "run_id": "eth-paper-1",
                    "ts_utc": datetime(2026, 5, 23, 2, tzinfo=UTC),
                    "symbol": "ETH-USDT",
                    "cost_source": "mixed_actual_proxy",
                    "paper_pnl_bps_4h": -10.0,
                    "paper_pnl_bps_8h": 15.0,
                    "paper_pnl_bps_12h": 20.0,
                    "paper_pnl_bps_24h": 25.0,
                    "paper_pnl_bps_48h": 35.0,
                },
                {
                    "as_of_date": "2026-05-24",
                    "strategy_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
                    "proposal_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
                    "run_id": "sol-paper-1",
                    "ts_utc": datetime(2026, 5, 23, 2, tzinfo=UTC),
                    "symbol": "SOL-USDT",
                    "cost_source": "mixed_actual_proxy",
                    "paper_pnl_bps_4h": -20.0,
                    "paper_pnl_bps_8h": -5.0,
                    "paper_pnl_bps_12h": 12.0,
                    "paper_pnl_bps_24h": 30.0,
                    "paper_pnl_bps_48h": 40.0,
                },
            ]
        ),
        lake / "gold" / "paper_strategy_runs",
    )


def _write_alt_impulse_evidence(lake) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "evidence_version": "strategy_evidence.v0.1",
                    "as_of_date": "2026-05-24",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "candidate_name": "v5.alt_impulse_shadow",
                    "symbol": "SOL-USDT",
                    "regime_state": "ALT_IMPULSE",
                    "horizon_hours": 24,
                    "sample_count": 35,
                    "complete_sample_count": 35,
                    "avg_net_bps": 60.0,
                    "median_net_bps": 55.0,
                    "p25_net_bps": -20.0,
                    "win_rate": 0.62,
                    "cost_source_mix": '{"public_spread_proxy":35}',
                    "decision": "KEEP_SHADOW",
                    "decision_reasons": '["regime_dependent"]',
                    "start_ts": datetime(2026, 5, 20, tzinfo=UTC),
                    "end_ts": datetime(2026, 5, 24, tzinfo=UTC),
                    "created_at": datetime(2026, 5, 24, tzinfo=UTC),
                    "source": "test",
                }
            ]
        ),
        lake / "gold" / "strategy_evidence",
    )
