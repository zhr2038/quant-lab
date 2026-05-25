from __future__ import annotations

import csv
import io
import zipfile
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.alpha_factory import (
    ALPHA_FACTORY_CANDIDATES,
    alpha_factory_decision,
    build_alpha_factory_results,
    build_and_publish_alpha_factory,
    build_default_template_registry,
    publish_alpha_factory_template_registry,
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


def test_expanded_relative_strength_uses_decision_time_ranking_not_future_labels(tmp_path):
    lake = tmp_path / "lake"
    _write_relative_strength_anti_leakage_fixture(lake)

    build_and_publish_second_stage_alpha_factory(
        lake,
        as_of_date="2026-05-24",
        lookback_days=3,
    )

    samples = read_parquet_dataset(lake / "gold" / "second_stage_alpha_factory_sample")
    decision_samples = read_parquet_dataset(
        lake / "gold" / "expanded_relative_strength_decision_sample"
    )
    decision_ts = datetime(2026, 5, 23, 11, tzinfo=UTC)
    top1 = samples.filter(
        (pl.col("source_type") == "second_stage_expanded_relative_strength_decision_time")
        & (pl.col("strategy_candidate") == "v5.expanded_relative_strength_top1_shadow")
        & (pl.col("decision_ts") == decision_ts)
        & (pl.col("rank_lookback_hours") == 4)
        & (pl.col("horizon_hours") == 4)
        & (pl.col("top_k") == 1)
    )

    assert not top1.is_empty()
    assert set(top1["symbol"].to_list()) == {"NEAR-USDT"}
    assert "WLD-USDT" not in set(top1["symbol"].to_list())
    assert set(top1["anti_leakage_check"].to_list()) == {"pass"}
    assert set(top1["label_reason"].to_list()) == {
        "decision_time_relative_strength_shadow_only"
    }

    explain = decision_samples.filter(
        (pl.col("decision_ts") == decision_ts)
        & (pl.col("lookback_hours") == 4)
        & (pl.col("top_k") == 1)
        & (pl.col("label_horizon_hours") == 4)
    )
    assert not explain.is_empty()
    assert {"NEAR-USDT", "WLD-USDT"}.issubset(set(explain["symbol"].to_list()))
    near_row = explain.filter(pl.col("symbol") == "NEAR-USDT").to_dicts()[0]
    wld_row = explain.filter(pl.col("symbol") == "WLD-USDT").to_dicts()[0]
    assert near_row["selected"] is True
    assert near_row["selected_rank"] == 1
    assert near_row["lookback_return_bps"] > wld_row["lookback_return_bps"]
    assert wld_row["selected"] is False
    assert wld_row["future_net_bps"] is not None
    assert set(explain["anti_leakage_check"].to_list()) == {"pass"}


def test_futures_shadow_is_labeled_as_spot_inverse_proxy_and_capped_to_shadow(tmp_path):
    lake = tmp_path / "lake"
    _write_market(lake)

    build_and_publish_second_stage_alpha_factory(
        lake,
        as_of_date="2026-05-24",
        lookback_days=30,
    )

    samples = read_parquet_dataset(lake / "gold" / "second_stage_alpha_factory_sample")
    futures = samples.filter(
        pl.col("strategy_candidate") == "v5.futures_risk_off_hedge_proxy_shadow"
    )
    assert not futures.is_empty()
    assert set(futures["source_type"].drop_nulls().to_list()) == {
        "second_stage_futures_short_spot_inverse_proxy"
    }
    assert set(futures["futures_data_available"].drop_nulls().to_list()) == {False}
    assert set(futures["funding_available"].drop_nulls().to_list()) == {False}
    assert "spot close inverse proxy" in futures["mark_price_source"][0]
    assert "funding_not_observable" in futures["label_reason"][0]

    summary = read_parquet_dataset(lake / "gold" / "second_stage_alpha_factory_summary")
    futures_summary = summary.filter(
        pl.col("strategy_candidate").str.contains("futures_")
    )
    assert not futures_summary.is_empty()
    assert "PAPER_READY" not in set(futures_summary["decision"].drop_nulls().to_list())
    assert all(
        "futures_data_missing" in reasons
        and "funding_not_observable" in reasons
        for reasons in futures_summary["decision_reasons"].to_list()
    )


def test_exit_policy_review_outputs_actual_vs_alternative_exits(tmp_path):
    lake = tmp_path / "lake"
    _write_market(lake)
    _write_exit_policy_inputs(lake)

    build_and_publish_second_stage_alpha_factory(
        lake,
        as_of_date="2026-05-24",
        lookback_days=30,
    )

    samples = read_parquet_dataset(lake / "gold" / "exit_policy_review_sample")
    summary = read_parquet_dataset(lake / "gold" / "exit_policy_review_summary")
    btc = samples.filter(pl.col("strategy_id") == "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW")

    assert not btc.is_empty()
    row = btc.to_dicts()[0]
    assert row["actual_exit_net_bps"] == pytest.approx(-37.7)
    assert row["fixed_hold_24h_net_bps"] == pytest.approx(45.95)
    assert row["best_alternative_exit_policy"] == "fixed_hold_24h"
    assert row["delta_vs_actual_bps"] == pytest.approx(83.65)
    assert row["decision"] == "REVIEW_EXIT_POLICY"

    btc_summary = summary.filter(pl.col("strategy_id") == "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW")
    assert not btc_summary.is_empty()
    summary_row = btc_summary.to_dicts()[0]
    assert summary_row["stop_loss_too_early_count"] == 1
    assert summary_row["hold_24h_better_than_actual_count"] == 1
    assert summary_row["avg_delta_hold24h_vs_actual"] == pytest.approx(83.65)
    assert summary_row["decision"] == "REVIEW_EXIT_POLICY"
    assert summary_row["recommended_mode"] == "shadow"


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
    assert "v5.futures_risk_off_hedge_proxy_shadow" not in set(
        results["strategy_candidate"].to_list()
    )


def test_alpha_factory_refreshes_template_registry_publication_timestamp(tmp_path):
    lake = tmp_path / "lake"
    old = datetime(2026, 5, 24, tzinfo=UTC)
    new = datetime(2026, 5, 25, 12, tzinfo=UTC)
    registry = build_default_template_registry(old)
    write_parquet_dataset(registry, lake / "gold" / "alpha_factory_template_registry")

    published = publish_alpha_factory_template_registry(lake, generated_at=new)

    assert set(published["created_at"].to_list()) == {new}


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
        validation_metrics={
            "sample_count": 10,
            "complete_sample_count": 10,
            "avg_net_bps": 10.0,
            "p25_net_bps": -10.0,
            "win_rate": 0.6,
        },
        recent_7d_metrics={
            "sample_count": 5,
            "complete_sample_count": 5,
            "avg_net_bps": 1.0,
            "p25_net_bps": -10.0,
            "win_rate": 0.6,
        },
    )[0] == "PAPER_READY"
    assert alpha_factory_decision(
        sample_count=35,
        complete_sample_count=35,
        avg_net_bps=-10.0,
        p25_net_bps=-100.0,
        win_rate=0.4,
    )[0] == "KILL"


def test_alpha_factory_result_blocks_paper_when_validation_is_negative():
    summary = _alpha_factory_summary_frame(
        avg_net_bps=56.0,
        p25_net_bps=10.0,
        win_rate=0.7,
        sample_count=30,
        complete_sample_count=30,
    )
    samples = _alpha_factory_sample_frame(
        [100.0] * 21 + [-50.0] * 9,
        start=datetime(2026, 5, 1, tzinfo=UTC),
    )

    results = build_alpha_factory_results(
        summary,
        samples=samples,
        as_of_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 24, tzinfo=UTC),
    )

    row = results.to_dicts()[0]
    assert row["decision"] != "PAPER_READY"
    assert row["recommended_mode"] != "paper"
    assert "validation_avg_net_bps_non_positive" in row["decision_reasons"]
    assert "validation_metrics_json" in results.columns


def test_alpha_factory_result_downgrades_when_recent_7d_is_negative():
    values = [100.0] * 21 + [800.0] + [-10.0] * 8
    summary = _alpha_factory_summary_frame(
        avg_net_bps=sum(values) / len(values),
        p25_net_bps=5.0,
        win_rate=0.7,
        sample_count=len(values),
        complete_sample_count=len(values),
    )
    samples = _alpha_factory_sample_frame(
        values,
        start=datetime(2026, 4, 25, tzinfo=UTC),
    )

    results = build_alpha_factory_results(
        summary,
        samples=samples,
        as_of_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 24, tzinfo=UTC),
    )

    row = results.to_dicts()[0]
    assert row["decision"] == "KEEP_SHADOW"
    assert "recent_7d_avg_net_bps_negative" in row["decision_reasons"]
    assert row["recent_degradation_penalty"] > 0.0


def test_alpha_factory_blocks_paper_when_recent_samples_are_insufficient():
    summary = _alpha_factory_summary_frame(
        avg_net_bps=45.0,
        p25_net_bps=5.0,
        win_rate=0.7,
        sample_count=30,
        complete_sample_count=30,
        cost_source_mix='{"mixed_actual_proxy":30}',
    )
    samples = _alpha_factory_sample_frame(
        [30.0, 35.0, 40.0, 45.0],
        start=datetime(2026, 5, 21, tzinfo=UTC),
    )

    results = build_alpha_factory_results(
        summary,
        samples=samples,
        as_of_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 24, tzinfo=UTC),
    )

    row = results.to_dicts()[0]
    assert row["decision"] == "KEEP_SHADOW"
    assert row["recent_sample_sufficient"] is False
    assert "insufficient_recent_samples" in row["paper_ready_block_reasons"]
    assert "insufficient_recent_samples" in row["decision_reasons"]


def test_alpha_factory_blocks_paper_when_cost_quality_is_degraded():
    summary = _alpha_factory_summary_frame(
        avg_net_bps=45.0,
        p25_net_bps=5.0,
        win_rate=0.7,
        sample_count=30,
        complete_sample_count=30,
        cost_source_mix='{"local_estimate":30}',
    )
    samples = _alpha_factory_sample_frame(
        [30.0] * 30,
        start=datetime(2026, 4, 25, tzinfo=UTC),
    )

    results = build_alpha_factory_results(
        summary,
        samples=samples,
        as_of_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 24, tzinfo=UTC),
    )

    row = results.to_dicts()[0]
    assert row["decision"] == "KEEP_SHADOW"
    assert row["cost_quality_score"] < 0.5
    assert "cost_quality_not_paper_ready" in row["paper_ready_block_reasons"]


def test_alpha_factory_blocks_paper_when_mae_is_too_deep():
    summary = _alpha_factory_summary_frame(
        avg_net_bps=45.0,
        p25_net_bps=5.0,
        win_rate=0.7,
        sample_count=30,
        complete_sample_count=30,
        cost_source_mix='{"mixed_actual_proxy":30}',
    )
    samples = _alpha_factory_sample_frame(
        [30.0] * 30,
        start=datetime(2026, 4, 25, tzinfo=UTC),
        mae_bps=-200.0,
        mfe_bps=80.0,
    )

    results = build_alpha_factory_results(
        summary,
        samples=samples,
        as_of_date=datetime(2026, 5, 24, tzinfo=UTC).date(),
        generated_at=datetime(2026, 5, 24, tzinfo=UTC),
    )

    row = results.to_dicts()[0]
    assert row["decision"] == "KEEP_SHADOW"
    assert row["avg_mae_bps"] == -200.0
    assert row["avg_mfe_bps"] == 80.0
    assert "mae_too_deep_for_paper" in row["paper_ready_block_reasons"]


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
        assert "reports/expanded_relative_strength_decision_samples.csv" in names
        assert "reports/alpha_factory_template_registry.csv" in names
        assert "reports/alpha_factory_candidates.csv" in names
        assert "reports/alpha_factory_results.csv" in names
        assert "reports/alpha_factory_promotion_queue.csv" in names
        assert "reports/alpha_factory_daily.md" in names
        assert "reports/exit_policy_review.csv" in names
        assert "reports/exit_policy_summary.md" in names
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode(
                        "utf-8"
                    )
                )
            )
        )
        decision_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "reports/expanded_relative_strength_decision_samples.csv"
                    ).decode("utf-8")
                )
            )
        )

    second_stage = [
        row for row in advisory if row["strategy_candidate"] in SECOND_STAGE_CANDIDATES
    ]
    assert second_stage
    assert decision_rows
    assert "lookback_return_bps" in decision_rows[0]
    assert "future_net_bps" in decision_rows[0]
    assert all(float(row["max_live_notional_usdt"] or 0.0) == 0.0 for row in second_stage)
    assert "LIVE_SMALL_READY" not in {row["decision"] for row in second_stage}


def _alpha_factory_summary_frame(
    *,
    avg_net_bps: float,
    p25_net_bps: float,
    win_rate: float,
    sample_count: int,
    complete_sample_count: int,
    cost_source_mix: str = '{"public_spread_proxy":30}',
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-24",
                "strategy_candidate": "v5.expanded_relative_strength_top1_shadow",
                "candidate_name": "v5.expanded_relative_strength_top1_shadow",
                "symbol": "NEAR-USDT",
                "regime_state": "TREND_UP",
                "horizon_hours": 24,
                "sample_count": sample_count,
                "complete_sample_count": complete_sample_count,
                "avg_net_bps": avg_net_bps,
                "median_net_bps": avg_net_bps,
                "p25_net_bps": p25_net_bps,
                "win_rate": win_rate,
                "cost_source_mix": cost_source_mix,
                "start_ts": datetime(2026, 5, 1, tzinfo=UTC),
                "end_ts": datetime(2026, 5, 24, tzinfo=UTC),
                "source_dataset": "test",
            }
        ]
    )


def _alpha_factory_sample_frame(
    values: list[float],
    *,
    start: datetime,
    mae_bps: float | None = None,
    mfe_bps: float | None = None,
) -> pl.DataFrame:
    rows = []
    for index, net_bps in enumerate(values):
        ts = start + timedelta(days=index)
        rows.append(
            {
                "as_of_date": "2026-05-24",
                "strategy_candidate": "v5.expanded_relative_strength_top1_shadow",
                "symbol": "NEAR-USDT",
                "regime_state": "TREND_UP",
                "horizon_hours": 24,
                "decision_ts": ts,
                "label_ts": ts + timedelta(hours=24),
                "net_bps_after_cost": net_bps,
                "win": net_bps > 0.0,
                "mae_bps": mae_bps,
                "mfe_bps": mfe_bps,
                "label_status": "complete",
            }
        )
    return pl.DataFrame(rows)


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
                "avg_spread_bps": 1.0,
                "btc_correlation": 0.2,
                "volume_24h_usdt": 12_000_000.0,
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


def _write_relative_strength_anti_leakage_fixture(lake) -> None:
    start = datetime(2026, 5, 23, tzinfo=UTC)
    rows = []
    for symbol in ["NEAR-USDT", "WLD-USDT"]:
        for hour in range(24):
            ts = start + timedelta(hours=hour)
            if symbol == "NEAR-USDT":
                close = 100.0 + hour * 2.0
            elif hour <= 11:
                close = 100.0
            else:
                close = 220.0 + hour
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": ts,
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1000.0,
                    "quote_volume": close * 1000.0,
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
                    "as_of_date": "2026-05-24",
                    "generated_at": datetime(2026, 5, 24, tzinfo=UTC),
                    "symbol": symbol,
                    "symbol_quality_score": 90.0,
                    "quality_score": 90.0,
                    "avg_spread_bps": 1.0,
                    "btc_correlation": 0.2,
                    "volume_24h_usdt": 12_000_000.0,
                }
                for symbol in ["NEAR-USDT", "WLD-USDT"]
            ]
        ),
        lake / "gold" / "expanded_universe_quality",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-24",
                    "symbol": symbol,
                    "source": "public_spread_proxy",
                    "roundtrip_all_in_cost_bps": 5.0,
                    "created_at": datetime(2026, 5, 24, tzinfo=UTC),
                }
                for symbol in ["NEAR-USDT", "WLD-USDT"]
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
