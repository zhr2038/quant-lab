import csv
import io
import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.export.daily import export_daily_pack
from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
from quant_lab.research.strategy_evidence import (
    build_and_publish_strategy_evidence,
    normalize_strategy_evidence_decisions,
    strategy_evidence_decision_ladder,
)


def test_strategy_evidence_ladder_requires_complete_sample_floor():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=12,
        complete_sample_count=2,
        avg_net_bps=-25.0,
        p25_net_bps=-40.0,
        win_rate=0.2,
    )

    assert decision == "RESEARCH_ONLY"
    assert "insufficient_complete_samples" in reasons

    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=8,
        complete_sample_count=8,
        avg_net_bps=25.0,
        p25_net_bps=10.0,
        win_rate=1.0,
    )
    assert decision == "RESEARCH_ONLY"
    assert "insufficient_total_samples" in reasons


def test_strategy_evidence_ladder_kills_only_after_complete_sample_floor():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=12,
        complete_sample_count=12,
        avg_net_bps=-25.0,
        p25_net_bps=-40.0,
        win_rate=0.2,
    )

    assert decision == "KILL"
    assert "non_positive_after_cost_edge" in reasons
    assert "win_rate_below_threshold" in reasons


def test_strategy_evidence_ladder_blocks_paper_ready_on_global_default_cost():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=72,
        complete_sample_count=72,
        avg_net_bps=25.0,
        p25_net_bps=1.0,
        win_rate=0.72,
        cost_source_mix={"global_default": 72},
    )

    assert decision == "KEEP_SHADOW"
    assert "cost_source_not_trusted" in reasons


def test_strategy_evidence_ladder_allows_paper_ready_on_mixed_actual_proxy():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=72,
        complete_sample_count=72,
        avg_net_bps=25.0,
        p25_net_bps=1.0,
        win_rate=0.72,
        cost_source_mix={"mixed_actual_proxy": 72},
    )

    assert decision == "PAPER_READY"
    assert "paper_ready_thresholds_met" in reasons


def test_strategy_evidence_ladder_allows_paper_not_live_on_public_spread_proxy():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=72,
        complete_sample_count=72,
        avg_net_bps=25.0,
        p25_net_bps=1.0,
        win_rate=0.72,
        paper_days=20,
        paper_slippage_coverage=0.9,
        cost_source_mix={"public_spread_proxy": 72},
    )

    assert decision == "PAPER_READY"
    assert "cost_source_not_trusted" in reasons


def test_strategy_evidence_ladder_requires_actual_or_mixed_for_live_small_ready():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=72,
        complete_sample_count=72,
        avg_net_bps=25.0,
        p25_net_bps=1.0,
        win_rate=0.72,
        paper_days=20,
        paper_slippage_coverage=0.9,
        cost_source_mix={"mixed_actual_proxy": 72},
    )

    assert decision == "LIVE_SMALL_READY"
    assert reasons == ["live_small_ready_thresholds_met"]


def test_alt_impulse_ladder_is_regime_shadow_only():
    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=72,
        complete_sample_count=72,
        avg_net_bps=25.0,
        p25_net_bps=1.0,
        win_rate=0.72,
        paper_days=20,
        paper_slippage_coverage=0.9,
        cost_source_mix={"mixed_actual_proxy": 72},
        candidate_name="v5.alt_impulse_shadow",
    )

    assert decision == "REGIME_SHADOW"
    assert "live_disabled" in reasons

    decision, reasons = strategy_evidence_decision_ladder(
        sample_count=12,
        complete_sample_count=12,
        avg_net_bps=-25.0,
        p25_net_bps=-40.0,
        win_rate=0.2,
        candidate_name="v5.alt_impulse_shadow",
    )
    assert decision == "KILL"
    assert "negative_regime_net_edge" in reasons


def test_strategy_evidence_normalizes_stale_low_complete_decision():
    stale = pl.DataFrame(
        [
            {
                "strategy": "v5",
                "evidence_version": "strategy-evidence-v0.1",
                "as_of_date": "2026-05-16",
                "strategy_candidate": "Alpha6Factor",
                "candidate_name": "Alpha6Factor",
                "symbol": "BTC-USDT",
                "regime_state": "Trending",
                "horizon_hours": 4,
                "sample_count": 12,
                "complete_sample_count": 2,
                "avg_net_bps": -50.0,
                "median_net_bps": -50.0,
                "p25_net_bps": -70.0,
                "win_rate": 0.0,
                "cost_source_mix": '{"local_estimate":12}',
                "decision": "KILL",
                "decision_reasons": '["legacy_kill"]',
                "start_ts": datetime(2026, 5, 16, tzinfo=UTC),
                "end_ts": datetime(2026, 5, 16, tzinfo=UTC),
                "created_at": datetime(2026, 5, 16, tzinfo=UTC),
                "source": "test",
            }
        ]
    )

    normalized = normalize_strategy_evidence_decisions(stale)
    row = normalized.to_dicts()[0]

    assert row["decision"] == "RESEARCH_ONLY"
    assert "insufficient_complete_samples" in json.loads(row["decision_reasons"])


def test_strategy_evidence_builds_candidate_board_without_broad_btc_mixing(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake)
    _write_strategy_sources(lake)
    _write_alpha_discovery_labels(lake)

    result = build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")

    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    summary = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    rows = {
        (
            row["strategy_candidate"],
            row["symbol"],
            row["regime_state"],
            row["horizon_hours"],
        ): row
        for row in summary.to_dicts()
    }

    assert result.extracted_sample_count == samples.height
    key = ("v5.sol_protect_exception", "SOL-USDT", "trend", 24)
    assert key in rows
    assert rows[key]["sample_count"] == 35
    assert rows[key]["complete_sample_count"] == 35
    assert rows[key]["avg_net_bps"] == 6.0
    assert rows[key]["win_rate"] == 1.0
    assert rows[key]["decision"] == "PAPER_READY"
    assert all(
        row["sample_count"] >= 30
        for row in summary.filter(pl.col("decision") == "LIVE_SMALL_READY").to_dicts()
    )
    assert "candidate_id" in samples.columns
    assert "net_bps_after_cost" in samples.columns


def test_strategy_evidence_incremental_mode_skips_old_raw_labels(tmp_path):
    lake = tmp_path / "lake"
    old_ts = datetime(2026, 5, 1, tzinfo=UTC)
    recent_ts = datetime(2026, 5, 10, tzinfo=UTC)
    _write_candidate_labels_for_incremental_test(lake, old_ts, recent_ts)

    result = build_and_publish_strategy_evidence(
        lake,
        as_of_date="2026-05-10",
        mode="incremental",
        lookback_days=2,
    )
    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")

    assert result.mode == "incremental"
    assert result.extracted_sample_count == 1
    assert set(samples["candidate_id"].to_list()) == {"recent"}


def test_strategy_evidence_incremental_mode_skips_historical_outcome_inputs(tmp_path):
    lake = tmp_path / "lake"
    now = datetime(2026, 5, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_ts": now,
                    "ingest_ts": now,
                    "source_path_inside_bundle": "summaries/alt_impulse_shadow_outcomes.csv",
                    "row_index": 0,
                    "ts_utc": now.isoformat().replace("+00:00", "Z"),
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.alt_impulse_shadow",
                    "label_4h_net_bps": 20.0,
                    "label_status": "complete",
                    "raw_payload_json": json.dumps(
                        {
                            "symbol": "SOL-USDT",
                            "strategy_candidate": "v5.alt_impulse_shadow",
                            "label_4h_net_bps": 20.0,
                            "label_status": "complete",
                        }
                    ),
                }
            ]
        ),
        lake / "silver" / "v5_shadow_outcome",
    )

    result = build_and_publish_strategy_evidence(
        lake,
        as_of_date="2026-05-10",
        mode="incremental",
        lookback_days=2,
    )

    assert result.extracted_sample_count == 0


def test_daily_export_includes_alpha_discovery_reports(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake)
    _write_strategy_sources(lake)
    _write_alpha_discovery_labels(lake)
    build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    result = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        names = set(archive.namelist())
        assert "reports/alpha_discovery_board.csv" in names
        assert "reports/strategy_evidence_summary.md" in names
        assert "reports/candidate_kill_list.csv" in names
        assert "reports/candidate_shadow_watchlist.csv" in names
        assert "reports/candidate_paper_ready.csv" in names
        assert "research/strategy_evidence.csv" in names
        assert "research/alt_impulse_shadow_by_regime.csv" in names
        assert "research/alt_impulse_shadow_by_symbol_regime_horizon.csv" in names
        assert "research/strategy_evidence_samples.csv" in names
        board = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/alpha_discovery_board.csv").decode("utf-8"))
            )
        )
        evidence_rows = list(
            csv.DictReader(
                io.StringIO(archive.read("research/strategy_evidence.csv").decode("utf-8"))
            )
        )
        sample_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("research/strategy_evidence_samples.csv").decode("utf-8")
                )
            )
        )
        watch = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/candidate_shadow_watchlist.csv").decode("utf-8")
                )
            )
        )

    board_by_candidate = {row["candidate_name"]: row for row in board}
    assert board_by_candidate["v5.sol_protect_exception"]["decision"] == "PAPER_READY"
    assert len(evidence_rows) > 0
    assert len(board) > 0
    assert any(row["strategy_candidate"] == "v5.sol_protect_exception" for row in sample_rows)
    assert "candidate_id" in sample_rows[0]
    assert "net_bps_after_cost" in sample_rows[0]
    assert not any(row["candidate_name"] == "v5.sol_protect_exception" for row in watch)


def test_strategy_evidence_replaces_legacy_schema(tmp_path):
    lake = tmp_path / "lake"
    _write_market_bars(lake)
    _write_strategy_sources(lake)
    _write_alpha_discovery_labels(lake)
    write_parquet_dataset(
        pl.DataFrame([{"candidate_name": "legacy", "sample_count": 1}]),
        lake / "gold" / "strategy_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame([{"candidate_name": "legacy", "ts_utc": datetime(2026, 5, 9, tzinfo=UTC)}]),
        lake / "gold" / "strategy_evidence_sample",
    )

    build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")

    evidence = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    assert "strategy_candidate" in evidence.columns
    assert "candidate_id" in samples.columns
    assert "legacy" not in set(evidence["candidate_name"].drop_nulls())


def test_strategy_evidence_uses_historical_shadow_and_blocked_outcomes(tmp_path):
    lake = tmp_path / "lake"
    _write_historical_outcomes(lake)

    result = build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    evidence = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    board = read_parquet_dataset(lake / "gold" / "alpha_discovery_board")
    summary = {
        (row["strategy_candidate"], row["symbol"], row["regime_state"], row["horizon_hours"]): row
        for row in evidence.to_dicts()
    }
    board_rows = {
        (row["strategy_candidate"], row["symbol"], row["regime_state"], row["horizon_hours"]): row
        for row in board.to_dicts()
    }

    assert result.extracted_sample_count == samples.height
    assert samples.filter(
        (pl.col("strategy_candidate") == "v5.alt_impulse_shadow")
        & (pl.col("horizon_hours") == 24)
    ).height == 16
    assert summary[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 4)]["decision"] == "KILL"
    assert summary[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)]["decision"] == "KILL"
    assert board_rows[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)]["decision"] == (
        "KILL"
    )
    assert summary[
        ("v5.sol_protect_alpha6_low_exception", "SOL-USDT", "protect", 24)
    ]["decision"] == "KEEP_SHADOW"
    assert summary[("v5.multi_position_k2", "BNB-USDT", "trend", 24)]["decision"] == "KILL"
    assert board_rows[("v5.multi_position_k2", "BNB-USDT", "trend", 24)]["decision"] == (
        "KILL"
    )
    assert summary[("v5.multi_position_k3", "SOL-USDT", "trend", 24)]["decision"] == "KILL"
    assert summary[("v5.multi_position_k2", "PORTFOLIO", "trend", 24)]["decision"] == (
        "RESEARCH_ONLY"
    )
    assert summary[("v5.multi_position_k2", "PORTFOLIO", "trend", 24)]["sample_count"] == 1
    assert summary[("v5.multi_position_k3", "PORTFOLIO", "trend", 24)]["decision"] == (
        "RESEARCH_ONLY"
    )
    assert ("v5.btc_leadership_probe_strict", "BTC-USDT", "trend", 24) in summary
    assert ("v5.btc_leadership_blocked_relaxed", "BTC-USDT", "trend", 24) in summary
    assert summary[("v5.f3_dominant_entry", "BNB-USDT", "trend", 24)]["decision"] == (
        "KEEP_SHADOW"
    )
    assert summary[("v5.f4_volume_expansion_entry", "ETH-USDT", "trend", 24)]["decision"] == (
        "KEEP_SHADOW"
    )
    assert {
        "high_score_blocked_outcome",
        "btc_leadership_blocked_outcome",
        "alt_impulse_shadow_outcome",
        "multi_position_swing_shadow_outcome",
        "factor_contribution_outcome",
        "protect_sol_exception_shadow_outcome",
    }.issubset(set(samples["source_type"].drop_nulls()))
    for key, evidence_row in summary.items():
        assert board_rows[key]["decision"] == evidence_row["decision"]


def test_strategy_evidence_counts_historical_outcomes_by_unique_event_not_aggregate_count(
    tmp_path,
):
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 8, tzinfo=UTC)
    duplicate_payload = {
        "candidate_name": "alt_impulse_shadow",
        "label_4h_net_bps": -31.0,
        "label_4h_status": "complete",
        "label_24h_net_bps": -42.0,
        "label_24h_status": "complete",
    }
    rows = [
        {
            "strategy": "v5",
            "bundle_sha256": f"bundle-{index}",
            "bundle_name": f"bundle-{index}.tar.gz",
            "bundle_ts": start + timedelta(minutes=index),
            "ingest_ts": start + timedelta(minutes=index),
            "source_path_inside_bundle": "summaries/alt_impulse_shadow_outcomes.csv",
            "row_index": index,
            "candidate_name": "alt_impulse_shadow",
            "ts_utc": start.isoformat().replace("+00:00", "Z"),
            "symbol": "ETH-USDT",
            "regime_state": "impulse",
            "sample_count": "999",
            "complete_sample_count": "0",
            "label_4h_net_bps": "-31",
            "label_4h_status": "complete",
            "label_24h_net_bps": "-42",
            "label_24h_status": "complete",
            "cost_bps": "4.0",
            "cost_source": "quant_lab_actual",
            "raw_payload_json": json.dumps(duplicate_payload),
        }
        for index in range(2)
    ]
    rows.append(
        {
            "strategy": "v5",
            "bundle_sha256": "unknown",
            "bundle_name": "unknown.tar.gz",
            "bundle_ts": start,
            "ingest_ts": start,
            "source_path_inside_bundle": "summaries/alt_impulse_shadow_outcomes.csv",
            "row_index": 99,
            "candidate_name": "alt_impulse_shadow",
            "ts_utc": (start + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "regime_state": "impulse",
            "label_24h_net_bps": "12",
            "label_24h_status": "complete",
            "raw_payload_json": json.dumps(
                {
                    "candidate_name": "alt_impulse_shadow",
                    "label_24h_net_bps": 12.0,
                    "label_24h_status": "complete",
                }
            ),
        }
    )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_shadow_outcome")

    result = build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    evidence = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    quality = read_parquet_dataset(lake / "gold" / "strategy_evidence_quality")
    board = read_parquet_dataset(lake / "gold" / "alpha_discovery_board")
    summary = {
        (row["strategy_candidate"], row["symbol"], row["regime_state"], row["horizon_hours"]): row
        for row in evidence.to_dicts()
    }

    assert "strategy_evidence_unknown_symbol_samples_skipped:1" in result.warnings
    quality_rows = {
        row["warning_type"]: row
        for row in quality.filter(pl.col("as_of_date") == "2026-05-10").to_dicts()
    }
    assert quality_rows["strategy_evidence_unknown_symbol_samples_skipped"][
        "warning_count"
    ] == 1
    assert "UNKNOWN" not in set(samples["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(evidence["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(board["symbol"].drop_nulls())
    assert samples.filter(pl.col("horizon_hours") == 24).height == 1
    assert summary[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)]["sample_count"] == 1
    assert (
        summary[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)][
            "complete_sample_count"
        ]
        == 1
    )
    assert summary[("v5.alt_impulse_shadow", "ETH-USDT", "impulse", 24)]["avg_net_bps"] == -42.0

    legacy_sample = samples.to_dicts()[0].copy()
    legacy_sample.update(
        {
            "candidate_id": "legacy-unknown",
            "source_event_key": "legacy-unknown",
            "symbol": "UNKNOWN",
            "as_of_date": "2026-05-10",
        }
    )
    legacy_evidence = evidence.to_dicts()[0].copy()
    legacy_evidence.update(
        {
            "strategy_candidate": "v5.alt_impulse_shadow",
            "candidate_name": "v5.alt_impulse_shadow",
            "symbol": "UNKNOWN",
            "as_of_date": "2026-05-10",
            "sample_count": 999,
            "complete_sample_count": 0,
        }
    )
    stale_evidence = evidence.to_dicts()[0].copy()
    stale_evidence.update(
        {
            "strategy_candidate": "v5.alt_impulse_shadow",
            "candidate_name": "v5.alt_impulse_shadow",
            "symbol": "UNKNOWN",
            "as_of_date": "2026-05-09",
            "sample_count": 999,
            "complete_sample_count": 0,
        }
    )
    legacy_board = board.to_dicts()[0].copy()
    legacy_board.update(
        {
            "strategy_candidate": "v5.alt_impulse_shadow",
            "candidate_name": "v5.alt_impulse_shadow",
            "symbol": "UNKNOWN",
            "as_of_date": "2026-05-10",
            "sample_count": 999,
            "complete_sample_count": 0,
        }
    )
    write_parquet_dataset(
        pl.concat(
            [samples, pl.DataFrame([legacy_sample], schema=samples.schema)],
            how="diagonal_relaxed",
        ),
        lake / "gold" / "strategy_evidence_sample",
    )
    write_parquet_dataset(
        pl.concat(
            [evidence, pl.DataFrame([legacy_evidence, stale_evidence], schema=evidence.schema)],
            how="diagonal_relaxed",
        ),
        lake / "gold" / "strategy_evidence",
    )
    write_parquet_dataset(
        pl.concat(
            [board, pl.DataFrame([legacy_board], schema=board.schema)],
            how="diagonal_relaxed",
        ),
        lake / "gold" / "alpha_discovery_board",
    )

    build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    evidence = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    board = read_parquet_dataset(lake / "gold" / "alpha_discovery_board")
    current_samples = samples.filter(pl.col("as_of_date") == "2026-05-10")
    current_evidence = evidence.filter(pl.col("as_of_date") == "2026-05-10")
    current_board = board.filter(pl.col("as_of_date") == "2026-05-10")
    assert "UNKNOWN" not in set(samples["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(current_samples["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(current_evidence["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(current_board["symbol"].drop_nulls())
    assert evidence.filter(
        (pl.col("as_of_date") == "2026-05-10") & (pl.col("sample_count") == 999)
    ).is_empty()
    assert board.filter(
        (pl.col("as_of_date") == "2026-05-10") & (pl.col("sample_count") == 999)
    ).is_empty()
    assert board.filter(
        (pl.col("strategy_candidate") == "v5.alt_impulse_shadow")
        & (pl.col("complete_sample_count") == 0)
        & (pl.col("sample_count") > 10)
    ).is_empty()


def test_alt_impulse_outcomes_preserve_symbol_horizon_and_complete_counts(tmp_path):
    lake = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    horizons = [4, 8, 12, 24, 48, 72, 120]
    rows = []
    expected: dict[tuple[str, int], float] = {}
    for symbol_index, symbol in enumerate(["SOL-USDT", "ETH-USDT", "BNB-USDT"]):
        payload = {
            "event_id": f"alt-{symbol_index}",
            "candidate_name": "alt_impulse_shadow",
            "symbol": symbol,
        }
        for horizon in horizons:
            net_bps = float((symbol_index + 1) * horizon)
            payload[f"label_{horizon}h_net_bps"] = net_bps
            payload[f"label_{horizon}h_status"] = "complete"
            expected[(symbol, horizon)] = net_bps
        rows.append(
            {
                "strategy": "v5",
                "bundle_sha256": "bundle",
                "bundle_name": "bundle.tar.gz",
                "bundle_ts": start,
                "ingest_ts": start,
                "source_path_inside_bundle": "summaries/alt_impulse_shadow_outcomes.csv",
                "row_index": symbol_index,
                "event_id": f"alt-{symbol_index}",
                "candidate_name": "alt_impulse_shadow",
                "ts_utc": (start + timedelta(minutes=symbol_index)).isoformat().replace(
                    "+00:00", "Z"
                ),
                "symbol": symbol,
                "regime_state": "impulse",
                "sample_count": "999",
                "complete_sample_count": "0",
                "cost_bps": "1.0",
                "cost_source": "mixed_actual_proxy",
                "raw_payload_json": json.dumps(payload),
            }
        )
    rows.append(
        {
            "strategy": "v5",
            "bundle_sha256": "pending",
            "bundle_name": "pending.tar.gz",
            "bundle_ts": start,
            "ingest_ts": start,
            "source_path_inside_bundle": "summaries/alt_impulse_shadow_outcomes.csv",
            "row_index": 99,
            "event_id": "pending-large",
            "candidate_name": "alt_impulse_shadow",
            "ts_utc": (start + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "symbol": "SOL-USDT",
            "regime_state": "impulse",
            "sample_count": "999",
            "complete_sample_count": "0",
            "label_24h_net_bps": "1000",
            "label_24h_status": "pending",
            "cost_source": "mixed_actual_proxy",
        }
    )
    rows.append(
        {
            "strategy": "v5",
            "bundle_sha256": "unknown",
            "bundle_name": "unknown.tar.gz",
            "bundle_ts": start,
            "ingest_ts": start,
            "source_path_inside_bundle": "summaries/alt_impulse_shadow_outcomes.csv",
            "row_index": 100,
            "event_id": "unknown-large",
            "candidate_name": "alt_impulse_shadow",
            "ts_utc": (start + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
            "symbol": "UNKNOWN",
            "regime_state": "impulse",
            "sample_count": "999",
            "complete_sample_count": "0",
            "label_24h_net_bps": "500",
            "label_24h_status": "complete",
        }
    )
    write_parquet_dataset(pl.DataFrame(rows), lake / "silver" / "v5_shadow_outcome")

    result = build_and_publish_strategy_evidence(lake, as_of_date="2026-05-10")
    build_and_publish_alpha_discovery_board(lake, as_of_date="2026-05-10")

    samples = read_parquet_dataset(lake / "gold" / "strategy_evidence_sample")
    evidence = read_parquet_dataset(lake / "gold" / "strategy_evidence")
    board = read_parquet_dataset(lake / "gold" / "alpha_discovery_board")
    quality = read_parquet_dataset(lake / "gold" / "strategy_evidence_quality")

    assert "strategy_evidence_unknown_symbol_samples_skipped:1" in result.warnings
    assert "UNKNOWN" not in set(samples["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(evidence["symbol"].drop_nulls())
    assert "UNKNOWN" not in set(board["symbol"].drop_nulls())
    assert set(
        samples.filter(pl.col("strategy_candidate") == "v5.alt_impulse_shadow")[
            "horizon_hours"
        ].unique()
    ) == set(horizons)

    evidence_rows = {
        (row["symbol"], row["horizon_hours"]): row
        for row in evidence.filter(
            pl.col("strategy_candidate") == "v5.alt_impulse_shadow"
        ).to_dicts()
    }
    for key, net_bps in expected.items():
        row = evidence_rows[key]
        if key == ("SOL-USDT", 24):
            continue
        assert row["sample_count"] == 1
        assert row["complete_sample_count"] == 1
        assert row["avg_net_bps"] == net_bps
    pending_row = evidence_rows[("SOL-USDT", 24)]
    assert pending_row["sample_count"] == 2
    assert pending_row["complete_sample_count"] == 1
    assert pending_row["decision"] == "KEEP_SHADOW"
    assert "insufficient_regime_complete_samples" in json.loads(
        pending_row["decision_reasons"]
    )
    assert board.filter(
        (pl.col("strategy_candidate") == "v5.alt_impulse_shadow")
        & (pl.col("symbol") == "UNKNOWN")
    ).is_empty()
    assert quality.filter(
        (pl.col("warning_type") == "strategy_evidence_unknown_symbol_samples_skipped")
        & (pl.col("warning_count") == 1)
    ).height == 1

    export = export_daily_pack(
        export_date="2026-05-10",
        lake_root=lake,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )
    with zipfile.ZipFile(export.zip_path) as archive:
        alt_by_regime = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("research/alt_impulse_shadow_by_regime.csv").decode("utf-8")
                )
            )
        )
        alt_by_symbol_regime = list(
            csv.DictReader(
                io.StringIO(
                    archive.read(
                        "research/alt_impulse_shadow_by_symbol_regime_horizon.csv"
                    ).decode("utf-8")
                )
            )
        )
    assert any(
        row["strategy_candidate"] == "v5.alt_impulse_shadow"
        and row["regime_state"] == "impulse"
        for row in alt_by_regime
    )
    assert any(
        row["strategy_candidate"] == "v5.alt_impulse_shadow"
        and row["symbol"] == "ETH-USDT"
        and row["regime_state"] == "impulse"
        for row in alt_by_symbol_regime
    )


def _write_market_bars(lake: Path) -> None:
    start = datetime(2026, 5, 9, tzinfo=UTC)
    rows = []
    for symbol, hourly_return in {
        "BTC-USDT": 0.0015,
        "SOL-USDT": 0.001,
        "ETH-USDT": -0.001,
    }.items():
        for index in range(180):
            close = 100.0 * ((1.0 + hourly_return) ** index)
            rows.append(
                {
                    "venue": "okx",
                    "symbol": symbol,
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": start + timedelta(hours=index),
                    "open": close,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close,
                    "volume": 100.0,
                    "quote_volume": close * 100.0,
                    "source": "test",
                    "ingest_ts": start + timedelta(hours=index, minutes=1),
                }
            )
    write_market_bars(lake, rows)


def _write_alpha_discovery_labels(lake: Path) -> None:
    start = datetime(2026, 5, 9, tzinfo=UTC)
    rows = []
    for index in range(35):
        rows.append(
            {
                "strategy": "v5",
                "candidate_id": f"sol-{index}",
                "run_id": f"run-sol-{index}",
                "ts_utc": start + timedelta(hours=index),
                "symbol": "SOL-USDT",
                "strategy_candidate": "v5.sol_protect_exception",
                "block_reason": "protect_exception",
                "final_decision": "SHADOW",
                "horizon_hours": 24,
                "gross_bps": 10.0,
                "net_bps_after_cost": 6.0,
                "mfe_bps": 12.0,
                "mae_bps": -4.0,
                "win": True,
                "label_status": "complete",
                "cost_bps": 4.0,
                "cost_source": "quant_lab",
                "regime_state": "trend",
                "created_at": start + timedelta(hours=index, minutes=1),
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "v5_candidate_label")


def _write_candidate_labels_for_incremental_test(
    lake: Path,
    old_ts: datetime,
    recent_ts: datetime,
) -> None:
    rows = []
    for candidate_id, ts in [("old", old_ts), ("recent", recent_ts)]:
        rows.append(
            {
                "strategy": "v5",
                "candidate_id": candidate_id,
                "run_id": f"run-{candidate_id}",
                "ts_utc": ts,
                "symbol": "SOL-USDT",
                "strategy_candidate": "v5.sol_protect_exception",
                "block_reason": "protect_exception",
                "final_decision": "SHADOW",
                "horizon_hours": 24,
                "gross_bps": 10.0,
                "net_bps_after_cost": 6.0,
                "mfe_bps": 12.0,
                "mae_bps": -4.0,
                "win": True,
                "label_status": "complete",
                "cost_bps": 4.0,
                "cost_source": "mixed_actual_proxy",
                "regime_state": "trend",
                "created_at": ts,
            }
        )
    write_parquet_dataset(pl.DataFrame(rows), lake / "gold" / "v5_candidate_label")


def _write_strategy_sources(lake: Path) -> None:
    start = datetime(2026, 5, 9, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "sol",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": start + timedelta(hours=index),
                    "ingest_ts": start + timedelta(hours=index, minutes=1),
                    "source_path_inside_bundle": "summaries/router_decisions.csv",
                    "row_index": index,
                    "candidate_name": "sol_protect_exception",
                    "ts_utc": (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
                    "symbol": "SOL-USDT",
                    "reason": "protect_exception",
                    "final_score": "0.71",
                    "f1": "0.1",
                    "f2": "0.2",
                    "f3": "0.3",
                    "f4": "0.4",
                    "f5": "0.5",
                    "alpha6_score": "0.8",
                    "alpha6_side": "long",
                    "regime_state": "trend",
                    "protect_level": "SOL_PROTECT",
                    "expected_edge_bps": "18",
                    "raw_payload_json": json.dumps({"candidate_name": "sol_protect_exception"}),
                }
                for index in range(35)
            ]
        ),
        lake / "silver" / "v5_router_decision",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "btc",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": start,
                    "ingest_ts": start,
                    "source_path_inside_bundle": "summaries/probe_diagnostics.csv",
                    "row_index": 0,
                    "probe_name": "btc_leadership_blocker",
                    "ts_utc": start.isoformat().replace("+00:00", "Z"),
                    "symbol": "BTC-USDT",
                    "raw_payload_json": json.dumps({"probe_name": "btc_leadership_blocker"}),
                },
                {
                    "strategy": "v5",
                    "bundle_sha256": "btc",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": start + timedelta(hours=1),
                    "ingest_ts": start + timedelta(hours=1),
                    "source_path_inside_bundle": "summaries/probe_diagnostics.csv",
                    "row_index": 1,
                    "probe_name": "btc_leadership_probe_strict",
                    "ts_utc": (start + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                    "symbol": "BTC-USDT",
                    "final_score": "0.9",
                    "alpha6_side": "long",
                    "raw_payload_json": json.dumps({"probe_name": "btc_leadership_probe_strict"}),
                },
                {
                    "strategy": "v5",
                    "bundle_sha256": "btc",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": start + timedelta(hours=2),
                    "ingest_ts": start + timedelta(hours=2),
                    "source_path_inside_bundle": "summaries/probe_diagnostics.csv",
                    "row_index": 2,
                    "probe_name": "strict_btc_leadership_probe",
                    "ts_utc": (start + timedelta(hours=2)).isoformat().replace("+00:00", "Z"),
                    "symbol": "BTC-USDT",
                    "final_score": "0.91",
                    "alpha6_side": "long",
                    "raw_payload_json": json.dumps({"probe_name": "strict_btc_leadership_probe"}),
                },
            ]
        ),
        lake / "silver" / "v5_probe_diagnostic",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "alt",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": start + timedelta(hours=index),
                    "ingest_ts": start + timedelta(hours=index),
                    "source_path_inside_bundle": "summaries/alt_impulse_shadow.csv",
                    "row_index": index,
                    "candidate_name": "alt_impulse_shadow",
                    "ts_utc": (start + timedelta(hours=index)).isoformat().replace("+00:00", "Z"),
                    "symbol": "ETH-USDT",
                    "final_score": "0.8",
                    "alpha6_side": "long",
                    "regime_state": "impulse",
                    "raw_payload_json": json.dumps({"candidate_name": "alt_impulse_shadow"}),
                }
                for index in range(5)
            ]
        ),
        lake / "silver" / "v5_shadow_outcome",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "cost",
                    "bundle_name": "bundle.tar.gz",
                    "bundle_ts": start,
                    "ingest_ts": start,
                    "source_path_inside_bundle": "summaries/quant_lab_cost_usage.csv",
                    "row_index": 0,
                    "symbol": symbol,
                    "cost_source": "quant_lab",
                    "cost_bps": "4.0",
                    "raw_payload_json": json.dumps(
                        {"symbol": symbol, "cost_source": "quant_lab", "cost_bps": 4.0}
                    ),
                }
                for symbol in ["BTC-USDT", "SOL-USDT", "ETH-USDT"]
            ]
        ),
        lake / "silver" / "v5_quant_lab_cost_usage",
    )


def _write_historical_outcomes(lake: Path) -> None:
    start = datetime(2026, 5, 8, tzinfo=UTC)
    high_score_rows = []
    shadow_rows = []
    _add_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/protect_sol_exception_shadow_outcomes.csv",
        candidate="sol_protect_alpha6_low_exception",
        symbol="SOL-USDT",
        regime="protect",
        net_values=[15.0] * 12,
    )
    _add_outcome_rows(
        high_score_rows,
        start=start,
        source_path="summaries/high_score_blocked_outcomes.csv",
        candidate="btc_leadership_blocked_relaxed",
        symbol="BTC-USDT",
        regime="trend",
        net_values=[8.0] * 12,
    )
    _add_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/btc_leadership_probe_blocked_outcomes.csv",
        candidate="btc_leadership_probe_strict",
        symbol="BTC-USDT",
        regime="trend",
        net_values=[12.0] * 12,
    )
    _add_multi_horizon_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/alt_impulse_shadow_outcomes.csv",
        candidate="alt_impulse_shadow",
        symbol="ETH-USDT",
        regime="impulse",
        net_values=[-35.0] * 16,
    )
    _add_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/multi_position_swing_shadow_outcomes.csv",
        candidate="multi_position_k2",
        symbol="BNB-USDT",
        regime="trend",
        net_values=[-12.0] * 12,
    )
    _add_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/multi_position_swing_shadow_outcomes.csv",
        candidate="multi_position_k3",
        symbol="SOL-USDT",
        regime="trend",
        net_values=[-10.0] * 12,
    )
    _add_multi_position_by_k_rows(
        shadow_rows,
        start=start,
        k=2,
        net_24h=-95.0,
        count=12,
    )
    _add_multi_position_by_k_rows(
        shadow_rows,
        start=start + timedelta(hours=1),
        k=3,
        net_24h=-120.0,
        count=12,
    )
    _add_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/factor_contribution_outcomes_by_factor.csv",
        candidate="f3_dominant_entry",
        symbol="BNB-USDT",
        regime="trend",
        net_values=[18.0] * 12,
    )
    _add_outcome_rows(
        shadow_rows,
        start=start,
        source_path="summaries/factor_contribution_outcomes_by_factor.csv",
        candidate="f4_volume_expansion_entry",
        symbol="ETH-USDT",
        regime="trend",
        net_values=[16.0] * 12,
    )
    write_parquet_dataset(
        pl.DataFrame(high_score_rows),
        lake / "silver/v5_high_score_blocked_outcome",
    )
    write_parquet_dataset(pl.DataFrame(shadow_rows), lake / "silver/v5_shadow_outcome")


def _add_multi_horizon_outcome_rows(
    rows: list[dict],
    *,
    start: datetime,
    source_path: str,
    candidate: str,
    symbol: str,
    regime: str,
    net_values: list[float],
) -> None:
    for index, net in enumerate(net_values):
        ts = start + timedelta(hours=index)
        payload = {
            "candidate_name": candidate,
            "event_id": f"{candidate}-{symbol}-{index}",
        }
        row = {
            "strategy": "v5",
            "bundle_sha256": "hist",
            "bundle_name": "bundle.tar.gz",
            "bundle_ts": ts,
            "ingest_ts": ts + timedelta(minutes=1),
            "source_path_inside_bundle": source_path,
            "row_index": index,
            "candidate_name": candidate,
            "event_id": f"{candidate}-{symbol}-{index}",
            "ts_utc": ts.isoformat().replace("+00:00", "Z"),
            "symbol": symbol,
            "regime_state": regime,
            "cost_bps": "4.0",
            "cost_source": "quant_lab_actual",
        }
        for horizon in [4, 8, 12, 24, 48, 72, 120]:
            adjusted = net - (horizon / 100.0)
            row[f"label_{horizon}h_net_bps"] = str(adjusted)
            row[f"label_{horizon}h_gross_bps"] = str(adjusted + 4.0)
            row[f"label_{horizon}h_win"] = str(adjusted > 0).lower()
            row[f"label_{horizon}h_status"] = "complete"
            payload[f"label_{horizon}h_net_bps"] = adjusted
        row["raw_payload_json"] = json.dumps(payload)
        rows.append(row)


def _add_multi_position_by_k_rows(
    rows: list[dict],
    *,
    start: datetime,
    k: int,
    net_24h: float,
    count: int,
) -> None:
    rows.append(
        {
            "strategy": "v5",
            "bundle_sha256": "hist",
            "bundle_name": "bundle.tar.gz",
            "bundle_ts": start,
            "ingest_ts": start + timedelta(minutes=1),
            "source_path_inside_bundle": "summaries/multi_position_swing_shadow_by_k.csv",
            "row_index": k,
            "ts_utc": start.isoformat().replace("+00:00", "Z"),
            "k": str(k),
            "count": str(count),
            "complete_count": str(count),
            "avg_24h_net_bps": str(net_24h),
            "win_rate_24h": "0.1",
            "raw_payload_json": json.dumps(
                {
                    "k": str(k),
                    "count": str(count),
                    "complete_count": str(count),
                    "avg_24h_net_bps": net_24h,
                    "win_rate_24h": 0.1,
                }
            ),
        }
    )


def _add_outcome_rows(
    rows: list[dict],
    *,
    start: datetime,
    source_path: str,
    candidate: str,
    symbol: str,
    regime: str,
    net_values: list[float],
) -> None:
    for index, net in enumerate(net_values):
        ts = start + timedelta(hours=index)
        rows.append(
            {
                "strategy": "v5",
                "bundle_sha256": "hist",
                "bundle_name": "bundle.tar.gz",
                "bundle_ts": ts,
                "ingest_ts": ts + timedelta(minutes=1),
                "source_path_inside_bundle": source_path,
                "row_index": index,
                "candidate_name": candidate,
                "event_id": f"{candidate}-{symbol}-{index}",
                "ts_utc": ts.isoformat().replace("+00:00", "Z"),
                "symbol": symbol,
                "regime_state": regime,
                "horizon_hours": "24",
                "net_bps_after_cost": str(net),
                "gross_bps": str(net + 4.0),
                "win": str(net > 0).lower(),
                "cost_bps": "4.0",
                "cost_source": "quant_lab_actual",
                "raw_payload_json": json.dumps(
                    {
                        "candidate_name": candidate,
                        "event_id": f"{candidate}-{symbol}-{index}",
                        "net_bps_after_cost": net,
                    }
                ),
            }
        )
