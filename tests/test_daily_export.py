import csv
import io
import json
import os
import zipfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from fastapi.testclient import TestClient

import quant_lab.api.main as api_main
import quant_lab.export.daily as daily_export_module
from quant_lab.api.main import app
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.file_index import build_lake_file_index
from quant_lab.data.lake import read_parquet_dataset, write_market_bars, write_parquet_dataset
from quant_lab.export.daily import (
    CSV_SCHEMAS,
    REQUIRED_MEMBERS,
    _collect_recent_heavy_files,
    _load_export_frame,
    _member_row_count,
    _recent_heavy_dataset_files,
    _sha256_payload,
    _strategy_evidence_has_required_candidates,
    export_daily_pack,
    validate_expert_pack,
)
from quant_lab.ops.metrics import record_api_request
from tests.v5_bundle_fixture import make_tar


@pytest.fixture(autouse=True)
def _isolate_default_v5_telemetry_config(monkeypatch, tmp_path):
    monkeypatch.delenv("QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES", raising=False)
    monkeypatch.delenv("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", raising=False)
    inbox = tmp_path / "isolated_v5_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    config = _v5_telemetry_config(tmp_path, inbox, tmp_path / "isolated_v5_lake")
    monkeypatch.setenv("QUANT_LAB_V5_TELEMETRY_CONFIG", str(config))


def test_collect_recent_heavy_files_uses_timestamp_not_physical_tail(tmp_path):
    dataset_path = tmp_path / "orderbook_spread_1m"
    dataset_path.mkdir()
    base = datetime(2026, 6, 11, 2, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for symbol in ("BNB-USDT", "BTC-USDT", "ZEC-USDT"):
        for minute in range(4):
            rows.append(
                {
                    "symbol": symbol,
                    "minute_ts": base + timedelta(minutes=minute),
                    "ts": base + timedelta(minutes=minute),
                    "spread_bps": 1.0 + minute,
                }
            )
    for minute in range(20):
        rows.append(
            {
                "symbol": "ZEC-USDT",
                "minute_ts": base - timedelta(hours=2, minutes=minute),
                "ts": base - timedelta(hours=2, minutes=minute),
                "spread_bps": 9.0,
            }
        )
    data_path = dataset_path / "data.parquet"
    pl.DataFrame(rows).write_parquet(data_path)

    collected = _collect_recent_heavy_files(
        [data_path],
        "orderbook_spread_1m",
        limit=9,
    )

    symbols = set(collected["symbol"].to_list())
    assert {"BNB-USDT", "BTC-USDT"}.issubset(symbols)
    assert collected["ts"].min() >= base + timedelta(minutes=1)


def test_export_v5_authoritative_scan_defaults_match_production_systemd(monkeypatch):
    monkeypatch.delenv("QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES", raising=False)
    monkeypatch.delenv("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", raising=False)

    max_pending = daily_export_module._export_v5_max_pending_bundles()

    assert max_pending == 1
    assert daily_export_module._export_v5_max_scan_bundles(max_pending) == 1000


def test_microstructure_rollup_limits_cover_forward_validation_window():
    min_rows_for_30_hourly_samples = 80_000

    assert (
        daily_export_module.HEAVY_EXPORT_DATASET_LIMITS["trade_activity_1m"]
        >= min_rows_for_30_hourly_samples
    )
    assert (
        daily_export_module.HEAVY_EXPORT_DATASET_LIMITS["orderbook_spread_1m"]
        >= min_rows_for_30_hourly_samples
    )


def test_v5_candidate_feature_completeness_by_strategy_breaks_down_gaps():
    frame = pl.DataFrame(
        [
            {
                "strategy_candidate": "f3_dominant_entry",
                "symbol": "BNB-USDT",
                "final_score": "0.9",
                "f1_mom_5d": "0.1",
                "f2_mom_20d": "0.2",
                "f3_vol_adj_ret": "1.8",
                "f4_volume_expansion": "0.4",
                "f5_rsi_trend_confirm": "0.5",
                "alpha6_score": "0.7",
                "alpha6_side": "buy",
                "ml_score": "0.9",
                "mean_reversion_score": "null",
                "expected_edge_bps": "80",
                "required_edge_bps": "45",
                "cost_source": "public_spread_proxy",
            },
            {
                "strategy_candidate": "portfolio_trend_following",
                "symbol": "BTC-USDT",
                "eligible_before_filters": "true",
                "final_score": "0.4",
                "f1_mom_5d": "null",
                "f2_mom_20d": "null",
                "f3_vol_adj_ret": "null",
                "f4_volume_expansion": "null",
                "f5_rsi_trend_confirm": "null",
                "alpha6_score": "null",
                "alpha6_side": "null",
                "ml_score": "0.4",
                "mean_reversion_score": "null",
                "expected_edge_bps": "10",
                "required_edge_bps": "45",
                "cost_source": "public_spread_proxy",
            },
            {
                "strategy_candidate": "portfolio_trend_following",
                "symbol": "ETH-USDT",
                "eligible_before_filters": "false",
                "final_decision": "no_order",
                "final_score": "null",
                "f1_mom_5d": "null",
                "f2_mom_20d": "null",
                "f3_vol_adj_ret": "null",
                "f4_volume_expansion": "null",
                "f5_rsi_trend_confirm": "null",
                "alpha6_score": "null",
                "alpha6_side": "null",
                "ml_score": "null",
                "mean_reversion_score": "null",
                "expected_edge_bps": "0",
                "required_edge_bps": "45",
                "cost_source": "local_estimate",
            },
        ]
    )

    report = daily_export_module._v5_candidate_feature_completeness_by_strategy(frame)
    rows = {row["strategy_candidate"]: row for row in report.to_dicts()}

    assert rows["f3_dominant_entry"]["core_signal_completeness"] == 1.0
    assert rows["f3_dominant_entry"]["edge_context_completeness"] == 1.0
    assert rows["portfolio_trend_following"]["row_count"] == 2
    assert rows["portfolio_trend_following"]["feature_denominator_row_count"] == 1
    assert rows["portfolio_trend_following"]["no_signal_context_row_count"] == 1
    assert rows["portfolio_trend_following"]["core_signal_completeness"] == 0.125
    assert rows["portfolio_trend_following"]["edge_context_completeness"] == 1.0
    assert "f1_mom_5d" in rows["portfolio_trend_following"]["missing_core_fields"]
    assert rows["portfolio_trend_following"]["live_order_effect"] == "read_only_no_live_order"


def test_v5_candidate_required_feature_completeness_prefers_new_metric():
    assert daily_export_module._v5_candidate_required_feature_completeness(
        {"feature_completeness": 0.2, "required_feature_completeness": 0.95}
    ) == pytest.approx(0.95)
    assert daily_export_module._v5_candidate_required_feature_completeness(
        {"feature_completeness": 0.75}
    ) == pytest.approx(0.75)
    assert daily_export_module._v5_candidate_required_feature_completeness(
        {
            "feature_completeness": 0.2,
            "feature_completeness_by_field_json": json.dumps(
                {
                    "final_score": 0.9,
                    "expected_edge_bps": 1.0,
                    "required_edge_bps": 1.0,
                    "mean_reversion_score": 0.0,
                }
            ),
        }
    ) == pytest.approx((0.9 + 1.0 + 1.0) / 3)


def test_export_daily_pack_writes_required_members(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    pack_path = Path(result.zip_path)
    export_index_path = tmp_path / "exports" / "export_index.json"
    validation = validate_expert_pack(pack_path)

    assert pack_path.exists()
    assert export_index_path.is_file()
    assert validation.valid
    structural_warnings = [
        warning
        for warning in validation.warnings
        if warning != "data_quality status is CRITICAL"
    ]
    assert structural_warnings == []
    assert validation.export_date == "2026-05-11"
    assert result.row_counts["market_bar"] == 2

    with zipfile.ZipFile(pack_path) as archive:
        names = set(archive.namelist())
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        provenance = json.loads(archive.read("provenance.json").decode("utf-8"))
        assert set(REQUIRED_MEMBERS).issubset(names)
        assert manifest["export_date"] == "2026-05-11"
        assert manifest["display_timezone"] == "Asia/Shanghai"
        assert manifest["generated_at_beijing"].endswith("+08:00")
        assert manifest["row_counts"]["market_bar"] == 2
        assert manifest["git_commit"]
        assert "git_branch" in manifest
        assert "dirty_worktree" in manifest

        assert "git_dirty" in manifest
        assert manifest["provenance_status"] in {"git_clean", "git_dirty"}
        assert manifest["code_provenance"] in {"ok", "degraded"}
        assert manifest["hostname"]
        assert manifest["python_version"]
        assert manifest["export_command"] == "qlab export-daily"
        assert manifest["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
        assert provenance["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
        assert provenance["display_timezone"] == "Asia/Shanghai"
        assert provenance["generated_at_beijing"].endswith("+08:00")
        assert "dataset_freshness" in manifest
        assert provenance["git_commit"]
        assert "git_dirty" in provenance
        assert provenance["provenance_status"] in {"git_clean", "git_dirty"}
        assert provenance["code_provenance"] in {"ok", "degraded"}
        assert "freshness_seconds" in provenance["datasets"][0]
        assert "freshness_status" in provenance["datasets"][0]

        assert "v5/v5_strategy_health.csv" in names
        assert "v5/v5_btc_probe_entry_quality_audit.csv" in names
        assert "reports/v5_enforce_readiness.json" in names
        assert "reports/v5_enforce_readiness.csv" in names
        assert "reports/system_acceptance_dashboard.csv" in names
        assert "reports/system_acceptance_dashboard.md" in names
        assert "reports/post_impulse_overextension_no_trigger_reasons.csv" in names
        assert "reports/bottom_zone_reversal_no_trigger_reasons.csv" in names
        assert "reports/factor_dedupe_decision.csv" in names
        assert "reports/factor_family_leaderboard.csv" in names
        assert "reports/factor_paper_review_queue.csv" in names
        assert "reports/composite_factor_candidates.csv" in names
        assert "reports/factor_regime_effectiveness.csv" in names
        assert "reports/factor_forward_validation.csv" in names
        assert "reports/factor_forward_validation.md" in names
        assert "reports/factor_strategy_bridge_candidates.csv" in names
        assert "reports/cost_bootstrap_readiness.csv" in names
        assert "reports/live_universe_cost_coverage.csv" in names
        assert "reports/v5_candidate_feature_completeness_by_strategy.csv" in names
        assert "reports/fast_microstructure_forward_test.csv" in names
        assert "reports/fast_microstructure_strategy_candidates.csv" in names
        assert "reports/fast_microstructure_strategy_review.csv" in names
        assert "reports/fast_microstructure_forward_summary.md" in names
        assert "reports/bottom_zone_probe_paper_readiness.csv" in names
        assert "reports/api_error_summary.csv" in names
        assert "diagnostics/export_timing.csv" in names
        assert "diagnostics/export_timing.json" in names
        acceptance_rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/system_acceptance_dashboard.csv").decode("utf-8"))
            )
        )
        acceptance_checks = {row["check_name"] for row in acceptance_rows}
        assert "v5_bundle_sync_ok" in acceptance_checks
        readiness_rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/cost_bootstrap_readiness.csv").decode("utf-8"))
            )
        )
        assert readiness_rows
        assert "bootstrap_state" in readiness_rows[0]
        assert "actual_or_mixed_bootstrap_coverage_live_universe" in readiness_rows[0]
        assert "web_rglob_fallback_zero" in acceptance_checks
        assert "api_latency_p95_ok" in acceptance_checks
        assert "System Acceptance Dashboard" in archive.read(
            "reports/system_acceptance_dashboard.md"
        ).decode("utf-8")
        export_timing = list(
            csv.DictReader(
                io.StringIO(archive.read("diagnostics/export_timing.csv").decode("utf-8"))
            )
        )
        assert any(row["stage"] == "load_snapshot" for row in export_timing)
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        executive_summary = archive.read("executive_summary.md").decode("utf-8")
        assert "quant_lab_enforce_readiness" in data_quality
        assert "shadow_only_recommended" in data_quality
        assert "dataset_governance" in data_quality
        assert "registry_quality" in data_quality
        assert "check_count" in data_quality["dataset_governance"]
        assert "check_count" in data_quality["registry_quality"]
        assert any(check["name"] == "code_provenance_clean" for check in data_quality["checks"])
        assert "quant_lab_enforce_readiness:" in executive_summary
        assert "charts/market_close.png" in names
        assert archive.read("charts/market_close.png").startswith(b"\x89PNG")

    export_index = json.loads(export_index_path.read_text(encoding="utf-8"))
    assert export_index["latest_pack"] == str(pack_path)
    assert export_index["manifest_summary"]["export_date"] == "2026-05-11"
    assert export_index["packs"]

    for dataset in daily_export_module.SNAPSHOT_META_DATASETS:
        dataset_path = lake_root / daily_export_module.readers.DATASET_PATHS.get(
            dataset,
            Path("gold") / dataset,
        )
        meta_path = dataset_path / "_snapshot_meta.json"
        assert meta_path.is_file(), dataset
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["dataset"] == dataset
        assert meta.get("source_sha"), dataset


def test_research_validation_v3_reports_export_forward_and_cost_coverage(tmp_path):
    lake_root = tmp_path / "lake"
    out_dir = tmp_path / "exports"
    start = datetime(2026, 6, 1, tzinfo=UTC)
    bars = []
    factor_values = []
    spreads = []
    trades = []
    for index in range(90):
        ts = start + timedelta(hours=index)
        close = 100.0 + 0.03 * index * index
        bars.append(
            {
                "venue": "okx",
                "symbol": "BTC-USDT",
                "market_type": "SPOT",
                "timeframe": "1H",
                "ts": ts,
                "open": close - 0.1,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": 1000.0 + index,
                "quote_volume": close * (1000.0 + index),
                "source": "test",
                "ingest_ts": ts + timedelta(minutes=1),
                "is_closed": True,
            }
        )
        factor_values.append(
            {
                "factor_id": "core.close_return_24",
                "factor_name": "Close return 24",
                "factor_family": "momentum",
                "factor_version": "v0.1",
                "symbol": "BTC-USDT",
                "timeframe": "1H",
                "ts": ts,
                "raw_value": float(index),
                "normalized_value": float(index),
                "rank_value": float(index),
                "value": float(index),
                "is_valid": True,
                "created_at": ts,
                "source": "test",
            }
        )
        spreads.append(
            {
                "symbol": "BTC-USDT",
                "channel": "books5",
                "minute_ts": ts,
                "ts": ts,
                "spread_bps": 2.0,
                "orderbook_imbalance": min(0.9, index / 100.0),
            }
        )
        trades.append(
            {
                "symbol": "BTC-USDT",
                "minute_ts": ts,
                "latest_trade_ts": ts,
                "trade_count": 5,
                "size_sum": 20.0,
                "taker_buy_size_sum": 10.0 + index,
                "taker_sell_size_sum": 10.0,
            }
        )

    write_market_bars(lake_root, bars)
    write_parquet_dataset(pl.DataFrame(factor_values), lake_root / "gold" / "factor_value")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-01",
                    "factor_id": "core.close_return_24",
                    "factor_name": "Close return 24",
                    "factor_family": "momentum",
                    "factor_version": "v0.1",
                    "timeframe": "1H",
                    "best_horizon_bars": 4,
                    "tested_horizon_count": 3,
                    "best_score": 10.0,
                    "avg_score": 8.0,
                    "best_rank_ic_mean": 0.2,
                    "best_rank_ic_tstat": 3.0,
                    "best_long_short_mean_bps": 25.0,
                    "candidate_state": "PAPER_READY",
                    "recommended_action": "PAPER_REVIEW",
                    "created_at": start,
                    "source": "test",
                }
            ]
        ),
        lake_root / "gold" / "factor_candidate",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-01",
                    "factor_id": "core.close_return_24",
                    "factor_family": "momentum",
                    "horizon_bars": 4,
                    "valid_sample_count": 120,
                    "rank_ic_mean": 0.2,
                    "rank_ic_tstat": 3.0,
                    "long_short_mean_bps": 25.0,
                    "win_rate": 0.65,
                    "score": 10.0,
                    "regime_state": "TREND_UP",
                    "created_at": start,
                }
            ]
        ),
        lake_root / "gold" / "factor_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-01",
                    "factor_id_left": "core.close_return_24",
                    "factor_id_right": "core.close_return_24",
                    "sample_count": 120,
                    "correlation": 1.0,
                }
            ]
        ),
        lake_root / "gold" / "factor_correlation_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [{"as_of_date": "2026-06-01", "current_regime": "TREND_UP", "created_at": start}]
        ),
        lake_root / "gold" / "market_regime_daily",
    )
    fresh_cost_ts = datetime.now(UTC)
    cost_rows = [
        {
            "day": "2026-06-01",
            "symbol": "BTC-USDT",
            "regime": "normal",
            "event_type": "fill",
            "notional_bucket": "all",
            "sample_count": 40,
            "fee_bps_p75": 0.5,
            "slippage_bps_p75": 0.2,
            "spread_bps_p75": 0.3,
            "total_cost_bps_p75": 1.0,
            "fallback_level": "actual_okx_fills_and_bills",
            "source": "actual_okx_fills_and_bills",
            "actual_fill_count": 40,
            "created_at": fresh_cost_ts,
        }
    ]
    for symbol in ["BNB-USDT", "ETH-USDT", "SOL-USDT"]:
        cost_rows.append(
            {
                "day": "2026-06-01",
                "symbol": symbol,
                "regime": "normal",
                "event_type": "spread_proxy",
                "notional_bucket": "all",
                "sample_count": 60,
                "fee_bps_p75": 0.0,
                "slippage_bps_p75": 0.0,
                "spread_bps_p75": 1.5,
                "total_cost_bps_p75": 1.5,
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "source": "public_spread_proxy",
                "proxy_sample_count": 60,
                "created_at": fresh_cost_ts,
            }
        )
    cost_rows.append(
        {
            "day": "2026-05-20",
            "symbol": "SOL-USDT",
            "regime": "normal",
            "event_type": "fill_proxy",
            "notional_bucket": "all",
            "sample_count": 4,
            "fee_bps_p75": 10.0,
            "slippage_bps_p75": 1.0,
            "spread_bps_p75": 1.0,
            "total_cost_bps_p75": 12.0,
            "fallback_level": "SPREAD_PROXY",
            "source": "mixed_actual_proxy",
            "mixed_fill_count": 4,
            "created_at": fresh_cost_ts - timedelta(days=7),
        }
    )
    write_parquet_dataset(pl.DataFrame(cost_rows), lake_root / "gold" / "cost_bucket_daily")
    write_parquet_dataset(pl.DataFrame(spreads), lake_root / "silver" / "orderbook_spread_1m")
    write_parquet_dataset(pl.DataFrame(trades), lake_root / "silver" / "trade_activity_1m")

    result = export_daily_pack(
        export_date="2026-06-01",
        lake_root=lake_root,
        out_dir=out_dir,
        profile="expert",
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        factor_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/factor_forward_validation.csv").decode("utf-8")
                )
            )
        )
        bridge_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/factor_strategy_bridge_candidates.csv").decode("utf-8")
                )
            )
        )
        cost_coverage_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/live_universe_cost_coverage.csv").decode("utf-8")
                )
            )
        )
        fast_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/fast_microstructure_forward_test.csv").decode("utf-8")
                )
            )
        )
        fast_candidate_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/fast_microstructure_strategy_candidates.csv").decode(
                        "utf-8"
                    )
                )
            )
        )
        fast_review_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/fast_microstructure_strategy_review.csv").decode(
                        "utf-8"
                    )
                )
            )
        )
        advisory_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        questions = archive.read("expert_questions.md").decode("utf-8")

    assert any(
        row["recommendation"] == "FORWARD_VALIDATION_PASS"
        and float(row["regime_stability"]) > 0
        and float(row["cost_adjusted_score"]) > 0
        for row in factor_rows
    )
    bridge_review_rows = [
        row
        for row in bridge_rows
        if row["recommended_action"] == "REVIEW_FOR_ALPHA_FACTORY_STRATEGY"
    ]
    assert bridge_review_rows
    assert {"symbol", "regime", "horizon", "horizon_hours"}.issubset(
        bridge_review_rows[0]
    )
    assert any(
        row["eligible_for_alpha_factory"] == "strategy_review_pending"
        for row in bridge_review_rows
    )
    assert not any(
        "factor_strategy_bridge_candidates dataset is missing_or_empty" in warning
        for warning in data_quality["warnings"]
    )
    bridge_review_candidate_ids = {
        row["bridge_candidate_id"] for row in bridge_review_rows if row["bridge_candidate_id"]
    }
    bridge_advisory_rows = [
        row
        for row in advisory_rows
        if row["strategy_candidate"] in bridge_review_candidate_ids
    ]
    assert bridge_advisory_rows
    assert bridge_review_candidate_ids.issubset(
        {row["strategy_candidate"] for row in bridge_advisory_rows}
    )
    assert not any(
        row["strategy_candidate"].startswith("v5.fast_microstructure_bridge.")
        for row in bridge_advisory_rows
    )
    assert all(row["decision"] == "KEEP_SHADOW" for row in bridge_advisory_rows)
    assert all(row["recommended_mode"] == "shadow" for row in bridge_advisory_rows)
    assert all(row["max_live_notional_usdt"] == "0.0" for row in bridge_advisory_rows)
    assert all(
        row["live_order_effect"] == "read_only_no_live_order"
        for row in bridge_advisory_rows
    )
    assert not any(
        "forward_validation_not_passed" in row["blocking_reasons"]
        or "regime_stability_not_positive_or_missing" in row["blocking_reasons"]
        for row in bridge_review_rows
    )
    lake_bridge = read_parquet_dataset(lake_root / "gold" / "factor_strategy_bridge_candidates")
    lake_advisory = read_parquet_dataset(lake_root / "gold" / "strategy_opportunity_advisory")
    assert lake_bridge.height == len(bridge_rows)
    lake_advisory_candidates = set(lake_advisory["strategy_candidate"].to_list())
    assert bridge_review_candidate_ids.issubset(lake_advisory_candidates)
    bnb_coverage = next(row for row in cost_coverage_rows if row["symbol"] == "BNB-USDT")
    readiness_coverage = data_quality["quant_lab_enforce_readiness"]["metrics"][
        "actual_or_mixed_cost_coverage_live_universe"
    ]
    assert bnb_coverage["effective_cost_source"] == "public_spread_proxy"
    assert float(bnb_coverage["actual_or_mixed_cost_coverage_live_universe"]) == readiness_coverage
    assert readiness_coverage == 0.25
    sol_coverage = next(row for row in cost_coverage_rows if row["symbol"] == "SOL-USDT")
    assert sol_coverage["stale_actual_or_mixed"].lower() == "true"
    assert sol_coverage["actual_or_mixed_direct"].lower() == "false"
    assert sol_coverage["anchored_mixed_proxy_candidate"].lower() == "true"
    assert sol_coverage["mixed_proxy_eligible"].lower() == "true"
    assert (
        sol_coverage["coverage_reason"]
        == "stale_actual_or_mixed_with_anchored_proxy_not_counted"
    )
    checks = {check["name"]: check for check in data_quality["checks"]}
    stale_live_cost = checks["live_universe_stale_actual_or_mixed_cost"]
    assert stale_live_cost["status"] == "WARN"
    assert "stale_actual_or_mixed_symbols=['SOL-USDT']" in stale_live_cost["detail"]
    assert "coverage_status=WARNING" in stale_live_cost["detail"]
    assert "stale_actual_or_mixed=SOL-USDT" in questions
    assert "不要把 public_spread_proxy 当 actual/mixed" in questions
    assert any(
        row["feature_name"] == "orderbook_imbalance_1m"
        and row["recommendation"] == "OOS_VALIDATION_WEAK_OR_MIXED"
        and row["lookback_bars"] == "2000"
        and int(row["effective_sample_count"]) > 0
        and row["purge_embargo_hours"] in {"1", "4", "8"}
        and row["oos_validation_pass"].lower() == "false"
        and float(row["build_elapsed_ms"]) >= 0.0
        for row in fast_rows
    )
    assert not fast_candidate_rows
    assert len(fast_review_rows) == 1
    assert fast_review_rows[0]["recommended_stage"] == "NO_SHADOW_REVIEW"
    assert fast_review_rows[0]["response_action"] == "diagnostic_only"
    assert fast_review_rows[0]["live_order_effect"] == "read_only_no_live_order"


def test_export_daily_pack_includes_expanded_universe_paper_advisory_from_maturity(
    tmp_path,
):
    lake_root = _fixture_lake(tmp_path)
    generated_at = datetime(2026, 5, 11, 8, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-11",
                    "generated_at": generated_at,
                    "symbol": "HYPE-USDT",
                    "strategy_candidate": "Alpha6Factor",
                    "universe_type": "expanded_paper",
                    "maturity_state": "KEEP_SHADOW",
                    "recommended_mode": "shadow",
                    "sample_count": 62,
                    "complete_sample_count": 61,
                    "avg_net_bps": 54.4,
                    "p25_net_bps": -350.5,
                    "win_rate": 0.72,
                    "cost_source_mix": '{"public_spread_proxy": 62}',
                },
                {
                    "as_of_date": "2026-05-11",
                    "generated_at": generated_at,
                    "symbol": "WLD-USDT",
                    "strategy_candidate": "Alpha6Factor",
                    "universe_type": "expanded_paper",
                    "maturity_state": "PAPER_READY",
                    "recommended_mode": "paper",
                    "sample_count": 62,
                    "complete_sample_count": 61,
                    "avg_net_bps": 230.5,
                    "p25_net_bps": 456.4,
                    "win_rate": 0.8,
                    "max_paper_notional_usdt": 80.0,
                    "cost_source_mix": '{"public_spread_proxy": 62}',
                    "cost_quality": "public_proxy",
                },
            ]
        ),
        lake_root / "gold" / "expanded_universe_candidate_maturity",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        advisory = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )

    by_strategy_id = {row["strategy_id"]: row for row in advisory}
    assert "HYPE_EXPANDED_UNIVERSE_PAPER_V1" not in by_strategy_id
    wld = by_strategy_id["WLD_EXPANDED_UNIVERSE_PAPER_V1"]
    assert wld["strategy_candidate"] == "v5.expanded_universe_wld_paper"
    assert wld["symbol"] == "WLD-USDT"
    assert wld["decision"] == "PAPER_READY"
    assert wld["recommended_mode"] == "paper"
    assert wld["universe_type"] == "expanded_paper"
    assert wld["expanded_universe_maturity_state"] == "PAPER_READY"
    assert wld["max_live_notional_usdt"] == "0.0"
    assert wld["max_paper_notional_usdt"] == "80.0"


def test_api_latency_summary_export_includes_cache_and_payload_fields(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/strategy-opportunity-advisory",
        status_code=200,
        duration_seconds=0.123,
        cache_hit=True,
        rows_returned=233,
        response_bytes=12_345,
        lake_scan_ms=1.2,
        serialize_ms=3.4,
        source_signature_ms=2.1,
        response_cache_hit=True,
        dependency_meta_missing=True,
    )

    frame = daily_export_module._api_latency_summary_for_export(lake)
    rows = {row["endpoint"]: row for row in frame.to_dicts()}
    row = rows["/v1/strategy-opportunity-advisory"]

    assert "cache_hit_rate" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "success_p95_ms" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "error_p95_ms" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "avg_source_signature_ms" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "response_cache_hit_rate" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "dependency_meta_missing_count" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "dependency_meta_missing_rate" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "auth_error_count" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert "auth_error_rate" in CSV_SCHEMAS["reports/api_latency_summary.csv"]
    assert row["count"] == 1
    assert row["success_p95_ms"] == 123.0
    assert row["error_p95_ms"] == ""
    assert row["cache_hit_rate"] == 1.0
    assert row["avg_rows_returned"] == 233.0
    assert row["avg_response_bytes"] == 12345.0
    assert row["avg_lake_scan_ms"] == 1.2
    assert row["avg_serialize_ms"] == 3.4
    assert row["avg_source_signature_ms"] == 2.1
    assert row["response_cache_hit_rate"] == 1.0
    assert row["dependency_meta_missing_count"] == 1
    assert row["dependency_meta_missing_rate"] == 1.0
    assert row["error_count"] == 0


def test_api_latency_summary_error_count_does_not_double_count_status_error(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    for status_code, error_type, duration, auth_result in (
        (200, None, 0.050, "token_ok"),
        (401, "HTTPException", 0.200, "missing_bearer_token"),
    ):
        record_api_request(
            lake_root=lake,
            method="GET",
            path="/v1/strategy-opportunity-advisory",
            status_code=status_code,
            duration_seconds=duration,
            error_type=error_type,
            auth_result=auth_result,
        )

    frame = daily_export_module._api_latency_summary_for_export(lake)
    rows = {row["endpoint"]: row for row in frame.to_dicts()}

    assert rows["__all__"]["count"] == 2
    assert rows["__all__"]["error_count"] == 1
    assert rows["__all__"]["success_p95_ms"] == 50.0
    assert rows["__all__"]["error_p95_ms"] == 200.0
    assert rows["__all__"]["auth_error_count"] == 1
    assert rows["__all__"]["auth_error_rate"] == 0.5
    assert rows["/v1/strategy-opportunity-advisory"]["count"] == 2
    assert rows["/v1/strategy-opportunity-advisory"]["error_count"] == 1
    assert rows["/v1/strategy-opportunity-advisory"]["success_p95_ms"] == 50.0
    assert rows["/v1/strategy-opportunity-advisory"]["error_p95_ms"] == 200.0


def test_api_latency_summary_export_window_is_configurable(
    tmp_path,
    monkeypatch,
):
    lake = tmp_path / "lake"
    now = datetime.now(UTC)
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_ROWS", "1")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_FLUSH_SECONDS", "3600")
    monkeypatch.setenv("QUANT_LAB_API_METRICS_EXPORT_WINDOW_MINUTES", "30")
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/strategy-opportunity-advisory",
        status_code=401,
        duration_seconds=0.001,
        auth_result="missing_bearer_token",
        request_ts=now - timedelta(hours=2),
    )
    record_api_request(
        lake_root=lake,
        method="GET",
        path="/v1/strategy-opportunity-advisory",
        status_code=200,
        duration_seconds=0.050,
        auth_result="token_ok",
        request_ts=now,
    )

    frame = daily_export_module._api_latency_summary_for_export(lake)
    rows = {row["endpoint"]: row for row in frame.to_dicts()}

    assert rows["__all__"]["count"] == 1
    assert rows["__all__"]["auth_error_count"] == 0
    assert rows["__all__"]["auth_error_rate"] == 0.0


def test_api_latency_summary_triggers_live_metrics_flush_when_token_configured(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, *_args):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["authorization"] = request.headers.get("Authorization")
        captured["headers"] = dict(request.header_items())
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "unit-test-token")
    monkeypatch.setattr(daily_export_module.urllib.request, "urlopen", fake_urlopen)

    daily_export_module._api_latency_summary_for_export(tmp_path / "lake")

    assert captured["url"] == "http://127.0.0.1:8027/v1/ops/api-metrics"
    assert captured["authorization"] == "Bearer unit-test-token"
    headers = {str(key).lower(): value for key, value in dict(captured["headers"]).items()}
    assert headers["x-quant-lab-client-id"] == "quant-lab.daily_export"
    assert headers["user-agent"] == "quant-lab-daily-export/1.0"
    assert captured["timeout"] == 1.5


def test_snapshot_meta_written_for_api_dependency_datasets(tmp_path):
    lake = tmp_path / "lake"
    generated_at = datetime(2026, 5, 31, 10, tzinfo=UTC)
    frames = {
        "strategy_opportunity_advisory": pl.DataFrame(
            [
                {
                    "strategy_candidate": "v5.f4_volume_expansion_entry",
                    "symbol": "SOL-USDT",
                    "generated_at": generated_at,
                    "expires_at": generated_at + timedelta(hours=1),
                }
            ]
        ),
        "risk_permission": pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "as_of_ts": generated_at,
                }
            ]
        ),
        "gate_decision": pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "created_at": generated_at,
                }
            ]
        ),
        "cost_health_daily": pl.DataFrame(
            [
                {
                    "date": generated_at.date().isoformat(),
                    "created_at": generated_at,
                }
            ]
        ),
    }
    snapshot = daily_export_module._DatasetSnapshot(
        frames=frames,
        row_counts={name: frame.height for name, frame in frames.items()},
        warnings=[],
    )

    snapshot = daily_export_module._publish_risk_permission_dependency_meta_snapshot(
        lake,
        snapshot,
    )
    warnings = daily_export_module._write_source_snapshot_metas(lake, snapshot.frames)

    assert warnings == []
    for dataset in daily_export_module.SNAPSHOT_META_DATASETS:
        dataset_path = lake / daily_export_module.readers.DATASET_PATHS.get(
            dataset,
            Path("gold") / dataset,
        )
        meta = json.loads((dataset_path / "_snapshot_meta.json").read_text(encoding="utf-8"))
        assert meta["dataset"] == dataset
        assert "source_sha" in meta

    dependency = read_parquet_dataset(lake / "gold/risk_permission_api_dependency_meta")
    row = dependency.to_dicts()[0]
    assert row["strategy"] == "v5"
    assert row["version"] == "5.0.0"
    assert row["source_sha"]


def test_bnb_paper_summary_uses_latest_strategy_day_view():
    from quant_lab.strategy_telemetry.analyze import _latest_bnb_paper_strategy_daily

    frame = pl.DataFrame(
        [
            {
                "strategy_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                "paper_date": "2026-05-31",
                "entry_count": 2,
                "bundle_ts": datetime(2026, 5, 31, 10, tzinfo=UTC),
            },
            {
                "strategy_id": "BNB_F3_DOMINANT_ENTRY_PAPER_V1",
                "paper_date": "2026-05-31",
                "entry_count": 7,
                "bundle_ts": datetime(2026, 5, 31, 12, tzinfo=UTC),
            },
        ]
    )

    latest = _latest_bnb_paper_strategy_daily(frame)
    summary = daily_export_module._bnb_paper_strategy_summary_md(latest)

    assert latest.height == 1
    assert latest["entry_count"][0] == 7
    assert "- entry_count: 7" in summary


def test_strategy_opportunity_advisory_adds_hype_wld_expanded_paper_rows():
    generated_at = datetime(2026, 6, 4, 10, tzinfo=UTC)
    frame = daily_export_module._strategy_opportunity_advisory_for_export(
        alpha_discovery_board=pl.DataFrame(),
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        expanded_universe_maturity=pl.DataFrame(
            [
                {
                    "symbol": "HYPE-USDT",
                    "expanded_universe_maturity_state": "PAPER_READY",
                    "generated_at": generated_at,
                    "sample_count": 33,
                    "complete_sample_count": 31,
                    "avg_net_bps": 88.0,
                    "max_paper_notional_usdt": 25.0,
                },
                {
                    "symbol": "WLD-USDT",
                    "maturity_state": "PAPER_READY",
                    "generated_at": generated_at,
                },
            ]
        ),
    )

    rows = {row["strategy_id"]: row for row in frame.to_dicts()}
    hype = rows["HYPE_EXPANDED_UNIVERSE_PAPER_V1"]
    wld = rows["WLD_EXPANDED_UNIVERSE_PAPER_V1"]
    assert hype["strategy_candidate"] == "v5.expanded_universe_hype_paper"
    assert hype["universe_type"] == "expanded_paper"
    assert hype["recommended_mode"] == "paper"
    assert hype["decision"] == "PAPER_READY"
    assert hype["would_enter"] is True
    assert hype["max_paper_notional_usdt"] == 25.0
    assert hype["max_live_notional_usdt"] == 0.0
    assert hype["live_order_effect"] == "read_only_no_live_order"
    assert wld["strategy_candidate"] == "v5.expanded_universe_wld_paper"
    assert wld["max_live_notional_usdt"] == 0.0


def test_strategy_opportunity_advisory_adds_bottom_zone_probe_paper_row():
    generated_at = datetime(2026, 6, 16, 10, tzinfo=UTC)
    frame = daily_export_module._strategy_opportunity_advisory_for_export(
        alpha_discovery_board=pl.DataFrame(),
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        bottom_zone_reversal_shadow=pl.DataFrame(
            [
                {
                    "generated_at": generated_at,
                    "symbol": "BNB-USDT",
                    "ts_utc": generated_at,
                    "bottom_zone_state": "BOTTOM_PROBE_ALLOWED",
                    "would_probe_paper": True,
                    "bottom_zone_score": 0.78,
                    "bounce_probability_4h": 0.78,
                    "live_order_effect": "read_only_no_live_order",
                },
                {
                    "generated_at": generated_at,
                    "symbol": "TON-USDT",
                    "ts_utc": generated_at,
                    "bottom_zone_state": "BOTTOM_PROBE_ALLOWED",
                    "would_probe_paper": True,
                    "bottom_zone_score": 0.81,
                    "bounce_probability_4h": 0.81,
                    "live_order_effect": "read_only_no_live_order",
                },
            ]
        ),
    )

    rows = {row["symbol"]: row for row in frame.to_dicts()}
    row = rows["BNB-USDT"]
    expanded = rows["TON-USDT"]

    assert row["strategy_id"] == "BOTTOM_ZONE_PROBE_PAPER_V1"
    assert row["strategy_candidate"] == "v5.bottom_zone_probe_paper"
    assert row["symbol"] == "BNB-USDT"
    assert row["universe_type"] == "v5_live_universe"
    assert row["decision"] == "PAPER_READY"
    assert row["recommended_mode"] == "paper"
    assert row["would_enter"] is True
    assert row["max_paper_notional_usdt"] > 0
    assert row["max_live_notional_usdt"] == 0.0
    assert row["live_order_effect"] == "read_only_no_live_order"
    assert expanded["universe_type"] == "expanded_paper"
    assert expanded["recommended_mode"] == "paper"
    assert expanded["max_live_notional_usdt"] == 0.0
    assert expanded["live_order_effect"] == "read_only_no_live_order"


def test_export_daily_pack_publishes_bottom_zone_canonical_gold_and_api(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(lake_root))
    api_main._STRATEGY_OPPORTUNITY_ADVISORY_CACHE.clear()
    api_main._STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE.clear()
    generated_at = datetime(2026, 6, 18, 8, tzinfo=UTC)
    bottom_zone = pl.DataFrame(
        [
            {
                "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
                "schema_version": "bottom_zone_reversal_shadow.v0.1",
                "symbol": "TON-USDT",
                "ts_utc": generated_at.isoformat().replace("+00:00", "Z"),
                "bottom_zone_state": "BOTTOM_PROBE_ALLOWED",
                "would_probe_paper": True,
                "bottom_zone_score": 0.82,
                "bounce_probability_score": 0.82,
                "bounce_probability_4h": 0.82,
                "live_order_effect": "read_only_no_live_order",
            }
        ],
        infer_schema_length=None,
    )
    bottom_zone_calls = []

    def fake_bottom_zone_reversal_shadow(**_):
        bottom_zone_calls.append(True)
        return bottom_zone

    monkeypatch.setattr(
        daily_export_module,
        "build_bottom_zone_reversal_shadow",
        fake_bottom_zone_reversal_shadow,
    )

    result = export_daily_pack(
        export_date="2026-06-18",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        advisory_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )
        acceptance_rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/system_acceptance_dashboard.csv").decode("utf-8"))
            )
        )

    zip_bottom = [
        row
        for row in advisory_rows
        if row["strategy_candidate"] == "v5.bottom_zone_probe_paper"
        and row["symbol"] == "TON-USDT"
    ]
    assert zip_bottom
    assert zip_bottom[0]["max_live_notional_usdt"] == "0.0"
    assert zip_bottom[0]["live_order_effect"] == "read_only_no_live_order"
    assert len(bottom_zone_calls) == 1
    acceptance = {row["check_name"]: row for row in acceptance_rows}
    assert acceptance["bottom_zone_paper_v5_rows_ok"]["status"] == "PASS"
    assert "TON-USDT" in acceptance["bottom_zone_paper_v5_rows_ok"]["observed_value"]

    lake_advisory = read_parquet_dataset(lake_root / "gold" / "strategy_opportunity_advisory")
    lake_bottom = lake_advisory.filter(
        (pl.col("strategy_candidate") == "v5.bottom_zone_probe_paper")
        & (pl.col("symbol") == "TON-USDT")
    )
    assert lake_bottom.height == 1
    meta = json.loads(
        (lake_root / "gold" / "strategy_opportunity_advisory" / "_snapshot_meta.json").read_text(
            encoding="utf-8"
        )
    )
    assert meta["row_count"] == lake_advisory.height
    assert meta["source_sha"]

    client = TestClient(app)
    full = client.get("/v1/strategy-opportunity-advisory")
    compact = client.get("/v1/strategy-opportunity-advisory/v5-compact")
    api_main._STRATEGY_OPPORTUNITY_ADVISORY_CACHE.clear()
    api_main._STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE.clear()

    assert full.status_code == 200
    assert compact.status_code == 200
    for payload in (full.json(), compact.json()):
        row = next(
            item
            for item in payload
            if item["strategy_candidate"] == "v5.bottom_zone_probe_paper"
            and item["symbol"] == "TON-USDT"
        )
        assert row["decision"] == "PAPER_READY"
        assert row["recommended_mode"] == "paper"
        assert row["max_live_notional_usdt"] == 0.0
        assert row["live_order_effect"] == "read_only_no_live_order"


def test_bottom_zone_probe_paper_readiness_applies_paper_thresholds():
    generated_at = datetime(2026, 6, 17, 10, tzinfo=UTC)
    frame = daily_export_module._bottom_zone_probe_paper_readiness_for_export(
        paper_daily=pl.DataFrame(
            [
                {
                    "paper_date": "2026-06-16",
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "BNB-USDT",
                    "paper_days_to_date": 14,
                    "entry_count": 20,
                    "avg_paper_pnl_bps": 32.5,
                },
                {
                    "paper_date": "2026-06-16",
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "IP-USDT",
                    "paper_days_to_date": 7,
                    "entry_count": 6,
                    "avg_paper_pnl_bps": -4.0,
                },
            ]
        ),
        paper_runs=pl.DataFrame(
            [
                {
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "BNB-USDT",
                    "paper_pnl_bps": 10.0,
                },
                {
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "BNB-USDT",
                    "paper_pnl_bps": 30.0,
                },
                {
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "BNB-USDT",
                    "paper_pnl_bps": 50.0,
                },
                {
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "IP-USDT",
                    "paper_pnl_bps": -70.0,
                },
                {
                    "strategy_id": "BOTTOM_ZONE_PROBE_PAPER_V1",
                    "strategy_candidate": "v5.bottom_zone_probe_paper",
                    "symbol": "IP-USDT",
                    "paper_pnl_bps": -30.0,
                },
            ]
        ),
        generated_at=generated_at,
    )

    rows = {row["symbol"]: row for row in frame.to_dicts()}
    assert rows["BNB-USDT"]["readiness_status"] == "READY_FOR_REVIEW"
    assert rows["BNB-USDT"]["recommended_stage"] == "PAPER_REVIEW"
    assert rows["BNB-USDT"]["blocking_reasons"] == "[]"
    assert rows["BNB-USDT"]["live_order_effect"] == "read_only_no_live_order"
    ip_reasons = json.loads(rows["IP-USDT"]["blocking_reasons"])
    assert rows["IP-USDT"]["readiness_status"] == "BLOCKED"
    assert rows["IP-USDT"]["recommended_stage"] == "KEEP_PAPER_SHADOW"
    assert ip_reasons == [
        "paper_days_lt_14",
        "paper_entries_lt_20",
        "avg_pnl_not_positive",
        "p25_not_above_minus_50",
    ]


def test_strategy_opportunity_advisory_adds_bridge_review_shadow_rows():
    generated_at = datetime(2026, 6, 16, 10, tzinfo=UTC)
    frame = daily_export_module._strategy_opportunity_advisory_for_export(
        alpha_discovery_board=pl.DataFrame(),
        strategy_evidence=pl.DataFrame(),
        paper_proposals=pl.DataFrame(),
        risk_permissions=pl.DataFrame(),
        cost_health=pl.DataFrame(),
        paper_daily=pl.DataFrame(),
        paper_slippage=pl.DataFrame(),
        factor_strategy_bridge_candidates=pl.DataFrame(
            [
                {
                    "as_of_date": "2026-06-16",
                    "generated_at": generated_at,
                    "factor_id": "core.mean_reversion_vol_adjusted_4",
                    "factor_family": "core",
                    "symbol": "BNB-USDT",
                    "regime": "SIDEWAYS",
                    "horizon": "4h",
                    "horizon_hours": 4,
                    "forward_sample_count": 44,
                    "forward_cost_adjusted_score": 18.5,
                    "bridge_candidate_id": (
                        "v5.factor_bridge.core.mean_reversion_vol_adjusted_4"
                    ),
                    "eligible_for_alpha_factory": "strategy_review_pending",
                    "blocking_reasons": (
                        '["needs_strategy_formulation","needs_paper_tracking"]'
                    ),
                    "recommended_action": "REVIEW_FOR_ALPHA_FACTORY_STRATEGY",
                    "live_order_effect": "read_only_no_live_order",
                },
                {
                    "as_of_date": "2026-06-16",
                    "generated_at": generated_at,
                    "factor_id": "fast_microstructure.orderbook_imbalance_1m",
                    "factor_family": "fast_microstructure",
                    "symbol": "SOL-USDT",
                    "regime": "RISK_ON",
                    "horizon": "4h",
                    "horizon_hours": 4,
                    "forward_sample_count": 58,
                    "forward_cost_adjusted_score": 24.0,
                    "bridge_candidate_id": (
                        "v5.fast_microstructure_bridge.orderbook_imbalance_1m."
                        "sol_usdt.risk_on.4h"
                    ),
                    "eligible_for_alpha_factory": "strategy_review_pending",
                    "blocking_reasons": (
                        '["needs_strategy_formulation","needs_paper_tracking"]'
                    ),
                    "recommended_action": "REVIEW_FOR_ALPHA_FACTORY_STRATEGY",
                    "live_order_effect": "read_only_no_live_order",
                },
                {
                    "as_of_date": "2026-06-16",
                    "generated_at": generated_at,
                    "symbol": "BTC-USDT",
                    "bridge_candidate_id": "v5.factor_bridge.display_only",
                    "eligible_for_alpha_factory": "false",
                    "blocking_reasons": '["forward_validation_not_passed"]',
                    "recommended_action": "DISPLAY_ONLY_FACTOR_REVIEW",
                    "live_order_effect": "read_only_no_live_order",
                },
            ]
        ),
    )

    rows = {row["strategy_candidate"]: row for row in frame.to_dicts()}
    factor = rows["v5.factor_bridge.core.mean_reversion_vol_adjusted_4"]
    fast = rows[
        "v5.fast_microstructure_bridge.orderbook_imbalance_1m.sol_usdt.risk_on.4h"
    ]

    assert set(rows) == {
        "v5.factor_bridge.core.mean_reversion_vol_adjusted_4",
        "v5.fast_microstructure_bridge.orderbook_imbalance_1m.sol_usdt.risk_on.4h",
    }
    assert factor["decision"] == "KEEP_SHADOW"
    assert factor["recommended_mode"] == "shadow"
    assert factor["would_enter"] is False
    assert factor["max_paper_notional_usdt"] == 0.0
    assert factor["max_live_notional_usdt"] == 0.0
    assert "needs_paper_tracking" in factor["live_block_reasons"]
    assert fast["template_family"] == "fast_microstructure_bridge"
    assert fast["live_order_effect"] == "read_only_no_live_order"


def test_system_acceptance_requires_expanded_reader_runs_and_daily_rows():
    generated_at = datetime(2026, 6, 4, 10, tzinfo=UTC)
    advisory = pl.DataFrame(
        [
            {
                "strategy_id": "HYPE_EXPANDED_UNIVERSE_PAPER_V1",
                "strategy_candidate": "v5.expanded_universe_hype_paper",
                "symbol": "HYPE-USDT",
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "universe_type": "expanded_paper",
                "generated_at": generated_at,
                "expires_at": generated_at + timedelta(hours=2),
            }
        ]
    )
    reader = pl.DataFrame(
        [{"symbol": "HYPE-USDT", "strategy_id": "HYPE_EXPANDED_UNIVERSE_PAPER_V1"}]
    )
    runs = pl.DataFrame([{"symbol": "HYPE-USDT", "strategy_id": "HYPE_EXPANDED_UNIVERSE_PAPER_V1"}])

    failed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={
            "strategy_opportunity_advisory": advisory,
            "expanded_universe_advisory_reader": reader,
            "expanded_universe_paper_runs": runs,
        },
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    failed_row = next(
        row
        for row in failed.to_dicts()
        if row["check_name"] == "expanded_universe_paper_v5_rows_ok"
    )
    assert failed_row["status"] == "FAIL"
    assert "missing_daily=['HYPE-USDT']" in failed_row["observed_value"]

    passed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={
            "strategy_opportunity_advisory": advisory,
            "expanded_universe_advisory_reader": reader,
            "expanded_universe_paper_runs": runs,
            "expanded_universe_paper_daily": pl.DataFrame(
                [
                    {
                        "symbol": "HYPE-USDT",
                        "strategy_id": "HYPE_EXPANDED_UNIVERSE_PAPER_V1",
                        "entry_count": 1,
                    }
                ]
            ),
        },
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    passed_row = next(
        row
        for row in passed.to_dicts()
        if row["check_name"] == "expanded_universe_paper_v5_rows_ok"
    )
    assert passed_row["status"] == "PASS"


def test_system_acceptance_requires_no_sample_reason_when_expanded_entry_count_zero():
    generated_at = datetime(2026, 6, 4, 10, tzinfo=UTC)
    advisory = pl.DataFrame(
        [
            {
                "strategy_id": "WLD_EXPANDED_UNIVERSE_PAPER_V1",
                "strategy_candidate": "v5.expanded_universe_wld_paper",
                "symbol": "WLD-USDT",
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "universe_type": "expanded_paper",
                "generated_at": generated_at,
                "expires_at": generated_at + timedelta(hours=2),
            }
        ]
    )
    reader = pl.DataFrame([{"symbol": "WLD-USDT", "strategy_id": "WLD_EXPANDED_UNIVERSE_PAPER_V1"}])
    runs_without_reason = pl.DataFrame(
        [
            {
                "symbol": "WLD-USDT",
                "strategy_id": "WLD_EXPANDED_UNIVERSE_PAPER_V1",
                "would_enter": False,
            }
        ]
    )
    daily_zero = pl.DataFrame(
        [{"symbol": "WLD-USDT", "strategy_id": "WLD_EXPANDED_UNIVERSE_PAPER_V1", "entry_count": 0}]
    )

    failed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={
            "strategy_opportunity_advisory": advisory,
            "expanded_universe_advisory_reader": reader,
            "expanded_universe_paper_runs": runs_without_reason,
            "expanded_universe_paper_daily": daily_zero,
        },
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    failed_row = next(
        row
        for row in failed.to_dicts()
        if row["check_name"] == "expanded_universe_paper_v5_rows_ok"
    )
    assert failed_row["status"] == "FAIL"
    assert "entry_count_zero_missing_no_sample_reason=['WLD-USDT']" in failed_row["observed_value"]
    assert "no_sample_reason_mix={'not_observable': 1}" in failed_row["observed_value"]

    runs_with_reason = pl.DataFrame(
        [
            {
                "symbol": "WLD-USDT",
                "strategy_id": "WLD_EXPANDED_UNIVERSE_PAPER_V1",
                "would_enter": False,
                "no_sample_reason": "cost_source_global_default",
            }
        ]
    )
    passed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={
            "strategy_opportunity_advisory": advisory,
            "expanded_universe_advisory_reader": reader,
            "expanded_universe_paper_runs": runs_with_reason,
            "expanded_universe_paper_daily": daily_zero,
        },
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    passed_row = next(
        row
        for row in passed.to_dicts()
        if row["check_name"] == "expanded_universe_paper_v5_rows_ok"
    )
    assert passed_row["status"] == "PASS"
    assert "no_sample_reason_mix={'cost_source_global_default': 1}" in passed_row["observed_value"]


def test_system_acceptance_label_join_uses_pending_ratio():
    generated_at = datetime(2026, 6, 4, 10, tzinfo=UTC)
    all_pending = pl.DataFrame(
        [
            {"symbol": "BNB-USDT", "future_4h_net_bps": None},
            {"symbol": "BNB-USDT", "future_4h_net_bps": None},
        ]
    )
    partial = pl.DataFrame(
        [
            {"symbol": "BNB-USDT", "future_4h_net_bps": 12.0},
            {"symbol": "BNB-USDT", "future_4h_net_bps": None},
        ]
    )
    failed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={"bnb_strong_alpha6_bypass_shadow": all_pending},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    failed_row = next(
        row for row in failed.to_dicts() if row["check_name"] == "bnb_bypass_label_join_ok"
    )
    assert failed_row["status"] == "FAIL"
    assert "pending_ratio=1.000000" in failed_row["observed_value"]
    assert "label_join_failure_reasons={'not_observable': 2}" in failed_row["observed_value"]

    passed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={"bnb_strong_alpha6_bypass_shadow": partial},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    passed_row = next(
        row for row in passed.to_dicts() if row["check_name"] == "bnb_bypass_label_join_ok"
    )
    assert passed_row["status"] == "PASS"
    assert "pending_ratio=0.500000" in passed_row["observed_value"]

    diagnostic_failure = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "future_4h_net_bps": None,
                "label_join_failure_reason": "missing_run_id",
            },
            {
                "symbol": "BNB-USDT",
                "future_4h_net_bps": None,
                "label_join_failure_reason": "nearest_label_too_far",
            },
        ]
    )
    diagnostic = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={"bnb_strong_alpha6_bypass_shadow": diagnostic_failure},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    diagnostic_row = next(
        row for row in diagnostic.to_dicts() if row["check_name"] == "bnb_bypass_label_join_ok"
    )
    assert (
        "label_join_failure_reasons={'missing_run_id': 1, 'nearest_label_too_far': 1}"
        in diagnostic_row["observed_value"]
    )


def test_system_acceptance_fast_microstructure_core_observability():
    generated_at = datetime(2026, 6, 4, 10, tzinfo=UTC)
    fast = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "orderbook_imbalance_1m": 0.1,
                "orderbook_imbalance_5m": 0.2,
                "taker_buy_sell_imbalance_5m": 0.3,
                "cvd_5m": 4.0,
                "cvd_divergence": 0.0,
                "spread_bps_change_5m": -0.5,
            }
        ]
    )
    dashboard = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={"fast_microstructure_features": fast},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=generated_at,
    )
    row = next(
        row
        for row in dashboard.to_dicts()
        if row["check_name"] == "fast_microstructure_core_observability_ok"
    )
    assert row["status"] == "PASS"
    assert "not_observable_ratio=0.000000" in row["observed_value"]


def test_system_acceptance_requires_api_auth_error_rate_below_one_percent():
    dashboard = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(
            [
                {
                    "endpoint": "__all__",
                    "count": 200,
                    "p95_ms": 12.0,
                    "auth_error_count": 1,
                    "auth_error_rate": 0.005,
                }
            ]
        ),
        lake_file_count=0,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    row = next(
        row
        for row in dashboard.to_dicts()
        if row["check_name"] == "api_auth_error_rate_ok"
    )
    assert row["status"] == "PASS"
    assert "auth_error_rate=0.005000" in row["observed_value"]

    failed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(
            [
                {
                    "endpoint": "__all__",
                    "count": 100,
                    "p95_ms": 12.0,
                    "auth_error_count": 1,
                    "auth_error_rate": 0.01,
                }
            ]
        ),
        lake_file_count=0,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    failed_row = next(
        row
        for row in failed.to_dicts()
        if row["check_name"] == "api_auth_error_rate_ok"
    )
    assert failed_row["status"] == "FAIL"


def test_system_acceptance_lake_file_count_and_growth_thresholds():
    warning = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=6_000,
        lake_file_growth_24h_count=251,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    checks = {row["check_name"]: row for row in warning.to_dicts()}
    assert checks["lake_parquet_file_count_under_threshold"]["status"] == "WARNING"
    assert checks["lake_parquet_file_growth_24h_ok"]["status"] == "WARNING"

    failed = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=7_501,
        lake_file_growth_24h_count=250,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    failed_checks = {row["check_name"]: row for row in failed.to_dicts()}
    assert failed_checks["lake_parquet_file_count_under_threshold"]["status"] == "FAIL"
    assert failed_checks["lake_parquet_file_growth_24h_ok"]["status"] == "PASS"


def test_system_acceptance_downgrades_stale_v5_bundle_to_warning():
    dashboard = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={"authoritative_snapshot": False, "stale_v5_bundle": True},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=datetime(2026, 6, 4, 10, tzinfo=UTC),
    )
    row = next(
        row
        for row in dashboard.to_dicts()
        if row["check_name"] == "v5_bundle_sync_ok"
    )
    assert row["status"] == "WARNING"
    assert "downgrade V5-derived conclusions" in row["next_action"]


def test_system_acceptance_passes_synced_non_authoritative_v5_bundle():
    dashboard = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={"authoritative_snapshot": False, "stale_v5_bundle": False},
        data_quality_warnings=[],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    row = next(
        row
        for row in dashboard.to_dicts()
        if row["check_name"] == "v5_bundle_sync_ok"
    )
    assert row["status"] == "PASS"
    assert "pre-export V5 refresh was disabled" in row["next_action"]
    assert "rerun telemetry sync" not in row["next_action"]


def test_system_acceptance_surfaces_blocked_enforce_readiness():
    dashboard = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[
            "quant_lab_enforce_readiness: readiness_status=BLOCKED; "
            "blocked=['actual_or_mixed_cost_coverage_live_universe']"
        ],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    row = next(
        row
        for row in dashboard.to_dicts()
        if row["check_name"] == "enforce_readiness_ok"
    )

    assert row["status"] == "FAIL"
    assert "readiness_status=BLOCKED" in row["observed_value"]
    assert "restore blocked readiness inputs" in row["next_action"]


def test_system_acceptance_surfaces_warn_enforce_readiness():
    dashboard = daily_export_module.build_system_acceptance_dashboard(
        frames={},
        report_frames={},
        row_counts={},
        pre_export_v5={},
        data_quality_warnings=[
            "quant_lab_enforce_readiness: readiness_status=WARN; warnings=['cost']"
        ],
        api_latency_summary=pl.DataFrame(),
        lake_file_count=0,
        generated_at=datetime(2026, 6, 11, 10, tzinfo=UTC),
    )
    row = next(
        row
        for row in dashboard.to_dicts()
        if row["check_name"] == "enforce_readiness_ok"
    )

    assert row["status"] == "WARNING"
    assert "readiness_status=WARN" in row["observed_value"]


def test_v5_bundle_sync_uses_latest_ingested_dataset_timestamp():
    generated_at = datetime(2026, 6, 11, 6, tzinfo=UTC)
    frames = {
        "strategy_health_daily": pl.DataFrame(
            [
                {
                    "date": "2026-06-11",
                    "latest_bundle_ts": "2026-06-11T05:33:11Z",
                }
            ]
        )
    }
    pre_export_v5 = {
        "sync_attempted": True,
        "sync_succeeded": True,
        "latest_v5_bundle_seen_at_export": "2026-06-11T05:29:51Z",
        "selected_v5_bundle_built_at": "2026-06-11T05:29:51Z",
        "selected_v5_bundle_ingested_at": "2026-06-11T05:20:00Z",
        "selected_v5_bundle_path": "/tmp/v5_latest.tar.gz",
        "selected_v5_bundle_sha256": "abc123",
        "selected_v5_bundle_manifest_match": True,
        "authoritative_snapshot": True,
        "stale_v5_bundle": True,
    }

    consistency = daily_export_module._v5_export_consistency(
        frames,
        pre_export_v5=pre_export_v5,
        pre_export_v5_refresh=True,
        allow_stale_v5=False,
    )
    diagnostics = daily_export_module._v5_bundle_sync_diagnostics_frame(
        frames=frames,
        pre_export_v5={**pre_export_v5, **consistency},
        generated_at=generated_at,
    )

    row = diagnostics.to_dicts()[0]
    assert consistency["stale_v5_bundle"] is False
    assert consistency["why_stale"] == ""
    assert row["latest_ingested_bundle_ts"] == "2026-06-11T05:33:11+00:00"
    assert row["latest_remote_bundle_ts"] == "2026-06-11T05:29:51+00:00"
    assert row["latest_uploaded_bundle_ts"] == "2026-06-11T05:29:51+00:00"
    assert row["selected_v5_bundle_ts"] == "2026-06-11T05:29:51+00:00"
    assert row["stale_seconds"] == 0
    assert row["why_stale"] == ""
    assert row["selected_v5_bundle_sha"] == "abc123"


def test_v5_bundle_sync_uses_selected_manifest_ingest_for_uploaded_latest_bundle():
    generated_at = datetime(2026, 6, 15, 13, tzinfo=UTC)
    frames = {
        "strategy_health_daily": pl.DataFrame(
            [
                {
                    "date": "2026-06-15",
                    "latest_bundle_ts": "2026-06-15T11:00:00Z",
                }
            ]
        )
    }
    pre_export_v5 = {
        "sync_attempted": True,
        "sync_succeeded": True,
        "latest_v5_bundle_seen_at_export": "2026-06-15T12:00:00Z",
        "selected_v5_bundle_built_at": "2026-06-15T12:00:00Z",
        "selected_v5_bundle_ingested_at": "2026-06-15T12:00:30Z",
        "selected_v5_bundle_path": "/tmp/v5_latest.tar.gz",
        "selected_v5_bundle_sha256": "latest-sha",
        "selected_v5_bundle_manifest_match": True,
    }

    consistency = daily_export_module._v5_export_consistency(
        frames,
        pre_export_v5=pre_export_v5,
        pre_export_v5_refresh=True,
        allow_stale_v5=False,
    )
    diagnostics = daily_export_module._v5_bundle_sync_diagnostics_frame(
        frames=frames,
        pre_export_v5={**pre_export_v5, **consistency},
        generated_at=generated_at,
    )

    row = diagnostics.to_dicts()[0]
    assert consistency["stale_v5_bundle"] is False
    assert consistency["authoritative_snapshot"] is True
    assert row["latest_uploaded_bundle_ts"] == "2026-06-15T12:00:00+00:00"
    assert row["selected_v5_bundle_ts"] == "2026-06-15T12:00:00+00:00"
    assert row["latest_ingested_bundle_ts"] == "2026-06-15T12:00:30+00:00"
    assert row["why_stale"] == ""


def test_v5_bundle_sync_keeps_refresh_disabled_provenance_out_of_failures():
    generated_at = datetime(2026, 6, 15, 13, tzinfo=UTC)
    frames = {
        "strategy_health_daily": pl.DataFrame(
            [
                {
                    "date": "2026-06-15",
                    "latest_bundle_ts": "2026-06-15T12:00:30Z",
                }
            ]
        )
    }
    pre_export_v5 = {
        "sync_attempted": False,
        "latest_v5_bundle_seen_at_export": "2026-06-15T12:00:00Z",
        "selected_v5_bundle_built_at": "2026-06-15T12:00:00Z",
        "selected_v5_bundle_ingested_at": "2026-06-15T12:00:30Z",
        "selected_v5_bundle_path": "/tmp/v5_latest.tar.gz",
        "selected_v5_bundle_sha256": "latest-sha",
        "selected_v5_bundle_manifest_match": True,
    }

    consistency = daily_export_module._v5_export_consistency(
        frames,
        pre_export_v5=pre_export_v5,
        pre_export_v5_refresh=False,
        allow_stale_v5=False,
    )
    diagnostics = daily_export_module._v5_bundle_sync_diagnostics_frame(
        frames=frames,
        pre_export_v5={**pre_export_v5, **consistency},
        generated_at=generated_at,
    )

    row = diagnostics.to_dicts()[0]
    assert consistency["authoritative_snapshot"] is False
    assert consistency["stale_v5_bundle"] is False
    assert consistency["warning_reason"] == ""
    assert (
        "pre_export_v5_refresh_disabled"
        in consistency["selected_v5_bundle_authoritative_reason"]
    )
    assert row["why_stale"] == ""
    assert row["failure_reason"] == ""


def test_v5_bundle_sync_flags_remote_newer_than_ingested():
    frames = {
        "strategy_health_daily": pl.DataFrame(
            [
                {
                    "date": "2026-06-11",
                    "latest_bundle_ts": "2026-06-11T05:10:00Z",
                }
            ]
        )
    }
    pre_export_v5 = {
        "latest_v5_bundle_seen_at_export": "2026-06-11T05:29:51Z",
        "selected_v5_bundle_path": "/tmp/v5_old.tar.gz",
        "selected_v5_bundle_sha256": "old",
        "authoritative_snapshot": True,
    }

    consistency = daily_export_module._v5_export_consistency(
        frames,
        pre_export_v5=pre_export_v5,
        pre_export_v5_refresh=True,
        allow_stale_v5=False,
    )

    assert consistency["stale_v5_bundle"] is True
    assert consistency["why_stale"] == "remote_newer_than_ingested"
    assert "why_stale=remote_newer_than_ingested" in consistency["warning_reason"]


def test_market_export_uses_rollup_frames_without_raw_shape():
    trade_rollup = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "minute_ts": datetime(2026, 5, 31, 10, tzinfo=UTC),
                "trade_count": 2,
                "size_sum": 1.5,
                "latest_trade_ts": datetime(2026, 5, 31, 10, 0, 30, tzinfo=UTC),
            },
            {
                "symbol": "BNB-USDT",
                "minute_ts": datetime(2026, 5, 31, 10, 1, tzinfo=UTC),
                "trade_count": 3,
                "size_sum": 2.5,
                "latest_trade_ts": datetime(2026, 5, 31, 10, 1, 30, tzinfo=UTC),
            },
        ]
    )
    spread_rollup = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "channel": "books5",
                "minute_ts": datetime(2026, 5, 31, 10, tzinfo=UTC),
                "spread_bps": 4.0,
                "ts": datetime(2026, 5, 31, 10, tzinfo=UTC),
            },
            {
                "symbol": "BNB-USDT",
                "channel": "books5",
                "minute_ts": datetime(2026, 5, 31, 10, 1, tzinfo=UTC),
                "spread_bps": 5.0,
                "ts": datetime(2026, 5, 31, 10, 1, tzinfo=UTC),
            },
        ]
    )

    trade = daily_export_module._trade_activity_export_table(trade_rollup).to_dicts()[0]
    spread = daily_export_module._orderbook_spread_export_table(spread_rollup).to_dicts()[0]

    assert trade["trade_count"] == 5
    assert trade["size_sum"] == 4.0
    assert spread["spread_bps"] == 5.0


def test_strategy_evidence_coverage_does_not_require_dormant_candidates():
    evidence = pl.DataFrame(
        {
            "candidate_name": [
                "v5.sol_protect_exception",
                "v5.alt_impulse_shadow",
                "v5.swing_f4_f5_alpha6",
                "v5.f3_dominant_entry",
                "v5.f4_volume_expansion_entry",
            ]
        }
    )

    assert _strategy_evidence_has_required_candidates(evidence)


def test_export_includes_v5_local_live_vs_quant_lab_shadow_report(tmp_path):
    lake_root = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 11, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                _bar(entry_ts, symbol="SOL-USDT", close=100.0),
                _bar(entry_ts + timedelta(hours=4), symbol="SOL-USDT", close=102.0),
                _bar(entry_ts + timedelta(hours=8), symbol="SOL-USDT", close=98.0),
                _bar(entry_ts + timedelta(hours=24), symbol="SOL-USDT", close=103.0),
            ]
        ),
        lake_root / "silver" / "market_bar",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-live-1",
                    "ts_utc": entry_ts.isoformat(),
                    "symbol": "SOL-USDT",
                    "normalized_symbol": "SOL-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 100.0,
                    "trade_id": "trade-sol-1",
                    "order_id": "order-sol-1",
                }
            ]
        ),
        lake_root / "silver" / "v5_trade_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-11",
                    "strategy_candidate": "v5.sol_guard_shadow",
                    "symbol": "SOL-USDT",
                    "decision": "KILL",
                    "recommended_mode": "none",
                    "horizon_hours": 24,
                    "sample_count": 30,
                    "complete_sample_count": 30,
                    "avg_net_bps": -20.0,
                    "p25_net_bps": -80.0,
                    "win_rate": 0.3,
                    "cost_source_mix": '{"mixed_actual_proxy":30}',
                    "decision_reasons": '["negative_after_cost_edge"]',
                    "created_at": entry_ts,
                }
            ]
        ),
        lake_root / "gold" / "alpha_discovery_board",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/v5_local_live_vs_quant_lab_shadow.csv").decode("utf-8")
                )
            )
        )

    assert len(rows) == 1
    row = rows[0]
    assert row["trade_id"] == "trade-sol-1"
    assert row["quant_lab_is_live_commander"].lower() == "false"
    assert row["quant_lab_would_block_if_enabled"].lower() == "true"
    assert row["block_outcome_4h"] == "missed_profit_if_blocked"
    assert row["block_outcome_8h"] == "avoided_loss_if_blocked"


def test_export_continues_when_derived_gold_publication_is_denied(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    risk_on_frame = pl.DataFrame(
        [
            {
                "generated_at": datetime(2026, 5, 25, 7, tzinfo=UTC),
                "schema_version": "v5_risk_on_multi_buy_shadow.v0.2",
                "run_id": "run-1",
                "decision_ts": datetime(2026, 5, 25, 6, tzinfo=UTC),
                "current_regime": "ALT_IMPULSE",
                "regime_source": "v5_candidate_event",
                "candidate_buy_count": 2,
                "market_regime_daily_state": "SIDEWAYS",
                "v5_candidate_regime_state": "Trending",
                "trigger_reason": "intraday_v5_candidate_risk_on",
                "broad_market_positive_count": 2,
                "btc_24h_return_bps": 120.0,
                "strategy_candidate": "v5.risk_on_multi_buy_top2_shadow",
                "top_k": 2,
                "selected_symbols": '["BNB-USDT","SOL-USDT"]',
                "selected_count": 2,
                "would_buy_symbol": "BNB-USDT",
                "would_buy": True,
                "would_size_usdt": 0.0,
                "final_score": 0.91,
                "expected_edge_bps": 40.0,
                "required_edge_bps": 10.0,
                "entry_px": 650.0,
                "future_4h_net_bps": 12.0,
                "future_8h_net_bps": 18.0,
                "future_24h_net_bps": 25.0,
                "avg_portfolio_net_bps": 21.5,
                "actual_v5_bought_symbols": '["BNB-USDT"]',
                "vs_actual_v5_net_bps": 4.0,
                "missed_symbols": '["SOL-USDT"]',
                "recommended_mode": "shadow",
                "max_live_notional_usdt": 0.0,
                "source": "quant_lab_risk_on_multi_buy_shadow",
            }
        ]
    )

    monkeypatch.setattr(
        daily_export_module,
        "_v5_missed_opportunity_audit_for_export",
        lambda **_: pl.DataFrame(),
    )
    monkeypatch.setattr(
        daily_export_module,
        "_risk_on_multi_buy_shadow_for_export",
        lambda **_: risk_on_frame,
    )
    original_write = daily_export_module.write_parquet_dataset

    def failing_write(frame, dataset_path, *args, **kwargs):
        if Path(dataset_path).name in {"v5_risk_on_multi_buy_shadow", "risk_on_multi_buy_shadow"}:
            raise PermissionError("simulated lake permission drift")
        return original_write(frame, dataset_path, *args, **kwargs)

    monkeypatch.setattr(daily_export_module, "write_parquet_dataset", failing_write)

    result = export_daily_pack(
        export_date="2026-05-25",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    assert any(
        warning.startswith("v5_risk_on_multi_buy_shadow publish skipped: PermissionError")
        for warning in result.warnings
    )
    with zipfile.ZipFile(result.zip_path) as archive:
        rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/risk_on_multi_buy_shadow.csv").decode("utf-8"))
            )
        )
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert rows[0]["strategy_candidate"] == "v5.risk_on_multi_buy_top2_shadow"
    assert manifest["row_counts"]["v5_risk_on_multi_buy_shadow"] == 1


def test_export_includes_missed_opportunity_and_risk_on_multi_buy_shadow(tmp_path):
    lake_root = tmp_path / "lake"
    entry_ts = datetime(2026, 5, 11, tzinfo=UTC)
    bars = []
    for symbol, close_4h, close_8h, close_24h in [
        ("BTC-USDT", 101.0, 101.5, 102.0),
        ("BNB-USDT", 102.0, 103.0, 104.0),
        ("SOL-USDT", 104.0, 105.0, 106.0),
    ]:
        bars.extend(
            [
                _bar(entry_ts, symbol=symbol, close=100.0),
                _bar(entry_ts + timedelta(hours=4), symbol=symbol, close=close_4h),
                _bar(entry_ts + timedelta(hours=8), symbol=symbol, close=close_8h),
                _bar(entry_ts + timedelta(hours=24), symbol=symbol, close=close_24h),
            ]
        )
    bars.append(_bar(entry_ts - timedelta(hours=24), symbol="BTC-USDT", close=98.0))
    write_parquet_dataset(pl.DataFrame(bars), lake_root / "silver" / "market_bar")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-risk-on",
                    "ts_utc": entry_ts.isoformat(),
                    "symbol": "BNB-USDT",
                    "alpha6_side": "buy",
                    "final_score": 90.0,
                    "expected_edge_bps": 80.0,
                    "required_edge_bps": 30.0,
                    "cost_gate_verified": True,
                    "final_decision": "OPEN_LONG",
                    "block_reason": "",
                    "regime_state": "Trending",
                    "broad_market_positive_count": 4,
                    "strategy_candidate": "v5.bnb_risk_on_shadow",
                },
                {
                    "run_id": "run-risk-on",
                    "ts_utc": entry_ts.isoformat(),
                    "symbol": "SOL-USDT",
                    "alpha6_side": "buy",
                    "final_score": 85.0,
                    "expected_edge_bps": 75.0,
                    "required_edge_bps": 30.0,
                    "cost_gate_verified": True,
                    "final_decision": "BLOCKED",
                    "block_reason": "permission_abort",
                    "regime_state": "Trending",
                    "broad_market_positive_count": 4,
                    "strategy_candidate": "v5.sol_risk_on_shadow",
                },
            ]
        ),
        lake_root / "silver" / "v5_candidate_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-risk-on",
                    "ts_utc": entry_ts.isoformat(),
                    "symbol": "BNB-USDT",
                    "normalized_symbol": "BNB-USDT",
                    "side": "buy",
                    "action": "entry",
                    "price": 100.0,
                    "trade_id": "trade-bnb-1",
                }
            ]
        ),
        lake_root / "silver" / "v5_trade_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-11",
                    "current_regime": "ALT_IMPULSE",
                    "broad_market_positive_count": 4,
                    "btc_24h_return_bps": 200.0,
                    "created_at": entry_ts,
                },
                {
                    "as_of_date": "2026-05-11",
                    "current_regime": "SIDEWAYS",
                    "broad_market_positive_count": 0,
                    "btc_24h_return_bps": -300.0,
                    "created_at": entry_ts + timedelta(hours=12),
                },
                {
                    "as_of_date": "2026-05-12",
                    "current_regime": "RISK_OFF",
                    "broad_market_positive_count": 1,
                    "btc_24h_return_bps": -300.0,
                    "created_at": entry_ts + timedelta(days=1),
                },
            ]
        ),
        lake_root / "gold" / "market_regime_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-11",
                    "strategy_candidate": "v5.bnb_risk_on_shadow",
                    "symbol": "BNB-USDT",
                    "decision": "KILL",
                    "recommended_mode": "none",
                    "horizon_hours": 24,
                    "sample_count": 30,
                    "complete_sample_count": 30,
                    "avg_net_bps": -20.0,
                    "p25_net_bps": -80.0,
                    "win_rate": 0.3,
                    "cost_source_mix": '{"mixed_actual_proxy":30}',
                    "decision_reasons": '["negative_after_cost_edge"]',
                    "created_at": entry_ts,
                },
                {
                    "as_of_date": "2026-05-11",
                    "strategy_candidate": "v5.sol_risk_on_shadow",
                    "symbol": "SOL-USDT",
                    "decision": "KILL",
                    "recommended_mode": "none",
                    "horizon_hours": 24,
                    "sample_count": 30,
                    "complete_sample_count": 30,
                    "avg_net_bps": -20.0,
                    "p25_net_bps": -80.0,
                    "win_rate": 0.3,
                    "cost_source_mix": '{"mixed_actual_proxy":30}',
                    "decision_reasons": '["negative_after_cost_edge"]',
                    "created_at": entry_ts,
                },
            ]
        ),
        lake_root / "gold" / "alpha_discovery_board",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        missed_rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/missed_opportunity_audit.csv").decode("utf-8"))
            )
        )
        risk_on_rows = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/risk_on_multi_buy_shadow.csv").decode("utf-8"))
            )
        )
        advisory_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_opportunity_advisory.csv").decode("utf-8")
                )
            )
        )
        summary = archive.read("reports/missed_opportunity_summary.md").decode("utf-8")
        risk_on_summary = archive.read("reports/risk_on_multi_buy_summary.md").decode("utf-8")

    outcomes = {row["symbol"]: row["outcome_if_blocked"] for row in missed_rows}
    assert outcomes["BNB-USDT"] == "quant_lab_would_have_missed_profit"
    assert outcomes["SOL-USDT"] == "v5_missed_profit_opportunity"
    assert any(
        row["strategy_candidate"] == "v5.risk_on_multi_buy_top1_shadow" for row in risk_on_rows
    )
    assert any(
        row["strategy_candidate"] == "v5.risk_on_multi_buy_top2_shadow" for row in risk_on_rows
    )
    top2 = next(row for row in risk_on_rows if row["top_k"] == "2")
    assert json.loads(top2["selected_symbols"]) == ["BNB-USDT", "SOL-USDT"]
    assert top2["would_buy_symbol"] in {"BNB-USDT", "SOL-USDT"}
    assert top2["actual_v5_bought_symbols"] == '["BNB-USDT"]'
    assert float(top2["avg_portfolio_net_bps"]) > 0
    assert any(
        row["strategy_candidate"] == "v5.risk_on_multi_buy_top2_shadow"
        and row["recommended_mode"] == "shadow"
        for row in advisory_rows
    )
    assert "quant_lab_would_have_missed_profit" in summary
    assert "top2" in risk_on_summary


def test_risk_on_multi_buy_uses_intraday_v5_candidate_regime_when_daily_sideways():
    entry_ts = datetime(2026, 5, 25, 6, tzinfo=UTC)
    bars = []
    for symbol, close_4h, close_8h, close_24h in [
        ("BTC-USDT", 101.0, 102.0, 103.0),
        ("BNB-USDT", 102.0, 103.0, 104.0),
        ("SOL-USDT", 103.0, 104.0, 105.0),
    ]:
        bars.extend(
            [
                _bar(entry_ts, symbol=symbol, close=100.0),
                _bar(entry_ts + timedelta(hours=4), symbol=symbol, close=close_4h),
                _bar(entry_ts + timedelta(hours=8), symbol=symbol, close=close_8h),
                _bar(entry_ts + timedelta(hours=24), symbol=symbol, close=close_24h),
            ]
        )
    candidate_events = pl.DataFrame(
        [
            {
                "run_id": "run-20260525",
                "ts_utc": entry_ts.isoformat(),
                "symbol": "SOL-USDT",
                "alpha6_side": "buy",
                "final_score": 91.0,
                "expected_edge_bps": 90.0,
                "required_edge_bps": 30.0,
                "cost_gate_verified": True,
                "final_decision": "BLOCKED",
                "regime_state": "Trending",
                "broad_market_positive_count": 1,
            },
            {
                "run_id": "run-20260525",
                "ts_utc": entry_ts.isoformat(),
                "symbol": "BNB-USDT",
                "alpha6_side": "buy",
                "final_score": 88.0,
                "expected_edge_bps": 85.0,
                "required_edge_bps": 30.0,
                "cost_gate_verified": True,
                "final_decision": "OPEN_LONG",
                "regime_state": "Trending",
                "broad_market_positive_count": 1,
            },
        ]
    )
    market_regime = pl.DataFrame(
        [
            {
                "as_of_date": "2026-05-25",
                "current_regime": "SIDEWAYS",
                "broad_market_positive_count": 1,
                "btc_24h_return_bps": -20.0,
                "created_at": entry_ts,
            }
        ]
    )

    risk_on = daily_export_module._risk_on_multi_buy_shadow_for_export(
        candidate_events=candidate_events,
        v5_trades=pl.DataFrame(),
        market_bars=pl.DataFrame(bars),
        market_regime=market_regime,
        cost_buckets=pl.DataFrame(),
    )

    assert risk_on.height > 0
    top2 = next(row for row in risk_on.to_dicts() if row["top_k"] == 2)
    assert json.loads(top2["selected_symbols"]) == ["SOL-USDT", "BNB-USDT"]
    assert top2["regime_source"] == "v5_candidate_event"
    assert top2["trigger_reason"] == "intraday_v5_candidate_risk_on"
    assert top2["candidate_buy_count"] == 2
    assert top2["market_regime_daily_state"] == "SIDEWAYS"
    assert top2["v5_candidate_regime_state"] == "TREND_UP"
    assert top2["recommended_mode"] == "shadow"
    assert float(top2["max_live_notional_usdt"]) == 0.0


def test_member_row_count_counts_csv_without_materializing_rows() -> None:
    text = "a,b\n1,2\n3,4\n"

    assert _member_row_count("example.csv", text) == 2
    assert (
        _sha256_payload(text)
        == daily_export_module.hashlib.sha256(text.encode("utf-8")).hexdigest()
    )


def test_csv_text_uses_native_writer_for_scalar_frames(monkeypatch) -> None:
    def fail_python_writer(*args, **kwargs):
        raise AssertionError("primitive CSV frames should use the native writer")

    monkeypatch.setattr(daily_export_module, "_python_csv_text", fail_python_writer)

    text = daily_export_module._csv_text(
        pl.DataFrame(
            {
                "symbol": ["BTC-USDT"],
                "row_count": [2],
                "observed": [True],
            }
        )
    )

    assert text == "symbol,row_count,observed\nBTC-USDT,2,True\n"


def test_csv_text_redacts_sensitive_values_on_native_path() -> None:
    text = daily_export_module._csv_text(
        pl.DataFrame(
            {
                "api_key": ["SHOULD_NOT_LEAK"],
                "normal": ["ok"],
            }
        )
    )

    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows == [{"api_key": "[REDACTED]", "normal": "ok"}]


def test_csv_text_keeps_python_json_serialization_for_complex_values() -> None:
    text = daily_export_module._csv_text(
        pl.DataFrame({"symbol": ["BTC-USDT"], "values": [[1, 2]]})
    )

    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["symbol"] == "BTC-USDT"
    assert json.loads(rows[0]["values"]) == [1, 2]


def test_secret_scan_prefilter_skips_market_rows() -> None:
    assert not daily_export_module._line_may_contain_secret(
        "BTC-USDT,2026-06-07T00:00:00Z,634.2,640.0,620.0,638.5,1234"
    )


def test_secret_scan_prefilter_keeps_known_secret_patterns() -> None:
    examples = [
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQCexample",
        "ed25519 PRIVATE KEY",
        "OK-ACCESS-KEY: live_secret",
        "api_key=live_secret",
        "apiSecret=live_secret",
        "secret_key=live_secret",
        "api_secret=live_secret",
        "passphrase=live_secret",
        "password=live_secret",
        "token=live_secret",
    ]

    for example in examples:
        assert daily_export_module._line_may_contain_secret(example)
        high, medium = daily_export_module._secret_severity_counts(example)
        assert high + medium > 0, example


def test_export_frame_samples_large_unlisted_dataset_without_full_read(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "strategy_evidence"
    dataset_path.mkdir(parents=True)
    for index in range(3):
        pl.DataFrame(
            [
                {
                    "strategy_candidate": f"candidate_{index}",
                    "symbol": "BTC-USDT",
                    "horizon_hours": 24,
                    "avg_net_bps": float(index),
                }
            ]
        ).write_parquet(dataset_path / f"part-{index}.parquet")

    monkeypatch.setenv("QUANT_LAB_EXPORT_FULL_READ_MAX_FILES", "1")

    def fail_full_read(*args, **kwargs):
        raise AssertionError("full dataset read should not be used")

    monkeypatch.setattr(daily_export_module.readers, "read_dataset", fail_full_read)

    frame, row_count, warning = _load_export_frame(lake_root, "strategy_evidence")

    assert warning is None
    assert row_count == 3
    assert frame.height == 3


def test_export_frame_reports_full_row_count_for_sampled_dataset(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    for file_index in range(3):
        rows = [
            {
                "symbol": "BTC-USDT",
                "trade_id": f"{file_index}-{row_index}",
                "price": 100.0,
                "size": 1.0,
                "ts": datetime(2026, 5, 10, file_index, row_index, tzinfo=UTC),
            }
            for row_index in range(5)
        ]
        pl.DataFrame(rows).write_parquet(dataset_path / f"batch-{file_index}.parquet")
    monkeypatch.setitem(daily_export_module.HEAVY_EXPORT_DATASET_LIMITS, "trade_print", 4)

    frame, row_count, warning = _load_export_frame(lake_root, "trade_print")

    assert warning is None
    assert frame.height == 4
    assert row_count == 15


def test_export_frame_uses_lake_file_index_for_rollup_skipped_raw_row_count(
    tmp_path,
    monkeypatch,
):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    for file_index in range(3):
        rows = [
            {
                "symbol": "BTC-USDT",
                "trade_id": f"{file_index}-{row_index}",
                "price": 100.0,
                "size": 1.0,
                "ts": datetime(2026, 5, 10, file_index, row_index, tzinfo=UTC),
            }
            for row_index in range(5)
        ]
        pl.DataFrame(rows).write_parquet(dataset_path / f"batch-{file_index}.parquet")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "minute_ts": datetime(2026, 5, 10, 0, tzinfo=UTC),
                    "trade_count": 15,
                    "notional_usdt": 1500.0,
                }
            ]
        ),
        lake_root / "silver" / "trade_activity_1m",
    )
    build_lake_file_index(lake_root, ["silver/trade_print"])

    def fail_metadata_row_count(_files):
        raise AssertionError("rollup-skipped raw row count should use lake_file_index")

    monkeypatch.setattr(
        daily_export_module,
        "_export_parquet_metadata_row_count",
        fail_metadata_row_count,
    )

    frame, row_count, warning = _load_export_frame(lake_root, "trade_print")

    assert frame.is_empty()
    assert row_count == 15
    assert warning == "trade_print export frame skipped; using trade_activity_1m rollup"


def test_export_frame_uses_lake_file_index_for_file_candidates(
    tmp_path,
    monkeypatch,
):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "strategy_evidence"
    dataset_path.mkdir(parents=True)
    for file_index in range(3):
        pl.DataFrame(
            [
                {
                    "strategy_candidate": f"candidate_{file_index}_{row_index}",
                    "symbol": "BTC-USDT",
                    "horizon_hours": 24,
                    "avg_net_bps": float(row_index),
                }
                for row_index in range(5)
            ]
        ).write_parquet(dataset_path / f"part-{file_index}.parquet")
    build_lake_file_index(lake_root, ["gold/strategy_evidence"])
    daily_export_module._EXPORT_FILE_INDEX_CACHE.clear()
    monkeypatch.setenv("QUANT_LAB_EXPORT_FULL_READ_MAX_FILES", "1")

    def fail_rglob(*args, **kwargs):
        raise AssertionError("export should use lake_file_index instead of rglob")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    frame, row_count, warning = _load_export_frame(lake_root, "strategy_evidence")

    assert warning is None
    assert row_count == 15
    assert frame.height == 15


def test_export_frame_skips_okx_public_ws_raw_when_health_rollup_exists(
    tmp_path,
    monkeypatch,
):
    lake_root = tmp_path / "lake"
    raw_path = lake_root / "bronze" / "okx_public_ws"
    raw_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "channel": "trades",
                "received_at": datetime(2026, 5, 10, tzinfo=UTC),
            }
        ]
    ).write_parquet(raw_path / "raw.parquet")
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "collector": "okx_public_ws",
                    "status": "OK",
                    "last_message_at": datetime(2026, 5, 10, tzinfo=UTC),
                    "updated_at": datetime(2026, 5, 10, tzinfo=UTC),
                }
            ]
        ),
        lake_root / "bronze" / "collector_health" / "okx_public_ws",
    )
    daily_export_module._EXPORT_FILE_INDEX_CACHE.clear()

    def fail_rglob(*args, **kwargs):
        raise AssertionError("okx_public_ws raw export should not rglob when health exists")

    monkeypatch.setattr(Path, "rglob", fail_rglob)

    frame, row_count, warning = _load_export_frame(lake_root, "okx_public_ws")

    assert frame.is_empty()
    assert row_count == 0
    assert warning == "okx_public_ws export frame skipped; using okx_public_ws_health rollup"
    missing = daily_export_module._missing_dataset_rows(
        {
            "okx_public_ws": frame,
            "okx_public_ws_health": pl.DataFrame(
                [
                    {
                        "collector": "okx_public_ws",
                        "status": "OK",
                    }
                ]
            ),
        }
    )
    assert missing.is_empty()


def test_data_quality_registry_skips_heavy_raw_when_rollups_are_available(
    tmp_path,
    monkeypatch,
):
    captured = {}

    class FakeDataQualitySummary:
        def to_dict(self, *, include_checks=True):
            captured["include_checks"] = include_checks
            return {
                "status": "PASS",
                "generated_at": "2026-05-10T00:00:00+00:00",
                "dataset_count": len(captured["dataset_names"]),
                "check_count": 0,
                "fail_count": 0,
                "warning_count": 0,
                "checks": [],
            }

    def fake_run_data_quality(_lake_root, *, dataset_names, reference_at):
        captured["dataset_names"] = list(dataset_names)
        captured["reference_at"] = reference_at
        return FakeDataQualitySummary()

    monkeypatch.setattr(daily_export_module, "run_data_quality", fake_run_data_quality)
    now = datetime(2026, 5, 10, tzinfo=UTC)
    snapshot = daily_export_module._DatasetSnapshot(
        frames={
            "market_bar": pl.DataFrame(
                [
                    {
                        "venue": "OKX",
                        "symbol": "BTC-USDT",
                        "timeframe": "1H",
                        "ts": now,
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.5,
                        "volume": 1.0,
                        "closed": True,
                    }
                ]
            ),
            "cost_bucket_daily": pl.DataFrame(),
            "cost_health_daily": pl.DataFrame(),
            "gate_decision": pl.DataFrame(),
            "alpha_evidence": pl.DataFrame(),
            "alpha_discovery_board": pl.DataFrame(),
            "strategy_evidence": pl.DataFrame(),
            "v5_candidate_event": pl.DataFrame(),
            "v5_candidate_label": pl.DataFrame(),
            "v5_candidate_quality_daily": pl.DataFrame(),
            "risk_permission": pl.DataFrame(),
        },
        row_counts={
            "market_bar": 1,
            "okx_public_ws": 100,
            "okx_public_ws_health": 1,
            "trade_print": 100,
            "trade_activity_1m": 10,
            "orderbook_snapshot": 100,
            "orderbook_spread_1m": 10,
        },
        warnings=[],
    )

    payload = daily_export_module._data_quality_payload(
        tmp_path / "lake",
        snapshot,
        now.date(),
        now,
    )

    assert "okx_public_ws" not in captured["dataset_names"]
    assert "trade_print" not in captured["dataset_names"]
    assert "orderbook_snapshot" not in captured["dataset_names"]
    assert "okx_public_ws_health" in captured["dataset_names"]
    assert "trade_activity_1m" in captured["dataset_names"]
    assert "orderbook_spread_1m" in captured["dataset_names"]
    assert payload["registry_quality"]["skipped_heavy_datasets"] == [
        "okx_public_ws",
        "orderbook_snapshot",
        "trade_print",
    ]


def test_expanded_universe_event_export_is_treated_as_heavy_dataset(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "expanded_universe_candidate_event"
    dataset_path.mkdir(parents=True)
    for file_index in range(3):
        rows = [
            {
                "candidate_id": f"{file_index}-{row_index}",
                "ts_utc": datetime(2026, 5, 10, file_index, row_index, tzinfo=UTC),
                "symbol": "TRX-USDT",
                "strategy_candidate": "expanded_shadow",
            }
            for row_index in range(5)
        ]
        pl.DataFrame(rows).write_parquet(dataset_path / f"batch-{file_index}.parquet")
    monkeypatch.setitem(
        daily_export_module.HEAVY_EXPORT_DATASET_LIMITS,
        "expanded_universe_candidate_event",
        4,
    )

    def fail_full_read(*args, **kwargs):
        raise AssertionError("expanded universe events should use sampled export reads")

    monkeypatch.setattr(daily_export_module.readers, "read_dataset", fail_full_read)

    frame, row_count, warning = _load_export_frame(
        lake_root,
        "expanded_universe_candidate_event",
    )

    assert warning is None
    assert frame.height == 4
    assert row_count == 15


def test_export_frame_ignores_internal_lake_paths(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    for index in range(3):
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "trade_id": f"live-{index}",
                    "price": 100.0,
                    "size": 1.0,
                    "ts": datetime(2026, 5, 10, index, tzinfo=UTC),
                }
            ]
        ).write_parquet(dataset_path / f"batch-{index}.parquet")
    tmp_dir = dataset_path / "._tmp"
    tmp_dir.mkdir()
    pl.DataFrame(
        [
            {
                "symbol": "TMP-USDT",
                "trade_id": "tmp",
                "price": 1.0,
                "size": 1.0,
                "ts": datetime(2026, 5, 10, 3, tzinfo=UTC),
            }
        ]
    ).write_parquet(tmp_dir / "staged.tmp.parquet")
    backup = dataset_path.parent / "__trade_print_backup_deadbeef"
    backup.mkdir()
    pl.DataFrame(
        [
            {
                "symbol": "BACKUP-USDT",
                "trade_id": "backup",
                "price": 1.0,
                "size": 1.0,
                "ts": datetime(2026, 5, 10, 4, tzinfo=UTC),
            }
        ]
    ).write_parquet(backup / "backup.parquet")
    monkeypatch.setenv("QUANT_LAB_EXPORT_FULL_READ_MAX_FILES", "1")

    frame, row_count, warning = _load_export_frame(lake_root, "trade_print")

    assert warning is None
    assert row_count == 3
    assert set(frame["symbol"].to_list()) == {"BTC-USDT"}


def test_export_marks_core_momentum_as_research_baseline_and_prioritizes_strategy_dashboard(
    tmp_path,
):
    lake_root = _fixture_lake(tmp_path)
    created_at = datetime(2026, 5, 11, 12, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "alpha_id": "v5.core.momentum",
                    "version": "v0.1",
                    "data_version": "market_bar:test",
                    "feature_version": "core:v0.1",
                    "cost_model_version": "cost:v0.1",
                    "universe_id": "okx-major-spot",
                    "start_ts": "2026-05-10T00:00:00+00:00",
                    "end_ts": "2026-05-11T00:00:00+00:00",
                    "coverage": 1.0,
                    "ic_mean": -0.02,
                    "ic_tstat": -1.1,
                    "rank_ic_mean": -0.01,
                    "rank_ic_tstat": -0.7,
                    "oos_sharpe": -0.2,
                    "oos_max_drawdown": 0.12,
                    "edge_cost_ratio": 0.4,
                    "paper_days": 0,
                    "created_at": created_at.isoformat(),
                    "evidence_status": "ok",
                }
            ]
        ),
        lake_root / "gold" / "alpha_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "alpha_id": "v5.core.momentum",
                    "version": "v0.1",
                    "gate_version": "default-v0.1",
                    "status": "DEAD",
                    "passed": False,
                    "reasons": '["non_positive_ic"]',
                    "metrics": '{"ic_mean":-0.02}',
                    "next_action": "discard_or_research_only",
                    "created_at": created_at.isoformat(),
                }
            ]
        ),
        lake_root / "gold" / "gate_decision",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-11",
                    "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
                    "symbol": "SOL-USDT",
                    "horizon_hours": 24,
                    "sample_count": 32,
                    "complete_sample_count": 30,
                    "avg_net_bps": 18.0,
                    "p25_net_bps": -12.0,
                    "win_rate": 0.62,
                    "cost_source_mix": '{"public_spread_proxy":30}',
                    "decision": "PAPER_READY",
                    "decision_reasons": '["paper_candidate"]',
                    "created_at": created_at.isoformat(),
                },
                {
                    "as_of_date": "2026-05-11",
                    "strategy_candidate": "v5.multi_position_k2",
                    "symbol": "BTC-USDT",
                    "horizon_hours": 24,
                    "sample_count": 20,
                    "complete_sample_count": 18,
                    "avg_net_bps": -20.0,
                    "p25_net_bps": -55.0,
                    "win_rate": 0.33,
                    "cost_source_mix": '{"public_spread_proxy":18}',
                    "decision": "KILL",
                    "decision_reasons": '["non_positive_after_cost_edge"]',
                    "created_at": created_at.isoformat(),
                },
            ]
        ),
        lake_root / "gold" / "alpha_discovery_board",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        evidence = list(
            csv.DictReader(io.StringIO(archive.read("research/alpha_evidence.csv").decode("utf-8")))
        )
        gates = list(
            csv.DictReader(io.StringIO(archive.read("research/gate_decisions.csv").decode("utf-8")))
        )
        dashboard = list(
            csv.DictReader(
                io.StringIO(archive.read("reports/strategy_level_dashboard.csv").decode("utf-8"))
            )
        )
        executive_summary = archive.read("executive_summary.md").decode("utf-8")

    baseline_evidence = next(row for row in evidence if row["alpha_id"] == "v5.core.momentum")
    baseline_gate = next(row for row in gates if row["alpha_id"] == "v5.core.momentum")
    assert baseline_evidence["role"] == "research_baseline"
    assert baseline_evidence["not_live_eligible"].lower() == "true"
    assert baseline_evidence["not_global_strategy_gate"].lower() == "true"
    assert baseline_gate["baseline_status"] == "DEAD"
    assert baseline_gate["not_live_eligible"].lower() == "true"
    assert baseline_gate["not_global_strategy_gate"].lower() == "true"
    assert {row["decision"] for row in dashboard} >= {"PAPER_READY", "KILL"}
    assert "Core momentum baseline:" in executive_summary
    assert "Strategy-level dashboard decision counts:" in executive_summary
    assert "Focus strategy candidates:" in executive_summary


def test_export_daily_ingests_pending_v5_inbox_before_snapshot(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    make_tar(
        inbox / "v5_live_followup_bundle_20260517T060000Z.tar.gz",
        {
            "summaries/window_summary.json": (
                '{"run_count": 1, "recent_24h_decision_audit_count": 1}'
            ),
            "reports/candidate_snapshot.csv": (
                "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,"
                "current_position,current_weight,target_weight_raw,"
                "target_weight_after_risk,final_score,rank,f1_mom_5d,f2_mom_20d,"
                "f3_vol_adj_ret,f4_volume_expansion,f5_rsi_trend_confirm,"
                "alpha6_score,alpha6_side,ml_score,mean_reversion_score,"
                "expected_edge_bps,required_edge_bps,cost_bps,cost_source,"
                "eligible_before_filters,final_decision,block_reason,"
                "strategy_candidate\n"
                "c1,run_001,2026-05-17T06:00:00Z,BNB-USDT,trend,LOW,0,0,"
                "0.05,0.05,0.91,1,0.12,0.24,0.31,0.42,0.53,0.74,long,"
                "0.66,0.11,18,5,3.5,quant_lab,true,KEEP_SHADOW,risk_gate,"
                "v5.f3_dominant_entry\n"
            ),
            "summaries/btc_probe_entry_quality_audit.csv": (
                "run_id,entry_ts,entry_px,final_score,expected_edge_bps,required_edge_bps,"
                "btc_trend_score,trend_buy_count,alpha6_score,alpha6_side,"
                "bypassed_negative_expectancy_reason,selected_symbol,selection_mode,"
                "negative_expectancy_state,same_symbol_reentry_bypass,"
                "price_distance_from_recent_low_bps,price_distance_from_recent_high_bps,"
                "anti_chase_flag,entry_quality_status\n"
                "run_001,2026-05-17T06:00:00Z,65000,0.91,18,5,0.8,3,0.74,long,"
                "none,BTC/USDT,market_impulse_probe,ok,"
                "probe_stop_loss_reentry_after_loss,42,180,true,accepted\n"
            ),
        },
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        provenance = json.loads(archive.read("provenance.json").decode("utf-8"))
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        btc_probe_rows = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("v5/v5_btc_probe_entry_quality_audit.csv").decode("utf-8")
                )
            )
        )

    assert manifest["pre_export_v5_refresh"]["processed_bundle_count"] == 1
    assert manifest["authoritative_snapshot"] is True
    assert manifest["stale_v5_bundle"] is False
    assert manifest["selected_v5_bundle_sha256"]
    assert (
        manifest["selected_v5_bundle_manifest_bundle_sha256"]
        == manifest["selected_v5_bundle_sha256"]
    )
    assert (
        manifest["v5_export_consistency"]["selected_v5_bundle_manifest_bundle_sha256"]
        == manifest["selected_v5_bundle_sha256"]
    )
    assert (
        manifest["v5_export_consistency"]["selected_v5_bundle_sha256"]
        == manifest["selected_v5_bundle_sha256"]
    )
    assert provenance["selected_v5_bundle_sha256"] == manifest["selected_v5_bundle_sha256"]
    assert (
        provenance["selected_v5_bundle_manifest_bundle_sha256"]
        == manifest["selected_v5_bundle_sha256"]
    )
    assert manifest["selected_v5_bundle_manifest_match"] is True
    assert (
        manifest["selected_v5_bundle_authoritative_reason"]
        == "selected_bundle_sha_matched_bundle_manifest"
    )
    assert manifest["selected_v5_bundle_manifest_bundle_name"].startswith(
        "v5_live_followup_bundle_20260517T060000Z"
    )
    assert manifest["selected_v5_bundle_built_at"].startswith("2026-05-17T06:00:00")
    assert manifest["selected_v5_bundle_ingested_at"]
    assert manifest["selected_v5_bundle_event_counts"]["v5_candidate_event"] >= 1
    assert manifest["selected_v5_bundle_event_counts"]["v5_btc_probe_entry_quality_audit"] == 1
    assert manifest["latest_v5_bundle_seen_at_export"].startswith("2026-05-17T06:00:00")
    assert manifest["latest_v5_bundle_ingested_at_export"].startswith("2026-05-17T06:00:00")
    assert manifest["candidate_event_rows"] >= 1
    assert manifest["candidate_event_latest_ts"].startswith("2026-05-17T06:00:00")
    assert manifest["latest_candidate_event_ts"].startswith("2026-05-17T06:00:00")
    assert manifest["v5_bundle_lag_seconds"] is not None
    assert data_quality["v5_pre_export"]["processed_bundle_count"] == 1
    assert not any(
        "build_strategy_evidence" in str(warning)
        for warning in manifest["pre_export_v5_refresh"]["warnings"]
    )
    assert btc_probe_rows[0]["same_symbol_reentry_bypass"] == "probe_stop_loss_reentry_after_loss"
    assert btc_probe_rows[0]["anti_chase_flag"] == "true"
    assert btc_probe_rows[0]["normalized_symbol"] == "BTC-USDT"
    assert btc_probe_rows[0]["live_order_effect"] == "none_read_only_v5_bundle_audit"


def test_export_daily_limits_pre_export_v5_ingest_to_latest_pending(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES", "1")
    make_tar(
        inbox / "v5_live_followup_bundle_20260516T120000Z.tar.gz",
        {
            "reports/candidate_snapshot.csv": (
                "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,"
                "current_position,current_weight,target_weight_raw,"
                "target_weight_after_risk,final_score,rank,f1_mom_5d,f2_mom_20d,"
                "f3_vol_adj_ret,f4_volume_expansion,f5_rsi_trend_confirm,"
                "alpha6_score,alpha6_side,ml_score,mean_reversion_score,"
                "expected_edge_bps,required_edge_bps,cost_bps,cost_source,"
                "eligible_before_filters,final_decision,block_reason,"
                "strategy_candidate\n"
                "old,run_old,2026-05-16T12:00:00Z,BTC-USDT,trend,LOW,0,0,"
                "0.05,0.05,0.91,1,0.12,0.24,0.31,0.42,0.53,0.74,long,"
                "0.66,0.11,18,5,3.5,quant_lab,true,KEEP_SHADOW,risk_gate,"
                "v5.old_candidate\n"
            ),
        },
    )
    make_tar(
        inbox / "v5_live_followup_bundle_20260517T060000Z.tar.gz",
        {
            "reports/candidate_snapshot.csv": (
                "candidate_id,run_id,ts_utc,symbol,regime_state,risk_level,"
                "current_position,current_weight,target_weight_raw,"
                "target_weight_after_risk,final_score,rank,f1_mom_5d,f2_mom_20d,"
                "f3_vol_adj_ret,f4_volume_expansion,f5_rsi_trend_confirm,"
                "alpha6_score,alpha6_side,ml_score,mean_reversion_score,"
                "expected_edge_bps,required_edge_bps,cost_bps,cost_source,"
                "eligible_before_filters,final_decision,block_reason,"
                "strategy_candidate\n"
                "new,run_new,2026-05-17T06:00:00Z,BNB-USDT,trend,LOW,0,0,"
                "0.05,0.05,0.91,1,0.12,0.24,0.31,0.42,0.53,0.74,long,"
                "0.66,0.11,18,5,3.5,quant_lab,true,KEEP_SHADOW,risk_gate,"
                "v5.new_candidate\n"
            ),
        },
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
        allow_stale_v5=True,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    refresh = manifest["pre_export_v5_refresh"]
    assert refresh["allow_stale_v5"] is True
    assert refresh["selected_bundle_count"] == 1
    assert refresh["processed_bundle_count"] == 1
    assert refresh["selected_bundle_names"] == ["v5_live_followup_bundle_20260517T060000Z.tar.gz"]
    assert manifest["authoritative_snapshot"] is False
    assert manifest["pending_uningested_bundle_count"] == 1
    assert manifest["candidate_event_latest_ts"].startswith("2026-05-17T06:00:00")


def test_data_quality_fails_when_v5_bundle_is_stale_at_export(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-10",
                    "status": "OK",
                    "latest_bundle_ts": datetime(2026, 5, 10, tzinfo=UTC),
                    "created_at": datetime(2026, 5, 10, tzinfo=UTC),
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["v5_bundle_fresh_at_export"]["status"] == "FAIL"
    assert checks["v5_bundle_fresh_at_export"]["severity"] == "critical"


def test_export_daily_observes_v5_inbox_when_refresh_is_disabled(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    make_tar(
        inbox / "v5_live_followup_bundle_20260517T100000Z.tar.gz",
        {"summaries/window_summary.json": "{}"},
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
        v5_telemetry_config=config,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["pre_export_v5_refresh"]["enabled"] is False
    assert manifest["authoritative_snapshot"] is False
    assert manifest["selected_v5_bundle_manifest_match"] is False
    assert "pre_export_v5_refresh_disabled" in manifest["selected_v5_bundle_authoritative_reason"]
    assert manifest["stale_v5_bundle"] is True
    assert manifest["latest_v5_bundle_seen_at_export"].startswith("2026-05-17T10:00:00")
    assert "latest_candidate_event_ts" in manifest
    assert "v5_bundle_lag_seconds" in manifest


def test_export_daily_fails_when_newer_v5_bundle_cannot_be_ingested(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    make_tar(
        inbox / "v5_live_followup_bundle_20260517T090000Z.tar.gz",
        {
            "reports/candidate_snapshot.csv": (
                "candidate_id,run_id,ts_utc,symbol\nold,run_old,2026-05-17T09:00:00Z,BTC-USDT\n"
            )
        },
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)
    export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
    )
    bad_bundle = inbox / "v5_live_followup_bundle_20260517T100000Z.tar.gz"
    bad_bundle.write_bytes(b"not a tar archive")

    with pytest.raises(RuntimeError, match="stale_v5_bundle_at_export"):
        export_daily_pack(
            export_date="2026-05-17",
            lake_root=lake_root,
            out_dir=tmp_path / "exports",
            command_line=["qlab", "export-daily"],
            pre_export_v5_refresh=True,
            v5_telemetry_config=config,
        )


def test_export_daily_refreshes_v5_derived_outputs_when_latest_bundle_already_ingested(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    latest_bundle = make_tar(
        inbox / "v5_live_followup_bundle_20260517T100000Z.tar.gz",
        {"summaries/window_summary.json": "{}"},
    )
    latest_sha = daily_export_module.compute_sha256(latest_bundle)
    latest_ts = datetime(2026, 5, 17, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": latest_sha,
                    "bundle_name": latest_bundle.name,
                    "bundle_ts": latest_ts,
                    "ingest_ts": datetime(2026, 5, 17, 10, 1, tzinfo=UTC),
                    "schema_version": "v5_quant_lab_telemetry.v0.1",
                }
            ]
        ),
        lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-17",
                    "status": "OK",
                    "latest_bundle_ts": datetime(2026, 5, 17, 9, tzinfo=UTC),
                    "created_at": datetime(2026, 5, 17, 9, tzinfo=UTC),
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    def fake_refresh(lake_root_arg: Path, export_day) -> list[str]:
        assert Path(lake_root_arg) == lake_root
        write_parquet_dataset(
            pl.DataFrame(
                [
                    {
                        "strategy": "v5",
                        "date": str(export_day),
                        "status": "OK",
                        "latest_bundle_ts": latest_ts,
                        "created_at": datetime(2026, 5, 17, 10, 2, tzinfo=UTC),
                    }
                ]
            ),
            lake_root / "gold" / "strategy_health_daily",
        )
        return []

    monkeypatch.setattr(daily_export_module, "_refresh_v5_derived_outputs", fake_refresh)
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["pre_export_v5_refresh"]["processed_bundle_count"] == 0
    assert manifest["pre_export_v5_refresh"]["derived_refresh_retry_attempted"] is True
    assert manifest["stale_v5_bundle"] is False
    assert manifest["authoritative_snapshot"] is True
    assert manifest["selected_v5_bundle_manifest_match"] is True
    assert manifest["latest_v5_bundle_seen_at_export"].startswith("2026-05-17T10:00:00")
    assert manifest["latest_v5_bundle_ingested_at_export"].startswith("2026-05-17T10:00:00")


def test_refresh_v5_derived_outputs_uses_incremental_research_builds(
    tmp_path,
    monkeypatch,
):
    import quant_lab.reports.enforce_readiness as enforce_readiness
    import quant_lab.research.alpha_discovery as alpha_discovery
    import quant_lab.research.alpha_factory as alpha_factory
    import quant_lab.research.candidate_labels as candidate_labels
    import quant_lab.research.diagnostics_refresh as diagnostics_refresh
    import quant_lab.research.entry_quality as entry_quality
    import quant_lab.research.expanded_universe as expanded_universe
    import quant_lab.research.paper_tracking as paper_tracking
    import quant_lab.research.portfolio as portfolio
    import quant_lab.research.regime_router as regime_router
    import quant_lab.research.strategy_evidence as strategy_evidence
    import quant_lab.strategy_telemetry.analyze as telemetry_analyze

    calls: dict[str, list[dict[str, object]]] = {}

    def capture(name: str):
        def _call(*args, **kwargs):
            calls.setdefault(name, []).append({"args": args, "kwargs": kwargs})
            return SimpleNamespace()

        return _call

    monkeypatch.setattr(telemetry_analyze, "analyze_v5_telemetry", capture("telemetry"))
    monkeypatch.setattr(
        candidate_labels,
        "build_and_publish_candidate_labels",
        capture("candidate_labels"),
    )
    monkeypatch.setattr(
        strategy_evidence,
        "build_and_publish_strategy_evidence",
        capture("strategy_evidence"),
    )
    monkeypatch.setattr(
        expanded_universe,
        "build_and_publish_expanded_crypto_universe_shadow",
        capture("expanded_universe"),
    )
    monkeypatch.setattr(
        alpha_discovery,
        "build_and_publish_alpha_discovery_board",
        capture("alpha_discovery"),
    )
    monkeypatch.setattr(
        paper_tracking,
        "build_and_publish_paper_strategy_tracking",
        capture("paper_tracking"),
    )
    monkeypatch.setattr(
        portfolio,
        "build_and_publish_research_portfolio_status",
        capture("portfolio"),
    )
    monkeypatch.setattr(
        diagnostics_refresh,
        "refresh_research_diagnostics",
        capture("diagnostics"),
    )
    monkeypatch.setattr(
        entry_quality,
        "build_and_publish_entry_quality",
        capture("entry_quality"),
    )
    monkeypatch.setattr(
        alpha_factory,
        "build_and_publish_alpha_factory",
        capture("alpha_factory"),
    )
    monkeypatch.setattr(
        regime_router,
        "build_and_publish_regime_router",
        capture("regime_router"),
    )
    monkeypatch.setattr(
        enforce_readiness,
        "write_enforce_readiness_report",
        capture("enforce_readiness"),
    )

    lake_root = tmp_path / "lake"
    export_day = datetime(2026, 5, 17, tzinfo=UTC).date()

    monkeypatch.setenv("QUANT_LAB_EXPORT_DERIVED_REFRESH_IN_PROCESS", "1")
    warnings = daily_export_module._refresh_v5_derived_outputs(lake_root, export_day)

    assert warnings == []
    assert calls["telemetry"][0]["kwargs"] == {
        "lake_root": lake_root,
        "date": export_day.isoformat(),
        "refresh_candidate_gold": False,
    }
    assert calls["candidate_labels"][0]["kwargs"] == {
        "as_of_date": export_day,
        "mode": "incremental",
        "lookback_days": daily_export_module.EXPORT_V5_DERIVED_LOOKBACK_DAYS,
    }
    assert calls["strategy_evidence"][0]["kwargs"] == {
        "as_of_date": export_day.isoformat(),
        "mode": "incremental",
        "lookback_days": daily_export_module.EXPORT_V5_DERIVED_LOOKBACK_DAYS,
        "include_historical_outcomes": False,
    }
    assert all(
        call["kwargs"].get("include_legacy_outcome_counts") is False
        for call in calls["alpha_discovery"]
    )
    assert calls["alpha_factory"][0]["kwargs"] == {
        "as_of_date": export_day,
        "lookback_days": daily_export_module.EXPORT_V5_ALPHA_FACTORY_LOOKBACK_DAYS,
        "max_candidates": daily_export_module.EXPORT_V5_ALPHA_FACTORY_MAX_CANDIDATES,
    }


def test_refresh_v5_derived_outputs_subprocess_uses_web_memory_limit_default(
    tmp_path,
    monkeypatch,
):
    captured: dict[str, object] = {}

    monkeypatch.delenv("QUANT_LAB_EXPORT_DERIVED_REFRESH_STEP_MEMORY_MB", raising=False)
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_MEMORY_LIMIT_MB", "0")
    monkeypatch.setattr(
        daily_export_module,
        "_v5_derived_refresh_step_bodies",
        lambda: [("noop", "print('noop')")],
    )

    def fake_run(args, **kwargs):
        captured["env"] = kwargs["env"]
        return daily_export_module.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(daily_export_module.subprocess, "run", fake_run)

    warnings = daily_export_module._refresh_v5_derived_outputs_subprocess(
        tmp_path / "lake",
        datetime(2026, 6, 15, tzinfo=UTC).date(),
    )

    assert warnings == []
    assert captured["env"]["QUANT_LAB_DERIVED_REFRESH_MEMORY_MB"] == "0"


def test_v5_derived_refresh_subprocess_code_rejects_negative_memory_limit(tmp_path):
    env = {
        **os.environ,
        "QUANT_LAB_DERIVED_REFRESH_LAKE_ROOT": str(tmp_path / "lake"),
        "QUANT_LAB_DERIVED_REFRESH_DATE": "2026-06-15",
        "QUANT_LAB_DERIVED_REFRESH_LOOKBACK_DAYS": "8",
        "QUANT_LAB_DERIVED_REFRESH_ALPHA_FACTORY_LOOKBACK_DAYS": "30",
        "QUANT_LAB_DERIVED_REFRESH_ALPHA_FACTORY_MAX_CANDIDATES": "200",
        "QUANT_LAB_DERIVED_REFRESH_MEMORY_MB": "-1",
    }

    result = daily_export_module.subprocess.run(
        [
            os.sys.executable,
            "-c",
            daily_export_module._v5_derived_refresh_subprocess_code("print('unreachable')"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "QUANT_LAB_DERIVED_REFRESH_MEMORY_MB must be >= 0" in result.stderr


def test_export_daily_non_authoritative_when_selected_bundle_sha_missing_from_manifest(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    bundle = make_tar(
        inbox / "v5_live_followup_bundle_20260517T100000Z.tar.gz",
        {"summaries/window_summary.json": "{}"},
    )
    latest_ts = datetime(2026, 5, 17, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-17",
                    "status": "OK",
                    "latest_bundle_ts": latest_ts,
                    "created_at": latest_ts,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    monkeypatch.setattr(
        daily_export_module,
        "_pending_v5_bundle_scan_for_export",
        lambda inbox_dir, lake_arg: {
            "pending_paths": [],
            "context": {
                "scanned_bundle_count": 1,
                "selected_bundle_count": 0,
                "selected_bundle_names": [],
                "unscanned_bundle_count": 0,
                "oldest_scanned_bundle_ts": latest_ts.isoformat(),
                "newest_scanned_bundle_ts": latest_ts.isoformat(),
                "possible_historical_gap": False,
                "max_scan_bundles": 20,
                "max_pending_bundles": 5,
            },
        },
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    with pytest.raises(RuntimeError, match="selected_v5_bundle_manifest_missing"):
        export_daily_pack(
            export_date="2026-05-17",
            lake_root=lake_root,
            out_dir=tmp_path / "exports",
            command_line=["qlab", "export-daily"],
            pre_export_v5_refresh=True,
            v5_telemetry_config=config,
        )

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
        allow_stale_v5=True,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["selected_v5_bundle_sha256"] == daily_export_module.compute_sha256(bundle)
    assert manifest["stale_v5_bundle"] is False
    assert manifest["authoritative_snapshot"] is False
    assert manifest["selected_v5_bundle_manifest_match"] is False
    assert (
        "selected_v5_bundle_manifest_missing" in manifest["selected_v5_bundle_authoritative_reason"]
    )


def test_authoritative_when_inbox_has_more_than_max_scan_but_all_old_bundles_ingested(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", "20")
    base_ts = datetime(2026, 5, 17, 10, tzinfo=UTC)
    manifests = []
    for index in range(25):
        ts = base_ts - timedelta(hours=index)
        bundle = make_tar(
            inbox / f"v5_live_followup_bundle_{ts:%Y%m%dT%H%M%SZ}.tar.gz",
            {"summaries/window_summary.json": f'{{"index":{index}}}'},
        )
        manifests.append(
            {
                "strategy": "v5",
                "bundle_sha256": daily_export_module.compute_sha256(bundle),
                "bundle_name": bundle.name,
                "bundle_ts": ts,
                "ingest_ts": ts + timedelta(minutes=1),
                "schema_version": "v5_quant_lab_telemetry.v0.1",
            }
        )
    write_parquet_dataset(
        pl.DataFrame(manifests),
        lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-17",
                    "status": "OK",
                    "latest_bundle_ts": base_ts,
                    "created_at": base_ts,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["audit_bundle_count"] == 25
    assert manifest["scanned_bundle_count"] == 20
    assert manifest["unscanned_bundle_count"] == 5
    assert manifest["pending_uningested_bundle_count"] == 0
    assert manifest["possible_historical_gap"] is False
    assert manifest["historical_gap_detected"] is False
    assert manifest["selected_v5_bundle_manifest_match"] is True
    assert manifest["authoritative_snapshot"] is True


def test_export_daily_marks_possible_historical_gap_when_scan_window_is_incomplete(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", "20")
    base_ts = datetime(2026, 5, 17, 10, tzinfo=UTC)
    manifests = []
    for index in range(25):
        ts = base_ts - timedelta(hours=index)
        bundle = make_tar(
            inbox / f"v5_live_followup_bundle_{ts:%Y%m%dT%H%M%SZ}.tar.gz",
            {"summaries/window_summary.json": f'{{"index":{index}}}'},
        )
        if index < 20:
            manifests.append(
                {
                    "strategy": "v5",
                    "bundle_sha256": daily_export_module.compute_sha256(bundle),
                    "bundle_name": bundle.name,
                    "bundle_ts": ts,
                    "ingest_ts": ts + timedelta(minutes=1),
                    "schema_version": "v5_quant_lab_telemetry.v0.1",
                }
            )
    write_parquet_dataset(
        pl.DataFrame(manifests),
        lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-17",
                    "status": "OK",
                    "latest_bundle_ts": base_ts,
                    "created_at": base_ts,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    with pytest.raises(RuntimeError, match="historical_gap_detected"):
        export_daily_pack(
            export_date="2026-05-17",
            lake_root=lake_root,
            out_dir=tmp_path / "exports",
            command_line=["qlab", "export-daily"],
            pre_export_v5_refresh=True,
            v5_telemetry_config=config,
        )

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
        allow_stale_v5=True,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["scanned_bundle_count"] == 20
    assert manifest["audit_bundle_count"] == 25
    assert manifest["unscanned_bundle_count"] == 5
    assert manifest["pending_uningested_bundle_count"] == 5
    assert manifest["possible_historical_gap"] is True
    assert manifest["historical_gap_detected"] is True
    assert manifest["authoritative_snapshot"] is False
    assert "historical_gap_detected" in manifest["selected_v5_bundle_authoritative_reason"]


def test_pending_uningested_count_blocks_authoritative_when_exceeds_max_pending(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    inbox = tmp_path / "inbox"
    monkeypatch.setenv("QUANT_LAB_EXPORT_V5_MAX_PENDING_BUNDLES", "1")
    monkeypatch.setenv("QUANT_LAB_EXPORT_V5_MAX_SCAN_BUNDLES", "10")
    base_ts = datetime(2026, 5, 17, 10, tzinfo=UTC)
    for index in range(3):
        ts = base_ts - timedelta(hours=index)
        make_tar(
            inbox / f"v5_live_followup_bundle_{ts:%Y%m%dT%H%M%SZ}.tar.gz",
            {
                "reports/candidate_snapshot.csv": (
                    "candidate_id,run_id,ts_utc,symbol\n"
                    f"c{index},run_{index},2026-05-17T{10 - index:02d}:00:00Z,"
                    "BTC-USDT\n"
                )
            },
        )
    config = _v5_telemetry_config(tmp_path, inbox, lake_root)

    result = export_daily_pack(
        export_date="2026-05-17",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=True,
        v5_telemetry_config=config,
        allow_stale_v5=True,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["authoritative_snapshot"] is False
    assert manifest["pending_uningested_bundle_count"] > manifest["max_pending_bundles"]
    assert manifest["historical_gap_detected"] is True
    assert "pending_uningested_bundle_count" in manifest["selected_v5_bundle_authoritative_reason"]

    with pytest.raises(RuntimeError, match="pending_uningested_bundle_count"):
        export_daily_pack(
            export_date="2026-05-17",
            lake_root=lake_root,
            out_dir=tmp_path / "exports",
            command_line=["qlab", "export-daily"],
            pre_export_v5_refresh=True,
            v5_telemetry_config=config,
        )


def test_export_daily_pack_uses_unique_same_day_file_names(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    out_dir = tmp_path / "exports"

    first = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=out_dir,
        profile="expert",
        command_line=["qlab", "export-daily"],
    )
    second = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=out_dir,
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    assert first.zip_path != second.zip_path
    assert Path(first.zip_path).exists()
    assert Path(second.zip_path).exists()
    assert Path(first.zip_path).name.startswith("quant_lab_expert_pack_2026-05-11_")
    assert Path(second.zip_path).name.startswith("quant_lab_expert_pack_2026-05-11_")


def test_export_daily_pack_prunes_old_packs_after_generation(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    out_dir = tmp_path / "exports"
    out_dir.mkdir()
    sidecar = out_dir / "v5_enforce_readiness.json"
    sidecar.write_text("{}", encoding="utf-8")
    unrelated_zip = out_dir / "manual_debug_bundle.zip"
    unrelated_zip.write_bytes(b"not an expert pack")
    monkeypatch.setenv("QUANT_LAB_EXPORT_KEEP_PACKS", "3")

    generated = [
        export_daily_pack(
            export_date="2026-05-11",
            lake_root=lake_root,
            out_dir=out_dir,
            profile="expert",
            command_line=["qlab", "export-daily"],
        )
        for _ in range(5)
    ]

    packs = sorted(out_dir.glob("quant_lab_expert_pack_*.zip"))
    kept_names = {path.name for path in packs}

    assert len(packs) == 3
    assert Path(generated[-1].zip_path).name in kept_names
    assert Path(generated[-2].zip_path).name in kept_names
    assert Path(generated[-3].zip_path).name in kept_names
    assert not Path(generated[0].zip_path).exists()
    assert not Path(generated[1].zip_path).exists()
    assert sidecar.exists()
    assert unrelated_zip.exists()


def test_export_empty_csv_members_have_fixed_headers(tmp_path):
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=tmp_path / "empty-lake",
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        for member in [
            "costs/cost_bucket_daily.csv",
            "costs/cost_health_daily.csv",
            "features/feature_snapshot.csv",
            "market/orderbook_spread.csv",
            "market/trade_activity.csv",
            "research/alpha_evidence.csv",
            "research/gate_decisions.csv",
            "research/strategy_evidence_quality.csv",
            "reports/alpha_discovery_board.csv",
            "reports/candidate_kill_list.csv",
            "reports/candidate_shadow_watchlist.csv",
            "reports/candidate_paper_ready.csv",
            "reports/historical_label_threshold_ready.csv",
            "reports/factor_forward_validation.csv",
            "reports/fast_microstructure_forward_test.csv",
            "reports/fast_microstructure_strategy_candidates.csv",
            "reports/fast_microstructure_strategy_review.csv",
            "reports/live_universe_cost_coverage.csv",
            "reports/bottom_zone_probe_paper_readiness.csv",
            "reports/api_error_summary.csv",
            "v5/v5_candidate_events.csv",
            "v5/v5_btc_probe_entry_quality_audit.csv",
            "v5/v5_candidate_labels.csv",
            "v5/v5_candidate_quality.csv",
            "v5/v5_candidate_outcome_summary.csv",
            "v5/v5_expanded_universe_advisory_reader.csv",
            "v5/v5_expanded_universe_paper_runs.csv",
            "v5/v5_expanded_universe_paper_daily.csv",
        ]:
            first_line = archive.read(member).decode("utf-8").splitlines()[0]
            assert first_line == ",".join(CSV_SCHEMAS[member])


def test_data_quality_critical_failures_are_not_downgraded_to_warn(tmp_path):
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=tmp_path / "empty-lake",
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert data_quality["status"] == "CRITICAL"
    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["market_bar_present"]["status"] == "FAIL"
    assert checks["market_bar_present"]["severity"] == "critical"
    assert any(
        failure.startswith("market_bar_present:")
        for failure in data_quality.get("failures", [])
    )


def test_data_quality_marks_warn_enforce_readiness_as_warning(tmp_path, monkeypatch):
    class FakeEnforceReadiness(SimpleNamespace):
        def model_dump(self, *, mode: str = "json") -> dict[str, object]:
            _ = mode
            return dict(self.__dict__)

    lake_root = _fixture_lake(tmp_path)
    monkeypatch.setattr(
        daily_export_module,
        "build_enforce_readiness_report",
        lambda _lake_root: FakeEnforceReadiness(
            readiness_status="WARN",
            blocked_reasons=[],
            warning_reasons=["actual_or_mixed_cost_coverage"],
            checks=[],
            metrics={},
            shadow_only_recommended=True,
        ),
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    readiness_check = checks["quant_lab_enforce_readiness"]
    assert readiness_check["status"] == "WARN"
    assert readiness_check["severity"] == "warning"
    assert data_quality["quant_lab_enforce_readiness"]["readiness_status"] == "WARN"
    assert any("quant_lab_enforce_readiness:" in warning for warning in data_quality["warnings"])
    assert not any(
        "quant_lab_enforce_readiness:" in failure for failure in data_quality.get("failures", [])
    )


def test_data_quality_marks_read_only_live_cost_block_as_pass_diagnostic(tmp_path, monkeypatch):
    class FakeEnforceReadiness(SimpleNamespace):
        def model_dump(self, *, mode: str = "json") -> dict[str, object]:
            _ = mode
            return dict(self.__dict__)

    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "v1",
                    "permission": "ABORT",
                    "allowed_modes": [],
                    "allowed_live_modes": [],
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-v1",
                    "gate_version": "default-v0.1",
                    "reasons": ["fixture_abort"],
                    "created_at": now,
                    "as_of_ts": now,
                    "expires_at": now + timedelta(minutes=30),
                    "permission_status": "ACTIVE_ABORT",
                    "enforceable": True,
                }
            ]
        ),
        lake_root / "gold" / "risk_permission",
    )
    monkeypatch.setattr(
        daily_export_module,
        "build_enforce_readiness_report",
        lambda _lake_root: FakeEnforceReadiness(
            readiness_status="BLOCKED",
            blocked_reasons=["actual_or_mixed_cost_coverage_live_universe"],
            warning_reasons=["actual_or_mixed_cost_coverage_research_universe"],
            checks=[],
            metrics={},
            shadow_only_recommended=True,
        ),
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    readiness_check = checks["quant_lab_enforce_readiness"]
    assert readiness_check["status"] == "PASS"
    assert readiness_check["severity"] == "warning"
    assert "live_order_effect=read_only_no_live_order" in readiness_check["detail"]
    assert not any(
        "quant_lab_enforce_readiness:" in warning
        for warning in data_quality.get("warnings", [])
    )
    assert not any(
        "quant_lab_enforce_readiness:" in failure for failure in data_quality.get("failures", [])
    )


def test_data_quality_keeps_live_allow_cost_block_critical(tmp_path, monkeypatch):
    class FakeEnforceReadiness(SimpleNamespace):
        def model_dump(self, *, mode: str = "json") -> dict[str, object]:
            _ = mode
            return dict(self.__dict__)

    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    _write_risk_permission_for_export(
        lake_root,
        as_of=now,
        expires_at=now + timedelta(minutes=30),
    )
    monkeypatch.setattr(
        daily_export_module,
        "build_enforce_readiness_report",
        lambda _lake_root: FakeEnforceReadiness(
            readiness_status="BLOCKED",
            blocked_reasons=["actual_or_mixed_cost_coverage_live_universe"],
            warning_reasons=[],
            checks=[],
            metrics={},
            shadow_only_recommended=True,
        ),
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    readiness_check = checks["quant_lab_enforce_readiness"]
    assert readiness_check["status"] == "FAIL"
    assert readiness_check["severity"] == "critical"
    assert any(
        "quant_lab_enforce_readiness:" in failure for failure in data_quality.get("failures", [])
    )


def test_cost_fallback_ratio_is_not_pass_when_cost_bucket_daily_is_empty(tmp_path):
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=tmp_path / "empty-lake",
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["cost_fallback_ratio"]["status"] in {"N/A", "FAIL"}


def test_soft_cost_fallbacks_do_not_create_hard_failure(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-11",
                    "symbol": "BNB-USDT",
                    "source": "mixed_actual_proxy",
                    "sample_count": 2,
                    "fallback_level": "SAMPLE_TOO_SMALL;SLIPPAGE_UNKNOWN;SPREAD_PROXY",
                },
                {
                    "day": "2026-05-11",
                    "symbol": "SOL-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 100,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                },
            ]
        ),
        lake_root / "gold" / "cost_bucket_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["cost_hard_fallback_ratio"]["status"] == "PASS"
    assert "ratio=0.00%" in checks["cost_hard_fallback_ratio"]["detail"]
    assert checks["cost_soft_fallback_ratio"]["status"] == "WARN"
    assert checks["cost_fallback_ratio"]["status"] == "PASS"
    assert "diagnostic legacy metric" in checks["cost_fallback_ratio"]["detail"]
    assert "hard_fallback_ratio=0.00%" in checks["cost_fallback_ratio"]["detail"]


def test_dataset_freshness_uses_dataset_specific_timestamps(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    created_at = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-11",
                    "symbol": "ETH-USDT",
                    "regime": "normal",
                    "notional_bucket": "GLOBAL",
                    "created_at": created_at,
                    "sample_count": 1,
                }
            ]
        ),
        lake_root / "gold" / "cost_bucket_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "channel": "tickers",
                    "inst_id": "BTC-USDT",
                    "received_at": created_at,
                    "raw_json": "{}",
                }
            ]
        ),
        lake_root / "bronze" / "okx_public_ws",
    )
    write_parquet_dataset(
        pl.DataFrame([{"payload": "legacy-without-time"}]),
        lake_root / "silver" / "decision_audit",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        provenance = json.loads(archive.read("provenance.json").decode("utf-8"))

    freshness = manifest["dataset_freshness"]
    assert freshness["cost_bucket_daily"]["freshness_status"] != "missing"
    assert freshness["cost_bucket_daily"]["timestamp_column"] == "created_at"
    assert freshness["gate_decision"]["freshness_status"] != "missing"
    assert freshness["risk_permission"]["freshness_status"] != "missing"
    assert freshness["okx_public_ws"]["freshness_status"] != "missing"
    assert freshness["okx_public_ws"]["timestamp_column"] == "received_at"
    assert freshness["decision_audit"]["freshness_status"] == "unknown"
    assert all(
        dataset["freshness_status"] != "missing"
        for dataset in provenance["datasets"]
        if dataset["row_count"] > 0
    )


def test_stale_dataset_check_ignores_optional_entry_quality_history_and_generated_risk_on():
    now = datetime.now(UTC)
    old = now - timedelta(days=5)
    stale = daily_export_module._stale_rows(
        {
            "risk_on_multi_buy_shadow": pl.DataFrame(
                [
                    {
                        "decision_ts": old,
                        "generated_at": now,
                        "strategy_candidate": "v5.risk_on_multi_buy_top1_shadow",
                    }
                ]
            ),
            "v5_entry_quality_history_anti_leakage_check": pl.DataFrame(
                [
                    {
                        "generated_at_utc": old,
                        "check_name": "anti_leakage",
                        "status": "PASS",
                    }
                ]
            ),
            "v5_btc_probe_entry_quality_audit": pl.DataFrame(
                [
                    {
                        "bundle_ts": old,
                        "ingest_ts": old,
                        "source_path_inside_bundle": (
                            "summaries/btc_probe_entry_quality_audit.csv"
                        ),
                        "same_symbol_reentry_bypass": False,
                        "anti_chase_flag": False,
                        "raw_payload_json": "{}",
                    }
                ]
            ),
            "v5_pullback_reversal_shadow": pl.DataFrame(),
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "risk_on_multi_buy_shadow" not in datasets
    assert "v5_entry_quality_history_anti_leakage_check" not in datasets
    assert "v5_btc_probe_entry_quality_audit" not in datasets
    assert "v5_pullback_reversal_shadow" not in datasets


def test_stale_dataset_check_ignores_event_driven_v5_cost_usage_when_current():
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    stale = daily_export_module._stale_rows(
        {
            "strategy_health_daily": pl.DataFrame(
                [
                    {
                        "strategy": "v5",
                        "date": now.date().isoformat(),
                        "status": "OK",
                        "latest_bundle_ts": now,
                    }
                ]
            ),
            "v5_quant_lab_cost_usage": pl.DataFrame(
                [{"strategy": "v5", "bundle_ts": old, "ingest_ts": old}]
            ),
            "v5_quant_lab_fallback": pl.DataFrame(
                [{"strategy": "v5", "bundle_ts": old, "ingest_ts": old}]
            ),
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "v5_quant_lab_cost_usage" not in datasets
    assert "v5_quant_lab_fallback" not in datasets


def test_stale_dataset_check_ignores_okx_bills_when_readonly_backfill_current():
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    stale = daily_export_module._stale_rows(
        {
            "okx_private_readonly_bills": pl.DataFrame(
                [
                    {
                        "endpoint": "/api/v5/account/bills",
                        "ingest_ts": old,
                        "raw_json": "{}",
                    }
                ]
            ),
            "job_run_history": pl.DataFrame(
                [
                    {
                        "day": now.date().isoformat(),
                        "job_name": "okx-backfill-readonly",
                        "status": "succeeded",
                        "started_at": now - timedelta(seconds=5),
                        "finished_at": now,
                        "duration_seconds": 5.0,
                    }
                ]
            ),
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "okx_private_readonly_bills" not in datasets


def test_stale_dataset_check_ignores_closed_research_diagnostics():
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    stale = daily_export_module._stale_rows(
        {
            "research_portfolio_status": pl.DataFrame(
                [
                    {
                        "as_of_date": "2026-05-26",
                        "research_id": "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW",
                        "strategy_candidate": "v5.btc_strict_probe_exit_policy_review",
                        "status": "KILL",
                        "action": "CLOSE_RESEARCH",
                        "created_at": now,
                    }
                ]
            ),
            "btc_probe_exit_policy_review": pl.DataFrame(
                [
                    {
                        "generated_at_utc": old,
                        "strategy_candidate": "v5.btc_strict_probe_exit_policy_review",
                    }
                ]
            ),
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "btc_probe_exit_policy_review" not in datasets


def test_stale_dataset_check_ignores_rollup_covered_heavy_raw_datasets():
    now = datetime.now(UTC)
    stale = daily_export_module._stale_rows(
        {
            "okx_public_ws": pl.DataFrame(),
            "okx_public_ws_health": pl.DataFrame(
                [{"last_message_at": now, "updated_at": now}]
            ),
            "orderbook_snapshot": pl.DataFrame(),
            "orderbook_spread_1m": pl.DataFrame(
                [{"ts": now, "symbol": "BTC-USDT", "spread_bps": 1.2}]
            ),
            "trade_print": pl.DataFrame(),
            "trade_activity_1m": pl.DataFrame(
                [{"latest_trade_ts": now, "symbol": "BTC-USDT", "trade_count": 10}]
            ),
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "okx_public_ws" not in datasets
    assert "orderbook_snapshot" not in datasets
    assert "trade_print" not in datasets


def test_stale_dataset_check_uses_risk_permission_dependency_meta_generated_at():
    now = datetime.now(UTC)
    stale = daily_export_module._stale_rows(
        {
            "risk_permission_api_dependency_meta": pl.DataFrame(
                [
                    {
                        "strategy": "v5",
                        "version": "5.0.0",
                        "generated_at": now,
                    }
                ]
            )
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "risk_permission_api_dependency_meta" not in datasets


def test_stale_dataset_check_uses_factor_bridge_generated_at():
    now = datetime.now(UTC)
    stale = daily_export_module._stale_rows(
        {
            "factor_strategy_bridge_candidates": pl.DataFrame(
                [
                    {
                        "as_of_date": (now - timedelta(days=3)).date().isoformat(),
                        "generated_at": now,
                        "factor_id": "core.volume_zscore_24",
                        "bridge_candidate_id": "v5.factor_bridge.core.volume_zscore_24",
                    }
                ]
            )
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "factor_strategy_bridge_candidates" not in datasets


def test_data_quality_failures_exclude_warning_severity_failures():
    failures = daily_export_module._data_quality_failure_lines(
        [
            {
                "name": "stale_dataset_check",
                "status": "FAIL",
                "severity": "warning",
                "detail": "stale_or_missing=1",
            },
            {
                "name": "market_bar_present",
                "status": "FAIL",
                "severity": "critical",
                "detail": "rows=0",
            },
        ]
    )

    assert failures == ["market_bar_present: rows=0"]


def test_load_snapshot_does_not_warn_for_optional_empty_entry_quality_outputs(tmp_path):
    snapshot = daily_export_module._load_snapshot(tmp_path / "empty-lake")

    assert not any(
        warning.startswith("factor_strategy_bridge_candidates dataset is ")
        for warning in snapshot.warnings
    )
    assert not any(
        warning.startswith("v5_pullback_reversal_shadow dataset is ")
        for warning in snapshot.warnings
    )
    assert not any(
        warning.startswith("v5_entry_quality_history_pullback_by_symbol dataset is ")
        for warning in snapshot.warnings
    )


def test_export_counts_heavy_ws_dataset_without_full_member_load(tmp_path):
    lake_root = tmp_path / "lake"
    base_ts = datetime(2026, 5, 11, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "channel": "tickers",
                    "inst_id": "BTC-USDT",
                    "received_at": base_ts + timedelta(seconds=index),
                    "raw_json": "{}",
                }
                for index in range(6001)
            ]
        ),
        lake_root / "bronze" / "okx_public_ws",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    assert manifest["row_counts"]["okx_public_ws"] == 6001
    assert manifest["dataset_freshness"]["okx_public_ws"]["freshness_status"] != "missing"


def test_recent_heavy_dataset_files_limits_to_newest_parquet_files(tmp_path):
    dataset = tmp_path / "heavy"
    dataset.mkdir()
    paths = []
    for index in range(5):
        path = dataset / f"part-{index}.parquet"
        pl.DataFrame([{"value": index}]).write_parquet(path)
        mtime = 1_700_000_000 + index
        path.touch()
        os.utime(path, (mtime, mtime))
        paths.append(path)

    recent = _recent_heavy_dataset_files(dataset, max_files=2)

    assert recent == paths[-2:]


def test_expert_questions_are_dynamic_for_missing_daily_inputs(tmp_path):
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=tmp_path / "empty-lake",
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        questions = archive.read("expert_questions.md").decode("utf-8")

    assert "是否已配置 OKX WebSocket trades/books 采集？" in questions
    assert "是否启用 OKX read-only fills/bills？" in questions
    assert "为什么 cost_bucket_daily 为空？" in questions
    assert "为什么 gate_decision / alpha_evidence 为空？" in questions
    assert "为什么 risk_permission 为空？" in questions
    assert "V5 telemetry 是否同步成功？" in questions


def test_expert_questions_include_proxy_feature_usage_and_config_gaps(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-11",
                    "symbol": "BTC-USDT",
                    "regime": "normal",
                    "notional_bucket": "GLOBAL",
                    "sample_count": 0,
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "source": "public_spread_proxy",
                    "created_at": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "gold" / "cost_bucket_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-11",
                    "status": "WARNING",
                    "config_not_consumed_count": 2,
                    "config_not_consumed_count_unknown": False,
                    "config_not_consumed_top_keys_json": '["alpha_threshold","risk.max_weight"]',
                    "latest_bundle_ts": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "gold" / "v5_config_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        questions = archive.read("expert_questions.md").decode("utf-8")

    assert "live universe actual/mixed 成本覆盖仅 0.00%" in questions
    assert "未覆盖=BNB-USDT, BTC-USDT, ETH-USDT, SOL-USDT" in questions
    assert "不要把 public_spread_proxy 当 actual/mixed" in questions
    assert "为什么 feature_value 为空？" in questions
    assert "V5 是否已接入 quant-lab API？当前 v5_quant_lab_usage 为空。" in questions
    assert "config_not_consumed_count 是否真实为 2" in questions
    assert "alpha_threshold" in questions


def test_export_alpha_evidence_missing_status_is_unknown(tmp_path):
    lake_root = _fixture_lake(tmp_path)

    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        rows = list(
            csv.DictReader(io.StringIO(archive.read("research/alpha_evidence.csv").decode("utf-8")))
        )

    assert rows[0]["evidence_status"] == "unknown"


def test_export_warns_when_risk_permission_older_than_v5_telemetry(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-12",
                    "status": "OK",
                    "latest_bundle_ts": datetime(2026, 5, 12, 21, tzinfo=UTC),
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        executive_summary = archive.read("executive_summary.md").decode("utf-8")
        risk_flags = list(
            csv.DictReader(io.StringIO(archive.read("risk/risk_flags.csv").decode("utf-8")))
        )

    risk_quality = data_quality["risk_permission"]
    assert risk_quality["permission_status"] == "STALE_ALLOW"
    assert risk_quality["enforceable"] is False
    assert risk_quality["telemetry_latest_ts"].startswith("2026-05-12T21:00:00")
    assert risk_quality["next_action"] == "run qlab publish-risk-permission after sync-v5-telemetry"
    assert any(
        warning.startswith(
            "risk_permission_fresh_vs_v5_telemetry: risk_permission_stale_vs_v5_telemetry"
        )
        for warning in data_quality["warnings"]
    )
    assert "Risk permission status: STALE_ALLOW" in executive_summary
    assert "Latest V5 telemetry ts: 2026-05-12T21:00:00+00:00" in executive_summary
    assert risk_flags[0]["permission_status"] == "STALE_ALLOW"
    assert risk_flags[0]["enforceable"].lower() == "false"


def test_export_prefers_v5_bundle_manifest_over_health_day_boundary(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    manifest_ts = datetime.now(UTC) - timedelta(minutes=5)
    _write_current_and_bootstrap_risk_permissions_for_export(
        lake_root,
        as_of=manifest_ts,
        expires_at=datetime.now(UTC) + timedelta(minutes=20),
    )
    _write_v5_bundle_manifest_for_export(lake_root, manifest_ts)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": manifest_ts.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": datetime.now(UTC) + timedelta(hours=24),
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    result = export_daily_pack(
        export_date=manifest_ts.date().isoformat(),
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        risk_flags = list(
            csv.DictReader(io.StringIO(archive.read("risk/risk_flags.csv").decode("utf-8")))
        )

    risk_quality = data_quality["risk_permission"]
    assert risk_quality["permission_status"] == "ACTIVE_ALLOW"
    assert risk_quality["enforceable"] is True
    assert risk_quality["telemetry_latest_ts"].startswith(manifest_ts.isoformat()[:16])
    assert not any(
        warning.startswith(
            "risk_permission_fresh_vs_v5_telemetry: risk_permission_stale_vs_v5_telemetry"
        )
        for warning in data_quality["warnings"]
    )
    assert risk_flags == []


def test_export_manifest_records_fresh_risk_permission_not_expired(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    _write_risk_permission_for_export(
        lake_root,
        as_of=now,
        expires_at=now + timedelta(minutes=20),
    )

    result = export_daily_pack(
        export_date=now.date().isoformat(),
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert manifest["export_generated_at"]
    assert manifest["risk_permission_generated_at"].startswith(now.isoformat()[:19])
    assert manifest["risk_permission_expires_at"]
    assert manifest["permission_expired_at_export"] is False
    assert checks["risk_permission_not_expired_at_export"]["status"] == "PASS"


def test_export_marks_expired_risk_permission_fail_with_rerun_action(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    _write_risk_permission_for_export(
        lake_root,
        as_of=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )

    result = export_daily_pack(
        export_date=now.date().isoformat(),
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    expired_check = checks["risk_permission_not_expired_at_export"]
    assert manifest["permission_expired_at_export"] is True
    assert data_quality["risk_permission"]["permission_status"] == "EXPIRED_ALLOW"
    assert data_quality["status"] == "CRITICAL"
    assert expired_check["status"] == "FAIL"
    assert "run qlab publish-risk-permission before qlab export-daily" in expired_check["detail"]
    assert any(
        warning.startswith("risk_permission_not_expired_at_export: risk_permission expired")
        for warning in data_quality["warnings"]
    )


def test_export_refreshes_expired_risk_permission_before_snapshot(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    _write_risk_permission_for_export(
        lake_root,
        as_of=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )
    calls = []

    def fake_publish_risk_permission(*, lake_root, strategy, version):
        calls.append((Path(lake_root), strategy, version))
        refreshed_at = datetime.now(UTC)
        _write_risk_permission_for_export(
            Path(lake_root),
            as_of=refreshed_at,
            expires_at=refreshed_at + timedelta(minutes=90),
        )
        return SimpleNamespace(warnings=[])

    monkeypatch.setattr(
        daily_export_module,
        "publish_risk_permission",
        fake_publish_risk_permission,
    )

    result = export_daily_pack(
        export_date=now.date().isoformat(),
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        refresh_risk_permission=True,
        risk_strategy="v5",
        risk_version="5.0.0",
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        executive_summary = archive.read("executive_summary.md").decode("utf-8")

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert calls == [(lake_root, "v5", "5.0.0")]
    assert manifest["permission_expired_at_export"] is False
    assert data_quality["risk_permission"]["permission_status"] == "ACTIVE_ALLOW"
    assert checks["risk_permission_pre_export_refresh"]["status"] == "PASS"
    assert checks["risk_permission_not_expired_at_export"]["status"] == "PASS"
    assert "EXPIRED_ALLOW" not in executive_summary


def test_export_warns_when_risk_permission_republish_fails(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    _write_risk_permission_for_export(
        lake_root,
        as_of=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=1),
    )

    def fail_publish_risk_permission(*, lake_root, strategy, version):
        raise RuntimeError("publisher unavailable")

    monkeypatch.setattr(
        daily_export_module,
        "publish_risk_permission",
        fail_publish_risk_permission,
    )

    result = export_daily_pack(
        export_date=now.date().isoformat(),
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        refresh_risk_permission=True,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        executive_summary = archive.read("executive_summary.md").decode("utf-8")

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert data_quality["risk_permission"]["permission_status"] == "EXPIRED_ALLOW"
    assert checks["risk_permission_pre_export_refresh"]["status"] == "WARN"
    assert (
        "risk_permission_republish_failed" in checks["risk_permission_pre_export_refresh"]["detail"]
    )
    assert checks["risk_permission_not_expired_at_export"]["status"] == "FAIL"
    assert any(
        "risk_permission_republish_failed" in warning for warning in data_quality["warnings"]
    )
    assert "Risk permission counts:" in executive_summary
    assert "EXPIRED_ALLOW" in executive_summary
    assert "ACTIVE_ALLOW" not in executive_summary


def test_export_warns_on_risk_version_mismatch_with_v5_telemetry(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "abc",
                    "bundle_name": "bundle.tar.gz",
                    "source_path_inside_bundle": "raw/reports/quant_lab_usage.jsonl",
                    "row_index": 0,
                    "strategy_version": "5.0.0",
                    "raw_payload_json": '{"strategy_version":"5.0.0"}',
                    "ingest_ts": datetime.now(UTC),
                    "bundle_ts": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "silver" / "v5_quant_lab_usage",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert any(
        "risk_permission_version_matches_v5" in warning for warning in data_quality["warnings"]
    )


def test_data_quality_separates_generic_and_v5_decision_audit(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                    "decision_audit_count_24h": 31,
                    "duplicate_rate": 0.0,
                    "fallback_rate": 0.0,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        missing_rows = list(
            csv.DictReader(io.StringIO(archive.read("anomalies/missing_data.csv").decode("utf-8")))
        )

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["generic_decision_audit_present"]["status"] == "PASS"
    assert (
        "legacy generic silver/decision_audit is optional for V5"
        in checks["generic_decision_audit_present"]["detail"]
    )
    assert checks["v5_decision_audit_present"]["status"] == "PASS"
    assert checks["v5_decision_audit_count"]["detail"] == "v5_decision_audit_count=31"
    assert data_quality["decision_audit"]["generic_decision_audit_present"] is False
    assert data_quality["decision_audit"]["v5_decision_audit_count"] == 31
    assert data_quality["decision_audit"]["v5_decision_audit_present"] is True
    assert any(
        row["dataset"] == "decision_audit"
        and row["reason"] == "legacy_optional_non_v5_research_missing"
        for row in missing_rows
    )
    assert data_quality["quant_lab_enforce_readiness"]["metrics"]["decision_audit_count"] == 31


def test_gate_evidence_quality_ignores_bootstrap_placeholder_gate():
    gates = pl.DataFrame(
        [
            {
                "alpha_id": "v5.core",
                "version": "bootstrap",
                "status": "QUARANTINE",
                "source": "bootstrap conservative placeholder",
            },
            {
                "alpha_id": "v5.core.momentum",
                "version": "v0.1",
                "status": "DEAD",
                "source": "gate_engine",
            },
        ]
    )
    evidence = pl.DataFrame(
        [
            {
                "alpha_id": "v5.core.momentum",
                "version": "v0.1",
                "evidence_status": "ok",
            }
        ]
    )

    assert daily_export_module._gates_have_evidence(gates, evidence) is True


def test_stale_rows_treat_v5_trade_event_as_event_driven_when_telemetry_current():
    now = datetime.now(UTC)
    frames = {
        "strategy_health_daily": pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                }
            ]
        ),
        "v5_trade_event": pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "symbol": "SOL-USDT",
                    "ts_utc": now - timedelta(days=3),
                    "raw_payload_json": "{}",
                }
            ]
        ),
    }

    stale = daily_export_module._stale_rows(frames)

    assert not any(row["dataset"] == "v5_trade_event" for row in stale.to_dicts())


def test_export_warns_public_proxy_cost_without_private_actuals(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-12",
                    "symbol": "BTC-USDT",
                    "regime": "normal",
                    "notional_bucket": "all",
                    "sample_count": 0,
                    "source": "public_spread_proxy",
                    "fallback_level": "PUBLIC_SPREAD_PROXY",
                    "created_at": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "gold" / "cost_bucket_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert (
        "okx_private_actual_cost_available: actual cost unavailable; "
        "cost model is public_spread_proxy only"
    ) in data_quality["warnings"]


def test_export_reports_private_fills_when_actual_cost_is_zero(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "endpoint": "/api/v5/trade/fills-history",
                    "ingest_ts": datetime.now(UTC),
                    "raw_json": "{}",
                }
            ]
        ),
        lake_root / "bronze" / "okx_private_readonly" / "fills_history",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-12",
                    "status": "CRITICAL",
                    "cost_model_version": "costs-v1",
                    "actual_rows": 0,
                    "proxy_rows": 1,
                    "global_default_rows": 0,
                    "fallback_ratio": 1.0,
                    "symbols_with_actual_cost": "[]",
                    "symbols_with_mixed_cost": "[]",
                    "symbols_with_proxy_only": '["BNB-USDT"]',
                    "symbols_proxy_only": '["BNB-USDT"]',
                    "symbols_missing_cost": "[]",
                    "actual_sample_count_by_symbol": "{}",
                    "data_quality_checks_json": (
                        '{"private_fills_present_but_actual_cost_zero":false}'
                    ),
                    "min_sample_count": 30,
                    "warnings_json": '["private_fills_present_but_actual_cost_zero"]',
                    "created_at": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "gold" / "cost_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert any(
        warning.startswith("private_fills_present_but_actual_cost_zero")
        for warning in data_quality["warnings"]
    )
    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["private_fills_present_but_actual_cost_zero"]["status"] == "WARN"
    assert "latest_health_check_passed=false" in checks[
        "private_fills_present_but_actual_cost_zero"
    ]["detail"]
    assert checks["actual_cost_symbol_coverage"]["status"] == "PASS"
    assert "latest_actual_or_mixed_symbols=0/" in checks["actual_cost_symbol_coverage"]["detail"]
    assert "historical_actual_or_mixed_symbols=" in checks["actual_cost_symbol_coverage"]["detail"]
    assert "relevant_private_fills=0" in checks["actual_cost_symbol_coverage"]["detail"]
    assert "private_fills_total=1" in checks["actual_cost_symbol_coverage"]["detail"]
    assert (
        "source=computed_from_latest_cost_health+cost_bucket_daily_snapshot"
        in checks["actual_cost_symbol_coverage"]["detail"]
    )


def test_export_flags_raw_private_fills_even_when_latest_health_check_is_ok(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "endpoint": "/api/v5/trade/fills-history",
                    "ingest_ts": datetime.now(UTC),
                    "raw_json": json.dumps(
                        {
                            "instType": "SPOT",
                            "instId": "BNB-USDT",
                            "tradeId": "fill-1",
                            "ordId": "order-1",
                            "side": "buy",
                            "fillPx": "300",
                            "fillSz": "1",
                            "fee": "-0.3",
                            "feeCcy": "USDT",
                            "execType": "T",
                            "ts": "1778544000000",
                        },
                        sort_keys=True,
                    ),
                }
            ]
        ),
        lake_root / "bronze" / "okx_private_readonly" / "fills_history",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-12",
                    "status": "WARNING",
                    "cost_model_version": "costs-v1",
                    "actual_rows": 0,
                    "mixed_rows": 0,
                    "proxy_rows": 1,
                    "global_default_rows": 0,
                    "fallback_ratio": 1.0,
                    "symbols_with_actual_cost": "[]",
                    "symbols_with_mixed_cost": "[]",
                    "symbols_with_proxy_only": '["BNB-USDT"]',
                    "symbols_proxy_only": '["BNB-USDT"]',
                    "symbols_missing_cost": "[]",
                    "actual_sample_count_by_symbol": "{}",
                    "data_quality_checks_json": (
                        '{"private_fills_present_but_actual_cost_zero":true}'
                    ),
                    "min_sample_count": 30,
                    "warnings_json": "[]",
                    "created_at": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "gold" / "cost_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    private_fill_check = checks["private_fills_present_but_actual_cost_zero"]
    assert private_fill_check["status"] == "WARN"
    assert "private_fills=1" in private_fill_check["detail"]
    assert "relevant_private_fills=1" in private_fill_check["detail"]
    assert "latest_health_check_passed=true" in private_fill_check["detail"]
    assert "export_raw_rows_without_latest_actual_or_mixed=true" in private_fill_check["detail"]
    assert any(
        warning.startswith("private_fills_present_but_actual_cost_zero")
        for warning in data_quality["warnings"]
    )


def test_export_does_not_flag_historical_raw_private_fills_for_latest_day(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "endpoint": "/api/v5/trade/fills-history",
                    "ingest_ts": datetime(2026, 6, 16, tzinfo=UTC),
                    "raw_json": json.dumps(
                        {
                            "instType": "SPOT",
                            "instId": "BNB-USDT",
                            "tradeId": "old-fill",
                            "ordId": "old-order",
                            "side": "buy",
                            "fillPx": "300",
                            "fillSz": "1",
                            "fee": "-0.3",
                            "feeCcy": "USDT",
                            "execType": "T",
                            "ts": "1778371200000",
                        },
                        sort_keys=True,
                    ),
                }
            ]
        ),
        lake_root / "bronze" / "okx_private_readonly" / "fills_history",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-06-16",
                    "status": "WARNING",
                    "cost_model_version": "costs-v1",
                    "actual_rows": 0,
                    "mixed_rows": 0,
                    "proxy_rows": 1,
                    "global_default_rows": 0,
                    "fallback_ratio": 1.0,
                    "symbols_with_actual_cost": "[]",
                    "symbols_with_mixed_cost": "[]",
                    "symbols_with_proxy_only": '["BNB-USDT"]',
                    "symbols_proxy_only": '["BNB-USDT"]',
                    "symbols_missing_cost": "[]",
                    "actual_sample_count_by_symbol": "{}",
                    "data_quality_checks_json": (
                        '{"private_fills_present_but_actual_cost_zero":true}'
                    ),
                    "min_sample_count": 30,
                    "warnings_json": "[]",
                    "created_at": datetime.now(UTC),
                }
            ]
        ),
        lake_root / "gold" / "cost_health_daily",
    )

    result = export_daily_pack(
        export_date="2026-06-16",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    private_fill_check = checks["private_fills_present_but_actual_cost_zero"]
    assert private_fill_check["status"] == "PASS"
    assert "private_fills=1" in private_fill_check["detail"]
    assert "relevant_private_fills=0" in private_fill_check["detail"]
    assert "export_raw_rows_without_latest_actual_or_mixed=false" in private_fill_check["detail"]


def test_actual_cost_symbol_coverage_splits_latest_and_historical_context():
    passed, detail = daily_export_module._actual_cost_symbol_coverage_check(
        export_day=date(2026, 5, 12),
        costs=pl.DataFrame(
            [
                {"symbol": "BTC-USDT", "source": "actual_fills"},
                {"symbol": "ETH-USDT", "source": "public_spread_proxy"},
            ]
        ),
        latest_cost_health={
            "symbols_with_actual_cost": "[]",
            "symbols_with_mixed_cost": "[]",
            "symbols_with_proxy_only": '["BTC-USDT","ETH-USDT"]',
            "symbols_missing_cost": "[]",
        },
        health_checks={"actual_cost_symbol_coverage": "n/a"},
        actual_or_mixed_rows=0,
        fills=pl.DataFrame([{"symbol": "BTC-USDT", "ts": "1778544000000"}]),
        v5_trades=pl.DataFrame(),
    )

    assert passed is False
    assert "latest_actual_or_mixed_symbols=0/2" in detail
    assert "latest_actual_or_mixed_rows=0" in detail
    assert "historical_actual_or_mixed_symbols=1/2" in detail
    assert "historical_actual_or_mixed_rows=1" in detail
    assert "relevant_private_fills=1" in detail
    assert "private_fills_total=1" in detail


def test_live_universe_stale_actual_or_mixed_detail_lists_uncovered_symbols():
    now = datetime.now(UTC)
    passed, detail = daily_export_module._live_universe_stale_actual_or_mixed_check(
        pl.DataFrame(
            [
                {
                    "day": now.date().isoformat(),
                    "symbol": "BTC-USDT",
                    "source": "actual_fills",
                    "sample_count": 4,
                    "created_at": now - timedelta(days=3),
                },
                {
                    "day": now.date().isoformat(),
                    "symbol": "BTC-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 5000,
                    "created_at": now,
                },
                {
                    "day": now.date().isoformat(),
                    "symbol": "ETH-USDT",
                    "source": "public_spread_proxy",
                    "sample_count": 5000,
                    "created_at": now,
                },
            ]
        )
    )

    assert passed is False
    assert "stale_actual_or_mixed_symbols=['BTC-USDT']" in detail
    assert (
        "uncovered_actual_or_mixed_symbols=['BNB-USDT', 'BTC-USDT', "
        "'ETH-USDT', 'SOL-USDT']"
    ) in detail
    assert "proxy_only_symbols=['BTC-USDT', 'ETH-USDT']" in detail
    assert "coverage_status=WARNING" in detail


def test_export_market_tables_keep_symbol_universe_visible(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {"symbol": "SOL-USDT", "ts": now, "size": 1.0},
                {"symbol": "BTC-USDT", "ts": now, "size": 2.0},
            ]
        ),
        lake_root / "silver" / "trade_print",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "SOL-USDT",
                    "channel": "books5",
                    "ts": now,
                    "asks_json": '[[101, "1"]]',
                    "bids_json": '[[100, "1"]]',
                },
                {
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "ts": now,
                    "asks_json": '[[201, "1"]]',
                    "bids_json": '[[200, "1"]]',
                },
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        trade_rows = list(
            csv.DictReader(io.StringIO(archive.read("market/trade_activity.csv").decode("utf-8")))
        )
        spread_rows = list(
            csv.DictReader(io.StringIO(archive.read("market/orderbook_spread.csv").decode("utf-8")))
        )
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert {row["symbol"] for row in trade_rows} >= {"BTC-USDT", "SOL-USDT"}
    assert {row["symbol"] for row in spread_rows} >= {"BTC-USDT", "SOL-USDT"}
    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["okx_ws_universe_complete"]["status"] == "WARN"
    assert "missing=['BNB-USDT', 'ETH-USDT']" in checks["okx_ws_universe_complete"]["detail"]


def test_okx_ws_universe_accepts_rollup_sources(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    now = datetime.now(UTC)
    symbols = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": symbol,
                    "latest_trade_ts": now,
                    "trade_count": 10,
                    "size_sum": 1.0,
                }
                for symbol in symbols
            ]
        ),
        lake_root / "silver" / "trade_activity_1m",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": symbol,
                    "ts": now,
                    "spread_bps": 1.0,
                }
                for symbol in symbols
            ]
        ),
        lake_root / "silver" / "orderbook_spread_1m",
    )

    result = export_daily_pack(
        export_date="2026-05-12",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
        pre_export_v5_refresh=False,
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["okx_ws_universe_complete"]["status"] == "PASS"
    assert "missing=[]" in checks["okx_ws_universe_complete"]["detail"]
    assert "trade_activity_1m:4" in checks["okx_ws_universe_complete"]["detail"]


def test_validate_expert_pack_rejects_possible_secret(tmp_path):
    pack_path = tmp_path / "quant_lab_expert_pack_2026-05-11.zip"
    with zipfile.ZipFile(pack_path, "w") as archive:
        for member in REQUIRED_MEMBERS:
            archive.writestr(member, _member_payload(member))
        archive.writestr("extra.txt", "OK-ACCESS-KEY: should-not-ship\n")

    result = validate_expert_pack(pack_path)

    assert result.rejected
    assert any("possible secrets" in reason for reason in result.reasons)


def test_validate_expert_pack_warns_on_critical_data_quality(tmp_path):
    pack_path = tmp_path / "quant_lab_expert_pack_2026-05-11.zip"
    with zipfile.ZipFile(pack_path, "w") as archive:
        for member in REQUIRED_MEMBERS:
            if member == "data_quality.json":
                archive.writestr(
                    member,
                    json.dumps(
                        {
                            "status": "CRITICAL",
                            "warnings": [],
                            "failures": ["market_bar_present: rows=0"],
                            "checks": [],
                        }
                    ),
                )
            else:
                archive.writestr(member, _member_payload(member))

    result = validate_expert_pack(pack_path)

    assert "data_quality status is CRITICAL" in result.warnings


def _fixture_lake(tmp_path) -> Path:
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 11, tzinfo=UTC)
    write_market_bars(
        lake_root,
        [
            _bar(start, close=100.0),
            _bar(start + timedelta(hours=1), close=101.0),
        ],
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-11",
                    "symbol": "BTC-USDT",
                    "regime": "normal",
                    "event_type": "fill",
                    "notional_bucket": "1k-10k",
                    "sample_count": 24,
                    "fee_bps_p50": 2.0,
                    "fee_bps_p75": 3.0,
                    "fee_bps_p90": 4.0,
                    "slippage_bps_p50": 1.0,
                    "slippage_bps_p75": 1.5,
                    "slippage_bps_p90": 2.0,
                    "spread_bps_p50": 3.0,
                    "spread_bps_p75": 4.0,
                    "spread_bps_p90": 5.0,
                    "total_cost_bps_p50": 6.0,
                    "total_cost_bps_p75": 8.5,
                    "total_cost_bps_p90": 11.0,
                    "fallback_level": "actual_okx_fills_and_bills",
                    "source": "okx_direct",
                }
            ]
        ),
        lake_root / "gold" / "cost_bucket_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "alpha_id": "alpha-1",
                    "version": "v1",
                    "coverage": 0.99,
                    "ic_mean": 0.04,
                    "ic_tstat": 3.1,
                    "oos_sharpe": 1.2,
                    "oos_max_drawdown": 0.1,
                    "edge_cost_ratio": 2.2,
                    "paper_days": 20,
                }
            ]
        ),
        lake_root / "gold" / "alpha_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "alpha_id": "alpha-1",
                    "version": "v1",
                    "gate_version": "default-v0.1",
                    "status": "LIVE_READY",
                    "passed": True,
                    "reasons": ["all_default_gates_passed"],
                    "metrics": {"ic_tstat": 3.1},
                    "next_action": "eligible_for_strategy_consumer_review",
                    "created_at": start,
                }
            ]
        ),
        lake_root / "gold" / "gate_decision",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "v1",
                    "permission": "ALLOW",
                    "allowed_modes": ["paper"],
                    "allowed_live_modes": [],
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-v1",
                    "gate_version": "default-v0.1",
                    "reasons": ["fixture"],
                    "created_at": start,
                }
            ]
        ),
        lake_root / "gold" / "risk_permission",
    )
    return lake_root


def _write_risk_permission_for_export(
    lake_root: Path,
    *,
    as_of: datetime,
    expires_at: datetime,
) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "v1",
                    "permission": "ALLOW",
                    "allowed_modes": ["paper"],
                    "allowed_live_modes": [],
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-v1",
                    "gate_version": "default-v0.1",
                    "reasons": ["fixture"],
                    "created_at": as_of,
                    "as_of_ts": as_of,
                    "expires_at": expires_at,
                    "permission_status": "ACTIVE_ALLOW",
                    "enforceable": True,
                }
            ]
        ),
        lake_root / "gold" / "risk_permission",
    )


def _write_current_and_bootstrap_risk_permissions_for_export(
    lake_root: Path,
    *,
    as_of: datetime,
    expires_at: datetime,
) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "version": "5.0.0",
                    "permission": "ALLOW",
                    "allowed_modes": ["paper"],
                    "allowed_live_modes": [],
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-v1",
                    "gate_version": "default-v0.1",
                    "reasons": ["fixture"],
                    "created_at": as_of,
                    "as_of_ts": as_of,
                    "expires_at": expires_at,
                    "permission_status": "ACTIVE_ALLOW",
                    "enforceable": True,
                },
                {
                    "strategy": "v5",
                    "version": "bootstrap",
                    "permission": "SELL_ONLY",
                    "allowed_modes": ["sell_only"],
                    "allowed_live_modes": [],
                    "max_gross_exposure": 0.0,
                    "max_single_weight": 0.0,
                    "cost_model_version": "costs-bootstrap",
                    "gate_version": "bootstrap.quarantine.v1",
                    "reasons": ["required_alpha_gate_quarantine"],
                    "created_at": datetime(2026, 5, 11, tzinfo=UTC),
                    "as_of_ts": None,
                    "expires_at": None,
                    "permission_status": None,
                    "enforceable": None,
                    "source": "bootstrap_gold_health",
                    "fallback_level": "BOOTSTRAP_CONSERVATIVE",
                },
            ],
            infer_schema_length=None,
        ),
        lake_root / "gold" / "risk_permission",
    )


def _write_v5_bundle_manifest_for_export(lake_root: Path, bundle_ts: datetime) -> None:
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "bundle_sha256": "sha-fixture",
                    "bundle_name": "v5_live_followup_bundle_fixture.tar.gz",
                    "bundle_ts": bundle_ts,
                    "ingest_ts": datetime.now(UTC),
                    "schema_version": "test",
                }
            ]
        ),
        lake_root / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest",
    )


def _v5_telemetry_config(tmp_path: Path, inbox: Path, lake_root: Path) -> Path:
    identity = tmp_path / "v5readonly_ed25519"
    identity.write_text("fixture-key", encoding="utf-8")
    config = tmp_path / "v5_telemetry_remote.yaml"
    config.write_text(
        "\n".join(
            [
                'strategy: "v5"',
                'remote_host: "example.invalid"',
                'remote_user: "v5readonly"',
                "remote_port: 22",
                'remote_bundle_dir: "/var/lib/v5/exports/bundles"',
                'filename_glob: "v5_live_followup_bundle_*.tar.gz"',
                f'ssh_identity_file: "{identity.as_posix()}"',
                f'local_inbox_dir: "{inbox.as_posix()}"',
                f'restricted_archive_dir: "{(tmp_path / "restricted").as_posix()}"',
                f'redacted_archive_dir: "{(tmp_path / "redacted").as_posix()}"',
                f'lake_root: "{lake_root.as_posix()}"',
                "max_bundle_size_mb: 512",
                "max_extracted_size_mb: 2048",
                "max_file_count: 5000",
                "min_stable_age_seconds: 0",
                "keep_remote_files: true",
                "dry_run: false",
            ]
        ),
        encoding="utf-8",
    )
    return config


def _bar(ts: datetime, close: float, symbol: str = "BTC-USDT") -> dict:
    return {
        "venue": "okx",
        "symbol": symbol,
        "market_type": "SPOT",
        "timeframe": "1H",
        "ts": ts,
        "open": close - 1.0,
        "high": close + 2.0,
        "low": close - 2.0,
        "close": close,
        "volume": 10.0,
        "quote_volume": 1000.0,
        "source": "test_fixture",
        "ingest_ts": ts + timedelta(minutes=1),
    }


def _member_payload(member: str) -> str:
    if member == "manifest.json":
        return json.dumps({"export_date": "2026-05-11", "files": []})
    if member == "data_quality.json":
        return json.dumps({"status": "PASS", "warnings": [], "checks": []})
    if member.endswith(".json"):
        return "{}"
    if member.endswith(".csv"):
        return "placeholder\n"
    return "placeholder\n"
