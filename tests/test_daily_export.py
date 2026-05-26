import csv
import io
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

import quant_lab.export.daily as daily_export_module
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import write_market_bars, write_parquet_dataset
from quant_lab.export.daily import (
    CSV_SCHEMAS,
    REQUIRED_MEMBERS,
    _load_export_frame,
    _member_row_count,
    _recent_heavy_dataset_files,
    _sha256_payload,
    _strategy_evidence_has_required_candidates,
    export_daily_pack,
    validate_expert_pack,
)
from tests.v5_bundle_fixture import make_tar


def test_export_daily_pack_writes_required_members(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=lake_root,
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
    )

    pack_path = Path(result.zip_path)
    validation = validate_expert_pack(pack_path)

    assert pack_path.exists()
    assert validation.valid
    assert validation.warnings == []
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
        assert "reports/v5_enforce_readiness.json" in names
        assert "reports/v5_enforce_readiness.csv" in names
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        executive_summary = archive.read("executive_summary.md").decode("utf-8")
        assert "quant_lab_enforce_readiness" in data_quality
        assert "shadow_only_recommended" in data_quality
        assert any(check["name"] == "code_provenance_clean" for check in data_quality["checks"])
        assert "quant_lab_enforce_readiness:" in executive_summary
        assert "charts/market_close.png" in names
        assert archive.read("charts/market_close.png").startswith(b"\x89PNG")


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
                }
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
        row["strategy_candidate"] == "v5.risk_on_multi_buy_top1_shadow"
        for row in risk_on_rows
    )
    assert any(
        row["strategy_candidate"] == "v5.risk_on_multi_buy_top2_shadow"
        for row in risk_on_rows
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
    assert _sha256_payload(text) == daily_export_module.hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


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
            csv.DictReader(
                io.StringIO(archive.read("research/alpha_evidence.csv").decode("utf-8"))
            )
        )
        gates = list(
            csv.DictReader(
                io.StringIO(archive.read("research/gate_decisions.csv").decode("utf-8"))
            )
        )
        dashboard = list(
            csv.DictReader(
                io.StringIO(
                    archive.read("reports/strategy_level_dashboard.csv").decode("utf-8")
                )
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
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert manifest["pre_export_v5_refresh"]["processed_bundle_count"] == 1
    assert manifest["authoritative_snapshot"] is True
    assert manifest["stale_v5_bundle"] is False
    assert manifest["selected_v5_bundle_sha256"]
    assert manifest["selected_v5_bundle_built_at"].startswith("2026-05-17T06:00:00")
    assert manifest["selected_v5_bundle_ingested_at"]
    assert manifest["selected_v5_bundle_event_counts"]["v5_candidate_event"] >= 1
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
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json").decode("utf-8"))

    refresh = manifest["pre_export_v5_refresh"]
    assert refresh["selected_bundle_count"] == 1
    assert refresh["processed_bundle_count"] == 1
    assert refresh["selected_bundle_names"] == [
        "v5_live_followup_bundle_20260517T060000Z.tar.gz"
    ]
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
                "candidate_id,run_id,ts_utc,symbol\n"
                "old,run_old,2026-05-17T09:00:00Z,BTC-USDT\n"
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
            "v5/v5_candidate_events.csv",
            "v5/v5_candidate_labels.csv",
            "v5/v5_candidate_quality.csv",
            "v5/v5_candidate_outcome_summary.csv",
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


def test_cost_fallback_ratio_is_not_pass_when_cost_bucket_daily_is_empty(tmp_path):
    result = export_daily_pack(
        export_date="2026-05-11",
        lake_root=tmp_path / "empty-lake",
        out_dir=tmp_path / "exports",
        profile="expert",
        command_line=["qlab", "export-daily"],
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
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    checks = {check["name"]: check for check in data_quality["checks"]}
    assert checks["cost_hard_fallback_ratio"]["status"] == "PASS"
    assert "ratio=0.00%" in checks["cost_hard_fallback_ratio"]["detail"]
    assert checks["cost_soft_fallback_ratio"]["status"] == "WARN"
    assert checks["cost_fallback_ratio"]["status"] == "WARN"


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
            "v5_pullback_reversal_shadow": pl.DataFrame(),
        }
    )

    datasets = set(stale["dataset"].to_list()) if "dataset" in stale.columns else set()
    assert "risk_on_multi_buy_shadow" not in datasets
    assert "v5_entry_quality_history_anti_leakage_check" not in datasets
    assert "v5_pullback_reversal_shadow" not in datasets


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


def test_load_snapshot_does_not_warn_for_optional_empty_entry_quality_outputs(tmp_path):
    snapshot = daily_export_module._load_snapshot(tmp_path / "empty-lake")

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

    assert "成本模型仍是 public spread proxy" in questions
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
            csv.DictReader(
                io.StringIO(archive.read("research/alpha_evidence.csv").decode("utf-8"))
            )
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
            "risk_permission_fresh_vs_v5_telemetry: "
            "risk_permission_stale_vs_v5_telemetry"
        )
        for warning in data_quality["warnings"]
    )
    assert "Risk permission status: STALE_ALLOW" in executive_summary
    assert "Latest V5 telemetry ts: 2026-05-12T21:00:00+00:00" in executive_summary
    assert risk_flags[0]["permission_status"] == "STALE_ALLOW"
    assert risk_flags[0]["enforceable"].lower() == "false"


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
        "risk_permission_republish_failed"
        in checks["risk_permission_pre_export_refresh"]["detail"]
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
        "risk_permission_version_matches_v5" in warning
        for warning in data_quality["warnings"]
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
    assert "legacy generic silver/decision_audit is optional for V5" in checks[
        "generic_decision_audit_present"
    ]["detail"]
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
    assert (
        data_quality["quant_lab_enforce_readiness"]["metrics"]["decision_audit_count"]
        == 31
    )


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
                    "symbols_with_proxy_only": "[\"BNB-USDT\"]",
                    "symbols_proxy_only": "[\"BNB-USDT\"]",
                    "symbols_missing_cost": "[]",
                    "actual_sample_count_by_symbol": "{}",
                    "data_quality_checks_json": (
                        "{\"private_fills_present_but_actual_cost_zero\":false}"
                    ),
                    "min_sample_count": 30,
                    "warnings_json": "[\"private_fills_present_but_actual_cost_zero\"]",
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
    )

    with zipfile.ZipFile(result.zip_path) as archive:
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))

    assert any(
        warning.startswith("private_fills_present_but_actual_cost_zero")
        for warning in data_quality["warnings"]
    )


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
    assert "okx_ws_universe_complete: okx_ws_universe_incomplete" in data_quality["warnings"]


def test_validate_expert_pack_rejects_possible_secret(tmp_path):
    pack_path = tmp_path / "quant_lab_expert_pack_2026-05-11.zip"
    with zipfile.ZipFile(pack_path, "w") as archive:
        for member in REQUIRED_MEMBERS:
            archive.writestr(member, _member_payload(member))
        archive.writestr("extra.txt", "OK-ACCESS-KEY: should-not-ship\n")

    result = validate_expert_pack(pack_path)

    assert result.rejected
    assert any("possible secrets" in reason for reason in result.reasons)


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
