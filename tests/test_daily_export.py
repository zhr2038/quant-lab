import csv
import io
import json
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import write_market_bars, write_parquet_dataset
from quant_lab.export.daily import (
    CSV_SCHEMAS,
    REQUIRED_MEMBERS,
    _recent_heavy_dataset_files,
    export_daily_pack,
    validate_expert_pack,
)


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
        assert manifest["row_counts"]["market_bar"] == 2
        assert manifest["git_commit"]
        assert "git_branch" in manifest
        assert "dirty_worktree" in manifest
        assert manifest["hostname"]
        assert manifest["python_version"]
        assert manifest["export_command"] == "qlab export-daily"
        assert manifest["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
        assert provenance["contract_version"] == V5_QUANT_LAB_CONTRACT_VERSION
        assert "dataset_freshness" in manifest
        assert provenance["git_commit"]
        assert "freshness_seconds" in provenance["datasets"][0]
        assert "freshness_status" in provenance["datasets"][0]
        assert "v5/v5_strategy_health.csv" in names
        assert "reports/v5_enforce_readiness.json" in names
        assert "reports/v5_enforce_readiness.csv" in names
        data_quality = json.loads(archive.read("data_quality.json").decode("utf-8"))
        executive_summary = archive.read("executive_summary.md").decode("utf-8")
        assert "quant_lab_enforce_readiness" in data_quality
        assert "shadow_only_recommended" in data_quality
        assert "quant_lab_enforce_readiness:" in executive_summary
        assert "charts/market_close.png" in names
        assert archive.read("charts/market_close.png").startswith(b"\x89PNG")


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
    assert checks["generic_decision_audit_present"]["status"] == "WARN"
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
                    "allowed_modes": ["paper", "live_canary"],
                    "max_gross_exposure": 0.25,
                    "max_single_weight": 0.05,
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


def _bar(ts: datetime, close: float) -> dict:
    return {
        "venue": "okx",
        "symbol": "BTC-USDT",
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
