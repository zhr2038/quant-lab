import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.data.lake import write_market_bars, write_parquet_dataset
from quant_lab.web import app as web_app
from quant_lab.web import readers
from quant_lab.web.pages import (
    alpha_gates,
    cost_model,
    data_health,
    expert_exports,
    market_regime,
    okx_collectors,
    overview,
    strategy_consumers,
    v5_telemetry,
)

FORBIDDEN_WEB_TERMS = [
    "place_order",
    "create_order",
    "cancel_order",
    "amend_order",
    "withdraw",
    "transfer_funds",
    "execute_trade",
    "live_order_mutation",
]


def test_web_source_has_no_trading_execution_surface():
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in Path("src/quant_lab/web").rglob("*.py")
    )

    for term in FORBIDDEN_WEB_TERMS:
        assert term not in source_text


def test_web_readers_do_not_write_lake_files():
    source_text = Path("src/quant_lab/web/readers.py").read_text(encoding="utf-8")
    forbidden_write_calls = [
        "write_parquet",
        "write_parquet_dataset",
        "upsert_parquet_dataset",
        "write_market_bars",
        ".unlink(",
        ".rmdir(",
        ".mkdir(",
    ]

    for call in forbidden_write_calls:
        assert call not in source_text


def test_secret_like_values_are_redacted():
    df = pl.DataFrame(
        {
            "api_key": ["visible-sensitive-value"],
            "passphrase": ["another-sensitive-value"],
            "normal": ["ok-access-key=should-redact"],
        }
    )

    redacted = readers.redact_frame(df)

    assert redacted["api_key"][0] == "[REDACTED]"
    assert redacted["passphrase"][0] == "[REDACTED]"
    assert redacted["normal"][0] == "[REDACTED]"


def test_dashboard_readers_load_fixture_lake(tmp_path):
    lake_root = _fixture_lake(tmp_path)

    overview_summary = readers.dashboard_overview(lake_root)
    data_health = readers.data_health_summary(lake_root)
    costs = readers.cost_model_summary(lake_root)
    gates = readers.alpha_gate_summary(lake_root)
    consumers = readers.strategy_consumer_summary(lake_root)
    v5 = readers.v5_telemetry_summary(lake_root)
    experts = readers.expert_export_summary(tmp_path / "exports")

    assert overview_summary["status"] in {"OK", "WARNING"}
    assert overview_summary["latest_market_bar_ts"] == datetime(2026, 5, 10, 2, tzinfo=UTC)
    assert overview_summary["okx_public_ws_status"] == "RUNNING"
    assert data_health["duplicate_bar_count"] == 0
    assert costs["costs"].height == 1
    assert gates["counts"] == {"LIVE_READY": 1}
    assert consumers["permissions"] == {"v5": "ALLOW", "v7": "SELL_ONLY"}
    assert v5["latest"]["status"] == "OK"
    assert experts["latest_pack"].endswith("quant_lab_expert_pack_2026-05-10.zip")


def test_key_pages_render_fixture_lake_without_network_or_mutation(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit()

    for page in [
        overview,
        data_health,
        okx_collectors,
        market_regime,
        cost_model,
        alpha_gates,
        strategy_consumers,
        v5_telemetry,
        expert_exports,
    ]:
        page.render(lake_root, fake)

    assert fake.calls
    assert any(call[0] == "title" for call in fake.calls)
    assert any(call[0] == "dataframe" for call in fake.calls)


def test_web_launcher_hides_streamlit_file_navigation(monkeypatch, tmp_path):
    captured = {}

    class Result:
        returncode = 0

    def fake_run(command, env, check):
        captured["command"] = command
        captured["env"] = env
        captured["check"] = check
        return Result()

    monkeypatch.setattr(web_app.subprocess, "run", fake_run)

    result = web_app.main(
        [
            "--host",
            "0.0.0.0",
            "--port",
            "8501",
            "--lake-root",
            str(tmp_path / "lake"),
        ]
    )

    command = captured["command"]
    option_index = command.index("--client.showSidebarNavigation")
    assert result == 0
    assert captured["check"] is False
    assert command[option_index + 1] == "false"
    assert option_index < command.index("--")


def test_pages_warn_but_do_not_crash_when_lake_is_empty(tmp_path):
    fake = FakeStreamlit()

    overview.render(tmp_path / "empty_lake", fake)
    data_health.render(tmp_path / "empty_lake", fake)
    cost_model.render(tmp_path / "empty_lake", fake)
    alpha_gates.render(tmp_path / "empty_lake", fake)

    assert any(call[0] == "warning" for call in fake.calls)
    assert any(call[0] == "info" for call in fake.calls)


def test_okx_collectors_warns_when_orderbook_parquet_is_invalid(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    bad_path = lake_root / "silver" / "orderbook_snapshot" / "bad.parquet"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_bytes(b"not parquet")
    fake = FakeStreamlit()

    okx_collectors.render(lake_root, fake)

    warnings = "\n".join(str(warning) for warning in _call_values(fake, "warning"))
    assert "orderbook_snapshot 数据集读取失败" in warnings
    assert str(bad_path.parent) in warnings


def test_overview_diagnostics_warn_when_lake_root_does_not_exist(tmp_path):
    missing_lake = tmp_path / "missing_lake"
    fake = FakeStreamlit()

    overview.render(missing_lake, fake)

    warnings = _call_values(fake, "warning")
    assert any("Lake 根目录不存在" in str(warning) for warning in warnings)
    assert ("Lake 根目录", str(missing_lake)) in _call_values(fake, "metric")


def test_overview_diagnostics_suggests_commands_when_lake_has_no_parquet(tmp_path):
    lake_root = tmp_path / "lake"
    lake_root.mkdir()
    fake = FakeStreamlit()

    overview.render(lake_root, fake)

    warnings = "\n".join(str(warning) for warning in _call_values(fake, "warning"))
    assert (
        "qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT "
        f"--lake-root {lake_root} --history --limit 100"
    ) in warnings
    assert "qlab okx-ws-collect-universe" in warnings
    assert "qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml" in warnings
    assert "qlab export-daily 尚未实现或尚未运行。" in warnings


def test_empty_lake_diagnostics_exposes_suggested_commands(tmp_path):
    lake_root = tmp_path / "lake"

    diagnostics = readers.lake_diagnostics(lake_root)

    commands = "\n".join(diagnostics["suggested_commands"]["command"].to_list())
    assert "qlab okx-fetch-candles" in commands
    assert "qlab okx-ws-collect-universe" in commands
    assert "qlab sync-v5-telemetry" in commands
    assert "qlab export-daily" in commands


def test_overview_diagnostics_shows_latest_market_bar_ts(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit()

    overview.render(lake_root, fake)

    metrics = _call_values(fake, "metric")
    assert ("最新 market_bar 时间", "2026-05-10 02:00:00+00:00") in metrics


def _call_values(fake: "FakeStreamlit", name: str) -> list[object]:
    return [value for call_name, value in fake.calls if call_name == name]


class FakeStreamlit:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.sidebar = self

    def set_page_config(self, **kwargs) -> None:
        self.calls.append(("set_page_config", kwargs))

    def title(self, value) -> None:
        self.calls.append(("title", value))

    def caption(self, value) -> None:
        self.calls.append(("caption", value))

    def metric(self, label, value) -> None:
        self.calls.append(("metric", (label, value)))

    def subheader(self, value) -> None:
        self.calls.append(("subheader", value))

    def dataframe(self, value) -> None:
        self.calls.append(("dataframe", value))

    def warning(self, value) -> None:
        self.calls.append(("warning", value))

    def info(self, value) -> None:
        self.calls.append(("info", value))

    def write(self, value) -> None:
        self.calls.append(("write", value))

    def selectbox(self, _label, options):
        return list(options)[0]

    def text_input(self, _label, value):
        return value


def _fixture_lake(tmp_path) -> Path:
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    write_market_bars(
        lake_root,
        [
            _bar(start + timedelta(hours=0), close=100.0),
            _bar(start + timedelta(hours=1), close=102.0),
            _bar(start + timedelta(hours=2), close=103.0),
        ],
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BTC-USDT",
                    "regime": "normal",
                    "event_type": "fill",
                    "notional_bucket": "1k-10k",
                    "sample_count": 42,
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
                _permission_row("v5", "ALLOW", ["paper", "live_canary"]),
                _permission_row("v7", "SELL_ONLY", ["sell_only"]),
            ]
        ),
        lake_root / "gold" / "risk_permission",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "channel": "tickers",
                    "inst_id": "BTC-USDT",
                    "received_at": start,
                    "raw_json": "{}",
                }
            ]
        ),
        lake_root / "bronze" / "okx_public_ws",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "collector_name": "okx_public_ws",
                    "started_at": start.isoformat().replace("+00:00", "Z"),
                    "last_message_at": start.isoformat().replace("+00:00", "Z"),
                    "messages_read": 1,
                    "reconnect_count": 0,
                    "error_count": 0,
                    "subscribed_symbols": '["BTC-USDT"]',
                    "subscribed_channels": '["tickers"]',
                    "status": "RUNNING",
                    "updated_at": start.isoformat().replace("+00:00", "Z"),
                }
            ]
        ),
        lake_root / "bronze" / "collector_health" / "okx_public_ws",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-10",
                    "status": "OK",
                    "latest_bundle_ts": start,
                    "latest_bundle_sha256": "fixture-sha",
                    "run_count_72h": 1,
                    "decision_audit_count_24h": 1,
                    "trade_count_24h": 0,
                    "trade_count_72h": 0,
                    "roundtrip_count_72h": 0,
                    "open_position_count": 0,
                    "dust_residual_position_count": 0,
                    "kill_switch_enabled": False,
                    "reconcile_ok": True,
                    "ledger_ok": True,
                    "auto_risk_level": "LOW",
                    "high_issue_count": 0,
                    "medium_issue_count": 0,
                    "config_not_consumed_count": 0,
                    "high_score_blocked_count": 0,
                    "high_score_blocked_matured_count": 0,
                    "warnings_json": "[]",
                    "critical_reasons_json": "[]",
                    "next_actions_json": "[]",
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-10",
                    "status": "OK",
                    "violation_count": 0,
                    "violations_json": "[]",
                }
            ]
        ),
        lake_root / "gold" / "v5_gate_compliance_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "symbol": "BTC-USDT",
                    "trade_id": "1",
                    "price": 103.0,
                    "size": 1.0,
                    "side": "buy",
                    "ts": start,
                    "source": "fixture",
                    "ingest_ts": start,
                    "raw_json": "{}",
                }
            ]
        ),
        lake_root / "silver" / "trade_print",
    )
    _expert_pack(tmp_path / "exports")
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


def _permission_row(strategy: str, permission: str, modes: list[str]) -> dict:
    return {
        "strategy": strategy,
        "version": "v1",
        "permission": permission,
        "allowed_modes": modes,
        "max_gross_exposure": 0.25,
        "max_single_weight": 0.05,
        "cost_model_version": "costs-v1",
        "gate_version": "default-v0.1",
        "reasons": ["fixture"],
        "created_at": datetime(2026, 5, 10, tzinfo=UTC),
    }


def _expert_pack(exports_root: Path) -> None:
    exports_root.mkdir(parents=True)
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-10.zip"
    with zipfile.ZipFile(pack_path, "w") as archive:
        archive.writestr("manifest.json", '{"export_date": "2026-05-10"}')
        archive.writestr("data_quality.json", '{"status": "ok"}')
        archive.writestr("expert_questions.md", "1. Review alpha-1 gate.\n")
