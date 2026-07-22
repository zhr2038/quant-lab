import json
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from quant_lab.cli import _refresh_web_file_index
from quant_lab.data.file_index import build_lake_file_index
from quant_lab.data.lake import write_market_bars, write_parquet_dataset
from quant_lab.web import app as web_app
from quant_lab.web import export_request, readers
from quant_lab.web import perf as web_perf
from quant_lab.web.pages import (
    alpha_gates,
    bigscreen_v2,
    cost_model,
    data_health,
    expert_exports,
    market_regime,
    okx_collectors,
    overview,
    strategy_consumers,
    v5_telemetry,
    web_performance,
)
from quant_lab.web.pages._common import localize_frame, show_frame

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


def test_show_frame_prefers_streamlit_width_stretch():
    class ModernStreamlit:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def dataframe(self, value, **kwargs) -> None:
            self.calls.append(kwargs)

    fake = ModernStreamlit()

    show_frame(fake, pl.DataFrame({"symbol": ["BTC-USDT"]}), "empty")

    assert fake.calls == [{"width": "stretch", "hide_index": True}]


def test_show_frame_falls_back_for_older_streamlit_dataframe_api():
    class LegacyStreamlit:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def dataframe(self, value, **kwargs) -> None:
            if "width" in kwargs:
                raise TypeError("width is not supported")
            self.calls.append(kwargs)

    fake = LegacyStreamlit()

    show_frame(fake, pl.DataFrame({"symbol": ["BTC-USDT"]}), "empty")

    assert fake.calls == [{"use_container_width": True, "hide_index": True}]


def test_bigscreen_v2_prefers_components_iframe_api():
    class Components:
        def __init__(self, calls: list[tuple[str, object]]) -> None:
            self.v1 = self
            self._calls = calls

        def iframe(self, url, **kwargs) -> None:
            self._calls.append(("components_iframe", {"url": url, **kwargs}))

    class StreamlitWithBothApis:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []
            self.components = Components(self.calls)

        def iframe(self, _url, **_kwargs) -> None:
            raise AssertionError("top-level iframe must not be used when components.v1 exists")

    fake = StreamlitWithBothApis()

    bigscreen_v2.show_iframe(fake, "http://example.test/web-v2", height=920, scrolling=False)

    assert fake.calls == [
        (
            "components_iframe",
            {"url": "http://example.test/web-v2", "height": 920, "scrolling": False},
        )
    ]


def test_bigscreen_v2_falls_back_to_top_level_iframe_without_scrolling():
    class TopLevelOnlyStreamlit:
        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def iframe(self, url, **kwargs) -> None:
            if "scrolling" in kwargs:
                raise TypeError("scrolling is not supported")
            self.calls.append(("iframe", {"url": url, **kwargs}))

    fake = TopLevelOnlyStreamlit()

    bigscreen_v2.show_iframe(fake, "http://example.test/web-v2", height=920, scrolling=False)

    assert fake.calls == [
        (
            "iframe",
            {"url": "http://example.test/web-v2", "height": 920},
        )
    ]


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


def test_dashboard_overview_omits_absent_v7_permission(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    write_parquet_dataset(
        pl.DataFrame([_permission_row("v5", "ALLOW", ["paper"])]),
        lake_root / "gold" / "risk_permission",
    )

    overview_summary = readers.dashboard_overview(lake_root)

    assert overview_summary["v5_permission"] == "ALLOW"
    assert "v7_permission" not in overview_summary


def test_web_diagnostics_use_dataset_freshness_not_missing_for_populated_tables(tmp_path):
    lake_root = _fixture_lake(tmp_path)

    diagnostics = readers.lake_diagnostics(lake_root)
    rows = {row["dataset"]: row for row in diagnostics["datasets"].to_dicts()}

    for dataset in [
        "gold/cost_bucket_daily",
        "gold/gate_decision",
        "gold/risk_permission",
        "gold/strategy_health_daily",
    ]:
        assert rows[dataset]["rows"] > 0
        assert rows[dataset]["freshness_status"] != "missing"


def test_lake_diagnostics_parquet_count_excludes_internal_compaction_dirs(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    baseline = readers.lake_diagnostics(lake_root)["parquet_file_count"]

    internal = lake_root / "silver" / "orderbook_snapshot" / "__direct_compact_leftover"
    internal.mkdir(parents=True)
    pl.DataFrame([{"symbol": "SOL-USDT", "value": 1}]).write_parquet(
        internal / "compact_temp.parquet"
    )
    tmp_dir = lake_root / "silver" / "orderbook_snapshot" / "._tmp"
    tmp_dir.mkdir()
    pl.DataFrame([{"symbol": "SOL-USDT", "value": 2}]).write_parquet(
        tmp_dir / "part.tmp.parquet"
    )

    diagnostics = readers.lake_diagnostics(lake_root)

    assert diagnostics["parquet_file_count"] == baseline


def test_lake_diagnostics_uses_metadata_for_core_datasets(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    original = readers.read_parquet_dataset
    core_paths = {
        (lake_root / relative_path).resolve()
        for relative_path in readers.CORE_DIAGNOSTIC_DATASETS.values()
    }

    def guarded_read_parquet_dataset(path):
        if Path(path).resolve() in core_paths:
            raise AssertionError("lake diagnostics should not full-read core datasets")
        return original(path)

    monkeypatch.setattr(readers, "read_parquet_dataset", guarded_read_parquet_dataset)

    diagnostics = readers.lake_diagnostics(lake_root)
    rows = {row["dataset"]: row for row in diagnostics["datasets"].to_dicts()}

    assert rows["silver/market_bar"]["rows"] == 3
    assert rows["gold/risk_permission"]["rows"] == 2
    assert diagnostics["latest_market_bar_ts"] == datetime(2026, 5, 10, 2, tzinfo=UTC)


def test_data_health_stale_datasets_use_metadata_instead_of_full_raw_reads(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name in {"okx_public_ws", "trade_print", "orderbook_snapshot"}:
            raise AssertionError(f"{dataset_name} should be summarized from parquet metadata")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.data_health_summary(lake_root)

    assert summary["latest_market_bar_ts"] == datetime(2026, 5, 10, 2, tzinfo=UTC)
    assert "stale_datasets" in summary


def test_stale_dataset_rows_reuses_cached_snapshot_table(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = _fixture_lake(tmp_path)
    original = readers._dataset_snapshot
    calls: list[str] = []

    def counted_snapshot(lake_root_arg, dataset_name, **kwargs):
        calls.append(str(dataset_name))
        return original(lake_root_arg, dataset_name, **kwargs)

    monkeypatch.setattr(readers, "_dataset_snapshot", counted_snapshot)

    first = readers._stale_dataset_rows(lake_root)
    first_call_count = len(calls)
    second = readers._stale_dataset_rows(lake_root)

    assert first_call_count > 0
    assert len(calls) == first_call_count
    assert first.to_dicts() == second.to_dicts()


def test_data_health_summary_uses_lazy_market_bar_aggregates(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name == "market_bar":
            raise AssertionError("data health should not fully read market_bar")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.data_health_summary(lake_root)

    assert summary["latest_market_bar_ts"] == datetime(2026, 5, 10, 2, tzinfo=UTC)
    assert summary["duplicate_bar_count"] == 0
    assert summary["unclosed_bar_count"] == 0
    assert summary["latest_per_symbol"].height == 1
    assert summary["missing_bar_ratio"] == 0.0


def test_data_health_summary_reuses_cached_market_bar_lazy_health(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = _fixture_lake(tmp_path)
    original = readers._scan_parquet_files
    scan_calls: list[tuple[str, ...]] = []

    def counted_scan(files):
        file_tuple = tuple(str(path) for path in files)
        scan_calls.append(file_tuple)
        return original(files)

    monkeypatch.setattr(readers, "_scan_parquet_files", counted_scan)

    first = readers.data_health_summary(lake_root)
    first_scan_count = len(
        [
            call
            for call in scan_calls
            if any("market_bar" in path for path in call)
        ]
    )
    second = readers.data_health_summary(lake_root)

    market_calls = [
        call
        for call in scan_calls
        if any("market_bar" in path for path in call)
    ]
    assert first["latest_market_bar_ts"] == second["latest_market_bar_ts"]
    assert first_scan_count > 0
    assert len(market_calls) == first_scan_count


def test_dashboard_overview_uses_lazy_market_bar_aggregates(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name == "market_bar":
            raise AssertionError("dashboard overview should not fully read market_bar")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.dashboard_overview(lake_root)

    assert summary["latest_market_bar_ts"] == datetime(2026, 5, 10, 2, tzinfo=UTC)
    assert summary["missing_bar_ratio"] == 0.0


def test_dashboard_overview_samples_advisory_tables(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    strategy_path = lake_root / "gold" / "strategy_opportunity_advisory"
    strategy_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "strategy_candidate": "old_candidate",
                "symbol": "OLD-USDT",
                "decision": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "as_of_ts": start,
            }
        ]
    ).write_parquet(strategy_path / "batch_20260510T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "strategy_candidate": "new_candidate",
                "symbol": "NEW-USDT",
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "as_of_ts": start + timedelta(hours=1),
            }
        ]
    ).write_parquet(strategy_path / "batch_20260510T010000000000Z.parquet")
    entry_path = lake_root / "gold" / "v5_entry_quality_advisory"
    entry_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "strategy_candidate": "old_entry",
                "symbol": "OLD-USDT",
                "recommended_mode": "shadow",
                "readiness_status": "RESEARCH_ONLY",
                "generated_at_utc": start,
            }
        ]
    ).write_parquet(entry_path / "batch_20260510T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "strategy_candidate": "new_entry",
                "symbol": "NEW-USDT",
                "recommended_mode": "shadow",
                "readiness_status": "KEEP_SHADOW",
                "generated_at_utc": start + timedelta(hours=1),
            }
        ]
    ).write_parquet(entry_path / "batch_20260510T010000000000Z.parquet")
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "strategy_opportunity_advisory", 1)
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "v5_entry_quality_advisory", 1)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name in {"strategy_opportunity_advisory", "v5_entry_quality_advisory"}:
            raise AssertionError(f"{dataset_name} should use recent sample reads")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.dashboard_overview(lake_root)

    assert summary["strategy_opportunity_advisory"]["strategy_candidate"].to_list() == [
        "new_candidate"
    ]
    assert summary["entry_quality_advisory"]["strategy_candidate"].to_list() == ["new_entry"]


def test_advisory_recent_sample_prefers_generated_at_over_business_as_of(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    dataset_path = lake_root / "gold" / "strategy_opportunity_advisory"
    dataset_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "strategy_candidate": "old_business_future",
                "symbol": "OLD-USDT",
                "decision": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "as_of_ts": (now + timedelta(hours=2)).isoformat(),
                "generated_at": (now - timedelta(days=2)).isoformat(),
            }
        ]
    ).write_parquet(dataset_path / "batch_old.parquet")
    pl.DataFrame(
        [
            {
                "strategy_candidate": "new_generated",
                "symbol": "NEW-USDT",
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "as_of_ts": (now - timedelta(hours=1)).isoformat(),
                "generated_at": now.isoformat(),
            }
        ]
    ).write_parquet(dataset_path / "batch_new.parquet")

    frame, warning = readers.read_recent_dataset_with_warning(
        lake_root,
        "strategy_opportunity_advisory",
        limit=1,
        lookback_hours=1,
    )

    assert warning is None
    assert frame["strategy_candidate"].to_list() == ["new_generated"]


def test_research_portfolio_web_table_normalizes_close_actions():
    frame = pl.DataFrame(
        [
            {
                "research_id": "btc_broad_leadership",
                "module": "research_portfolio",
                "strategy_candidate": "v5.btc_broad_leadership",
                "status": "KILL",
                "action": "close_research",
                "reason": "negative_after_cost_edge",
                "sample_count": 40,
                "complete_sample_count": 40,
                "avg_net_bps": -12.5,
                "win_rate": 0.3,
                "p25_net_bps": -80.0,
            },
            {
                "research_id": "eth_f3",
                "module": "paper",
                "strategy_candidate": "ETH_F3_DOMINANT_ENTRY_PAPER_V1",
                "status": "DOWNGRADED_FROM_PAPER",
                "action": "continue_paper",
                "reason": "paper_negative_streak",
                "sample_count": 20,
                "complete_sample_count": 20,
                "avg_net_bps": -4.0,
                "win_rate": 0.45,
                "p25_net_bps": -55.0,
            },
        ]
    )

    table = readers._research_portfolio_table(frame)
    localized = localize_frame(table)

    assert "研究动作" in localized.columns
    assert "重点结论" in localized.columns
    assert "关键指标" in localized.columns
    actions = localized.get_column("研究动作").to_list()
    assert "关闭研究" in actions
    assert "转影子观察/复查" in actions
    assert "收盘价" not in actions
    takeaways = localized.get_column("重点结论").to_list()
    assert any("关闭" in value for value in takeaways)
    assert any("降级" in value for value in takeaways)
    assert any("ETH F3" in value for value in takeaways)
    assert all("complete_sample_count" not in value for value in localized.get_column("关键指标"))


def test_strategy_opportunity_web_table_keeps_high_signal_extension_fields():
    frame = pl.DataFrame(
        [
            {
                "strategy_candidate": "v5.af.expanded_relative_strength",
                "symbol": "TRX-USDT",
                "decision": "KEEP_SHADOW",
                "recommended_mode": "shadow",
                "horizon_hours": 48,
                "sample_count": 18,
                "complete_sample_count": 18,
                "avg_net_bps": 14.0,
                "p25_net_bps": -10.0,
                "win_rate": 0.61,
                "source_module": "alpha_factory",
                "promotion_state": "KEEP_SHADOW",
                "universe_type": "expanded_paper",
                "alpha_factory_score": 0.72,
                "cost_quality_score": 0.4,
                "paper_ready_block_reasons": '["cost_quality_not_paper_ready"]',
                "max_live_notional_usdt": 0.0,
            }
        ]
    )

    table = readers._strategy_opportunity_table(frame)
    localized = localize_frame(table)

    assert "Alpha Factory 分数" in localized.columns
    assert "币池类型" in localized.columns
    assert "重点结论" in localized.columns
    assert "关键指标" in localized.columns
    assert localized["推荐模式"][0] == "影子观察"
    assert localized["币池类型"][0] == "扩展币池纸面"
    assert "完整样本 18" in localized["关键指标"][0]
    assert "Alpha Factory" in localized["重点结论"][0]
    assert "v5.af." not in localized["重点结论"][0]


def test_strategy_opportunity_web_takeaway_localizes_regime_router_candidates():
    frame = pl.DataFrame(
        [
            {
                "strategy_candidate": "regime_router:ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1",
                "symbol": "ALL",
                "decision": "PAPER_READY",
                "recommended_mode": "paper",
                "max_live_notional_usdt": 0.0,
                "as_of_ts": "2026-05-25T00:00:00+00:00",
            }
        ]
    )

    table = readers._strategy_opportunity_table(frame)
    localized = localize_frame(table)

    takeaway = localized["重点结论"][0]
    assert "行情路由：ETH F3 主导纸面策略" in takeaway
    assert "regime_router:" not in takeaway
    assert "ETH_USDT_F3" not in takeaway


def test_feature_summary_samples_feature_value(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "feature_value"
    dataset_path.mkdir(parents=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "feature_set": "core",
                "feature_name": "close_return_1",
                "feature_version": "v0.1",
                "symbol": "OLD-USDT",
                "timeframe": "1H",
                "ts": start,
                "value": 0.1,
                "created_at": start,
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "feature_set": "core",
                "feature_name": "close_return_1",
                "feature_version": "v0.1",
                "symbol": "NEW-USDT",
                "timeframe": "1H",
                "ts": start + timedelta(hours=1),
                "value": 0.2,
                "created_at": start + timedelta(hours=1),
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T010000000000Z.parquet")
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "feature_value", 1)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name == "feature_value":
            raise AssertionError("feature summary should not fully read feature_value")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.feature_summary(lake_root)

    assert summary["features"]["symbol"].to_list() == ["NEW-USDT"]
    assert summary["latest"]["symbol"].to_list() == ["NEW-USDT"]
    assert "feature_value 数据集缺失或为空" not in summary["warnings"]


def test_strategy_consumer_summary_samples_decision_audit(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    dataset_path = lake_root / "silver" / "decision_audit"
    dataset_path.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "strategy": "v5",
                "message": "old fallback path",
                "ingest_ts": start,
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "strategy": "v5",
                "message": "new fallback path",
                "ingest_ts": start + timedelta(hours=1),
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T010000000000Z.parquet")
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "decision_audit", 1)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name == "decision_audit":
            raise AssertionError("strategy consumers should not fully read decision_audit")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.strategy_consumer_summary(lake_root)

    assert summary["permissions"]["v5"] == "ALLOW"
    assert summary["fallback_rows"]["message"].to_list() == ["new fallback path"]


def test_cost_model_summary_samples_cost_bucket_daily(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "cost_bucket_daily"
    dataset_path.mkdir(parents=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "day": "2026-05-09",
                "symbol": "OLD/USDT",
                "regime": "normal",
                "fallback_level": "GLOBAL_DEFAULT",
                "source": "global_default",
                "created_at": start,
            }
        ]
    ).write_parquet(dataset_path / "batch_20260509T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "day": "2026-05-10",
                "symbol": "NEW/USDT",
                "regime": "normal",
                "fallback_level": "NONE",
                "source": "actual_fills",
                "created_at": start + timedelta(hours=1),
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T010000000000Z.parquet")
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "cost_bucket_daily", 1)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name == "cost_bucket_daily":
            raise AssertionError("cost model page should not fully read cost_bucket_daily")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.cost_model_summary(lake_root)

    assert summary["costs"]["symbol"].to_list() == ["NEW-USDT"]
    assert "cost_bucket_daily 数据集缺失或为空" not in summary["warnings"]


def test_cost_model_web_tables_emphasize_hard_soft_fallbacks(tmp_path):
    lake_root = tmp_path / "lake"
    created = datetime(2026, 5, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "status": "WARNING",
                    "actual_rows": 1,
                    "mixed_rows": 1,
                    "proxy_rows": 2,
                    "global_default_rows": 0,
                    "hard_fallback_count": 0,
                    "hard_fallback_ratio": 0.0,
                    "soft_fallback_count": 2,
                    "soft_fallback_ratio": 0.5,
                    "proxy_only_count": 2,
                    "api_global_default_count": 0,
                    "api_symbol_proxy_hit_count": 4,
                    "api_regime_fallback_count": 1,
                    "api_degraded_cost_count": 2,
                    "symbols_with_actual_cost": "BTC-USDT",
                    "symbols_with_mixed_cost": "BNB-USDT",
                    "symbols_with_proxy_only": "SOL-USDT,ETH-USDT",
                    "warnings_json": '["soft_fallback_ratio_high"]',
                    "created_at": created,
                }
            ]
        ),
        lake_root / "gold" / "cost_health_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": "2026-05-10",
                    "symbol": "BNB/USDT",
                    "regime": "normal",
                    "source": "mixed_actual_proxy",
                    "cost_source": "mixed_actual_proxy",
                    "sample_count": 35,
                    "fee_bps_p75": 1.0,
                    "spread_bps_p75": 1.5,
                    "slippage_bps_p75": 0.0,
                    "total_cost_bps_p50": 2.0,
                    "total_cost_bps_p75": 2.5,
                    "total_cost_bps_p90": 3.0,
                    "roundtrip_all_in_cost_bps": 5.0,
                    "cost_trust_level": "CANARY",
                    "cost_trusted_for_live_canary": True,
                    "cost_trusted_for_live_scale": False,
                    "fallback_level": "SLIPPAGE_UNKNOWN",
                    "fallback_reason": "slippage_unknown",
                    "created_at": created,
                }
            ]
        ),
        lake_root / "gold" / "cost_bucket_daily",
    )

    summary = readers.cost_model_summary(lake_root)
    health = summary["cost_health"]
    costs = summary["costs"]
    localized_health = localize_frame(health)
    localized_costs = localize_frame(costs)

    assert "重点结论" in localized_health.columns
    assert "关键指标" in localized_health.columns
    assert "硬回退" in localized_health["关键指标"][0]
    assert "软回退" in localized_health["关键指标"][0]
    assert "重点结论" in localized_costs.columns
    assert localized_costs["标的"][0] == "BNB-USDT"
    assert "混合成本" in localized_costs["重点结论"][0]
    assert "往返 all-in" in localized_costs["关键指标"][0]


def test_data_health_hides_pending_v5_paper_telemetry(tmp_path):
    lake_root = tmp_path / "lake"
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    datasets = {row["dataset"] for row in stale_rows}

    assert "v5_paper_strategy_run" not in datasets
    assert "v5_paper_strategy_proposal_ack" not in datasets
    assert "v5_paper_strategy_daily" not in datasets
    assert "v5_paper_slippage_coverage" not in datasets
    assert "v5_trade_opportunity_funnel" not in datasets


def test_data_health_uses_ingest_time_for_v5_expanded_paper_daily(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old_business_date = now - timedelta(days=2)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy_id": "EXPANDED_PAPER",
                    "paper_date": old_business_date.date().isoformat(),
                    "symbol": "WLD-USDT",
                    "entry_count": 0,
                    "ingest_ts": now,
                    "bundle_ts": now,
                }
            ]
        ),
        lake_root / "silver" / "v5_expanded_universe_paper_daily",
    )

    snapshot = readers._dataset_snapshot(lake_root, "v5_expanded_universe_paper_daily")
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["timestamp_column"] == "ingest_ts"
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert "v5_expanded_universe_paper_daily" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_hides_stale_v5_paper_telemetry_when_v5_sync_current(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy_id": "EXPANDED_PAPER",
                    "paper_date": old.date().isoformat(),
                    "symbol": "WLD-USDT",
                    "entry_count": 0,
                }
            ]
        ),
        lake_root / "silver" / "v5_expanded_universe_paper_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    snapshot = readers._dataset_snapshot(lake_root, "v5_expanded_universe_paper_daily")
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["freshness_status"] == "stale"
    assert "v5_expanded_universe_paper_daily" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_hides_stale_entry_quality_dataset_without_freshness_sla(tmp_path):
    lake_root = tmp_path / "lake"
    old = datetime.now(UTC) - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "bundle_ts": old,
                    "ingest_ts": old,
                    "source_path_inside_bundle": "summaries/btc_probe_entry_quality_audit.csv",
                    "same_symbol_reentry_bypass": False,
                    "anti_chase_flag": False,
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake_root / "silver" / "v5_btc_probe_entry_quality_audit",
    )

    snapshot = readers._dataset_snapshot(lake_root, "v5_btc_probe_entry_quality_audit")
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["freshness_status"] == "stale"
    assert "v5_btc_probe_entry_quality_audit" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_hides_optional_empty_ai_research_output(tmp_path):
    stale_rows = readers.data_health_summary(tmp_path / "lake")["stale_datasets"].to_dicts()

    assert "ai_paper_strategy_draft" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_hides_stale_audit_table_without_freshness_sla(tmp_path):
    lake_root = tmp_path / "lake"
    old = datetime.now(UTC) - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_id": "proposal-1",
                    "conflict_field": "proposal_hash",
                    "active_conflict": False,
                    "created_at": old,
                }
            ]
        ),
        lake_root / "gold" / "paper_strategy_identity_conflict",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert "paper_strategy_identity_conflict" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_honors_weekly_registry_freshness_sla(tmp_path):
    lake_root = tmp_path / "lake"
    created_at = datetime.now(UTC) - timedelta(days=2)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": created_at.date().isoformat(),
                    "trial_id": "trial-1",
                    "factor_id": "factor-1",
                    "created_at": created_at,
                }
            ]
        ),
        lake_root / "gold" / "factor_attribution",
    )

    snapshot = readers._dataset_snapshot(lake_root, "factor_attribution")
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["freshness_status"] == "stale"
    assert "factor_attribution" not in {row["dataset"] for row in stale_rows}


def test_data_health_still_reports_dataset_beyond_registry_sla(tmp_path):
    lake_root = tmp_path / "lake"
    created_at = datetime.now(UTC) - timedelta(days=9)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": created_at.date().isoformat(),
                    "trial_id": "trial-1",
                    "factor_id": "factor-1",
                    "created_at": created_at,
                }
            ]
        ),
        lake_root / "gold" / "factor_attribution",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert "factor_attribution" in {row["dataset"] for row in stale_rows}


def test_data_health_does_not_age_out_versioned_alpha_template_controls(tmp_path):
    lake_root = tmp_path / "lake"
    created_at = datetime.now(UTC) - timedelta(days=30)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "template_id": "template-1",
                    "created_at": created_at,
                }
            ]
        ),
        lake_root / "gold" / "alpha_factory_template_registry",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert "alpha_factory_template_registry" not in {
        row["dataset"] for row in stale_rows
    }


def test_dataset_snapshot_falls_back_when_sidecar_lacks_business_time(tmp_path):
    lake_root = tmp_path / "lake"
    dataset = lake_root / "gold" / "paper_strategy_proposal_snapshot"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "proposal_snapshot_id": "snapshot-1",
                    "snapshot_generated_at": now,
                }
            ]
        ),
        dataset,
    )
    (dataset / "_snapshot_meta.json").write_text(
        json.dumps(
            {
                "dataset": "paper_strategy_proposal_snapshot",
                "generated_at": "",
                "row_count": 1,
                "file_count": 1,
            }
        ),
        encoding="utf-8",
    )

    snapshot = readers._dataset_snapshot(
        lake_root,
        "paper_strategy_proposal_snapshot",
        now=now + timedelta(minutes=1),
    )

    assert snapshot.freshness["timestamp_column"] == "snapshot_generated_at"
    assert snapshot.freshness["freshness_status"] == "fresh"


def test_data_health_hides_stale_bnb_latest_when_source_daily_is_current(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy_id": "BNB_PAPER",
                    "paper_date": old.date().isoformat(),
                    "ingest_ts": old,
                    "bundle_ts": old,
                    "entry_count": 1,
                }
            ]
        ),
        lake_root / "gold" / "v5_bnb_paper_strategy_daily_latest",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy_id": "BNB_PAPER",
                    "paper_date": now.date().isoformat(),
                    "ingest_ts": now,
                    "bundle_ts": now,
                    "entry_count": 2,
                }
            ]
        ),
        lake_root / "gold" / "v5_bnb_paper_strategy_daily",
    )

    snapshot = readers._dataset_snapshot(lake_root, "v5_bnb_paper_strategy_daily_latest")
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["freshness_status"] == "stale"
    assert "v5_bnb_paper_strategy_daily_latest" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_keeps_stale_bnb_latest_when_source_daily_is_stale(tmp_path):
    lake_root = tmp_path / "lake"
    old = datetime.now(UTC) - timedelta(days=3)
    daily_row = {
        "strategy_id": "BNB_PAPER",
        "paper_date": old.date().isoformat(),
        "ingest_ts": old,
        "bundle_ts": old,
        "entry_count": 1,
    }
    write_parquet_dataset(
        pl.DataFrame([daily_row]),
        lake_root / "gold" / "v5_bnb_paper_strategy_daily_latest",
    )
    write_parquet_dataset(
        pl.DataFrame([daily_row]),
        lake_root / "gold" / "v5_bnb_paper_strategy_daily",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    by_dataset = {row["dataset"]: row for row in stale_rows}

    assert by_dataset["v5_bnb_paper_strategy_daily_latest"]["status"] == "stale"


def test_data_health_hides_stale_bnb_paper_when_v5_telemetry_is_current(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=5)
    paper_daily = {
        "strategy_id": "BNB_PAPER",
        "paper_date": old.date().isoformat(),
        "ingest_ts": old,
        "bundle_ts": old,
        "entry_count": 1,
    }
    paper_run = {
        "strategy_id": "BNB_PAPER",
        "entry_ts": old,
        "ingest_ts": old,
        "bundle_ts": old,
        "status": "closed",
    }
    write_parquet_dataset(
        pl.DataFrame([paper_daily]),
        lake_root / "gold" / "v5_bnb_paper_strategy_daily_latest",
    )
    write_parquet_dataset(
        pl.DataFrame([paper_daily]),
        lake_root / "gold" / "v5_bnb_paper_strategy_daily",
    )
    write_parquet_dataset(
        pl.DataFrame([paper_run]),
        lake_root / "gold" / "v5_bnb_paper_strategy_runs",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now.isoformat(),
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    datasets = {row["dataset"] for row in stale_rows}

    assert "v5_bnb_paper_strategy_daily" not in datasets
    assert "v5_bnb_paper_strategy_daily_latest" not in datasets
    assert "v5_bnb_paper_strategy_runs" not in datasets


def test_stale_dataset_rows_keep_schema_when_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(readers, "DATASET_PATHS", {})

    table = readers._stale_dataset_rows(tmp_path / "lake")

    assert table.is_empty()
    assert {"takeaway", "severity", "next_action", "dataset", "status"}.issubset(
        table.columns
    )


def test_data_health_stale_rows_include_takeaway_severity_and_action(tmp_path):
    lake_root = tmp_path / "lake"
    old = datetime.now(UTC) - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "venue": "okx",
                    "symbol": "BTC-USDT",
                    "market_type": "SPOT",
                    "timeframe": "1H",
                    "ts": old,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "volume": 1.0,
                    "source": "test",
                    "ingest_ts": old,
                }
            ]
        ),
        lake_root / "silver" / "market_bar",
    )

    table = readers.data_health_summary(lake_root)["stale_datasets"]
    localized = localize_frame(table)
    market_row = next(row for row in table.to_dicts() if row["dataset"] == "market_bar")

    assert {"takeaway", "severity", "next_action"}.issubset(table.columns)
    assert "行情 K 线 已过期" in market_row["takeaway"]
    assert market_row["severity"] == "CRITICAL"
    assert "OKX REST/WS" in market_row["next_action"]
    assert "重点结论" in localized.columns
    assert "影响级别" in localized.columns
    assert "建议动作" in localized.columns


def test_data_health_uses_generated_at_for_factor_strategy_bridge_freshness(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": (now - timedelta(days=3)).date().isoformat(),
                    "generated_at": now.isoformat(),
                    "factor_id": "core.close_return_24",
                    "factor_family": "momentum",
                    "correlation_cluster_id": "cluster_001",
                    "symbol": "SOL-USDT",
                    "regime": "TREND_UP",
                    "horizon": "8h",
                    "horizon_hours": "8",
                    "forward_sample_count": 120,
                    "forward_cost_adjusted_score": 12.5,
                    "bridge_candidate_id": "v5.factor_bridge.core.close_return_24",
                    "eligible_for_alpha_factory": "strategy_review_pending",
                    "blocking_reasons": "[]",
                    "recommended_action": "REVIEW_FOR_ALPHA_FACTORY_STRATEGY",
                    "live_order_effect": "none_read_only_research",
                }
            ]
        ),
        lake_root / "gold" / "factor_strategy_bridge_candidates",
    )

    snapshot = readers._dataset_snapshot(
        lake_root,
        "factor_strategy_bridge_candidates",
        now=now,
    )
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["timestamp_column"] == "generated_at"
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert "factor_strategy_bridge_candidates" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_snapshot_uses_generated_at_for_strategy_opportunity_without_meta(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "SOL-USDT",
                    "as_of_ts": (now + timedelta(hours=4)).isoformat(),
                    "generated_at": now.isoformat(),
                    "created_at": now.isoformat(),
                    "state": "PAPER",
                    "live_order_effect": "none_read_only_research",
                }
            ]
        ),
        lake_root / "gold" / "strategy_opportunity_advisory",
    )

    snapshot = readers._dataset_snapshot(
        lake_root,
        "strategy_opportunity_advisory",
        now=now,
    )
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["timestamp_column"] == "generated_at"
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert "strategy_opportunity_advisory" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_snapshot_uses_created_at_for_opportunity_cost_daily_without_meta(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": (now - timedelta(days=45)).date(),
                    "false_block_count": 1,
                    "loss_saved_count": 0,
                    "veto_net_value_bps": -10.0,
                    "opportunity_cost_status": "VETO_VALUE_NEGATIVE_REVIEW_EXCEPTIONS",
                    "created_at": now.isoformat(),
                    "source": "test",
                }
            ]
        ),
        lake_root / "gold" / "quant_lab_opportunity_cost_daily",
    )

    snapshot = readers._dataset_snapshot(
        lake_root,
        "quant_lab_opportunity_cost_daily",
        now=now,
    )
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["timestamp_column"] == "created_at"
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert "quant_lab_opportunity_cost_daily" not in {
        row["dataset"] for row in stale_rows
    }


def test_data_health_uses_generated_at_for_cost_probe_fill_bill_match(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "generated_at": now.isoformat(),
                    "symbol": "SOL-USDT",
                    "authorization_id": "auth-sol",
                    "roundtrip_id": "entry:exit",
                    "entry_order_id": "entry",
                    "exit_order_id": "exit",
                    "entry_trade_id": "entry-trade",
                    "exit_trade_id": "exit-trade",
                    "entry_bill_id": "entry-bill",
                    "exit_bill_id": "exit-bill",
                    "entry_fee_from_fill": "0.004",
                    "entry_fee_from_bill": "0.004",
                    "exit_fee_from_fill": "0.004",
                    "exit_fee_from_bill": "0.004",
                    "fee_diff_usdt": "0",
                    "bill_match_status": "PASS",
                }
            ]
        ),
        lake_root / "gold" / "cost_probe_fill_bill_match",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "generated_at": now.isoformat(),
                    "symbol": "SOL-USDT",
                    "authorization_id": "auth-sol",
                    "roundtrip_id": "entry:exit",
                    "v5_roundtrip_cost_bps": "10",
                    "quant_lab_roundtrip_cost_bps": "11",
                    "okx_bill_roundtrip_cost_bps": "10.5",
                    "diff_bps": "1",
                    "status": "PASS",
                    "reason": "comparable_values=okx_bill,quant_lab,v5",
                    "cost_bucket_source": "bootstrap_cost_probe",
                    "bill_match_status": "PASS",
                }
            ]
        ),
        lake_root / "gold" / "cost_probe_cost_disagreement",
    )

    snapshot = readers._dataset_snapshot(lake_root, "cost_probe_fill_bill_match", now=now)
    disagreement_snapshot = readers._dataset_snapshot(
        lake_root,
        "cost_probe_cost_disagreement",
        now=now,
    )
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.freshness["timestamp_column"] == "generated_at"
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert disagreement_snapshot.freshness["timestamp_column"] == "generated_at"
    assert disagreement_snapshot.freshness["freshness_status"] == "fresh"
    assert "cost_probe_fill_bill_match" not in {row["dataset"] for row in stale_rows}
    assert "cost_probe_cost_disagreement" not in {row["dataset"] for row in stale_rows}


def test_data_health_treats_missing_cost_probe_audits_as_optional(tmp_path):
    lake_root = tmp_path / "lake"

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert "cost_probe_fill_bill_match" not in {row["dataset"] for row in stale_rows}
    assert "cost_probe_cost_disagreement" not in {row["dataset"] for row in stale_rows}


def test_cost_model_summary_exposes_cost_probe_disagreement_failures(tmp_path):
    lake_root = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "generated_at": "2026-07-03T15:20:09Z",
                    "symbol": "SOL-USDT",
                    "authorization_id": "auth-sol",
                    "roundtrip_id": "entry:exit",
                    "v5_roundtrip_cost_bps": "21.4",
                    "quant_lab_roundtrip_cost_bps": "51.9",
                    "okx_bill_roundtrip_cost_bps": "31.4",
                    "diff_bps": "30.5",
                    "status": "FAIL",
                    "reason": "comparable_values=okx_bill,quant_lab,v5",
                    "cost_bucket_source": "mixed_actual_proxy",
                    "bill_match_status": "PASS",
                }
            ]
        ),
        lake_root / "gold" / "cost_probe_cost_disagreement",
    )

    summary = readers.cost_model_summary(lake_root)

    rows = summary["cost_probe_cost_disagreement"].to_dicts()
    assert rows[0]["symbol"] == "SOL-USDT"
    assert rows[0]["status"] == "FAIL"
    assert "cost_probe_cost_disagreement_fail: fail_count=1" in summary["warnings"]


def test_dataset_snapshot_falls_back_when_file_index_misses_partitioned_files(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    partition = lake_root / "gold" / "market_regime_daily" / "as_of_date=2026-06-08"
    partition.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-08",
                "current_regime": "SIDEWAYS",
                "active_regimes_json": "[]",
                "regime_confidence": 0.7,
                "btc_24h_return_bps": 0.0,
                "eth_24h_return_bps": 0.0,
                "sol_24h_return_bps": 0.0,
                "bnb_24h_return_bps": 0.0,
                "broad_market_positive_count": 0,
                "realized_vol_bps": 1.0,
                "avg_spread_bps": 1.0,
                "liquidity_thin": False,
                "created_at": now,
                "source": "test",
                "schema_version": "test",
            }
        ]
    ).write_parquet(partition / "part.parquet")
    write_parquet_dataset(
        pl.DataFrame(
            [{"dataset": "gold/other_dataset", "path": "gold/other_dataset/data.parquet"}]
        ),
        lake_root / "bronze" / "lake_file_index",
    )

    snapshot = readers._dataset_snapshot(lake_root, "market_regime_daily", now=now)
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.rows == 1
    assert snapshot.warning is None
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert "market_regime_daily" not in {row["dataset"] for row in stale_rows}


def test_dataset_snapshot_merges_stale_file_index_with_new_partitioned_files(tmp_path):
    lake_root = tmp_path / "lake"
    dataset = lake_root / "gold" / "market_regime_daily"
    old = datetime.now(UTC) - timedelta(days=2)
    now = datetime.now(UTC)
    old_partition = dataset / "as_of_date=2026-06-06"
    old_partition.mkdir(parents=True)
    old_file = old_partition / "part.parquet"
    pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-06",
                "current_regime": "SIDEWAYS",
                "created_at": old,
            }
        ]
    ).write_parquet(old_file)
    old_mtime = old.timestamp()
    os.utime(old_file, (old_mtime, old_mtime))
    os.utime(old_partition, (old_mtime, old_mtime))
    os.utime(dataset, (old_mtime, old_mtime))
    build_lake_file_index(lake_root, ["gold/market_regime_daily"])

    new_partition = dataset / "as_of_date=2026-06-08"
    new_partition.mkdir(parents=True)
    new_file = new_partition / "part.parquet"
    pl.DataFrame(
        [
            {
                "as_of_date": "2026-06-08",
                "current_regime": "TREND_UP",
                "created_at": now,
            }
        ]
    ).write_parquet(new_file)
    new_mtime = now.timestamp()
    os.utime(new_file, (new_mtime, new_mtime))
    os.utime(new_partition, (new_mtime, new_mtime))
    os.utime(dataset, (new_mtime, new_mtime))

    snapshot = readers._dataset_snapshot(lake_root, "market_regime_daily", now=now)
    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert snapshot.rows == 2
    assert snapshot.warning is None
    assert snapshot.freshness["freshness_status"] == "fresh"
    assert snapshot.freshness["latest_timestamp"] == now.isoformat()
    assert "market_regime_daily" not in {row["dataset"] for row in stale_rows}


def test_data_health_marks_missing_market_rollups_as_pending_when_sources_exist(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [{"symbol": "BNB-USDT", "ts": now, "size": 1.0}]
        ),
        lake_root / "silver" / "trade_print",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "channel": "books5",
                    "ts": now,
                    "asks_json": "[[\"101\", \"1\"]]",
                    "bids_json": "[[\"100\", \"1\"]]",
                }
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )

    rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    by_dataset = {row["dataset"]: row for row in rows}

    for dataset in ("trade_activity_1m", "orderbook_spread_1m"):
        assert by_dataset[dataset]["status"] == "derived_rollup_pending"
        assert by_dataset[dataset]["severity"] == "WARNING"
        assert "build-market-data-rollups" in by_dataset[dataset]["next_action"]
        assert "暂无派生数据" in by_dataset[dataset]["takeaway"]


def test_data_health_reads_v5_paper_run_schema_evolution_without_warning(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "v5_paper_strategy_run"
    dataset.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "strategy_id": "SOL_PAPER",
                "symbol": "SOL-USDT",
                "created_at": datetime.now(UTC),
            }
        ]
    ).write_parquet(dataset / "part-old.parquet")
    pl.DataFrame(
        [
            {
                "strategy_id": "SOL_PAPER",
                "symbol": "SOL-USDT",
                "created_at": datetime.now(UTC),
                "arrival_mid": 150.0,
                "estimated_spread_bps": 1.5,
            }
        ]
    ).write_parquet(dataset / "part-new.parquet")

    states = {state.name: state for state in readers.dataset_states(tmp_path / "lake")}

    state = states["v5_paper_strategy_run"]
    assert state.rows == 2
    assert state.warning is None


def test_data_health_treats_v5_trades_as_event_driven_when_telemetry_is_current(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                    "decision_audit_count_24h": 1,
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
                    "symbol": "SOL-USDT",
                    "side": "buy",
                    "ts_utc": old,
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake_root / "silver" / "v5_trade_event",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert not any(row["dataset"] == "v5_trade_event" for row in stale_rows)


def test_data_health_treats_bnb_profit_lock_as_event_driven_when_telemetry_current(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                    "decision_audit_count_24h": 1,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy_id": "BNB_SWING_EXIT_POLICY_REVIEW",
                    "symbol": "BNB-USDT",
                    "entry_ts": old,
                    "exit_ts": old + timedelta(hours=24),
                    "bundle_ts": old,
                    "ingest_ts": old,
                }
            ]
        ),
        lake_root / "silver" / "v5_bnb_profit_lock_shadow",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert not any(row["dataset"] == "v5_bnb_profit_lock_shadow" for row in stale_rows)


def test_data_health_treats_v5_usage_and_attribution_as_event_driven_when_current(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                    "decision_audit_count_24h": 1,
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
                    "run_id": "20260603_01",
                    "mode": "shadow",
                    "ts_utc": old,
                    "bundle_ts": old,
                    "ingest_ts": old,
                }
            ]
        ),
        lake_root / "silver" / "v5_quant_lab_usage",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "event_key": "req-old",
                    "run_id": "20260601_01",
                    "endpoint": "/v1/strategy-opportunity-advisory/v5-compact",
                    "status_code": 200,
                    "ts_utc": old,
                    "bundle_ts": old,
                    "ingest_ts": old,
                }
            ]
        ),
        lake_root / "silver" / "v5_quant_lab_request",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BNB-USDT",
                    "exit_ts": old,
                    "bundle_ts": old,
                    "ingest_ts": old,
                    "attribution": "min_hold_violation",
                }
            ]
        ),
        lake_root / "silver" / "v5_bnb_negative_expectancy_attribution",
    )
    attribution = pl.DataFrame(
        [
            {
                "symbol": "BNB-USDT",
                "exit_ts": old,
                "bundle_ts": old,
                "ingest_ts": old,
                "attribution": "min_hold_violation",
            }
        ]
    )
    write_parquet_dataset(
        attribution,
        lake_root / "silver" / "v5_negative_expectancy_attribution",
    )
    write_parquet_dataset(
        attribution,
        lake_root / "gold" / "v5_negative_expectancy_attribution",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    datasets = {row["dataset"] for row in stale_rows}

    assert "v5_quant_lab_usage" not in datasets
    assert "v5_quant_lab_request" not in datasets
    assert "v5_bnb_negative_expectancy_attribution" not in datasets
    assert "v5_negative_expectancy_attribution" not in datasets
    assert "v5_negative_expectancy_attribution_silver" not in datasets


def test_web_tracks_v5_quant_lab_request_freshness(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "event_key": "req-1",
                    "run_id": "20260603_01",
                    "endpoint": "/v1/strategy-opportunity-advisory/v5-compact",
                    "status_code": 200,
                    "ts_utc": now,
                    "bundle_ts": now,
                    "ingest_ts": now,
                }
            ]
        ),
        lake_root / "silver" / "v5_quant_lab_request",
    )

    states = {state.name: state for state in readers.dataset_states(lake_root)}
    snapshot = readers._dataset_snapshot(lake_root, "v5_quant_lab_request")

    assert states["v5_quant_lab_request"].rows == 1
    assert snapshot.freshness["freshness_status"] == "fresh"


def test_web_exposes_cost_probe_event_streams(tmp_path):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "event_key": "order-event-key",
                    "event_id": "probe-entry-1|order:entry:filled|2026-05-14T08:30:02Z",
                    "event_type": "order:entry:filled",
                    "event_ts": "2026-05-14T08:30:02Z",
                    "order_key": "probe-entry-1",
                    "symbol": "BTC-USDT",
                    "live_order_effect": "live_cost_probe_order",
                    "ingest_ts": now,
                    "bundle_ts": now,
                }
            ]
        ),
        lake_root / "silver" / "v5_cost_probe_order_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "event_key": "roundtrip-event-key",
                    "event_id": "probe-roundtrip-1|roundtrip:closed|2026-05-14T08:30:20Z",
                    "event_type": "roundtrip:closed",
                    "event_ts": "2026-05-14T08:30:20Z",
                    "roundtrip_key": "probe-roundtrip-1",
                    "symbol": "BTC-USDT",
                    "live_order_effect": "live_cost_probe_roundtrip",
                    "ingest_ts": now,
                    "bundle_ts": now,
                }
            ]
        ),
        lake_root / "silver" / "v5_cost_probe_roundtrip_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "event_type": "cost_probe_live_execution_status",
                    "generated_at_utc": now,
                    "status": "AUTH_VALIDATED",
                    "manual_probe_symbol": "BTC/USDT",
                    "authorization_issued_at": now - timedelta(minutes=10),
                    "authorization_expires_at": now - timedelta(minutes=5),
                    "authorization_age_sec": 0,
                    "authorization_fresh": True,
                    "authorization_validated": True,
                    "authorization_consumed": False,
                    "recovery_required": False,
                    "ingest_ts": now,
                    "bundle_ts": now,
                    "raw_payload_json": "{}",
                }
            ]
        ),
        lake_root / "silver" / "v5_cost_probe_live_execution_status",
    )

    states = {state.name: state for state in readers.dataset_states(lake_root)}
    summary = readers.v5_telemetry_summary(lake_root)

    assert states["v5_cost_probe_order_event"].rows == 1
    assert states["v5_cost_probe_roundtrip_event"].rows == 1
    assert summary["latest"]["cost_probe_order_event"]["event_type"] == "order:entry:filled"
    assert summary["latest"]["cost_probe_roundtrip_event"]["event_type"] == "roundtrip:closed"
    live_status = summary["latest"]["cost_probe_live_execution_status"]
    assert live_status["authorization_fresh"] is True
    assert live_status["authorization_fresh_at_read"] is False
    assert live_status["authorization_expired_at_read"] is True
    assert live_status["authorization_seconds_to_expiry_at_read"] < 0


def test_data_health_treats_v5_cost_usage_and_fallback_as_event_driven_when_current(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": now.date().isoformat(),
                    "status": "OK",
                    "latest_bundle_ts": now,
                }
            ]
        ),
        lake_root / "gold" / "strategy_health_daily",
    )
    for dataset in ("v5_quant_lab_cost_usage", "v5_quant_lab_fallback"):
        write_parquet_dataset(
            pl.DataFrame(
                [
                    {
                        "strategy": "v5",
                        "bundle_ts": old,
                        "ingest_ts": old,
                        "ts_utc": old.isoformat(),
                    }
                ]
            ),
            lake_root / "silver" / dataset,
        )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    datasets = {row["dataset"] for row in stale_rows}

    assert "v5_quant_lab_cost_usage" not in datasets
    assert "v5_quant_lab_fallback" not in datasets


def test_data_health_treats_okx_bills_as_event_driven_when_backfill_current(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "endpoint": "/api/v5/account/bills",
                    "ingest_ts": old,
                    "raw_json": "{}",
                }
            ]
        ),
        lake_root / "bronze" / "okx_private_readonly" / "bills",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "day": now.date().isoformat(),
                    "job_name": "okx-backfill-readonly",
                    "status": "succeeded",
                    "started_at": now - timedelta(seconds=5),
                    "finished_at": now,
                    "duration_seconds": 5.0,
                    "error_type": None,
                    "error_message": None,
                }
            ]
        ),
        lake_root / "gold" / "job_run_history",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert "okx_private_readonly_bills" not in {row["dataset"] for row in stale_rows}


def test_data_health_hides_empty_legacy_expanded_shadow_when_automation_is_active(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            schema={
                "as_of_date": pl.Utf8,
                "generated_at": pl.Datetime(time_zone="UTC"),
                "symbol": pl.Utf8,
                "quality_score": pl.Float64,
            }
        ),
        lake_root / "gold" / "expanded_crypto_universe_shadow",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": now.date().isoformat(),
                    "generated_at": now,
                    "symbol": "TRX-USDT",
                    "universe_type": "expanded_paper",
                }
            ]
        ),
        lake_root / "gold" / "expanded_universe_candidate",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()

    assert not any(row["dataset"] == "expanded_crypto_universe_shadow" for row in stale_rows)


def test_data_health_uses_generation_time_for_risk_on_shadow_and_hides_history_snapshots(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=5)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "run_id": "run-1",
                    "decision_ts": old,
                    "generated_at": now,
                    "strategy_candidate": "v5.risk_on_multi_buy_top1_shadow",
                }
            ]
        ),
        lake_root / "gold" / "risk_on_multi_buy_shadow",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "generated_at_utc": old,
                    "check_name": "anti_leakage",
                    "status": "PASS",
                }
            ]
        ),
        lake_root / "gold" / "v5_entry_quality_history_anti_leakage_check",
    )

    stale_rows = readers.data_health_summary(lake_root)["stale_datasets"].to_dicts()
    datasets = {row["dataset"] for row in stale_rows}

    assert "risk_on_multi_buy_shadow" not in datasets
    assert "v5_entry_quality_history_anti_leakage_check" not in datasets


def test_okx_collector_summary_uses_metadata_for_raw_collector_counts(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name in {"market_bar", "okx_public_ws", "trade_print", "orderbook_snapshot"}:
            raise AssertionError(f"{dataset_name} should be summarized from parquet metadata")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.okx_collector_summary(lake_root)

    assert summary["trade_print_rows"] == 1
    assert summary["orderbook_snapshot_rows"] == 0
    assert summary["okx_public_ws_status"] == "RUNNING"


def test_market_regime_summary_samples_heavy_market_datasets(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    original = readers.read_dataset_with_warning

    def guarded_read_dataset_with_warning(lake_root_arg, dataset_name):
        if dataset_name in {"market_bar", "trade_print", "orderbook_snapshot"}:
            raise AssertionError(f"{dataset_name} should be sampled for market regime")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset_with_warning)

    summary = readers.market_regime_summary(lake_root)

    assert not summary["regimes"].is_empty()
    assert not summary["trade_activity"].is_empty()


def test_market_regime_summary_uses_lazy_aggregation_for_heavy_streams(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "ts": start,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["99", "1"]]',
                }
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )
    original = readers.read_recent_dataset_with_warning

    def guarded_read_recent_dataset_with_warning(lake_root_arg, dataset_name, **kwargs):
        if dataset_name in {"trade_print", "orderbook_snapshot"}:
            raise AssertionError(f"{dataset_name} should be lazily aggregated for market status")
        return original(lake_root_arg, dataset_name, **kwargs)

    monkeypatch.setattr(
        readers,
        "read_recent_dataset_with_warning",
        guarded_read_recent_dataset_with_warning,
    )

    summary = readers.market_regime_summary(lake_root)

    assert not summary["regimes"].is_empty()
    assert not summary["spread_bps"].is_empty()
    assert not summary["trade_activity"].is_empty()


def test_market_regime_summary_hides_symbols_lagging_current_market_window(tmp_path):
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = []
    for ts, symbol, close in [
        (start, "AAVE-USDT", 80.0),
        (start + timedelta(hours=5), "BTC-USDT", 100.0),
        (start + timedelta(hours=5), "ETH-USDT", 200.0),
    ]:
        row = _bar(ts, close=close)
        row["symbol"] = symbol
        rows.append(row)
    write_market_bars(lake_root, rows)

    summary = readers.market_regime_summary(lake_root)
    regimes = summary["regimes"].to_dicts()

    assert {row["symbol"] for row in regimes} == {"BTC-USDT", "ETH-USDT"}
    assert all(row["latest_ts"] == start + timedelta(hours=5) for row in regimes)
    assert summary["hidden_stale_regime_rows"] == 1
    assert summary["market_regime_latest_ts"] == start + timedelta(hours=5)
    assert not summary["warnings"]


def test_recent_heavy_dataset_read_uses_latest_file_window(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "symbol": "OLD-USDT",
                "trade_id": "old",
                "price": 1.0,
                "size": 1.0,
                "ts": start,
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "symbol": "NEW-USDT",
                "trade_id": "new",
                "price": 2.0,
                "size": 1.0,
                "ts": start + timedelta(hours=1),
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T010000000000Z.parquet")
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "trade_print", 1)

    recent, warning = readers.read_recent_dataset_with_warning(lake_root, "trade_print")

    assert warning is None
    assert recent["symbol"].to_list() == ["NEW-USDT"]


def test_recent_heavy_dataset_keeps_per_symbol_samples(tmp_path):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rows = [
        {
            "symbol": "BTC-USDT",
            "trade_id": f"btc-{index}",
            "price": 100.0,
            "size": 1.0,
            "ts": start + timedelta(seconds=index),
        }
        for index in range(30)
    ]
    rows.extend(
        [
            {
                "symbol": "ETH-USDT",
                "trade_id": "eth-1",
                "price": 200.0,
                "size": 1.0,
                "ts": start + timedelta(seconds=5),
            },
            {
                "symbol": "ETH-USDT",
                "trade_id": "eth-2",
                "price": 201.0,
                "size": 1.0,
                "ts": start + timedelta(seconds=6),
            },
        ]
    )
    pl.DataFrame(rows).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")

    recent, warning = readers.read_recent_dataset_with_warning(
        lake_root,
        "trade_print",
        limit=10,
        lookback_hours=1,
    )

    assert warning is None
    assert set(recent["symbol"].to_list()) == {"BTC-USDT", "ETH-USDT"}


def test_recent_heavy_dataset_filters_string_timestamps_by_lookback(tmp_path):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    rows = [
        {
            "symbol": "OLD-USDT",
            "trade_id": "old",
            "price": 1.0,
            "size": 1.0,
            "ts": "2026-05-10T00:00:00Z",
        },
        {
            "symbol": "NEW-USDT",
            "trade_id": "new",
            "price": 2.0,
            "size": 1.0,
            "ts": "2026-05-10T10:00:00Z",
        },
    ]
    pl.DataFrame(rows).write_parquet(dataset_path / "batch_20260510T100000000000Z.parquet")

    recent, warning = readers.read_recent_dataset_with_warning(
        lake_root,
        "trade_print",
        limit=100,
        lookback_hours=1,
    )

    assert warning is None
    assert recent["symbol"].to_list() == ["NEW-USDT"]


def test_recent_heavy_dataset_ignores_internal_lake_paths(tmp_path):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "trade_id": "live",
                "price": 100.0,
                "size": 1.0,
                "ts": datetime(2026, 5, 10, tzinfo=UTC),
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")
    tmp_dir = dataset_path / "._tmp"
    tmp_dir.mkdir()
    pl.DataFrame(
        [
            {
                "symbol": "TMP-USDT",
                "trade_id": "tmp",
                "price": 1.0,
                "size": 1.0,
                "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
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
                "ts": datetime(2026, 5, 10, 2, tzinfo=UTC),
            }
        ]
    ).write_parquet(backup / "backup.parquet")

    recent, warning = readers.read_recent_dataset_with_warning(lake_root, "trade_print")

    assert warning is None
    assert recent["symbol"].to_list() == ["BTC-USDT"]


def test_recent_heavy_dataset_ignores_stale_data_file_when_batches_exist(tmp_path):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "symbol": "SOL-USDT",
                "trade_id": f"old-sol-{index}",
                "price": 90.0,
                "size": 1.0,
                "ts": start,
            }
            for index in range(30)
        ]
    ).write_parquet(dataset_path / "data.parquet")
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "trade_id": "new-btc",
                "price": 100.0,
                "size": 1.0,
                "ts": start + timedelta(hours=1),
            },
            {
                "symbol": "ETH-USDT",
                "trade_id": "new-eth",
                "price": 200.0,
                "size": 1.0,
                "ts": start + timedelta(hours=1),
            },
        ]
    ).write_parquet(dataset_path / "batch_20260510T010000000000Z.parquet")

    recent, warning = readers.read_recent_dataset_with_warning(lake_root, "trade_print")

    assert warning is None
    assert set(recent["symbol"].to_list()) == {"BTC-USDT", "ETH-USDT"}


def test_research_display_dataset_uses_recent_file_sampling(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "v5_candidate_event"
    dataset_path.mkdir(parents=True)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "candidate_id": "old",
                "ts_utc": start,
                "symbol": "OLD-USDT",
                "strategy_candidate": "old",
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")
    pl.DataFrame(
        [
            {
                "candidate_id": "new",
                "ts_utc": start + timedelta(hours=1),
                "symbol": "NEW-USDT",
                "strategy_candidate": "new",
            }
        ]
    ).write_parquet(dataset_path / "batch_20260510T010000000000Z.parquet")
    monkeypatch.setitem(readers.WEB_RECENT_FILE_LIMITS, "v5_candidate_event", 1)

    recent, warning = readers.read_recent_dataset_with_warning(
        lake_root,
        "v5_candidate_event",
    )

    assert warning is None
    assert recent["symbol"].to_list() == ["NEW-USDT"]


def test_alpha_gate_summary_samples_research_detail_tables(tmp_path, monkeypatch):
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    for dataset_name in readers.WEB_RESEARCH_SAMPLE_DATASETS:
        dataset_path = readers.dataset_path_for(lake_root, dataset_name)
        dataset_path.mkdir(parents=True)
        pl.DataFrame(
            [
                {
                    "candidate_id": f"{dataset_name}-sample",
                    "ts_utc": start,
                    "label_ts": start + timedelta(hours=4),
                    "generated_at": start,
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.sol_protect_exception",
                    "candidate_name": "v5.sol_protect_exception",
                    "source_dataset": "test",
                    "horizon_hours": 24,
                    "recommendation": "shadow_only",
                }
            ]
        ).write_parquet(dataset_path / "batch_20260510T000000000000Z.parquet")
    original = readers.read_dataset_with_warning

    def guarded_read_dataset(lake_root_arg, dataset_name):
        if dataset_name in readers.WEB_RESEARCH_SAMPLE_DATASETS:
            raise AssertionError(f"{dataset_name} should use recent sample reads")
        return original(lake_root_arg, dataset_name)

    monkeypatch.setattr(readers, "read_dataset_with_warning", guarded_read_dataset)

    summary = readers.alpha_gate_summary(lake_root)

    assert summary["candidate_events"].height == 1
    assert summary["candidate_labels"].height == 1
    assert summary["strategy_samples"].height == 1
    assert "v5_candidate_event 候选快照尚未入湖" not in summary["warnings"]
    assert "v5_candidate_label 前向标签尚未生成" not in summary["warnings"]


def test_market_regime_warns_when_ws_universe_is_incomplete(tmp_path):
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    eth_bar = _bar(start, close=200.0)
    eth_bar["symbol"] = "ETH-USDT"
    write_market_bars(lake_root, [_bar(start, close=100.0), eth_bar])
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "ts": start,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["99", "1"]]',
                }
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "trade_id": "1",
                    "price": 100.0,
                    "size": 1.0,
                    "ts": start,
                }
            ]
        ),
        lake_root / "silver" / "trade_print",
    )

    summary = readers.market_regime_summary(lake_root)

    warnings = "\n".join(summary["warnings"])
    assert "OKX WebSocket universe 不完整" in warnings
    assert "ETH-USDT" in warnings


def test_market_regime_ignores_symbols_outside_configured_ws_universe(tmp_path):
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    rlusd_bar = _bar(start, close=100.0)
    rlusd_bar["symbol"] = "RLUSD-USDT"
    write_market_bars(lake_root, [_bar(start, close=100.0), rlusd_bar])
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "ts": start,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["99", "1"]]',
                }
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "trade_id": "1",
                    "price": 100.0,
                    "size": 1.0,
                    "ts": start,
                }
            ]
        ),
        lake_root / "silver" / "trade_print",
    )

    summary = readers.market_regime_summary(lake_root)

    warnings = "\n".join(summary["warnings"])
    assert "RLUSD-USDT" not in warnings


def test_market_regime_uses_collector_subscription_for_universe_completeness(tmp_path):
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    eth_bar = _bar(start, close=200.0)
    eth_bar["symbol"] = "ETH-USDT"
    write_market_bars(lake_root, [_bar(start, close=100.0), eth_bar])
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "channel": "books5",
                    "ts": start,
                    "asks_json": '[["101", "1"]]',
                    "bids_json": '[["99", "1"]]',
                }
            ]
        ),
        lake_root / "silver" / "orderbook_snapshot",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "symbol": "BTC-USDT",
                    "trade_id": "1",
                    "price": 100.0,
                    "size": 1.0,
                    "ts": start,
                }
            ]
        ),
        lake_root / "silver" / "trade_print",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "collector_name": "okx_public_ws",
                    "started_at": start,
                    "last_message_at": start,
                    "updated_at": start,
                    "messages_read": 10,
                    "reconnect_count": 0,
                    "error_count": 0,
                    "subscribed_symbols": '["BTC-USDT", "ETH-USDT"]',
                    "subscribed_channels": '["trades", "books5"]',
                    "status": "RUNNING",
                }
            ]
        ),
        lake_root / "bronze" / "collector_health" / "okx_public_ws",
    )

    summary = readers.market_regime_summary(lake_root)

    warnings = "\n".join(summary["warnings"])
    assert "ETH-USDT" not in warnings
    assert "OKX WebSocket universe" not in warnings


def test_dataset_freshness_unknown_when_populated_table_has_no_timestamp():
    payload = readers.dataset_freshness_payload(
        "decision_audit",
        pl.DataFrame([{"payload": "no timestamp"}]),
    )

    assert payload["freshness_status"] == "unknown"
    assert payload["freshness_status"] != "missing"


def test_dataset_freshness_flags_future_timestamps():
    now = datetime(2026, 6, 28, 3, tzinfo=UTC)

    payload = readers.dataset_freshness_payload(
        "custom_future_dataset",
        pl.DataFrame([{"ts": now + timedelta(hours=2)}]),
        now=now,
    )

    assert payload["freshness_status"] == "future"
    assert payload["freshness_seconds"] < 0
    assert payload["future_seconds"] > 0
    assert payload["freshness_warning"] == "timestamp_after_export_reference"


def test_candidate_label_freshness_prefers_generated_at_over_future_label_ts():
    now = datetime(2026, 6, 28, 3, tzinfo=UTC)

    payload = readers.dataset_freshness_payload(
        "expanded_universe_candidate_label",
        pl.DataFrame(
            [
                {
                    "label_ts": now + timedelta(days=2),
                    "generated_at": now - timedelta(minutes=15),
                }
            ]
        ),
        now=now,
    )

    assert payload["freshness_status"] == "fresh"
    assert payload["timestamp_column"] == "generated_at"
    assert payload["latest_timestamp"] == (now - timedelta(minutes=15)).isoformat()


def test_current_paper_tracker_freshness_uses_bundle_ingest_not_creation_time():
    now = datetime(2026, 7, 17, 11, tzinfo=UTC)
    created_at = now - timedelta(days=2)
    ingest_ts = now - timedelta(minutes=10)

    payload = readers.dataset_freshness_payload(
        "paper_strategy_trackers_current",
        pl.DataFrame(
            [
                {
                    "proposal_id": "proposal-1",
                    "created_at": created_at,
                    "updated_at": created_at,
                    "ingest_ts": ingest_ts,
                }
            ]
        ),
        now=now,
    )

    assert payload["freshness_status"] == "fresh"
    assert payload["timestamp_column"] == "ingest_ts"
    assert payload["latest_timestamp"] == ingest_ts.isoformat()


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


def test_alpha_gates_page_shows_strategy_evidence_discovery(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    start = datetime(2026, 5, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "board_schema_version": "alpha_discovery_board.v1",
                    "as_of_date": "2026-05-10",
                    "strategy_candidate": "v5.sol_protect_exception",
                    "candidate_name": "v5.sol_protect_exception",
                    "symbol": "SOL-USDT",
                    "regime_state": "trend",
                    "horizon_hours": 24,
                    "sample_count": 35,
                    "complete_sample_count": 35,
                    "avg_net_bps": 12.5,
                    "median_net_bps": 12.5,
                    "p25_net_bps": -8.0,
                    "win_rate": 0.6,
                    "avg_mfe_bps": 20.0,
                    "avg_mae_bps": -6.0,
                    "cost_source_mix": '[{"cost_source":"quant_lab","count":35,"ratio":1.0}]',
                    "stability_by_day": "[]",
                    "paper_days": 0,
                    "cost_source_has_global_default": False,
                    "decision": "KEEP_SHADOW",
                    "decision_reasons": '["sol_protect_exception_requires_shadow_review"]',
                    "risk_permission": "UNKNOWN",
                    "risk_permission_status": "UNKNOWN",
                    "enforce_readiness_status": "WARN",
                    "block_reason_mix": '{"protect":35}',
                    "final_decision_mix": '{"SHADOW":35}',
                    "created_at": start,
                    "source": "test",
                }
            ]
        ),
        lake_root / "gold" / "alpha_discovery_board",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_name": "v5.sol_protect_exception",
                    "evidence_version": "strategy-evidence-v0.1",
                    "as_of_date": "2026-05-10",
                    "sample_count": 35,
                    "complete_sample_count": 35,
                    "avg_net_bps_by_horizon": '{"24h":12.5}',
                    "win_rate_by_horizon": '{"24h":0.6}',
                    "downside_p25_by_horizon": '{"24h":-8.0}',
                    "max_drawdown_proxy": 22.0,
                    "cost_sensitivity": '{"avg_cost_bps":4.0}',
                    "symbol_breakdown": '[{"symbol":"SOL-USDT"}]',
                    "regime_breakdown": '[{"regime_state":"trend"}]',
                    "decision": "KEEP_SHADOW",
                    "decision_reasons": '["candidate_is_shadow_or_protect_exception"]',
                    "start_ts": start,
                    "end_ts": start + timedelta(hours=35),
                    "created_at": start,
                    "source": "test",
                }
            ]
        ),
        lake_root / "gold" / "strategy_evidence",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "ts_utc": start,
                    "symbol": "SOL-USDT",
                    "candidate_name": "v5.sol_protect_exception",
                    "entry_condition_signal": "protect_exception",
                    "block_reason": "protect",
                    "final_score": 0.71,
                    "f1": 0.1,
                    "f2": 0.2,
                    "f3": 0.3,
                    "f4": 0.4,
                    "f5": 0.5,
                    "alpha6_score": 0.8,
                    "alpha6_side": "long",
                    "regime_state": "trend",
                    "protect_level": "SOL_PROTECT",
                    "expected_edge_bps": 18.0,
                    "required_edge_bps": 4.0,
                    "cost_source": "quant_lab",
                    "cost_bps": 4.0,
                    "label_status": "complete",
                    "net_bps_after_cost_24h": 12.5,
                    "win_24h": True,
                    "drawdown_proxy_bps_24h": 22.0,
                    "net_bps_after_cost_72h": 15.0,
                    "win_72h": True,
                    "drawdown_proxy_bps_72h": 25.0,
                    "source_dataset": "v5_router_decision",
                    "created_at": start,
                    "source": "test",
                }
            ]
        ),
        lake_root / "gold" / "strategy_evidence_sample",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand_test",
                    "run_id": "run_001",
                    "ts_utc": start,
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.sol_protect_exception",
                    "final_decision": "KEEP_SHADOW",
                    "block_reason": "protect",
                    "final_score": 0.71,
                    "alpha6_score": 0.8,
                    "alpha6_side": "long",
                    "expected_edge_bps": 18.0,
                    "required_edge_bps": 4.0,
                    "cost_bps": 4.0,
                    "cost_source": "quant_lab",
                }
            ]
        ),
        lake_root / "silver" / "v5_candidate_event",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "candidate_id": "cand_test",
                    "run_id": "run_001",
                    "ts_utc": start,
                    "symbol": "SOL-USDT",
                    "strategy_candidate": "v5.sol_protect_exception",
                    "block_reason": "protect",
                    "horizon_hours": 24,
                    "gross_bps": 16.5,
                    "net_bps_after_cost": 12.5,
                    "mfe_bps": 20.0,
                    "mae_bps": -6.0,
                    "win": True,
                    "label_status": "complete",
                    "label_reason": "ok",
                    "cost_bps": 4.0,
                    "cost_source": "quant_lab",
                }
            ]
        ),
        lake_root / "gold" / "v5_candidate_label",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-10",
                    "status": "PASS",
                    "candidate_event_rows": 1,
                    "feature_completeness": 1.0,
                    "label_completeness": 1.0,
                    "cost_source_coverage": 1.0,
                    "created_at": start,
                }
            ]
        ),
        lake_root / "gold" / "v5_candidate_quality_daily",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "date": "2026-05-10",
                    "block_reason": "protect",
                    "strategy_candidate": "v5.sol_protect_exception",
                    "symbol": "SOL-USDT",
                    "horizon_hours": 24,
                    "sample_count": 1,
                    "complete_sample_count": 1,
                    "avg_net_bps": 12.5,
                    "median_net_bps": 12.5,
                    "win_rate": 1.0,
                    "downside_p25_bps": 12.5,
                    "created_at": start,
                }
            ]
        ),
        lake_root / "gold" / "v5_candidate_outcome_summary",
    )
    fake = FakeStreamlit()

    alpha_gates.render(lake_root, fake)

    subheaders = _call_values(fake, "subheader")
    metrics = _call_values(fake, "metric")
    frames = _call_values(fake, "dataframe")
    assert "📊 候选决策面板" in subheaders
    assert "🧪 策略证据聚合" in subheaders
    assert "🧾 V5 候选质量" in subheaders
    assert "V5 候选事件" not in subheaders
    assert "V5 候选前向标签" not in subheaders
    assert ("继续影子观察", 1) in metrics
    assert any(
        "v5.sol_protect_exception" in frame.get_column("策略候选").to_list()
        for frame in frames
        if isinstance(frame, pl.DataFrame) and "策略候选" in frame.columns
    )
    assert any(
        "v5.sol_protect_exception" in frame.get_column("候选名称").to_list()
        for frame in frames
        if isinstance(frame, pl.DataFrame) and "候选名称" in frame.columns
    )


def test_alpha_gates_does_not_warn_for_empty_legacy_strategy_evidence_when_board_exists(
    tmp_path,
):
    lake_root = tmp_path / "lake"
    start = datetime(2026, 5, 10, tzinfo=UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "strategy": "v5",
                    "board_schema_version": "alpha_discovery_board.v1",
                    "as_of_date": "2026-05-10",
                    "strategy_candidate": "v5.f3_dominant_entry",
                    "candidate_name": "v5.f3_dominant_entry",
                    "symbol": "BNB-USDT",
                    "regime_state": "trend",
                    "horizon_hours": 24,
                    "sample_count": 12,
                    "complete_sample_count": 12,
                    "avg_net_bps": 8.0,
                    "median_net_bps": 8.0,
                    "p25_net_bps": 2.0,
                    "win_rate": 0.75,
                    "decision": "KEEP_SHADOW",
                    "decision_reasons": "[]",
                    "created_at": start,
                    "source": "test",
                }
            ]
        ),
        lake_root / "gold" / "alpha_discovery_board",
    )

    summary = readers.alpha_gate_summary(lake_root)

    assert "strategy_evidence 候选发现证据尚未生成" not in (summary["warnings"])


def test_web_launcher_hides_streamlit_file_navigation(monkeypatch, tmp_path):
    captured = {}

    def fake_run_streamlit_process(command, *, env):
        captured["command"] = command
        captured["env"] = env
        return 0

    monkeypatch.setattr(web_app, "_run_streamlit_process", fake_run_streamlit_process)

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
    assert command[option_index + 1] == "false"
    assert option_index < command.index("--")


def test_web_launcher_stops_streamlit_child_on_signal(monkeypatch):
    signal_handlers = {}

    class FakeProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.poll_count = 0

        def poll(self):
            self.poll_count += 1
            if self.poll_count == 1:
                signal_handlers[web_app.signal.SIGTERM](web_app.signal.SIGTERM, None)
                return None
            return -web_app.signal.SIGTERM

        def terminate(self) -> None:
            self.terminated = True

        def kill(self) -> None:
            self.killed = True

    fake_process = FakeProcess()

    def fake_popen(command, env):
        return fake_process

    def fake_signal(signum, handler):
        previous = signal_handlers.get(signum)
        signal_handlers[signum] = handler
        return previous

    monkeypatch.setattr(web_app.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(web_app.signal, "signal", fake_signal)
    monkeypatch.setattr(web_app, "sleep", lambda _seconds: None)

    result = web_app._run_streamlit_process(["streamlit"], env={})

    assert result == 0
    assert fake_process.terminated is True
    assert fake_process.killed is False


def test_dashboard_uses_flat_page_radio_not_dropdown(tmp_path):
    fake = FakeStreamlit(radio_value="专家包导出")

    web_app.render_dashboard(tmp_path / "lake", fake)

    assert any(call["label"] == "页面" for call in _call_values(fake, "radio"))
    assert not _call_values(fake, "selectbox")
    assert any("专家包导出" in str(value) for value in _call_values(fake, "title"))


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
    assert "orderbook_snapshot 已忽略无效 Parquet 文件" in warnings
    assert str(bad_path) in warnings


def test_overview_diagnostics_warn_when_lake_root_does_not_exist(tmp_path):
    missing_lake = tmp_path / "missing_lake"
    fake = FakeStreamlit()

    overview.render(missing_lake, fake)

    warnings = _call_values(fake, "warning")
    assert any("Lake 根目录不存在" in str(warning) for warning in warnings)
    assert any(str(missing_lake) in str(caption) for caption in _call_values(fake, "caption"))


def test_overview_diagnostics_suggests_commands_when_lake_has_no_parquet(tmp_path):
    lake_root = tmp_path / "lake"
    lake_root.mkdir()
    fake = FakeStreamlit()

    overview.render(lake_root, fake)

    warnings = "\n".join(str(warning) for warning in _call_values(fake, "warning"))
    command_texts = []
    for frame in _call_values(fake, "dataframe"):
        if isinstance(frame, pl.DataFrame):
            for column in frame.columns:
                if "命令" in column:
                    command_texts.extend(str(value) for value in frame.get_column(column).to_list())
    frames = "\n".join(command_texts)
    assert (
        "qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT "
        f"--lake-root {lake_root} --limit 100"
    ) in frames
    assert "--history" not in frames
    assert "qlab okx-ws-collect-universe" in frames
    assert "qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml" in frames
    assert "qlab export-daily 尚未实现或尚未运行。" in warnings


def test_empty_lake_diagnostics_exposes_suggested_commands(tmp_path):
    lake_root = tmp_path / "lake"

    diagnostics = readers.lake_diagnostics(lake_root)

    commands = "\n".join(diagnostics["suggested_commands"]["command"].to_list())
    assert "qlab okx-fetch-candles" in commands
    assert "qlab okx-ws-collect-universe" in commands
    assert "qlab sync-v5-telemetry" in commands
    assert "qlab export-daily" in commands


def test_web_readers_use_lake_file_index_before_rglob(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = _fixture_lake(tmp_path)
    dataset_path = lake_root / "gold" / "cost_bucket_daily"
    build_lake_file_index(lake_root, ["gold/cost_bucket_daily"])

    def fail_rglob(_path):
        raise AssertionError("web reader should use lake_file_index before rglob")

    monkeypatch.setattr(readers, "_parquet_file_candidates_rglob", fail_rglob)

    files, warning = readers._valid_parquet_files_with_warning(
        dataset_path,
        "cost_bucket_daily",
    )

    assert files
    assert warning is None


def test_web_readers_cache_shared_lake_file_index(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = tmp_path / "lake"
    cost_dataset = lake_root / "gold" / "cost_bucket_daily"
    evidence_dataset = lake_root / "gold" / "strategy_evidence"
    write_parquet_dataset(
        pl.DataFrame({"ts": [datetime(2026, 6, 4, tzinfo=UTC)], "cost_bps": [12.0]}),
        cost_dataset,
    )
    write_parquet_dataset(
        pl.DataFrame({"ts": [datetime(2026, 6, 4, tzinfo=UTC)], "alpha_id": ["a"]}),
        evidence_dataset,
    )
    build_lake_file_index(
        lake_root,
        ["gold/cost_bucket_daily", "gold/strategy_evidence"],
    )

    original = readers.read_parquet_dataset
    index_reads: list[Path] = []

    def counted_read(path):
        path_obj = Path(path)
        if path_obj.name == "lake_file_index" and path_obj.parent.name == "bronze":
            index_reads.append(path_obj)
        return original(path)

    def fail_rglob(_path):
        raise AssertionError("web reader should use cached lake_file_index before rglob")

    monkeypatch.setattr(readers, "read_parquet_dataset", counted_read)
    monkeypatch.setattr(readers, "_parquet_file_candidates_rglob", fail_rglob)

    cost_files, cost_warning = readers._valid_parquet_files_with_warning(
        cost_dataset,
        "cost_bucket_daily",
    )
    evidence_files, evidence_warning = readers._valid_parquet_files_with_warning(
        evidence_dataset,
        "strategy_evidence",
    )

    assert len(cost_files) == 1
    assert len(evidence_files) == 1
    assert cost_warning is None
    assert evidence_warning is None
    assert len(index_reads) == 1


def test_web_file_index_stale_check_ignores_snapshot_meta_mtime(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "strategy_opportunity_advisory"
    write_parquet_dataset(
        pl.DataFrame(
            {
                "generated_at": [datetime(2026, 6, 4, tzinfo=UTC)],
                "strategy_candidate": ["v5.f3_dominant_entry"],
            }
        ),
        dataset_path,
    )
    build_lake_file_index(lake_root, ["gold/strategy_opportunity_advisory"])

    meta_path = dataset_path / "_snapshot_meta.json"
    meta_path.write_text('{"row_count": 1}', encoding="utf-8")
    future = datetime(2026, 6, 4, 1, tzinfo=UTC).timestamp()
    os.utime(meta_path, (future, future))

    def fail_rglob(_path):
        raise AssertionError("snapshot metadata mtime should not make file index stale")

    monkeypatch.setattr(readers, "_parquet_file_candidates_rglob", fail_rglob)

    files, warning = readers._valid_parquet_files_with_warning(
        dataset_path,
        "strategy_opportunity_advisory",
    )

    assert len(files) == 1
    assert warning is None


def test_web_readers_use_bounded_heavy_fallback_when_file_index_missing(tmp_path):
    readers.clear_web_cache()
    lake_root = _fixture_lake(tmp_path)
    dataset_path = lake_root / "silver" / "trade_print"

    files, warning = readers._valid_parquet_files_with_warning(
        dataset_path,
        "trade_print",
    )

    assert files
    assert warning is None


def test_refresh_web_file_index_covers_web_reader_datasets(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = tmp_path / "lake"
    cost_dataset = lake_root / "gold" / "cost_bucket_daily"
    evidence_dataset = lake_root / "gold" / "strategy_evidence"
    write_parquet_dataset(
        pl.DataFrame({"ts": [datetime(2026, 6, 4, tzinfo=UTC)], "cost_bps": [12.0]}),
        cost_dataset,
    )
    write_parquet_dataset(
        pl.DataFrame({"ts": [datetime(2026, 6, 4, tzinfo=UTC)], "alpha_id": ["a"]}),
        evidence_dataset,
    )
    monkeypatch.setattr(
        readers,
        "DATASET_PATHS",
        {
            "cost_bucket_daily": Path("gold") / "cost_bucket_daily",
            "strategy_evidence": Path("gold") / "strategy_evidence",
        },
    )

    result = _refresh_web_file_index(lake_root)

    assert result["dataset_count"] == 2
    assert result["indexed_rows"] == 2

    def fail_rglob(_path):
        raise AssertionError("web reader should use lake_file_index after refresh")

    monkeypatch.setattr(readers, "_parquet_file_candidates_rglob", fail_rglob)
    files, warning = readers._valid_parquet_files_with_warning(
        cost_dataset,
        "cost_bucket_daily",
    )

    assert len(files) == 1
    assert warning is None


def test_dataset_snapshot_uses_snapshot_meta_without_scanning_parquet(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "gold" / "strategy_health_daily"
    dataset_path.mkdir(parents=True)
    (dataset_path / "_snapshot_meta.json").write_text(
        json.dumps(
            {
                "row_count": 7,
                "file_count": 3,
                "latest_timestamp": "2026-05-10T12:00:00Z",
                "timestamp_column": "created_at",
                "generated_at": "2026-05-10T12:01:00Z",
                "source_sha": "meta-sha",
            }
        ),
        encoding="utf-8",
    )

    def fail_scan(_files):
        raise AssertionError("snapshot meta should avoid parquet scan")

    monkeypatch.setattr(readers, "_scan_parquet_files", fail_scan)

    snapshot = readers._dataset_snapshot(lake_root, "strategy_health_daily")

    assert snapshot.rows == 7
    assert snapshot.parquet_file_count == 3
    assert snapshot.warning is None
    assert snapshot.freshness["latest_timestamp"].startswith("2026-05-10T12:00:00")


def test_web_dataset_source_signature_tracks_snapshot_meta_field_changes(tmp_path):
    readers.clear_web_cache()
    dataset_path = tmp_path / "lake" / "gold" / "strategy_health_daily"
    dataset_path.mkdir(parents=True)
    meta_path = dataset_path / "_snapshot_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_sha": "same-source",
                "row_count": 1,
                "file_count": 1,
                "latest_timestamp": "2026-05-10T12:00:00Z",
                "created_at": "2026-05-10T12:00:01Z",
            }
        ),
        encoding="utf-8",
    )
    first = readers._web_dataset_source_signature(dataset_path)
    meta_path.write_text(
        json.dumps(
            {
                "source_sha": "same-source",
                "row_count": 2,
                "file_count": 1,
                "latest_timestamp": "2026-05-10T13:00:00Z",
                "created_at": "2026-05-10T13:00:01Z",
            }
        ),
        encoding="utf-8",
    )

    second = readers._web_dataset_source_signature(dataset_path)

    assert first != second


def test_web_dataset_source_signature_tracks_snapshot_meta_dataset_dir_changes(tmp_path):
    readers.clear_web_cache()
    dataset_path = tmp_path / "lake" / "gold" / "strategy_health_daily"
    dataset_path.mkdir(parents=True)
    meta_path = dataset_path / "_snapshot_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "source_sha": "same-source",
                "row_count": 1,
                "file_count": 1,
                "latest_timestamp": "2026-05-10T12:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    os.utime(dataset_path, (1_780_000_000, 1_780_000_000))
    first = readers._web_dataset_source_signature(dataset_path)
    os.utime(dataset_path, (1_780_000_100, 1_780_000_100))

    second = readers._web_dataset_source_signature(dataset_path)

    assert first != second


def test_heavy_web_dataset_source_signature_uses_file_index_summary(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
                "price": 100.0,
            }
        ]
    ).write_parquet(dataset_path / "part-1.parquet")
    build_lake_file_index(lake_root, ["silver/trade_print"])

    def fail_indexed_files(_path):
        raise AssertionError("heavy source signature should not stat every indexed parquet file")

    monkeypatch.setattr(readers, "_indexed_files_for_web", fail_indexed_files)

    signature = readers._web_dataset_source_signature(dataset_path)

    assert signature[0] == "lake_file_index_dataset"
    assert signature[1] == 1


def test_heavy_web_dataset_source_signature_changes_when_file_index_changes(tmp_path):
    readers.clear_web_cache()
    lake_root = tmp_path / "lake"
    dataset_path = lake_root / "silver" / "trade_print"
    dataset_path.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "ts": datetime(2026, 5, 10, 1, tzinfo=UTC),
                "price": 100.0,
            }
        ]
    ).write_parquet(dataset_path / "part-1.parquet")
    build_lake_file_index(lake_root, ["silver/trade_print"])
    first = readers._web_dataset_source_signature(dataset_path)

    pl.DataFrame(
        [
            {
                "symbol": "BTC-USDT",
                "ts": datetime(2026, 5, 10, 2, tzinfo=UTC),
                "price": 101.0,
            }
        ]
    ).write_parquet(dataset_path / "part-2.parquet")
    build_lake_file_index(lake_root, ["silver/trade_print"])
    readers.clear_web_cache()
    second = readers._web_dataset_source_signature(dataset_path)

    assert first != second
    assert second[0] == "lake_file_index_dataset"
    assert second[1] == 2


def test_web_recent_dataset_cache_avoids_second_scan(tmp_path, monkeypatch):
    readers.clear_web_cache()
    lake_root = _fixture_lake(tmp_path)

    first, first_warning = readers.read_recent_dataset_with_warning(
        lake_root,
        "cost_bucket_daily",
    )
    assert first.height == 1
    assert first_warning is None

    def fail_scan(_files):
        raise AssertionError("second read should use TTL cache")

    monkeypatch.setattr(readers, "_scan_parquet_files", fail_scan)

    second, _second_warning = readers.read_recent_dataset_with_warning(
        lake_root,
        "cost_bucket_daily",
    )

    assert second.height == 1


def test_web_performance_page_renders_recent_events(tmp_path):
    web_perf.clear_events()
    web_perf.record_event(
        "dataset_read",
        dataset_name="cost_bucket_daily",
        elapsed_ms=12.3,
        cache_hit=True,
        rows_rendered=1,
    )
    fake = FakeStreamlit()

    web_performance.render(tmp_path / "lake", fake)

    assert any("Web 性能" in str(value) for value in _call_values(fake, "title"))
    assert ("cache hit", 1) in _call_values(fake, "metric")
    frames = [frame for frame in _call_values(fake, "dataframe") if isinstance(frame, pl.DataFrame)]
    assert frames
    assert frames[-1].height == 1


def test_overview_diagnostics_shows_latest_market_bar_ts(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit()

    overview.render(lake_root, fake)

    metrics = _call_values(fake, "metric")
    assert ("最新行情 K 线", "2026-05-10 10:00:00 Asia/Shanghai") in metrics


def test_overview_uses_hard_soft_cost_fallback_metrics(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit()

    overview.render(lake_root, fake)

    metrics = dict(_call_values(fake, "metric"))
    assert metrics["成本硬回退比例"] == "0.00%"
    assert metrics["成本软回退比例"] == "50.00%"
    assert "成本回退比例" not in metrics


def test_expert_exports_page_downloads_listed_packs(tmp_path):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit()

    expert_exports.render(lake_root, fake, exports_root=tmp_path / "exports")

    downloads = _call_values(fake, "download_button")
    assert downloads
    assert downloads[0]["file_name"] == "quant_lab_expert_pack_2026-05-10.zip"
    assert downloads[0]["mime"] == "application/zip"
    assert downloads[0]["data"].startswith(b"PK")


def test_expert_export_summary_prefers_export_index(tmp_path, monkeypatch):
    readers.clear_web_cache()
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    pack = exports_root / "quant_lab_expert_pack_2026-05-16.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"marker": "zip"}))
        archive.writestr("data_quality.json", "{}")
        archive.writestr("expert_questions.md", "")
    (exports_root / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(pack),
                "packs": [
                    {
                        "path": str(pack),
                        "name": pack.name,
                        "size_bytes": pack.stat().st_size,
                        "modified_at": "2026-05-16T00:00:00Z",
                    }
                ],
                "manifest_summary": {"marker": "index"},
                "data_quality_summary": {"status": "OK"},
                "expert_questions": ["indexed question"],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    def fail_zip(_pack, _member):
        raise AssertionError("expert_export_summary should not read zip when index exists")

    monkeypatch.setattr(readers, "_read_json_from_zip", fail_zip)

    summary = readers.expert_export_summary(exports_root)

    assert summary["latest_pack"] == str(pack)
    assert summary["manifest_summary"]["marker"] == "index"
    assert summary["expert_questions"] == ["indexed question"]


def test_expert_export_summary_filters_missing_index_pack_rows(tmp_path):
    readers.clear_web_cache()
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    pack = exports_root / "quant_lab_expert_pack_2026-05-16_120000.zip"
    missing = exports_root / "quant_lab_expert_pack_2026-05-16_110000.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"marker": "zip"}))
        archive.writestr("data_quality.json", "{}")
        archive.writestr("expert_questions.md", "")
    (exports_root / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(pack),
                "packs": [
                    {
                        "path": str(pack),
                        "name": pack.name,
                        "size_bytes": pack.stat().st_size,
                        "modified_at": "2026-05-16T12:00:00Z",
                    },
                    {
                        "path": str(missing),
                        "name": missing.name,
                        "size_bytes": 1,
                        "modified_at": "2026-05-16T11:00:00Z",
                    },
                ],
                "manifest_summary": {"marker": "index"},
                "data_quality_summary": {"status": "OK"},
                "expert_questions": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    summary = readers.expert_export_summary(exports_root)
    packs = summary["packs"].to_dicts()

    assert summary["latest_pack"] == str(pack.resolve())
    assert [row["name"] for row in packs] == [pack.name]
    assert packs[0]["path"] == str(pack.resolve())


def test_expert_export_summary_ignores_index_when_newer_pack_is_unindexed(tmp_path):
    readers.clear_web_cache()
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    old_pack = exports_root / "quant_lab_expert_pack_2026-06-29_20260629T020417.zip"
    new_pack = exports_root / "quant_lab_expert_pack_2026-06-28_20260629T043920.zip"
    for pack, marker in [(old_pack, "old"), (new_pack, "new")]:
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr(
                "manifest.json",
                json.dumps({"marker": marker, "authoritative_snapshot": True}),
            )
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
    old_time = datetime(2026, 6, 28, 18, tzinfo=UTC).timestamp()
    new_time = datetime(2026, 6, 28, 20, 39, tzinfo=UTC).timestamp()
    os.utime(old_pack, (old_time, old_time))
    os.utime(new_pack, (new_time, new_time))
    (exports_root / "export_index.json").write_text(
        json.dumps(
            {
                "latest_pack": str(old_pack),
                "packs": [
                    {
                        "path": str(old_pack),
                        "name": old_pack.name,
                        "size_bytes": old_pack.stat().st_size,
                        "modified_at": "2026-06-28T18:00:00Z",
                        "authoritative_snapshot": True,
                    }
                ],
                "manifest_summary": {"marker": "index-old"},
                "data_quality_summary": {"status": "OK"},
                "expert_questions": [],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )

    summary = readers.expert_export_summary(exports_root)

    assert summary["latest_pack"] == str(new_pack)
    assert summary["manifest_summary"]["marker"] == "new"


def test_expert_export_summary_cache_updates_when_old_pack_deleted(tmp_path):
    readers.clear_web_cache()
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    latest = exports_root / "quant_lab_expert_pack_2026-05-16_120000.zip"
    old = exports_root / "quant_lab_expert_pack_2026-05-16_110000.zip"
    for pack in (latest, old):
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"marker": pack.name}))
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
    latest_time = datetime(2026, 5, 16, 12, tzinfo=UTC).timestamp()
    old_time = datetime(2026, 5, 16, 11, tzinfo=UTC).timestamp()
    os.utime(latest, (latest_time, latest_time))
    os.utime(old, (old_time, old_time))

    first = readers.expert_export_summary(exports_root)
    old.unlink()
    second = readers.expert_export_summary(exports_root)

    assert [row["name"] for row in first["packs"].to_dicts()] == [latest.name, old.name]
    assert [row["name"] for row in second["packs"].to_dicts()] == [latest.name]


def test_expert_exports_generate_today_button_invokes_export(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit(button_values={"生成今日专家包": True})
    captured = {}

    def fake_export_daily_pack(**kwargs):
        captured.update(kwargs)
        pack_path = Path(kwargs["out_dir"]) / "quant_lab_expert_pack_test.zip"
        with zipfile.ZipFile(pack_path, "w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
        return SimpleNamespace(zip_path=str(pack_path), warnings=[])

    monkeypatch.setenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "true")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_BACKGROUND", "false")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_MODE", "in_process")
    monkeypatch.setattr(expert_exports, "export_daily_pack", fake_export_daily_pack)

    expert_exports.render(lake_root, fake, exports_root=tmp_path / "exports")

    assert captured["lake_root"] == lake_root
    assert captured["out_dir"] == tmp_path / "exports"
    assert captured["export_date"]
    assert captured["profile"] == "expert"
    assert captured["refresh_risk_permission"] is False
    assert captured["pre_export_v5_refresh"] is True
    assert captured["allow_stale_v5"] is False
    assert "--no-refresh-risk-permission" in captured["command_line"]
    assert "--pre-export-v5-refresh" in captured["command_line"]
    assert "--allow-stale-v5" not in captured["command_line"]
    assert any("已生成专家包" in str(value) for value in _call_values(fake, "success"))
    downloads = _call_values(fake, "download_button")
    assert any(item["file_name"] == "quant_lab_expert_pack_test.zip" for item in downloads)


def test_expert_exports_generate_today_button_starts_background_job(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    exports_root = tmp_path / "exports"
    fake = RerunStreamlit(
        button_values={"generate_today_expert_pack": True},
        session_state={},
    )
    captured = {}

    class FakeProcess:
        pid = 4242

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "true")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_BACKGROUND", "true")
    monkeypatch.setattr(expert_exports, "beijing_today", lambda: datetime(2026, 5, 10).date())
    monkeypatch.setattr(expert_exports.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(expert_exports, "_pid_is_running", lambda pid: pid == 4242)

    expert_exports.render(lake_root, fake, exports_root=exports_root)

    status = json.loads(
        (exports_root / ".quant_lab_web_export_2026-05-10.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "running"
    assert status["mode"] == "authoritative_snapshot"
    assert status["pid"] == 4242
    assert captured["command"][0]
    assert captured["command"][1] == "-c"
    assert "refresh_risk_permission=False" in captured["command"][2]
    assert "pre_export_v5_refresh=True" in captured["command"][2]
    assert "allow_stale_v5=False" in captured["command"][2]
    assert captured["kwargs"]["env"]["POLARS_MAX_THREADS"] == "1"
    assert captured["kwargs"]["env"]["MALLOC_ARENA_MAX"] == "2"
    assert fake.rerun_count == 1
    assert any("PID=4242" in str(value) for value in _call_values(fake, "info"))


def test_expert_exports_generate_today_writes_request_file_for_systemd_worker(
    tmp_path, monkeypatch
):
    lake_root = _fixture_lake(tmp_path)
    exports_root = tmp_path / "exports"
    fake = RerunStreamlit(
        button_values={"generate_today_expert_pack": True},
        session_state={},
    )

    monkeypatch.setenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "true")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_BACKGROUND", "true")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_BACKGROUND_TRIGGER", "request_file")
    monkeypatch.setattr(expert_exports, "beijing_today", lambda: datetime(2026, 5, 16).date())

    expert_exports.render(lake_root, fake, exports_root=exports_root)

    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    request_path = exports_root / ".quant_lab_web_export_request.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    request = json.loads(request_path.read_text(encoding="utf-8"))
    assert status["state"] == "running"
    assert status["trigger"] == "request_file"
    assert status["systemd_unit"] == "quant-lab-web-export-request.service"
    assert status["request_id"]
    assert request["request_id"] == status["request_id"]
    assert request["export_date"] == "2026-05-16"
    assert request["lake_root"] == str(lake_root)
    assert request["exports_root"] == str(exports_root)
    assert request["status_path"] == str(status_path)
    assert fake.rerun_count == 1
    assert any(
        "quant-lab-web-export-request.service" in str(value)
        for value in _call_values(fake, "info")
    )


def test_expert_exports_generate_today_reuses_recent_success(tmp_path, monkeypatch):
    lake_root = _fixture_lake(tmp_path)
    exports_root = tmp_path / "exports"
    exports_root.mkdir(exist_ok=True)
    export_date = "2026-05-16"
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_recent.zip"
    pack_path.write_bytes(b"PK")
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "succeeded",
                "export_date": export_date,
                "request_id": "first-request",
                "zip_path": str(pack_path),
                "finished_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    def fail_popen(*args, **kwargs):  # pragma: no cover - only used if dedupe breaks
        raise AssertionError("recent successful export should be reused")

    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_REGENERATE_COOLDOWN_SECONDS", "180")
    monkeypatch.setattr(expert_exports.subprocess, "Popen", fail_popen)

    status = expert_exports._start_export_job(
        export_date=export_date,
        lake_root=lake_root,
        exports_root=exports_root,
    )

    assert status["state"] == "succeeded"
    assert status["zip_path"] == str(pack_path)
    assert status["regenerate_skipped"] is True
    assert status["regenerate_cooldown_remaining_seconds"] > 0
    assert status["regenerate_reuse_pack_name"] == pack_path.name
    assert not (exports_root / ".quant_lab_web_export_request.json").exists()


def test_web_export_request_worker_writes_completion_status(tmp_path, monkeypatch):
    exports_root = tmp_path / "exports"
    lake_root = tmp_path / "lake"
    exports_root.mkdir()
    lake_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    request_path = exports_root / ".quant_lab_web_export_request.json"
    request_path.write_text(
        json.dumps(
            {
                "export_date": "2026-05-16",
                "lake_root": str(lake_root),
                "exports_root": str(exports_root),
                "status_path": str(status_path),
                "request_id": "request-20260516",
                "requested_at": "2026-05-16T01:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_export_daily_pack(**kwargs):
        captured.update(kwargs)
        pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_worker.zip"
        pack_path.write_bytes(b"zip")
        return SimpleNamespace(zip_path=str(pack_path), warnings=["sample_warning"])

    monkeypatch.setattr(export_request, "export_daily_pack", fake_export_daily_pack)

    status = export_request.run_web_export_request(request_path)

    stored = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["state"] == "succeeded"
    assert stored["state"] == "succeeded"
    assert stored["trigger"] == "request_file"
    assert stored["request_id"] == "request-20260516"
    assert isinstance(stored["pid"], int)
    assert stored["worker_pid"] == stored["pid"]
    assert stored["zip_path"].endswith("quant_lab_expert_pack_2026-05-16_worker.zip")
    assert stored["warnings"] == ["sample_warning"]
    assert captured["pre_export_v5_refresh"] is True
    assert captured["allow_stale_v5"] is False
    assert "--pre-export-v5-refresh" in captured["command_line"]
    assert "--allow-stale-v5" not in captured["command_line"]
    assert not request_path.exists()


def test_expert_exports_background_job_truncates_stale_log(tmp_path, monkeypatch):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    log_path = exports_root / ".quant_lab_web_export_2026-05-10.log"
    log_path.write_text("old memory allocation error", encoding="utf-8")

    class FakeProcess:
        pid = 4242

    monkeypatch.setattr(expert_exports.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(expert_exports, "_pid_is_running", lambda _pid: False)

    status = expert_exports._start_export_job(
        export_date="2026-05-10",
        lake_root=tmp_path / "lake",
        exports_root=exports_root,
    )

    assert status["state"] == "running"
    assert status["pid"] == 4242
    assert log_path.read_text(encoding="utf-8") == ""


def test_expert_exports_generate_today_uses_beijing_date_and_creates_export_dir(
    tmp_path,
    monkeypatch,
):
    lake_root = tmp_path / "lake"
    exports_root = tmp_path / "missing_exports"
    fake = FakeStreamlit()
    captured = {}

    def fake_export_daily_pack(**kwargs):
        captured.update(kwargs)
        pack_path = Path(kwargs["out_dir"]) / "quant_lab_expert_pack_test.zip"
        with zipfile.ZipFile(pack_path, "w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
        return SimpleNamespace(zip_path=str(pack_path), warnings=[])

    monkeypatch.setattr(expert_exports, "beijing_today", lambda: datetime(2026, 5, 16).date())
    monkeypatch.setenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "true")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_BACKGROUND", "false")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_MODE", "in_process")
    monkeypatch.setattr(expert_exports, "export_daily_pack", fake_export_daily_pack)

    pack_path = expert_exports._generate_today_pack(
        fake,
        lake_root=lake_root,
        exports_root=exports_root,
    )

    assert exports_root.exists()
    assert pack_path == exports_root / "quant_lab_expert_pack_test.zip"
    assert captured["export_date"] == "2026-05-16"
    assert captured["refresh_risk_permission"] is False
    assert captured["pre_export_v5_refresh"] is True
    assert captured["allow_stale_v5"] is False


def test_expert_exports_generated_pack_is_listed_first(tmp_path):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    old_pack = exports_root / "quant_lab_expert_pack_2026-05-16_old.zip"
    generated = exports_root / "quant_lab_expert_pack_2026-05-16_generated.zip"
    for pack, marker in [(old_pack, "old"), (generated, "new")]:
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"marker": marker}))
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
    os.utime(generated, (datetime(2026, 5, 16, 1, tzinfo=UTC).timestamp(),) * 2)
    os.utime(old_pack, (datetime(2026, 5, 16, 2, tzinfo=UTC).timestamp(),) * 2)

    summary = expert_exports._summary_preferring_generated_pack(exports_root, generated)
    packs = summary["packs"].to_dicts()

    assert summary["latest_pack"] == str(generated)
    assert summary["manifest_summary"]["marker"] == "new"
    assert packs[0]["path"] == str(generated)


def test_expert_exports_generated_pack_is_first_download_even_with_older_mtime(tmp_path):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    old_pack = exports_root / "quant_lab_expert_pack_2026-05-16_old.zip"
    generated = exports_root / "quant_lab_expert_pack_2026-05-16_generated.zip"
    for pack, marker in [(old_pack, "old"), (generated, "new")]:
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"marker": marker}))
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
    os.utime(generated, (datetime(2026, 5, 16, 1, tzinfo=UTC).timestamp(),) * 2)
    os.utime(old_pack, (datetime(2026, 5, 16, 2, tzinfo=UTC).timestamp(),) * 2)

    summary = expert_exports._summary_preferring_generated_pack(exports_root, generated)
    fake = FakeStreamlit()
    expert_exports._render_pack_downloads(fake, summary["packs"])

    downloads = _call_values(fake, "download_button")
    assert downloads[0]["file_name"] == generated.name


def test_expert_exports_download_buttons_are_bounded_to_recent_packs(tmp_path, monkeypatch):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    rows = []
    for index in range(7):
        pack = exports_root / f"quant_lab_expert_pack_2026-05-16_{index}.zip"
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", "{}")
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
        rows.append({"path": str(pack), "modified_at": f"2026-05-16T00:0{index}:00Z"})
    monkeypatch.setenv("QUANT_LAB_WEB_DOWNLOAD_MAX_PACKS", "3")
    expert_exports._download_pack_bytes.cache_clear()
    fake = FakeStreamlit()

    expert_exports._render_pack_downloads(fake, pl.DataFrame(rows))

    downloads = _call_values(fake, "download_button")
    assert [item["file_name"] for item in downloads] == [
        "quant_lab_expert_pack_2026-05-16_0.zip",
        "quant_lab_expert_pack_2026-05-16_1.zip",
        "quant_lab_expert_pack_2026-05-16_2.zip",
    ]
    assert all(item["data"].startswith(b"PK") for item in downloads)
    captions = _call_values(fake, "caption")
    assert any("直接下载按钮仅加载最近 3 个" in str(value) for value in captions)


def test_expert_exports_generate_today_persists_pack_across_streamlit_rerun(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    exports_root = tmp_path / "exports"
    session_state = {}
    first_fake = RerunStreamlit(
        button_values={"生成今日专家包": True},
        session_state=session_state,
    )

    def fake_export_daily_pack(**kwargs):
        pack_path = Path(kwargs["out_dir"]) / "quant_lab_expert_pack_test.zip"
        with zipfile.ZipFile(pack_path, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"fresh": True}))
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
        return SimpleNamespace(zip_path=str(pack_path), warnings=[])

    monkeypatch.setenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "true")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_BACKGROUND", "false")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_MODE", "in_process")
    monkeypatch.setattr(expert_exports, "export_daily_pack", fake_export_daily_pack)

    expert_exports.render(lake_root, first_fake, exports_root=exports_root)

    assert first_fake.rerun_count == 1
    assert session_state["expert_exports_generated_pack"].endswith("quant_lab_expert_pack_test.zip")

    second_fake = RerunStreamlit(session_state=session_state)
    expert_exports.render(lake_root, second_fake, exports_root=exports_root)

    assert "expert_exports_generated_pack" not in session_state
    assert any("已生成专家包" in str(value) for value in _call_values(second_fake, "success"))
    downloads = _call_values(second_fake, "download_button")
    assert downloads[0]["file_name"] == "quant_lab_expert_pack_test.zip"


def test_expert_exports_generate_today_starts_snapshot_even_when_today_pack_exists(
    tmp_path,
    monkeypatch,
):
    lake_root = _fixture_lake(tmp_path)
    exports_root = tmp_path / "exports"
    exports_root.mkdir(exist_ok=True)
    pack = exports_root / "quant_lab_expert_pack_2026-05-16_existing.zip"
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"fresh": True}))
        archive.writestr("data_quality.json", "{}")
        archive.writestr("expert_questions.md", "")
    fake = RerunStreamlit(
        button_values={"generate_today_expert_pack": True},
        session_state={},
    )
    captured = {}

    class FakeProcess:
        pid = 5252

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.delenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", raising=False)
    monkeypatch.delenv("QUANT_LAB_WEB_EXPORT_BACKGROUND", raising=False)
    monkeypatch.setattr(expert_exports.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(expert_exports, "_pid_is_running", lambda pid: pid == 5252)
    monkeypatch.setattr(expert_exports, "beijing_today", lambda: datetime(2026, 5, 16).date())

    expert_exports.render(lake_root, fake, exports_root=exports_root)

    assert fake.rerun_count == 1
    assert "expert_exports_generated_pack" not in fake.session_state
    assert captured["command"][1] == "-c"
    status = json.loads(
        (exports_root / ".quant_lab_web_export_2026-05-16.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "running"
    assert status["pid"] == 5252


def test_expert_exports_generate_today_missing_pack_reuses_only_when_disabled(
    tmp_path, monkeypatch
):
    lake_root = _fixture_lake(tmp_path)
    fake = FakeStreamlit(button_values={"生成今日专家包": True})

    def fail_export_daily_pack(**_kwargs):
        raise AssertionError("web should not run on-demand export by default")

    monkeypatch.setenv("QUANT_LAB_WEB_ON_DEMAND_EXPORT", "false")
    monkeypatch.setattr(expert_exports, "beijing_today", lambda: datetime(2026, 5, 16).date())
    monkeypatch.setattr(expert_exports, "export_daily_pack", fail_export_daily_pack)

    expert_exports.render(lake_root, fake, exports_root=tmp_path / "exports")

    warnings = _call_values(fake, "warning")
    assert any("Web 不直接运行重计算" in str(value) for value in warnings)


def test_expert_exports_stale_running_job_is_marked_failed(tmp_path, monkeypatch):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "pid": 4242,
                "started_at": "2026-05-16T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "1")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "failed"
    assert "exceeded web status timeout" in status["error"]


def test_expert_exports_running_job_recovers_when_process_exited_after_pack(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "pid": 4242,
                "started_at": "2026-05-16T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
    pack_path.write_bytes(b"zip")
    new_mtime = datetime(2026, 5, 16, 1, 0, tzinfo=UTC).timestamp()
    os.utime(pack_path, (new_mtime, new_mtime))
    monkeypatch.setattr(expert_exports, "_pid_is_running", lambda pid: False)
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "999999999")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "succeeded"
    assert status["zip_path"] == str(pack_path)
    assert status["recovered_from_failed_status"] is True
    assert status["recovery_reason"] == "pack_created_after_export_job"
    assert status["recovered_pack_mtime"] == datetime.fromtimestamp(new_mtime, UTC).isoformat()


def test_expert_exports_running_job_does_not_recover_while_worker_is_alive(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "worker_pid": 4242,
                "started_at": "2026-05-16T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
    pack_path.write_bytes(b"partial-zip")
    new_mtime = datetime(2026, 5, 16, 1, 0, tzinfo=UTC).timestamp()
    os.utime(pack_path, (new_mtime, new_mtime))
    monkeypatch.setattr(expert_exports, "_pid_is_running", lambda pid: True)

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "running"
    assert "zip_path" not in status
    assert "recovered_from_failed_status" not in status


def test_expert_exports_running_request_file_job_stays_running_without_pid(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    request_path = exports_root / ".quant_lab_web_export_request.json"
    request_path.write_text("{}", encoding="utf-8")
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "request_file",
                "request_path": str(request_path),
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "999999999")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "running"
    assert "error" not in status


def test_expert_exports_request_file_job_tolerates_slow_systemd_prestart(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    request_path = exports_root / ".quant_lab_web_export_request.json"
    request_path.write_text("{}", encoding="utf-8")
    slow_prestart_time = (datetime.now(UTC) - timedelta(minutes=6)).timestamp()
    os.utime(request_path, (slow_prestart_time, slow_prestart_time))
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "request_file",
                "request_path": str(request_path),
                "started_at": datetime.now(UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "999999999")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "running"
    assert "did not pick up web export request" not in str(status.get("error") or "")


def test_expert_exports_request_file_job_uses_pickup_timeout_before_generic_stale(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    request_path = exports_root / ".quant_lab_web_export_request.json"
    request_path.write_text("{}", encoding="utf-8")
    old_time = (datetime.now(UTC) - timedelta(minutes=20)).timestamp()
    os.utime(request_path, (old_time, old_time))
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "request_file",
                "request_path": str(request_path),
                "started_at": datetime.fromtimestamp(old_time, UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "1")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_REQUEST_PICKUP_STALE_SECONDS", "5400")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "running"
    assert "exceeded web status timeout" not in str(status.get("error") or "")


def test_expert_exports_request_file_job_fails_when_worker_never_picks_it_up(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    request_path = exports_root / ".quant_lab_web_export_request.json"
    request_path.write_text("{}", encoding="utf-8")
    old_time = datetime(2026, 5, 16, 1, 0, tzinfo=UTC).timestamp()
    os.utime(request_path, (old_time, old_time))
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "request_file",
                "request_path": str(request_path),
                "started_at": "2026-05-16T01:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "999999999")
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_REQUEST_PICKUP_STALE_SECONDS", "1")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "failed"
    assert "did not pick up web export request" in status["error"]


def test_expert_exports_linux_zombie_pid_is_not_running(monkeypatch):
    monkeypatch.setattr(expert_exports.os, "name", "posix")
    monkeypatch.setattr(expert_exports, "_linux_proc_state", lambda pid: "Z")

    assert expert_exports._pid_is_running(4242) is False


def test_expert_exports_failed_status_recovers_when_new_pack_exists(tmp_path):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "MemoryError",
                "finished_at": "2026-05-16T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
    pack_path.write_bytes(b"zip")
    new_mtime = datetime(2026, 5, 16, 2, 0, tzinfo=UTC).timestamp()
    os.utime(pack_path, (new_mtime, new_mtime))

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "succeeded"
    assert status["zip_path"] == str(pack_path)
    assert status["recovered_from_failed_status"] is True
    assert status["recovery_reason"] == "pack_created_after_export_job"
    assert "error" not in status


def test_expert_exports_failed_status_does_not_recover_old_pack(tmp_path):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "MemoryError",
                "started_at": "2026-05-16T03:00:00+00:00",
                "finished_at": "2026-05-16T03:05:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
    pack_path.write_bytes(b"zip")
    old_mtime = datetime(2026, 5, 16, 2, 0, tzinfo=UTC).timestamp()
    os.utime(pack_path, (old_mtime, old_mtime))

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "failed"
    assert status["error"] == "MemoryError"
    assert "zip_path" not in status


def test_expert_exports_worker_trigger_failure_does_not_recover_old_same_day_pack(
    tmp_path,
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "export process exited before writing completion status",
                "started_at": "2026-05-16T03:00:00+00:00",
                "finished_at": "2026-05-16T03:01:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
    pack_path.write_bytes(b"zip")
    old_mtime = datetime(2026, 5, 16, 2, 0, tzinfo=UTC).timestamp()
    os.utime(pack_path, (old_mtime, old_mtime))

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "failed"
    assert status["error"] == "export process exited before writing completion status"
    assert "zip_path" not in status
    assert "recovered_from_failed_status" not in status


def test_expert_exports_stale_request_file_job_does_not_recover_old_pack(
    tmp_path, monkeypatch
):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "trigger": "request_file",
                "started_at": "2026-05-16T03:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
    pack_path.write_bytes(b"zip")
    old_mtime = datetime(2026, 5, 16, 2, 0, tzinfo=UTC).timestamp()
    os.utime(pack_path, (old_mtime, old_mtime))
    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_STATUS_STALE_SECONDS", "1")

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "failed"
    assert "exceeded web status timeout" in status["error"]
    assert "zip_path" not in status
    assert "recovered_from_failed_status" not in status


def test_expert_exports_failed_status_recovers_pack_between_start_and_failure(tmp_path):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "failed",
                "error": "export process exited before writing completion status",
                "started_at": "2026-05-16T03:00:00+00:00",
                "finished_at": "2026-05-16T03:05:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pack_path = exports_root / "quant_lab_expert_pack_2026-05-16_20260516T110300+0800.zip"
    pack_path.write_bytes(b"zip")
    pack_mtime = datetime(2026, 5, 16, 3, 3, tzinfo=UTC).timestamp()
    os.utime(pack_path, (pack_mtime, pack_mtime))

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "succeeded"
    assert status["zip_path"] == str(pack_path)
    assert status["recovered_from_failed_status"] is True
    assert "error" not in status


def test_expert_exports_succeeded_recovered_status_normalizes_inverted_times(tmp_path):
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    status_path = exports_root / ".quant_lab_web_export_2026-05-16.json"
    status_path.write_text(
        json.dumps(
            {
                "state": "succeeded",
                "zip_path": str(
                    exports_root
                    / "quant_lab_expert_pack_2026-05-16_20260516T100000+0800.zip"
                ),
                "started_at": "2026-05-16T04:00:00+00:00",
                "finished_at": "2026-05-16T02:00:00+00:00",
                "recovered_from_failed_status": True,
            }
        ),
        encoding="utf-8",
    )

    status = expert_exports._poll_export_job(exports_root, "2026-05-16")

    assert status["state"] == "succeeded"
    assert status["started_at"] == "2026-05-16T02:00:00+00:00"
    assert status["finished_at"] == "2026-05-16T02:00:00+00:00"
    assert status["timestamp_normalized"] is True
    stored = json.loads(status_path.read_text(encoding="utf-8"))
    assert stored["started_at"] == "2026-05-16T02:00:00+00:00"


def test_expert_exports_subprocess_mode_parses_generated_pack(tmp_path, monkeypatch):
    pack_path = tmp_path / "exports" / "quant_lab_expert_pack_subprocess.zip"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"zip_path": str(pack_path), "warnings": ["sample_warning"]}),
            stderr="",
        )

    monkeypatch.setenv("QUANT_LAB_WEB_EXPORT_MODE", "subprocess")
    monkeypatch.setattr(expert_exports.subprocess, "run", fake_run)

    result_path, warnings = expert_exports._export_daily_from_web(
        export_date="2026-05-17",
        lake_root=tmp_path / "lake",
        exports_root=tmp_path / "exports",
    )

    assert result_path == pack_path
    assert warnings == ["sample_warning"]
    assert captured["command"][0]
    assert captured["command"][1] == "-c"
    assert "refresh_risk_permission=False" in captured["command"][2]
    assert "pre_export_v5_refresh=True" in captured["command"][2]
    assert "allow_stale_v5=False" in captured["command"][2]
    assert "str(result.zip_path)" in captured["command"][2]
    assert captured["kwargs"]["capture_output"] is True


def test_expert_exports_background_script_serializes_zip_path_as_string():
    script = expert_exports._export_job_script()

    assert '"zip_path": str(result.zip_path)' in script
    assert '"zip_path": result.zip_path' not in script


def test_expert_exports_summary_uses_mtime_for_latest_pack(tmp_path):
    readers.clear_web_cache()
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    old_pack = exports_root / "quant_lab_expert_pack_2026-05-15_999999.zip"
    new_pack = exports_root / "quant_lab_expert_pack_2026-05-15_000001.zip"
    for pack, marker in [(old_pack, "old"), (new_pack, "new")]:
        with zipfile.ZipFile(pack, "w") as archive:
            archive.writestr("manifest.json", json.dumps({"marker": marker}))
            archive.writestr("data_quality.json", "{}")
            archive.writestr("expert_questions.md", "")
    old_time = datetime(2026, 5, 15, 1, tzinfo=UTC).timestamp()
    new_time = datetime(2026, 5, 15, 2, tzinfo=UTC).timestamp()
    old_pack.touch()
    new_pack.touch()
    os.utime(old_pack, (old_time, old_time))
    os.utime(new_pack, (new_time, new_time))

    summary = readers.expert_export_summary(exports_root)

    assert summary["latest_pack"] == str(new_pack)
    assert summary["manifest_summary"]["marker"] == "new"


def test_expert_exports_summary_prefers_authoritative_pack_over_newer_debug_pack(tmp_path):
    readers.clear_web_cache()
    exports_root = tmp_path / "exports"
    exports_root.mkdir()
    authoritative_pack = exports_root / "quant_lab_expert_pack_2026-05-15_100000.zip"
    debug_pack = exports_root / "quant_lab_expert_pack_2026-05-15_110000.zip"
    with zipfile.ZipFile(authoritative_pack, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps({"marker": "authoritative", "authoritative_snapshot": True}),
        )
        archive.writestr("data_quality.json", "{}")
        archive.writestr("expert_questions.md", "")
    with zipfile.ZipFile(debug_pack, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "marker": "debug",
                    "authoritative_snapshot": False,
                    "selected_v5_bundle_authoritative_reason": (
                        "pre_export_v5_refresh_disabled"
                    ),
                }
            ),
        )
        archive.writestr("data_quality.json", "{}")
        archive.writestr("expert_questions.md", "")
    old_time = datetime(2026, 5, 15, 1, tzinfo=UTC).timestamp()
    new_time = datetime(2026, 5, 15, 2, tzinfo=UTC).timestamp()
    os.utime(authoritative_pack, (old_time, old_time))
    os.utime(debug_pack, (new_time, new_time))

    summary = readers.expert_export_summary(exports_root)
    latest_for_date = expert_exports._latest_pack_for_export_date(
        exports_root,
        "2026-05-15",
    )

    assert summary["latest_pack"] == str(authoritative_pack)
    assert summary["manifest_summary"]["marker"] == "authoritative"
    assert latest_for_date == authoritative_pack


def _call_values(fake: "FakeStreamlit", name: str) -> list[object]:
    return [value for call_name, value in fake.calls if call_name == name]


class FakeStreamlit:
    def __init__(
        self,
        button_values: dict[str, bool] | None = None,
        radio_value: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.sidebar = self
        self.button_values = button_values or {}
        self.radio_value = radio_value

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

    def button(self, label, **kwargs):
        self.calls.append(("button", {"label": label, **kwargs}))
        return self.button_values.get(label, self.button_values.get(kwargs.get("key"), False))

    def download_button(self, **kwargs) -> None:
        self.calls.append(("download_button", kwargs))

    def success(self, value) -> None:
        self.calls.append(("success", value))

    def error(self, value) -> None:
        self.calls.append(("error", value))

    def spinner(self, value):
        self.calls.append(("spinner", value))
        return _FakeSpinner()

    def selectbox(self, _label, options):
        self.calls.append(("selectbox", {"label": _label, "options": list(options)}))
        return list(options)[0]

    def radio(self, label, options):
        values = list(options)
        self.calls.append(("radio", {"label": label, "options": values}))
        if self.radio_value is not None:
            return self.radio_value
        return values[0]

    def text_input(self, _label, value):
        return value


class RerunStreamlit(FakeStreamlit):
    def __init__(
        self,
        button_values: dict[str, bool] | None = None,
        session_state: dict[str, str] | None = None,
    ) -> None:
        super().__init__(button_values=button_values)
        self.session_state = session_state if session_state is not None else {}
        self.rerun_count = 0

    def rerun(self) -> None:
        self.rerun_count += 1
        self.calls.append(("rerun", None))


class _FakeSpinner:
    def __enter__(self) -> None:
        return None

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        return False


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
                    "day": "2026-05-10",
                    "status": "WARNING",
                    "actual_rows": 1,
                    "mixed_rows": 0,
                    "proxy_rows": 1,
                    "global_default_rows": 0,
                    "hard_fallback_count": 0,
                    "hard_fallback_ratio": 0.0,
                    "soft_fallback_count": 1,
                    "soft_fallback_ratio": 0.5,
                    "proxy_only_count": 1,
                    "api_global_default_count": 0,
                    "api_symbol_proxy_hit_count": 1,
                    "api_regime_fallback_count": 0,
                    "api_degraded_cost_count": 1,
                    "symbols_with_actual_cost": "BTC-USDT",
                    "symbols_with_mixed_cost": "",
                    "symbols_with_proxy_only": "SOL-USDT",
                    "warnings_json": '["soft_fallback_ratio_high"]',
                    "created_at": start,
                }
            ]
        ),
        lake_root / "gold" / "cost_health_daily",
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
                _permission_row("v5", "ALLOW", ["paper"]),
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
