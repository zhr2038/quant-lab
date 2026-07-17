from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from csv import DictReader
from datetime import UTC, date, datetime
from functools import lru_cache
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.market_bar_time import (
    DEFAULT_MARKET_BAR_TIMEFRAME,
    market_bar_close_ts,
    market_bar_freshness_seconds,
)
from quant_lab.factors.composite_factory import build_factor_factory_v2_reports
from quant_lab.ops.api_metrics import api_metrics_summary
from quant_lab.symbols import normalize_symbol
from quant_lab.time_display import beijing_today
from quant_lab.web import perf, readers

# Keep the backend cache below the frontend polling interval. If this drifts
# higher, the page can claim it refreshed while still rendering an old snapshot.
SNAPSHOT_CACHE_TTL_SECONDS = 12.0
SNAPSHOT_CACHE_STALE_GRACE_SECONDS = 0.0
MARKET_BAR_WARNING_DELAY_SECONDS = 2 * 60 * 60
MARKET_BAR_CRITICAL_DELAY_SECONDS = 3 * 60 * 60
EXPERT_PACK_V5_LAG_WARNING_SECONDS = 60 * 60
WEB_V2_SMOKE_STATUS_PATH = Path("/var/lib/quant-lab/ops/web_v2_smoke/latest.json")
WEB_V2_SMOKE_MAX_AGE_SECONDS = 25 * 60
WEB_V2_API_METRICS_WINDOW_MINUTES = 60
V5_BUNDLE_NAME_RE = re.compile(r"v5_live_followup_bundle_(\d{8}T\d{6})Z\.tar\.gz")
ACTIONABLE_DATA_MATRIX_COLUMNS = {"market_bar", "spread", "trade"}
_SNAPSHOT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_SNAPSHOT_CACHE_SOURCE_SIGNATURES: dict[str, tuple[Any, ...]] = {}
_SNAPSHOT_CACHE_LOCK = threading.RLock()
_SNAPSHOT_REFRESHING: set[str] = set()


def bigscreen_snapshot(lake_root: str | Path) -> dict[str, Any]:
    payload, _meta = bigscreen_snapshot_with_meta(lake_root)
    return payload


def bigscreen_snapshot_with_meta(lake_root: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    start = time.perf_counter()
    root = Path(lake_root)
    cache_key = str(root.resolve())
    source_signature = _snapshot_source_signature(root)
    now = time.monotonic()
    ttl_seconds = _snapshot_cache_ttl_seconds()
    stale_grace_seconds = _snapshot_cache_stale_grace_seconds()
    with _SNAPSHOT_CACHE_LOCK:
        cached = _SNAPSHOT_CACHE.get(cache_key)
        cached_source_signature = _SNAPSHOT_CACHE_SOURCE_SIGNATURES.get(cache_key)
    if cached_source_signature != source_signature:
        cached = None
    if cached is not None:
        age_seconds = max(0.0, now - cached[0])
        if age_seconds <= ttl_seconds:
            perf.record_event(
                "bigscreen_snapshot",
                elapsed_ms=(time.perf_counter() - start) * 1000,
                cache_hit=True,
            )
            return cached[1], {
                "cache_hit": True,
                "cache_age_seconds": age_seconds,
                "cache_ttl_seconds": ttl_seconds,
                "cache_stale": False,
                "cache_stale_grace_seconds": stale_grace_seconds,
            }
        if age_seconds <= ttl_seconds + stale_grace_seconds:
            refresh_scheduled = _schedule_snapshot_refresh(cache_key, root)
            perf.record_event(
                "bigscreen_snapshot",
                elapsed_ms=(time.perf_counter() - start) * 1000,
                cache_hit=True,
            )
            return cached[1], {
                "cache_hit": True,
                "cache_age_seconds": age_seconds,
                "cache_ttl_seconds": ttl_seconds,
                "cache_stale": True,
                "cache_stale_grace_seconds": stale_grace_seconds,
                "cache_refresh_scheduled": refresh_scheduled,
            }

    payload = _build_bigscreen_snapshot_payload(root)
    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE[cache_key] = (time.monotonic(), payload)
        _SNAPSHOT_CACHE_SOURCE_SIGNATURES[cache_key] = source_signature
    perf.record_event(
        "bigscreen_snapshot",
        elapsed_ms=(time.perf_counter() - start) * 1000,
        cache_miss=True,
    )
    return payload, {
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "cache_ttl_seconds": ttl_seconds,
        "cache_stale": False,
        "cache_stale_grace_seconds": stale_grace_seconds,
    }


def _build_bigscreen_snapshot_payload(root: Path) -> dict[str, Any]:
    generated_at = datetime.now(UTC)
    data_health = _safe_summary("data_health_summary", readers.data_health_summary, root)
    cost = _safe_summary("cost_model_summary", readers.cost_model_summary, root)
    strategy = _safe_strategy_summary(root)
    ai_research = _safe_ai_research_summary(root, generated_at=generated_at)
    market = _safe_summary("market_regime_summary", readers.market_regime_summary, root)
    collectors = _safe_summary("okx_collector_summary", readers.okx_collector_summary, root)
    v5 = _safe_summary("v5_telemetry_summary", readers.v5_telemetry_summary, root)
    current_readiness = _safe_summary(
        "current_enforce_readiness_summary",
        readers.current_enforce_readiness_summary,
        root,
    )
    consumers = _safe_summary(
        "strategy_consumer_summary",
        readers.strategy_consumer_summary,
        root,
    )
    export_history = _safe_summary(
        "expert_export_summary",
        readers.expert_export_summary,
        readers.default_exports_root(root),
    )
    exports = _web_v2_export_summary_from_history(root, export_history, generated_at)
    web_events = perf.recent_events(limit=50)
    api_metrics = _safe_api_metrics(root)
    web_smoke = _web_v2_smoke_status(generated_at)
    overview = _overview_from_summaries(data_health, v5, consumers)
    data_matrix = _data_matrix(market, collectors, cost, strategy, data_health, overview)

    raw_warnings = _dedupe(
        [
            *_warnings(data_health),
            *_warnings(cost),
            *_warnings(strategy),
            *_warnings(market),
            *_warnings(collectors),
            *_warnings(v5),
            *_warnings(consumers),
            *_warnings(exports),
            *_export_quality_warnings(exports),
            *_data_matrix_warnings(data_matrix),
        ]
    )
    warnings = _system_warnings(raw_warnings, exports, data_health, generated_at)
    advisories = [warning for warning in raw_warnings if warning not in warnings]
    status = _status_from_inputs(
        overview,
        data_health,
        cost,
        v5,
        warnings,
        exports,
        generated_at=generated_at,
    )
    overview["status"] = status
    health_score = _health_score(status, data_health, cost, v5, web_events, exports)
    legacy_anomalies = _legacy_web_anomalies(data_health)
    server_resources = _server_resources(root)
    v5_payload = _v5_payload(v5, current_readiness)
    exports_payload = _exports_payload(exports)
    exports_payload.update(_expert_pack_v5_lag_status_payload(exports, v5))

    payload = {
        "generated_at": _json_value(generated_at),
        "lake_root": str(root),
        "mode": "read-only",
        "status": status,
        "health_score": health_score,
        "kpis": _kpis(generated_at, overview, collectors, cost, v5, web_events, api_metrics),
        "actions": _build_actions(
            overview,
            data_health,
            cost,
            v5,
            web_events,
            exports,
            legacy_anomalies,
            data_matrix,
            web_smoke,
        )[:8],
        "data_matrix": data_matrix,
        "strategy_flow": _strategy_flow(strategy),
        "ai_research": ai_research,
        "v5": v5_payload,
        "cost": _cost_payload(cost),
        "market": _market_payload(market),
        "collectors": _collector_payload(collectors),
        "data_health": _data_health_payload(data_health, data_matrix),
        "legacy_anomalies": legacy_anomalies,
        "server_resources": server_resources,
        "web_perf": _web_perf_payload(web_events, api_metrics, web_smoke),
        "consumers": _consumer_payload(consumers),
        "exports": exports_payload,
        "warnings": warnings[:30],
        "advisories": advisories[:30],
    }
    return payload


def _schedule_snapshot_refresh(cache_key: str, root: Path) -> bool:
    if not _snapshot_async_refresh_enabled():
        return False
    with _SNAPSHOT_CACHE_LOCK:
        if cache_key in _SNAPSHOT_REFRESHING:
            return False
        _SNAPSHOT_REFRESHING.add(cache_key)
    thread = threading.Thread(
        target=_refresh_snapshot_cache,
        args=(cache_key, root),
        name="quant-lab-bigscreen-cache-refresh",
        daemon=True,
    )
    thread.start()
    return True


def _refresh_snapshot_cache(cache_key: str, root: Path) -> None:
    start = time.perf_counter()
    try:
        source_signature = _snapshot_source_signature(root)
        payload = _build_bigscreen_snapshot_payload(root)
        with _SNAPSHOT_CACHE_LOCK:
            _SNAPSHOT_CACHE[cache_key] = (time.monotonic(), payload)
            _SNAPSHOT_CACHE_SOURCE_SIGNATURES[cache_key] = source_signature
        perf.record_event(
            "bigscreen_snapshot_refresh",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            cache_miss=True,
        )
    except Exception as exc:
        perf.record_event(
            "bigscreen_snapshot_refresh",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            warning=f"{type(exc).__name__}: {exc}",
        )
    finally:
        with _SNAPSHOT_CACHE_LOCK:
            _SNAPSHOT_REFRESHING.discard(cache_key)


def clear_bigscreen_cache() -> None:
    with _SNAPSHOT_CACHE_LOCK:
        _SNAPSHOT_CACHE.clear()
        _SNAPSHOT_CACHE_SOURCE_SIGNATURES.clear()
        _SNAPSHOT_REFRESHING.clear()


def _snapshot_source_signature(root: Path) -> tuple[Any, ...]:
    exports_root = readers.default_exports_root(root)
    export_date = beijing_today(datetime.now(UTC)).isoformat()
    return (
        _directory_signature(root / "silver" / "market_bar_health"),
        _lake_dataset_signature(root / "silver" / "market_bar"),
        _lake_dataset_signature(root / "silver" / "orderbook_snapshot"),
        _lake_dataset_signature(root / "silver" / "trade_print"),
        _directory_signature(root / "gold" / "cost_bucket_daily"),
        _directory_signature(root / "gold" / "cost_bootstrap_readiness"),
        _directory_signature(root / "gold" / "cost_probe_fill_bill_match"),
        _directory_signature(root / "gold" / "cost_probe_cost_disagreement"),
        _path_signature(exports_root / f".quant_lab_web_export_{export_date}.json"),
        _path_signature(exports_root / "export_index.json"),
        _expert_pack_collection_signature(exports_root),
        _path_signature(root / "reports" / "v5_enforce_readiness.json"),
        _directory_signature(root / "bronze" / "v5_bundle_manifest"),
        _directory_signature(root / "gold" / "strategy_health_daily"),
        _directory_signature(root / "gold" / "quant_lab_opportunity_cost_event"),
        _directory_signature(root / "gold" / "quant_lab_opportunity_cost_daily"),
        _directory_signature(root / "gold" / "opportunity_cost_by_bucket"),
        _directory_signature(root / "gold" / "quant_lab_decision_regret"),
        _directory_signature(root / "gold" / "paper_strategy_proposal"),
        _directory_signature(root / "gold" / "paper_strategy_registry"),
        _directory_signature(root / "gold" / "paper_strategy_promotion_gate"),
        _directory_signature(root / "gold" / "strategy_cost_trust"),
        _directory_signature(root / "gold" / "ai_research_run"),
        _directory_signature(root / "gold" / "ai_research_finding"),
        _directory_signature(root / "gold" / "ai_factor_proposal"),
        _directory_signature(root / "gold" / "ai_paper_strategy_draft"),
        _directory_signature(root / "gold" / "ai_experiment_proposal"),
        _directory_signature(root / "gold" / "ai_code_review_target"),
        _directory_signature(_ai_queue_root() / "state"),
        _directory_signature(_ai_queue_root() / "pending"),
        _directory_signature(_ai_queue_root() / "running"),
        _directory_signature(_ai_queue_root() / "failed"),
        _path_signature(_web_v2_smoke_status_path()),
    )


def _lake_dataset_signature(path: Path) -> tuple[Any, ...]:
    signature = readers.web_dataset_source_signature(path)
    signature_kind = signature[0] if signature else None
    if signature_kind in {"snapshot_meta", "lake_file_index"}:
        return (str(path), signature)
    return _directory_signature(path)


def _path_signature(path: Path) -> tuple[Any, ...]:
    try:
        stat = path.stat()
    except OSError:
        return (str(path), "missing")
    return (str(path), stat.st_mtime_ns, stat.st_size)


def _directory_signature(path: Path, *, limit: int = 24, nested_limit: int = 8) -> tuple[Any, ...]:
    try:
        stat = path.stat()
        children_by_name = sorted(path.iterdir(), key=lambda child: child.name)
    except OSError:
        return (str(path), "missing")

    selected_children: dict[str, Path] = {
        child.name: child
        for child in children_by_name[:limit] + children_by_name[-limit:]
    }
    child_signatures: list[tuple[str, Any, Any, tuple[Any, ...]]] = []
    for child in selected_children.values():
        try:
            child_stat = child.stat()
            child_mtime: Any = child_stat.st_mtime_ns
            child_size: Any = child_stat.st_size
        except OSError:
            child_mtime = "unreadable"
            child_size = "unreadable"
        nested_signature = _shallow_nested_signature(child, limit=nested_limit)
        child_signatures.append((child.name, child_mtime, child_size, nested_signature))
    return (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        len(children_by_name),
        tuple(sorted(child_signatures, key=lambda item: item[0])),
    )


def _shallow_nested_signature(path: Path, *, limit: int) -> tuple[Any, ...]:
    if not path.is_dir():
        return ()
    try:
        children = list(path.iterdir())
    except OSError:
        return ("unreadable",)
    nested_stats: list[tuple[Path, Any, Any]] = []
    for child in children:
        try:
            child_stat = child.stat()
        except OSError:
            nested_stats.append((child, "unreadable", "unreadable"))
            continue
        nested_stats.append((child, child_stat.st_mtime_ns, child_stat.st_size))
    selected = sorted(
        nested_stats,
        key=lambda item: (item[1] if isinstance(item[1], int) else -1, item[0].name),
        reverse=True,
    )[:limit]
    return tuple(
        (child.name, child_mtime, child_size)
        for child, child_mtime, child_size in sorted(selected, key=lambda item: item[0].name)
    )


def _expert_pack_collection_signature(root: Path, *, limit: int = 24) -> tuple[Any, ...]:
    try:
        packs = sorted(
            root.glob("quant_lab_expert_pack_*.zip"),
            key=lambda path: (path.stat().st_mtime, path.name),
            reverse=True,
        )
    except OSError:
        return (str(root), "unreadable")
    signatures: list[tuple[str, Any, Any]] = []
    for path in packs[:limit]:
        try:
            stat = path.stat()
        except OSError:
            signatures.append((path.name, "unreadable", "unreadable"))
            continue
        signatures.append((path.name, stat.st_mtime_ns, stat.st_size))
    return (str(root), len(packs), tuple(signatures))


def _snapshot_cache_ttl_seconds() -> float:
    raw = os.environ.get("QUANT_LAB_BIGSCREEN_CACHE_TTL_SECONDS", "")
    try:
        value = float(raw) if raw else SNAPSHOT_CACHE_TTL_SECONDS
    except ValueError:
        return SNAPSHOT_CACHE_TTL_SECONDS
    return max(0.0, value)


def _snapshot_cache_stale_grace_seconds() -> float:
    raw = os.environ.get("QUANT_LAB_BIGSCREEN_STALE_GRACE_SECONDS", "")
    try:
        value = float(raw) if raw else SNAPSHOT_CACHE_STALE_GRACE_SECONDS
    except ValueError:
        return SNAPSHOT_CACHE_STALE_GRACE_SECONDS
    return max(0.0, value)


def _snapshot_async_refresh_enabled() -> bool:
    raw = os.environ.get("QUANT_LAB_BIGSCREEN_ASYNC_REFRESH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _market_bar_warning_delay_seconds() -> int:
    return _seconds_env(
        "QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS",
        MARKET_BAR_WARNING_DELAY_SECONDS,
    )


def _market_bar_critical_delay_seconds() -> int:
    warning = _market_bar_warning_delay_seconds()
    configured = _seconds_env(
        "QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS",
        MARKET_BAR_CRITICAL_DELAY_SECONDS,
    )
    return max(warning, configured)


def _web_v2_api_metrics_window_minutes() -> int:
    return _nonnegative_int_env(
        "QUANT_LAB_WEB_V2_API_METRICS_WINDOW_MINUTES",
        WEB_V2_API_METRICS_WINDOW_MINUTES,
    )


def _nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(float(raw)) if raw else default
    except ValueError:
        return default
    return max(0, value)


def _seconds_env(name: str, default: int) -> int:
    return _nonnegative_int_env(name, default)


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _web_v2_smoke_status_path() -> Path:
    raw = os.environ.get("QUANT_LAB_WEB_V2_SMOKE_STATUS_PATH", "").strip()
    return Path(raw) if raw else WEB_V2_SMOKE_STATUS_PATH


def _web_v2_smoke_max_age_seconds() -> int:
    return _seconds_env("QUANT_LAB_WEB_V2_SMOKE_MAX_AGE_SECONDS", WEB_V2_SMOKE_MAX_AGE_SECONDS)


def _web_v2_smoke_status_required() -> bool:
    return _bool_env("QUANT_LAB_WEB_V2_SMOKE_REQUIRE_STATUS", default=False)


def _safe_summary(name: str, fn: Any, *args: Any) -> dict[str, Any]:
    try:
        value = fn(*args)
    except Exception as exc:
        return {"warnings": [f"{name} failed: {exc}"], "status": "UNKNOWN"}
    return value if isinstance(value, dict) else {"warnings": [f"{name} returned non-dict"]}


def _safe_api_metrics(root: Path) -> dict[str, Any]:
    window_minutes = _web_v2_api_metrics_window_minutes()
    since_minutes = window_minutes if window_minutes > 0 else None
    try:
        summary = api_metrics_summary(root, since_minutes=since_minutes)
    except Exception as exc:
        return {
            "window_minutes": since_minutes,
            "warnings": [f"api_metrics_summary failed: {exc}"],
        }
    if isinstance(summary, dict):
        summary.setdefault("window_minutes", since_minutes)
        return summary
    return {
        "window_minutes": since_minutes,
        "warnings": ["api_metrics_summary returned non-dict"],
    }


def _safe_strategy_summary(root: Path) -> dict[str, Any]:
    warnings: list[str] = []
    frames: dict[str, pl.DataFrame] = {}
    for dataset_name, output_key in [
        ("strategy_opportunity_advisory", "strategy_opportunity_advisory"),
        ("alpha_discovery_board", "alpha_discovery_board"),
        ("alpha_factory_promotion_queue", "alpha_factory_promotion_queue"),
        ("risk_on_multi_buy_shadow", "risk_on_multi_buy_shadow"),
        ("research_portfolio_status", "research_portfolio_status"),
        ("quant_lab_opportunity_cost_daily", "quant_lab_opportunity_cost_daily"),
        ("opportunity_cost_by_bucket", "opportunity_cost_by_bucket"),
        ("quant_lab_decision_regret", "quant_lab_decision_regret"),
        ("factor_candidate", "factor_candidate"),
        ("factor_evidence", "factor_evidence"),
        ("factor_correlation_daily", "factor_correlation_daily"),
        ("paper_strategy_proposal", "paper_strategy_proposal"),
        ("paper_strategy_registry", "paper_strategy_registry"),
        ("paper_strategy_promotion_gate", "paper_strategy_promotion_gate"),
        ("strategy_cost_trust", "strategy_cost_trust"),
    ]:
        frame, warning = _read_display_frame(root, dataset_name)
        if frame.is_empty() and dataset_name == "risk_on_multi_buy_shadow":
            frame, warning = _read_display_frame(root, "v5_risk_on_multi_buy_shadow")
        frames[output_key] = frame
        if warning:
            warnings.append(warning)
    frames["factor_strategy_bridge_candidates"] = _latest_factor_strategy_bridge_candidates(
        root
    )
    frames["fast_microstructure_forward_test"] = _latest_export_report_frame(
        root,
        "reports/fast_microstructure_forward_test.csv",
    )

    advisory = frames.get("strategy_opportunity_advisory", pl.DataFrame())
    discovery = frames.get("alpha_discovery_board", pl.DataFrame())
    full_advisory, full_advisory_warning = _read_strategy_advisory_for_counts(root)
    advisory_for_counts = full_advisory if not full_advisory.is_empty() else advisory
    if full_advisory_warning and not full_advisory.is_empty():
        warnings.append(f"strategy_opportunity_advisory 全量统计警告：{full_advisory_warning}")
    elif full_advisory_warning and not advisory.is_empty():
        warnings.append(
            "strategy_opportunity_advisory 全量统计不可用，策略流统计退回展示样本："
            f"{full_advisory_warning}"
        )
    if advisory.is_empty():
        warnings.append("strategy_opportunity_advisory 数据集缺失或为空")
    return {
        **frames,
        "counts": {},
        "strategy_opportunity_advisory_rank": advisory_for_counts,
        "strategy_flow_counts": _strategy_flow_counts_from_frame(advisory_for_counts),
        "strategy_counts": _count_by_column(advisory_for_counts, "decision"),
        "discovery_counts": _count_by_column(discovery, "decision"),
        "warnings": warnings,
    }


def _safe_ai_research_summary(
    root: Path,
    *,
    generated_at: datetime,
) -> dict[str, Any]:
    dataset_names = (
        "ai_research_run",
        "ai_research_finding",
        "ai_factor_proposal",
        "ai_paper_strategy_draft",
        "ai_experiment_proposal",
        "ai_code_review_target",
    )
    frames: dict[str, pl.DataFrame] = {}
    warnings: list[str] = []
    for dataset_name in dataset_names:
        frame, warning = _read_display_frame(root, dataset_name)
        frames[dataset_name] = frame
        if warning and not frame.is_empty():
            warnings.append(warning)

    queue = _ai_queue_status()
    run_rows = _ai_rows(frames["ai_research_run"], sort_by="completed_at", limit=8)
    latest_run = run_rows[0] if run_rows else {}
    latest_task_id = str(latest_run.get("task_id") or "")
    counts = {
        "run_count": frames["ai_research_run"].height,
        "finding_count": frames["ai_research_finding"].height,
        "factor_proposal_count": frames["ai_factor_proposal"].height,
        "paper_draft_count": frames["ai_paper_strategy_draft"].height,
        "experiment_count": frames["ai_experiment_proposal"].height,
        "code_review_target_count": frames["ai_code_review_target"].height,
    }
    queue_counts = queue.get("counts") if isinstance(queue.get("counts"), dict) else {}
    if int(queue_counts.get("running") or 0) > 0:
        status = "RUNNING"
    elif int(queue_counts.get("pending") or 0) > 0:
        status = "PENDING"
    elif run_rows:
        status = "AI_RESULT_AVAILABLE"
    elif int(queue_counts.get("failed") or 0) > 0:
        status = "FAILED"
    else:
        status = "WAITING_FOR_FIRST_RESULT"

    current_findings = _ai_rows_for_task(
        frames["ai_research_finding"], latest_task_id, sort_by="completed_at", limit=12
    )
    current_factors = _ai_rows_for_task(
        frames["ai_factor_proposal"], latest_task_id, sort_by="completed_at", limit=10
    )
    current_paper_drafts = _ai_rows_for_task(
        frames["ai_paper_strategy_draft"], latest_task_id, sort_by="completed_at", limit=8
    )
    current_experiments = _ai_rows_for_task(
        frames["ai_experiment_proposal"], latest_task_id, sort_by="completed_at", limit=8
    )
    current_code_targets = _ai_rows_for_task(
        frames["ai_code_review_target"], latest_task_id, sort_by="completed_at", limit=8
    )

    return {
        "status": status,
        "mode": "diagnostic_only",
        "research_only": True,
        "requires_human_review": True,
        "live_order_effect": "none_read_only_research",
        "latest_run": latest_run,
        "latest_run_age_seconds": _age_seconds(generated_at, latest_run.get("completed_at")),
        "counts": counts,
        "queue": queue,
        "recent_runs": run_rows,
        "primary_bottleneck_id": latest_run.get("primary_bottleneck_id"),
        "root_cause_tree": _json_array(latest_run.get("root_cause_tree_json")),
        "next_actions": _json_array(latest_run.get("next_actions_json")),
        "continuity": _json_object(latest_run.get("continuity_json")),
        "validation_events": _json_array(latest_run.get("validation_events_json")),
        "findings": current_findings,
        "factor_proposals": current_factors,
        "paper_strategy_drafts": current_paper_drafts,
        "experiment_proposals": current_experiments,
        "code_review_targets": current_code_targets,
        "warnings": warnings,
    }


def _ai_rows(frame: pl.DataFrame, *, sort_by: str, limit: int) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    ordered = frame
    if sort_by in frame.columns:
        ordered = frame.sort(sort_by, descending=True, nulls_last=True)
    return _frame_rows(ordered, limit=limit)


def _ai_rows_for_task(
    frame: pl.DataFrame,
    task_id: str,
    *,
    sort_by: str,
    limit: int,
) -> list[dict[str, Any]]:
    if frame.is_empty() or not task_id or "task_id" not in frame.columns:
        return []
    return _ai_rows(
        frame.filter(pl.col("task_id") == task_id),
        sort_by=sort_by,
        limit=limit,
    )


def _json_array(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _ai_queue_root() -> Path:
    value = os.environ.get("QUANT_LAB_AI_QUEUE_ROOT", "").strip()
    return Path(value) if value else Path("/var/lib/quant-lab/ai_queue")


def _ai_queue_status() -> dict[str, Any]:
    root = _ai_queue_root()
    counts: dict[str, int] = {}
    warnings: list[str] = []
    for state in ("pending", "running", "completed", "failed"):
        path = root / state
        try:
            counts[state] = sum(1 for item in path.iterdir() if item.is_dir())
        except FileNotFoundError:
            counts[state] = 0
        except OSError as exc:
            counts[state] = 0
            warnings.append(f"{state}:{type(exc).__name__}")
    for state in ("inbox", "imported", "rejected"):
        path = root / "results" / state
        try:
            counts[f"result_{state}"] = sum(
                1
                for item in path.iterdir()
                if item.is_dir() and not item.name.startswith(".")
            )
        except FileNotFoundError:
            counts[f"result_{state}"] = 0
        except OSError as exc:
            counts[f"result_{state}"] = 0
            warnings.append(f"result_{state}:{type(exc).__name__}")
    state_payload: dict[str, Any] = {}
    try:
        raw = json.loads((root / "state" / "last_task.json").read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            state_payload = _json_value(raw)
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "counts": counts,
        "last_task": state_payload,
        "warnings": warnings,
    }


def _read_strategy_advisory_for_counts(root: Path) -> tuple[pl.DataFrame, str | None]:
    try:
        frame, warning = readers.read_dataset_with_warning(
            root,
            "strategy_opportunity_advisory",
        )
    except Exception as exc:
        return pl.DataFrame(), f"strategy_opportunity_advisory 全量统计读取失败：{exc}"
    return frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(), warning


def _read_display_frame(root: Path, dataset_name: str) -> tuple[pl.DataFrame, str | None]:
    try:
        frame, warning = readers._read_web_display_dataset_with_warning(root, dataset_name)
        return frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(), warning
    except Exception as exc:
        return pl.DataFrame(), f"{dataset_name} 大屏抽样读取失败：{exc}"


def _latest_export_report_frame(root: Path, member_name: str) -> pl.DataFrame:
    exports_root = readers.default_exports_root(root)
    if not exports_root.exists():
        return pl.DataFrame()
    packs = sorted(
        exports_root.glob("quant_lab_expert_pack_*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for pack_path in packs[:5]:
        try:
            with zipfile.ZipFile(pack_path) as archive:
                with archive.open(member_name) as handle:
                    rows = list(DictReader(TextIOWrapper(handle, encoding="utf-8")))
            return pl.DataFrame(rows) if rows else pl.DataFrame()
        except Exception:
            continue
    return pl.DataFrame()


def _web_v2_export_summary_from_history(
    root: Path,
    history: dict[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    exports_root = readers.default_exports_root(root)
    export_date = beijing_today(generated_at).isoformat()
    packs = _frame_rows(history.get("packs"), limit=12)
    available_pack = _first_existing_pack_path(exports_root, packs) or _safe_export_pack_path(
        exports_root,
        history.get("latest_pack"),
    )
    status = _read_web_export_status(exports_root, export_date)
    state = str(status.get("state") or ("manual_missing" if available_pack else "missing"))
    state_lower = state.lower()
    manual_pack = (
        _safe_export_pack_path(exports_root, status.get("zip_path"))
        if state_lower == "succeeded"
        else None
    )
    latest_pack = None
    latest_source = None
    if state_lower not in {"starting", "running"} and manual_pack is not None:
        if available_pack is not None and _pack_name_matches_export_date(
            available_pack.name,
            export_date,
        ):
            latest_pack = available_pack
            latest_source = (
                "manual_web_request"
                if _same_export_pack_path(manual_pack, available_pack)
                else "requested_date_pack"
            )
        elif manual_pack is not None and _pack_name_matches_export_date(
            manual_pack.name,
            export_date,
        ):
            latest_pack = manual_pack
            latest_source = "manual_web_request"

    display_available_pack = (
        available_pack
        if (
            available_pack is not None
            and _pack_name_matches_export_date(available_pack.name, export_date)
        )
        else None
    )
    display_pack = latest_pack or display_available_pack
    display_source = (
        latest_source
        or ("available_pack" if display_available_pack is not None else None)
    )

    if display_pack is not None:
        manifest = readers._read_json_from_zip(display_pack, "manifest.json")
        data_quality = readers._read_json_from_zip(display_pack, "data_quality.json")
        questions = readers._read_text_from_zip(display_pack, "expert_questions.md").splitlines()
    else:
        manifest = {}
        data_quality = {}
        questions = []

    return {
        **history,
        "latest_pack": str(latest_pack) if latest_pack is not None else None,
        "latest_pack_source": latest_source,
        "display_pack": str(display_pack) if display_pack is not None else None,
        "display_pack_source": display_source,
        "manual_latest_pack": str(manual_pack) if manual_pack is not None else None,
        "manual_latest_pack_name": manual_pack.name if manual_pack is not None else None,
        "available_pack": str(available_pack) if available_pack is not None else None,
        "available_pack_name": available_pack.name if available_pack is not None else None,
        "manual_state": state,
        "manual_status": status,
        "export_date": export_date,
        "manifest_summary": manifest,
        "data_quality_summary": data_quality,
        "expert_questions": [line for line in questions if str(line).strip()][:20],
        "warnings": _manual_export_warnings(status),
    }


def _read_web_export_status(exports_root: Path, export_date: str) -> dict[str, Any]:
    path = exports_root / f".quant_lab_web_export_{export_date}.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _first_existing_pack_path(exports_root: Path, rows: list[dict[str, Any]]) -> Path | None:
    for row in rows:
        path = _safe_export_pack_path(exports_root, row.get("path"))
        if path is not None:
            return path
        name = row.get("name")
        if name:
            path = _safe_export_pack_path(exports_root, exports_root / str(name))
            if path is not None:
                return path
    return None


def _safe_export_pack_path(exports_root: Path, raw_value: Any) -> Path | None:
    if not raw_value:
        return None
    try:
        path = Path(str(raw_value))
        if not path.is_absolute():
            path = exports_root / path
        resolved = path.resolve()
        resolved.relative_to(exports_root.resolve())
    except (OSError, ValueError):
        return None
    if resolved.is_file() and _is_expert_pack_name(resolved.name):
        return resolved
    return None


def _same_export_pack_path(left: Path | None, right: Path | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left == right


def _pack_name_matches_export_date(file_name: str, export_date: str) -> bool:
    if file_name == f"quant_lab_expert_pack_{export_date}.zip" or file_name.startswith(
        f"quant_lab_expert_pack_{export_date}_"
    ):
        return True
    target_day = str(export_date or "").replace("-", "")
    match = re.match(
        r"^quant_lab_expert_pack_\d{4}-\d{2}-\d{2}_(\d{8})T\d{6}",
        str(file_name or ""),
    )
    return bool(target_day and match and match.group(1) == target_day)


def _manual_export_warnings(status: dict[str, Any]) -> list[str]:
    state = str(status.get("state") or "").lower()
    if state != "failed":
        return []
    error = str(status.get("error") or "unknown").strip()
    return [f"expert_pack_manual_export_failed: {error}"]


def _latest_factor_strategy_bridge_candidates(root: Path) -> pl.DataFrame:
    return _latest_export_report_frame(
        root,
        "reports/factor_strategy_bridge_candidates.csv",
    )


def _count_by_column(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    return {
        str(row[column]): int(row["count"])
        for row in frame.group_by(column).len(name="count").to_dicts()
    }


def _overview_from_summaries(
    data_health: dict[str, Any],
    v5: dict[str, Any],
    consumers: dict[str, Any],
) -> dict[str, Any]:
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    permissions = (
        consumers.get("permissions") if isinstance(consumers.get("permissions"), dict) else {}
    )
    overview = {
        "status": "UNKNOWN",
        "v5_permission": permissions.get("v5", "UNKNOWN"),
        "latest_market_bar_ts": data_health.get("latest_market_bar_ts"),
        "latest_market_bar_close_ts": data_health.get("latest_market_bar_close_ts"),
        "market_bar_timeframe": data_health.get("market_bar_timeframe"),
        "latest_v5_bundle_ts": latest.get("latest_bundle_ts"),
        "diagnostics": {
            "latest_v5_bundle_ts": latest.get("latest_bundle_ts"),
        },
    }
    if "v7" in permissions:
        overview["v7_permission"] = permissions["v7"]
    return overview


def _frame_rows(frame: Any, limit: int = 20) -> list[dict[str, Any]]:
    if isinstance(frame, pl.DataFrame) and not frame.is_empty():
        return [_json_row(row) for row in readers.redact_frame(frame).head(limit).to_dicts()]
    if isinstance(frame, list):
        return [_json_row(row) for row in frame[:limit] if isinstance(row, dict)]
    return []


def _json_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in row.items()}


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_value(child) for child in value]
    return value


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _warnings(summary: dict[str, Any]) -> list[str]:
    values = summary.get("warnings")
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _system_warnings(
    warnings: list[str],
    exports: dict[str, Any],
    data_health: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> list[str]:
    out = list(warnings)
    if data_health is not None:
        issue = _market_bar_freshness_issue(data_health, generated_at or datetime.now(UTC))
        if issue is not None:
            label = "critical" if issue["severity"] == "CRITICAL" else "warning"
            out.append(
                f"market_bar_freshness_{label}: latest={issue['latest_ts']}; "
                f"latest_close={issue.get('latest_close_ts')}; "
                f"delay_seconds={issue['age_seconds']}; "
                f"threshold_seconds={issue['threshold_seconds']}"
            )
    data_quality = _export_data_quality(exports)
    if not _export_quality_is_read_only_cost_advisory(data_quality):
        return out
    return [
        warning for warning in out if not _is_export_quality_advisory_notice(warning)
    ]


def _is_export_quality_advisory_notice(value: str) -> bool:
    text = str(value).strip().lower()
    return text.startswith(("expert_pack_data_quality_warning:", "expert_pack_warning:"))


def _export_data_quality(exports: dict[str, Any]) -> dict[str, Any]:
    value = exports.get("data_quality_summary")
    return value if isinstance(value, dict) else {}


def _export_quality_level(exports: dict[str, Any]) -> str:
    data_quality = _export_data_quality(exports)
    status = str(data_quality.get("status") or "").upper()
    if _export_quality_is_live_readiness_blocked_only(data_quality):
        return "WARNING"
    if status in {"CRITICAL", "FAIL", "FAILED", "ERROR"}:
        return "CRITICAL"
    if status in {"WARN", "WARNING"}:
        return "WARNING"
    if _quality_warning_count(data_quality) > 0:
        return "WARNING"
    return "OK"


def _export_quality_is_live_readiness_blocked_only(data_quality: dict[str, Any]) -> bool:
    if str(data_quality.get("status") or "").upper() not in {"CRITICAL", "FAIL"}:
        return False
    failures = [
        str(value).strip()
        for value in data_quality.get("failures", [])
        if str(value).strip()
    ]
    if not failures:
        return False
    return all(_is_read_only_live_readiness_failure(value) for value in failures)


def _is_read_only_live_readiness_failure(value: str) -> bool:
    text = value.lower()
    return (
        "quant_lab_enforce_readiness" in text
        and "readiness_status=blocked" in text
        and "actual_or_mixed_cost_coverage_live_universe" in text
    )


def _export_quality_is_read_only_cost_advisory(data_quality: dict[str, Any]) -> bool:
    if str(data_quality.get("status") or "").upper() not in {"WARN", "WARNING"}:
        return False
    if _quality_failure_count(data_quality):
        return False
    warnings = [
        str(value).strip()
        for value in data_quality.get("warnings", [])
        if str(value).strip()
    ]
    if not warnings:
        return False
    has_read_only_live_context = any(
        "live_order_effect=read_only_no_live_order" in warning.lower()
        for warning in warnings
    )
    has_entry_or_scale_context = any(
        "live_order_effect=entry_or_scale_block_only" in warning.lower()
        for warning in warnings
    )
    has_readiness_cost_context = any(
        _is_cost_coverage_readiness_advisory(warning) for warning in warnings
    )
    return (
        has_read_only_live_context
        or has_entry_or_scale_context
        or has_readiness_cost_context
    ) and all(
        _is_read_only_cost_quality_warning(warning) for warning in warnings
    )


def _is_read_only_cost_quality_warning(value: str) -> bool:
    text = value.lower()
    if text.startswith("cost_soft_fallback_ratio:"):
        return True
    if text.startswith("live_universe_stale_actual_or_mixed_cost:"):
        return "live_order_effect=read_only_no_live_order" in text
    if not text.startswith("quant_lab_enforce_readiness:"):
        return False
    if "live_order_effect=entry_or_scale_block_only" in text:
        return True
    if _is_cost_coverage_readiness_advisory(value):
        return True
    return (
        _is_read_only_live_readiness_failure(value)
        and "live_order_effect=read_only_no_live_order" in text
    )


def _is_cost_coverage_readiness_advisory(value: str) -> bool:
    text = value.lower()
    return (
        text.startswith("quant_lab_enforce_readiness:")
        and "actual_or_mixed_cost_coverage" in text
        and "blocked=[]" in text
        and "failures=" not in text
        and (
            "readiness_status=warn" in text
            or "readiness_status=advisory_ready" in text
        )
    )


def _cost_soft_fallback_requires_action(cost: dict[str, Any]) -> bool:
    if (_float(cost.get("soft_fallback_ratio")) or 0.0) <= 0.8:
        return False
    return not _cost_soft_fallback_is_read_only_advisory(cost)


def _cost_probe_disagreement_issue(cost: dict[str, Any]) -> dict[str, Any] | None:
    rows = _frame_rows(cost.get("cost_probe_cost_disagreement"), limit=100)
    if not rows:
        return None
    fail_rows = [
        row for row in rows if str(row.get("status") or "").strip().upper() == "FAIL"
    ]
    warn_rows = [
        row
        for row in rows
        if str(row.get("status") or "").strip().upper() in {"WARN", "WARNING"}
    ]
    issue_rows = fail_rows or warn_rows
    if not issue_rows:
        return None
    severity = "CRITICAL" if fail_rows else "WARNING"
    symbols = sorted({str(row.get("symbol") or "") for row in issue_rows if row.get("symbol")})
    max_diff = max((_float(row.get("diff_bps")) or 0.0) for row in issue_rows)
    status_text = "FAIL" if fail_rows else "WARN"
    return {
        "severity": severity,
        "summary": (
            f"{len(issue_rows)} 个探针成本对账 {status_text}；"
            f"最大差异 {max_diff:.1f}bps；标的 {', '.join(symbols[:4])}"
        ),
    }


def _cost_soft_fallback_is_read_only_advisory(cost: dict[str, Any]) -> bool:
    if (_float(cost.get("hard_fallback_ratio")) or 0.0) > 0.25:
        return False
    coverage_rows = _frame_rows(cost.get("live_universe_cost_coverage"), limit=100)
    if not coverage_rows:
        return False
    actionable_rows = [
        row
        for row in coverage_rows
        if str(row.get("coverage_status") or "").upper()
        in {"CRITICAL", "FAIL", "FAILED", "WARNING", "WARN"}
    ]
    return all(
        str(row.get("live_order_effect") or "").strip().lower() == "read_only_no_live_order"
        for row in (actionable_rows or coverage_rows)
    )


def _quality_failure_count(data_quality: dict[str, Any]) -> int:
    failures = data_quality.get("failures")
    return len(failures) if isinstance(failures, list) else 0


def _export_quality_warnings(exports: dict[str, Any]) -> list[str]:
    data_quality = _export_data_quality(exports)
    level = _export_quality_level(exports)
    if level not in {"WARNING", "CRITICAL"}:
        return []
    status = str(data_quality.get("status") or level)
    warning_count = _quality_warning_count(data_quality)
    failure_count = _quality_failure_count(data_quality)
    label = "critical" if level == "CRITICAL" else "warning"
    out = [
        (
            f"expert_pack_data_quality_{label}: status={status}; "
            f"warnings={warning_count}; failures={failure_count}"
        )
    ]
    warnings = data_quality.get("warnings")
    if isinstance(warnings, list):
        out.extend(
            f"expert_pack_warning: {warning}"
            for warning in warnings[:5]
            if str(warning).strip()
        )
    failures = data_quality.get("failures")
    if isinstance(failures, list):
        out.extend(
            f"expert_pack_failure: {failure}"
            for failure in failures[:5]
            if str(failure).strip()
        )
    return out


def _status_from_inputs(
    overview: dict[str, Any],
    data_health: dict[str, Any],
    cost: dict[str, Any],
    v5: dict[str, Any],
    warnings: list[str],
    exports: dict[str, Any],
    *,
    generated_at: datetime | None = None,
) -> str:
    if _int(data_health.get("schema_violation_count")) or _int(
        data_health.get("unclosed_bar_count")
    ):
        return "CRITICAL"
    if (_float(cost.get("hard_fallback_ratio")) or 0.0) > 0.25:
        return "CRITICAL"
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    if latest.get("kill_switch_enabled") is True:
        return "CRITICAL"
    if latest.get("reconcile_ok") is False or latest.get("ledger_ok") is False:
        return "CRITICAL"
    export_quality_level = _export_quality_level(exports)
    if export_quality_level == "CRITICAL":
        return "CRITICAL"
    market_issue = _market_bar_freshness_issue(data_health, generated_at or datetime.now(UTC))
    if market_issue is not None and market_issue["severity"] == "CRITICAL":
        return "CRITICAL"
    overview_status = str(overview.get("status") or "").upper()
    if overview_status == "CRITICAL":
        return "CRITICAL"
    if warnings or overview_status == "WARNING":
        return "WARNING"
    return "OK"


def _health_score(
    status: str,
    data_health: dict[str, Any],
    cost: dict[str, Any],
    v5: dict[str, Any],
    web_events: list[dict[str, Any]],
    exports: dict[str, Any],
) -> int:
    score = 100
    if status == "CRITICAL":
        score -= 25
    elif status == "WARNING":
        score -= 8
    score -= min((_int(data_health.get("schema_violation_count")) or 0) * 10, 30)
    if _int(data_health.get("unclosed_bar_count")):
        score -= 15
    if (_float(cost.get("hard_fallback_ratio")) or 0.0) > 0.25:
        score -= 20
    if (_float(cost.get("soft_fallback_ratio")) or 0.0) > 0.8:
        score -= 8
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    if latest.get("kill_switch_enabled") is True:
        score -= 15
    if latest.get("reconcile_ok") is False or latest.get("ledger_ok") is False:
        score -= 15
    if sum(1 for row in web_events if row.get("rglob_fallback")):
        score -= 5
    if str(exports.get("manual_state") or "").lower() == "failed":
        score -= 5
    market_issue = _market_bar_freshness_issue(data_health, datetime.now(UTC))
    if market_issue is not None:
        score -= 15 if market_issue["severity"] == "CRITICAL" else 5
    export_quality_level = _export_quality_level(exports)
    if export_quality_level == "CRITICAL":
        score -= 15
    elif export_quality_level == "WARNING":
        score -= 5
    return max(0, min(100, score))


def _kpis(
    generated_at: datetime,
    overview: dict[str, Any],
    collectors: dict[str, Any],
    cost: dict[str, Any],
    v5: dict[str, Any],
    web_events: list[dict[str, Any]],
    api_metrics: dict[str, Any],
) -> dict[str, Any]:
    latest_bundle_ts = _latest_v5_bundle_ts(overview, v5)
    kpis = {
        "platform_status": overview.get("status", "UNKNOWN"),
        "v5_permission": overview.get("v5_permission", "UNKNOWN"),
        "latest_market_bar_ts": _json_value(overview.get("latest_market_bar_ts")),
        "latest_market_bar_close_ts": _json_value(overview.get("latest_market_bar_close_ts")),
        "latest_v5_bundle_ts": _json_value(latest_bundle_ts),
        "market_delay_seconds": _market_bar_delay_seconds(overview, generated_at),
        "v5_bundle_delay_seconds": _age_seconds(generated_at, latest_bundle_ts),
        "cost_hard_fallback_ratio": _float(cost.get("hard_fallback_ratio")),
        "cost_soft_fallback_ratio": _float(cost.get("soft_fallback_ratio")),
        "ws_message_count": collectors.get("orderbook_snapshot_rows", 0),
        "trade_print_rows": collectors.get("trade_print_rows", 0),
        "orderbook_snapshot_rows": collectors.get("orderbook_snapshot_rows", 0),
        "api_p50_ms": _metric_latency(api_metrics, "p50") or _web_latency(web_events, "p50"),
        "api_p95_ms": _metric_latency(api_metrics, "p95") or _web_latency(web_events, "p95"),
    }
    if "v7_permission" in overview:
        kpis["v7_permission"] = overview["v7_permission"]
    return kpis


def _latest_v5_bundle_ts(overview: dict[str, Any], v5: dict[str, Any]) -> Any:
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    return (
        latest.get("latest_bundle_ts")
        or overview.get("diagnostics", {}).get("latest_v5_bundle_ts")
        or overview.get("latest_v5_bundle_ts")
    )


def _age_seconds(now: datetime, value: Any) -> int | None:
    ts = _parse_dt(value)
    if ts is None:
        return None
    return max(0, int((now - ts).total_seconds()))


def _market_bar_reference_ts(payload: dict[str, Any]) -> datetime | None:
    close_ts = _parse_dt(payload.get("latest_market_bar_close_ts"))
    if close_ts is not None:
        return close_ts
    latest_ts = payload.get("latest_market_bar_ts")
    timeframe = str(payload.get("market_bar_timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME)
    return market_bar_close_ts(latest_ts, timeframe)


def _market_bar_delay_seconds(payload: dict[str, Any], now: datetime) -> int | None:
    age_seconds = market_bar_freshness_seconds(
        payload.get("latest_market_bar_ts"),
        latest_close_ts=payload.get("latest_market_bar_close_ts"),
        timeframe=str(payload.get("market_bar_timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME),
        now=now,
    )
    return age_seconds


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _parse_v5_bundle_name_ts(value: Any) -> datetime | None:
    text = str(value or "")
    match = V5_BUNDLE_NAME_RE.search(text)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def _market_bar_freshness_issue(
    data_health: dict[str, Any],
    now: datetime,
) -> dict[str, Any] | None:
    latest_ts = data_health.get("latest_market_bar_ts")
    age_seconds = _market_bar_delay_seconds(data_health, now)
    if age_seconds is None:
        return None
    reference_ts = _market_bar_reference_ts(data_health)
    critical_threshold = _market_bar_critical_delay_seconds()
    warning_threshold = _market_bar_warning_delay_seconds()
    if age_seconds >= critical_threshold:
        return {
            "severity": "CRITICAL",
            "age_seconds": age_seconds,
            "latest_ts": _json_value(latest_ts),
            "latest_close_ts": _json_value(reference_ts),
            "threshold_seconds": critical_threshold,
        }
    if age_seconds >= warning_threshold:
        return {
            "severity": "WARNING",
            "age_seconds": age_seconds,
            "latest_ts": _json_value(latest_ts),
            "latest_close_ts": _json_value(reference_ts),
            "threshold_seconds": warning_threshold,
        }
    return None


def _market_bar_status(data_health: dict[str, Any], now: datetime) -> str:
    if not data_health.get("latest_market_bar_ts"):
        return "WARNING"
    issue = _market_bar_freshness_issue(data_health, now)
    return str(issue["severity"]) if issue is not None else "OK"


def _web_latency(events: list[dict[str, Any]], key: str) -> float | None:
    values = sorted(
        _float(row.get("elapsed_ms"))
        for row in events
        if _float(row.get("elapsed_ms")) is not None
    )
    if not values:
        return None
    if key == "p50":
        index = int((len(values) - 1) * 0.5)
    else:
        index = int((len(values) - 1) * 0.95)
    return round(float(values[index]), 3)


def _metric_latency(metrics: dict[str, Any], key: str) -> float | None:
    latency = metrics.get("latency_ms")
    if isinstance(latency, dict):
        return _float(latency.get(key))
    return None


def _data_matrix(
    market: dict[str, Any],
    collectors: dict[str, Any],
    cost: dict[str, Any],
    strategy: dict[str, Any],
    data_health: dict[str, Any],
    overview: dict[str, Any],
) -> dict[str, Any]:
    symbols = _matrix_symbols(market, cost, strategy)
    spread_by_symbol = _rows_by_symbol(_frame_rows(market.get("spread_bps"), limit=80))
    trade_by_symbol = _rows_by_symbol(_frame_rows(market.get("trade_activity"), limit=80))
    regime_by_symbol = _rows_by_symbol(_frame_rows(market.get("regimes"), limit=80))
    cost_by_symbol = _rows_by_symbol(_frame_rows(cost.get("costs"), limit=80))
    live_cost_by_symbol = _rows_by_symbol(
        _frame_rows(cost.get("live_universe_cost_coverage"), limit=80)
    )
    advisory_by_symbol = _latest_by_symbol(
        _frame_rows(strategy.get("strategy_opportunity_advisory"), limit=80)
    )
    evidence_by_symbol = _latest_by_symbol(_frame_rows(strategy.get("strategy_evidence"), limit=80))
    latest_market_by_symbol = _rows_by_symbol(
        _frame_rows(data_health.get("latest_per_symbol"), limit=500)
    )

    now = datetime.now(UTC)
    fallback_market_status = _market_bar_status(data_health, now)
    fallback_market_delay_seconds = _market_bar_delay_seconds(data_health, now)
    fallback_market_latest_ts = data_health.get("latest_market_bar_ts")
    fallback_market_close_ts = _market_bar_reference_ts(data_health)
    ws_status = _status_label(collectors.get("okx_public_ws_status"))
    rows: list[dict[str, Any]] = []
    for symbol in symbols[:16]:
        spread = spread_by_symbol.get(symbol, {})
        trade = trade_by_symbol.get(symbol, {})
        regime = regime_by_symbol.get(symbol, {})
        market_bar = _symbol_market_bar_cell(
            latest_market_by_symbol.get(symbol),
            data_health=data_health,
            fallback_status=fallback_market_status,
            fallback_delay_seconds=fallback_market_delay_seconds,
            fallback_latest_ts=fallback_market_latest_ts,
            fallback_close_ts=fallback_market_close_ts,
            now=now,
        )
        cost_row = cost_by_symbol.get(symbol, {})
        live_cost_row = live_cost_by_symbol.get(symbol, {})
        advisory = advisory_by_symbol.get(symbol, {})
        evidence = evidence_by_symbol.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "market_bar": {
                    **market_bar,
                    "regime": regime.get("volatility_regime") or regime.get("regime"),
                },
                "ws": {
                    "status": ws_status,
                    "messages": collectors.get("orderbook_snapshot_rows"),
                },
                "spread": _spread_cell(spread.get("spread_bps")),
                "trade": {
                    "status": "OK" if _int(trade.get("trade_count")) else "WARNING",
                    "trade_count": _int(trade.get("trade_count")),
                },
                "cost": {
                    "status": _cost_status(cost_row, cost, live_cost_row),
                    "source": _matrix_cost_source(cost_row, live_cost_row),
                    "coverage_status": live_cost_row.get("coverage_status"),
                    "coverage_reason": live_cost_row.get("coverage_reason"),
                },
                "evidence": {
                    "status": "OK"
                    if evidence.get("complete_sample_count") or evidence.get("sample_count")
                    else "INFO",
                    "complete_sample_count": _int(evidence.get("complete_sample_count")),
                },
                "advisory": {
                    "status": _advisory_status(advisory),
                    "recommended_mode": advisory.get("recommended_mode"),
                    "decision": advisory.get("decision"),
                },
            }
        )
    return {
        "columns": ["market_bar", "ws", "spread", "trade", "cost", "evidence", "advisory"],
        "rows": rows,
    }


def _symbol_market_bar_cell(
    row: dict[str, Any] | None,
    *,
    data_health: dict[str, Any],
    fallback_status: str,
    fallback_delay_seconds: int | None,
    fallback_latest_ts: Any,
    fallback_close_ts: datetime | None,
    now: datetime,
) -> dict[str, Any]:
    if row:
        latest_ts = row.get("latest_ts")
        timeframe = str(
            row.get("timeframe")
            or data_health.get("market_bar_timeframe")
            or DEFAULT_MARKET_BAR_TIMEFRAME
        )
        latest_close_ts = market_bar_close_ts(latest_ts, timeframe)
        freshness_seconds = market_bar_freshness_seconds(
            latest_ts,
            latest_close_ts=latest_close_ts,
            timeframe=timeframe,
            now=now,
        )
        return {
            "status": _market_bar_cell_status(freshness_seconds),
            "freshness_seconds": freshness_seconds,
            "latest_ts": _json_value(latest_ts),
            "latest_close_ts": _json_value(latest_close_ts),
            "source": "per_symbol_latest",
        }
    return {
        "status": fallback_status,
        "freshness_seconds": fallback_delay_seconds,
        "latest_ts": _json_value(fallback_latest_ts),
        "latest_close_ts": _json_value(fallback_close_ts),
        "source": "global_market_bar_health",
    }


def _market_bar_cell_status(freshness_seconds: int | None) -> str:
    if freshness_seconds is None:
        return "WARNING"
    if freshness_seconds >= _market_bar_critical_delay_seconds():
        return "CRITICAL"
    if freshness_seconds >= _market_bar_warning_delay_seconds():
        return "WARNING"
    return "OK"


def _data_matrix_issue_summary(data_matrix: dict[str, Any]) -> dict[str, Any]:
    columns = data_matrix.get("columns")
    rows = data_matrix.get("rows")
    if not isinstance(columns, list) or not isinstance(rows, list):
        return {
            "critical": 0,
            "warning": 0,
            "top": [],
            "critical_by_metric": {},
            "warning_by_metric": {},
        }

    critical = 0
    warning = 0
    critical_top: list[str] = []
    warning_top: list[str] = []
    critical_by_metric: dict[str, int] = {}
    warning_by_metric: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or "").strip() or "unknown"
        for column in columns:
            column_key = str(column)
            if column_key not in ACTIONABLE_DATA_MATRIX_COLUMNS:
                continue
            cell = row.get(column)
            if not isinstance(cell, dict):
                continue
            status = str(cell.get("status") or "").upper()
            if status == "CRITICAL":
                critical += 1
                critical_by_metric[column_key] = critical_by_metric.get(column_key, 0) + 1
                if len(critical_top) < 3:
                    critical_top.append(f"{symbol}.{column}")
            elif status == "WARNING":
                warning += 1
                warning_by_metric[column_key] = warning_by_metric.get(column_key, 0) + 1
                if len(warning_top) < 3:
                    warning_top.append(f"{symbol}.{column}")
    return {
        "critical": critical,
        "warning": warning,
        "top": critical_top or warning_top,
        "critical_by_metric": critical_by_metric,
        "warning_by_metric": warning_by_metric,
    }


def _data_matrix_warnings(data_matrix: dict[str, Any]) -> list[str]:
    summary = _data_matrix_issue_summary(data_matrix)
    critical = int(summary.get("critical") or 0)
    warning = int(summary.get("warning") or 0)
    if critical <= 0 and warning <= 0:
        return []
    top = ",".join(str(value) for value in summary.get("top", []) if str(value).strip())
    return [f"data_matrix_attention: critical={critical}; warning={warning}; top={top}"]


def _data_matrix_action_text(summary: dict[str, Any]) -> tuple[str, str, str]:
    critical = int(summary.get("critical") or 0)
    warning = int(summary.get("warning") or 0)
    critical_by_metric = summary.get("critical_by_metric")
    warning_by_metric = summary.get("warning_by_metric")
    if not isinstance(critical_by_metric, dict):
        critical_by_metric = {}
    if not isinstance(warning_by_metric, dict):
        warning_by_metric = {}
    spread_critical = int(critical_by_metric.get("spread") or 0)
    spread_warning = int(warning_by_metric.get("spread") or 0)
    market_critical = int(critical_by_metric.get("market_bar") or 0)
    market_warning = int(warning_by_metric.get("market_bar") or 0)
    top = ", ".join(str(value) for value in summary.get("top", []))
    top_text = top or "矩阵明细"
    if market_critical:
        return (
            "逐币行情数据过期",
            f"{market_critical} 个逐币行情严重滞后 / {critical} 个严重单元；"
            f"重点查看 {top_text}",
            "打开数据成本页复核 market_bar latest_ts / latest_close_ts，"
            "不要只看全局 market freshness。",
        )
    if market_warning and not critical:
        return (
            "逐币行情新鲜度需关注",
            f"{market_warning} 个逐币行情注意项 / {warning} 个注意单元；"
            f"重点查看 {top_text}",
            "打开数据成本页复核 market_bar latest_ts / latest_close_ts，"
            "确认扩展币池是否需要补采。",
        )
    if spread_critical and market_warning:
        return (
            "价差偏高且逐币行情需复核",
            f"{spread_critical} 个严重价差 + {market_warning} 个逐币行情注意项 / "
            f"{warning} 个注意单元；重点查看 {top_text}",
            "先按实时盘口成本风险处理；同时复核逐币 market_bar latest_ts / "
            "latest_close_ts，确认扩展币池是否需要补采。",
        )
    if spread_critical:
        return (
            "市场价差偏高",
            f"{spread_critical} 个严重价差 / {warning} 个注意单元；重点查看 {top_text}",
            "这是实时盘口成本风险；打开数据成本页复核 spread、cost、advisory 的逐币状态",
        )
    if spread_warning and not critical:
        return (
            "市场价差需关注",
            f"{spread_warning} 个价差注意项 / {warning} 个注意单元；重点查看 {top_text}",
            "这是实时盘口成本风险；打开数据成本页复核 spread、cost、advisory 的逐币状态",
        )
    return (
        "数据矩阵存在注意项",
        f"{critical} 个严重单元 / {warning} 个注意单元；重点查看 {top_text}",
        "打开数据成本页复核 spread、trade、cost、advisory 的逐币状态",
    )


def _matrix_symbols(
    market: dict[str, Any],
    cost: dict[str, Any],
    strategy: dict[str, Any],
) -> list[str]:
    symbols: list[str] = []
    ws_observable_symbols = set(readers.OKX_WS_UNIVERSE_SYMBOLS)

    for row in _frame_rows(market.get("regimes"), limit=80):
        symbol = _normalize_display_symbol(row.get("symbol"))
        if (
            _is_usdt_market_symbol(symbol)
            and symbol in ws_observable_symbols
            and symbol not in symbols
        ):
            symbols.append(symbol)

    for rows in [
        _frame_rows(cost.get("costs"), limit=80),
        _frame_rows(strategy.get("strategy_opportunity_advisory"), limit=80),
        _frame_rows(strategy.get("alpha_discovery_board"), limit=80),
    ]:
        for row in rows:
            symbol = _normalize_display_symbol(row.get("symbol"))
            if _is_usdt_market_symbol(symbol) and symbol not in symbols:
                symbols.append(symbol)
    preferred = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]
    return [symbol for symbol in preferred if symbol in symbols] + [
        symbol for symbol in symbols if symbol not in preferred
    ]


def _normalize_display_symbol(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return normalize_symbol(text)
    except Exception:
        return text.replace("/", "-").upper()


def _is_usdt_market_symbol(symbol: str | None) -> bool:
    if not symbol:
        return False
    return symbol.endswith("-USDT")


def _rows_by_symbol(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = _normalize_display_symbol(row.get("symbol"))
        if symbol:
            result[symbol] = row
    return result


def _latest_by_symbol(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return _rows_by_symbol(rows)


def _strategy_flow(strategy: dict[str, Any]) -> dict[str, Any]:
    advisory_rows = _frame_rows(strategy.get("strategy_opportunity_advisory"), limit=200)
    rank_source = strategy.get("strategy_opportunity_advisory_rank")
    if not isinstance(rank_source, pl.DataFrame) or rank_source.is_empty():
        rank_source = strategy.get("strategy_opportunity_advisory")
    ranking_rows = _frame_rows(rank_source, limit=50_000)
    counts = _strategy_counts(strategy, advisory_rows)
    top = _top_strategy_candidates(ranking_rows or advisory_rows)
    factor_factory = _factor_factory_payload(strategy)
    opportunity_cost = _opportunity_cost_payload(strategy)
    paper_lifecycle = _paper_lifecycle_payload(strategy)
    return {
        "counts": counts,
        "top_candidates": top[:8],
        "top_live_candidates": _top_live_candidates(top)[:5],
        "research_portfolio": _frame_rows(strategy.get("research_portfolio_status"), limit=8),
        "alpha_factory": _frame_rows(strategy.get("alpha_factory_promotion_queue"), limit=8),
        "risk_on_multi_buy": _frame_rows(strategy.get("risk_on_multi_buy_shadow"), limit=6),
        "factor_factory": factor_factory,
        "opportunity_cost": opportunity_cost,
        "paper_lifecycle": paper_lifecycle,
        "fast_microstructure_forward": _fast_microstructure_forward_payload(
            strategy.get("fast_microstructure_forward_test")
        ),
        "advisory_fresh": True,
    }


_PAPER_LIFECYCLE_STATES = (
    "RESEARCH_ONLY",
    "BACKTEST_CANDIDATE",
    "PAPER_PROPOSAL_READY",
    "PAPER_ACK_PENDING",
    "PAPER_TRACKER_ACTIVE",
    "PAPER_EVIDENCE_INSUFFICIENT",
    "PAPER_PROMOTION_READY",
    "CANARY_READY",
    "LIVE_SCALE_READY",
    "REJECTED",
    "KILLED",
)


def _paper_lifecycle_payload(strategy: dict[str, Any]) -> dict[str, Any]:
    sources = (
        ("proposal", _frame_rows(strategy.get("paper_strategy_proposal"), limit=2_000)),
        ("registry", _frame_rows(strategy.get("paper_strategy_registry"), limit=2_000)),
        (
            "promotion",
            _frame_rows(strategy.get("paper_strategy_promotion_gate"), limit=2_000),
        ),
    )
    latest: dict[str, dict[str, Any]] = {}
    for source, rows in sources:
        for row in rows:
            proposal_id = str(row.get("proposal_id") or "").strip()
            strategy_id = str(row.get("strategy_id") or "").strip()
            strategy_version = str(row.get("strategy_version") or "").strip()
            key = proposal_id or f"{strategy_id}:{strategy_version}"
            if not key.strip(":"):
                continue
            merged = {**latest.get(key, {}), **row}
            merged["lifecycle_source"] = source
            merged["lifecycle_state"] = _normalized_paper_lifecycle_state(merged, source)
            latest[key] = merged

    cost_by_strategy: dict[str, dict[str, Any]] = {}
    for row in _frame_rows(strategy.get("strategy_cost_trust"), limit=2_000):
        strategy_id = str(row.get("strategy_id") or "").strip()
        if strategy_id:
            cost_by_strategy[strategy_id] = row

    counts = {state: 0 for state in _PAPER_LIFECYCLE_STATES}
    rows: list[dict[str, Any]] = []
    for row in latest.values():
        state = str(row.get("lifecycle_state") or "RESEARCH_ONLY")
        if state in counts:
            counts[state] += 1
        strategy_id = str(row.get("strategy_id") or "").strip()
        cost = cost_by_strategy.get(strategy_id, {})
        rows.append(
            {
                "proposal_id": row.get("proposal_id"),
                "proposal_hash": row.get("proposal_hash"),
                "strategy_id": strategy_id,
                "strategy_version": row.get("strategy_version"),
                "symbol": row.get("symbol"),
                "lifecycle_state": state,
                "lifecycle_reason": row.get("lifecycle_reason"),
                "blocked_reasons": row.get("promotion_block_reasons")
                or row.get("blocked_reasons"),
                "next_required_actions": row.get("next_required_actions"),
                "paper_only": row.get("paper_only", True),
                "live_order_effect": row.get("live_order_effect")
                or "paper_only_no_live_order",
                "promotion_score": row.get("promotion_score"),
                "cost_trust_level": cost.get("cost_trust_level") or "BLOCK",
                "created_at": row.get("created_at"),
            }
        )
    rows.sort(
        key=lambda row: (
            str(row.get("created_at") or ""),
            str(row.get("proposal_id") or ""),
        ),
        reverse=True,
    )

    cost_counts: dict[str, int] = {}
    for row in cost_by_strategy.values():
        level = str(row.get("cost_trust_level") or "BLOCK")
        cost_counts[level] = cost_counts.get(level, 0) + 1
    return {
        "contract_version": "quant_lab.paper_strategy.v1",
        "runtime_mode": "PAPER_ONLY",
        "live_order_effect": "none",
        "counts": counts,
        "cost_trust_counts": cost_counts,
        "rows": rows[:8],
        "total": len(rows),
    }


def _normalized_paper_lifecycle_state(row: dict[str, Any], source: str) -> str:
    raw = str(row.get("lifecycle_state") or row.get("status") or "").strip().upper()
    if raw == "PAPER_READY":
        return "PAPER_PROPOSAL_READY"
    if raw in _PAPER_LIFECYCLE_STATES:
        return raw
    if str(row.get("reject_reason") or "").strip():
        return "REJECTED"
    if source == "proposal":
        return "PAPER_PROPOSAL_READY"
    if row.get("accepted") is True and row.get("paper_tracker_effective") is True:
        return "PAPER_TRACKER_ACTIVE"
    if row.get("accepted") is True:
        return "PAPER_ACK_PENDING"
    return "PAPER_ACK_PENDING"


def _opportunity_cost_payload(strategy: dict[str, Any]) -> dict[str, Any]:
    daily = _as_frame(strategy.get("quant_lab_opportunity_cost_daily"))
    buckets = _as_frame(strategy.get("opportunity_cost_by_bucket"))
    regrets = _as_frame(strategy.get("quant_lab_decision_regret"))
    daily_rows = _frame_rows(daily, limit=200)
    daily_rows.sort(key=lambda row: str(row.get("day") or ""), reverse=True)
    latest = daily_rows[0] if daily_rows else {}
    recent_7d = daily_rows[:7]
    bucket_rows = _frame_rows(buckets, limit=200)
    bucket_rows.sort(
        key=lambda row: (
            bool(row.get("opportunity_exception_candidate")) is not True,
            _float(row.get("veto_net_value_bps")) or 0.0,
        )
    )
    return {
        "latest_day": latest.get("day"),
        "veto_net_value_bps": _float(latest.get("veto_net_value_bps")),
        "missed_profit_bps": _float(latest.get("false_block_profit_bps_sum")) or 0.0,
        "loss_saved_bps": _float(latest.get("loss_saved_bps_sum")) or 0.0,
        "veto_net_value_bps_7d": _sum_rows(recent_7d, "veto_net_value_bps"),
        "missed_profit_bps_7d": _sum_rows(recent_7d, "false_block_profit_bps_sum"),
        "loss_saved_bps_7d": _sum_rows(recent_7d, "loss_saved_bps_sum"),
        "false_block_count": _int(latest.get("false_block_count")) or 0,
        "loss_saved_count": _int(latest.get("loss_saved_count")) or 0,
        "false_block_count_7d": _sum_int_rows(recent_7d, "false_block_count"),
        "loss_saved_count_7d": _sum_int_rows(recent_7d, "loss_saved_count"),
        "high_confidence_false_block_count": (
            _int(latest.get("high_confidence_false_block_count")) or 0
        ),
        "high_confidence_loss_saved_count": (
            _int(latest.get("high_confidence_loss_saved_count")) or 0
        ),
        "high_confidence_false_block_count_7d": _sum_int_rows(
            recent_7d, "high_confidence_false_block_count"
        ),
        "high_confidence_loss_saved_count_7d": _sum_int_rows(
            recent_7d, "high_confidence_loss_saved_count"
        ),
        "status": latest.get("opportunity_cost_status") or "NO_DATA",
        "top_buckets": bucket_rows[:4],
        "recent_regrets": _frame_rows(regrets, limit=4),
    }


def _sum_rows(rows: list[dict[str, Any]], field: str) -> float:
    return float(sum(_float(row.get(field)) or 0.0 for row in rows))


def _sum_int_rows(rows: list[dict[str, Any]], field: str) -> int:
    return int(sum(_int(row.get(field)) or 0 for row in rows))


def _factor_factory_payload(strategy: dict[str, Any]) -> dict[str, Any]:
    candidates = _as_frame(strategy.get("factor_candidate"))
    evidence = _as_frame(strategy.get("factor_evidence"))
    correlations = _as_frame(strategy.get("factor_correlation_daily"))
    v2_reports = build_factor_factory_v2_reports(
        candidates=candidates,
        evidence=evidence,
        correlations=correlations,
    )
    bridge_candidates = _as_frame(strategy.get("factor_strategy_bridge_candidates"))
    if bridge_candidates.is_empty():
        bridge_candidates = v2_reports["factor_strategy_bridge_candidates"]
    candidate_rows = _factor_candidate_rows(candidates)
    state_counts = _count_by_column(candidates, "candidate_state")
    high_correlation_pairs = _high_correlation_rows(correlations)
    evidence_by_horizon = _factor_evidence_by_horizon(evidence)
    paper_ready_candidates = [
        row
        for row in candidate_rows
        if str(row.get("candidate_state") or "").upper() == "PAPER_READY"
    ]
    warnings: list[str] = []
    if candidates.is_empty():
        warnings.append("factor_candidate_missing_or_empty")
    if evidence.is_empty():
        warnings.append("factor_evidence_missing_or_empty")
    return {
        "title": "Factor Factory",
        "live_order_effect": "none_read_only_research",
        "paper_ready_meaning": "paper review candidate only, not live eligibility",
        "candidate_count": candidates.height,
        "evidence_count": evidence.height,
        "correlation_pair_count": correlations.height,
        "paper_ready_count": state_counts.get("PAPER_READY", 0),
        "paper_review_queue_count": v2_reports["factor_paper_review_queue"].height,
        "strategy_bridge_candidate_count": bridge_candidates.height,
        "high_correlation_pair_count": len(high_correlation_pairs),
        "state_counts": state_counts,
        "latest_candidate_created_at": _max_column_value(candidates, "created_at"),
        "latest_evidence_created_at": _max_column_value(evidence, "created_at"),
        "top_candidates": candidate_rows[:8],
        "paper_ready_candidates": paper_ready_candidates[:8],
        "evidence_by_horizon": evidence_by_horizon,
        "high_correlation_pairs": high_correlation_pairs[:8],
        "dedupe_decisions": _frame_rows(v2_reports["factor_dedupe_decision"], limit=8),
        "family_leaderboard": _frame_rows(v2_reports["factor_family_leaderboard"], limit=8),
        "paper_review_queue": _frame_rows(v2_reports["factor_paper_review_queue"], limit=8),
        "composite_candidates": _frame_rows(v2_reports["composite_factor_candidates"], limit=8),
        "regime_effectiveness": _frame_rows(v2_reports["factor_regime_effectiveness"], limit=8),
        "strategy_bridge_candidates": _frame_rows(
            bridge_candidates,
            limit=8,
        ),
        "warnings": warnings,
    }


def _fast_microstructure_forward_payload(value: Any) -> dict[str, Any]:
    frame = _as_frame(value)
    rows = _frame_rows(frame, limit=1000)
    counts: dict[str, int] = {}
    for row in rows:
        recommendation = str(row.get("recommendation") or "UNKNOWN")
        counts[recommendation] = counts.get(recommendation, 0) + 1
    pass_rows = [
        row
        for row in rows
        if str(row.get("recommendation") or "") == "FORWARD_VALIDATION_PASS"
    ]

    def pass_rank(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            _float(row.get("sample_count")) or 0.0,
            _float(row.get("long_short_bps")) or -999999.0,
            _float(row.get("rank_ic")) or -999999.0,
        )

    top_passes = sorted(pass_rows, key=pass_rank, reverse=True)
    return {
        "title": "Fast Microstructure Forward Test",
        "live_order_effect": "read_only_no_live_order",
        "row_count": len(rows),
        "pass_count": counts.get("FORWARD_VALIDATION_PASS", 0),
        "weak_or_mixed_count": counts.get("FORWARD_VALIDATION_WEAK_OR_MIXED", 0),
        "needs_more_samples_count": counts.get("NEEDS_MORE_FORWARD_SAMPLES", 0),
        "recommendation_counts": counts,
        "latest_generated_at": _max_column_value(frame, "generated_at"),
        "lookback_bars": _max_column_value(frame, "lookback_bars"),
        "top_passes": top_passes[:8],
    }


def _as_frame(value: Any) -> pl.DataFrame:
    return value if isinstance(value, pl.DataFrame) else pl.DataFrame()


def _factor_candidate_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = _frame_rows(frame, limit=500)

    def priority(row: dict[str, Any]) -> tuple[int, float, float, float]:
        state = str(row.get("candidate_state") or "").upper()
        state_priority = {
            "PAPER_READY": 5,
            "KEEP_SHADOW": 4,
            "RESEARCH": 3,
            "KILL": 1,
        }.get(state, 2)
        return (
            state_priority,
            _float(row.get("best_score")) or -999999.0,
            _float(row.get("best_long_short_mean_bps")) or -999999.0,
            _float(row.get("best_rank_ic_mean")) or -999999.0,
        )

    sorted_rows = sorted(rows, key=priority, reverse=True)
    return [_factor_candidate_payload(row) for row in sorted_rows]


def _factor_candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "factor_id": row.get("factor_id"),
        "factor_name": row.get("factor_name"),
        "factor_family": row.get("factor_family"),
        "factor_version": row.get("factor_version"),
        "timeframe": row.get("timeframe"),
        "candidate_state": row.get("candidate_state"),
        "best_horizon_bars": _int(row.get("best_horizon_bars")),
        "tested_horizon_count": _int(row.get("tested_horizon_count")),
        "best_score": _float(row.get("best_score")),
        "avg_score": _float(row.get("avg_score")),
        "best_rank_ic_mean": _float(row.get("best_rank_ic_mean")),
        "best_rank_ic_tstat": _float(row.get("best_rank_ic_tstat")),
        "best_long_short_mean_bps": _float(row.get("best_long_short_mean_bps")),
        "recommended_action": row.get("recommended_action"),
        "manual_review_required": row.get("manual_review_required"),
        "created_at": row.get("created_at"),
        "source": row.get("source"),
        "live_order_effect": "none_read_only_research",
    }


def _factor_evidence_by_horizon(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty() or "horizon_bars" not in frame.columns:
        return []
    rows = _frame_rows(frame, limit=1000)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        horizon = _int(row.get("horizon_bars"))
        if horizon is None:
            continue
        grouped.setdefault(horizon, []).append(row)
    output: list[dict[str, Any]] = []
    for horizon, horizon_rows in sorted(grouped.items()):
        scores = [_float(row.get("score")) for row in horizon_rows]
        rank_ics = [_float(row.get("rank_ic_mean")) for row in horizon_rows]
        spreads = [_float(row.get("long_short_mean_bps")) for row in horizon_rows]
        output.append(
            {
                "horizon_bars": horizon,
                "factor_count": len(horizon_rows),
                "paper_ready_count": sum(
                    1
                    for row in horizon_rows
                    if str(row.get("decision") or "").upper() == "PAPER_READY"
                ),
                "avg_score": _mean(scores),
                "avg_rank_ic_mean": _mean(rank_ics),
                "avg_long_short_mean_bps": _mean(spreads),
            }
        )
    return output


def _high_correlation_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    rows = _frame_rows(frame, limit=1000)
    filtered = [
        row
        for row in rows
        if (_float(row.get("correlation")) is not None)
        and abs(_float(row.get("correlation")) or 0.0) >= 0.90
    ]
    filtered.sort(key=lambda row: abs(_float(row.get("correlation")) or 0.0), reverse=True)
    return [
        {
            "factor_id_left": row.get("factor_id_left"),
            "factor_id_right": row.get("factor_id_right"),
            "correlation": _float(row.get("correlation")),
            "sample_count": _int(row.get("sample_count")),
            "timeframe": row.get("timeframe"),
            "as_of_date": row.get("as_of_date"),
        }
        for row in filtered
    ]


def _mean(values: list[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    if not observed:
        return None
    return round(sum(observed) / len(observed), 6)


def _max_column_value(frame: pl.DataFrame, column: str) -> Any:
    if frame.is_empty() or column not in frame.columns:
        return None
    values = frame.get_column(column).drop_nulls()
    if values.is_empty():
        return None
    try:
        return _json_value(values.max())
    except Exception:
        return None


def _strategy_counts(strategy: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, int]:
    flow_counts = strategy.get("strategy_flow_counts")
    if isinstance(flow_counts, dict) and any(flow_counts.values()):
        return _normalized_strategy_flow_counts(flow_counts)
    if rows:
        return _strategy_flow_counts_from_rows(rows)
    raw = strategy.get("strategy_counts") or strategy.get("alpha_discovery_counts") or {}
    if isinstance(raw, dict):
        return {
            "research": _strategy_count_sum(raw, "RESEARCH_ONLY", "research"),
            "shadow": _strategy_count_sum(raw, "KEEP_SHADOW", "REGIME_SHADOW", "shadow"),
            "paper": _strategy_count_sum(raw, "PAPER_READY", "paper"),
            "kill": _strategy_count_sum(raw, "KILL", "kill"),
        }
    return {"research": 0, "shadow": 0, "paper": 0, "kill": 0}


def _strategy_flow_counts_from_frame(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty():
        return {}
    columns = [column for column in ["decision", "recommended_mode"] if column in frame.columns]
    if not columns:
        return {}
    return _strategy_flow_counts_from_rows(frame.select(columns).to_dicts())


def _strategy_flow_counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"research": 0, "shadow": 0, "paper": 0, "kill": 0}
    for row in rows:
        decision = str(row.get("decision") or "").upper()
        mode = str(row.get("recommended_mode") or "").lower()
        if decision == "KILL":
            counts["kill"] += 1
        elif mode == "paper" or decision == "PAPER_READY":
            counts["paper"] += 1
        elif mode == "shadow" or "SHADOW" in decision:
            counts["shadow"] += 1
        else:
            counts["research"] += 1
    return counts


def _normalized_strategy_flow_counts(raw: dict[str, Any]) -> dict[str, int]:
    return {
        "research": _int(raw.get("research")) or 0,
        "shadow": _int(raw.get("shadow")) or 0,
        "paper": _int(raw.get("paper")) or 0,
        "kill": _int(raw.get("kill")) or 0,
    }


def _strategy_count_sum(raw: dict[str, Any], *keys: str) -> int:
    return sum((_int(raw.get(key)) or 0) for key in keys)


def _top_strategy_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def priority(row: dict[str, Any]) -> tuple[int, float, float, int]:
        decision = str(row.get("decision") or "").upper()
        mode = str(row.get("recommended_mode") or "").lower()
        mode_priority = 0
        if mode == "paper" or decision == "PAPER_READY":
            mode_priority = 4
        elif mode == "shadow" or "SHADOW" in decision:
            mode_priority = 3
        elif mode == "research":
            mode_priority = 2
        elif decision == "KILL":
            mode_priority = 1
        return (
            mode_priority,
            _float(row.get("avg_net_bps")) or -999999.0,
            _float(row.get("p25_net_bps")) or -999999.0,
            _int(row.get("complete_sample_count")) or 0,
        )

    sorted_rows = _dedupe_strategy_candidate_rows(sorted(rows, key=priority, reverse=True))
    return [_strategy_candidate_payload(row) for row in sorted_rows]


def _dedupe_strategy_candidate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None]] = set()
    for row in rows:
        symbol = _normalize_display_symbol(row.get("symbol")) or str(row.get("symbol") or "")
        key = (symbol, _int(row.get("horizon_hours")))
        if symbol and key in seen:
            continue
        if symbol:
            seen.add(key)
        deduped.append(row)
    return deduped


def _strategy_candidate_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "strategy_candidate": row.get("strategy_candidate") or row.get("takeaway"),
        "symbol": _normalize_display_symbol(row.get("symbol")) or row.get("symbol"),
        "decision": row.get("decision"),
        "recommended_mode": row.get("recommended_mode"),
        "horizon_hours": _int(row.get("horizon_hours")),
        "avg_net_bps": _float(row.get("avg_net_bps")),
        "p25_net_bps": _float(row.get("p25_net_bps")),
        "win_rate": _float(row.get("win_rate")),
        "complete_sample_count": _int(row.get("complete_sample_count")),
        "source_module": row.get("source_module"),
        "promotion_state": row.get("promotion_state"),
        "cost_quality_score": _float(row.get("cost_quality_score")),
        "live_block_reasons": row.get("live_block_reasons"),
        "takeaway": row.get("takeaway"),
        "key_metrics": row.get("key_metrics"),
    }


def _top_live_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("recommended_mode") or "").lower() in {"paper", "shadow"}
        and str(row.get("decision") or "").upper() != "KILL"
    ]


def _v5_payload(
    v5: dict[str, Any],
    current_readiness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    payload = {str(key): _json_value(value) for key, value in latest.items()}
    if current_readiness:
        payload["current_enforce_readiness"] = _json_value(current_readiness)
    if "latest_bundle_sha256" in payload and "latest_bundle_sha256_short" not in payload:
        payload["latest_bundle_sha256_short"] = _short_hash(payload["latest_bundle_sha256"])
    return payload


def _short_hash(value: Any) -> str:
    text = str(value or "")
    if len(text) <= 16:
        return text
    return f"{text[:8]}…{text[-6:]}"


def _cost_payload(cost: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "actual_rows",
        "mixed_rows",
        "bootstrap_probe_rows",
        "proxy_rows",
        "global_default_rows",
        "hard_fallback_count",
        "hard_fallback_ratio",
        "soft_fallback_count",
        "soft_fallback_ratio",
        "proxy_only_count",
        "fallback_ratio",
        "fallback_ratio_status",
        "symbols_with_actual_cost",
        "symbols_with_proxy_only",
        "cost_quality_basis",
    ]
    payload = {key: _json_value(cost.get(key)) for key in keys}
    payload["cost_rows"] = _frame_rows(cost.get("costs"), limit=12)
    payload["cost_health"] = _frame_rows(cost.get("cost_health"), limit=5)
    payload["cost_bootstrap_readiness"] = _frame_rows(
        cost.get("cost_bootstrap_readiness"), limit=12
    )
    payload["cost_probe_cost_disagreement"] = _frame_rows(
        cost.get("cost_probe_cost_disagreement"), limit=12
    )
    payload["live_universe_cost_coverage"] = _frame_rows(
        cost.get("live_universe_cost_coverage"), limit=12
    )
    return payload


def _market_payload(market: dict[str, Any]) -> dict[str, Any]:
    regimes = _merge_market_rows(
        _frame_rows(market.get("regimes"), limit=16),
        _frame_rows(market.get("spread_bps"), limit=16),
        _frame_rows(market.get("trade_activity"), limit=16),
    )
    return {
        "regimes": regimes,
        "spread_bps": _frame_rows(market.get("spread_bps"), limit=16),
        "trade_activity": _frame_rows(market.get("trade_activity"), limit=16),
        "abnormal_symbols": _frame_rows(market.get("abnormal_symbols"), limit=16),
    }


def _merge_market_rows(
    regimes: list[dict[str, Any]],
    spreads: list[dict[str, Any]],
    trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spread_by_symbol = _rows_by_symbol(spreads)
    trade_by_symbol = _rows_by_symbol(trades)
    out: list[dict[str, Any]] = []
    for row in regimes:
        symbol = _normalize_display_symbol(row.get("symbol"))
        if not symbol:
            continue
        merged = dict(row)
        merged["symbol"] = symbol
        merged.update(
            {
                "spread_bps": _float(spread_by_symbol.get(symbol, {}).get("spread_bps")),
                "trade_count": _int(trade_by_symbol.get(symbol, {}).get("trade_count")),
            }
        )
        out.append(merged)
    return out


def _collector_payload(collectors: dict[str, Any]) -> dict[str, Any]:
    return {
        "okx_public_ws_status": collectors.get("okx_public_ws_status"),
        "trade_print_rows": collectors.get("trade_print_rows"),
        "orderbook_snapshot_rows": collectors.get("orderbook_snapshot_rows"),
        "collectors": _frame_rows(collectors.get("collectors"), limit=8),
        "collector_health": _frame_rows(collectors.get("collector_health"), limit=8),
    }


def _data_health_payload(
    data_health: dict[str, Any],
    data_matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latest_per_symbol = _current_latest_per_symbol_rows(data_health, data_matrix)
    return {
        "duplicate_bar_count": _int(data_health.get("duplicate_bar_count")) or 0,
        "unclosed_bar_count": _int(data_health.get("unclosed_bar_count")) or 0,
        "schema_violation_count": _int(data_health.get("schema_violation_count")) or 0,
        "missing_bar_ratio": _float(data_health.get("missing_bar_ratio")) or 0.0,
        "latest_market_bar_ts": _json_value(data_health.get("latest_market_bar_ts")),
        "latest_market_bar_close_ts": _json_value(_market_bar_reference_ts(data_health)),
        "market_bar_timeframe": data_health.get("market_bar_timeframe")
        or DEFAULT_MARKET_BAR_TIMEFRAME,
        "market_bar_delay_seconds": _market_bar_delay_seconds(data_health, datetime.now(UTC)),
        "market_bar_freshness_status": _market_bar_status(data_health, datetime.now(UTC)),
        "stale_dataset_count": len(_frame_rows(data_health.get("stale_datasets"), limit=1000)),
        "stale_datasets": _frame_rows(data_health.get("stale_datasets"), limit=8),
        "latest_per_symbol": latest_per_symbol,
        "latest_per_symbol_scope": "current_matrix_symbols_first"
        if data_matrix is not None
        else "raw_latest_per_symbol",
        "latest_per_symbol_total_symbols": len(
            _frame_rows(data_health.get("latest_per_symbol"), limit=5000)
        ),
    }


def _current_latest_per_symbol_rows(
    data_health: dict[str, Any],
    data_matrix: dict[str, Any] | None,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    raw_rows = _frame_rows(data_health.get("latest_per_symbol"), limit=5000)
    if not raw_rows:
        return []
    if data_matrix is None:
        return raw_rows[:limit]

    by_symbol = _rows_by_symbol(raw_rows)
    preferred: list[str] = []
    matrix_rows = data_matrix.get("rows")
    if isinstance(matrix_rows, list):
        for row in matrix_rows:
            if not isinstance(row, dict):
                continue
            symbol = _normalize_display_symbol(row.get("symbol"))
            if _is_usdt_market_symbol(symbol) and symbol not in preferred:
                preferred.append(symbol)
    for symbol in readers.OKX_WS_UNIVERSE_SYMBOLS:
        if symbol not in preferred:
            preferred.append(symbol)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for symbol in preferred:
        row = by_symbol.get(symbol)
        if row is None or symbol in seen:
            continue
        selected.append(row)
        seen.add(symbol)
        if len(selected) >= limit:
            return selected

    remaining = []
    for row in raw_rows:
        symbol = _normalize_display_symbol(row.get("symbol"))
        if not _is_usdt_market_symbol(symbol) or symbol in seen:
            continue
        remaining.append(row)
    remaining.sort(key=_latest_per_symbol_sort_key, reverse=True)
    return (selected + remaining)[:limit]


def _latest_per_symbol_sort_key(row: dict[str, Any]) -> tuple[datetime, int]:
    latest_ts = _parse_dt(row.get("latest_ts")) or datetime.min.replace(tzinfo=UTC)
    return latest_ts, _int(row.get("rows")) or 0


def _legacy_web_anomalies(data_health: dict[str, Any]) -> dict[str, Any]:
    stale_rows = _frame_rows(data_health.get("stale_datasets"), limit=12)
    missing_rows = _frame_rows(data_health.get("missing_bars"), limit=8)
    schema_violations = [
        str(value)
        for value in data_health.get("schema_violations", [])
        if str(value).strip()
    ]
    warning_values = _warnings(data_health)[:8]
    duplicate_count = _int(data_health.get("duplicate_bar_count")) or 0
    unclosed_count = _int(data_health.get("unclosed_bar_count")) or 0
    schema_count = _int(data_health.get("schema_violation_count")) or len(schema_violations)
    missing_ratio = _float(data_health.get("missing_bar_ratio")) or 0.0
    missing_count = sum(_int(row.get("missing_bars")) or 0 for row in missing_rows)
    critical_count = 0
    warning_count = 0
    items: list[dict[str, Any]] = []

    def add_item(
        severity: str,
        title: str,
        summary: str,
        source: str,
        next_action: str,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        nonlocal critical_count, warning_count
        normalized = severity.upper()
        if normalized == "CRITICAL":
            critical_count += 1
        elif normalized == "WARNING":
            warning_count += 1
        items.append(
            {
                "severity": normalized,
                "title": title,
                "summary": summary,
                "source": source,
                "next_action": next_action,
                "rows": rows or [],
            }
        )

    if stale_rows:
        top_severity = "WARNING"
        for row in stale_rows:
            if str(row.get("severity") or "").upper() == "CRITICAL":
                top_severity = "CRITICAL"
                break
        add_item(
            top_severity,
            "数据集过期或缺失",
            f"{len(stale_rows)} 个数据集需要关注；首页只展示前 {min(len(stale_rows), 12)} 条。",
            "legacy_data_health.stale_datasets",
            "检查对应采集 / refresh / telemetry sync 任务，避免下游页面看到旧结果。",
            stale_rows,
        )

    if schema_count or unclosed_count or duplicate_count:
        add_item(
            "CRITICAL" if schema_count or unclosed_count else "WARNING",
            "行情数据结构异常",
            f"schema={schema_count}，未闭合K线={unclosed_count}，重复bar={duplicate_count}。",
            "legacy_data_health.market_bar",
            "优先修复 market_bar 结构 / 未闭合 / 重复问题，再解释策略证据。",
            [{"warning": value} for value in schema_violations[:8]],
        )

    if missing_rows or missing_ratio > 0:
        add_item(
            "WARNING",
            "历史 K 线存在缺口",
            f"missing_bars={missing_count}，missing_ratio={missing_ratio:.4f}。",
            "legacy_data_health.missing_bars",
            "补齐对应 symbol/timeframe 的 market_bar，再刷新研究报告。",
            missing_rows,
        )

    if warning_values and not items:
        add_item(
            "WARNING",
            "数据健康提示",
            warning_values[0],
            "legacy_data_health.warnings",
            "打开数据健康页查看 warning 明细。",
            [{"warning": value} for value in warning_values],
        )

    return {
        "live_order_effect": "none_read_only_display",
        "source": "legacy_streamlit_data_health_overview",
        "has_anomalies": bool(items),
        "total_count": len(items),
        "critical_count": critical_count,
        "warning_count": warning_count,
        "stale_dataset_count": len(stale_rows),
        "schema_violation_count": schema_count,
        "unclosed_bar_count": unclosed_count,
        "duplicate_bar_count": duplicate_count,
        "missing_bar_ratio": missing_ratio,
        "missing_bar_row_count": len(missing_rows),
        "missing_bar_count": missing_count,
        "latest_market_bar_ts": _json_value(data_health.get("latest_market_bar_ts")),
        "latest_market_bar_close_ts": _json_value(_market_bar_reference_ts(data_health)),
        "items": items,
        "warnings": warning_values,
    }


def _server_resources(root: Path) -> dict[str, Any]:
    sampled_at = datetime.now(UTC)
    cpu_count = os.cpu_count() or 1
    load_1m: float | None = None
    load_5m: float | None = None
    load_15m: float | None = None
    try:
        load_1m, load_5m, load_15m = os.getloadavg()
    except (AttributeError, OSError):
        pass
    cpu_usage_percent = _cpu_usage_percent_sample()
    cpu_load_percent = (
        min(999.0, max(0.0, (load_1m / max(cpu_count, 1)) * 100.0))
        if load_1m is not None
        else None
    )
    cpu_display_percent = cpu_usage_percent if cpu_usage_percent is not None else cpu_load_percent
    memory = _memory_usage()
    disk = _disk_usage(root)
    uptime_seconds = _uptime_seconds()
    swap_percent = _percent(memory.get("swap_used_bytes"), memory.get("swap_total_bytes"))
    statuses = [
        _resource_status(cpu_display_percent, warning=80.0, critical=95.0),
        _resource_status(memory.get("memory_used_percent"), warning=80.0, critical=92.0),
        _resource_status(disk.get("disk_used_percent"), warning=80.0, critical=90.0),
        _resource_status(swap_percent, warning=20.0, critical=60.0),
    ]
    status = "CRITICAL" if "CRITICAL" in statuses else "WARNING" if "WARNING" in statuses else "OK"
    return {
        "live_order_effect": "none_read_only_display",
        "source": "server_runtime_readonly",
        "sampled_at": _json_value(sampled_at),
        "hostname": _hostname(),
        "status": status,
        "cpu_count": cpu_count,
        "cpu_usage_percent": _rounded(cpu_usage_percent),
        "cpu_load_percent": _rounded(cpu_load_percent),
        "cpu_display_percent": _rounded(cpu_display_percent),
        "load_1m": _rounded(load_1m, digits=2),
        "load_5m": _rounded(load_5m, digits=2),
        "load_15m": _rounded(load_15m, digits=2),
        "memory_total_bytes": memory.get("memory_total_bytes"),
        "memory_available_bytes": memory.get("memory_available_bytes"),
        "memory_used_bytes": memory.get("memory_used_bytes"),
        "memory_used_percent": _rounded(memory.get("memory_used_percent")),
        "swap_total_bytes": memory.get("swap_total_bytes"),
        "swap_used_bytes": memory.get("swap_used_bytes"),
        "swap_used_percent": _rounded(swap_percent),
        **disk,
        "uptime_seconds": _rounded(uptime_seconds, digits=0),
    }


def _hostname() -> str:
    try:
        uname = os.uname()
    except AttributeError:
        return os.environ.get("COMPUTERNAME") or "unknown"
    return uname.nodename or "unknown"


def _cpu_usage_percent_sample() -> float | None:
    first = _read_proc_stat_cpu()
    if first is None:
        return None
    time.sleep(0.05)
    second = _read_proc_stat_cpu()
    if second is None:
        return None
    total_delta = second[0] - first[0]
    idle_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, (1.0 - (idle_delta / total_delta)) * 100.0))


def _read_proc_stat_cpu() -> tuple[int, int] | None:
    path = Path("/proc/stat")
    try:
        line = path.read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return None
    parts = line.split()
    if not parts or parts[0] != "cpu":
        return None
    try:
        values = [int(value) for value in parts[1:]]
    except ValueError:
        return None
    if len(values) < 4:
        return None
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _memory_usage() -> dict[str, Any]:
    meminfo = _read_meminfo()
    total = meminfo.get("MemTotal")
    available = meminfo.get("MemAvailable")
    if total is None:
        return {}
    if available is None:
        available = meminfo.get("MemFree", 0) + meminfo.get("Buffers", 0) + meminfo.get("Cached", 0)
    used = max(0, total - available)
    swap_total = meminfo.get("SwapTotal", 0)
    swap_free = meminfo.get("SwapFree", 0)
    swap_used = max(0, swap_total - swap_free)
    return {
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "memory_used_bytes": used,
        "memory_used_percent": _percent(used, total),
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_used,
    }


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        parts = rest.strip().split()
        if not parts:
            continue
        try:
            kib = int(parts[0])
        except ValueError:
            continue
        values[key] = kib * 1024
    return values


def _disk_usage(root: Path) -> dict[str, Any]:
    disk_path = root if root.exists() else Path("/")
    try:
        usage = shutil.disk_usage(disk_path)
    except OSError:
        try:
            usage = shutil.disk_usage("/")
            disk_path = Path("/")
        except OSError:
            return {}
    used = max(0, usage.total - usage.free)
    return {
        "disk_path": str(disk_path),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "disk_used_bytes": used,
        "disk_used_percent": _percent(used, usage.total),
    }


def _uptime_seconds() -> float | None:
    try:
        raw = Path("/proc/uptime").read_text(encoding="utf-8").split()[0]
        return float(raw)
    except (OSError, IndexError, ValueError):
        return None


def _percent(numerator: Any, denominator: Any) -> float | None:
    try:
        top = float(numerator)
        bottom = float(denominator)
    except (TypeError, ValueError):
        return None
    if bottom <= 0:
        return None
    return max(0.0, (top / bottom) * 100.0)


def _rounded(value: Any, *, digits: int = 1) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return round(numeric, digits)


def _resource_status(value: Any, *, warning: float, critical: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"
    if not math.isfinite(numeric):
        return "UNKNOWN"
    if numeric >= critical:
        return "CRITICAL"
    if numeric >= warning:
        return "WARNING"
    return "OK"


def _web_v2_smoke_status(now: datetime) -> dict[str, Any]:
    path = _web_v2_smoke_status_path()
    required = _web_v2_smoke_status_required()
    max_age_seconds = _web_v2_smoke_max_age_seconds()
    base: dict[str, Any] = {
        "source_path": str(path),
        "required": required,
        "max_age_seconds": max_age_seconds,
    }
    if not path.exists():
        return {**base, "status": "missing", "ok": None, "age_seconds": None}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            **base,
            "status": "invalid",
            "ok": False,
            "age_seconds": None,
            "error": f"{type(exc).__name__}: {exc}",
        }
    if not isinstance(payload, dict):
        return {**base, "status": "invalid", "ok": False, "age_seconds": None}

    checked_at = _parse_dt(payload.get("checked_at"))
    age_seconds = None if checked_at is None else max(0, int((now - checked_at).total_seconds()))
    failures = payload.get("failures") if isinstance(payload.get("failures"), list) else []
    warnings = payload.get("warnings") if isinstance(payload.get("warnings"), list) else []
    ok = payload.get("ok") if isinstance(payload.get("ok"), bool) else None
    stale = age_seconds is None or age_seconds > max_age_seconds
    status = "ok"
    if ok is False or failures:
        status = "failing"
    elif stale:
        status = "stale"
    elif ok is None:
        status = "unknown"

    return {
        **base,
        "status": status,
        "ok": ok,
        "checked_at": _json_value(checked_at),
        "age_seconds": age_seconds,
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "failures": failures[:6],
        "warnings": warnings[:6],
        "snapshot": payload.get("snapshot") if isinstance(payload.get("snapshot"), dict) else {},
        "expert_pack": (
            payload.get("expert_pack") if isinstance(payload.get("expert_pack"), dict) else {}
        ),
        "deep_health": (
            payload.get("deep_health") if isinstance(payload.get("deep_health"), dict) else {}
        ),
    }


def _web_v2_smoke_action_issue(web_smoke: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(web_smoke, dict):
        return None
    status = str(web_smoke.get("status") or "")
    required = web_smoke.get("required") is True
    if status == "missing" and not required:
        return None
    if status in {"ok", ""}:
        return None
    if status == "failing":
        severity = "CRITICAL"
        summary = f"最近一次 Web V2/API smoke 失败，failures={web_smoke.get('failure_count') or 0}"
    elif status == "stale":
        severity = "WARNING"
        summary = (
            "Web V2/API smoke 结果已过期，"
            f"age={web_smoke.get('age_seconds')}s，阈值={web_smoke.get('max_age_seconds')}s"
        )
    elif status == "missing":
        severity = "WARNING"
        summary = "生产 API 要求 Web V2/API smoke 状态文件，但当前未找到 latest.json"
    else:
        severity = "WARNING"
        summary = f"Web V2/API smoke 状态不可用：{status}"
    return {"severity": severity, "summary": summary}


def _web_perf_payload(
    events: list[dict[str, Any]],
    api_metrics: dict[str, Any],
    web_smoke: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "cache_hit": sum(1 for row in events if row.get("cache_hit")),
        "cache_miss": sum(1 for row in events if row.get("cache_miss")),
        "rglob_fallback": sum(1 for row in events if row.get("rglob_fallback")),
        "recent_events": events[:20],
        "api_latency_ms": api_metrics.get("latency_ms", {}),
        "api_latency_by_path_ms": api_metrics.get("latency_by_path_ms", {}),
        "api_p50_ms": _metric_latency(api_metrics, "p50") or _web_latency(events, "p50"),
        "api_p95_ms": _metric_latency(api_metrics, "p95") or _web_latency(events, "p95"),
        "slow_paths": api_metrics.get("slow_paths", []),
        "api_metrics_window_minutes": api_metrics.get("window_minutes"),
        "web_v2_smoke": web_smoke or {},
    }


def _consumer_payload(consumers: dict[str, Any]) -> dict[str, Any]:
    fallback_rows = _frame_rows(consumers.get("fallback_rows"), limit=8)
    return {
        "permissions": consumers.get("permissions", {}),
        "permission_rows": _frame_rows(consumers.get("permission_rows"), limit=8),
        "fallback_rows": len(fallback_rows),
        "fallback_details": fallback_rows,
    }


def _exports_payload(exports: dict[str, Any]) -> dict[str, Any]:
    packs = _frame_rows(exports.get("packs"), limit=5)
    for row in packs:
        name = str(row.get("name") or Path(str(row.get("path") or "")).name)
        if _is_expert_pack_name(name):
            row["name"] = name
            row["download_url"] = f"/web-v2/expert-pack/download/{name}"
    manifest = (
        exports.get("manifest_summary")
        if isinstance(exports.get("manifest_summary"), dict)
        else {}
    )
    data_quality = (
        exports.get("data_quality_summary")
        if isinstance(exports.get("data_quality_summary"), dict)
        else {}
    )
    questions = (
        exports.get("expert_questions")
        if isinstance(exports.get("expert_questions"), list)
        else []
    )
    latest_pack = exports.get("latest_pack")
    latest_name = Path(str(latest_pack)).name if latest_pack else ""
    available_pack = exports.get("available_pack")
    available_name = Path(str(available_pack)).name if available_pack else ""
    manual_state = str(exports.get("manual_state") or "").strip()
    v5_bundle_name, v5_bundle_ts = _latest_export_pack_v5_bundle_metadata(exports)
    v5_attachment = _latest_export_pack_v5_attachment_metadata(exports)
    code_lag = _latest_export_pack_code_lag_metadata(exports)
    return {
        "latest_pack": exports.get("latest_pack"),
        "latest_pack_name": latest_name if _is_expert_pack_name(latest_name) else None,
        "latest_pack_source": exports.get("latest_pack_source"),
        "latest_pack_v5_bundle_name": v5_bundle_name,
        "latest_pack_v5_bundle_ts": _json_value(v5_bundle_ts),
        "latest_pack_v5_bundle_sha256": v5_attachment.get("selected_sha"),
        "latest_pack_embedded_v5_bundle_present": v5_attachment.get("embedded_present"),
        "latest_pack_embedded_v5_bundle_member_path": v5_attachment.get("embedded_member"),
        "latest_pack_embedded_v5_bundle_sha256": v5_attachment.get("embedded_sha"),
        "latest_pack_embedded_v5_bundle_matches_selected": v5_attachment.get(
            "embedded_matches_selected"
        ),
        "latest_pack_quant_lab_git_commit": code_lag.get("pack_git_commit"),
        "current_quant_lab_git_commit": code_lag.get("current_git_commit"),
        "latest_pack_matches_current_quant_lab_commit": code_lag.get("matches_current"),
        "latest_pack_code_lag_status": code_lag.get("code_lag_status"),
        "latest_download_url": (
            f"/web-v2/expert-pack/download/{latest_name}"
            if _is_expert_pack_name(latest_name)
            else None
        ),
        "available_pack": available_pack,
        "available_pack_name": (
            available_name if _is_expert_pack_name(available_name) else None
        ),
        "available_download_url": (
            f"/web-v2/expert-pack/download/{available_name}"
            if _is_expert_pack_name(available_name)
            else None
        ),
        "pack_count": len(packs),
        "packs": packs,
        "manifest_summary": _json_value(manifest),
        "data_quality_summary": _json_value(data_quality),
        "manifest_status": manifest.get("status") or "not_observable",
        "data_quality_status": data_quality.get("status") or "not_observable",
        "data_quality_warning_count": _quality_warning_count(data_quality),
        "expert_question_count": len(questions),
        "expert_questions": [_json_value(line) for line in questions[:8]],
        "manual_state": manual_state or None,
        "manual_status": _json_value(exports.get("manual_status") or {}),
        "export_date": exports.get("export_date"),
        "job_state": "pack_available" if latest_pack else (manual_state or "manual_missing"),
    }


def _is_expert_pack_name(value: str) -> bool:
    return value.startswith("quant_lab_expert_pack_") and value.endswith(".zip")


def _latest_export_pack_v5_bundle_metadata(
    exports: dict[str, Any],
) -> tuple[str | None, datetime | None]:
    row = _latest_export_pack_row(exports)
    if row is None:
        return None, None
    bundle_name = (
        row.get("selected_v5_bundle_manifest_bundle_name")
        or row.get("selected_v5_bundle_name")
        or row.get("v5_bundle_name")
        or row.get("selected_v5_bundle")
    )
    if not bundle_name:
        manifest = exports.get("manifest_summary")
        if isinstance(manifest, dict):
            bundle_name = (
                manifest.get("selected_v5_bundle_manifest_bundle_name")
                or manifest.get("selected_v5_bundle_name")
                or manifest.get("selected_v5_bundle")
            )
    bundle_name_text = str(bundle_name or "").strip()
    if not bundle_name_text:
        return None, None
    return bundle_name_text, _parse_v5_bundle_name_ts(bundle_name_text)


def _latest_export_pack_v5_attachment_metadata(exports: dict[str, Any]) -> dict[str, Any]:
    row = _latest_export_pack_row(exports) or {}
    manifest = exports.get("manifest_summary")
    if not isinstance(manifest, dict):
        manifest = {}
    selected_sha = _first_present(
        row.get("selected_v5_bundle_sha256"),
        row.get("selected_v5_bundle_manifest_bundle_sha256"),
        row.get("embedded_v5_bundle_source_sha256"),
        manifest.get("selected_v5_bundle_sha256"),
        manifest.get("selected_v5_bundle_manifest_bundle_sha256"),
        manifest.get("embedded_v5_bundle_source_sha256"),
    )
    embedded_present = _first_present(
        row.get("embedded_v5_bundle_present"),
        manifest.get("embedded_v5_bundle_present"),
    )
    embedded_member = _first_present(
        row.get("embedded_v5_bundle_member_path"),
        manifest.get("embedded_v5_bundle_member_path"),
    )
    embedded_sha = _first_present(
        row.get("embedded_v5_bundle_sha256"),
        manifest.get("embedded_v5_bundle_sha256"),
    )
    embedded_matches = _first_present(
        row.get("embedded_v5_bundle_matches_selected"),
        manifest.get("embedded_v5_bundle_matches_selected"),
    )
    return {
        "selected_sha": str(selected_sha).strip() if selected_sha not in (None, "") else None,
        "embedded_present": _bool_or_none(embedded_present),
        "embedded_member": (
            str(embedded_member).strip() if embedded_member not in (None, "") else None
        ),
        "embedded_sha": str(embedded_sha).strip() if embedded_sha not in (None, "") else None,
        "embedded_matches_selected": _bool_or_none(embedded_matches),
    }


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _latest_export_pack_code_lag_metadata(exports: dict[str, Any]) -> dict[str, Any]:
    if not (exports.get("latest_pack") or exports.get("display_pack")):
        return {}
    manifest = exports.get("manifest_summary")
    if not isinstance(manifest, dict):
        manifest = {}
    pack_commit = str(manifest.get("git_commit") or "").strip()
    current_commit = _current_git_commit()
    matches_current = _git_commits_match(pack_commit, current_commit or "")
    return {
        "pack_git_commit": pack_commit or None,
        "current_git_commit": current_commit,
        "matches_current": matches_current,
        "code_lag_status": _pack_code_lag_status(matches_current),
    }


@lru_cache(maxsize=1)
def _current_git_commit() -> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = result.stdout.strip()
    return commit or None


def _git_commits_match(pack_commit: str, current_commit: str) -> bool | None:
    pack = str(pack_commit or "").strip()
    current = str(current_commit or "").strip()
    if not pack or not current:
        return None
    return pack.startswith(current) or current.startswith(pack)


def _pack_code_lag_status(matches_current: bool | None) -> str:
    if matches_current is True:
        return "OK"
    if matches_current is False:
        return "WARNING"
    return "UNKNOWN"


def _bool_or_none(value: Any) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "pass", "passed"}:
        return True
    if text in {"0", "false", "no", "n", "off", "fail", "failed"}:
        return False
    return bool(value)


def _latest_export_pack_row(exports: dict[str, Any]) -> dict[str, Any] | None:
    rows = _frame_rows(exports.get("packs"), limit=24)
    if not rows:
        return None
    latest_name = Path(
        str(exports.get("latest_pack") or exports.get("display_pack") or "")
    ).name
    if not latest_name:
        return None
    for row in rows:
        row_name = str(row.get("name") or Path(str(row.get("path") or "")).name)
        if row_name == latest_name:
            return row
    return None


def _quality_warning_count(data_quality: dict[str, Any]) -> int:
    for key in ["warning_count", "warnings_count", "data_quality_warning_count"]:
        value = _int(data_quality.get(key))
        if value is not None:
            return value
    warnings = data_quality.get("warnings")
    return len(warnings) if isinstance(warnings, list) else 0


def _build_actions(
    overview: dict[str, Any],
    data_health: dict[str, Any],
    cost: dict[str, Any],
    v5: dict[str, Any],
    web_events: list[dict[str, Any]],
    exports: dict[str, Any],
    legacy_anomalies: dict[str, Any] | None = None,
    data_matrix: dict[str, Any] | None = None,
    web_smoke: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    web_smoke_issue = _web_v2_smoke_action_issue(web_smoke)
    if web_smoke_issue is not None:
        actions.append(
            _action(
                str(web_smoke_issue["severity"]),
                "Web V2/API 巡检异常",
                str(web_smoke_issue["summary"]),
                "web_v2_smoke",
                "检查 quant-lab-web-v2-smoke.timer/service journal 与 latest.json 写入状态",
                "/data-ops",
            )
        )
    if data_matrix is not None:
        matrix_issue = _data_matrix_issue_summary(data_matrix)
        critical = int(matrix_issue.get("critical") or 0)
        warning = int(matrix_issue.get("warning") or 0)
        if critical or warning:
            title, summary, next_action = _data_matrix_action_text(matrix_issue)
            actions.append(
                _action(
                    "WARNING",
                    title,
                    summary,
                    "data_matrix",
                    next_action,
                    "/data",
                )
            )
    if (_float(cost.get("hard_fallback_ratio")) or 0.0) > 0.25:
        actions.append(
            _action(
                "CRITICAL",
                "成本硬回退偏高",
                "存在 global_default 或成本服务缺口，不能当作 live-ready 证据",
                "cost_model_summary",
                "优先排查成本桶刷新、symbol 命中和 API 全局默认命中",
                "/cost",
            )
        )
    cost_probe_issue = _cost_probe_disagreement_issue(cost)
    if cost_probe_issue is not None:
        actions.append(
            _action(
                str(cost_probe_issue["severity"]),
                "成本探针对账不一致",
                str(cost_probe_issue["summary"]),
                "cost_probe_cost_disagreement",
                "复核 V5 roundtrip cost、quant-lab cost_bucket 与 OKX bill 净现金流口径",
                "/cost",
            )
        )
    if _cost_soft_fallback_requires_action(cost):
        actions.append(
            _action(
                "WARNING",
                "成本软回退偏高",
                "当前主要依赖 public spread proxy，只适合 paper/shadow 参考",
                "cost_model_summary",
                "补真实/混合成本样本，降低 proxy-only",
                "/cost",
            )
        )
    if _int(data_health.get("schema_violation_count")) or _int(
        data_health.get("unclosed_bar_count")
    ):
        actions.append(
            _action(
                "CRITICAL",
                "行情数据结构异常",
                "market_bar 有结构违规或未闭合 K 线",
                "data_health_summary",
                "先修复 market_bar，再解释策略证据",
                "/data-ops",
            )
        )
    market_issue = _market_bar_freshness_issue(data_health, datetime.now(UTC))
    if market_issue is not None:
        actions.append(
            _action(
                str(market_issue["severity"]),
                "行情 K 线延迟",
                (
                    f"market_bar 最新 {market_issue['latest_ts']}，"
                    f"close {market_issue.get('latest_close_ts')}，"
                    f"延迟 {market_issue['age_seconds']} 秒"
                ),
                "data_health_summary",
                "检查 OKX REST backfill timer 与 feature publish 是否及时刷新",
                "/data-ops",
            )
        )
    legacy_items = (
        legacy_anomalies.get("items")
        if isinstance(legacy_anomalies, dict) and isinstance(legacy_anomalies.get("items"), list)
        else []
    )
    for item in legacy_items[:2]:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "")
        if source in {
            "legacy_data_health.market_bar",
            "legacy_data_health.missing_bars",
        }:
            continue
        actions.append(
            _action(
                str(item.get("severity") or "WARNING"),
                str(item.get("title") or "旧页面异常"),
                str(item.get("summary") or "旧 Streamlit 页面发现异常"),
                source or "legacy_web",
                str(item.get("next_action") or "打开旧数据健康页查看明细"),
                "/data-ops",
            )
        )
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    if latest.get("kill_switch_enabled") is True or latest.get("reconcile_ok") is False:
        actions.append(
            _action(
                "CRITICAL",
                "V5 风控/对账异常",
                "kill-switch 或 reconcile 状态需要复核",
                "v5_telemetry_summary",
                "进入 V5 遥测页查看 bundle/reconcile/ledger",
                "/v5-consumers",
            )
        )
    if sum(1 for row in web_events if row.get("rglob_fallback")):
        actions.append(
            _action(
                "WARNING",
                "Web 触发 fallback glob",
                "文件索引缺失时 reader 会退回路径扫描",
                "web_perf",
                "刷新 lake_file_index，避免页面和导出变慢",
                "/data-ops",
            )
        )
    if str(exports.get("manual_state") or "").lower() == "failed":
        actions.append(
            _action(
                "WARNING",
                "专家包生成失败",
                "手动 expert pack 导出任务失败",
                "expert_export_summary",
                "进入专家包导出页查看失败原因并重新生成",
                "/exports",
            )
        )
    if not actions:
        actions.append(
            _action(
                "OK",
                "研究中台可读",
                "核心数据、成本和消费者摘要可展示",
                "bigscreen_snapshot",
                "继续观察策略机会与成本质量",
                "/strategy",
            )
        )
    return actions


def _action(
    severity: str,
    title: str,
    summary: str,
    source: str,
    next_action: str,
    drilldown: str,
) -> dict[str, str]:
    return {
        "severity": severity,
        "title": title,
        "summary": summary,
        "source": source,
        "next_action": next_action,
        "drilldown": drilldown,
    }


def _expert_pack_v5_lag_issue(
    exports: dict[str, Any],
    v5: dict[str, Any],
) -> dict[str, Any] | None:
    current_pack = exports.get("latest_pack") or exports.get("display_pack")
    if not current_pack:
        return None
    pack_bundle_name, pack_bundle_ts = _latest_export_pack_v5_bundle_metadata(exports)
    latest_v5_ts = _parse_dt(_latest_v5_bundle_ts({}, v5))
    if pack_bundle_ts is None or latest_v5_ts is None:
        return None
    lag_seconds = int((latest_v5_ts - pack_bundle_ts).total_seconds())
    if lag_seconds <= EXPERT_PACK_V5_LAG_WARNING_SECONDS:
        return None
    return {
        "pack_bundle_name": pack_bundle_name or "not_observable",
        "pack_bundle_ts": _json_value(pack_bundle_ts),
        "latest_v5_bundle_ts": _json_value(latest_v5_ts),
        "lag_seconds": lag_seconds,
        "lag_minutes": max(1, round(lag_seconds / 60)),
    }


def _expert_pack_v5_lag_status_payload(
    exports: dict[str, Any],
    v5: dict[str, Any],
) -> dict[str, Any]:
    current_pack = exports.get("latest_pack") or exports.get("display_pack")
    if not current_pack:
        return {}
    _pack_bundle_name, pack_bundle_ts = _latest_export_pack_v5_bundle_metadata(exports)
    latest_v5_ts = _parse_dt(_latest_v5_bundle_ts({}, v5))
    if pack_bundle_ts is None or latest_v5_ts is None:
        return {}
    lag_seconds = int((latest_v5_ts - pack_bundle_ts).total_seconds())
    bounded_lag_seconds = max(0, lag_seconds)
    status = "OK" if lag_seconds <= EXPERT_PACK_V5_LAG_WARNING_SECONDS else "WARNING"
    return {
        "latest_pack_v5_lag_status": status,
        "latest_pack_v5_lag_seconds": bounded_lag_seconds,
        "latest_pack_v5_lag_minutes": max(0, round(bounded_lag_seconds / 60)),
        "latest_live_v5_bundle_ts": _json_value(latest_v5_ts),
        "latest_pack_v5_lag_pack_bundle_ts": _json_value(pack_bundle_ts),
    }


def _cost_status(
    row: dict[str, Any],
    cost: dict[str, Any],
    live_row: dict[str, Any] | None = None,
) -> str:
    if live_row:
        source = str(live_row.get("effective_cost_source") or "").lower()
        covered = _truthy(live_row.get("actual_or_mixed_covered"))
        coverage_status = str(live_row.get("coverage_status") or "").upper()
        if "global" in source:
            return "CRITICAL"
        if covered and ("actual" in source or "mixed" in source):
            return "OK"
        if coverage_status in {"WARNING", "FAIL", "CRITICAL"} or source == "missing":
            return "WARNING"
    source = str(row.get("cost_source") or row.get("source") or "").lower()
    fallback = str(row.get("fallback_level") or "").lower()
    if "global" in source or "global" in fallback:
        return "CRITICAL"
    if "actual" in source or "mixed" in source or "actual" in fallback:
        return "OK"
    if row or (_float(cost.get("soft_fallback_ratio")) or 0.0) > 0.0:
        return "WARNING"
    return "INFO"


def _matrix_cost_source(row: dict[str, Any], live_row: dict[str, Any]) -> Any:
    live_source = str(live_row.get("effective_cost_source") or "").strip()
    if live_source and live_source.lower() != "missing":
        return live_source
    return row.get("cost_source") or row.get("source")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def _spread_status(value: Any) -> str:
    spread = _float(value)
    if spread is None:
        return "INFO"
    if spread >= 6:
        return "CRITICAL"
    if spread >= 3:
        return "WARNING"
    return "OK"


def _spread_cell(value: Any) -> dict[str, Any]:
    spread = _float(value)
    status = _spread_status(value)
    cell: dict[str, Any] = {
        "status": status,
        "spread_bps": spread,
    }
    if spread is None:
        cell["reason"] = "spread_not_observable"
    elif status == "CRITICAL":
        cell["reason"] = f"spread_bps {spread:.2f} >= 6.00 critical threshold"
    elif status == "WARNING":
        cell["reason"] = f"spread_bps {spread:.2f} >= 3.00 warning threshold"
    else:
        cell["reason"] = "spread_bps below warning threshold"
    return cell


def _advisory_status(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "").upper()
    mode = str(row.get("recommended_mode") or "").lower()
    if decision == "KILL":
        return "INFO"
    if mode == "paper" or decision == "PAPER_READY":
        return "OK"
    if mode == "shadow" or "SHADOW" in decision:
        return "INFO"
    return "INFO" if row else "WARNING"


def _status_label(value: Any) -> str:
    text = str(value or "").upper()
    if text in {"OK", "RUNNING", "ALLOW", "FRESH"}:
        return "OK"
    if text in {"CRITICAL", "ABORT", "FAIL", "FAILED", "STALE"}:
        return "CRITICAL"
    if not text or text in {"UNKNOWN", "NONE"}:
        return "WARNING"
    return "WARNING"


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result
