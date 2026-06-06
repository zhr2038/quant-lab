from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.ops.api_metrics import api_metrics_summary
from quant_lab.symbols import normalize_symbol
from quant_lab.web import perf, readers

SNAPSHOT_CACHE_TTL_SECONDS = 20.0
_SNAPSHOT_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def bigscreen_snapshot(lake_root: str | Path) -> dict[str, Any]:
    root = Path(lake_root)
    cache_key = str(root.resolve())
    cached = _SNAPSHOT_CACHE.get(cache_key)
    now = time.monotonic()
    if cached is not None and now - cached[0] <= SNAPSHOT_CACHE_TTL_SECONDS:
        return cached[1]

    generated_at = datetime.now(UTC)
    data_health = _safe_summary("data_health_summary", readers.data_health_summary, root)
    cost = _safe_summary("cost_model_summary", readers.cost_model_summary, root)
    strategy = _safe_strategy_summary(root)
    market = _safe_summary("market_regime_summary", readers.market_regime_summary, root)
    collectors = _safe_summary("okx_collector_summary", readers.okx_collector_summary, root)
    v5 = _safe_summary("v5_telemetry_summary", readers.v5_telemetry_summary, root)
    consumers = _safe_summary(
        "strategy_consumer_summary",
        readers.strategy_consumer_summary,
        root,
    )
    exports = _safe_summary(
        "expert_export_summary",
        readers.expert_export_summary,
        readers.default_exports_root(root),
    )
    web_events = perf.recent_events(limit=50)
    api_metrics = _safe_api_metrics(root)
    overview = _overview_from_summaries(data_health, v5, consumers)

    warnings = _dedupe(
        [
            *_warnings(data_health),
            *_warnings(cost),
            *_warnings(strategy),
            *_warnings(market),
            *_warnings(collectors),
            *_warnings(v5),
            *_warnings(consumers),
            *_warnings(exports),
        ]
    )
    status = _status_from_inputs(overview, data_health, cost, v5, warnings)
    overview["status"] = status
    health_score = _health_score(status, data_health, cost, v5, web_events, exports)
    payload = {
        "generated_at": _json_value(generated_at),
        "lake_root": str(root),
        "mode": "read-only",
        "status": status,
        "health_score": health_score,
        "kpis": _kpis(generated_at, overview, collectors, cost, v5, web_events, api_metrics),
        "actions": _build_actions(overview, data_health, cost, v5, web_events, exports)[:8],
        "data_matrix": _data_matrix(market, collectors, cost, strategy, data_health, overview),
        "strategy_flow": _strategy_flow(strategy),
        "v5": _v5_payload(v5),
        "cost": _cost_payload(cost),
        "market": _market_payload(market),
        "collectors": _collector_payload(collectors),
        "data_health": _data_health_payload(data_health),
        "web_perf": _web_perf_payload(web_events, api_metrics),
        "consumers": _consumer_payload(consumers),
        "exports": _exports_payload(exports),
        "warnings": warnings[:30],
    }
    _SNAPSHOT_CACHE[cache_key] = (now, payload)
    return payload


def clear_bigscreen_cache() -> None:
    _SNAPSHOT_CACHE.clear()


def _safe_summary(name: str, fn: Any, *args: Any) -> dict[str, Any]:
    try:
        value = fn(*args)
    except Exception as exc:
        return {"warnings": [f"{name} failed: {exc}"], "status": "UNKNOWN"}
    return value if isinstance(value, dict) else {"warnings": [f"{name} returned non-dict"]}


def _safe_api_metrics(root: Path) -> dict[str, Any]:
    try:
        return api_metrics_summary(root)
    except Exception as exc:
        return {"warnings": [f"api_metrics_summary failed: {exc}"]}


def _safe_strategy_summary(root: Path) -> dict[str, Any]:
    warnings: list[str] = []
    frames: dict[str, pl.DataFrame] = {}
    for dataset_name, output_key in [
        ("strategy_opportunity_advisory", "strategy_opportunity_advisory"),
        ("alpha_discovery_board", "alpha_discovery_board"),
        ("alpha_factory_promotion_queue", "alpha_factory_promotion_queue"),
        ("risk_on_multi_buy_shadow", "risk_on_multi_buy_shadow"),
        ("research_portfolio_status", "research_portfolio_status"),
        ("factor_candidate", "factor_candidate"),
        ("factor_evidence", "factor_evidence"),
        ("factor_correlation_daily", "factor_correlation_daily"),
    ]:
        frame, warning = _read_display_frame(root, dataset_name)
        if frame.is_empty() and dataset_name == "risk_on_multi_buy_shadow":
            frame, warning = _read_display_frame(root, "v5_risk_on_multi_buy_shadow")
        frames[output_key] = frame
        if warning:
            warnings.append(warning)

    advisory = frames.get("strategy_opportunity_advisory", pl.DataFrame())
    discovery = frames.get("alpha_discovery_board", pl.DataFrame())
    if advisory.is_empty():
        warnings.append("strategy_opportunity_advisory 数据集缺失或为空")
    return {
        **frames,
        "counts": {},
        "strategy_counts": _count_by_column(advisory, "decision"),
        "discovery_counts": _count_by_column(discovery, "decision"),
        "warnings": warnings,
    }


def _read_display_frame(root: Path, dataset_name: str) -> tuple[pl.DataFrame, str | None]:
    try:
        frame, warning = readers._read_web_display_dataset_with_warning(root, dataset_name)
        return frame if isinstance(frame, pl.DataFrame) else pl.DataFrame(), warning
    except Exception as exc:
        return pl.DataFrame(), f"{dataset_name} 大屏抽样读取失败：{exc}"


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
    return {
        "status": "UNKNOWN",
        "v5_permission": permissions.get("v5", "UNKNOWN"),
        "v7_permission": permissions.get("v7", "UNKNOWN"),
        "latest_market_bar_ts": data_health.get("latest_market_bar_ts"),
        "latest_v5_bundle_ts": latest.get("latest_bundle_ts"),
        "diagnostics": {
            "latest_v5_bundle_ts": latest.get("latest_bundle_ts"),
        },
    }


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


def _status_from_inputs(
    overview: dict[str, Any],
    data_health: dict[str, Any],
    cost: dict[str, Any],
    v5: dict[str, Any],
    warnings: list[str],
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
    overview_status = str(overview.get("status") or "").upper()
    if warnings or overview_status == "WARNING":
        return "WARNING"
    if overview_status == "CRITICAL":
        return "CRITICAL"
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
    if not exports.get("latest_pack"):
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
    return {
        "platform_status": overview.get("status", "UNKNOWN"),
        "v5_permission": overview.get("v5_permission", "UNKNOWN"),
        "v7_permission": overview.get("v7_permission", "UNKNOWN"),
        "latest_market_bar_ts": _json_value(overview.get("latest_market_bar_ts")),
        "latest_v5_bundle_ts": _json_value(latest_bundle_ts),
        "market_delay_seconds": _age_seconds(generated_at, overview.get("latest_market_bar_ts")),
        "v5_bundle_delay_seconds": _age_seconds(generated_at, latest_bundle_ts),
        "cost_hard_fallback_ratio": _float(cost.get("hard_fallback_ratio")),
        "cost_soft_fallback_ratio": _float(cost.get("soft_fallback_ratio")),
        "ws_message_count": collectors.get("orderbook_snapshot_rows", 0),
        "trade_print_rows": collectors.get("trade_print_rows", 0),
        "orderbook_snapshot_rows": collectors.get("orderbook_snapshot_rows", 0),
        "api_p50_ms": _metric_latency(api_metrics, "p50") or _web_latency(web_events, "p50"),
        "api_p95_ms": _metric_latency(api_metrics, "p95") or _web_latency(web_events, "p95"),
    }


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


def _parse_dt(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


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
    advisory_by_symbol = _latest_by_symbol(
        _frame_rows(strategy.get("strategy_opportunity_advisory"), limit=80)
    )
    evidence_by_symbol = _latest_by_symbol(_frame_rows(strategy.get("strategy_evidence"), limit=80))

    market_status = "OK" if overview.get("latest_market_bar_ts") else "WARNING"
    ws_status = _status_label(collectors.get("okx_public_ws_status"))
    rows: list[dict[str, Any]] = []
    for symbol in symbols[:16]:
        spread = spread_by_symbol.get(symbol, {})
        trade = trade_by_symbol.get(symbol, {})
        regime = regime_by_symbol.get(symbol, {})
        cost_row = cost_by_symbol.get(symbol, {})
        advisory = advisory_by_symbol.get(symbol, {})
        evidence = evidence_by_symbol.get(symbol, {})
        rows.append(
            {
                "symbol": symbol,
                "market_bar": {
                    "status": market_status,
                    "freshness_seconds": _age_seconds(
                        datetime.now(UTC),
                        data_health.get("latest_market_bar_ts"),
                    ),
                    "latest_ts": _json_value(data_health.get("latest_market_bar_ts")),
                    "regime": regime.get("volatility_regime") or regime.get("regime"),
                },
                "ws": {
                    "status": ws_status,
                    "messages": collectors.get("orderbook_snapshot_rows"),
                },
                "spread": {
                    "status": _spread_status(spread.get("spread_bps")),
                    "spread_bps": _float(spread.get("spread_bps")),
                },
                "trade": {
                    "status": "OK" if _int(trade.get("trade_count")) else "WARNING",
                    "trade_count": _int(trade.get("trade_count")),
                },
                "cost": {
                    "status": _cost_status(cost_row, cost),
                    "source": cost_row.get("cost_source") or cost_row.get("source"),
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


def _matrix_symbols(
    market: dict[str, Any],
    cost: dict[str, Any],
    strategy: dict[str, Any],
) -> list[str]:
    symbols: list[str] = []
    for rows in [
        _frame_rows(market.get("regimes"), limit=80),
        _frame_rows(cost.get("costs"), limit=80),
        _frame_rows(strategy.get("strategy_opportunity_advisory"), limit=80),
        _frame_rows(strategy.get("alpha_discovery_board"), limit=80),
    ]:
        for row in rows:
            symbol = _normalize_display_symbol(row.get("symbol"))
            if symbol and symbol not in symbols:
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
    counts = _strategy_counts(strategy, advisory_rows)
    top = _top_strategy_candidates(advisory_rows)
    factor_factory = _factor_factory_payload(strategy)
    return {
        "counts": counts,
        "top_candidates": top[:8],
        "top_live_candidates": _top_live_candidates(top)[:5],
        "research_portfolio": _frame_rows(strategy.get("research_portfolio_status"), limit=8),
        "alpha_factory": _frame_rows(strategy.get("alpha_factory_promotion_queue"), limit=8),
        "risk_on_multi_buy": _frame_rows(strategy.get("risk_on_multi_buy_shadow"), limit=6),
        "factor_factory": factor_factory,
        "advisory_fresh": True,
    }


def _factor_factory_payload(strategy: dict[str, Any]) -> dict[str, Any]:
    candidates = _as_frame(strategy.get("factor_candidate"))
    evidence = _as_frame(strategy.get("factor_evidence"))
    correlations = _as_frame(strategy.get("factor_correlation_daily"))
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
        "high_correlation_pair_count": len(high_correlation_pairs),
        "state_counts": state_counts,
        "latest_candidate_created_at": _max_column_value(candidates, "created_at"),
        "latest_evidence_created_at": _max_column_value(evidence, "created_at"),
        "top_candidates": candidate_rows[:8],
        "paper_ready_candidates": paper_ready_candidates[:8],
        "evidence_by_horizon": evidence_by_horizon,
        "high_correlation_pairs": high_correlation_pairs[:8],
        "warnings": warnings,
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
    if rows:
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
    raw = strategy.get("strategy_counts") or strategy.get("alpha_discovery_counts") or {}
    if isinstance(raw, dict):
        return {
            "research": _int(raw.get("RESEARCH_ONLY") or raw.get("research")) or 0,
            "shadow": _int(raw.get("KEEP_SHADOW") or raw.get("REGIME_SHADOW") or raw.get("shadow"))
            or 0,
            "paper": _int(raw.get("PAPER_READY") or raw.get("paper")) or 0,
            "kill": _int(raw.get("KILL") or raw.get("kill")) or 0,
        }
    return {"research": 0, "shadow": 0, "paper": 0, "kill": 0}


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

    sorted_rows = sorted(rows, key=priority, reverse=True)
    return [_strategy_candidate_payload(row) for row in sorted_rows]


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


def _v5_payload(v5: dict[str, Any]) -> dict[str, Any]:
    latest = v5.get("latest") if isinstance(v5.get("latest"), dict) else {}
    payload = {str(key): _json_value(value) for key, value in latest.items()}
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
    ]
    payload = {key: _json_value(cost.get(key)) for key in keys}
    payload["cost_rows"] = _frame_rows(cost.get("costs"), limit=12)
    payload["cost_health"] = _frame_rows(cost.get("cost_health"), limit=5)
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


def _data_health_payload(data_health: dict[str, Any]) -> dict[str, Any]:
    return {
        "duplicate_bar_count": _int(data_health.get("duplicate_bar_count")) or 0,
        "unclosed_bar_count": _int(data_health.get("unclosed_bar_count")) or 0,
        "schema_violation_count": _int(data_health.get("schema_violation_count")) or 0,
        "missing_bar_ratio": _float(data_health.get("missing_bar_ratio")) or 0.0,
        "latest_market_bar_ts": _json_value(data_health.get("latest_market_bar_ts")),
        "stale_dataset_count": len(_frame_rows(data_health.get("stale_datasets"), limit=1000)),
        "stale_datasets": _frame_rows(data_health.get("stale_datasets"), limit=8),
        "latest_per_symbol": _frame_rows(data_health.get("latest_per_symbol"), limit=8),
    }


def _web_perf_payload(events: list[dict[str, Any]], api_metrics: dict[str, Any]) -> dict[str, Any]:
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
    return {
        "latest_pack": exports.get("latest_pack"),
        "latest_download_url": (
            f"/web-v2/expert-pack/download/{latest_name}"
            if _is_expert_pack_name(latest_name)
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
        "job_state": "pack_available" if latest_pack else "missing",
    }


def _is_expert_pack_name(value: str) -> bool:
    return value.startswith("quant_lab_expert_pack_") and value.endswith(".zip")


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
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
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
    if (_float(cost.get("soft_fallback_ratio")) or 0.0) > 0.8:
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
    if not exports.get("latest_pack"):
        actions.append(
            _action(
                "WARNING",
                "专家包缺失",
                "未找到最新 expert pack",
                "expert_export_summary",
                "进入专家包页面生成或检查导出任务",
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


def _cost_status(row: dict[str, Any], cost: dict[str, Any]) -> str:
    source = str(row.get("cost_source") or row.get("source") or "").lower()
    fallback = str(row.get("fallback_level") or "").lower()
    if "global" in source or "global" in fallback:
        return "CRITICAL"
    if "actual" in source or "mixed" in source or "actual" in fallback:
        return "OK"
    if row or (_float(cost.get("soft_fallback_ratio")) or 0.0) > 0.0:
        return "WARNING"
    return "INFO"


def _spread_status(value: Any) -> str:
    spread = _float(value)
    if spread is None:
        return "INFO"
    if spread >= 6:
        return "CRITICAL"
    if spread >= 3:
        return "WARNING"
    return "OK"


def _advisory_status(row: dict[str, Any]) -> str:
    decision = str(row.get("decision") or "").upper()
    mode = str(row.get("recommended_mode") or "").lower()
    if decision == "KILL":
        return "CRITICAL"
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
