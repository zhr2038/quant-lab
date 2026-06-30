from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import RiskPermission
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.costs.model import (
    _row_is_stale,
    estimate_cost_from_cost_bucket_daily_rows,
    evaluate_live_universe_cost_coverage,
)
from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy
from quant_lab.risk.publish import is_permission_status_enforceable, parse_risk_permission_row
from quant_lab.symbols import normalize_symbol

DEFAULT_COST_SYMBOLS = ["BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"]
LIVE_UNIVERSE_SYMBOLS = DEFAULT_COST_SYMBOLS
ENFORCE_READINESS_JSON = "v5_enforce_readiness.json"
ENFORCE_READINESS_CSV = "v5_enforce_readiness.csv"
ACTUAL_OR_MIXED_COST_SOURCES = {
    "actual_fills",
    "actual_okx_fills_and_bills",
    "actual_okx_fills_fee_missing",
    "mixed_actual_proxy",
}
PUBLIC_PROXY_COST_SOURCES = {"public_spread_proxy", "public_proxy"}
ENTRY_READINESS_REASONS = {
    "cost_api_global_default_rate",
    "cost_live_symbol_hit_rate",
    "actual_or_mixed_cost_coverage_live_universe",
    "fallback_rate",
    "alpha_gate_status_not_dead",
}
SCALE_READINESS_REASONS = {
    "cost_api_global_default_rate",
    "cost_symbol_hit_rate",
    "cost_live_symbol_hit_rate",
    "actual_or_mixed_cost_coverage",
    "actual_or_mixed_cost_coverage_research_universe",
    "actual_or_mixed_cost_coverage_live_universe",
    "actual_or_mixed_cost_coverage_expanded_universe",
    "fallback_rate",
    "alpha_gate_status_not_dead",
}


class EnforceReadinessThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cost_api_global_default_rate_max: float = Field(default=0.0, ge=0, le=1)
    cost_symbol_hit_rate_min: float = Field(default=0.95, ge=0, le=1)
    actual_or_mixed_cost_coverage_min: float = Field(default=0.50, ge=0, le=1)
    telemetry_duplicate_rate_warn: float = Field(default=0.10, ge=0, le=1)
    telemetry_event_key_coverage_min: float = Field(default=0.95, ge=0, le=1)
    fallback_rate_max: float = Field(default=0.05, ge=0, le=1)


class EnforceReadinessCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    status: str
    passed: bool
    value: Any = None
    threshold: Any = None
    detail: str = ""


class EnforceReadinessReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str = "v5"
    version: str = "5.0.0"
    readiness_status: str
    veto_status: str = "VETO_READY"
    entry_status: str = "ENTRY_READY"
    scale_status: str = "SCALE_READY"
    shadow_only_recommended: bool
    blocked_reasons: list[str]
    warning_reasons: list[str]
    veto_blocked_reasons: list[str] = Field(default_factory=list)
    entry_blocked_reasons: list[str] = Field(default_factory=list)
    scale_blocked_reasons: list[str] = Field(default_factory=list)
    required_actions: list[str]
    as_of_ts: datetime
    contract_version: str = V5_QUANT_LAB_CONTRACT_VERSION
    checks: list[EnforceReadinessCheck]
    metrics: dict[str, Any]


def build_enforce_readiness_report(
    lake_root: str | Path,
    *,
    strategy: str = "v5",
    version: str = "5.0.0",
    thresholds: EnforceReadinessThresholds | None = None,
    cost_symbols: list[str] | None = None,
    cost_regime: str = "Trending",
) -> EnforceReadinessReport:
    root = Path(lake_root)
    limits = thresholds or EnforceReadinessThresholds()
    now = datetime.now(UTC)

    cost_rows = _rows(root / "gold" / "cost_bucket_daily")
    live_symbols, live_universe_source = _live_universe_symbols(root)
    symbols = _cost_symbols(cost_rows, cost_symbols, live_symbols=live_symbols)
    expanded_symbols = _expanded_universe_symbols(root, cost_rows, live_symbols)
    cost_metrics = _cost_api_metrics(
        cost_rows=cost_rows,
        symbols=symbols,
        live_symbols=live_symbols,
        live_universe_source=live_universe_source,
        expanded_symbols=expanded_symbols,
        cost_regime=cost_regime,
    )

    risk_rows = _rows(root / "gold" / "risk_permission")
    risk_metrics = _risk_metrics(
        risk_rows=risk_rows,
        strategy=strategy,
        version=version,
        now=now,
    )
    strategy_health = _latest_row(root / "gold" / "strategy_health_daily")
    telemetry_metrics = _telemetry_metrics(root, strategy_health, limits)
    gate_metrics = _gate_metrics(root)

    checks: list[EnforceReadinessCheck] = [
        _check(
            "risk_permission_fresh",
            risk_metrics["risk_permission_fresh"],
            value=risk_metrics["permission_status"],
            detail=risk_metrics["detail"],
        ),
        _check(
            "risk_permission_api_consistent_with_gold",
            risk_metrics["permission_api_consistent_with_gold"],
            value=risk_metrics["permission_api_lag_sec"],
            detail=risk_metrics["consistency_detail"],
        ),
        _check(
            "cost_api_global_default_rate",
            cost_metrics["cost_api_global_default_rate"]
            <= limits.cost_api_global_default_rate_max,
            value=cost_metrics["cost_api_global_default_rate"],
            threshold=f"<= {limits.cost_api_global_default_rate_max}",
            detail=cost_metrics["cost_detail"],
        ),
        _check(
            "cost_symbol_hit_rate",
            cost_metrics["cost_symbol_hit_rate"] >= limits.cost_symbol_hit_rate_min,
            value=cost_metrics["cost_symbol_hit_rate"],
            threshold=f">= {limits.cost_symbol_hit_rate_min}",
            detail=cost_metrics["cost_detail"],
            warn_only=True,
        ),
        _check(
            "cost_live_symbol_hit_rate",
            cost_metrics["cost_live_symbol_hit_rate"] >= limits.cost_symbol_hit_rate_min,
            value=cost_metrics["cost_live_symbol_hit_rate"],
            threshold=f">= {limits.cost_symbol_hit_rate_min}",
            detail=cost_metrics["live_cost_detail"],
        ),
        _check(
            "actual_or_mixed_cost_coverage_research_universe",
            cost_metrics["actual_or_mixed_cost_coverage_research_universe"]
            >= limits.actual_or_mixed_cost_coverage_min,
            value=cost_metrics["actual_or_mixed_cost_coverage_research_universe"],
            threshold=f">= {limits.actual_or_mixed_cost_coverage_min}",
            detail=cost_metrics["research_coverage_detail"],
            warn_only=True,
        ),
        _check(
            "actual_or_mixed_cost_coverage_live_universe",
            cost_metrics["actual_or_mixed_cost_coverage_live_universe"]
            >= limits.actual_or_mixed_cost_coverage_min,
            value=cost_metrics["actual_or_mixed_cost_coverage_live_universe"],
            threshold=f">= {limits.actual_or_mixed_cost_coverage_min}",
            detail=cost_metrics["live_coverage_detail"],
        ),
        _check(
            "telemetry_dedupe_health",
            telemetry_metrics["dedupe_health_status"] == "PASS",
            value=telemetry_metrics["dedupe_health_status"],
            threshold="not BLOCKED",
            detail=telemetry_metrics["detail"],
            warn_only=telemetry_metrics["dedupe_health_status"] == "WARN",
        ),
        _check(
            "fallback_rate",
            telemetry_metrics["fallback_rate"] <= limits.fallback_rate_max,
            value=telemetry_metrics["fallback_rate"],
            threshold=f"<= {limits.fallback_rate_max}",
            detail=telemetry_metrics["detail"],
        ),
        _check(
            "alpha_gate_status_not_dead",
            not gate_metrics["has_dead_gate"],
            value=gate_metrics["gate_status_counts"],
            detail=gate_metrics["detail"],
        ),
        _check(
            "decision_audit_present",
            telemetry_metrics["decision_audit_present"],
            value=telemetry_metrics["decision_audit_count"],
            detail=telemetry_metrics["detail"],
        ),
    ]

    blocked = [check.name for check in checks if check.status == "BLOCKED"]
    warnings = [check.name for check in checks if check.status == "WARN"]
    read_only_cost_advisory = (
        set(blocked) == {"actual_or_mixed_cost_coverage_live_universe"}
        and _read_only_permission_context(risk_metrics)
    )
    status = "READY"
    if read_only_cost_advisory:
        warnings = sorted({*warnings, *blocked})
        blocked = []
        status = "ADVISORY_READY"
    elif blocked:
        status = "BLOCKED"
    elif warnings:
        status = "WARN"

    metrics = {
        **risk_metrics,
        **cost_metrics,
        **telemetry_metrics,
        **gate_metrics,
    }
    split = _readiness_split(blocked, warnings)
    return EnforceReadinessReport(
        strategy=strategy,
        version=version,
        readiness_status=status,
        veto_status=split["veto_status"],
        entry_status=split["entry_status"],
        scale_status=split["scale_status"],
        shadow_only_recommended=status != "READY",
        blocked_reasons=blocked,
        warning_reasons=warnings,
        veto_blocked_reasons=split["veto_blocked_reasons"],
        entry_blocked_reasons=split["entry_blocked_reasons"],
        scale_blocked_reasons=split["scale_blocked_reasons"],
        required_actions=_required_actions(blocked, warnings),
        as_of_ts=now,
        checks=checks,
        metrics=metrics,
    )


def _readiness_split(blocked: list[str], warnings: list[str]) -> dict[str, Any]:
    blocked_set = {str(reason).strip() for reason in blocked if str(reason).strip()}
    warning_set = {str(reason).strip() for reason in warnings if str(reason).strip()}
    entry_blocked = sorted((blocked_set | warning_set) & ENTRY_READINESS_REASONS)
    scale_blocked = sorted((blocked_set | warning_set) & SCALE_READINESS_REASONS)
    veto_blocked = sorted(blocked_set - ENTRY_READINESS_REASONS)
    return {
        "veto_status": "VETO_BLOCKED" if veto_blocked else "VETO_READY",
        "entry_status": "ENTRY_BLOCKED" if entry_blocked else "ENTRY_READY",
        "scale_status": "SCALE_BLOCKED" if scale_blocked else "SCALE_READY",
        "veto_blocked_reasons": veto_blocked,
        "entry_blocked_reasons": entry_blocked,
        "scale_blocked_reasons": scale_blocked,
    }


def _read_only_permission_context(risk_metrics: dict[str, Any]) -> bool:
    status = str(risk_metrics.get("permission_status") or "").strip().upper()
    return status in {"ACTIVE_ABORT", "ACTIVE_SELL_ONLY"}


def write_enforce_readiness_report(
    lake_root: str | Path,
    *,
    out_dir: str | Path | None = None,
    strategy: str = "v5",
    version: str = "5.0.0",
    thresholds: EnforceReadinessThresholds | None = None,
) -> EnforceReadinessReport:
    report = build_enforce_readiness_report(
        lake_root,
        strategy=strategy,
        version=version,
        thresholds=thresholds,
    )
    target = Path(out_dir) if out_dir is not None else Path(lake_root) / "reports"
    target.mkdir(parents=True, exist_ok=True)
    (target / ENFORCE_READINESS_JSON).write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(target / ENFORCE_READINESS_CSV, report)
    return report


def enforce_readiness_members(
    lake_root: str | Path,
    *,
    strategy: str = "v5",
    version: str = "5.0.0",
) -> dict[str, str]:
    report = build_enforce_readiness_report(
        lake_root,
        strategy=strategy,
        version=version,
    )
    return {
        f"reports/{ENFORCE_READINESS_JSON}": json.dumps(
            report.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
        ),
        f"reports/{ENFORCE_READINESS_CSV}": _csv_text(report),
    }


def _rows(dataset_path: Path) -> list[dict[str, Any]]:
    try:
        frame = read_parquet_dataset(dataset_path)
    except Exception:
        return []
    return [] if frame.is_empty() else frame.to_dicts()


def _latest_row(dataset_path: Path) -> dict[str, Any]:
    rows = _rows(dataset_path)
    if not rows:
        return {}
    return max(rows, key=_row_sort_ts)


def _cost_symbols(
    rows: list[dict[str, Any]],
    explicit: list[str] | None,
    *,
    live_symbols: list[str],
) -> list[str]:
    if explicit:
        return sorted({normalize_symbol(symbol) for symbol in explicit})
    symbols = {
        normalize_symbol(row.get("symbol"))
        for row in rows
        if str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    symbols.update(_normalized_symbol_set(live_symbols))
    return sorted(symbols) or DEFAULT_COST_SYMBOLS


def _live_universe_symbols(root: Path) -> tuple[list[str], str]:
    configured = _live_universe_from_effective_config(root)
    if configured:
        return configured, "effective_live_config"
    telemetry_symbols = _live_universe_from_candidate_events(root)
    if telemetry_symbols:
        return telemetry_symbols, "v5_candidate_event"
    return sorted({normalize_symbol(symbol) for symbol in LIVE_UNIVERSE_SYMBOLS}), (
        "default_fallback"
    )


def _live_universe_from_effective_config(root: Path) -> list[str]:
    rows = _rows(root / "silver" / "v5_state_snapshot")
    candidates: list[str] = []
    for row in rows:
        state_type = str(row.get("state_type") or "").lower()
        source_path = str(row.get("source_path_inside_bundle") or "").lower()
        if "effective_live_config" not in state_type and "effective_live_config" not in source_path:
            continue
        payload = _json_dict(row.get("raw_payload_json"))
        for key in [
            "symbols",
            "live_symbols",
            "trade_symbols",
            "symbol_universe",
            "allowed_symbols",
        ]:
            candidates.extend(_symbols_from_value(payload.get(key)))
        if "universe" in payload:
            candidates.extend(_symbols_from_value(payload.get("universe")))
    return _live_symbols_if_complete(candidates)


def _live_universe_from_candidate_events(root: Path) -> list[str]:
    rows = _rows(root / "silver" / "v5_candidate_event")
    if not rows:
        return []
    latest_ts = max((_parse_dt(row.get("ts_utc")) for row in rows), default=None)
    if latest_ts is None:
        return []
    recent_cutoff = latest_ts - timedelta(hours=48)
    symbols = [
        normalize_symbol(row.get("symbol"))
        for row in rows
        if (_parse_dt(row.get("ts_utc")) or latest_ts) >= recent_cutoff
        and normalize_symbol(row.get("symbol")) in set(DEFAULT_COST_SYMBOLS)
    ]
    return _live_symbols_if_complete(symbols)


def _live_symbols_if_complete(symbols: list[str]) -> list[str]:
    normalized = sorted(
        {normalize_symbol(symbol) for symbol in symbols if normalize_symbol(symbol)}
    )
    default_set = set(DEFAULT_COST_SYMBOLS)
    return normalized if default_set.issubset(set(normalized)) else []


def _symbols_from_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_symbol(item) for item in value if normalize_symbol(item)]
    if isinstance(value, dict):
        symbols: list[str] = []
        for key in ["symbols", "live_symbols", "trade_symbols", "items"]:
            symbols.extend(_symbols_from_value(value.get(key)))
        return symbols
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [normalize_symbol(part) for part in text.replace(";", ",").split(",")]
        return _symbols_from_value(parsed)
    return []


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _cost_api_metrics(
    *,
    cost_rows: list[dict[str, Any]],
    symbols: list[str],
    live_symbols: list[str],
    live_universe_source: str,
    expanded_symbols: list[str],
    cost_regime: str,
) -> dict[str, Any]:
    estimates = []
    for symbol in symbols:
        estimates.append(
            estimate_cost_from_cost_bucket_daily_rows(
                symbol=symbol,
                regime=cost_regime,
                notional_usdt=1_000.0,
                quantile="p75",
                rows=cost_rows,
            ).model_dump(mode="json")
        )
    total = max(len(estimates), 1)
    global_default = [
        item for item in estimates if item.get("cost_source") == "global_default"
    ]
    symbol_hits = [item for item in estimates if item.get("cost_source") != "global_default"]
    hit_symbols = {
        normalize_symbol(item.get("normalized_symbol") or item.get("symbol"))
        for item in symbol_hits
        if normalize_symbol(item.get("normalized_symbol") or item.get("symbol"))
    }
    fresh_cost_rows = [row for row in cost_rows if not _row_is_stale(row)]
    stale_cost_rows = [row for row in cost_rows if _row_is_stale(row)]
    cost_symbols = {
        normalize_symbol(row.get("symbol"))
        for row in cost_rows
        if str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    fresh_cost_symbols = {
        normalize_symbol(row.get("symbol"))
        for row in fresh_cost_rows
        if str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    denominator_symbols = fresh_cost_symbols or cost_symbols
    actual_or_mixed = {
        normalize_symbol(row.get("symbol"))
        for row in fresh_cost_rows
        if _is_live_cost_coverage_row(row)
        and str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    stale_actual_or_mixed = {
        normalize_symbol(row.get("symbol"))
        for row in stale_cost_rows
        if _is_live_cost_coverage_row(row)
        and str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    proxy_only = {
        normalize_symbol(row.get("symbol"))
        for row in fresh_cost_rows
        if _cost_source(row) in PUBLIC_PROXY_COST_SOURCES
        and str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    } - actual_or_mixed
    live_symbol_set = _normalized_symbol_set(live_symbols)
    live_coverage = evaluate_live_universe_cost_coverage(
        pl.DataFrame(cost_rows) if cost_rows else pl.DataFrame(),
        live_symbols=sorted(live_symbol_set),
    )
    actual_or_mixed_live = set(live_coverage["covered_symbols"])
    proxy_only_live = set(live_coverage["proxy_only_symbols"])
    missing_actual_or_mixed_live = live_symbol_set - actual_or_mixed_live
    expanded_symbol_set = _normalized_symbol_set(expanded_symbols) - live_symbol_set
    actual_or_mixed_expanded = actual_or_mixed & expanded_symbol_set
    proxy_only_expanded = proxy_only & expanded_symbol_set
    live_cost_source_detail = _live_cost_source_detail(
        cost_rows=cost_rows,
        live_symbols=live_symbol_set,
    )
    stale_actual_or_mixed_live = {
        symbol
        for symbol, item in live_cost_source_detail.items()
        if item.get("latest_actual_or_mixed_source") != "missing"
        and item.get("latest_actual_or_mixed_stale") is True
    }
    missing_live_cost_symbols = live_symbol_set - hit_symbols
    coverage_denominator = max(len(denominator_symbols), 1)
    research_actual_or_mixed_coverage = len(actual_or_mixed) / coverage_denominator
    research_coverage_detail = (
        f"research_universe_actual_or_mixed_symbols={sorted(actual_or_mixed)}; "
        f"actual_or_mixed_live={sorted(actual_or_mixed_live)}; "
        f"actual_or_mixed_expanded={sorted(actual_or_mixed_expanded)}; "
        f"fresh_cost_symbols={sorted(fresh_cost_symbols)}; "
        f"proxy_only_symbols={sorted(proxy_only)}; "
        f"proxy_only_symbols_live={sorted(proxy_only_live)}; "
        f"proxy_only_symbols_expanded={sorted(proxy_only_expanded)}; "
        f"stale_actual_or_mixed_symbols={sorted(stale_actual_or_mixed)}"
    )
    return {
        "cost_symbols_checked": symbols,
        "cost_symbols_live_universe": sorted(live_symbol_set),
        "cost_symbols_expanded_universe": sorted(expanded_symbol_set),
        "live_universe_source": live_universe_source,
        "cost_api_global_default_count": len(global_default),
        "cost_api_global_default_rate": len(global_default) / total,
        "cost_symbol_hit_count": len(symbol_hits),
        "cost_symbol_hit_rate": len(symbol_hits) / total,
        "cost_live_symbol_hit_count": len(hit_symbols & live_symbol_set),
        "cost_live_symbol_hit_rate": _coverage_rate(hit_symbols & live_symbol_set, live_symbol_set),
        "missing_live_cost_symbols": sorted(missing_live_cost_symbols),
        "actual_or_mixed_cost_symbol_count": len(actual_or_mixed),
        "actual_or_mixed_cost_coverage": research_actual_or_mixed_coverage,
        "actual_or_mixed_cost_symbol_count_research_universe": len(actual_or_mixed),
        "actual_or_mixed_cost_coverage_research_universe": (
            research_actual_or_mixed_coverage
        ),
        "actual_or_mixed_cost_symbol_count_live_universe": len(actual_or_mixed_live),
        "actual_or_mixed_cost_coverage_live_universe": live_coverage["coverage_rate"],
        "actual_or_mixed_cost_symbol_count_expanded_universe": len(
            actual_or_mixed_expanded
        ),
        "actual_or_mixed_cost_coverage_expanded_universe": _coverage_rate(
            actual_or_mixed_expanded,
            expanded_symbol_set,
        ),
        "fresh_cost_symbol_count": len(fresh_cost_symbols),
        "fresh_cost_symbols": sorted(fresh_cost_symbols),
        "stale_actual_or_mixed_cost_symbols": sorted(stale_actual_or_mixed),
        "stale_actual_or_mixed_symbols_live": sorted(stale_actual_or_mixed_live),
        "proxy_only_cost_symbols": sorted(proxy_only),
        "proxy_only_symbols_live": sorted(proxy_only_live),
        "proxy_only_symbols_expanded": sorted(proxy_only_expanded),
        "missing_actual_or_mixed_symbols_live": sorted(missing_actual_or_mixed_live),
        "live_cost_source_detail": live_cost_source_detail,
        "cost_estimate_examples": estimates,
        "cost_detail": (
            f"global_default={len(global_default)}; symbol_hits={len(symbol_hits)}; "
            f"symbols={symbols}"
        ),
        "live_cost_detail": (
            f"live_symbols={sorted(live_symbol_set)}; "
            f"live_symbol_hits={sorted(hit_symbols & live_symbol_set)}; "
            f"missing_live_cost_symbols={sorted(missing_live_cost_symbols)}; "
            f"live_universe_source={live_universe_source}"
        ),
        "coverage_detail": research_coverage_detail,
        "research_coverage_detail": research_coverage_detail,
        "live_coverage_detail": (
            f"actual_or_mixed_live={sorted(actual_or_mixed_live)}; "
            f"proxy_only_symbols_live={sorted(proxy_only_live)}; "
            f"missing_live_cost_symbols={sorted(missing_live_cost_symbols)}; "
            f"missing_actual_or_mixed_symbols_live={sorted(missing_actual_or_mixed_live)}; "
            f"stale_actual_or_mixed_symbols_live={sorted(stale_actual_or_mixed_live)}; "
            f"live_universe_source={live_universe_source}"
        ),
    }


def _live_cost_source_detail(
    *,
    cost_rows: list[dict[str, Any]],
    live_symbols: set[str],
) -> dict[str, dict[str, Any]]:
    detail: dict[str, dict[str, Any]] = {}
    for symbol in sorted(live_symbols):
        symbol_rows = [
            row
            for row in cost_rows
            if normalize_symbol(row.get("symbol")) == symbol
        ]
        latest_row = _latest_cost_row(symbol_rows)
        latest_actual_or_mixed_row = _latest_cost_row(
            [
                row
                for row in symbol_rows
                if _is_live_cost_coverage_row(row)
            ]
        )
        latest_actual_or_mixed_ts = _cost_row_ts(latest_actual_or_mixed_row)
        latest_actual_or_mixed_age_sec = (
            (datetime.now(UTC) - latest_actual_or_mixed_ts).total_seconds()
            if latest_actual_or_mixed_ts is not None
            else None
        )
        detail[symbol] = {
            "latest_source": _cost_source(latest_row) if latest_row else "missing",
            "latest_created_at": _iso(_cost_row_ts(latest_row)),
            "latest_sample_count": _int(latest_row.get("sample_count")) if latest_row else 0,
            "latest_actual_or_mixed_source": (
                _cost_source(latest_actual_or_mixed_row)
                if latest_actual_or_mixed_row
                else "missing"
            ),
            "latest_actual_or_mixed_created_at": _iso(latest_actual_or_mixed_ts),
            "latest_actual_or_mixed_age_sec": latest_actual_or_mixed_age_sec,
            "latest_actual_or_mixed_stale": (
                _row_is_stale(latest_actual_or_mixed_row)
                if latest_actual_or_mixed_row
                else True
            ),
        }
    return detail


def _latest_cost_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: _cost_row_ts(row) or datetime.min.replace(tzinfo=UTC))


def _is_live_cost_coverage_row(row: dict[str, Any]) -> bool:
    if _cost_source(row) not in ACTUAL_OR_MIXED_COST_SOURCES:
        return False
    eligible = _bool_or_none(row.get("eligible_for_live_cost_coverage"))
    if eligible is False:
        return False
    origin_mix = str(row.get("sample_origin_mix") or "").strip().lower()
    if origin_mix == "cost_probe_only":
        return False
    cost_probe_count = _int(row.get("cost_probe_fill_count"))
    live_count = max(
        _int(row.get("strategy_live_fill_count")),
        _int(row.get("private_fill_count")),
        _int(row.get("actual_fill_count")),
        _int(row.get("mixed_fill_count")),
    )
    return not (cost_probe_count > 0 and live_count <= 0)


def _cost_row_ts(row: dict[str, Any] | None) -> datetime | None:
    if not row:
        return None
    return _parse_dt(row.get("created_at")) or _parse_dt(row.get("as_of_ts")) or _parse_dt(
        row.get("day")
    )


def _expanded_universe_symbols(
    root: Path,
    cost_rows: list[dict[str, Any]],
    live_symbols: list[str],
) -> list[str]:
    live_symbol_set = _normalized_symbol_set(live_symbols)
    symbols: set[str] = set()
    for dataset in [
        "expanded_universe_candidate",
        "expanded_universe_quality",
        "expanded_universe_watchlist",
        "expanded_universe_candidate_maturity",
        "expanded_universe_promotion_queue",
    ]:
        symbols.update(
            _row_symbols(_rows(root / "gold" / dataset), exclude=live_symbol_set)
        )
    if not symbols:
        symbols.update(_row_symbols(cost_rows, exclude=live_symbol_set))
    return sorted(symbols)


def _row_symbols(
    rows: list[dict[str, Any]],
    *,
    exclude: set[str] | None = None,
) -> set[str]:
    excluded = exclude or set()
    symbols: set[str] = set()
    for row in rows:
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol or symbol == "GLOBAL" or symbol in excluded:
            continue
        symbols.add(symbol)
    return symbols


def _normalized_symbol_set(symbols: list[str]) -> set[str]:
    return {symbol for symbol in (normalize_symbol(item) for item in symbols) if symbol}


def _coverage_rate(numerator: set[str], denominator: set[str]) -> float:
    return len(numerator) / len(denominator) if denominator else 0.0


def _cost_source(row: dict[str, Any]) -> str:
    return str(row.get("source") or row.get("cost_source") or "").lower()


def _risk_metrics(
    *,
    risk_rows: list[dict[str, Any]],
    strategy: str,
    version: str,
    now: datetime,
) -> dict[str, Any]:
    candidates = [
        row
        for row in risk_rows
        if str(row.get("strategy") or row.get("strategy_id") or "") == strategy
        and str(row.get("version") or "") == version
    ]
    if not candidates:
        return {
            "risk_permission_fresh": False,
            "permission_status": "NO_FRESH_PERMISSION",
            "permission_enforceable": False,
            "risk_permission_as_of_ts": None,
            "gold_permission_as_of_ts": None,
            "api_permission_as_of_ts": None,
            "permission_api_lag_sec": None,
            "permission_api_consistent_with_gold": False,
            "detail": "risk_permission rows missing for strategy/version",
            "consistency_detail": "gold_latest missing",
        }
    parsed = [_permission_from_row(row) for row in candidates]
    gold_latest = max(
        parsed,
        key=lambda item: (_dt_sort(item.as_of_ts), _dt_sort(item.source_bundle_ts)),
    )
    active = [
        permission
        for permission in parsed
        if str(permission.permission_status or "").startswith("ACTIVE_")
        and _expires_at_ok(permission.expires_at, now)
        and is_permission_status_enforceable(permission.permission_status)
    ]
    api_permission = max(
        active,
        key=lambda item: (_dt_sort(item.as_of_ts), _dt_sort(item.source_bundle_ts)),
        default=gold_latest,
    )
    api_as_of = api_permission.as_of_ts or api_permission.created_at
    gold_as_of = gold_latest.as_of_ts or gold_latest.created_at
    lag = max(0, int((gold_as_of - api_as_of).total_seconds()))
    consistent = api_as_of >= gold_as_of
    fresh = (
        str(api_permission.permission_status or "").startswith("ACTIVE_")
        and _expires_at_ok(api_permission.expires_at, now)
        and bool(api_permission.enforceable)
    )
    return {
        "risk_permission_fresh": fresh,
        "permission_status": str(api_permission.permission_status or ""),
        "permission_enforceable": bool(api_permission.enforceable),
        "risk_permission_as_of_ts": api_as_of.isoformat(),
        "api_permission_as_of_ts": api_as_of.isoformat(),
        "gold_permission_as_of_ts": gold_as_of.isoformat(),
        "permission_api_lag_sec": lag,
        "permission_api_consistent_with_gold": consistent,
        "detail": (
            f"status={api_permission.permission_status}; enforceable={api_permission.enforceable}; "
            f"expires_at={_iso(api_permission.expires_at)}"
        ),
        "consistency_detail": (
            f"api_as_of={api_as_of.isoformat()}; gold_as_of={gold_as_of.isoformat()}; "
            f"lag_sec={lag}"
        ),
    }


def _permission_from_row(row: dict[str, Any]) -> RiskPermission:
    permission = parse_risk_permission_row(row)
    if permission is None:
        raise ValueError("invalid risk_permission row")
    return permission


def _telemetry_metrics(
    root: Path,
    strategy_health: dict[str, Any],
    limits: EnforceReadinessThresholds,
) -> dict[str, Any]:
    strategy_decision_count = strategy_health.get("decision_audit_count_24h")
    decision_count = (
        int(strategy_decision_count)
        if strategy_decision_count is not None
        else _dataset_row_count(root / "silver" / "v5_decision_audit")
    )
    duplicate_rate = _float(strategy_health.get("duplicate_rate"))
    fallback_rate = _float(strategy_health.get("fallback_rate"))
    raw_imported_rows = _int(strategy_health.get("raw_imported_rows"))
    unique_event_rows = _int(strategy_health.get("unique_event_rows"))
    duplicate_event_rows = _int(
        strategy_health.get("duplicate_event_rows")
        or strategy_health.get("duplicate_event_count")
    )
    exact_duplicate_event_rows = _int(strategy_health.get("exact_duplicate_event_rows"))
    conflicting_duplicate_event_rows = _int(
        strategy_health.get("conflicting_duplicate_event_rows")
    )
    conflicting_duplicate_event_key_count = _int(
        strategy_health.get("conflicting_duplicate_event_key_count")
    )
    unique_request_count = _int(strategy_health.get("unique_request_count"))
    unique_actual_fallback_count = _int(strategy_health.get("unique_actual_fallback_count"))
    v5_event_metrics = _v5_telemetry_event_dedupe_metrics(root)
    if v5_event_metrics["raw_imported_rows"] > 0:
        raw_imported_rows = v5_event_metrics["raw_imported_rows"]
        unique_event_rows = v5_event_metrics["unique_event_rows"]
        duplicate_event_rows = v5_event_metrics["duplicate_event_rows"]
        duplicate_rate = v5_event_metrics["duplicate_rate"]
        exact_duplicate_event_rows = v5_event_metrics["exact_duplicate_event_rows"]
        conflicting_duplicate_event_rows = v5_event_metrics["conflicting_duplicate_event_rows"]
        conflicting_duplicate_event_key_count = v5_event_metrics[
            "conflicting_duplicate_event_key_count"
        ]
    if duplicate_rate == 0 and not strategy_health:
        duplicate_rate = 1.0
    if duplicate_event_rows == 0 and raw_imported_rows and unique_event_rows:
        duplicate_event_rows = max(raw_imported_rows - unique_event_rows, 0)
    event_key_coverage = _event_key_coverage(root)
    dedupe_status, dedupe_reason = _dedupe_health(
        duplicate_rate=duplicate_rate,
        raw_imported_rows=raw_imported_rows,
        unique_event_rows=unique_event_rows,
        duplicate_event_rows=duplicate_event_rows,
        conflicting_duplicate_event_rows=conflicting_duplicate_event_rows,
        unique_request_count=unique_request_count,
        unique_actual_fallback_count=unique_actual_fallback_count,
        event_key_coverage=event_key_coverage,
        limits=limits,
    )
    return {
        "telemetry_duplicate_rate": duplicate_rate,
        "duplicate_rate": duplicate_rate,
        "raw_imported_rows": raw_imported_rows,
        "unique_event_rows": unique_event_rows,
        "duplicate_event_rows": duplicate_event_rows,
        "duplicate_event_count": duplicate_event_rows,
        "exact_duplicate_event_rows": exact_duplicate_event_rows,
        "conflicting_duplicate_event_rows": conflicting_duplicate_event_rows,
        "conflicting_duplicate_event_key_count": conflicting_duplicate_event_key_count,
        "duplicate_explanation": strategy_health.get("duplicate_explanation") or "",
        "event_key_coverage": event_key_coverage,
        "unique_request_count": unique_request_count,
        "unique_actual_fallback_count": unique_actual_fallback_count,
        "dedupe_health_status": dedupe_status,
        "dedupe_block_reason": dedupe_reason,
        "fallback_rate": fallback_rate,
        "decision_audit_count": decision_count,
        "decision_audit_present": decision_count > 0,
        "latest_bundle_ts": _iso(_parse_dt(strategy_health.get("latest_bundle_ts"))),
        "first_seen_bundle_ts": _iso(_parse_dt(strategy_health.get("first_seen_bundle_ts"))),
        "last_seen_bundle_ts": _iso(_parse_dt(strategy_health.get("last_seen_bundle_ts"))),
        "detail": (
            f"duplicate_rate={duplicate_rate:.4f}; fallback_rate={fallback_rate:.4f}; "
            f"exact_duplicate_event_rows={exact_duplicate_event_rows}; "
            f"conflicting_duplicate_event_rows={conflicting_duplicate_event_rows}; "
            f"decision_audit_count={decision_count}; dedupe_health_status={dedupe_status}; "
            f"dedupe_block_reason={dedupe_reason or 'none'}"
        ),
    }


def _v5_telemetry_event_dedupe_metrics(root: Path) -> dict[str, Any]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for dataset_name, dataset in (
        ("v5_quant_lab_request", root / "silver" / "v5_quant_lab_request"),
        ("v5_quant_lab_fallback", root / "silver" / "v5_quant_lab_fallback"),
    ):
        try:
            lazy = read_parquet_lazy(dataset)
            schema = set(lazy.collect_schema().names())
            selected = [
                column
                for column in [
                    "event_id",
                    "event_key",
                    "request_id",
                    "payload_hash",
                    "raw_payload_hash",
                ]
                if column in schema
            ]
            if not selected:
                continue
            rows.extend((dataset_name, row) for row in lazy.select(selected).collect().to_dicts())
        except Exception:
            continue
    raw_rows = len(rows)
    if raw_rows == 0:
        return {
            "raw_imported_rows": 0,
            "unique_event_rows": 0,
            "duplicate_event_rows": 0,
            "duplicate_rate": 0.0,
            "exact_duplicate_event_rows": 0,
            "conflicting_duplicate_event_rows": 0,
            "conflicting_duplicate_event_key_count": 0,
        }
    by_key: dict[str, set[str]] = {}
    for dataset_name, row in rows:
        key = (
            _observable_telemetry_key_part(row.get("event_id"))
            or _observable_telemetry_key_part(row.get("event_key"))
            or _observable_telemetry_key_part(row.get("request_id"))
        )
        if not key:
            key = json.dumps(row, sort_keys=True, default=str)
        namespaced_key = f"{dataset_name}:{key}"
        payload_hash = (
            _observable_telemetry_key_part(row.get("payload_hash"))
            or _observable_telemetry_key_part(row.get("raw_payload_hash"))
        )
        by_key.setdefault(namespaced_key, set()).add(payload_hash)
    unique_rows = len(by_key)
    duplicate_rows = max(raw_rows - unique_rows, 0)
    conflicting_keys = sum(1 for hashes in by_key.values() if len({h for h in hashes if h}) > 1)
    conflicting_rows = sum(
        max(len(hashes) - 1, 0)
        for hashes in by_key.values()
        if len({h for h in hashes if h}) > 1
    )
    exact_rows = max(duplicate_rows - conflicting_rows, 0)
    return {
        "raw_imported_rows": raw_rows,
        "unique_event_rows": unique_rows,
        "duplicate_event_rows": duplicate_rows,
        "duplicate_rate": duplicate_rows / raw_rows if raw_rows else 0.0,
        "exact_duplicate_event_rows": exact_rows,
        "conflicting_duplicate_event_rows": conflicting_rows,
        "conflicting_duplicate_event_key_count": conflicting_keys,
    }


_UNOBSERVABLE_TELEMETRY_KEY_VALUES = {
    "",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not_available",
    "not_observable",
    "unknown",
}


def _observable_telemetry_key_part(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in _UNOBSERVABLE_TELEMETRY_KEY_VALUES:
        return ""
    return text


def _event_key_coverage(root: Path) -> float:
    total = 0
    keyed = 0
    for dataset in [
        root / "silver" / "v5_quant_lab_request",
        root / "silver" / "v5_quant_lab_fallback",
    ]:
        metrics = _event_key_dataset_metrics(dataset)
        total += metrics["total"]
        keyed += metrics["keyed"]
    if total == 0:
        return 1.0
    return keyed / total


def _dataset_row_count(dataset_path: Path) -> int:
    try:
        frame = read_parquet_lazy(dataset_path).select(pl.len().alias("rows"))
        metrics = _collect_lazy(frame)
    except Exception:
        return 0
    if metrics.is_empty() or "rows" not in metrics.columns:
        return 0
    return int(metrics["rows"][0] or 0)


def _event_key_dataset_metrics(dataset_path: Path) -> dict[str, int]:
    try:
        lazy = read_parquet_lazy(dataset_path)
        schema = lazy.collect_schema()
        columns = set(schema.names())
        if "event_key" in columns:
            cleaned_key = pl.col("event_key").cast(pl.Utf8).str.strip_chars().str.to_lowercase()
            keyed_expr = (
                (~cleaned_key.is_in(sorted(_UNOBSERVABLE_TELEMETRY_KEY_VALUES)))
                .fill_null(False)
                .sum()
                .alias("keyed")
            )
        else:
            keyed_expr = pl.lit(0).alias("keyed")
        metrics = _collect_lazy(
            lazy.select(
                pl.len().alias("total"),
                keyed_expr,
            )
        )
    except Exception:
        return {"total": 0, "keyed": 0}
    if metrics.is_empty():
        return {"total": 0, "keyed": 0}
    return {
        "total": int(metrics["total"][0] or 0),
        "keyed": int(metrics["keyed"][0] or 0),
    }


def _collect_lazy(frame: pl.LazyFrame) -> pl.DataFrame:
    try:
        return frame.collect(engine="streaming")
    except TypeError:
        try:
            return frame.collect(streaming=True)
        except TypeError:
            return frame.collect()


def _dedupe_health(
    *,
    duplicate_rate: float,
    raw_imported_rows: int,
    unique_event_rows: int,
    duplicate_event_rows: int,
    conflicting_duplicate_event_rows: int,
    unique_request_count: int,
    unique_actual_fallback_count: int,
    event_key_coverage: float,
    limits: EnforceReadinessThresholds,
) -> tuple[str, str]:
    if event_key_coverage < limits.telemetry_event_key_coverage_min:
        missing_rate = 1.0 - event_key_coverage
        return "BLOCKED", f"event_key_missing_rate={missing_rate:.4f}"
    if raw_imported_rows and unique_event_rows <= 0:
        return "BLOCKED", "raw_rows_present_but_unique_event_rows_zero"
    if raw_imported_rows and unique_event_rows > raw_imported_rows:
        return "BLOCKED", "unique_event_rows_exceeds_raw_imported_rows"
    if raw_imported_rows and duplicate_event_rows < 0:
        return "BLOCKED", "negative_duplicate_event_rows"
    if conflicting_duplicate_event_rows > 0:
        return "BLOCKED", f"conflicting_duplicate_event_rows={conflicting_duplicate_event_rows}"
    if unique_event_rows and unique_request_count > unique_event_rows:
        return "BLOCKED", "unique_request_count_exceeds_unique_event_rows"
    if unique_request_count and unique_actual_fallback_count > unique_request_count:
        return "BLOCKED", "unique_actual_fallback_count_exceeds_unique_request_count"
    if duplicate_rate > limits.telemetry_duplicate_rate_warn:
        return "WARN", "high_duplicate_rate_expected_for_rolling_followup_bundles"
    return "PASS", ""


def _gate_metrics(root: Path) -> dict[str, Any]:
    rows = _rows(root / "gold" / "gate_decision")
    if not rows:
        return {
            "has_dead_gate": True,
            "gate_status_counts": {},
            "global_gate_status_counts": {},
            "not_global_gate_status_counts": {},
            "not_global_gate_ignored_count": 0,
            "alpha_gate_latest_status": "missing",
            "detail": "gate_decision missing",
        }
    counts: dict[str, int] = {}
    global_counts: dict[str, int] = {}
    not_global_counts: dict[str, int] = {}
    global_rows: list[dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        if _is_not_global_strategy_gate(row):
            not_global_counts[status] = not_global_counts.get(status, 0) + 1
        else:
            global_counts[status] = global_counts.get(status, 0) + 1
            global_rows.append(row)
    if not global_rows:
        return {
            "has_dead_gate": True,
            "gate_status_counts": counts,
            "global_gate_status_counts": {},
            "not_global_gate_status_counts": not_global_counts,
            "not_global_gate_ignored_count": len(rows),
            "alpha_gate_latest_status": "missing_global_gate",
            "detail": (
                f"global gate_decision missing; gate_status_counts={counts}; "
                f"not_global_gate_status_counts={not_global_counts}"
            ),
        }
    latest = max(global_rows, key=_row_sort_ts)
    return {
        "has_dead_gate": "DEAD" in global_counts,
        "gate_status_counts": counts,
        "global_gate_status_counts": global_counts,
        "not_global_gate_status_counts": not_global_counts,
        "not_global_gate_ignored_count": len(rows) - len(global_rows),
        "alpha_gate_latest_status": str(latest.get("status") or "unknown"),
        "detail": (
            f"global_gate_status_counts={global_counts}; "
            f"not_global_gate_status_counts={not_global_counts}; "
            f"not_global_gate_ignored_count={len(rows) - len(global_rows)}"
        ),
    }


def _is_not_global_strategy_gate(row: dict[str, Any]) -> bool:
    if _truthy(row.get("not_global_strategy_gate")):
        return True
    role = str(row.get("role") or "").strip().lower()
    if role == "research_baseline":
        return True
    if _truthy(row.get("not_live_eligible")) and str(row.get("baseline_status") or "").strip():
        return True
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _check(
    name: str,
    passed: bool,
    *,
    value: Any = None,
    threshold: Any = None,
    detail: str = "",
    warn_only: bool = False,
) -> EnforceReadinessCheck:
    status = "PASS" if passed else ("WARN" if warn_only else "BLOCKED")
    return EnforceReadinessCheck(
        name=name,
        status=status,
        passed=passed,
        value=value,
        threshold=threshold,
        detail=detail,
    )


def _required_actions(blocked: list[str], warnings: list[str]) -> list[str]:
    mapping = {
        "risk_permission_fresh": "run qlab publish-risk-permission after sync-v5-telemetry",
        "risk_permission_api_consistent_with_gold": (
            "verify /v1/risk/live-permission returns latest gold row"
        ),
        "cost_api_global_default_rate": "fix cost_bucket_daily/API symbol matching before enforce",
        "cost_symbol_hit_rate": "review non-live cost symbol coverage for advisory reporting",
        "cost_live_symbol_hit_rate": (
            "ensure every V5 live symbol hits a symbol-level cost bucket before enforce"
        ),
        "actual_or_mixed_cost_coverage": (
            "improve actual/mixed cost coverage for research and expanded symbols"
        ),
        "actual_or_mixed_cost_coverage_research_universe": (
            "improve actual/mixed cost coverage for research and expanded symbols"
        ),
        "actual_or_mixed_cost_coverage_live_universe": (
            "enable OKX read-only fills/bills or V5 order lifecycle cost backfill "
            "for BTC/ETH/SOL/BNB before enforce"
        ),
        "telemetry_dedupe_health": (
            "fix V5 telemetry event_id/event_key coverage before enforce"
        ),
        "fallback_rate": "reduce Quant Lab API timeout/local fallback rate",
        "alpha_gate_status_not_dead": (
            "publish valid alpha evidence and remove DEAD gates before enforce"
        ),
        "decision_audit_present": "sync V5 telemetry before evaluating enforce readiness",
    }
    return [mapping[name] for name in [*blocked, *warnings] if name in mapping]


def _write_csv(path: Path, report: EnforceReadinessReport) -> None:
    path.write_text(_csv_text(report), encoding="utf-8")


def _csv_text(report: EnforceReadinessReport) -> str:
    output = []
    header = [
        "as_of_ts",
        "strategy",
        "version",
        "readiness_status",
        "veto_status",
        "entry_status",
        "scale_status",
        "shadow_only_recommended",
        "raw_imported_rows",
        "unique_event_rows",
        "duplicate_event_rows",
        "duplicate_rate",
        "dedupe_health_status",
        "dedupe_block_reason",
        "cost_symbol_hit_rate",
        "cost_live_symbol_hit_rate",
        "missing_live_cost_symbols",
        "live_universe_source",
        "actual_or_mixed_cost_coverage",
        "actual_or_mixed_cost_coverage_research_universe",
        "actual_or_mixed_cost_coverage_live_universe",
        "actual_or_mixed_cost_coverage_expanded_universe",
        "proxy_only_symbols_live",
        "proxy_only_symbols_expanded",
        "blocked_reasons",
        "warning_reasons",
        "veto_blocked_reasons",
        "entry_blocked_reasons",
        "scale_blocked_reasons",
        "required_actions",
        "contract_version",
    ]
    row = {
        "as_of_ts": report.as_of_ts.isoformat(),
        "strategy": report.strategy,
        "version": report.version,
        "readiness_status": report.readiness_status,
        "veto_status": report.veto_status,
        "entry_status": report.entry_status,
        "scale_status": report.scale_status,
        "shadow_only_recommended": str(report.shadow_only_recommended).lower(),
        "raw_imported_rows": report.metrics.get("raw_imported_rows", 0),
        "unique_event_rows": report.metrics.get("unique_event_rows", 0),
        "duplicate_event_rows": report.metrics.get("duplicate_event_rows", 0),
        "duplicate_rate": report.metrics.get("duplicate_rate", 0.0),
        "dedupe_health_status": report.metrics.get("dedupe_health_status", "PASS"),
        "dedupe_block_reason": report.metrics.get("dedupe_block_reason", ""),
        "cost_symbol_hit_rate": report.metrics.get("cost_symbol_hit_rate", 0.0),
        "cost_live_symbol_hit_rate": report.metrics.get("cost_live_symbol_hit_rate", 0.0),
        "missing_live_cost_symbols": json.dumps(
            report.metrics.get("missing_live_cost_symbols", []),
            sort_keys=True,
        ),
        "live_universe_source": report.metrics.get("live_universe_source", "unknown"),
        "actual_or_mixed_cost_coverage": report.metrics.get(
            "actual_or_mixed_cost_coverage",
            0.0,
        ),
        "actual_or_mixed_cost_coverage_research_universe": report.metrics.get(
            "actual_or_mixed_cost_coverage_research_universe",
            report.metrics.get("actual_or_mixed_cost_coverage", 0.0),
        ),
        "actual_or_mixed_cost_coverage_live_universe": report.metrics.get(
            "actual_or_mixed_cost_coverage_live_universe",
            0.0,
        ),
        "actual_or_mixed_cost_coverage_expanded_universe": report.metrics.get(
            "actual_or_mixed_cost_coverage_expanded_universe",
            0.0,
        ),
        "proxy_only_symbols_live": json.dumps(
            report.metrics.get("proxy_only_symbols_live", []),
            sort_keys=True,
        ),
        "proxy_only_symbols_expanded": json.dumps(
            report.metrics.get("proxy_only_symbols_expanded", []),
            sort_keys=True,
        ),
        "blocked_reasons": json.dumps(report.blocked_reasons, sort_keys=True),
        "warning_reasons": json.dumps(report.warning_reasons, sort_keys=True),
        "veto_blocked_reasons": json.dumps(report.veto_blocked_reasons, sort_keys=True),
        "entry_blocked_reasons": json.dumps(report.entry_blocked_reasons, sort_keys=True),
        "scale_blocked_reasons": json.dumps(report.scale_blocked_reasons, sort_keys=True),
        "required_actions": json.dumps(report.required_actions, sort_keys=True),
        "contract_version": report.contract_version,
    }
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=header)
    writer.writeheader()
    writer.writerow(row)
    output.append(buffer.getvalue())
    return "".join(output)


def _row_sort_ts(row: dict[str, Any]) -> float:
    for field in ["as_of_ts", "latest_bundle_ts", "created_at", "date", "bundle_ts"]:
        parsed = _parse_dt(row.get(field))
        if parsed is not None:
            return parsed.timestamp()
    return 0.0


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        text = f"{text}T00:00:00+00:00"
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _dt_sort(value: datetime | None) -> float:
    if value is None:
        return 0.0
    return (value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)).timestamp()


def _expires_at_ok(value: datetime | None, now: datetime) -> bool:
    return value is None or value >= now


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
