from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import RiskPermission
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.costs.model import estimate_cost_from_cost_bucket_daily_rows
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.risk.publish import is_permission_status_enforceable
from quant_lab.symbols import normalize_symbol

DEFAULT_COST_SYMBOLS = ["BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT"]
ENFORCE_READINESS_JSON = "v5_enforce_readiness.json"
ENFORCE_READINESS_CSV = "v5_enforce_readiness.csv"


class EnforceReadinessThresholds(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cost_api_global_default_rate_max: float = Field(default=0.0, ge=0, le=1)
    cost_symbol_hit_rate_min: float = Field(default=0.95, ge=0, le=1)
    actual_or_mixed_cost_coverage_min: float = Field(default=0.50, ge=0, le=1)
    telemetry_duplicate_rate_warn: float = Field(default=0.10, ge=0, le=1)
    telemetry_duplicate_rate_block: float = Field(default=0.50, ge=0, le=1)
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
    shadow_only_recommended: bool
    blocked_reasons: list[str]
    warning_reasons: list[str]
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
    symbols = _cost_symbols(cost_rows, cost_symbols)
    cost_metrics = _cost_api_metrics(
        cost_rows=cost_rows,
        symbols=symbols,
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
    telemetry_metrics = _telemetry_metrics(root, strategy_health)
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
        ),
        _check(
            "actual_or_mixed_cost_coverage",
            cost_metrics["actual_or_mixed_cost_coverage"]
            >= limits.actual_or_mixed_cost_coverage_min,
            value=cost_metrics["actual_or_mixed_cost_coverage"],
            threshold=f">= {limits.actual_or_mixed_cost_coverage_min}",
            detail=cost_metrics["coverage_detail"],
        ),
        _check(
            "telemetry_duplicate_rate",
            telemetry_metrics["duplicate_rate"] <= limits.telemetry_duplicate_rate_warn,
            value=telemetry_metrics["duplicate_rate"],
            threshold=f"<= {limits.telemetry_duplicate_rate_warn}",
            detail=telemetry_metrics["detail"],
            warn_only=(
                telemetry_metrics["duplicate_rate"] <= limits.telemetry_duplicate_rate_block
            ),
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
    status = "READY"
    if blocked:
        status = "BLOCKED"
    elif warnings:
        status = "WARN"

    metrics = {
        **risk_metrics,
        **cost_metrics,
        **telemetry_metrics,
        **gate_metrics,
    }
    return EnforceReadinessReport(
        strategy=strategy,
        version=version,
        readiness_status=status,
        shadow_only_recommended=status != "READY",
        blocked_reasons=blocked,
        warning_reasons=warnings,
        required_actions=_required_actions(blocked, warnings),
        as_of_ts=now,
        checks=checks,
        metrics=metrics,
    )


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


def _cost_symbols(rows: list[dict[str, Any]], explicit: list[str] | None) -> list[str]:
    if explicit:
        return sorted({normalize_symbol(symbol) for symbol in explicit})
    symbols = {
        normalize_symbol(row.get("symbol"))
        for row in rows
        if str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    return sorted(symbols) or DEFAULT_COST_SYMBOLS


def _cost_api_metrics(
    *,
    cost_rows: list[dict[str, Any]],
    symbols: list[str],
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
    actual_or_mixed = {
        normalize_symbol(row.get("symbol"))
        for row in cost_rows
        if str(row.get("source") or row.get("cost_source") or "").lower()
        in {"actual_fills", "actual_okx_fills_and_bills", "mixed_actual_proxy"}
        and str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    cost_symbols = {
        normalize_symbol(row.get("symbol"))
        for row in cost_rows
        if str(row.get("symbol") or "").upper() not in {"", "GLOBAL"}
    }
    coverage_denominator = max(len(cost_symbols), 1)
    return {
        "cost_symbols_checked": symbols,
        "cost_api_global_default_count": len(global_default),
        "cost_api_global_default_rate": len(global_default) / total,
        "cost_symbol_hit_count": len(symbol_hits),
        "cost_symbol_hit_rate": len(symbol_hits) / total,
        "actual_or_mixed_cost_symbol_count": len(actual_or_mixed),
        "actual_or_mixed_cost_coverage": len(actual_or_mixed) / coverage_denominator,
        "cost_estimate_examples": estimates,
        "cost_detail": (
            f"global_default={len(global_default)}; symbol_hits={len(symbol_hits)}; "
            f"symbols={symbols}"
        ),
        "coverage_detail": (
            f"actual_or_mixed_symbols={sorted(actual_or_mixed)}; "
            f"cost_symbols={sorted(cost_symbols)}"
        ),
    }


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
    cleaned = dict(row)
    cleaned.pop("source", None)
    cleaned.pop("fallback_level", None)
    cleaned.pop("permission_source", None)
    for field in ["reasons", "risk_reason_codes", "allowed_modes"]:
        value = cleaned.get(field)
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
                cleaned[field] = parsed if isinstance(parsed, list) else [str(parsed)]
            except json.JSONDecodeError:
                cleaned[field] = [value] if value else []
    return RiskPermission.model_validate(cleaned)


def _telemetry_metrics(root: Path, strategy_health: dict[str, Any]) -> dict[str, Any]:
    decisions = _rows(root / "silver" / "v5_decision_audit")
    decision_count = int(strategy_health.get("decision_audit_count_24h") or len(decisions))
    duplicate_rate = _float(strategy_health.get("duplicate_rate"))
    fallback_rate = _float(strategy_health.get("fallback_rate"))
    if duplicate_rate == 0 and not strategy_health:
        duplicate_rate = 1.0
    return {
        "telemetry_duplicate_rate": duplicate_rate,
        "duplicate_rate": duplicate_rate,
        "fallback_rate": fallback_rate,
        "decision_audit_count": decision_count,
        "decision_audit_present": decision_count > 0,
        "latest_bundle_ts": _iso(_parse_dt(strategy_health.get("latest_bundle_ts"))),
        "detail": (
            f"duplicate_rate={duplicate_rate:.4f}; fallback_rate={fallback_rate:.4f}; "
            f"decision_audit_count={decision_count}"
        ),
    }


def _gate_metrics(root: Path) -> dict[str, Any]:
    rows = _rows(root / "gold" / "gate_decision")
    if not rows:
        return {
            "has_dead_gate": True,
            "gate_status_counts": {},
            "alpha_gate_latest_status": "missing",
            "detail": "gate_decision missing",
        }
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    latest = max(rows, key=_row_sort_ts)
    return {
        "has_dead_gate": "DEAD" in counts,
        "gate_status_counts": counts,
        "alpha_gate_latest_status": str(latest.get("status") or "unknown"),
        "detail": f"gate_status_counts={counts}",
    }


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
        "cost_symbol_hit_rate": "ensure all V5 symbols hit symbol-level cost buckets",
        "actual_or_mixed_cost_coverage": (
            "enable OKX read-only fills/bills or V5 trades cost backfill"
        ),
        "telemetry_duplicate_rate": "dedupe overlapping V5 follow-up bundles by event_key",
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
        "shadow_only_recommended",
        "blocked_reasons",
        "warning_reasons",
        "required_actions",
        "contract_version",
    ]
    row = {
        "as_of_ts": report.as_of_ts.isoformat(),
        "strategy": report.strategy,
        "version": report.version,
        "readiness_status": report.readiness_status,
        "shadow_only_recommended": str(report.shadow_only_recommended).lower(),
        "blocked_reasons": json.dumps(report.blocked_reasons, sort_keys=True),
        "warning_reasons": json.dumps(report.warning_reasons, sort_keys=True),
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


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None
