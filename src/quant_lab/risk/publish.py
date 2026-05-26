import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import GateDecision, RiskPermission, RiskPermissionStatus
from quant_lab.contracts.v5_quant_lab import RISK_PERMISSION_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy, upsert_parquet_dataset
from quant_lab.risk.advisory import (
    apply_risk_advisory_context,
    build_risk_advisory_context,
    json_list,
    required_gate_decisions,
)
from quant_lab.risk.permissions import evaluate_live_permission

GATE_DECISION_DATASET = Path("gold") / "gate_decision"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
COST_HEALTH_DAILY_DATASET = Path("gold") / "cost_health_daily"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
V5_GATE_COMPLIANCE_DATASET = Path("gold") / "v5_gate_compliance_daily"
STRATEGY_HEALTH_DAILY_DATASET = Path("gold") / "strategy_health_daily"
DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS = 90 * 60

RISK_PERMISSION_SCHEMA = {
    "schema_version": pl.Utf8,
    "strategy": pl.Utf8,
    "version": pl.Utf8,
    "permission": pl.Utf8,
    "allowed_modes": pl.Utf8,
    "max_gross_exposure": pl.Float64,
    "max_single_weight": pl.Float64,
    "max_gross_exposure_usdt": pl.Float64,
    "max_single_order_usdt": pl.Float64,
    "cost_model_version": pl.Utf8,
    "gate_version": pl.Utf8,
    "reasons": pl.Utf8,
    "reason": pl.Utf8,
    "created_at": pl.Utf8,
    "as_of_ts": pl.Utf8,
    "source_bundle_ts": pl.Utf8,
    "expires_at": pl.Utf8,
    "telemetry_latest_ts": pl.Utf8,
    "permission_freshness_sec": pl.Int64,
    "contract_version": pl.Utf8,
    "permission_status": pl.Utf8,
    "enforceable": pl.Boolean,
    "risk_reason_codes": pl.Utf8,
    "system_safety_status": pl.Utf8,
    "core_alpha_gate_status": pl.Utf8,
    "core_alpha_dead": pl.Boolean,
    "baseline_status": pl.Utf8,
    "baseline_alpha_id": pl.Utf8,
    "baseline_role": pl.Utf8,
    "baseline_not_live_eligible": pl.Boolean,
    "baseline_not_global_strategy_gate": pl.Boolean,
    "strategy_opportunities_available": pl.Boolean,
    "allowed_advisory_modes": pl.Utf8,
    "allowed_live_modes": pl.Utf8,
    "live_block_reasons": pl.Utf8,
    "source": pl.Utf8,
    "fallback_level": pl.Utf8,
}


class RiskPermissionPublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    strategy: str
    version: str
    permission: str
    risk_permission_rows: int = Field(ge=0)
    gate_decision_rows: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def publish_risk_permission(
    lake_root: str | Path,
    strategy: str,
    version: str,
) -> RiskPermissionPublishResult:
    root = Path(lake_root)
    gate_decisions = load_gate_decisions(root, strategy)
    cost_health = lake_cost_health(root)
    data_health = lake_data_health(root)
    telemetry_reasons = strategy_telemetry_reasons(root, strategy)
    telemetry_latest_ts = latest_strategy_telemetry_ts(root, strategy)
    if telemetry_reasons:
        data_health = {
            **data_health,
            "status": "critical",
            "is_critical": True,
            "reasons": [*data_health.get("reasons", []), *telemetry_reasons],
        }
    permission = evaluate_live_permission(
        strategy=strategy,
        version=version,
        gate_decisions=required_gate_decisions(gate_decisions),
        cost_health=cost_health,
        data_health=data_health,
    )
    permission = apply_risk_advisory_context(
        permission,
        build_risk_advisory_context(
            root,
            gate_decisions=gate_decisions,
            data_health=data_health,
            cost_health=cost_health,
            telemetry_reasons=telemetry_reasons,
        ),
    )
    permission = annotate_risk_permission(
        permission,
        telemetry_latest_ts=telemetry_latest_ts,
        as_of_ts=datetime.now(UTC),
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    )
    frame = pl.DataFrame(
        [risk_permission_row(permission)],
        schema=RISK_PERMISSION_SCHEMA,
        orient="row",
    )
    rows = upsert_parquet_dataset(
        frame,
        root / RISK_PERMISSION_DATASET,
        key_columns=["strategy", "version"],
    )
    warnings: list[str] = []
    if not gate_decisions:
        warnings.append("gate_decision missing or empty")
    if cost_health.get("missing"):
        warnings.append("cost health missing")
    if data_health.get("is_critical"):
        warnings.append("market data critical")
    return RiskPermissionPublishResult(
        lake_root=str(root),
        strategy=strategy,
        version=version,
        permission=permission.permission.value,
        risk_permission_rows=rows,
        gate_decision_rows=len(gate_decisions),
        reasons=permission.reasons,
        warnings=warnings,
    )


def load_gate_decisions(root: Path, strategy: str) -> list[GateDecision]:
    df = read_parquet_dataset(root / GATE_DECISION_DATASET)
    if df.is_empty():
        return []
    if "strategy" in df.columns:
        df = df.filter(pl.col("strategy") == strategy)
    decisions: list[GateDecision] = []
    for row in df.to_dicts():
        cleaned = dict(row)
        cleaned.pop("strategy", None)
        cleaned.pop("source", None)
        cleaned.pop("fallback_level", None)
        cleaned.pop("role", None)
        cleaned.pop("baseline_status", None)
        cleaned.pop("not_live_eligible", None)
        cleaned.pop("not_global_strategy_gate", None)
        if isinstance(cleaned.get("metrics"), str):
            cleaned["metrics"] = _json_dict(cleaned["metrics"])
        if isinstance(cleaned.get("reasons"), str):
            cleaned["reasons"] = _json_list(cleaned["reasons"])
        try:
            decisions.append(GateDecision.model_validate(cleaned))
        except Exception:
            continue
    return _prefer_research_gate_decisions(decisions)


def _prefer_research_gate_decisions(decisions: list[GateDecision]) -> list[GateDecision]:
    research_decisions = [
        decision for decision in decisions if not decision.gate_version.startswith("bootstrap.")
    ]
    return research_decisions or decisions


def lake_cost_health(root: Path) -> dict[str, Any]:
    health = read_parquet_dataset(root / COST_HEALTH_DAILY_DATASET)
    if not health.is_empty():
        row = _latest_row(health, "day")
        status = str(row.get("status") or "ok").lower()
        return {
            "status": status,
            "missing": status == "critical" and int(row.get("actual_rows") or 0) == 0,
            "high_fallback": float(row.get("fallback_ratio") or 0.0) > 0.5,
            "fallback_ratio": float(row.get("fallback_ratio") or 0.0),
            "cost_model_version": str(row.get("cost_model_version") or "cost_health_daily"),
        }

    buckets = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
    if buckets.is_empty():
        return {"status": "missing", "missing": True, "cost_model_version": "missing"}
    latest_day = (
        str(buckets.select(pl.col("day").max()).item()) if "day" in buckets.columns else "unknown"
    )
    sources = _string_values(buckets, "source")
    fallback_ratio = _fallback_ratio(buckets)
    status = "ok"
    if latest_day != "unknown" and _is_day_stale(latest_day, 3):
        status = "stale"
    return {
        "status": status,
        "stale": status == "stale",
        "high_fallback": fallback_ratio > 0.5,
        "fallback_ratio": fallback_ratio,
        "cost_model_version": f"cost_bucket_daily:{latest_day}",
        "sources": sorted(sources),
    }


def lake_data_health(root: Path) -> dict[str, Any]:
    market_bars = read_parquet_dataset(root / MARKET_BAR_DATASET)
    latest_ts = _latest_market_bar_ts(market_bars)
    if latest_ts is None:
        return {"status": "critical", "is_critical": True, "reasons": ["market_bar_missing"]}
    if latest_ts < datetime.now(UTC) - timedelta(hours=24):
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_stale"],
            "latest_market_bar_ts": latest_ts.isoformat(),
        }
    return {
        "status": "ok",
        "latest_market_bar_ts": latest_ts.isoformat(),
        "allowed_modes": ["paper"],
        "max_gross_exposure": 0.25,
        "max_single_weight": 0.05,
    }


def strategy_telemetry_reasons(root: Path, strategy: str) -> list[str]:
    if strategy != "v5":
        return []
    reasons: list[str] = []
    latest = _latest_lazy_row(
        root / V5_GATE_COMPLIANCE_DATASET,
        strategy=strategy,
        sort_columns=["date", "created_at"],
    )
    if latest is not None:
        if int(latest.get("violation_count") or 0) > 0:
            reasons.append("v5_gate_compliance_violation")
    latest = _latest_lazy_row(
        root / STRATEGY_HEALTH_DAILY_DATASET,
        strategy=strategy,
        sort_columns=["latest_bundle_ts", "created_at", "date"],
    )
    if latest is not None:
        if str(latest.get("status") or "").upper() == "CRITICAL":
            reasons.append("v5_strategy_health_critical")
    return reasons


def latest_strategy_telemetry_ts(root: Path, strategy: str) -> datetime | None:
    if strategy != "v5":
        return None
    lazy, columns = _safe_parquet_lazy(root / STRATEGY_HEALTH_DAILY_DATASET)
    if lazy is None:
        return None
    scoped = lazy.filter(pl.col("strategy") == strategy) if "strategy" in columns else lazy
    timestamp_columns = [
        column for column in ["latest_bundle_ts", "created_at", "date"] if column in columns
    ]
    if not timestamp_columns:
        return None
    try:
        frame = scoped.select(
            [_lazy_utc_datetime(column).max().alias(column) for column in timestamp_columns]
        ).collect()
    except Exception:
        return None
    parsed = [
        _coerce_timestamp(frame.item(0, column))
        for column in timestamp_columns
        if column in frame.columns
    ]
    clean = [value for value in parsed if value is not None]
    return max(clean) if clean else None


def annotate_risk_permission(
    permission: RiskPermission,
    *,
    telemetry_latest_ts: datetime | None,
    as_of_ts: datetime | None = None,
    threshold_seconds: int = DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
) -> RiskPermission:
    as_of = as_of_ts or datetime.now(UTC)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        as_of = as_of.replace(tzinfo=UTC)
    as_of = as_of.astimezone(UTC)
    telemetry_ts = _ensure_utc(telemetry_latest_ts)
    stale = risk_permission_stale_vs_telemetry(
        as_of_ts=as_of,
        telemetry_latest_ts=telemetry_ts,
        threshold_seconds=threshold_seconds,
    )
    freshness = permission_freshness_sec(
        as_of_ts=as_of,
        telemetry_latest_ts=telemetry_ts,
    )
    expires_at = as_of + timedelta(seconds=max(threshold_seconds, 0))
    return permission.model_copy(
        update={
            "as_of_ts": as_of,
            "source_bundle_ts": telemetry_ts,
            "expires_at": expires_at,
            "telemetry_latest_ts": telemetry_ts,
            "permission_freshness_sec": freshness,
            "contract_version": RISK_PERMISSION_CONTRACT_VERSION,
            "permission_status": permission_status(permission.permission.value, stale=stale),
            "enforceable": not stale,
            "risk_reason_codes": permission.reasons,
            "reason": permission.reasons[0] if permission.reasons else None,
        }
    )


def risk_permission_stale_vs_telemetry(
    *,
    as_of_ts: datetime | None,
    telemetry_latest_ts: datetime | None,
    threshold_seconds: int = DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
) -> bool:
    if as_of_ts is None or telemetry_latest_ts is None:
        return False
    as_of = _ensure_utc(as_of_ts)
    telemetry_ts = _ensure_utc(telemetry_latest_ts)
    if as_of is None or telemetry_ts is None:
        return False
    return telemetry_ts > as_of + timedelta(seconds=max(threshold_seconds, 0))


def permission_freshness_sec(
    *,
    as_of_ts: datetime | None,
    telemetry_latest_ts: datetime | None,
) -> int:
    if as_of_ts is None or telemetry_latest_ts is None:
        return 0
    as_of = _ensure_utc(as_of_ts)
    telemetry_ts = _ensure_utc(telemetry_latest_ts)
    if as_of is None or telemetry_ts is None or telemetry_ts <= as_of:
        return 0
    return int((telemetry_ts - as_of).total_seconds())


def permission_status(
    permission: str,
    *,
    stale: bool = False,
    expired: bool = False,
) -> RiskPermissionStatus:
    normalized = str(permission).strip().upper()
    if normalized not in {"ALLOW", "SELL_ONLY", "ABORT"}:
        return RiskPermissionStatus.NO_FRESH_PERMISSION
    if expired:
        prefix = "EXPIRED"
    else:
        prefix = "STALE" if stale else "ACTIVE"
    return RiskPermissionStatus(f"{prefix}_{normalized}")


def is_permission_status_enforceable(status: str | RiskPermissionStatus | None) -> bool:
    return str(status or "").startswith("ACTIVE_")


def risk_permission_row(permission: RiskPermission) -> dict[str, Any]:
    return {
        **permission.model_dump(mode="json"),
        "allowed_modes": _json(permission.allowed_modes),
        "reasons": _json(permission.reasons),
        "source": "research.risk_permission.v0.1",
        "fallback_level": "NONE",
        "risk_reason_codes": _json(permission.risk_reason_codes or permission.reasons),
        "allowed_advisory_modes": _json(permission.allowed_advisory_modes),
        "allowed_live_modes": _json(permission.allowed_live_modes),
        "live_block_reasons": _json(permission.live_block_reasons),
    }


def parse_risk_permission_row(row: dict[str, Any]) -> RiskPermission | None:
    cleaned = dict(row)
    cleaned.pop("source", None)
    cleaned.pop("fallback_level", None)
    cleaned.pop("permission_source", None)
    if isinstance(cleaned.get("allowed_modes"), str):
        cleaned["allowed_modes"] = _json_list(cleaned["allowed_modes"])
    if isinstance(cleaned.get("reasons"), str):
        cleaned["reasons"] = _json_list(cleaned["reasons"])
    if isinstance(cleaned.get("risk_reason_codes"), str):
        cleaned["risk_reason_codes"] = _json_list(cleaned["risk_reason_codes"])
    for field in ["allowed_advisory_modes", "allowed_live_modes", "live_block_reasons"]:
        if isinstance(cleaned.get(field), str):
            cleaned[field] = json_list(cleaned[field])
    for field in [
        "as_of_ts",
        "source_bundle_ts",
        "expires_at",
        "telemetry_latest_ts",
        "permission_freshness_sec",
    ]:
        if cleaned.get(field) in {"", "None", "none", "null"}:
            cleaned[field] = None
    if cleaned.get("permission_status") in {"", "None", "none", "null"}:
        cleaned["permission_status"] = None
    if cleaned.get("contract_version") in {"", "None", "none", "null"}:
        cleaned.pop("contract_version", None)
    try:
        return RiskPermission.model_validate(cleaned)
    except Exception:
        return None


def _latest_timestamp_value(df: pl.DataFrame, columns: list[str]) -> datetime | None:
    for column in columns:
        if column not in df.columns:
            continue
        values = [_coerce_timestamp(value) for value in df[column].to_list()]
        parsed = [value for value in values if value is not None]
        if parsed:
            return max(parsed)
    return None


def _safe_parquet_lazy(path: Path) -> tuple[pl.LazyFrame | None, set[str]]:
    try:
        lazy = read_parquet_lazy(path)
        columns = set(lazy.collect_schema().names())
    except Exception:
        return None, set()
    return lazy, columns


def _latest_lazy_row(
    path: Path,
    *,
    strategy: str | None = None,
    sort_columns: list[str] | None = None,
) -> dict[str, Any] | None:
    lazy, columns = _safe_parquet_lazy(path)
    if lazy is None:
        return None
    scoped = lazy
    if strategy is not None and "strategy" in columns:
        scoped = scoped.filter(pl.col("strategy") == strategy)
    sort_column = next((column for column in (sort_columns or []) if column in columns), None)
    if sort_column is not None:
        scoped = scoped.with_columns(_lazy_utc_datetime(sort_column).alias(sort_column)).sort(
            sort_column,
            descending=True,
            nulls_last=True,
        )
    try:
        frame = scoped.limit(1).collect()
    except Exception:
        return None
    return None if frame.is_empty() else frame.to_dicts()[0]


def _lazy_utc_datetime(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return _ensure_utc(value)
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
    return _ensure_utc(parsed)


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _latest_market_bar_ts(df: pl.DataFrame) -> datetime | None:
    if df.is_empty() or "ts" not in df.columns:
        return None
    normalized = df
    try:
        if normalized.schema.get("ts") == pl.String:
            normalized = normalized.with_columns(
                pl.col("ts").str.to_datetime(time_zone="UTC", strict=False)
            )
        else:
            normalized = normalized.with_columns(
                pl.col("ts").cast(pl.Datetime(time_zone="UTC")).alias("ts")
            )
    except Exception:
        return None
    latest = normalized.select(pl.col("ts").max()).item()
    return latest.astimezone(UTC) if isinstance(latest, datetime) else None


def _fallback_ratio(df: pl.DataFrame) -> float:
    if df.is_empty():
        return 1.0
    if "source" in df.columns:
        fallback = df.filter(pl.col("source") != "actual_okx_fills_and_bills").height
        return fallback / df.height
    if "fallback_level" in df.columns:
        fallback = df.filter(pl.col("fallback_level") != "NONE").height
        return fallback / df.height
    return 1.0


def _string_values(df: pl.DataFrame, column: str) -> set[str]:
    if df.is_empty() or column not in df.columns:
        return set()
    return {str(value) for value in df[column].drop_nulls().to_list()}


def _latest_row(df: pl.DataFrame, column: str) -> dict[str, Any]:
    if column in df.columns:
        return df.sort(column).tail(1).to_dicts()[0]
    return df.tail(1).to_dicts()[0]


def _is_day_stale(day: str, max_age_days: int) -> bool:
    try:
        day_date = datetime.fromisoformat(day).date()
    except ValueError:
        return True
    return day_date < (datetime.now(UTC).date() - timedelta(days=max_age_days))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _json_dict(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _json_list(value: str) -> list[str]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(loaded, list):
        return [str(item) for item in loaded]
    return [str(loaded)]
