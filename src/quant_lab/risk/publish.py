import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import GateDecision, RiskPermission
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.risk.permissions import evaluate_live_permission

GATE_DECISION_DATASET = Path("gold") / "gate_decision"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
COST_HEALTH_DAILY_DATASET = Path("gold") / "cost_health_daily"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
V5_GATE_COMPLIANCE_DATASET = Path("gold") / "v5_gate_compliance_daily"
STRATEGY_HEALTH_DAILY_DATASET = Path("gold") / "strategy_health_daily"

RISK_PERMISSION_SCHEMA = {
    "strategy": pl.Utf8,
    "version": pl.Utf8,
    "permission": pl.Utf8,
    "allowed_modes": pl.Utf8,
    "max_gross_exposure": pl.Float64,
    "max_single_weight": pl.Float64,
    "cost_model_version": pl.Utf8,
    "gate_version": pl.Utf8,
    "reasons": pl.Utf8,
    "created_at": pl.Utf8,
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
        gate_decisions=gate_decisions,
        cost_health=cost_health,
        data_health=data_health,
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
        if isinstance(cleaned.get("metrics"), str):
            cleaned["metrics"] = _json_dict(cleaned["metrics"])
        if isinstance(cleaned.get("reasons"), str):
            cleaned["reasons"] = _json_list(cleaned["reasons"])
        try:
            decisions.append(GateDecision.model_validate(cleaned))
        except Exception:
            continue
    return decisions


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
        "allowed_modes": ["paper", "live_canary"],
        "max_gross_exposure": 0.25,
        "max_single_weight": 0.05,
    }


def strategy_telemetry_reasons(root: Path, strategy: str) -> list[str]:
    if strategy != "v5":
        return []
    reasons: list[str] = []
    compliance = read_parquet_dataset(root / V5_GATE_COMPLIANCE_DATASET)
    if not compliance.is_empty():
        latest = _latest_row(compliance, "date")
        if int(latest.get("violation_count") or 0) > 0:
            reasons.append("v5_gate_compliance_violation")
    health = read_parquet_dataset(root / STRATEGY_HEALTH_DAILY_DATASET)
    if not health.is_empty():
        latest = _latest_row(health, "date")
        if str(latest.get("status") or "").upper() == "CRITICAL":
            reasons.append("v5_strategy_health_critical")
    return reasons


def risk_permission_row(permission: RiskPermission) -> dict[str, Any]:
    return {
        **permission.model_dump(mode="json"),
        "allowed_modes": _json(permission.allowed_modes),
        "reasons": _json(permission.reasons),
        "source": "research.risk_permission.v0.1",
        "fallback_level": "NONE",
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
    try:
        return RiskPermission.model_validate(cleaned)
    except Exception:
        return None


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
