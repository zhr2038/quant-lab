import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import GateDecision, GateStatus, RiskPermission
from quant_lab.costs.calibrate import calibrate_costs_for_day
from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.research.baselines import alpha_role
from quant_lab.risk.permissions import evaluate_live_permission
from quant_lab.risk.publish import annotate_risk_permission, latest_strategy_telemetry_ts

COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
GATE_DECISION_DATASET = Path("gold") / "gate_decision"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"
MARKET_BAR_DATASET = Path("silver") / "market_bar"

BOOTSTRAP_ALPHA_ID = "v5.core"
BOOTSTRAP_GATE_VERSION = "bootstrap.quarantine.v1"
BOOTSTRAP_COST_MODEL_VERSION = "bootstrap.cost.v1"
BOOTSTRAP_SOURCE = "bootstrap_gold_health"
BOOTSTRAP_WARNING = "bootstrap conservative placeholder, not live-ready evidence"

GATE_DECISION_SCHEMA = {
    "strategy": pl.Utf8,
    "alpha_id": pl.Utf8,
    "version": pl.Utf8,
    "gate_version": pl.Utf8,
    "status": pl.Utf8,
    "passed": pl.Boolean,
    "reasons": pl.Utf8,
    "metrics": pl.Utf8,
    "next_action": pl.Utf8,
    "created_at": pl.Utf8,
    "role": pl.Utf8,
    "baseline_status": pl.Utf8,
    "not_live_eligible": pl.Boolean,
    "not_global_strategy_gate": pl.Boolean,
    "source": pl.Utf8,
    "fallback_level": pl.Utf8,
}

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


class BootstrapGoldHealthResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    strategy: str
    version: str
    day: str
    cost_bootstrapped: bool
    gate_bootstrapped: bool
    risk_bootstrapped: bool
    cost_bucket_daily_rows: int = Field(ge=0)
    gate_decision_rows: int = Field(ge=0)
    risk_permission_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


def bootstrap_gold_health(
    lake_root: str | Path,
    strategy: str,
    version: str,
    day: str = "auto",
) -> BootstrapGoldHealthResult:
    root = Path(lake_root)
    resolved_day = _resolve_day(root, day)
    warnings: list[str] = []

    cost_bootstrapped = False
    if _dataset_empty(root / COST_BUCKET_DAILY_DATASET):
        calibration = calibrate_costs_for_day(root, resolved_day)
        cost_bootstrapped = True
        if calibration.sources == ["global_default"]:
            warnings.append(BOOTSTRAP_WARNING)

    gate_bootstrapped = False
    if _dataset_empty(root / GATE_DECISION_DATASET):
        gate_rows = [_bootstrap_gate_row(strategy=strategy, version=version)]
        gate_decision_rows = upsert_parquet_dataset(
            pl.DataFrame(gate_rows, schema=GATE_DECISION_SCHEMA, orient="row"),
            root / GATE_DECISION_DATASET,
            key_columns=["strategy", "alpha_id", "version", "gate_version"],
        )
        gate_bootstrapped = True
        warnings.append(BOOTSTRAP_WARNING)
    else:
        gate_decision_rows = read_parquet_dataset(root / GATE_DECISION_DATASET).height

    risk_bootstrapped = False
    if _dataset_empty(root / RISK_PERMISSION_DATASET):
        permission = _evaluate_bootstrap_permission(root, strategy=strategy, version=version)
        risk_permission_rows = upsert_parquet_dataset(
            pl.DataFrame(
                [_risk_permission_row(permission)],
                schema=RISK_PERMISSION_SCHEMA,
                orient="row",
            ),
            root / RISK_PERMISSION_DATASET,
            key_columns=["strategy", "version"],
        )
        risk_bootstrapped = True
        warnings.append(BOOTSTRAP_WARNING)
    else:
        risk_permission_rows = read_parquet_dataset(root / RISK_PERMISSION_DATASET).height

    cost_bucket_daily_rows = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET).height
    gate_decision_rows = read_parquet_dataset(root / GATE_DECISION_DATASET).height
    risk_permission_rows = read_parquet_dataset(root / RISK_PERMISSION_DATASET).height

    return BootstrapGoldHealthResult(
        lake_root=str(root),
        strategy=strategy,
        version=version,
        day=resolved_day,
        cost_bootstrapped=cost_bootstrapped,
        gate_bootstrapped=gate_bootstrapped,
        risk_bootstrapped=risk_bootstrapped,
        cost_bucket_daily_rows=cost_bucket_daily_rows,
        gate_decision_rows=gate_decision_rows,
        risk_permission_rows=risk_permission_rows,
        warnings=_dedupe(warnings),
    )


def _resolve_day(lake_root: Path, day: str) -> str:
    if day != "auto":
        return day
    market_bars = read_parquet_dataset(lake_root / MARKET_BAR_DATASET)
    latest_ts = _latest_market_bar_ts(market_bars)
    if latest_ts is None:
        return datetime.now(UTC).date().isoformat()
    return latest_ts.date().isoformat()


def _dataset_empty(dataset_path: Path) -> bool:
    return read_parquet_dataset(dataset_path).is_empty()


def _bootstrap_gate_row(strategy: str, version: str) -> dict[str, Any]:
    created_at = _utc_now()
    role = alpha_role(BOOTSTRAP_ALPHA_ID)
    return {
        "strategy": strategy,
        "alpha_id": BOOTSTRAP_ALPHA_ID,
        "version": version,
        "gate_version": BOOTSTRAP_GATE_VERSION,
        "status": GateStatus.QUARANTINE.value,
        "passed": False,
        "reasons": _json(["bootstrap_gate", "missing_alpha_evidence", "not_ready_for_live"]),
        "metrics": _json({"coverage": None, "ic_tstat": None, "bootstrap": True}),
        "next_action": "generate_alpha_evidence_before_live",
        "created_at": created_at,
        "role": role,
        "baseline_status": GateStatus.QUARANTINE.value if role == "research_baseline" else "",
        "not_live_eligible": role == "research_baseline",
        "not_global_strategy_gate": role == "research_baseline",
        "source": BOOTSTRAP_SOURCE,
        "fallback_level": "BOOTSTRAP_CONSERVATIVE",
    }


def _evaluate_bootstrap_permission(root: Path, strategy: str, version: str) -> RiskPermission:
    gate_decisions = _load_gate_decisions(root, strategy)
    cost_health = _bootstrap_cost_health(root)
    data_health = _bootstrap_data_health(root)
    permission = evaluate_live_permission(
        strategy=strategy,
        version=version,
        gate_decisions=gate_decisions,
        cost_health=cost_health,
        data_health=data_health,
    )
    extra_reasons = data_health.get("reasons", [])
    if isinstance(extra_reasons, list):
        reasons = _dedupe(
            [
                *permission.reasons,
                *_bootstrap_gate_reasons(gate_decisions),
                *[str(reason) for reason in extra_reasons],
            ]
        )
        permission = permission.model_copy(update={"reasons": reasons})
    return annotate_risk_permission(
        permission,
        telemetry_latest_ts=latest_strategy_telemetry_ts(root, strategy),
        as_of_ts=datetime.now(UTC),
    )


def _load_gate_decisions(root: Path, strategy: str) -> list[GateDecision]:
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
    return decisions


def _bootstrap_cost_health(root: Path) -> dict[str, Any]:
    df = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
    if df.is_empty():
        return {
            "status": "missing",
            "missing": True,
            "cost_model_version": "missing",
        }
    latest_day = _latest_string(df, "day") or "unknown"
    if _all_rows_are_global_default(df):
        return {
            "status": "missing",
            "missing": True,
            "cost_model_version": BOOTSTRAP_COST_MODEL_VERSION,
        }
    return {
        "status": "ok",
        "cost_model_version": f"cost_bucket_daily:{latest_day}",
    }


def _bootstrap_data_health(root: Path) -> dict[str, Any]:
    market_bars = read_parquet_dataset(root / MARKET_BAR_DATASET)
    latest_ts = _latest_market_bar_ts(market_bars)
    if latest_ts is None:
        return {
            "status": "warning",
            "reasons": ["market_bar_missing"],
        }
    if latest_ts < datetime.now(UTC) - timedelta(hours=24):
        return {
            "status": "warning",
            "reasons": ["market_bar_stale"],
            "latest_market_bar_ts": latest_ts.isoformat(),
        }
    return {
        "status": "ok",
        "latest_market_bar_ts": latest_ts.isoformat(),
    }


def _risk_permission_row(permission: RiskPermission) -> dict[str, Any]:
    return {
        **permission.model_dump(mode="json"),
        "allowed_modes": _json(permission.allowed_modes),
        "reasons": _json(permission.reasons),
        "source": BOOTSTRAP_SOURCE,
        "fallback_level": "BOOTSTRAP_CONSERVATIVE",
    }


def _bootstrap_gate_reasons(gate_decisions: list[GateDecision]) -> list[str]:
    if any(decision.gate_version == BOOTSTRAP_GATE_VERSION for decision in gate_decisions):
        return ["required_alpha_gate_quarantine"]
    return []


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
    if not isinstance(latest, datetime):
        return None
    return latest.astimezone(UTC)


def _all_rows_are_global_default(df: pl.DataFrame) -> bool:
    if df.is_empty():
        return False
    checks: list[pl.Expr] = []
    if "source" in df.columns:
        checks.append(pl.col("source") == "global_default")
    if "fallback_level" in df.columns:
        checks.append(pl.col("fallback_level") == "GLOBAL_DEFAULT")
    if not checks:
        return False
    condition = checks[0]
    for check in checks[1:]:
        condition = condition | check
    return df.filter(condition).height == df.height


def _latest_string(df: pl.DataFrame, column: str) -> str | None:
    if df.is_empty() or column not in df.columns:
        return None
    value = df.select(pl.col(column).max()).item()
    return str(value) if value is not None else None


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
