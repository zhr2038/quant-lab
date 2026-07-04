import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import GateDecision, RiskPermission, RiskPermissionStatus
from quant_lab.contracts.v5_quant_lab import RISK_PERMISSION_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy, upsert_parquet_dataset
from quant_lab.data.market_bar_time import (
    DEFAULT_MARKET_BAR_TIMEFRAME,
    market_bar_close_ts,
    market_bar_freshness_seconds,
)
from quant_lab.ops.dataset_registry import get_dataset_spec
from quant_lab.risk.advisory import (
    apply_risk_advisory_context,
    build_risk_advisory_context,
    json_list,
    required_gate_decisions,
)
from quant_lab.risk.permissions import evaluate_live_permission
from quant_lab.trade_level.judgment import trade_level_risk_summary

GATE_DECISION_DATASET = Path("gold") / "gate_decision"
RISK_PERMISSION_DATASET = Path("gold") / "risk_permission"
RISK_PERMISSION_API_DEPENDENCY_META_DATASET = Path("gold") / "risk_permission_api_dependency_meta"
TRADE_LEVEL_JUDGMENT_DATASET = Path("gold") / "trade_level_judgment"
FALSE_BLOCK_AUDIT_DATASET = Path("gold") / "quant_lab_false_block_audit"
OPPORTUNITY_COST_BY_BUCKET_DATASET = Path("gold") / "opportunity_cost_by_bucket"
TRADE_LEVEL_BUCKET_POLICY_DATASET = Path("gold") / "trade_level_bucket_policy"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
COST_HEALTH_DAILY_DATASET = Path("gold") / "cost_health_daily"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
V5_GATE_COMPLIANCE_DATASET = Path("gold") / "v5_gate_compliance_daily"
STRATEGY_HEALTH_DAILY_DATASET = Path("gold") / "strategy_health_daily"
V5_BUNDLE_MANIFEST_DATASET = Path("bronze") / "strategy_telemetry" / "v5" / "bundle_manifest"
DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS = 90 * 60
MARKET_BAR_WARNING_DELAY_SECONDS = 2 * 60 * 60
MARKET_BAR_CRITICAL_DELAY_SECONDS = 3 * 60 * 60
ACTIVE_PERMISSION_STATUSES = {
    RiskPermissionStatus.ACTIVE_ALLOW.value,
    RiskPermissionStatus.ACTIVE_SELL_ONLY.value,
    RiskPermissionStatus.ACTIVE_ABORT.value,
}

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
    "trade_level_decision_summary": pl.Utf8,
    "micro_canary_review_count": pl.Int64,
    "micro_canary_review_bucket_count": pl.Int64,
    "blocked_by_observability_count": pl.Int64,
    "reviewable_abort_count": pl.Int64,
    "micro_canary_review_ready_count": pl.Int64,
    "micro_canary_review_blocked_by_observability_count": pl.Int64,
    "micro_canary_allow_candidate_count": pl.Int64,
    "risk_block_bucket_count": pl.Int64,
    "recommended_next_permission_mode": pl.Utf8,
    "top_micro_canary_review_buckets": pl.Utf8,
    "false_block_rate": pl.Float64,
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
    advisory_context = build_risk_advisory_context(
        root,
        gate_decisions=gate_decisions,
        data_health=data_health,
        cost_health=cost_health,
        telemetry_reasons=telemetry_reasons,
    )
    permission = annotate_risk_permission(
        permission,
        telemetry_latest_ts=telemetry_latest_ts,
        as_of_ts=datetime.now(UTC),
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    )
    permission = apply_risk_advisory_context(permission, advisory_context)
    row = risk_permission_row(permission)
    trade_level_summary = _trade_level_summary_for_risk_permission(root)
    row.update(trade_level_summary)
    if str(trade_level_summary.get("recommended_next_permission_mode") or "").startswith(
        "MICRO_CANARY_REVIEW"
    ):
        advisory_modes = json_list(row.get("allowed_advisory_modes") or "[]")
        row["allowed_advisory_modes"] = _json(
            list(dict.fromkeys([*advisory_modes, "micro_canary_review"]))
        )
    frame = pl.DataFrame(
        [row],
        schema=RISK_PERMISSION_SCHEMA,
        orient="row",
    )
    rows = upsert_parquet_dataset(
        frame,
        root / RISK_PERMISSION_DATASET,
        key_columns=["strategy", "version"],
    )
    _publish_risk_permission_api_dependency_meta(
        root,
        strategy=strategy,
        version=version,
        permission=permission,
        telemetry_latest_ts=telemetry_latest_ts,
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


def _publish_risk_permission_api_dependency_meta(
    root: Path,
    *,
    strategy: str,
    version: str,
    permission: RiskPermission,
    telemetry_latest_ts: datetime | None,
) -> None:
    generated_at = datetime.now(UTC)
    frame = _risk_permission_dependency_meta_frame(
        [
            _risk_permission_dependency_meta_row(
                root,
                strategy=strategy,
                version=version,
                permission=permission,
                telemetry_latest_ts=telemetry_latest_ts,
                generated_at=generated_at,
            )
        ],
    )
    upsert_parquet_dataset(
        frame,
        root / RISK_PERMISSION_API_DEPENDENCY_META_DATASET,
        key_columns=["strategy", "version"],
    )
    stored = read_parquet_dataset(root / RISK_PERMISSION_API_DEPENDENCY_META_DATASET)
    snapshot_frame = stored if not stored.is_empty() else frame
    _write_dependency_snapshot_meta(
        root / RISK_PERMISSION_API_DEPENDENCY_META_DATASET,
        frame=snapshot_frame,
        source_sha=_dependency_meta_source_sha(snapshot_frame),
        generated_at=generated_at,
    )


def rebuild_risk_permission_api_dependency_meta_from_latest_permission(
    lake_root: str | Path,
) -> int:
    root = Path(lake_root)
    risk_permission = read_parquet_dataset(root / RISK_PERMISSION_DATASET)
    if risk_permission.is_empty() or not {"strategy", "version"}.issubset(
        set(risk_permission.columns)
    ):
        return 0
    generated_at = datetime.now(UTC)
    rows: list[dict[str, Any]] = []
    for row in _latest_risk_permission_rows(risk_permission):
        permission = parse_risk_permission_row(row)
        if permission is None:
            continue
        strategy = str(row.get("strategy") or permission.strategy or "").strip()
        version = str(row.get("version") or permission.version or "").strip()
        if not strategy or not version:
            continue
        telemetry_latest_ts = _ensure_utc(
            permission.telemetry_latest_ts or permission.source_bundle_ts
        )
        rows.append(
            _risk_permission_dependency_meta_row(
                root,
                strategy=strategy,
                version=version,
                permission=permission,
                telemetry_latest_ts=telemetry_latest_ts,
                generated_at=generated_at,
            )
        )
    if not rows:
        return 0
    frame = _risk_permission_dependency_meta_frame(rows)
    written = upsert_parquet_dataset(
        frame,
        root / RISK_PERMISSION_API_DEPENDENCY_META_DATASET,
        key_columns=["strategy", "version"],
    )
    stored = read_parquet_dataset(root / RISK_PERMISSION_API_DEPENDENCY_META_DATASET)
    snapshot_frame = stored if not stored.is_empty() else frame
    _write_dependency_snapshot_meta(
        root / RISK_PERMISSION_API_DEPENDENCY_META_DATASET,
        frame=snapshot_frame,
        source_sha=_dependency_meta_source_sha(snapshot_frame),
        generated_at=generated_at,
    )
    return written


def _risk_permission_dependency_meta_row(
    root: Path,
    *,
    strategy: str,
    version: str,
    permission: RiskPermission,
    telemetry_latest_ts: datetime | None,
    generated_at: datetime,
) -> dict[str, Any]:
    source_sha = _dependency_source_sha(
        strategy=strategy,
        version=version,
        permission=permission,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    return {
        "strategy": strategy,
        "version": version,
        "risk_permission_source_sha": source_sha,
        "gate_decision_source_sha": _dataset_quick_signature(root / GATE_DECISION_DATASET),
        "cost_health_source_sha": _dataset_quick_signature(root / COST_HEALTH_DAILY_DATASET),
        "telemetry_latest_ts": telemetry_latest_ts.isoformat() if telemetry_latest_ts else "",
        "source_sha": source_sha,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
    }


def _risk_permission_dependency_meta_frame(
    rows: list[dict[str, Any]],
) -> pl.DataFrame:
    return pl.DataFrame(rows, infer_schema_length=None)


def _latest_risk_permission_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    sortable_columns = [
        column
        for column in ("as_of_ts", "source_bundle_ts", "created_at")
        if column in frame.columns
    ]
    normalized = frame
    for column in sortable_columns:
        normalized = normalized.with_columns(_lazy_utc_datetime(column).alias(column))
    if sortable_columns:
        normalized = normalized.sort(sortable_columns, descending=True, nulls_last=True)
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for row in normalized.to_dicts():
        key = (str(row.get("strategy") or ""), str(row.get("version") or ""))
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def _dependency_meta_source_sha(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "empty"
    import hashlib

    digest = hashlib.sha256()
    digest.update(str(frame.height).encode("ascii"))
    for row in sorted(
        frame.to_dicts(),
        key=lambda item: (
            str(item.get("strategy") or ""),
            str(item.get("version") or ""),
            str(item.get("source_sha") or ""),
        ),
    ):
        digest.update(str(row.get("strategy") or "").encode("utf-8"))
        digest.update(str(row.get("version") or "").encode("utf-8"))
        digest.update(str(row.get("source_sha") or "").encode("utf-8"))
    return digest.hexdigest()


def _write_dependency_snapshot_meta(
    dataset_path: Path,
    *,
    frame: pl.DataFrame,
    source_sha: str,
    generated_at: datetime,
) -> None:
    dataset_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": "risk_permission_api_dependency_meta",
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "expires_at": "",
        "row_count": frame.height,
        "source_sha": source_sha,
        "file_count": sum(1 for path in dataset_path.rglob("*.parquet") if path.is_file()),
        "schema_version": "risk_permission_api_dependency_meta",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    tmp_path = dataset_path / "._snapshot_meta.tmp"
    tmp_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_path.replace(dataset_path / "_snapshot_meta.json")


def _dependency_source_sha(
    *,
    strategy: str,
    version: str,
    permission: RiskPermission,
    telemetry_latest_ts: datetime | None,
) -> str:
    import hashlib

    permission_payload = permission.model_dump(mode="json")
    permission_payload.update(
        {
            "strategy": strategy,
            "version": version,
            "telemetry_latest_ts": telemetry_latest_ts.isoformat()
            if telemetry_latest_ts is not None
            else "",
        }
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            permission_payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _dataset_quick_signature(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        stat = path.stat()
    except OSError:
        return "unavailable"
    return f"{stat.st_mtime_ns}:{stat.st_size}"


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
    latest_ts, timeframe = _latest_market_bar_metadata(market_bars)
    if latest_ts is None:
        return {"status": "critical", "is_critical": True, "reasons": ["market_bar_missing"]}
    latest_utc = latest_ts.astimezone(UTC) if latest_ts.tzinfo else latest_ts.replace(tzinfo=UTC)
    close_utc = market_bar_close_ts(latest_utc, timeframe)
    age_seconds = market_bar_freshness_seconds(
        latest_utc,
        latest_close_ts=close_utc,
        timeframe=timeframe,
    )
    age_seconds = age_seconds if age_seconds is not None else 0
    critical_threshold = _market_bar_critical_delay_seconds()
    warning_threshold = _market_bar_warning_delay_seconds()
    if age_seconds >= critical_threshold:
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_stale"],
            "latest_market_bar_ts": latest_utc.isoformat(),
            "latest_market_bar_close_ts": close_utc.isoformat() if close_utc else None,
            "market_bar_timeframe": timeframe or DEFAULT_MARKET_BAR_TIMEFRAME,
            "freshness_seconds": age_seconds,
            "stale_threshold_seconds": critical_threshold,
        }
    if age_seconds >= warning_threshold:
        return {
            "status": "warning",
            "is_critical": False,
            "reasons": ["market_bar_delayed"],
            "latest_market_bar_ts": latest_utc.isoformat(),
            "latest_market_bar_close_ts": close_utc.isoformat() if close_utc else None,
            "market_bar_timeframe": timeframe or DEFAULT_MARKET_BAR_TIMEFRAME,
            "freshness_seconds": age_seconds,
            "warning_threshold_seconds": warning_threshold,
            "stale_threshold_seconds": critical_threshold,
            "allowed_modes": ["paper"],
            "max_gross_exposure": 0.25,
            "max_single_weight": 0.05,
        }
    return {
        "status": "ok",
        "latest_market_bar_ts": latest_utc.isoformat(),
        "latest_market_bar_close_ts": close_utc.isoformat() if close_utc else None,
        "market_bar_timeframe": timeframe or DEFAULT_MARKET_BAR_TIMEFRAME,
        "freshness_seconds": age_seconds,
        "warning_threshold_seconds": warning_threshold,
        "stale_threshold_seconds": critical_threshold,
        "allowed_modes": ["paper"],
        "max_gross_exposure": 0.25,
        "max_single_weight": 0.05,
    }


def _market_bar_warning_delay_seconds() -> int:
    return _seconds_env(
        "QUANT_LAB_MARKET_BAR_WARNING_DELAY_SECONDS",
        MARKET_BAR_WARNING_DELAY_SECONDS,
    )


def _market_bar_critical_delay_seconds() -> int:
    spec = get_dataset_spec("market_bar")
    default = (
        spec.freshness_seconds
        if spec and spec.freshness_seconds
        else MARKET_BAR_CRITICAL_DELAY_SECONDS
    )
    configured = _seconds_env("QUANT_LAB_MARKET_BAR_CRITICAL_DELAY_SECONDS", int(default))
    return max(_market_bar_warning_delay_seconds(), configured)


def _seconds_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    try:
        value = int(float(raw)) if raw else default
    except ValueError:
        return default
    return max(0, value)


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
    manifest_ts = _latest_v5_bundle_manifest_ts(root, strategy)
    if manifest_ts is not None:
        return manifest_ts
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


def _latest_v5_bundle_manifest_ts(root: Path, strategy: str) -> datetime | None:
    lazy, columns = _safe_parquet_lazy(root / V5_BUNDLE_MANIFEST_DATASET)
    if lazy is None or "bundle_ts" not in columns:
        return None
    scoped = lazy.filter(pl.col("strategy") == strategy) if "strategy" in columns else lazy
    try:
        frame = scoped.select(_lazy_utc_datetime("bundle_ts").max().alias("bundle_ts")).collect()
    except Exception:
        return None
    return _coerce_timestamp(frame.item(0, "bundle_ts")) if frame.height else None


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
    value = status.value if hasattr(status, "value") else str(status or "")
    return value in ACTIVE_PERMISSION_STATUSES


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
        "trade_level_decision_summary": "{}",
        "micro_canary_review_count": 0,
        "micro_canary_review_bucket_count": 0,
        "blocked_by_observability_count": 0,
        "reviewable_abort_count": 0,
        "micro_canary_review_ready_count": 0,
        "micro_canary_review_blocked_by_observability_count": 0,
        "micro_canary_allow_candidate_count": 0,
        "risk_block_bucket_count": 0,
        "recommended_next_permission_mode": "PAPER_ONLY",
        "top_micro_canary_review_buckets": "[]",
        "false_block_rate": 0.0,
    }


def _trade_level_summary_for_risk_permission(root: Path) -> dict[str, Any]:
    try:
        judgments = read_parquet_dataset(root / TRADE_LEVEL_JUDGMENT_DATASET)
    except Exception:
        judgments = pl.DataFrame()
    try:
        false_block_audit = read_parquet_dataset(root / FALSE_BLOCK_AUDIT_DATASET)
    except Exception:
        false_block_audit = pl.DataFrame()
    try:
        opportunity_buckets = read_parquet_dataset(root / OPPORTUNITY_COST_BY_BUCKET_DATASET)
    except Exception:
        opportunity_buckets = pl.DataFrame()
    try:
        bucket_policy = read_parquet_dataset(root / TRADE_LEVEL_BUCKET_POLICY_DATASET)
    except Exception:
        bucket_policy = pl.DataFrame()
    return trade_level_risk_summary(
        judgments,
        false_block_audit,
        opportunity_buckets,
        bucket_policy,
    )


def parse_risk_permission_row(row: dict[str, Any]) -> RiskPermission | None:
    cleaned = dict(row)
    cleaned.pop("source", None)
    cleaned.pop("fallback_level", None)
    cleaned.pop("permission_source", None)
    cleaned.pop("trade_level_decision_summary", None)
    cleaned.pop("micro_canary_review_count", None)
    cleaned.pop("micro_canary_review_bucket_count", None)
    cleaned.pop("blocked_by_observability_count", None)
    cleaned.pop("reviewable_abort_count", None)
    cleaned.pop("micro_canary_review_ready_count", None)
    cleaned.pop("micro_canary_review_blocked_by_observability_count", None)
    cleaned.pop("micro_canary_allow_candidate_count", None)
    cleaned.pop("risk_block_bucket_count", None)
    cleaned.pop("recommended_next_permission_mode", None)
    cleaned.pop("top_micro_canary_review_buckets", None)
    cleaned.pop("false_block_rate", None)
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


def _latest_market_bar_metadata(df: pl.DataFrame) -> tuple[datetime | None, str]:
    if df.is_empty() or "ts" not in df.columns:
        return None, DEFAULT_MARKET_BAR_TIMEFRAME
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
        return None, DEFAULT_MARKET_BAR_TIMEFRAME
    try:
        latest_row = normalized.sort("ts").tail(1).to_dicts()[0]
    except Exception:
        return None, DEFAULT_MARKET_BAR_TIMEFRAME
    latest = latest_row.get("ts")
    latest_utc = latest.astimezone(UTC) if isinstance(latest, datetime) else None
    timeframe = str(latest_row.get("timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME)
    return latest_utc, timeframe


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
