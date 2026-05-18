"""Read-only FastAPI surface for quant-lab strategy consumers.

This API is read-only and must not mutate strategy state or exchange state.
"""

import hmac
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.models import (
    AlphaEvidence,
    CostEstimate,
    GateDecision,
    MarketBar,
    RiskPermission,
    RiskPermissionStatus,
)
from quant_lab.costs.health import read_cost_health_daily
from quant_lab.costs.model import CostBucket, estimate_cost_bps, estimate_cost_from_lake
from quant_lab.data.lake import read_market_bars, read_parquet_dataset
from quant_lab.gates.defaults import evaluate_alpha_gate
from quant_lab.ops.metrics import api_metrics_summary, record_api_request
from quant_lab.research.bootstrap_gold import BOOTSTRAP_GATE_VERSION
from quant_lab.risk.permissions import evaluate_live_permission
from quant_lab.risk.publish import (
    DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    annotate_risk_permission,
    is_permission_status_enforceable,
    latest_strategy_telemetry_ts,
    parse_risk_permission_row,
    permission_status,
    risk_permission_stale_vs_telemetry,
)


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = "ok"
    service: str = "quant-lab"
    mode: str = "read-only"


class CatalogDatasetsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[str] = Field(default_factory=list)


class LatestFeatureRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    ts: datetime
    features: dict[str, float | None]


class LatestFeaturesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    feature_set: str
    feature_version: str | None = None
    timeframe: str | None = None
    created_at: datetime
    rows: list[LatestFeatureRow] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ResearchAlphaResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    alpha_id: str
    evidence: dict[str, Any] | None = None
    gate_decision: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)


class LivePermissionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permission: RiskPermission
    permission_source: str
    permission_freshness_seconds: int | None = None
    published_permission_stale: bool
    permission_health: dict[str, Any]
    data_health: dict[str, Any]
    cost_health: dict[str, Any]
    gate_summary: dict[str, Any]
    v5_telemetry_summary: dict[str, Any]
    api_consistency: dict[str, Any] = Field(default_factory=dict)


class ApiMetricsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_count: int
    by_path: dict[str, int]
    by_status_code: dict[str, int]
    latency_ms: dict[str, float | None]


KNOWN_DATASETS = [
    "market_bar",
    "feature_value",
    "feature_coverage_daily",
    "feature_anomaly_daily",
    "cost_bucket_daily",
    "cost_health_daily",
    "alpha_evidence",
    "gate_decision",
    "risk_permission",
    "api_request_metrics",
    "job_run_history",
    "lake_file_health_daily",
    "v5_quant_lab_mode_daily",
    "v5_quant_lab_enforcement_daily",
]


def _bool_env(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def create_app() -> FastAPI:
    disable_docs = _bool_env("QUANT_LAB_DISABLE_DOCS", default=False)
    app = FastAPI(
        title="quant-lab",
        version=__version__,
        description="Read-only quantitative research middle platform.",
        docs_url=None if disable_docs else "/docs",
        redoc_url=None if disable_docs else "/redoc",
        openapi_url=None if disable_docs else "/openapi.json",
    )

    @app.middleware("http")
    async def require_bearer_token_and_record_metrics(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        status_code = 500
        try:
            _authorize_v1_request(request)
        except HTTPException as exc:
            status_code = exc.status_code
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
            _record_api_request_metric(request, status_code, time.perf_counter() - started)
            return response
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            _record_api_request_metric(request, status_code, time.perf_counter() - started)

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/catalog/datasets", response_model=CatalogDatasetsResponse)
    def catalog_datasets() -> CatalogDatasetsResponse:
        return CatalogDatasetsResponse(datasets=KNOWN_DATASETS)

    @app.get("/v1/ops/api-metrics", response_model=ApiMetricsResponse)
    def ops_api_metrics(day: str | None = None) -> ApiMetricsResponse:
        return ApiMetricsResponse(**api_metrics_summary(_lake_root(), day=day))

    @app.get("/v1/gates/example", response_model=GateDecision)
    def gate_example() -> GateDecision:
        evidence = AlphaEvidence.example_live_ready()
        return evaluate_alpha_gate(evidence)

    @app.get("/v1/gates/decision/{alpha_id}", response_model=GateDecision)
    def gate_decision(alpha_id: str) -> GateDecision:
        decision = _load_latest_gate_decision_for_alpha(_lake_root(), alpha_id=alpha_id)
        if decision is not None:
            return decision
        return _missing_gate_decision(alpha_id)

    @app.get("/v1/costs/example", response_model=CostEstimate)
    def costs_example() -> CostEstimate:
        buckets = [
            CostBucket(
                bucket_id="btc-default",
                symbol="BTCUSDT",
                regime="normal",
                min_notional_usdt=0,
                max_notional_usdt=100_000,
                cost_bps=4.2,
            )
        ]
        return estimate_cost_bps("BTCUSDT", "normal", 10_000, buckets)

    @app.get("/v1/costs/estimate", response_model=CostEstimate)
    def costs_estimate(
        symbol: str,
        regime: str,
        notional_usdt: float,
        quantile: str = "p75",
        notional_bucket: str | None = None,
    ) -> CostEstimate:
        return estimate_cost_from_lake(
            lake_root=_lake_root(),
            symbol=symbol,
            regime=regime,
            notional_usdt=notional_usdt,
            quantile=quantile,
            notional_bucket=notional_bucket,
        )

    @app.get("/v1/features/latest", response_model=LatestFeaturesResponse)
    def features_latest(
        feature_set: str,
        feature_version: str | None = None,
        symbols: str | None = None,
        timeframe: str | None = None,
        feature_names: str | None = None,
    ) -> LatestFeaturesResponse:
        return _latest_features_response(
            lake_root=_lake_root(),
            feature_set=feature_set,
            feature_version=feature_version,
            symbols=_split_csv(symbols),
            timeframe=timeframe,
            feature_names=_split_csv(feature_names),
        )

    @app.get("/v1/research/alpha/{alpha_id}", response_model=ResearchAlphaResponse)
    def research_alpha(alpha_id: str) -> ResearchAlphaResponse:
        lake_root = _lake_root()
        evidence = _load_latest_alpha_evidence(lake_root, alpha_id=alpha_id)
        gate = _load_latest_gate_decision_for_alpha(lake_root, alpha_id=alpha_id)
        warnings: list[str] = []
        if evidence is None:
            warnings.append("alpha_evidence missing for alpha_id")
        if gate is None:
            warnings.append("gate_decision missing for alpha_id")
        return ResearchAlphaResponse(
            alpha_id=alpha_id,
            evidence=evidence,
            gate_decision=gate.model_dump(mode="json") if gate else None,
            warnings=warnings,
        )

    @app.get("/v1/market/bars", response_model=list[MarketBar])
    def market_bars(
        venue: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[MarketBar]:
        lake_root = Path(os.environ.get("QUANT_LAB_LAKE_ROOT", "/var/lib/quant-lab/lake"))
        return read_market_bars(
            lake_root=lake_root,
            venue=venue,
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
        )

    @app.get("/v1/risk/live-permission", response_model=RiskPermission)
    def live_permission(strategy: str, version: str) -> RiskPermission:
        return _live_permission_evaluation(
            _lake_root(),
            strategy=strategy,
            version=version,
        )["permission"]

    @app.get("/v1/risk/live-permission-detail", response_model=LivePermissionDetailResponse)
    def live_permission_detail(strategy: str, version: str) -> LivePermissionDetailResponse:
        evaluation = _live_permission_evaluation(
            _lake_root(),
            strategy=strategy,
            version=version,
        )
        return LivePermissionDetailResponse(**evaluation)

    return app


def _compute_live_permission(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
) -> RiskPermission:
    return _compute_live_permission_with_context(
        lake_root,
        strategy=strategy,
        version=version,
    )["permission"]


def _live_permission_evaluation(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
) -> dict[str, Any]:
    telemetry_latest_ts = latest_strategy_telemetry_ts(lake_root, strategy)
    selection = _select_published_risk_permission(
        lake_root,
        strategy=strategy,
        version=version,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    published = selection["active"] or selection["stale"]
    computed = _compute_live_permission_with_context(
        lake_root,
        strategy=strategy,
        version=version,
    )
    recomputed = computed["permission"]
    stale = selection["active"] is None
    source = "recomputed"
    permission = recomputed
    if selection["active"] is not None and not _is_more_conservative(
        recomputed, selection["active"]
    ):
        source = "published_cache"
        permission = selection["active"]
    elif selection["active"] is None and selection["stale"] is not None:
        stale_permission = selection["stale"]
        if _permission_expired(stale_permission):
            source = "no_fresh_expired_published_permission"
            permission = _force_no_fresh_permission(
                recomputed,
                telemetry_latest_ts=telemetry_latest_ts,
            )
        else:
            source = "published_stale"
            permission = stale_permission
            if _is_more_conservative(recomputed, stale_permission):
                source = "recomputed_stale_context"
                permission = recomputed
            permission = _force_non_enforceable_permission(
                permission,
                telemetry_latest_ts=telemetry_latest_ts,
                reason="no_fresh_published_permission",
            )
    elif selection["active"] is None and selection["stale"] is None:
        source = "no_fresh_permission"
        permission = _force_no_fresh_permission(
            recomputed,
            telemetry_latest_ts=telemetry_latest_ts,
        )
    return {
        "permission": permission,
        "permission_source": source,
        "permission_freshness_seconds": _permission_freshness_seconds(published),
        "published_permission_stale": stale,
        "permission_health": _risk_permission_health(
            response_permission=permission,
            gold_latest=selection["gold_latest"],
            stale_permission_consecutive_count=_stale_permission_consecutive_count(
                lake_root,
                strategy,
            ),
        ),
        "data_health": computed["data_health"],
        "cost_health": computed["cost_health"],
        "gate_summary": computed["gate_summary"],
        "v5_telemetry_summary": computed["v5_telemetry_summary"],
        "api_consistency": _risk_permission_api_consistency(
            response_permission=permission,
            gold_latest=selection["gold_latest"],
        ),
    }


def _compute_live_permission_with_context(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
) -> dict[str, Any]:
    gate_decisions = _load_gate_decisions(lake_root, strategy=strategy)
    data_health = _lake_data_health(lake_root)
    cost_health = _lake_cost_health(lake_root)
    telemetry_reasons = _strategy_telemetry_reasons(lake_root, strategy=strategy)
    telemetry_latest_ts = latest_strategy_telemetry_ts(lake_root, strategy)
    original_data_health = dict(data_health)
    if telemetry_reasons:
        data_health = {
            **data_health,
            "status": "critical",
            "is_critical": True,
            "reasons": [*data_health.get("reasons", []), *telemetry_reasons],
        }
    if _has_bootstrap_gate(gate_decisions) and _data_health_is_critical(data_health):
        data_reasons = list(data_health.get("reasons", []))
        data_health = {
            **data_health,
            "status": "warning",
            "is_critical": False,
        }
        permission = evaluate_live_permission(
            strategy=strategy,
            version=version,
            gate_decisions=gate_decisions,
            cost_health=cost_health,
            data_health=data_health,
        )
        permission = permission.model_copy(
            update={
                "reasons": _dedupe(
                    [
                        *permission.reasons,
                        "required_alpha_gate_quarantine",
                        *data_reasons,
                    ]
                )
            }
        )
        return {
            "permission": annotate_risk_permission(
                permission,
                telemetry_latest_ts=telemetry_latest_ts,
                as_of_ts=datetime.now(UTC),
                threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
            ),
            "data_health": data_health,
            "cost_health": cost_health,
            "gate_summary": _gate_summary(gate_decisions),
            "v5_telemetry_summary": _v5_telemetry_summary(telemetry_reasons),
            "original_data_health": original_data_health,
        }

    permission = evaluate_live_permission(
        strategy=strategy,
        version=version,
        gate_decisions=gate_decisions,
        cost_health=cost_health,
        data_health=data_health,
    )
    return {
        "permission": annotate_risk_permission(
            permission,
            telemetry_latest_ts=telemetry_latest_ts,
            as_of_ts=datetime.now(UTC),
            threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
        ),
        "data_health": data_health,
        "cost_health": cost_health,
        "gate_summary": _gate_summary(gate_decisions),
        "v5_telemetry_summary": _v5_telemetry_summary(telemetry_reasons),
        "original_data_health": original_data_health,
    }


app = create_app()


def _lake_root() -> Path:
    return Path(os.environ.get("QUANT_LAB_LAKE_ROOT", "/var/lib/quant-lab/lake"))


def _authorize_v1_request(request: Request) -> None:
    if not request.url.path.startswith("/v1/"):
        return
    if not _client_ip_allowed(request):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="quant-lab API client IP is not allowed",
        )
    token = os.environ.get("QUANT_LAB_API_TOKEN")
    if not token:
        return
    provided = _bearer_token(request.headers.get("authorization", ""))
    if provided is None or not hmac.compare_digest(provided, token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="quant-lab API bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )


def _bearer_token(authorization: str) -> str | None:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :]


def _client_ip_allowed(request: Request) -> bool:
    allowed = os.environ.get("QUANT_LAB_ALLOWED_CLIENT_IPS")
    if not allowed:
        return True
    allowed_hosts = {item.strip() for item in allowed.split(",") if item.strip()}
    host = request.client.host if request.client else ""
    return host in allowed_hosts


def _record_api_request_metric(request: Request, status_code: int, duration_seconds: float) -> None:
    if not request.url.path.startswith("/v1/"):
        return
    if not _bool_env("QUANT_LAB_API_METRICS_ENABLED", default=True):
        return
    try:
        record_api_request(
            lake_root=_lake_root(),
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_seconds=duration_seconds,
            client_host=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
        )
    except Exception:
        return


def _risk_permission_ttl_seconds() -> int:
    default_ttl = DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS
    value = os.environ.get("QUANT_LAB_RISK_PERMISSION_TTL_SECONDS", str(default_ttl))
    try:
        ttl = int(value)
    except ValueError:
        return default_ttl
    return max(ttl, 0)


def _risk_permission_is_stale(
    permission: RiskPermission,
    *,
    telemetry_latest_ts: datetime | None,
) -> bool:
    if permission.permission_status and not is_permission_status_enforceable(
        permission.permission_status
    ):
        return True
    if risk_permission_stale_vs_telemetry(
        as_of_ts=permission.as_of_ts or permission.created_at,
        telemetry_latest_ts=telemetry_latest_ts or permission.telemetry_latest_ts,
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    ):
        return True
    if permission.expires_at and permission.expires_at < datetime.now(UTC):
        return True
    ttl_seconds = _risk_permission_ttl_seconds()
    if ttl_seconds == 0:
        return True
    reference = permission.as_of_ts or permission.created_at
    return reference < datetime.now(UTC) - timedelta(seconds=ttl_seconds)


def _permission_freshness_seconds(permission: RiskPermission | None) -> int | None:
    if permission is None:
        return None
    reference = permission.as_of_ts or permission.created_at
    return max(0, int((datetime.now(UTC) - reference).total_seconds()))


def _is_more_conservative(candidate: RiskPermission, baseline: RiskPermission) -> bool:
    return _permission_rank(candidate.permission.value) > _permission_rank(
        baseline.permission.value
    )


def _permission_rank(permission: str) -> int:
    return {"ALLOW": 0, "SELL_ONLY": 1, "ABORT": 2}.get(permission, 2)


def _gate_summary(gate_decisions: list[GateDecision]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    alpha_ids: list[str] = []
    gate_versions: list[str] = []
    for decision in gate_decisions:
        status_value = decision.status.value
        counts[status_value] = counts.get(status_value, 0) + 1
        alpha_ids.append(decision.alpha_id)
        gate_versions.append(decision.gate_version)
    return {
        "total": len(gate_decisions),
        "status_counts": counts,
        "alpha_ids": sorted(set(alpha_ids)),
        "gate_versions": sorted(set(gate_versions)),
    }


def _v5_telemetry_summary(reasons: list[str]) -> dict[str, Any]:
    return {
        "status": "critical" if reasons else "ok",
        "reasons": list(reasons),
    }


def _latest_features_response(
    *,
    lake_root: Path,
    feature_set: str,
    feature_version: str | None,
    symbols: list[str] | None,
    timeframe: str | None,
    feature_names: list[str] | None,
) -> LatestFeaturesResponse:
    df = read_parquet_dataset(lake_root / "gold" / "feature_value")
    created_at = datetime.now(UTC)
    if df.is_empty():
        return LatestFeaturesResponse(
            feature_set=feature_set,
            feature_version=feature_version,
            timeframe=timeframe,
            created_at=created_at,
            rows=[],
            warnings=["feature_value dataset is missing or empty"],
        )
    filtered = df.filter(pl.col("feature_set") == feature_set)
    if "ts" in filtered.columns:
        filtered = _normalize_datetime_column(filtered, "ts")
    if "created_at" in filtered.columns:
        filtered = _normalize_datetime_column(filtered, "created_at")
        latest_created = filtered.select(pl.col("created_at").max()).item()
        if isinstance(latest_created, datetime):
            created_at = latest_created.astimezone(UTC)
    if feature_version is None:
        if filtered.is_empty() or "feature_version" not in filtered.columns:
            version = None
        else:
            version = str(filtered.select(pl.col("feature_version").max()).item())
            filtered = filtered.filter(pl.col("feature_version") == version)
    else:
        version = feature_version
        filtered = filtered.filter(pl.col("feature_version") == feature_version)
    if timeframe:
        filtered = filtered.filter(pl.col("timeframe") == timeframe)
    if symbols:
        filtered = filtered.filter(pl.col("symbol").is_in(symbols))
    if feature_names:
        filtered = filtered.filter(pl.col("feature_name").is_in(feature_names))
    if filtered.is_empty():
        return LatestFeaturesResponse(
            feature_set=feature_set,
            feature_version=version,
            timeframe=timeframe,
            created_at=created_at,
            rows=[],
            warnings=["no feature rows matched query"],
        )

    latest_ts = filtered.group_by("symbol").agg(pl.col("ts").max().alias("ts"))
    latest = filtered.join(latest_ts, on=["symbol", "ts"], how="inner")
    rows: list[LatestFeatureRow] = []
    for (symbol, ts), group in latest.group_by(["symbol", "ts"], maintain_order=True):
        features = {
            str(row["feature_name"]): row["value"]
            for row in group.sort("feature_name").to_dicts()
        }
        rows.append(LatestFeatureRow(symbol=str(symbol), ts=ts, features=features))
    return LatestFeaturesResponse(
        feature_set=feature_set,
        feature_version=version,
        timeframe=timeframe,
        created_at=created_at,
        rows=rows,
        warnings=[],
    )


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    return parsed or None


def _load_latest_alpha_evidence(lake_root: Path, alpha_id: str) -> dict[str, Any] | None:
    df = read_parquet_dataset(lake_root / "gold" / "alpha_evidence")
    if df.is_empty() or "alpha_id" not in df.columns:
        return None
    filtered = df.filter(pl.col("alpha_id") == alpha_id)
    if filtered.is_empty():
        return None
    return _latest_row(filtered, "created_at")


def _load_latest_gate_decision_for_alpha(lake_root: Path, alpha_id: str) -> GateDecision | None:
    df = read_parquet_dataset(lake_root / "gold" / "gate_decision")
    if df.is_empty() or "alpha_id" not in df.columns:
        return None
    filtered = df.filter(pl.col("alpha_id") == alpha_id)
    if filtered.is_empty():
        return None
    row = _latest_row(filtered, "created_at")
    return _gate_decision_from_row(row)


def _missing_gate_decision(alpha_id: str) -> GateDecision:
    return GateDecision(
        alpha_id=alpha_id,
        version="unknown",
        gate_version="missing-gate-decision-v0.1",
        status="QUARANTINE",
        passed=False,
        reasons=["missing_gate_decision"],
        metrics={},
        next_action="build_alpha_evidence_before_gate",
        created_at=datetime.now(UTC),
    )


def _load_gate_decisions(lake_root: Path, strategy: str) -> list[GateDecision]:
    df = read_parquet_dataset(lake_root / "gold" / "gate_decision")
    if df.is_empty():
        return []
    if "strategy" in df.columns:
        df = df.filter(pl.col("strategy") == strategy)
    rows = df.to_dicts()
    decisions: list[GateDecision] = []
    for row in rows:
        cleaned = dict(row)
        cleaned.pop("strategy", None)
        cleaned.pop("source", None)
        cleaned.pop("fallback_level", None)
        parsed = _gate_decision_from_row(cleaned)
        if parsed is not None:
            decisions.append(parsed)
    return _prefer_research_gate_decisions(decisions)


def _prefer_research_gate_decisions(decisions: list[GateDecision]) -> list[GateDecision]:
    research_decisions = [
        decision
        for decision in decisions
        if not decision.gate_version.startswith("bootstrap.")
    ]
    return research_decisions or decisions


def _gate_decision_from_row(row: dict[str, Any]) -> GateDecision | None:
    cleaned = dict(row)
    cleaned.pop("strategy", None)
    cleaned.pop("source", None)
    cleaned.pop("fallback_level", None)
    if isinstance(cleaned.get("metrics"), str):
        cleaned["metrics"] = _json_dict(cleaned["metrics"])
    if isinstance(cleaned.get("reasons"), str):
        cleaned["reasons"] = _json_list(cleaned["reasons"])
    try:
        return GateDecision.model_validate(cleaned)
    except Exception:
        return None


def _select_published_risk_permission(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
    telemetry_latest_ts: datetime | None,
) -> dict[str, RiskPermission | None]:
    df = read_parquet_dataset(lake_root / "gold" / "risk_permission")
    if df.is_empty():
        return {"active": None, "stale": None, "gold_latest": None}
    filtered = df
    if "strategy" in filtered.columns:
        filtered = filtered.filter(pl.col("strategy") == strategy)
    elif "strategy_id" in filtered.columns:
        filtered = filtered.filter(pl.col("strategy_id") == strategy)
    if "version" in filtered.columns:
        filtered = filtered.filter(pl.col("version") == version)
    if filtered.is_empty():
        return {"active": None, "stale": None, "gold_latest": None}

    parsed = [
        permission
        for row in filtered.to_dicts()
        if (permission := parse_risk_permission_row(row)) is not None
    ]
    if not parsed:
        return {"active": None, "stale": None, "gold_latest": None}

    normalized = [
        _normalize_published_risk_permission(
            permission,
            telemetry_latest_ts=telemetry_latest_ts,
        )
        for permission in parsed
    ]
    gold_latest = _latest_permission(normalized)
    active_candidates = [
        permission for permission in normalized if _published_permission_is_active(permission)
    ]
    stale_candidates = [
        permission for permission in normalized if not _published_permission_is_active(permission)
    ]
    return {
        "active": _latest_permission(active_candidates),
        "stale": _latest_permission(stale_candidates),
        "gold_latest": gold_latest,
    }


def _normalize_published_risk_permission(
    permission: RiskPermission,
    *,
    telemetry_latest_ts: datetime | None,
) -> RiskPermission:
    as_of = _ensure_utc(permission.as_of_ts or permission.created_at)
    latest_telemetry = _latest_datetime(
        [
            telemetry_latest_ts,
            permission.telemetry_latest_ts,
            permission.source_bundle_ts,
        ]
    )
    expires_at = _ensure_utc(permission.expires_at)
    if expires_at is None and as_of is not None:
        expires_at = as_of + timedelta(seconds=max(_risk_permission_ttl_seconds(), 0))
    expired = expires_at is not None and expires_at < datetime.now(UTC)
    stale_vs_telemetry = risk_permission_stale_vs_telemetry(
        as_of_ts=as_of,
        telemetry_latest_ts=latest_telemetry,
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    )
    status_value = permission_status(
        permission.permission.value,
        stale=stale_vs_telemetry,
        expired=expired,
    )
    reasons = list(permission.reasons)
    if expired and "permission_expired" not in reasons:
        reasons.append("permission_expired")
    if stale_vs_telemetry and "risk_permission_stale_vs_v5_telemetry" not in reasons:
        reasons.append("risk_permission_stale_vs_v5_telemetry")
    return permission.model_copy(
        update={
            "as_of_ts": as_of,
            "expires_at": expires_at,
            "telemetry_latest_ts": latest_telemetry,
            "permission_freshness_sec": _telemetry_freshness_seconds(as_of, latest_telemetry),
            "permission_status": status_value,
            "enforceable": is_permission_status_enforceable(status_value) and not expired,
            "risk_reason_codes": reasons,
            "reasons": reasons,
            "reason": reasons[0] if reasons else None,
        }
    )


def _published_permission_is_active(permission: RiskPermission) -> bool:
    return (
        is_permission_status_enforceable(permission.permission_status)
        and not _permission_expired(permission)
    )


def _permission_expired(permission: RiskPermission) -> bool:
    return permission.expires_at is not None and permission.expires_at < datetime.now(UTC)


def _force_non_enforceable_permission(
    permission: RiskPermission,
    *,
    telemetry_latest_ts: datetime | None,
    reason: str,
) -> RiskPermission:
    as_of = _ensure_utc(permission.as_of_ts or permission.created_at)
    latest_telemetry = _latest_datetime(
        [telemetry_latest_ts, permission.telemetry_latest_ts, permission.source_bundle_ts]
    )
    reasons = _dedupe([*permission.reasons, reason])
    expired = _permission_expired(permission)
    status_value = permission_status(
        permission.permission.value,
        stale=not expired,
        expired=expired,
    )
    return permission.model_copy(
        update={
            "as_of_ts": as_of,
            "telemetry_latest_ts": latest_telemetry,
            "permission_freshness_sec": _telemetry_freshness_seconds(as_of, latest_telemetry),
            "permission_status": status_value,
            "enforceable": False,
            "risk_reason_codes": reasons,
            "reasons": reasons,
            "reason": reasons[0] if reasons else None,
        }
    )


def _force_no_fresh_permission(
    permission: RiskPermission,
    *,
    telemetry_latest_ts: datetime | None,
) -> RiskPermission:
    reasons = list(permission.reasons)
    reason = "no_fresh_published_permission"
    risk_reason_codes = _dedupe([*permission.risk_reason_codes, *reasons, reason])
    return permission.model_copy(
        update={
            "allowed_modes": [],
            "max_gross_exposure": 0.0,
            "max_single_weight": 0.0,
            "max_gross_exposure_usdt": 0.0,
            "max_single_order_usdt": 0.0,
            "telemetry_latest_ts": _ensure_utc(telemetry_latest_ts),
            "permission_status": RiskPermissionStatus.NO_FRESH_PERMISSION,
            "enforceable": False,
            "risk_reason_codes": risk_reason_codes,
            "reason": reason,
        }
    )


def _latest_permission(permissions: list[RiskPermission]) -> RiskPermission | None:
    if not permissions:
        return None
    return sorted(
        permissions,
        key=lambda permission: (
            _datetime_sort_value(permission.as_of_ts or permission.created_at),
            _datetime_sort_value(permission.source_bundle_ts),
        ),
    )[-1]


def _risk_permission_api_consistency(
    *,
    response_permission: RiskPermission,
    gold_latest: RiskPermission | None,
) -> dict[str, Any]:
    response_as_of = response_permission.as_of_ts or response_permission.created_at
    gold_as_of = gold_latest.as_of_ts or gold_latest.created_at if gold_latest is not None else None
    status_value = "PASS"
    if gold_as_of is not None and response_as_of < gold_as_of:
        status_value = "FAIL"
    lag_seconds = (
        max(0, int((gold_as_of - response_as_of).total_seconds()))
        if gold_as_of is not None
        else None
    )
    return {
        "compare_gold_latest_vs_api_response": status_value,
        "api_response_as_of_ts": response_as_of.isoformat(),
        "gold_latest_as_of_ts": gold_as_of.isoformat() if gold_as_of else None,
        "api_permission_as_of_ts": response_as_of.isoformat(),
        "gold_permission_as_of_ts": gold_as_of.isoformat() if gold_as_of else None,
        "permission_api_lag_sec": lag_seconds,
        "permission_api_consistent_with_gold": status_value == "PASS",
        "api_response_permission_status": str(response_permission.permission_status or ""),
        "gold_latest_permission_status": str(gold_latest.permission_status or "")
        if gold_latest
        else None,
    }


def _risk_permission_health(
    *,
    response_permission: RiskPermission,
    gold_latest: RiskPermission | None,
    stale_permission_consecutive_count: int = 0,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    response_as_of = response_permission.as_of_ts or response_permission.created_at
    response_expires_at = response_permission.expires_at
    gold_as_of = gold_latest.as_of_ts or gold_latest.created_at if gold_latest else None
    refresh_lag = max(0, int((now - gold_as_of).total_seconds())) if gold_as_of else None
    return {
        "latest_permission_status": str(response_permission.permission_status or ""),
        "permission_age_sec": max(0, int((now - response_as_of).total_seconds())),
        "expires_in_sec": int((response_expires_at - now).total_seconds())
        if response_expires_at
        else None,
        "permission_refresh_lag_sec": refresh_lag,
        "refresh_lag_sec": refresh_lag,
        "stale_permission_consecutive_count": stale_permission_consecutive_count,
        "gold_latest_permission_status": str(gold_latest.permission_status or "")
        if gold_latest
        else None,
    }


def _stale_permission_consecutive_count(lake_root: Path, strategy: str) -> int:
    health = read_parquet_dataset(lake_root / "gold" / "strategy_health_daily")
    if health.is_empty() or "stale_permission_consecutive_count" not in health.columns:
        return 0
    scoped = health
    if "strategy" in scoped.columns:
        scoped = scoped.filter(pl.col("strategy") == strategy)
    if scoped.is_empty():
        return 0
    for column in ["latest_bundle_ts", "created_at", "date"]:
        if column in scoped.columns:
            scoped = _normalize_datetime_column(scoped, column).sort(column)
            break
    latest = scoped.tail(1).to_dicts()[0]
    try:
        return max(0, int(latest.get("stale_permission_consecutive_count") or 0))
    except (TypeError, ValueError):
        return 0


def _datetime_sort_value(value: datetime | None) -> float:
    normalized = _ensure_utc(value)
    return normalized.timestamp() if normalized is not None else 0.0


def _latest_datetime(values: list[datetime | None]) -> datetime | None:
    parsed = [_ensure_utc(value) for value in values]
    clean = [value for value in parsed if value is not None]
    return max(clean) if clean else None


def _telemetry_freshness_seconds(
    as_of_ts: datetime | None,
    telemetry_latest_ts: datetime | None,
) -> int:
    if as_of_ts is None or telemetry_latest_ts is None or telemetry_latest_ts <= as_of_ts:
        return 0
    return int((telemetry_latest_ts - as_of_ts).total_seconds())


def _ensure_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)


def _lake_cost_health(lake_root: Path) -> dict[str, Any]:
    health = read_cost_health_daily(lake_root)
    if health.get("rows"):
        status = str(health.get("status") or "").lower()
        return {
            "status": status,
            "missing": status == "critical" and int(health.get("actual_rows") or 0) == 0,
            "high_fallback": float(health.get("fallback_ratio") or 0.0) > 0.5,
            "fallback_ratio": float(health.get("fallback_ratio") or 0.0),
            "cost_model_version": str(health.get("cost_model_version") or "cost_health_daily"),
        }
    df = read_parquet_dataset(lake_root / "gold" / "cost_bucket_daily")
    if df.is_empty():
        return {
            "status": "missing",
            "missing": True,
            "cost_model_version": "missing",
        }
    latest_day = _latest_string(df, "day")
    status = "ok"
    health: dict[str, Any] = {
        "status": status,
        "cost_model_version": f"cost_bucket_daily:{latest_day or 'unknown'}",
    }
    if _all_cost_rows_global_default(df):
        health.update(
            {
                "status": "missing",
                "missing": True,
                "cost_model_version": "bootstrap.cost.v1",
            }
        )
    if latest_day and _is_day_stale(latest_day, max_age_days=3):
        health.update({"status": "stale", "stale": True})
    return health


def _lake_data_health(lake_root: Path) -> dict[str, Any]:
    df = read_parquet_dataset(lake_root / "silver" / "market_bar")
    if df.is_empty() or "ts" not in df.columns:
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_missing"],
        }
    normalized = _normalize_datetime_column(df, "ts")
    latest_ts = normalized.select(pl.col("ts").max()).item()
    if not isinstance(latest_ts, datetime):
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_invalid_timestamp"],
        }
    latest_utc = latest_ts.astimezone(UTC)
    if latest_utc < datetime.now(UTC) - timedelta(hours=24):
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_stale"],
            "latest_market_bar_ts": latest_utc.isoformat(),
        }
    return {
        "status": "ok",
        "latest_market_bar_ts": latest_utc.isoformat(),
        "allowed_modes": ["paper", "live_canary"],
        "max_gross_exposure": 0.25,
        "max_single_weight": 0.05,
    }


def _strategy_telemetry_reasons(lake_root: Path, strategy: str) -> list[str]:
    reasons: list[str] = []
    compliance = read_parquet_dataset(lake_root / "gold" / "v5_gate_compliance_daily")
    if strategy == "v5" and not compliance.is_empty():
        latest = _latest_row(compliance, "date")
        violation_count = int(latest.get("violation_count") or 0)
        if violation_count > 0:
            reasons.append("v5_gate_compliance_violation")

    health = read_parquet_dataset(lake_root / "gold" / "strategy_health_daily")
    if strategy == "v5" and not health.is_empty():
        latest = _latest_row(health, "date")
        if str(latest.get("status") or "").upper() == "CRITICAL":
            reasons.append("v5_strategy_health_critical")
    return reasons


def _has_bootstrap_gate(gate_decisions: list[GateDecision]) -> bool:
    return any(
        decision.gate_version == BOOTSTRAP_GATE_VERSION
        and decision.status.value == "QUARANTINE"
        for decision in gate_decisions
    )


def _data_health_is_critical(data_health: dict[str, Any]) -> bool:
    return bool(data_health.get("is_critical")) or str(data_health.get("status")) == "critical"


def _all_cost_rows_global_default(df: pl.DataFrame) -> bool:
    if df.is_empty():
        return False
    expressions: list[pl.Expr] = []
    if "source" in df.columns:
        expressions.append(pl.col("source") == "global_default")
    if "fallback_level" in df.columns:
        expressions.append(pl.col("fallback_level") == "GLOBAL_DEFAULT")
    if not expressions:
        return False
    condition = expressions[0]
    for expression in expressions[1:]:
        condition = condition | expression
    return df.filter(condition).height == df.height


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _latest_row(df: pl.DataFrame, sort_column: str) -> dict[str, Any]:
    if sort_column in df.columns:
        return df.sort(sort_column).tail(1).to_dicts()[0]
    return df.tail(1).to_dicts()[0]


def _latest_string(df: pl.DataFrame, column: str) -> str | None:
    if df.is_empty() or column not in df.columns:
        return None
    return str(df.select(pl.col(column).max()).item())


def _is_day_stale(day: str, max_age_days: int) -> bool:
    try:
        day_date = datetime.fromisoformat(day).date()
    except ValueError:
        return True
    return day_date < (datetime.now(UTC).date() - timedelta(days=max_age_days))


def _normalize_datetime_column(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if df.schema.get(column) == pl.String:
        return df.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=False))
    return df.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column))


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
