"""Read-only FastAPI surface for quant-lab strategy consumers.

This API is read-only and must not mutate strategy state or exchange state.
"""

import csv
import hmac
import io
import json
import os
import subprocess
import threading
import time
import zipfile
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.models import (
    CostEstimate,
    GateDecision,
    MarketBar,
    RiskPermission,
    RiskPermissionStatus,
)
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.costs.health import read_cost_health_daily
from quant_lab.costs.model import CostBucket, estimate_cost_bps, estimate_cost_from_lake
from quant_lab.data.lake import (
    read_market_bars,
    read_parquet_dataset,  # noqa: F401 - kept as a monkeypatch guard for no-eager-read tests.
    read_parquet_lazy,
)
from quant_lab.gates.defaults import conservative_example_gate_decision
from quant_lab.ops.metrics import api_metrics_summary, record_api_request
from quant_lab.research.advisory_overrides import (
    portfolio_overridden_decision_mode,
    portfolio_override_for_row,
    portfolio_status_overrides_by_identifier,
)
from quant_lab.research.alpha_factory import ALPHA_FACTORY_CANDIDATES
from quant_lab.research.bootstrap_gold import BOOTSTRAP_GATE_VERSION
from quant_lab.risk.advisory import (
    apply_risk_advisory_context,
    build_risk_advisory_context,
    required_gate_decisions,
)
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
from quant_lab.symbols import normalize_symbol

STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION = "strategy_opportunity_advisory.v0.1"
STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS = 3 * 60 * 60
STRATEGY_OPPORTUNITY_ADVISORY_MAX_API_ROWS = 50_000
UNOBSERVABLE_TEXT_VALUES = {"", "none", "null", "nan", "not_observable", "unknown"}
_STRATEGY_OPPORTUNITY_ADVISORY_CACHE_LOCK = threading.Lock()
_STRATEGY_OPPORTUNITY_ADVISORY_CACHE: dict[
    str,
    tuple[tuple[Any, ...], tuple["StrategyOpportunityAdvisoryRow", ...], bytes | None],
] = {}


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
    latency_by_path_ms: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    slow_paths: list[dict[str, float | int | str | None]] = Field(default_factory=list)


class StrategyOpportunityAdvisoryRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_ts: datetime
    generated_at: datetime
    expires_at: datetime
    contract_version: str
    schema_version: str = STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION
    quant_lab_git_commit: str | None = None
    source_version: str
    would_block_if_enabled: bool = False
    would_enter: bool = False
    no_sample_reason: str | None = None
    strategy_id: str
    strategy_candidate: str
    symbol: str
    decision: str
    recommended_mode: str
    horizon_hours: int | None = None
    sample_count: int | None = None
    complete_sample_count: int | None = None
    avg_net_bps: float | None = None
    p25_net_bps: float | None = None
    win_rate: float | None = None
    cost_source_mix: str | None = None
    source_module: str | None = None
    template_family: str | None = None
    candidate_id: str | None = None
    promotion_state: str | None = None
    alpha_factory_score: float | None = None
    universe_type: str | None = None
    cost_quality_score: float | None = None
    paper_ready_block_reasons: list[str] = Field(default_factory=list)
    advisory_intent: str = "research_only"
    live_block_reasons: list[str] = Field(default_factory=list)
    max_paper_notional_usdt: float | None = None
    max_live_notional_usdt: float = 0.0


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
        return conservative_example_gate_decision()

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

    @app.get(
        "/v1/strategy-opportunity-advisory",
        response_model=list[StrategyOpportunityAdvisoryRow],
    )
    @app.get(
        "/v1/strategy_opportunity_advisory",
        response_model=list[StrategyOpportunityAdvisoryRow],
    )
    @app.get(
        "/v1/reports/strategy-opportunity-advisory",
        response_model=list[StrategyOpportunityAdvisoryRow],
    )
    def strategy_opportunity_advisory() -> Response:
        return _strategy_opportunity_advisory_response(_lake_root())

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
        telemetry_latest_ts=telemetry_latest_ts,
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
    permission = apply_risk_advisory_context(
        permission,
        computed["advisory_context"],
    )
    if not is_permission_status_enforceable(permission.permission_status):
        permission = _clear_non_enforceable_live_modes(permission)
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
    telemetry_latest_ts: datetime | None = None,
) -> dict[str, Any]:
    gate_decisions = _load_gate_decisions(lake_root, strategy=strategy)
    data_health = _lake_data_health(lake_root)
    cost_health = _lake_cost_health(lake_root)
    telemetry_reasons = _strategy_telemetry_reasons(lake_root, strategy=strategy)
    if telemetry_latest_ts is None:
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
            gate_decisions=required_gate_decisions(gate_decisions),
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
        advisory_context = build_risk_advisory_context(
            lake_root,
            gate_decisions=gate_decisions,
            data_health=data_health,
            cost_health=cost_health,
            telemetry_reasons=telemetry_reasons,
        )
        permission = apply_risk_advisory_context(permission, advisory_context)
        return {
            "permission": annotate_risk_permission(
                permission,
                telemetry_latest_ts=telemetry_latest_ts,
                as_of_ts=datetime.now(UTC),
                threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
            ),
            "advisory_context": advisory_context,
            "data_health": data_health,
            "cost_health": cost_health,
            "gate_summary": _gate_summary(gate_decisions),
            "v5_telemetry_summary": _v5_telemetry_summary(telemetry_reasons),
            "original_data_health": original_data_health,
        }

    permission = evaluate_live_permission(
        strategy=strategy,
        version=version,
        gate_decisions=required_gate_decisions(gate_decisions),
        cost_health=cost_health,
        data_health=data_health,
    )
    advisory_context = build_risk_advisory_context(
        lake_root,
        gate_decisions=gate_decisions,
        data_health=data_health,
        cost_health=cost_health,
        telemetry_reasons=telemetry_reasons,
    )
    permission = apply_risk_advisory_context(permission, advisory_context)
    return {
        "permission": annotate_risk_permission(
            permission,
            telemetry_latest_ts=telemetry_latest_ts,
            as_of_ts=datetime.now(UTC),
            threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
        ),
        "advisory_context": advisory_context,
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
    created_at = datetime.now(UTC)
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "feature_value")
    required_columns = {
        "feature_set",
        "feature_name",
        "feature_version",
        "symbol",
        "ts",
        "value",
    }
    if lazy is None:
        return LatestFeaturesResponse(
            feature_set=feature_set,
            feature_version=feature_version,
            timeframe=timeframe,
            created_at=created_at,
            rows=[],
            warnings=["feature_value dataset is missing or empty"],
        )
    missing_columns = sorted(required_columns.difference(columns))
    if missing_columns:
        return LatestFeaturesResponse(
            feature_set=feature_set,
            feature_version=feature_version,
            timeframe=timeframe,
            created_at=created_at,
            rows=[],
            warnings=[f"feature_value missing required columns: {','.join(missing_columns)}"],
        )

    filtered = lazy.filter(pl.col("feature_set") == feature_set).with_columns(
        _lazy_utc_datetime("ts").alias("ts")
    )
    if "created_at" in columns:
        filtered = filtered.with_columns(_lazy_utc_datetime("created_at").alias("created_at"))
    if feature_version is None:
        version_frame = _collect_lazy_or_empty(
            filtered.select(pl.col("feature_version").cast(pl.Utf8).max().alias("feature_version"))
        )
        version_value = (
            version_frame.item(0, "feature_version") if not version_frame.is_empty() else None
        )
        version = str(version_value) if version_value is not None else None
        if version:
            filtered = filtered.filter(pl.col("feature_version") == version)
    else:
        version = feature_version
        filtered = filtered.filter(pl.col("feature_version") == feature_version)
    if timeframe:
        filtered = filtered.filter(pl.col("timeframe") == timeframe)
    if symbols:
        filtered = filtered.filter(
            pl.col("symbol").is_in([normalize_symbol(symbol) for symbol in symbols])
        )
    if feature_names:
        filtered = filtered.filter(pl.col("feature_name").is_in(feature_names))
    if "created_at" in columns:
        latest_created_frame = _collect_lazy_or_empty(
            filtered.select(pl.col("created_at").max().alias("created_at"))
        )
        latest_created = (
            latest_created_frame.item(0, "created_at")
            if not latest_created_frame.is_empty()
            else None
        )
        if isinstance(latest_created, datetime):
            created_at = latest_created.astimezone(UTC)
    latest_ts = filtered.group_by("symbol").agg(pl.col("ts").max().alias("ts"))
    latest = _collect_lazy_or_empty(filtered.join(latest_ts, on=["symbol", "ts"], how="inner"))
    if latest.is_empty():
        return LatestFeaturesResponse(
            feature_set=feature_set,
            feature_version=version,
            timeframe=timeframe,
            created_at=created_at,
            rows=[],
            warnings=["no feature rows matched query"],
        )
    rows: list[LatestFeatureRow] = []
    for (symbol, ts), group in latest.group_by(["symbol", "ts"], maintain_order=True):
        features = {
            str(row["feature_name"]): row["value"] for row in group.sort("feature_name").to_dicts()
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


def _strategy_opportunity_advisory_rows(
    lake_root: Path,
) -> list[StrategyOpportunityAdvisoryRow]:
    signature = _strategy_opportunity_advisory_source_signature(lake_root)
    cache_key = str(lake_root.resolve())
    with _STRATEGY_OPPORTUNITY_ADVISORY_CACHE_LOCK:
        cached = _STRATEGY_OPPORTUNITY_ADVISORY_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature:
            return list(cached[1])
    rows = _strategy_opportunity_advisory_rows_uncached(lake_root)
    with _STRATEGY_OPPORTUNITY_ADVISORY_CACHE_LOCK:
        _STRATEGY_OPPORTUNITY_ADVISORY_CACHE[cache_key] = (signature, tuple(rows), None)
    return rows


def _strategy_opportunity_advisory_response(lake_root: Path) -> Response:
    signature = _strategy_opportunity_advisory_source_signature(lake_root)
    cache_key = str(lake_root.resolve())
    with _STRATEGY_OPPORTUNITY_ADVISORY_CACHE_LOCK:
        cached = _STRATEGY_OPPORTUNITY_ADVISORY_CACHE.get(cache_key)
        if cached is not None and cached[0] == signature and cached[2] is not None:
            return Response(content=cached[2], media_type="application/json")
    rows = _strategy_opportunity_advisory_rows(lake_root)
    payload = _strategy_opportunity_advisory_json_bytes(rows)
    with _STRATEGY_OPPORTUNITY_ADVISORY_CACHE_LOCK:
        _STRATEGY_OPPORTUNITY_ADVISORY_CACHE[cache_key] = (signature, tuple(rows), payload)
    return Response(content=payload, media_type="application/json")


def _strategy_opportunity_advisory_json_bytes(
    rows: list[StrategyOpportunityAdvisoryRow],
) -> bytes:
    return b"[" + b",".join(row.model_dump_json().encode("utf-8") for row in rows) + b"]"


def _strategy_opportunity_advisory_rows_uncached(
    lake_root: Path,
) -> list[StrategyOpportunityAdvisoryRow]:
    raw_rows: list[dict[str, Any]]
    raw_rows = _strategy_opportunity_advisory_gold_rows(lake_root)
    if not raw_rows:
        raw_rows = _strategy_opportunity_advisory_report_rows(lake_root)
    parsed: list[StrategyOpportunityAdvisoryRow] = []
    for row in raw_rows:
        advisory = _strategy_opportunity_advisory_row(row)
        if advisory is not None:
            parsed.append(advisory)
    deduped = _dedupe_strategy_opportunity_advisory_rows(parsed)
    promotion_overridden = _apply_alpha_factory_promotion_overrides_to_advisory_rows(
        deduped,
        _strategy_opportunity_alpha_factory_promotions(lake_root),
    )
    return _apply_research_portfolio_overrides_to_advisory_rows(
        promotion_overridden,
        _strategy_opportunity_portfolio_overrides(lake_root),
    )


def _strategy_opportunity_advisory_source_signature(lake_root: Path) -> tuple[Any, ...]:
    gold_path = lake_root / "gold" / "strategy_opportunity_advisory"
    gold_files = _parquet_file_signature(gold_path)
    portfolio_files = _parquet_file_signature(lake_root / "gold" / "research_portfolio_status")
    promotion_files = _parquet_file_signature(lake_root / "gold" / "alpha_factory_promotion_queue")
    if gold_files:
        return (
            "gold",
            gold_files,
            "research_portfolio_status",
            portfolio_files,
            "alpha_factory_promotion_queue",
            promotion_files,
        )
    latest_report = _latest_strategy_opportunity_report_pack(lake_root)
    if latest_report is not None:
        try:
            stat = latest_report.stat()
        except OSError:
            return (
                "report",
                str(latest_report),
                None,
                None,
                "research_portfolio_status",
                portfolio_files,
                "alpha_factory_promotion_queue",
                promotion_files,
            )
        return (
            "report",
            str(latest_report),
            stat.st_mtime_ns,
            stat.st_size,
            "research_portfolio_status",
            portfolio_files,
            "alpha_factory_promotion_queue",
            promotion_files,
        )
    return (
        "missing",
        "research_portfolio_status",
        portfolio_files,
        "alpha_factory_promotion_queue",
        promotion_files,
    )


def _parquet_file_signature(path: Path) -> tuple[tuple[str, int, int], ...]:
    if path.is_file() and path.suffix == ".parquet":
        candidates = [path]
    elif path.exists() and path.is_dir():
        candidates = sorted(
            file_path
            for file_path in path.rglob("*.parquet")
            if file_path.is_file() and not _is_internal_api_lake_path(file_path)
        )
    else:
        return ()
    signature: list[tuple[str, int, int]] = []
    for file_path in candidates:
        try:
            stat = file_path.stat()
        except OSError:
            continue
        relative_to = path if path.is_dir() else path.parent
        signature.append(
            (str(file_path.relative_to(relative_to)), stat.st_mtime_ns, stat.st_size)
        )
    return tuple(signature)


def _is_internal_api_lake_path(path: Path) -> bool:
    return any(
        part == "._tmp" or part.startswith("__") or part.startswith(".")
        for part in path.parts
    )


def _strategy_opportunity_advisory_gold_rows(lake_root: Path) -> list[dict[str, Any]]:
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "strategy_opportunity_advisory")
    if lazy is None:
        return []
    selected = lazy.select([pl.col(column) for column in columns])
    timestamp_column = next(
        (
            column
            for column in ("generated_at", "as_of_ts", "created_at", "as_of_date")
            if column in columns
        ),
        None,
    )
    if timestamp_column is not None:
        selected = (
            selected.with_columns(
                _lazy_utc_datetime(timestamp_column).alias("__api_advisory_sort_ts")
            )
            .sort("__api_advisory_sort_ts", descending=True, nulls_last=True)
            .limit(STRATEGY_OPPORTUNITY_ADVISORY_MAX_API_ROWS)
            .drop("__api_advisory_sort_ts")
        )
    else:
        selected = selected.tail(STRATEGY_OPPORTUNITY_ADVISORY_MAX_API_ROWS)
    frame = _collect_lazy_or_empty(selected)
    return frame.to_dicts()


def _dedupe_strategy_opportunity_advisory_rows(
    rows: list[StrategyOpportunityAdvisoryRow],
) -> list[StrategyOpportunityAdvisoryRow]:
    selected: dict[tuple[str, str, int | None], StrategyOpportunityAdvisoryRow] = {}
    for row in rows:
        key = (row.strategy_candidate, row.symbol, row.horizon_hours)
        current = selected.get(key)
        if current is None or _advisory_row_preferred(row, current):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            row.strategy_candidate,
            row.symbol,
            row.horizon_hours if row.horizon_hours is not None else -1,
        ),
    )


def _strategy_opportunity_portfolio_overrides(
    lake_root: Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "research_portfolio_status")
    if lazy is None:
        return {}
    selected = lazy.select([pl.col(column) for column in columns])
    frame = _collect_lazy_or_empty(selected)
    return portfolio_status_overrides_by_identifier(frame)


def _strategy_opportunity_alpha_factory_promotions(
    lake_root: Path,
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "alpha_factory_promotion_queue")
    if lazy is None:
        return {}
    selected = lazy.select([pl.col(column) for column in columns])
    frame = _collect_lazy_or_empty(selected)
    return _alpha_factory_promotion_overrides_by_candidate_symbol(frame)


def _alpha_factory_promotion_overrides_by_candidate_symbol(
    promotion_queue: pl.DataFrame,
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    overrides: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    if promotion_queue.is_empty():
        return overrides
    rows = promotion_queue.to_dicts()
    latest_as_of_date = _latest_advisory_as_of_date(rows)
    if latest_as_of_date:
        rows = [
            row
            for row in rows
            if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    for row in rows:
        candidate = _text_value(row.get("strategy_candidate"))
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if not candidate:
            continue
        key = (candidate, symbol, _optional_int(row.get("horizon_hours")))
        current = overrides.get(key)
        if current is None or _advisory_dict_time(row) >= _advisory_dict_time(current):
            overrides[key] = row
    return overrides


def _latest_advisory_as_of_date(rows: list[dict[str, Any]]) -> str:
    values = [
        str(row.get("as_of_date") or "").strip()
        for row in rows
        if str(row.get("as_of_date") or "").strip()
    ]
    return max(values) if values else ""


def _advisory_dict_time(row: dict[str, Any]) -> datetime:
    for field in ("generated_at", "created_at", "as_of_ts", "as_of_date"):
        parsed = _parse_advisory_datetime(row.get(field))
        if parsed is not None:
            return parsed
    return datetime.min.replace(tzinfo=UTC)


def _is_alpha_factory_advisory_model(row: StrategyOpportunityAdvisoryRow) -> bool:
    candidate = row.strategy_candidate
    return (
        candidate in ALPHA_FACTORY_CANDIDATES
        or candidate.startswith("v5.af.")
        or candidate.startswith("v5.expanded_relative_strength")
        or (row.source_module or "").lower() == "alpha_factory"
        or bool(row.template_family)
        or row.alpha_factory_score is not None
    )


def _alpha_factory_promotion_for_advisory_row(
    row: StrategyOpportunityAdvisoryRow,
    promotions: dict[tuple[str, str, int | None], dict[str, Any]],
) -> dict[str, Any] | None:
    for key in [
        (row.strategy_candidate, row.symbol, row.horizon_hours),
        (row.strategy_candidate, row.symbol, None),
        (row.strategy_candidate, "UNKNOWN", row.horizon_hours),
        (row.strategy_candidate, "UNKNOWN", None),
    ]:
        promotion = promotions.get(key)
        if promotion is not None:
            return promotion
    return None


def _alpha_factory_promotion_decision_mode(
    promotion: dict[str, Any],
) -> tuple[str, str, list[str]]:
    state = _text_value(
        promotion.get("promotion_state")
        or promotion.get("decision")
        or "RESEARCH"
    ).upper()
    decision = "RESEARCH_ONLY" if state == "RESEARCH" else state
    mode = _text_value(promotion.get("recommended_mode")).lower()
    if not mode:
        mode = _advisory_recommended_mode(None, decision)
    reasons = _advisory_reason_list(promotion.get("reasons"))
    if decision != "PAPER_READY":
        reasons.append("alpha_factory_promotion_queue_not_paper_ready")
    return decision, mode, sorted(set(reasons))


def _apply_alpha_factory_promotion_overrides_to_advisory_rows(
    rows: list[StrategyOpportunityAdvisoryRow],
    promotions: dict[tuple[str, str, int | None], dict[str, Any]],
) -> list[StrategyOpportunityAdvisoryRow]:
    output: list[StrategyOpportunityAdvisoryRow] = []
    for row in rows:
        if not _is_alpha_factory_advisory_model(row):
            output.append(row.model_copy(update={"max_live_notional_usdt": 0.0}))
            continue
        promotion = _alpha_factory_promotion_for_advisory_row(row, promotions)
        if promotion is None:
            paper_decision = row.decision in {"PAPER_READY", "LIVE_SMALL_READY"}
            paper_mode = row.recommended_mode in {"paper", "live_small"}
            if not paper_decision and not paper_mode:
                output.append(row.model_copy(update={"max_live_notional_usdt": 0.0}))
                continue
            decision = "KEEP_SHADOW"
            mode = "shadow"
            reasons = ["alpha_factory_promotion_queue_missing"]
            promotion_state = row.promotion_state or "KEEP_SHADOW"
        else:
            decision, mode, reasons = _alpha_factory_promotion_decision_mode(promotion)
            promotion_state = _text_value(
                promotion.get("promotion_state")
                or promotion.get("decision")
                or decision
            ).upper()
        live_block_reasons = sorted({*row.live_block_reasons, *reasons})
        paper_ready_block_reasons = sorted(
            {
                *row.paper_ready_block_reasons,
                *(reason for reason in reasons if decision != "PAPER_READY"),
            }
        )
        no_sample_reason = row.no_sample_reason
        if mode in {"research", "shadow"} and not no_sample_reason:
            no_sample_reason = "shadow_only" if mode == "shadow" else "research_only"
        output.append(
            row.model_copy(
                update={
                    "decision": decision,
                    "recommended_mode": mode,
                    "promotion_state": promotion_state,
                    "live_block_reasons": live_block_reasons,
                    "paper_ready_block_reasons": paper_ready_block_reasons,
                    "advisory_intent": _strategy_advisory_intent(mode),
                    "max_paper_notional_usdt": _api_advisory_max_paper_notional(mode),
                    "max_live_notional_usdt": 0.0,
                    "would_enter": (
                        False if mode in {"none", "research", "shadow"} else row.would_enter
                    ),
                    "no_sample_reason": no_sample_reason,
                }
            )
        )
    return output


def _apply_research_portfolio_overrides_to_advisory_rows(
    rows: list[StrategyOpportunityAdvisoryRow],
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> list[StrategyOpportunityAdvisoryRow]:
    if not overrides:
        return [row.model_copy(update={"max_live_notional_usdt": 0.0}) for row in rows]
    output: list[StrategyOpportunityAdvisoryRow] = []
    for row in rows:
        override = portfolio_override_for_row(row.model_dump(mode="python"), overrides)
        decision, mode, reasons = portfolio_overridden_decision_mode(
            decision=row.decision,
            recommended_mode=row.recommended_mode,
            portfolio_override=override,
        )
        if not reasons:
            output.append(row.model_copy(update={"max_live_notional_usdt": 0.0}))
            continue
        live_block_reasons = sorted({*row.live_block_reasons, *reasons})
        no_sample_reason = row.no_sample_reason
        if mode == "research":
            if "research_paused" in reasons:
                no_sample_reason = "research_paused"
            elif "baseline_only" in reasons:
                no_sample_reason = "baseline_only"
            elif not no_sample_reason:
                no_sample_reason = "research_only"
        output.append(
            row.model_copy(
                update={
                    "decision": decision,
                    "recommended_mode": mode,
                    "live_block_reasons": live_block_reasons,
                    "advisory_intent": _strategy_advisory_intent(mode),
                    "max_paper_notional_usdt": _api_advisory_max_paper_notional(mode),
                    "max_live_notional_usdt": 0.0,
                    "would_block_if_enabled": True
                    if decision == "KILL"
                    else row.would_block_if_enabled,
                    "would_enter": False
                    if mode in {"none", "research"}
                    else row.would_enter,
                    "no_sample_reason": no_sample_reason,
                }
            )
        )
    return output


def _api_advisory_max_paper_notional(recommended_mode: str) -> float:
    return 1000.0 if recommended_mode == "paper" else 0.0


def _advisory_row_preferred(
    candidate: StrategyOpportunityAdvisoryRow,
    current: StrategyOpportunityAdvisoryRow,
) -> bool:
    if candidate.generated_at != current.generated_at:
        return candidate.generated_at > current.generated_at
    preferred_schema = STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION
    if candidate.schema_version == preferred_schema and current.schema_version != preferred_schema:
        return True
    return candidate.as_of_ts > current.as_of_ts


def _latest_strategy_opportunity_frame(df: pl.DataFrame) -> pl.DataFrame:
    for column in ["as_of_ts", "created_at"]:
        if column not in df.columns:
            continue
        try:
            normalized = _normalize_datetime_column(df, column)
            latest = normalized.select(pl.col(column).max()).item()
        except Exception:
            continue
        if latest is not None:
            return normalized.filter(pl.col(column) == latest)
    return df


def _strategy_opportunity_advisory_report_rows(lake_root: Path) -> list[dict[str, Any]]:
    latest_report = _latest_strategy_opportunity_report_pack(lake_root)
    if latest_report is None:
        return []
    try:
        with zipfile.ZipFile(latest_report) as archive:
            with archive.open("reports/strategy_opportunity_advisory.csv") as handle:
                text = handle.read().decode("utf-8")
    except (KeyError, OSError, UnicodeDecodeError, zipfile.BadZipFile):
        return []
    return list(csv.DictReader(io.StringIO(text)))


def _latest_strategy_opportunity_report_pack(lake_root: Path) -> Path | None:
    exports_root = _exports_root_for_lake(lake_root)
    if not exports_root.exists():
        return None
    packs = sorted(
        exports_root.glob("quant_lab_expert_pack_*.zip"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    for pack in packs:
        try:
            with zipfile.ZipFile(pack) as archive:
                archive.getinfo("reports/strategy_opportunity_advisory.csv")
        except (KeyError, OSError, zipfile.BadZipFile):
            continue
        return pack
    return None


def _exports_root_for_lake(lake_root: Path) -> Path:
    return lake_root.parent / "exports" if lake_root.name == "lake" else lake_root / "exports"


def _strategy_opportunity_advisory_row(
    row: dict[str, Any],
) -> StrategyOpportunityAdvisoryRow | None:
    strategy_candidate = _text_value(row.get("strategy_candidate"))
    symbol = normalize_symbol(row.get("symbol") or row.get("v5_symbol"))
    if not strategy_candidate or not symbol:
        return None
    decision = (_text_value(row.get("decision")) or "RESEARCH_ONLY").upper()
    recommended_mode = _advisory_recommended_mode(row.get("recommended_mode"), decision)
    max_live_notional = 0.0
    as_of_ts = _advisory_datetime(
        row,
        ("as_of_ts", "generated_at", "generated_at_utc", "created_at", "as_of_date"),
    )
    generated_at = _advisory_datetime(
        row,
        ("generated_at", "generated_at_utc", "created_at", "as_of_ts", "as_of_date"),
        fallback=as_of_ts,
    )
    expires_at = _parse_advisory_datetime(row.get("expires_at"))
    if expires_at is None:
        expires_at = generated_at + timedelta(seconds=STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS)
    git_commit = _observable_text(row.get("quant_lab_git_commit")) or _git_commit()
    source_version = _observable_text(row.get("source_version")) or _source_version(
        "strategy_opportunity_advisory",
        git_commit,
    )
    sample_count = _optional_int(row.get("sample_count"))
    would_enter = _optional_bool(row.get("would_enter"))
    if would_enter is None:
        would_enter = _default_advisory_would_enter(decision, recommended_mode, sample_count)
    would_block = _optional_bool(row.get("would_block_if_enabled"))
    if would_block is None:
        would_block = _default_advisory_would_block(decision, recommended_mode, sample_count)
    no_sample_reason = _text_value(row.get("no_sample_reason")) or _default_no_sample_reason(
        decision,
        recommended_mode,
        sample_count,
    )
    strategy_id = _text_value(row.get("strategy_id")) or _advisory_strategy_id(
        strategy_candidate,
        symbol,
    )
    return StrategyOpportunityAdvisoryRow(
        as_of_ts=as_of_ts,
        generated_at=generated_at,
        expires_at=expires_at,
        contract_version=V5_QUANT_LAB_CONTRACT_VERSION,
        schema_version=_text_value(row.get("schema_version"))
        or STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION,
        quant_lab_git_commit=git_commit,
        source_version=source_version,
        would_block_if_enabled=would_block,
        would_enter=would_enter,
        no_sample_reason=no_sample_reason,
        strategy_id=strategy_id,
        strategy_candidate=strategy_candidate,
        symbol=symbol,
        decision=decision,
        recommended_mode=recommended_mode,
        horizon_hours=_optional_int(row.get("horizon_hours")),
        sample_count=_optional_int(row.get("sample_count")),
        complete_sample_count=_optional_int(row.get("complete_sample_count")),
        avg_net_bps=_optional_float(row.get("avg_net_bps")),
        p25_net_bps=_optional_float(row.get("p25_net_bps")),
        win_rate=_optional_float(row.get("win_rate")),
        cost_source_mix=_jsonish_text(row.get("cost_source_mix")),
        source_module=_text_value(row.get("source_module")),
        template_family=_text_value(row.get("template_family")),
        candidate_id=_text_value(row.get("candidate_id")),
        promotion_state=_text_value(row.get("promotion_state")),
        alpha_factory_score=_optional_float(row.get("alpha_factory_score")),
        universe_type=_text_value(row.get("universe_type")),
        cost_quality_score=_optional_float(row.get("cost_quality_score")),
        paper_ready_block_reasons=_advisory_reason_list(row.get("paper_ready_block_reasons")),
        advisory_intent=_text_value(row.get("advisory_intent"))
        or _strategy_advisory_intent(recommended_mode),
        live_block_reasons=_advisory_reason_list(row.get("live_block_reasons")),
        max_paper_notional_usdt=_optional_float(row.get("max_paper_notional_usdt")),
        max_live_notional_usdt=max_live_notional,
    )


def _advisory_recommended_mode(value: Any, decision: str) -> str:
    if decision == "KILL":
        return "none"
    if decision == "PAPER_READY":
        return "paper"
    if decision == "LIVE_SMALL_READY":
        return _text_value(value).lower() or "live_small"
    if decision in {"KEEP_SHADOW", "REGIME_SHADOW"}:
        return _text_value(value).lower() or "shadow"
    return _text_value(value).lower() or "research"


def _strategy_advisory_intent(recommended_mode: str) -> str:
    if recommended_mode in {"paper", "shadow", "live_small"}:
        return "paper_shadow"
    return "research_only"


def _default_advisory_would_enter(
    decision: str,
    recommended_mode: str,
    sample_count: int | None,
) -> bool:
    if (sample_count or 0) <= 0:
        return False
    return decision in {"PAPER_READY", "LIVE_SMALL_READY"} or recommended_mode in {
        "paper",
        "live_small",
    }


def _default_advisory_would_block(
    decision: str,
    recommended_mode: str,
    sample_count: int | None,
) -> bool:
    if decision == "KILL":
        return True
    return recommended_mode == "none" and (sample_count or 0) > 0


def _default_no_sample_reason(
    decision: str,
    recommended_mode: str,
    sample_count: int | None,
) -> str | None:
    if (sample_count or 0) <= 0:
        return "no_strategy_evidence_sample"
    if decision == "KILL":
        return "killed_candidate"
    if recommended_mode == "research":
        return "research_only"
    if recommended_mode == "shadow":
        return "shadow_only"
    return None


def _advisory_strategy_id(strategy_candidate: str, symbol: str) -> str:
    raw = f"{symbol}_{strategy_candidate}"
    normalized = "".join(character if character.isalnum() else "_" for character in raw.upper())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _advisory_datetime(
    row: dict[str, Any],
    keys: tuple[str, ...],
    *,
    fallback: datetime | None = None,
) -> datetime:
    for key in keys:
        parsed = _parse_advisory_datetime(row.get(key))
        if parsed is not None:
            return parsed
    return fallback or datetime.now(UTC)


def _parse_advisory_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    if len(text) == 10 and text[4] == "-" and text[7] == "-":
        try:
            return datetime.fromisoformat(text).replace(tzinfo=UTC)
        except ValueError:
            return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def _advisory_reason_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, dict):
        return [json.dumps(value, sort_keys=True)]
    text = str(value).strip()
    if not text:
        return []
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return [item.strip() for item in text.replace("|", ",").split(",") if item.strip()]
    if isinstance(loaded, list):
        return [str(item) for item in loaded if str(item).strip()]
    if isinstance(loaded, dict):
        return [json.dumps(loaded, sort_keys=True)]
    return [str(loaded)] if str(loaded).strip() else []


def _jsonish_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict | list):
        return json.dumps(value, sort_keys=True)
    text = str(value).strip()
    return text or None


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def _observable_text(value: Any) -> str | None:
    text = _text_value(value)
    return None if text.lower() in UNOBSERVABLE_TEXT_VALUES else text


def _source_version(component: str, git_commit: str | None) -> str:
    version = git_commit or __version__
    return f"{component}:{version}"


@lru_cache(maxsize=1)
def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            cwd=root,
            text=True,
        )
    except OSError:
        return None
    value = result.stdout.strip()
    return value or None


def _optional_float(value: Any) -> float | None:
    text = _text_value(value)
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return None if parsed != parsed else parsed


def _optional_int(value: Any) -> int | None:
    parsed = _optional_float(value)
    return None if parsed is None else int(parsed)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _load_latest_alpha_evidence(lake_root: Path, alpha_id: str) -> dict[str, Any] | None:
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "alpha_evidence")
    if lazy is None or "alpha_id" not in columns:
        return None
    filtered = lazy.filter(pl.col("alpha_id") == alpha_id)
    if "created_at" in columns:
        filtered = filtered.with_columns(_lazy_utc_datetime("created_at").alias("created_at")).sort(
            "created_at",
            descending=True,
            nulls_last=True,
        )
    row_frame = _collect_lazy_or_empty(filtered.limit(1))
    if row_frame.is_empty():
        return None
    return row_frame.to_dicts()[0]


def _load_latest_gate_decision_for_alpha(lake_root: Path, alpha_id: str) -> GateDecision | None:
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "gate_decision")
    if lazy is None or "alpha_id" not in columns:
        return None
    sort_column = "created_at" if "created_at" in columns else None
    filtered = lazy.filter(pl.col("alpha_id") == alpha_id)
    if sort_column:
        filtered = filtered.sort(sort_column, descending=True, nulls_last=True)
    row_frame = _collect_lazy_or_empty(filtered.limit(1))
    if row_frame.is_empty():
        return None
    return _gate_decision_from_row(row_frame.to_dicts()[0])


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
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "gate_decision")
    if lazy is None:
        return []
    if "strategy" in columns:
        lazy = lazy.filter(pl.col("strategy") == strategy)
    rows_frame = _collect_lazy_or_empty(lazy)
    rows = rows_frame.to_dicts()
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
        decision for decision in decisions if not decision.gate_version.startswith("bootstrap.")
    ]
    return research_decisions or decisions


def _gate_decision_from_row(row: dict[str, Any]) -> GateDecision | None:
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
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "risk_permission")
    if lazy is None:
        return {"active": None, "stale": None, "gold_latest": None}
    filtered = lazy
    if "strategy" in columns:
        filtered = filtered.filter(pl.col("strategy") == strategy)
    elif "strategy_id" in columns:
        filtered = filtered.filter(pl.col("strategy_id") == strategy)
    if "version" in columns:
        filtered = filtered.filter(pl.col("version") == version)
    frame = _collect_lazy_or_empty(filtered)
    if frame.is_empty():
        return {"active": None, "stale": None, "gold_latest": None}

    parsed = [
        permission
        for row in frame.to_dicts()
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
    return is_permission_status_enforceable(
        permission.permission_status
    ) and not _permission_expired(permission)


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
            "allowed_live_modes": [],
            "live_block_reasons": _dedupe(
                [*permission.live_block_reasons, "no_fresh_published_permission"]
            ),
            "risk_reason_codes": risk_reason_codes,
            "reason": reason,
        }
    )


def _clear_non_enforceable_live_modes(permission: RiskPermission) -> RiskPermission:
    return permission.model_copy(
        update={
            "allowed_live_modes": [],
            "live_block_reasons": _dedupe(
                [*permission.live_block_reasons, "risk_permission_not_enforceable"]
            ),
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
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "strategy_health_daily")
    if lazy is None or "stale_permission_consecutive_count" not in columns:
        return 0
    scoped = lazy.filter(pl.col("strategy") == strategy) if "strategy" in columns else lazy
    for column in ["latest_bundle_ts", "created_at", "date"]:
        if column in columns:
            scoped = scoped.with_columns(_lazy_utc_datetime(column).alias(column)).sort(
                column,
                descending=True,
                nulls_last=True,
            )
            break
    latest_frame = _collect_lazy_or_empty(scoped.limit(1))
    if latest_frame.is_empty():
        return 0
    latest = latest_frame.to_dicts()[0]
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
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "cost_bucket_daily")
    if lazy is None:
        return {
            "status": "missing",
            "missing": True,
            "cost_model_version": "missing",
        }
    aggregations: list[pl.Expr] = [pl.len().alias("rows")]
    if "day" in columns:
        aggregations.append(pl.col("day").cast(pl.String, strict=False).max().alias("latest_day"))
    else:
        aggregations.append(pl.lit(None, dtype=pl.String).alias("latest_day"))
    global_default_exprs: list[pl.Expr] = []
    if "source" in columns:
        global_default_exprs.append(
            pl.col("source").cast(pl.String, strict=False) == "global_default"
        )
    if "fallback_level" in columns:
        global_default_exprs.append(
            pl.col("fallback_level").cast(pl.String, strict=False) == "GLOBAL_DEFAULT"
        )
    if global_default_exprs:
        global_default_condition = global_default_exprs[0]
        for expression in global_default_exprs[1:]:
            global_default_condition = global_default_condition | expression
        aggregations.append(
            global_default_condition.fill_null(False)
            .cast(pl.Int64)
            .sum()
            .alias("global_default_rows")
        )
    else:
        aggregations.append(pl.lit(0, dtype=pl.Int64).alias("global_default_rows"))
    stats = _collect_lazy_or_empty(lazy.select(aggregations))
    if stats.is_empty() or int(stats.item(0, "rows") or 0) == 0:
        return {
            "status": "missing",
            "missing": True,
            "cost_model_version": "missing",
        }
    rows = int(stats.item(0, "rows") or 0)
    global_default_rows = int(stats.item(0, "global_default_rows") or 0)
    latest_day = stats.item(0, "latest_day")
    latest_day = str(latest_day) if latest_day else None
    status = "ok"
    health: dict[str, Any] = {
        "status": status,
        "cost_model_version": f"cost_bucket_daily:{latest_day or 'unknown'}",
    }
    if global_default_rows == rows:
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
    latest_from_health = _latest_market_bar_health_ts(lake_root)
    if latest_from_health is not None:
        return _market_bar_data_health_from_latest(latest_from_health)

    lazy, columns = _safe_parquet_lazy(lake_root / "silver" / "market_bar")
    if lazy is None or "ts" not in columns:
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_missing"],
        }
    latest_frame = _collect_lazy_or_empty(
        lazy.select(_lazy_utc_datetime("ts").max().alias("latest_ts"))
    )
    latest_ts = latest_frame.item(0, "latest_ts") if not latest_frame.is_empty() else None
    if not isinstance(latest_ts, datetime):
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_invalid_timestamp"],
        }
    return _market_bar_data_health_from_latest(latest_ts)


def _latest_market_bar_health_ts(lake_root: Path) -> datetime | None:
    lazy, columns = _safe_parquet_lazy(lake_root / "silver" / "market_bar_health")
    if lazy is None or "latest_ts" not in columns:
        return None
    latest_frame = _collect_lazy_or_empty(
        lazy.select(_lazy_utc_datetime("latest_ts").max().alias("latest_ts"))
    )
    latest_ts = latest_frame.item(0, "latest_ts") if not latest_frame.is_empty() else None
    if not isinstance(latest_ts, datetime):
        return None
    return latest_ts.astimezone(UTC)


def _market_bar_data_health_from_latest(latest_ts: datetime) -> dict[str, Any]:
    latest_utc = latest_ts.astimezone(UTC) if latest_ts.tzinfo else latest_ts.replace(tzinfo=UTC)
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
    compliance = _latest_lazy_row(
        lake_root / "gold" / "v5_gate_compliance_daily",
        filters={"strategy": strategy} if strategy == "v5" else None,
        sort_columns=["date", "created_at"],
    )
    if strategy == "v5" and compliance is not None:
        violation_count = int(compliance.get("violation_count") or 0)
        if violation_count > 0:
            reasons.append("v5_gate_compliance_violation")

    health = _latest_lazy_row(
        lake_root / "gold" / "strategy_health_daily",
        filters={"strategy": strategy} if strategy == "v5" else None,
        sort_columns=["latest_bundle_ts", "created_at", "date"],
    )
    if strategy == "v5" and health is not None:
        if str(health.get("status") or "").upper() == "CRITICAL":
            reasons.append("v5_strategy_health_critical")
    return reasons


def _has_bootstrap_gate(gate_decisions: list[GateDecision]) -> bool:
    return any(
        decision.gate_version == BOOTSTRAP_GATE_VERSION and decision.status.value == "QUARANTINE"
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


def _safe_parquet_lazy(path: Path) -> tuple[pl.LazyFrame | None, set[str]]:
    try:
        lazy = read_parquet_lazy(path)
        columns = set(lazy.collect_schema().names())
    except Exception:
        return None, set()
    return lazy, columns


def _collect_lazy_or_empty(lazy: pl.LazyFrame) -> pl.DataFrame:
    try:
        return lazy.collect()
    except Exception:
        return pl.DataFrame()


def _latest_lazy_row(
    path: Path,
    *,
    filters: dict[str, Any] | None = None,
    sort_columns: list[str] | None = None,
) -> dict[str, Any] | None:
    lazy, columns = _safe_parquet_lazy(path)
    if lazy is None:
        return None
    scoped = lazy
    for column, value in (filters or {}).items():
        if column in columns:
            scoped = scoped.filter(pl.col(column) == value)
    sort_column = next((column for column in (sort_columns or []) if column in columns), None)
    if sort_column is not None:
        scoped = scoped.with_columns(_lazy_utc_datetime(sort_column).alias(sort_column)).sort(
            sort_column,
            descending=True,
            nulls_last=True,
        )
    frame = _collect_lazy_or_empty(scoped.limit(1))
    return None if frame.is_empty() else frame.to_dicts()[0]


def _lazy_utc_datetime(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)


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
