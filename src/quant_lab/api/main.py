"""Read-only FastAPI surface for quant-lab strategy consumers.

This API is read-only and must not mutate strategy state or exchange state.
"""

import csv
import hashlib
import hmac
import io
import json
import os
import subprocess
import time
import zipfile
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.gzip import GZipMiddleware

from quant_lab import __version__
from quant_lab.api.cache import (
    CostBucketCache,
    ExactKeyCache,
    StrategyOpportunityAdvisoryCache,
    StrategyOpportunityAdvisoryResponseCache,
)
from quant_lab.contracts.models import (
    CostEstimate,
    GateDecision,
    MarketBar,
    RiskPermission,
    RiskPermissionStatus,
)
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.costs.health import read_cost_health_daily
from quant_lab.costs.model import (
    CostBucket,
    estimate_cost_bps,
    estimate_cost_from_cost_bucket_table_rows,
    evaluate_live_universe_cost_coverage,
)
from quant_lab.data.lake import (
    read_market_bars,
    read_parquet_dataset,  # noqa: F401 - kept as a monkeypatch guard for no-eager-read tests.
    read_parquet_lazy,
)
from quant_lab.data.market_bar_time import (
    DEFAULT_MARKET_BAR_TIMEFRAME,
    ensure_utc_datetime,
    market_bar_close_ts,
    market_bar_freshness_seconds,
)
from quant_lab.gates.defaults import conservative_example_gate_decision
from quant_lab.ops.api_metrics import api_metrics_summary, record_api_request
from quant_lab.ops.dataset_registry import dataset_names, get_dataset_spec
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
    rebuild_risk_permission_api_dependency_meta_from_latest_permission,
    risk_permission_stale_vs_telemetry,
)
from quant_lab.symbols import normalize_symbol

MARKET_BAR_WARNING_DELAY_SECONDS = 2 * 60 * 60
MARKET_BAR_CRITICAL_DELAY_SECONDS = 3 * 60 * 60

STRATEGY_OPPORTUNITY_ADVISORY_SCHEMA_VERSION = "strategy_opportunity_advisory.v0.1"
STRATEGY_OPPORTUNITY_ADVISORY_TTL_SECONDS = 3 * 60 * 60
STRATEGY_OPPORTUNITY_ADVISORY_MAX_API_ROWS = 50_000
BOTTOM_ZONE_PROBE_STRATEGY_ID = "BOTTOM_ZONE_PROBE_PAPER_V1"
BOTTOM_ZONE_PROBE_CANDIDATES = {
    "v5.bottom_zone_probe_paper",
    "bottom_zone_probe_paper",
}
UNOBSERVABLE_TEXT_VALUES = {"", "none", "null", "nan", "not_observable", "unknown"}
_STRATEGY_OPPORTUNITY_ADVISORY_CACHE: StrategyOpportunityAdvisoryCache[
    "StrategyOpportunityAdvisoryRow"
] = StrategyOpportunityAdvisoryCache()
_STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE = StrategyOpportunityAdvisoryResponseCache()
_RISK_PERMISSION_EVALUATION_CACHE: ExactKeyCache[dict[str, Any]] = ExactKeyCache()
_COST_ESTIMATE_CACHE: ExactKeyCache[CostEstimate] = ExactKeyCache()
_COST_BUCKET_CACHE = CostBucketCache()


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
    cache_hit_count: int = 0
    rows_returned_total: float = 0.0
    response_bytes_total: float = 0.0
    lake_scan_ms_total: float = 0.0
    serialize_ms_total: float = 0.0
    source_signature_ms_total: float = 0.0
    response_cache_hit_count: int = 0
    dependency_meta_missing_count: int = 0
    by_error_type: dict[str, int] = Field(default_factory=dict)
    by_auth_result: dict[str, int] = Field(default_factory=dict)
    auth_error_count: int = 0


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
    expanded_universe_maturity_state: str | None = None
    cost_quality: str | None = None
    cost_quality_score: float | None = None
    paper_ready_block_reasons: list[str] = Field(default_factory=list)
    advisory_intent: str = "research_only"
    live_block_reasons: list[str] = Field(default_factory=list)
    max_paper_notional_usdt: float | None = None
    max_live_notional_usdt: float = 0.0
    live_order_effect: str = "read_only_no_live_order"


KNOWN_DATASETS = dataset_names()


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
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    _mount_bigscreen_static(app)

    @app.middleware("http")
    async def require_bearer_token_and_record_metrics(request: Request, call_next: Any) -> Any:
        started = time.perf_counter()
        status_code = 500
        response: Response | None = None
        error_type: str | None = None
        try:
            _authorize_v1_request(request)
        except HTTPException as exc:
            status_code = exc.status_code
            response = JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers,
            )
            _record_api_request_metric(
                request,
                status_code,
                time.perf_counter() - started,
                response=response,
                error_type="HTTPException",
            )
            return response
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception as exc:
            error_type = type(exc).__name__
            raise
        finally:
            _record_api_request_metric(
                request,
                status_code,
                time.perf_counter() - started,
                response=response,
                error_type=error_type,
            )

    @app.on_event("startup")
    def warm_strategy_opportunity_advisory_cache() -> None:
        try:
            _ensure_risk_permission_dependency_meta(_lake_root())
        except Exception:
            pass
        try:
            _strategy_opportunity_advisory_snapshot(_lake_root())
        except Exception:
            pass
        try:
            _cost_bucket_snapshot(_lake_root())
        except Exception:
            pass
        try:
            from quant_lab.web.bigscreen import bigscreen_snapshot

            bigscreen_snapshot(_lake_root())
        except Exception:
            pass

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/health/deep")
    def health_deep() -> dict[str, Any]:
        lake_root = _lake_root()
        data_health = _lake_data_health(lake_root)
        cost_health = _lake_cost_health(lake_root)
        dependency_meta_health = _risk_permission_dependency_meta_health(lake_root)
        overall_status, warnings = _deep_health_status(
            data_health=data_health,
            cost_health=cost_health,
            risk_permission_dependency_meta=dependency_meta_health,
        )
        live_entry_readiness = _live_entry_readiness_health(lake_root)
        return {
            "status": "ok",
            "overall_status": overall_status,
            "service": "quant-lab",
            "mode": "read-only",
            "service_health": {
                "status": "OK",
                "mode": "read-only",
                "transport": "OK",
            },
            "data_quality": _deep_data_quality_status(overall_status, warnings),
            "live_entry_readiness": live_entry_readiness,
            "data_health": data_health,
            "cost_health": cost_health,
            "risk_permission_dependency_meta": dependency_meta_health,
            "warnings": warnings,
        }

    @app.get("/v1/web/bigscreen-snapshot")
    def web_bigscreen_snapshot() -> JSONResponse:
        from quant_lab.web.bigscreen import bigscreen_snapshot_with_meta

        payload, cache_meta = bigscreen_snapshot_with_meta(_lake_root())
        return _bigscreen_snapshot_response(payload, cache_meta)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/web-v2/snapshot")
    def web_bigscreen_snapshot_for_static_page() -> JSONResponse:
        from quant_lab.web.bigscreen import bigscreen_snapshot_with_meta

        payload, cache_meta = bigscreen_snapshot_with_meta(_lake_root())
        return _bigscreen_snapshot_response(payload, cache_meta)

    @app.post("/web-v2/expert-pack/generate")
    def web_v2_generate_expert_pack(response: Response) -> dict[str, Any]:
        status_payload = _start_web_v2_expert_pack_job()
        _mark_no_store(response)
        response.headers["X-Quant-Lab-Mode"] = "read-only-export"
        return status_payload

    @app.get("/web-v2/expert-pack/status")
    def web_v2_expert_pack_status(
        response: Response,
        export_date: str | None = None,
    ) -> dict[str, Any]:
        status_payload = _web_v2_expert_pack_status(export_date=export_date)
        _mark_no_store(response)
        response.headers["X-Quant-Lab-Mode"] = "read-only-export"
        return status_payload

    @app.get("/web-v2/expert-pack/download/{file_name}")
    def web_v2_download_expert_pack(file_name: str) -> FileResponse:
        exports_root = _web_v2_exports_root()
        pack_path = _safe_web_v2_export_pack_path(exports_root, file_name)
        return FileResponse(
            pack_path,
            media_type="application/zip",
            filename=pack_path.name,
        )

    @app.get("/web-v2")
    @app.get("/web-v2/")
    def web_bigscreen_page() -> Response:
        return _bigscreen_index_response()

    @app.get("/web-v2/legacy")
    def web_legacy_streamlit_page(request: Request) -> RedirectResponse:
        return RedirectResponse(url=_legacy_streamlit_url(request), status_code=307)

    @app.get("/web-v2/{_path:path}")
    def web_bigscreen_spa_fallback(_path: str) -> Response:
        return _bigscreen_index_response()

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
        response: Response,
        symbol: str,
        regime: str,
        notional_usdt: float,
        quantile: str = "p75",
        notional_bucket: str | None = None,
    ) -> CostEstimate:
        (
            estimate,
            request_cache_hit,
            bucket_cache_hit,
            compute_ms,
            lake_scan_ms,
        ) = _cost_estimate_cached(
            _lake_root(),
            symbol=symbol,
            regime=regime,
            notional_usdt=notional_usdt,
            quantile=quantile,
            notional_bucket=notional_bucket,
        )
        response.headers["X-Cost-Cache-Hit"] = "true" if request_cache_hit else "false"
        response.headers["X-Cost-Bucket-Cache-Hit"] = (
            "true" if bucket_cache_hit else "false"
        )
        response.headers["X-Cost-Compute-Ms"] = f"{compute_ms:.3f}"
        response.headers["X-Quant-Lab-Api-Cache-Hit"] = (
            "true" if request_cache_hit or bucket_cache_hit else "false"
        )
        response.headers["X-Quant-Lab-Lake-Scan-Ms"] = f"{lake_scan_ms:.3f}"
        response.headers["X-Quant-Lab-Serialize-Ms"] = "0.000"
        fallback = _signature_fallback_header(
            _cost_signature_paths(_lake_root()),
        )
        if fallback:
            response.headers["X-Quant-Lab-Signature-Fallback"] = fallback
        return estimate

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
        "/v1/reports/strategy-opportunity-advisory",
        response_model=list[StrategyOpportunityAdvisoryRow],
    )
    def strategy_opportunity_advisory(
        request: Request,
        symbols: str | None = None,
        families: str | None = None,
        latest_only: bool = False,
        fresh_only: bool = False,
        fields: str | None = None,
    ) -> Response:
        return _strategy_opportunity_advisory_response(
            _lake_root(),
            request=request,
            symbols=_split_csv(symbols),
            families=_split_csv(families),
            latest_only=latest_only,
            fresh_only=fresh_only,
            fields=fields,
        )

    @app.get("/v1/strategy_opportunity_advisory")
    def strategy_opportunity_advisory_legacy(request: Request) -> RedirectResponse:
        target = "/v1/strategy-opportunity-advisory/v5-compact"
        if request.url.query:
            target = f"{target}?{request.url.query}"
        return RedirectResponse(url=target, status_code=308)

    @app.get("/v1/strategy-opportunity-advisory/v5-compact")
    def strategy_opportunity_advisory_v5_compact(
        request: Request,
        symbols: str | None = None,
        families: str | None = None,
        latest_only: bool = True,
        fresh_only: bool = False,
    ) -> Response:
        return _strategy_opportunity_advisory_response(
            _lake_root(),
            request=request,
            symbols=_split_csv(symbols),
            families=_split_csv(families),
            latest_only=latest_only,
            fresh_only=fresh_only,
            fields="minimal",
            compact_for_v5=True,
        )

    @app.get("/v1/risk/live-permission", response_model=RiskPermission)
    def live_permission(response: Response, strategy: str, version: str) -> RiskPermission:
        evaluation, cache_hit, compute_ms = _live_permission_evaluation_cached(
            _lake_root(),
            strategy=strategy,
            version=version,
        )
        response.headers["X-Risk-Permission-Cache-Hit"] = "true" if cache_hit else "false"
        response.headers["X-Risk-Permission-Compute-Ms"] = f"{compute_ms:.3f}"
        response.headers["X-Quant-Lab-Api-Cache-Hit"] = "true" if cache_hit else "false"
        response.headers["X-Quant-Lab-Lake-Scan-Ms"] = f"{compute_ms:.3f}"
        response.headers["X-Quant-Lab-Serialize-Ms"] = "0.000"
        fallback = _signature_fallback_header(_risk_permission_signature_paths(_lake_root()))
        if fallback:
            response.headers["X-Quant-Lab-Signature-Fallback"] = fallback
        dependency_meta_status = _risk_permission_dependency_meta_status(_lake_root())
        if dependency_meta_status != "ok":
            response.headers["X-Risk-Permission-Dependency-Meta"] = dependency_meta_status
        return evaluation["permission"]

    @app.get("/v1/risk/live-permission-detail", response_model=LivePermissionDetailResponse)
    def live_permission_detail(
        response: Response,
        strategy: str,
        version: str,
    ) -> LivePermissionDetailResponse:
        evaluation, cache_hit, compute_ms = _live_permission_evaluation_cached(
            _lake_root(),
            strategy=strategy,
            version=version,
        )
        response.headers["X-Risk-Permission-Cache-Hit"] = "true" if cache_hit else "false"
        response.headers["X-Risk-Permission-Compute-Ms"] = f"{compute_ms:.3f}"
        response.headers["X-Quant-Lab-Api-Cache-Hit"] = "true" if cache_hit else "false"
        response.headers["X-Quant-Lab-Lake-Scan-Ms"] = f"{compute_ms:.3f}"
        response.headers["X-Quant-Lab-Serialize-Ms"] = "0.000"
        fallback = _signature_fallback_header(_risk_permission_signature_paths(_lake_root()))
        if fallback:
            response.headers["X-Quant-Lab-Signature-Fallback"] = fallback
        dependency_meta_status = _risk_permission_dependency_meta_status(_lake_root())
        if dependency_meta_status != "ok":
            response.headers["X-Risk-Permission-Dependency-Meta"] = dependency_meta_status
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
        missing_expiry = _permission_missing_required_expiry(stale_permission)
        if _permission_expired(stale_permission) or missing_expiry:
            source = (
                "no_fresh_incomplete_published_permission"
                if missing_expiry
                else "no_fresh_expired_published_permission"
            )
            permission = _force_no_fresh_permission(
                recomputed,
                telemetry_latest_ts=telemetry_latest_ts,
                extra_reasons=["published_permission_missing_expiry"] if missing_expiry else [],
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


def _live_permission_evaluation_cached(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
) -> tuple[dict[str, Any], bool, float]:
    telemetry_latest_ts = latest_strategy_telemetry_ts(lake_root, strategy)
    key = _risk_permission_cache_key(
        lake_root,
        strategy=strategy,
        version=version,
        telemetry_latest_ts=telemetry_latest_ts,
    )
    cached = _RISK_PERMISSION_EVALUATION_CACHE.get(key)
    if cached is not None and _cached_live_permission_valid(cached):
        return cached, True, 0.0
    started = time.perf_counter()
    evaluation = _live_permission_evaluation(
        lake_root,
        strategy=strategy,
        version=version,
    )
    compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
    _RISK_PERMISSION_EVALUATION_CACHE.set(key, evaluation)
    return evaluation, False, compute_ms


def _risk_permission_cache_key(
    lake_root: Path,
    *,
    strategy: str,
    version: str,
    telemetry_latest_ts: datetime | None,
) -> tuple[Any, ...]:
    return (
        str(lake_root.resolve()),
        str(strategy or "").strip(),
        str(version or "").strip(),
        "risk_permission_api_dependency_meta",
        _risk_permission_dependency_meta_signature(lake_root),
        "telemetry_latest_ts",
        telemetry_latest_ts.isoformat() if telemetry_latest_ts is not None else "",
    )


def _cached_live_permission_valid(evaluation: dict[str, Any]) -> bool:
    permission = evaluation.get("permission")
    if not isinstance(permission, RiskPermission):
        return False
    expires_at = permission.expires_at
    if expires_at is None:
        return False
    return expires_at >= datetime.now(UTC)


def _cost_estimate_cached(
    lake_root: Path,
    *,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str,
    notional_bucket: str | None,
) -> tuple[CostEstimate, bool, bool, float, float]:
    snapshot, bucket_cache_hit, _source_signature_ms = _cost_bucket_snapshot(lake_root)
    lake_scan_ms = 0.0 if bucket_cache_hit else snapshot.lake_scan_ms
    key = _cost_estimate_cache_key(
        source_sha=snapshot.source_sha,
        symbol=symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        notional_bucket=notional_bucket,
    )
    cached = _COST_ESTIMATE_CACHE.get(key)
    if cached is not None:
        return cached, True, bucket_cache_hit, 0.0, lake_scan_ms
    started = time.perf_counter()
    estimate = estimate_cost_from_cost_bucket_table_rows(
        symbol=symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        rows=snapshot.rows,
        dataset_has_rows=snapshot.dataset_has_rows,
        notional_bucket=notional_bucket,
    )
    compute_ms = round((time.perf_counter() - started) * 1000.0, 3)
    _COST_ESTIMATE_CACHE.set(key, estimate)
    return estimate, False, bucket_cache_hit, compute_ms, lake_scan_ms


def _cost_estimate_cache_key(
    *,
    source_sha: str,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str,
    notional_bucket: str | None,
) -> tuple[Any, ...]:
    return (
        source_sha,
        normalize_symbol(symbol),
        str(regime or "").strip(),
        round(float(notional_usdt or 0.0), 2),
        str(quantile or "p75").strip(),
        str(notional_bucket or "").strip(),
    )


def _cost_bucket_snapshot(lake_root: Path):
    return _COST_BUCKET_CACHE.get_snapshot(
        lake_root,
        signature_builder=_cost_bucket_source_signature,
        loader=_cost_bucket_rows_for_api,
        monotonic_seconds=time.perf_counter,
    )


def _cost_bucket_source_signature(lake_root: Path) -> tuple[Any, ...]:
    return (
        "cost_bucket_daily",
        _dataset_snapshot_signature(lake_root / "gold" / "cost_bucket_daily"),
        "cost_health_daily",
        _dataset_snapshot_signature(lake_root / "gold" / "cost_health_daily"),
    )


def _cost_bucket_rows_for_api(lake_root: Path) -> tuple[list[dict[str, Any]], bool]:
    dataset_path = lake_root / "gold" / "cost_bucket_daily"
    try:
        frame = read_parquet_lazy(dataset_path).collect()
    except Exception:
        return [], False
    if frame.is_empty():
        return [], False
    return frame.to_dicts(), True


def _cost_signature_paths(lake_root: Path) -> list[Path]:
    return [
        lake_root / "gold" / "cost_bucket_daily",
        lake_root / "gold" / "cost_health_daily",
    ]


def _risk_permission_signature_paths(lake_root: Path) -> list[Path]:
    meta_path = lake_root / "gold" / "risk_permission_api_dependency_meta"
    if meta_path.exists():
        return [meta_path]
    return [
        lake_root / "gold" / "risk_permission",
        lake_root / "gold" / "gate_decision",
        lake_root / "gold" / "cost_health_daily",
    ]


def _risk_permission_dependency_meta_status(lake_root: Path) -> str:
    meta_path = lake_root / "gold" / "risk_permission_api_dependency_meta"
    if not meta_path.exists():
        return "missing"
    if meta_path.is_dir() and not (meta_path / "_snapshot_meta.json").exists():
        return "missing_snapshot_meta"
    return "ok"


def _ensure_risk_permission_dependency_meta(lake_root: Path) -> int:
    if _risk_permission_dependency_meta_status(lake_root) == "ok":
        return 0
    return rebuild_risk_permission_api_dependency_meta_from_latest_permission(lake_root)


def _risk_permission_dependency_meta_health(lake_root: Path) -> dict[str, Any]:
    status = _risk_permission_dependency_meta_status(lake_root)
    if status == "ok":
        return {"status": "ok", "warning": ""}
    return {
        "status": "warning",
        "warning": f"risk_permission_dependency_meta_{status}",
    }


def _risk_permission_dependency_meta_signature(lake_root: Path) -> tuple[Any, ...]:
    meta_path = lake_root / "gold" / "risk_permission_api_dependency_meta"
    if meta_path.exists():
        return _dataset_snapshot_signature(meta_path)
    return (
        "fallback_dependency_meta_missing",
        _dataset_snapshot_signature(lake_root / "gold" / "risk_permission"),
        _dataset_snapshot_signature(lake_root / "gold" / "gate_decision"),
        _dataset_snapshot_signature(lake_root / "gold" / "cost_health_daily"),
    )


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
        permission = annotate_risk_permission(
            permission,
            telemetry_latest_ts=telemetry_latest_ts,
            as_of_ts=datetime.now(UTC),
            threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
        )
        permission = apply_risk_advisory_context(permission, advisory_context)
        return {
            "permission": permission,
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
    permission = annotate_risk_permission(
        permission,
        telemetry_latest_ts=telemetry_latest_ts,
        as_of_ts=datetime.now(UTC),
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    )
    permission = apply_risk_advisory_context(permission, advisory_context)
    return {
        "permission": permission,
        "advisory_context": advisory_context,
        "data_health": data_health,
        "cost_health": cost_health,
        "gate_summary": _gate_summary(gate_decisions),
        "v5_telemetry_summary": _v5_telemetry_summary(telemetry_reasons),
        "original_data_health": original_data_health,
    }


def _lake_root() -> Path:
    return Path(os.environ.get("QUANT_LAB_LAKE_ROOT", "/var/lib/quant-lab/lake"))


def _utf8_json_response(content: Any, *, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        content=content,
        headers=headers,
        media_type="application/json; charset=utf-8",
    )


def _bigscreen_snapshot_response(
    payload: dict[str, Any],
    cache_meta: dict[str, Any],
) -> JSONResponse:
    cache_hit = bool(cache_meta.get("cache_hit"))
    response = _utf8_json_response(
        payload,
        headers={
            "Cache-Control": "no-store, max-age=0",
            "Pragma": "no-cache",
            "X-Quant-Lab-Bigscreen-Mode": "read-only",
            "X-Quant-Lab-Bigscreen-Cache-Hit": "true" if cache_hit else "false",
            "X-Quant-Lab-Bigscreen-Cache-Age-Seconds": _header_float_text(
                cache_meta.get("cache_age_seconds")
            ),
            "X-Quant-Lab-Bigscreen-Cache-TTL-Seconds": _header_float_text(
                cache_meta.get("cache_ttl_seconds")
            ),
            "X-Quant-Lab-Bigscreen-Cache-Stale": (
                "true" if cache_meta.get("cache_stale") else "false"
            ),
            "X-Quant-Lab-Bigscreen-Stale-Grace-Seconds": _header_float_text(
                cache_meta.get("cache_stale_grace_seconds")
            ),
            "X-Quant-Lab-Api-Cache-Hit": "true" if cache_hit else "false",
            "X-Quant-Lab-Response-Cache-Hit": "true" if cache_hit else "false",
        },
    )
    response.headers["X-Quant-Lab-Response-Bytes"] = str(len(response.body or b""))
    return response


def _mark_no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["Pragma"] = "no-cache"


def _header_float_text(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _bigscreen_static_root() -> Path:
    return Path(__file__).resolve().parents[1] / "web" / "bigscreen_static"


def _mount_bigscreen_static(app: FastAPI) -> None:
    static_root = _bigscreen_static_root()
    assets_root = static_root / "assets"
    if assets_root.is_dir():
        app.mount(
            "/web-v2/assets",
            StaticFiles(directory=str(assets_root)),
            name="quant-lab-bigscreen-assets",
        )


def _bigscreen_index_response() -> Response:
    index_path = _bigscreen_static_root() / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return JSONResponse(
        status_code=503,
        content={
            "detail": "quant-lab bigscreen frontend is not built",
            "expected_path": str(index_path),
        },
    )


def _legacy_streamlit_url(request: Request) -> str:
    port = os.environ.get("QUANT_LAB_LEGACY_WEB_PORT", "8501").strip() or "8501"
    hostname = request.url.hostname or "qyun2.hrhome.top"
    scheme = request.url.scheme or "http"
    return f"{scheme}://{hostname}:{port}/"


def _web_v2_exports_root() -> Path:
    from quant_lab.web import readers

    return readers.default_exports_root(_lake_root())


def _web_v2_export_date(export_date: str | None = None) -> str:
    if export_date:
        return export_date
    from quant_lab.time_display import beijing_today

    return beijing_today().isoformat()


def _start_web_v2_expert_pack_job() -> dict[str, Any]:
    from quant_lab.web.pages.expert_exports import (
        _start_export_job,
        _web_background_export_enabled,
        _web_on_demand_export_enabled,
    )

    export_date = _web_v2_export_date()
    lake_root = _lake_root()
    exports_root = _web_v2_exports_root()
    if not _web_on_demand_export_enabled():
        return _decorate_web_v2_expert_pack_status(
            {
                "state": "on_demand_disabled",
                "export_date": export_date,
                "message": "QUANT_LAB_WEB_ON_DEMAND_EXPORT is disabled",
            },
            export_date=export_date,
            exports_root=exports_root,
        )
    if not _web_background_export_enabled():
        return _decorate_web_v2_expert_pack_status(
            {
                "state": "background_disabled",
                "export_date": export_date,
                "message": "QUANT_LAB_WEB_EXPORT_BACKGROUND is disabled",
            },
            export_date=export_date,
            exports_root=exports_root,
        )
    status_payload = _start_export_job(
        export_date=export_date,
        lake_root=lake_root,
        exports_root=exports_root,
    )
    return _decorate_web_v2_expert_pack_status(
        status_payload,
        export_date=export_date,
        exports_root=exports_root,
    )


def _web_v2_expert_pack_status(*, export_date: str | None = None) -> dict[str, Any]:
    from quant_lab.web.pages.expert_exports import _poll_export_job

    day = _web_v2_export_date(export_date)
    exports_root = _web_v2_exports_root()
    status_payload = _poll_export_job(exports_root, day)
    return _decorate_web_v2_expert_pack_status(
        status_payload,
        export_date=day,
        exports_root=exports_root,
    )


def _decorate_web_v2_expert_pack_status(
    status_payload: dict[str, Any],
    *,
    export_date: str,
    exports_root: Path,
) -> dict[str, Any]:
    from quant_lab.web import readers
    from quant_lab.web.pages.expert_exports import _latest_pack_for_export_date

    status = dict(status_payload)
    requested_date_pack = _latest_pack_for_export_date(exports_root, export_date)
    zip_path = _safe_existing_pack_from_status(exports_root, status.get("zip_path"))
    packs = _expert_pack_rows_with_downloads(readers.expert_export_summary(exports_root))
    latest_available_pack = _pack_path_from_row(packs[0]) if packs else None
    state = status.get("state")
    if not state:
        state = (
            "succeeded"
            if requested_date_pack is not None
            else ("missing_requested_date" if latest_available_pack is not None else "missing")
        )
    is_running = str(state).lower() in {"starting", "running"}
    latest_pack = None if is_running else requested_date_pack or latest_available_pack or zip_path
    payload = {
        "mode": "read_only_export",
        "live_order_effect": "none",
        "export_date": export_date,
        "exports_root": str(exports_root),
        "state": state,
        "status": status,
        "requested_date_pack": (
            str(requested_date_pack) if requested_date_pack is not None else None
        ),
        "requested_date_pack_name": (
            requested_date_pack.name if requested_date_pack is not None else None
        ),
        "previous_pack": str(requested_date_pack) if is_running and requested_date_pack else None,
        "previous_pack_name": (
            requested_date_pack.name if is_running and requested_date_pack else None
        ),
        "latest_pack": str(latest_pack) if latest_pack is not None else None,
        "latest_pack_name": latest_pack.name if latest_pack is not None else None,
        "latest_pack_is_requested_date": (
            latest_pack is not None
            and requested_date_pack is not None
            and latest_pack.resolve() == requested_date_pack.resolve()
        ),
        "latest_download_url": (
            _web_v2_export_download_url(latest_pack.name) if latest_pack is not None else None
        ),
        "packs": packs,
        "pack_count": len(packs),
    }
    if latest_pack is not None:
        try:
            stat = latest_pack.stat()
        except OSError:
            stat = None
        if stat is not None:
            payload["latest_size_bytes"] = stat.st_size
            payload["latest_modified_at"] = datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
    return payload


def _pack_path_from_row(row: dict[str, Any]) -> Path | None:
    name = str(row.get("name") or Path(str(row.get("path") or "")).name)
    if not _is_valid_export_pack_name(name):
        return None
    path_text = row.get("path")
    if path_text:
        path = Path(str(path_text))
        if path.is_file():
            return path
    return None


def _safe_existing_pack_from_status(exports_root: Path, value: Any) -> Path | None:
    if not value:
        return None
    try:
        path = Path(str(value)).resolve()
        root = exports_root.resolve()
    except OSError:
        return None
    try:
        path.relative_to(root)
    except ValueError:
        return None
    if path.is_file() and path.name.startswith("quant_lab_expert_pack_") and path.suffix == ".zip":
        return path
    return None


def _expert_pack_rows_with_downloads(
    summary: dict[str, Any],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    packs = summary.get("packs")
    rows: list[dict[str, Any]] = []
    if isinstance(packs, pl.DataFrame) and not packs.is_empty():
        source_rows = packs.head(limit).to_dicts()
    elif isinstance(packs, list):
        source_rows = [row for row in packs[:limit] if isinstance(row, dict)]
    else:
        source_rows = []
    for row in source_rows:
        item = {str(key): _api_json_value(value) for key, value in row.items()}
        name = str(item.get("name") or Path(str(item.get("path") or "")).name)
        if _is_valid_export_pack_name(name):
            item["name"] = name
            item["download_url"] = _web_v2_export_download_url(name)
        rows.append(item)
    return rows


def _safe_web_v2_export_pack_path(exports_root: Path, file_name: str) -> Path:
    if not _is_valid_export_pack_name(file_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="expert pack not found",
        )
    root = exports_root.resolve()
    path = (root / file_name).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="expert pack not found",
        ) from exc
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="expert pack not found",
        )
    return path


def _is_valid_export_pack_name(file_name: str) -> bool:
    if not file_name or "/" in file_name or "\\" in file_name:
        return False
    if file_name != Path(file_name).name:
        return False
    return file_name.startswith("quant_lab_expert_pack_") and file_name.endswith(".zip")


def _web_v2_export_download_url(file_name: str) -> str:
    return f"/web-v2/expert-pack/download/{file_name}"


def _api_json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and value != value:
        return None
    if isinstance(value, dict):
        return {str(key): _api_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_api_json_value(child) for child in value]
    return value


app = create_app()


def _authorize_v1_request(request: Request) -> None:
    if not request.url.path.startswith("/v1/"):
        return
    if not _client_ip_allowed(request):
        _set_auth_result(request, "client_ip_denied")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="quant-lab API client IP is not allowed",
        )
    token = os.environ.get("QUANT_LAB_API_TOKEN")
    if not token:
        _set_auth_result(request, "auth_not_configured")
        return
    if _local_health_check_without_token_allowed(request):
        _set_auth_result(request, "local_health_bypass")
        return
    provided = _bearer_token(request.headers.get("authorization", ""))
    if provided is None:
        _set_auth_result(request, "missing_bearer_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="quant-lab API bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(provided, token):
        _set_auth_result(request, "invalid_bearer_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="quant-lab API bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    _set_auth_result(request, "token_ok")


def _set_auth_result(request: Request, value: str) -> None:
    try:
        request.state.quant_lab_auth_result = value
    except Exception:
        return


def _bearer_token(authorization: str) -> str | None:
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :]


def _local_health_check_without_token_allowed(request: Request) -> bool:
    if request.url.path not in {"/v1/health", "/v1/health/deep"}:
        return False
    if not _bool_env("QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH", default=True):
        return False
    return _request_is_localhost(request)


def _request_is_localhost(request: Request) -> bool:
    host = request.client.host if request.client else ""
    return host in {"127.0.0.1", "::1", "localhost"} or host.startswith("::ffff:127.")


def _client_ip_allowed(request: Request) -> bool:
    allowed = os.environ.get("QUANT_LAB_ALLOWED_CLIENT_IPS")
    if not allowed:
        return True
    allowed_hosts = {item.strip() for item in allowed.split(",") if item.strip()}
    host = request.client.host if request.client else ""
    return host in allowed_hosts


def _record_api_request_metric(
    request: Request,
    status_code: int,
    duration_seconds: float,
    *,
    response: Response | None = None,
    error_type: str | None = None,
) -> None:
    if not request.url.path.startswith("/v1/"):
        return
    if not _bool_env("QUANT_LAB_API_METRICS_ENABLED", default=True):
        return
    headers = getattr(response, "headers", {}) or {}
    effective_error_type = error_type
    dependency_meta = _header_lookup(headers, "x-risk-permission-dependency-meta")
    dependency_meta_missing = dependency_meta == "missing"
    if effective_error_type is None and dependency_meta:
        effective_error_type = f"dependency_meta_{dependency_meta}"
    try:
        record_api_request(
            lake_root=_lake_root(),
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_seconds=duration_seconds,
            client_host=request.client.host if request.client else None,
            client_id=_request_client_id(request),
            user_agent=request.headers.get("user-agent"),
            auth_result=str(
                getattr(request.state, "quant_lab_auth_result", "") or "not_observable"
            ),
            cache_hit=_truthy(_header_lookup(headers, "x-quant-lab-api-cache-hit"))
            or _truthy(_header_lookup(headers, "x-advisory-cache-hit")),
            rows_returned=_int_or_none(_header_lookup(headers, "x-quant-lab-advisory-row-count"))
            or _int_or_none(_header_lookup(headers, "x-advisory-row-count")),
            response_bytes=_int_or_none(_header_lookup(headers, "x-quant-lab-response-bytes"))
            or _int_or_none(_header_lookup(headers, "x-advisory-response-bytes"))
            or _int_or_none(_header_lookup(headers, "content-length")),
            lake_scan_ms=_float_or_none(_header_lookup(headers, "x-quant-lab-lake-scan-ms")),
            serialize_ms=_float_or_none(_header_lookup(headers, "x-quant-lab-serialize-ms")),
            source_signature_ms=_float_or_none(
                _header_lookup(headers, "x-quant-lab-source-signature-ms")
            )
            or _float_or_none(_header_lookup(headers, "x-advisory-source-signature-ms")),
            response_cache_hit=_truthy(
                _header_lookup(headers, "x-quant-lab-response-cache-hit")
            )
            or _truthy(_header_lookup(headers, "x-advisory-response-cache-hit")),
            dependency_meta_missing=dependency_meta_missing,
            error_type=effective_error_type,
        )
    except Exception:
        return


def _request_client_id(request: Request) -> str | None:
    for header in ("x-quant-lab-client-id", "x-client-id"):
        value = request.headers.get(header)
        if value:
            return value
    return None


def _header_lookup(headers: Any, name: str) -> str:
    target = str(name or "").lower()
    try:
        for key, value in dict(headers or {}).items():
            if str(key).lower() == target:
                return str(value)
    except Exception:
        return ""
    return ""


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


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
    snapshot, _cache_hit, _signature_ms = _strategy_opportunity_advisory_snapshot(lake_root)
    return list(snapshot.rows)


def _strategy_opportunity_advisory_snapshot(lake_root: Path) -> tuple[Any, bool, float]:
    return _STRATEGY_OPPORTUNITY_ADVISORY_CACHE.get_snapshot(
        lake_root,
        signature_builder=_strategy_opportunity_advisory_source_signature,
        loader=_strategy_opportunity_advisory_rows_uncached,
        serializer=_strategy_opportunity_advisory_json_bytes,
        monotonic_seconds=time.perf_counter,
    )


def _strategy_opportunity_advisory_response(
    lake_root: Path,
    *,
    request: Request | None = None,
    symbols: list[str] | None = None,
    families: list[str] | None = None,
    latest_only: bool = False,
    fresh_only: bool = False,
    fields: str | None = None,
    compact_for_v5: bool = False,
) -> Response:
    request_now = datetime.now(UTC)
    snapshot, cache_hit, source_signature_ms = _strategy_opportunity_advisory_snapshot(lake_root)
    _STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE.clear_except_source_sha(
        snapshot.source_sha
    )
    response_key = _strategy_opportunity_advisory_response_cache_key(
        source_sha=snapshot.source_sha,
        symbols=symbols,
        families=families,
        latest_only=latest_only,
        fresh_only=fresh_only,
        fields=fields,
        compact_for_v5=compact_for_v5,
    )
    cached_response = _STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE.get(
        response_key,
        now=request_now,
    )
    if cached_response is not None:
        headers = _strategy_opportunity_advisory_response_headers(
            lake_root,
            row_count=cached_response.row_count,
            latest_generated_text=cached_response.latest_generated_at,
            cache_hit=cache_hit,
            response_cache_hit=True,
            source_sha=snapshot.source_sha,
            source_signature_ms=source_signature_ms,
            lake_scan_ms=0.0,
            serialize_ms=0.0,
            etag=cached_response.etag,
        )
        fallback = _signature_fallback_header(
            _strategy_opportunity_signature_paths(lake_root)
        )
        if fallback:
            headers["X-Quant-Lab-Signature-Fallback"] = fallback
        headers["X-Advisory-Response-Bytes"] = str(len(cached_response.payload))
        headers["X-Quant-Lab-Response-Bytes"] = str(len(cached_response.payload))
        if request is not None and request.headers.get("if-none-match") == cached_response.etag:
            return Response(status_code=304, headers=headers)
        return Response(
            content=cached_response.payload,
            media_type="application/json",
            headers=headers,
        )
    rows = _filter_strategy_opportunity_advisory_rows(
        list(snapshot.rows),
        symbols=symbols,
        families=families,
        latest_only=latest_only,
        fresh_only=fresh_only,
        now=request_now,
    )
    if compact_for_v5:
        rows = _compact_strategy_opportunity_advisory_for_v5(rows)
    minimal = str(fields or "").strip().lower() in {"minimal", "compact", "v5"}
    serialize_started = time.perf_counter()
    payload = (
        _strategy_opportunity_advisory_minimal_json_bytes(rows)
        if minimal
        else _strategy_opportunity_advisory_json_bytes(rows)
    )
    serialize_ms = round((time.perf_counter() - serialize_started) * 1000.0, 3)
    etag = _advisory_etag(snapshot.source_sha, payload)
    latest_generated_text = _latest_advisory_generated_text(rows)
    _STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE.set(
        response_key,
        payload=payload,
        etag=etag,
        row_count=len(rows),
        latest_generated_at=latest_generated_text,
        serialize_ms=serialize_ms,
        valid_until=_fresh_only_response_valid_until(rows) if fresh_only else None,
    )
    headers = _strategy_opportunity_advisory_response_headers(
        lake_root,
        row_count=len(rows),
        latest_generated_text=latest_generated_text,
        cache_hit=cache_hit,
        response_cache_hit=False,
        source_sha=snapshot.source_sha,
        source_signature_ms=source_signature_ms,
        lake_scan_ms=0.0 if cache_hit else snapshot.lake_scan_ms,
        serialize_ms=serialize_ms,
        etag=etag,
    )
    fallback = _signature_fallback_header(_strategy_opportunity_signature_paths(lake_root))
    if fallback:
        headers["X-Quant-Lab-Signature-Fallback"] = fallback
    headers["X-Advisory-Response-Bytes"] = str(len(payload))
    headers["X-Quant-Lab-Response-Bytes"] = str(len(payload))
    if request is not None and request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return Response(content=payload, media_type="application/json", headers=headers)


def _strategy_opportunity_advisory_response_headers(
    lake_root: Path,
    *,
    row_count: int,
    latest_generated_text: str,
    cache_hit: bool,
    response_cache_hit: bool,
    source_sha: str,
    source_signature_ms: float,
    lake_scan_ms: float,
    serialize_ms: float,
    etag: str,
) -> dict[str, str]:
    try:
        resolved_lake = str(lake_root.resolve())
    except OSError:
        resolved_lake = str(lake_root)
    lake_root_hash = hashlib.sha256(resolved_lake.encode("utf-8")).hexdigest()[:16]
    api_cache_hit = cache_hit or response_cache_hit
    return {
        "X-Advisory-Dataset-Generated-At": latest_generated_text,
        "X-Advisory-Row-Count": str(row_count),
        "X-Advisory-Source-Sha": source_sha,
        "X-Lake-Root-Hash": lake_root_hash,
        "X-Advisory-Cache-Hit": "true" if cache_hit else "false",
        "X-Advisory-Response-Cache-Hit": "true" if response_cache_hit else "false",
        "X-Advisory-Source-Signature-Ms": f"{float(source_signature_ms):.3f}",
        "X-Advisory-Lake-Scan-Ms": f"{float(lake_scan_ms):.3f}",
        "X-Advisory-Serialize-Ms": f"{float(serialize_ms):.3f}",
        "X-Quant-Lab-Advisory-Dataset-Generated-At": latest_generated_text,
        "X-Quant-Lab-Advisory-Row-Count": str(row_count),
        "X-Quant-Lab-Advisory-Source-Sha": source_sha,
        "X-Quant-Lab-Lake-Root-Hash": lake_root_hash,
        "X-Quant-Lab-Api-Cache-Hit": "true" if api_cache_hit else "false",
        "X-Quant-Lab-Advisory-Response-Cache-Hit": (
            "true" if response_cache_hit else "false"
        ),
        "X-Quant-Lab-Source-Signature-Ms": f"{float(source_signature_ms):.3f}",
        "X-Quant-Lab-Lake-Scan-Ms": f"{float(lake_scan_ms):.3f}",
        "X-Quant-Lab-Serialize-Ms": f"{float(serialize_ms):.3f}",
        "X-Quant-Lab-Advisory-Response-Cache-Size": str(
            _STRATEGY_OPPORTUNITY_ADVISORY_RESPONSE_CACHE.size()
        ),
        "ETag": etag,
    }


def _strategy_opportunity_advisory_json_bytes(
    rows: list[StrategyOpportunityAdvisoryRow],
) -> bytes:
    return b"[" + b",".join(row.model_dump_json().encode("utf-8") for row in rows) + b"]"


def _strategy_opportunity_advisory_response_cache_key(
    *,
    source_sha: str,
    symbols: list[str] | None,
    families: list[str] | None,
    latest_only: bool,
    fresh_only: bool,
    fields: str | None,
    compact_for_v5: bool = False,
) -> tuple[Any, ...]:
    return (
        source_sha,
        tuple(sorted(_normalize_query_symbol(symbol) for symbol in (symbols or []))),
        tuple(sorted(str(family).strip().lower() for family in (families or []))),
        bool(latest_only),
        bool(fresh_only),
        str(fields or "").strip().lower(),
        bool(compact_for_v5),
    )


def _latest_advisory_generated_text(rows: list[StrategyOpportunityAdvisoryRow]) -> str:
    latest_generated = max((row.generated_at for row in rows), default=None)
    return (
        latest_generated.astimezone(UTC).isoformat().replace("+00:00", "Z")
        if latest_generated is not None
        else ""
    )


def _strategy_opportunity_advisory_minimal_json_bytes(
    rows: list[StrategyOpportunityAdvisoryRow],
) -> bytes:
    compact_rows = [_strategy_opportunity_advisory_minimal_row(row) for row in rows]
    return json.dumps(compact_rows, separators=(",", ":"), default=str).encode("utf-8")


def _strategy_opportunity_advisory_minimal_row(
    row: StrategyOpportunityAdvisoryRow,
) -> dict[str, Any]:
    return {
        "as_of_ts": row.as_of_ts,
        "generated_at": row.generated_at,
        "expires_at": row.expires_at,
        "contract_version": row.contract_version,
        "quant_lab_git_commit": row.quant_lab_git_commit,
        "source_version": row.source_version,
        "strategy_id": row.strategy_id,
        "strategy_candidate": row.strategy_candidate,
        "symbol": row.symbol,
        "decision": row.decision,
        "recommended_mode": row.recommended_mode,
        "horizon_hours": row.horizon_hours,
        "source_module": row.source_module,
        "template_family": row.template_family,
        "candidate_id": row.candidate_id,
        "promotion_state": row.promotion_state,
        "alpha_factory_score": row.alpha_factory_score,
        "universe_type": row.universe_type,
        "expanded_universe_maturity_state": row.expanded_universe_maturity_state,
        "cost_quality": row.cost_quality,
        "would_block_if_enabled": row.would_block_if_enabled,
        "would_enter": row.would_enter,
        "no_sample_reason": row.no_sample_reason,
        "live_block_reasons": row.live_block_reasons,
        "max_paper_notional_usdt": row.max_paper_notional_usdt,
        "max_live_notional_usdt": row.max_live_notional_usdt,
        "live_order_effect": row.live_order_effect,
        "sample_count": row.sample_count,
        "complete_sample_count": row.complete_sample_count,
        "avg_net_bps": row.avg_net_bps,
        "win_rate": row.win_rate,
        "cost_source_mix": row.cost_source_mix,
    }


def _filter_strategy_opportunity_advisory_rows(
    rows: list[StrategyOpportunityAdvisoryRow],
    *,
    symbols: list[str] | None,
    families: list[str] | None,
    latest_only: bool,
    fresh_only: bool,
    now: datetime | None = None,
) -> list[StrategyOpportunityAdvisoryRow]:
    selected = rows
    if symbols:
        wanted_symbols = {_normalize_query_symbol(symbol) for symbol in symbols}
        selected = [
            row for row in selected if _normalize_query_symbol(row.symbol) in wanted_symbols
        ]
    if families:
        wanted_families = {
            str(family).strip().lower()
            for family in families
            if str(family).strip()
        }
        selected = [
            row for row in selected if _advisory_row_matches_family(row, wanted_families)
        ]
    if fresh_only:
        reference = now or datetime.now(UTC)
        selected = [row for row in selected if row.expires_at >= reference]
    if latest_only:
        latest: dict[tuple[str, str, str, str], StrategyOpportunityAdvisoryRow] = {}
        for row in selected:
            key = (
                row.strategy_candidate,
                row.symbol,
                row.source_module or "",
                row.candidate_id or "",
            )
            current = latest.get(key)
            if current is None or row.generated_at > current.generated_at:
                latest[key] = row
        selected = list(latest.values())
    return sorted(
        selected,
        key=lambda row: (
            -_datetime_sort_value(row.generated_at),
            row.strategy_candidate,
            row.symbol,
            row.horizon_hours or -1,
        ),
    )


def _fresh_only_response_valid_until(
    rows: list[StrategyOpportunityAdvisoryRow],
) -> datetime | None:
    expires_values = [row.expires_at for row in rows if row.expires_at is not None]
    return min(expires_values) if expires_values else None


def _compact_strategy_opportunity_advisory_for_v5(
    rows: list[StrategyOpportunityAdvisoryRow],
) -> list[StrategyOpportunityAdvisoryRow]:
    risk_on_latest: dict[
        tuple[str, str, str, int | None], StrategyOpportunityAdvisoryRow
    ] = {}
    selected: list[StrategyOpportunityAdvisoryRow] = []
    for row in rows:
        if _advisory_row_matches_family(row, {"risk_on_multi_buy"}):
            key = (
                row.strategy_candidate,
                row.symbol,
                row.source_module or "",
                row.horizon_hours,
            )
            current = risk_on_latest.get(key)
            if current is None or _v5_compact_row_sort_key(row) > _v5_compact_row_sort_key(current):
                risk_on_latest[key] = row
            continue
        selected.append(row)
    selected.extend(risk_on_latest.values())
    return sorted(
        selected,
        key=lambda row: (
            -_datetime_sort_value(row.generated_at),
            row.strategy_candidate,
            row.symbol,
            row.horizon_hours or -1,
        ),
    )


def _v5_compact_row_sort_key(row: StrategyOpportunityAdvisoryRow) -> tuple[float, float, str]:
    return (
        _datetime_sort_value(row.generated_at),
        _datetime_sort_value(row.as_of_ts),
        str(row.candidate_id or ""),
    )


def _advisory_row_matches_family(row: StrategyOpportunityAdvisoryRow, families: set[str]) -> bool:
    haystack = " ".join(
        str(value or "").lower()
        for value in (
            row.source_module,
            row.template_family,
            row.strategy_candidate,
            row.strategy_id,
            row.candidate_id,
        )
    )
    return any(family in haystack for family in families)


def _normalize_query_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    try:
        return normalize_symbol(text).replace("/", "-").upper()
    except Exception:
        return text.replace("/", "-").upper()


def _advisory_etag(source_sha: str, payload: bytes) -> str:
    digest = hashlib.sha256(source_sha.encode("utf-8") + b":" + payload).hexdigest()
    return f'"{digest}"'


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
    alpha_factory_enriched = _apply_alpha_factory_result_metadata_to_advisory_rows(
        deduped,
        _strategy_opportunity_alpha_factory_result_metadata(lake_root),
    )
    promotion_overridden = _apply_alpha_factory_promotion_overrides_to_advisory_rows(
        alpha_factory_enriched,
        _strategy_opportunity_alpha_factory_promotions(lake_root),
    )
    portfolio_overridden = _apply_research_portfolio_overrides_to_advisory_rows(
        promotion_overridden,
        _strategy_opportunity_portfolio_overrides(lake_root),
    )
    return _dedupe_strategy_opportunity_advisory_rows(portfolio_overridden)


def _strategy_opportunity_advisory_source_signature(lake_root: Path) -> tuple[Any, ...]:
    gold_path = lake_root / "gold" / "strategy_opportunity_advisory"
    gold_files = _dataset_snapshot_signature(gold_path)
    portfolio_files = _dataset_snapshot_signature(
        lake_root / "gold" / "research_portfolio_status"
    )
    promotion_files = _dataset_snapshot_signature(
        lake_root / "gold" / "alpha_factory_promotion_queue"
    )
    result_files = _dataset_snapshot_signature(lake_root / "gold" / "alpha_factory_result")
    if gold_files:
        return (
            "gold",
            gold_files,
            "research_portfolio_status",
            portfolio_files,
            "alpha_factory_promotion_queue",
            promotion_files,
            "alpha_factory_result",
            result_files,
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
                "alpha_factory_result",
                result_files,
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
            "alpha_factory_result",
            result_files,
        )
    return (
        "missing",
        "research_portfolio_status",
        portfolio_files,
        "alpha_factory_promotion_queue",
        promotion_files,
        "alpha_factory_result",
        result_files,
    )


def _strategy_opportunity_signature_paths(lake_root: Path) -> list[Path]:
    return [
        lake_root / "gold" / "strategy_opportunity_advisory",
        lake_root / "gold" / "research_portfolio_status",
        lake_root / "gold" / "alpha_factory_promotion_queue",
        lake_root / "gold" / "alpha_factory_result",
    ]


def _dataset_snapshot_signature(path: Path) -> tuple[Any, ...]:
    meta = path / "_snapshot_meta.json" if path.is_dir() or not path.suffix else None
    if meta is not None and meta.exists():
        try:
            meta_stat = meta.stat()
            root_stat = path.stat() if path.exists() else None
            payload = json.loads(meta.read_text(encoding="utf-8"))
            parquet_signature = _parquet_file_signature(path)
            return (
                "meta",
                meta_stat.st_mtime_ns,
                meta_stat.st_size,
                root_stat.st_mtime_ns if root_stat is not None else None,
                root_stat.st_size if root_stat is not None else None,
                parquet_signature,
                str(payload.get("source_sha") or ""),
                str(payload.get("generated_at") or ""),
                int(payload.get("row_count") or 0),
            )
        except Exception:
            return _parquet_file_signature(path)
    return _parquet_file_signature(path)


def _signature_fallback_header(paths: list[Path]) -> str:
    for path in paths:
        if path.exists() and path.is_dir() and not (path / "_snapshot_meta.json").exists():
            return "parquet_rglob"
    return ""


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
    selected: dict[tuple[str, str, int | None, str, str], StrategyOpportunityAdvisoryRow] = {}
    for row in rows:
        key = (
            row.strategy_candidate,
            row.symbol,
            row.horizon_hours,
            row.source_module or "",
            row.candidate_id or "",
        )
        current = selected.get(key)
        if current is None or _advisory_row_preferred(row, current):
            selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            row.strategy_candidate,
            row.symbol,
            row.horizon_hours if row.horizon_hours is not None else -1,
            row.source_module or "",
            row.candidate_id or "",
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


def _strategy_opportunity_alpha_factory_result_metadata(
    lake_root: Path,
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    lazy, columns = _safe_parquet_lazy(lake_root / "gold" / "alpha_factory_result")
    if lazy is None:
        return {}
    selected = lazy.select([pl.col(column) for column in columns])
    frame = _collect_lazy_or_empty(selected)
    return _alpha_factory_result_metadata_by_candidate_symbol(frame)


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


def _alpha_factory_result_metadata_by_candidate_symbol(
    results: pl.DataFrame,
) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    metadata: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    if results.is_empty():
        return metadata
    rows = results.to_dicts()
    latest_as_of_date = _latest_advisory_as_of_date(rows)
    if latest_as_of_date:
        rows = [
            row
            for row in rows
            if _text_value(row.get("as_of_date")) == latest_as_of_date
        ]
    for row in rows:
        candidate = _text_value(row.get("strategy_candidate"))
        symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
        if not candidate:
            continue
        enriched = {
            "source_module": "alpha_factory",
            "template_family": _alpha_factory_template_family(row),
            "candidate_id": _text_value(row.get("candidate_id")),
            "promotion_state": _text_value(row.get("promotion_state") or row.get("decision")),
            "alpha_factory_score": _optional_float(row.get("alpha_factory_score")),
            "cost_quality_score": _optional_float(row.get("cost_quality_score")),
            "paper_ready_block_reasons": row.get("paper_ready_block_reasons"),
        }
        key = (candidate, symbol, _optional_int(row.get("horizon_hours")))
        current = metadata.get(key)
        if current is None or _advisory_dict_time(row) >= _advisory_dict_time(current):
            metadata[key] = {**row, **enriched}
    return metadata


def _alpha_factory_template_family(row: dict[str, Any]) -> str:
    explicit = _text_value(row.get("template_family"))
    if explicit:
        return explicit
    template_name = _text_value(row.get("template_name"))
    if not template_name:
        return ""
    if "_v" in template_name:
        prefix, suffix = template_name.rsplit("_v", 1)
        if prefix and suffix.isdigit():
            return prefix
    return template_name


def _alpha_factory_metadata_for_advisory_row(
    row: StrategyOpportunityAdvisoryRow,
    metadata: dict[tuple[str, str, int | None], dict[str, Any]],
) -> dict[str, Any] | None:
    for key in [
        (row.strategy_candidate, row.symbol, row.horizon_hours),
        (row.strategy_candidate, row.symbol, None),
        (row.strategy_candidate, "UNKNOWN", row.horizon_hours),
        (row.strategy_candidate, "UNKNOWN", None),
    ]:
        value = metadata.get(key)
        if value is not None:
            return value
    return None


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
    if _is_bottom_zone_probe_paper_advisory(row):
        return False
    source_module = (row.source_module or "").lower()
    if source_module in {"regime_router", "expanded_universe"}:
        return False
    candidate = row.strategy_candidate
    return (
        candidate in ALPHA_FACTORY_CANDIDATES
        or candidate.startswith("v5.af.")
        or candidate.startswith("v5.expanded_relative_strength")
        or source_module == "alpha_factory"
        or bool(row.template_family)
        or row.alpha_factory_score is not None
    )


def _is_alpha_factory_promotion_capped_model(row: StrategyOpportunityAdvisoryRow) -> bool:
    if _is_bottom_zone_probe_paper_advisory(row):
        return False
    candidate = row.strategy_candidate
    candidate_key = (
        candidate.split(":", 1)[1]
        if candidate.startswith("regime_router:")
        else candidate
    )
    source_module = (row.source_module or "").lower()
    if source_module == "expanded_universe":
        return False
    return (
        candidate_key in ALPHA_FACTORY_CANDIDATES
        or candidate_key.startswith("v5.af.")
        or candidate_key.startswith("v5.expanded_relative_strength")
        or source_module == "alpha_factory"
        or bool(row.template_family)
        or row.alpha_factory_score is not None
    )


def _apply_alpha_factory_result_metadata_to_advisory_rows(
    rows: list[StrategyOpportunityAdvisoryRow],
    metadata: dict[tuple[str, str, int | None], dict[str, Any]],
) -> list[StrategyOpportunityAdvisoryRow]:
    if not metadata:
        return [row.model_copy(update={"max_live_notional_usdt": 0.0}) for row in rows]
    output: list[StrategyOpportunityAdvisoryRow] = []
    for row in rows:
        if not _is_alpha_factory_advisory_model(row):
            output.append(row.model_copy(update={"max_live_notional_usdt": 0.0}))
            continue
        meta = _alpha_factory_metadata_for_advisory_row(row, metadata)
        if meta is None:
            output.append(row.model_copy(update={"max_live_notional_usdt": 0.0}))
            continue
        paper_ready_block_reasons = row.paper_ready_block_reasons
        if not paper_ready_block_reasons:
            paper_ready_block_reasons = _advisory_reason_list(
                meta.get("paper_ready_block_reasons")
            )
        output.append(
            row.model_copy(
                update={
                    "source_module": row.source_module or _text_value(meta.get("source_module")),
                    "template_family": row.template_family
                    or _text_value(meta.get("template_family")),
                    "candidate_id": row.candidate_id or _text_value(meta.get("candidate_id")),
                    "promotion_state": row.promotion_state
                    or _text_value(meta.get("promotion_state")).upper(),
                    "alpha_factory_score": row.alpha_factory_score
                    if row.alpha_factory_score is not None
                    else _optional_float(meta.get("alpha_factory_score")),
                    "cost_quality_score": row.cost_quality_score
                    if row.cost_quality_score is not None
                    else _optional_float(meta.get("cost_quality_score")),
                    "paper_ready_block_reasons": paper_ready_block_reasons,
                    "max_live_notional_usdt": 0.0,
                }
            )
        )
    return output


def _alpha_factory_promotion_for_advisory_row(
    row: StrategyOpportunityAdvisoryRow,
    promotions: dict[tuple[str, str, int | None], dict[str, Any]],
) -> dict[str, Any] | None:
    candidates = [row.strategy_candidate]
    if row.strategy_candidate.startswith("regime_router:"):
        candidates.append(row.strategy_candidate.split(":", 1)[1])
    for candidate in candidates:
        for key in [
            (candidate, row.symbol, row.horizon_hours),
            (candidate, row.symbol, None),
            (candidate, "UNKNOWN", row.horizon_hours),
            (candidate, "UNKNOWN", None),
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
        if not _is_alpha_factory_promotion_capped_model(row):
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
                    "source_module": row.source_module or "alpha_factory",
                    "template_family": row.template_family
                    or _alpha_factory_template_family(promotion or {}),
                    "candidate_id": row.candidate_id
                    or _text_value((promotion or {}).get("candidate_id")),
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
        if _is_bottom_zone_probe_paper_advisory(row):
            output.append(row.model_copy(update={"max_live_notional_usdt": 0.0}))
            continue
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


def _is_bottom_zone_probe_paper_advisory(row: StrategyOpportunityAdvisoryRow) -> bool:
    identifiers = {
        str(row.strategy_id or "").strip().lower(),
        str(row.strategy_candidate or "").strip().lower(),
        str(row.template_family or "").strip().lower(),
    }
    return bool(
        BOTTOM_ZONE_PROBE_STRATEGY_ID.lower() in identifiers
        or identifiers.intersection(BOTTOM_ZONE_PROBE_CANDIDATES)
    )


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
        expanded_universe_maturity_state=_text_value(
            row.get("expanded_universe_maturity_state")
            or row.get("maturity_state")
            or row.get("decision")
        ),
        cost_quality=_text_value(row.get("cost_quality") or row.get("cost_source_quality")),
        cost_quality_score=_optional_float(row.get("cost_quality_score")),
        paper_ready_block_reasons=_advisory_reason_list(row.get("paper_ready_block_reasons")),
        advisory_intent=_text_value(row.get("advisory_intent"))
        or _strategy_advisory_intent(recommended_mode),
        live_block_reasons=_advisory_reason_list(row.get("live_block_reasons")),
        max_paper_notional_usdt=_optional_float(row.get("max_paper_notional_usdt")),
        max_live_notional_usdt=max_live_notional,
        live_order_effect=_text_value(row.get("live_order_effect"))
        or "read_only_no_live_order",
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
    missing_required_expiry = expires_at is None and permission.enforceable is True
    if expires_at is None and as_of is not None and not missing_required_expiry:
        expires_at = as_of + timedelta(seconds=max(_risk_permission_ttl_seconds(), 0))
    expired = expires_at is not None and expires_at < datetime.now(UTC)
    stale_vs_telemetry = risk_permission_stale_vs_telemetry(
        as_of_ts=as_of,
        telemetry_latest_ts=latest_telemetry,
        threshold_seconds=DEFAULT_TELEMETRY_STALE_THRESHOLD_SECONDS,
    )
    published_not_enforceable = permission.enforceable is False
    status_value = (
        RiskPermissionStatus.NO_FRESH_PERMISSION
        if missing_required_expiry
        else permission_status(
            permission.permission.value,
            stale=stale_vs_telemetry or published_not_enforceable,
            expired=expired,
        )
    )
    reasons = list(permission.reasons)
    if missing_required_expiry and "published_permission_missing_expiry" not in reasons:
        reasons.append("published_permission_missing_expiry")
    if expired and "permission_expired" not in reasons:
        reasons.append("permission_expired")
    if stale_vs_telemetry and "risk_permission_stale_vs_v5_telemetry" not in reasons:
        reasons.append("risk_permission_stale_vs_v5_telemetry")
    if published_not_enforceable and "published_permission_not_enforceable" not in reasons:
        reasons.append("published_permission_not_enforceable")
    return permission.model_copy(
        update={
            "as_of_ts": as_of,
            "expires_at": expires_at,
            "telemetry_latest_ts": latest_telemetry,
            "permission_freshness_sec": _telemetry_freshness_seconds(as_of, latest_telemetry),
            "permission_status": status_value,
            "enforceable": (
                is_permission_status_enforceable(status_value)
                and not expired
                and not missing_required_expiry
                and not published_not_enforceable
            ),
            "risk_reason_codes": reasons,
            "reasons": reasons,
            "reason": reasons[0] if reasons else None,
        }
    )


def _published_permission_is_active(permission: RiskPermission) -> bool:
    return is_permission_status_enforceable(
        permission.permission_status
    ) and permission.enforceable is True and not _permission_expired(permission)


def _permission_expired(permission: RiskPermission) -> bool:
    return permission.expires_at is not None and permission.expires_at < datetime.now(UTC)


def _permission_missing_required_expiry(permission: RiskPermission) -> bool:
    return permission.expires_at is None and "published_permission_missing_expiry" in set(
        permission.risk_reason_codes or permission.reasons
    )


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
    extra_reasons: list[str] | None = None,
) -> RiskPermission:
    reasons = list(permission.reasons)
    reason = "no_fresh_published_permission"
    risk_reason_codes = _dedupe(
        [*permission.risk_reason_codes, *reasons, *(extra_reasons or []), reason]
    )
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
        actual_rows = _optional_int(health.get("actual_rows")) or 0
        fallback_ratio = _optional_float(health.get("fallback_ratio")) or 0.0
        live_coverage = _lake_live_universe_cost_coverage(lake_root)
        warnings = list(health.get("warnings") if isinstance(health.get("warnings"), list) else [])
        if live_coverage.get("coverage_status") == "WARNING":
            status = "warning" if status == "ok" else status
            warnings.append("live_universe_cost_coverage_warning")
        warnings = list(dict.fromkeys(warnings))
        return {
            "status": status,
            "missing": status == "critical" and actual_rows == 0,
            "high_fallback": fallback_ratio > 0.5,
            "fallback_ratio": fallback_ratio,
            "hard_fallback_ratio": _optional_float(health.get("hard_fallback_ratio")) or 0.0,
            "soft_fallback_ratio": _optional_float(health.get("soft_fallback_ratio")) or 0.0,
            "actual_rows": actual_rows,
            "mixed_rows": _optional_int(health.get("mixed_rows")) or 0,
            "proxy_rows": _optional_int(health.get("proxy_rows")) or 0,
            "global_default_rows": _optional_int(health.get("global_default_rows")) or 0,
            "proxy_only_count": _optional_int(health.get("proxy_only_count")) or 0,
            "warnings": warnings,
            "symbols_missing_cost": health.get("symbols_missing_cost")
            if isinstance(health.get("symbols_missing_cost"), list)
            else [],
            "cost_model_version": str(health.get("cost_model_version") or "cost_health_daily"),
            "live_universe_cost_coverage": live_coverage,
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


def _lake_live_universe_cost_coverage(lake_root: Path) -> dict[str, Any]:
    lazy, _columns = _safe_parquet_lazy(lake_root / "gold" / "cost_bucket_daily")
    if lazy is None:
        return {
            "status": "not_observable",
            "coverage_status": "NOT_OBSERVABLE",
            "coverage_rate": None,
            "target_coverage": None,
            "reason": "cost_bucket_daily_missing",
        }
    frame = _collect_lazy_or_empty(lazy)
    if frame.is_empty():
        return {
            "status": "not_observable",
            "coverage_status": "NOT_OBSERVABLE",
            "coverage_rate": None,
            "target_coverage": None,
            "reason": "cost_bucket_daily_empty",
        }
    try:
        evaluation = evaluate_live_universe_cost_coverage(frame)
    except Exception as exc:
        return {
            "status": "not_observable",
            "coverage_status": "NOT_OBSERVABLE",
            "coverage_rate": None,
            "target_coverage": None,
            "reason": "live_universe_cost_coverage_unavailable",
            "error_type": type(exc).__name__,
        }
    coverage_status = str(evaluation.get("coverage_status") or "UNKNOWN")
    return {
        "status": "ok" if coverage_status == "PASS" else "warning",
        "coverage_status": coverage_status,
        "coverage_rate": evaluation.get("coverage_rate"),
        "target_coverage": evaluation.get("target_coverage"),
        "covered_symbols": evaluation.get("covered_symbols", []),
        "direct_symbols": evaluation.get("direct_symbols", []),
        "mixed_proxy_symbols": evaluation.get("mixed_proxy_symbols", []),
        "stale_actual_or_mixed_symbols": evaluation.get("stale_actual_or_mixed_symbols", []),
        "missing_symbols": evaluation.get("missing_symbols", []),
        "proxy_only_symbols": evaluation.get("proxy_only_symbols", []),
        "detail_by_symbol": evaluation.get("detail_by_symbol", {}),
    }


def _deep_health_status(
    *,
    data_health: dict[str, Any],
    cost_health: dict[str, Any],
    risk_permission_dependency_meta: dict[str, Any],
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    critical = False
    for name, payload in (
        ("data_health", data_health),
        ("cost_health", cost_health),
        ("risk_permission_dependency_meta", risk_permission_dependency_meta),
    ):
        status_value = str(payload.get("status") or "").strip().lower()
        is_missing = status_value == "missing" or bool(payload.get("missing"))
        is_critical = status_value == "critical" or bool(payload.get("is_critical"))
        if is_critical or (name in {"data_health", "cost_health"} and is_missing):
            critical = True
            warnings.append(f"{name}_{status_value or 'critical'}")
            continue
        if (
            status_value in {"warning", "stale"}
            or bool(payload.get("stale"))
            or bool(payload.get("high_fallback"))
        ):
            warnings.append(f"{name}_{status_value or 'warning'}")
    if critical:
        return "critical", _dedupe(warnings)
    if warnings:
        return "warning", _dedupe(warnings)
    return "ok", []


def _deep_data_quality_status(overall_status: str, warnings: list[str]) -> dict[str, Any]:
    status = str(overall_status or "ok").strip().upper()
    if status == "OK":
        normalized = "OK"
    elif status == "WARNING":
        normalized = "WARN"
    else:
        normalized = status
    return {
        "status": normalized,
        "warnings": list(warnings),
        "warning_count": len(warnings),
    }


def _live_entry_readiness_health(lake_root: Path) -> dict[str, Any]:
    payload = _read_json_file(lake_root / "reports" / "v5_enforce_readiness.json")
    status = str(payload.get("readiness_status") or "UNKNOWN").strip().upper()
    veto_status = str(payload.get("veto_status") or "VETO_READY").strip().upper()
    entry_status = str(payload.get("entry_status") or "").strip().upper()
    scale_status = str(payload.get("scale_status") or "").strip().upper()
    if entry_status and scale_status:
        pass
    elif status == "READY":
        entry_status = "ENTRY_READY"
        scale_status = "SCALE_READY"
    elif status in {"ADVISORY_READY", "WARN"}:
        entry_status = "ENTRY_BLOCKED"
        scale_status = "SCALE_BLOCKED"
    else:
        entry_status = "ENTRY_BLOCKED"
        scale_status = "SCALE_BLOCKED"
    return {
        "status": status,
        "veto_status": veto_status,
        "entry_status": entry_status,
        "scale_status": scale_status,
        "blocked_reasons": list(payload.get("blocked_reasons") or []),
        "warning_reasons": list(payload.get("warning_reasons") or []),
        "veto_blocked_reasons": list(payload.get("veto_blocked_reasons") or []),
        "entry_blocked_reasons": list(payload.get("entry_blocked_reasons") or []),
        "scale_blocked_reasons": list(payload.get("scale_blocked_reasons") or []),
        "source": "reports/v5_enforce_readiness.json" if payload else "missing",
    }


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _lake_data_health(lake_root: Path) -> dict[str, Any]:
    latest_from_health = _latest_market_bar_health(lake_root)
    if latest_from_health is not None:
        return _market_bar_data_health_from_latest(
            latest_from_health["latest_ts"],
            timeframe=latest_from_health.get("timeframe"),
            latest_close_ts=latest_from_health.get("latest_close_ts"),
        )

    lazy, columns = _safe_parquet_lazy(lake_root / "silver" / "market_bar")
    if lazy is None or "ts" not in columns:
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_missing"],
        }
    scoped = lazy.with_columns(_lazy_utc_datetime("ts").alias("latest_ts")).sort(
        "latest_ts",
        descending=True,
        nulls_last=True,
    )
    select_exprs: list[pl.Expr] = [pl.col("latest_ts")]
    if "timeframe" in columns:
        select_exprs.append(pl.col("timeframe").cast(pl.Utf8, strict=False).alias("timeframe"))
    else:
        select_exprs.append(pl.lit(DEFAULT_MARKET_BAR_TIMEFRAME).alias("timeframe"))
    latest_frame = _collect_lazy_or_empty(scoped.select(select_exprs).limit(1))
    latest_ts = latest_frame.item(0, "latest_ts") if not latest_frame.is_empty() else None
    timeframe = (
        str(latest_frame.item(0, "timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME)
        if not latest_frame.is_empty()
        else DEFAULT_MARKET_BAR_TIMEFRAME
    )
    if not isinstance(latest_ts, datetime):
        return {
            "status": "critical",
            "is_critical": True,
            "reasons": ["market_bar_invalid_timestamp"],
        }
    return _market_bar_data_health_from_latest(latest_ts, timeframe=timeframe)


def _latest_market_bar_health(lake_root: Path) -> dict[str, Any] | None:
    lazy, columns = _safe_parquet_lazy(lake_root / "silver" / "market_bar_health")
    if lazy is None or "latest_ts" not in columns:
        return None
    scoped = lazy.with_columns(_lazy_utc_datetime("latest_ts").alias("latest_ts")).sort(
        "latest_ts",
        descending=True,
        nulls_last=True,
    )
    select_exprs: list[pl.Expr] = [pl.col("latest_ts")]
    if "latest_timeframe" in columns:
        select_exprs.append(
            pl.col("latest_timeframe").cast(pl.Utf8, strict=False).alias("timeframe")
        )
    else:
        select_exprs.append(pl.lit(DEFAULT_MARKET_BAR_TIMEFRAME).alias("timeframe"))
    if "latest_close_ts" in columns:
        select_exprs.append(_lazy_utc_datetime("latest_close_ts").alias("latest_close_ts"))
    else:
        select_exprs.append(pl.lit(None).alias("latest_close_ts"))
    latest_frame = _collect_lazy_or_empty(scoped.select(select_exprs).limit(1))
    latest_ts = latest_frame.item(0, "latest_ts") if not latest_frame.is_empty() else None
    if not isinstance(latest_ts, datetime):
        return None
    timeframe = str(latest_frame.item(0, "timeframe") or DEFAULT_MARKET_BAR_TIMEFRAME)
    latest_close_ts = ensure_utc_datetime(latest_frame.item(0, "latest_close_ts"))
    return {
        "latest_ts": latest_ts.astimezone(UTC),
        "timeframe": timeframe,
        "latest_close_ts": latest_close_ts,
    }


def _market_bar_data_health_from_latest(
    latest_ts: datetime,
    *,
    timeframe: str | None = None,
    latest_close_ts: datetime | None = None,
) -> dict[str, Any]:
    latest_utc = latest_ts.astimezone(UTC) if latest_ts.tzinfo else latest_ts.replace(tzinfo=UTC)
    close_utc = latest_close_ts or market_bar_close_ts(latest_utc, timeframe)
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
