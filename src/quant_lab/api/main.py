"""Read-only FastAPI surface for quant-lab strategy consumers.

This API is read-only and must not mutate strategy state or exchange state.
"""

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.models import (
    AlphaEvidence,
    CostEstimate,
    GateDecision,
    MarketBar,
    RiskPermission,
)
from quant_lab.costs.model import CostBucket, estimate_cost_bps, estimate_cost_from_lake
from quant_lab.data.lake import read_market_bars, read_parquet_dataset
from quant_lab.gates.defaults import evaluate_alpha_gate
from quant_lab.risk.permissions import evaluate_live_permission


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: str = "ok"
    service: str = "quant-lab"
    mode: str = "read-only"


class CatalogDatasetsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[str] = Field(default_factory=list)


KNOWN_DATASETS = [
    "market_bar",
    "feature_value",
    "cost_bucket_daily",
    "alpha_evidence",
    "gate_decision",
    "risk_permission",
]


def create_app() -> FastAPI:
    app = FastAPI(
        title="quant-lab",
        version=__version__,
        description="Read-only quantitative research middle platform.",
    )

    @app.get("/v1/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/catalog/datasets", response_model=CatalogDatasetsResponse)
    def catalog_datasets() -> CatalogDatasetsResponse:
        return CatalogDatasetsResponse(datasets=KNOWN_DATASETS)

    @app.get("/v1/gates/example", response_model=GateDecision)
    def gate_example() -> GateDecision:
        evidence = AlphaEvidence.example_live_ready()
        return evaluate_alpha_gate(evidence)

    @app.get("/v1/gates/decision/{alpha_id}", response_model=GateDecision)
    def gate_decision(alpha_id: str) -> GateDecision:
        evidence = AlphaEvidence.example_live_ready().model_copy(update={"alpha_id": alpha_id})
        return evaluate_alpha_gate(evidence)

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
        lake_root = _lake_root()
        gate_decisions = _load_gate_decisions(lake_root, strategy=strategy)
        data_health = _lake_data_health(lake_root)
        cost_health = _lake_cost_health(lake_root)
        telemetry_reasons = _strategy_telemetry_reasons(lake_root, strategy=strategy)
        if telemetry_reasons:
            data_health = {
                **data_health,
                "status": "critical",
                "is_critical": True,
                "reasons": [*data_health.get("reasons", []), *telemetry_reasons],
            }
        return evaluate_live_permission(
            strategy=strategy,
            version=version,
            gate_decisions=gate_decisions,
            cost_health=cost_health,
            data_health=data_health,
        )

    return app


app = create_app()


def _lake_root() -> Path:
    return Path(os.environ.get("QUANT_LAB_LAKE_ROOT", "/var/lib/quant-lab/lake"))


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
        if isinstance(cleaned.get("metrics"), str):
            cleaned["metrics"] = _json_dict(cleaned["metrics"])
        if isinstance(cleaned.get("reasons"), str):
            cleaned["reasons"] = _json_list(cleaned["reasons"])
        try:
            decisions.append(GateDecision.model_validate(cleaned))
        except Exception:
            continue
    return decisions


def _lake_cost_health(lake_root: Path) -> dict[str, Any]:
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
