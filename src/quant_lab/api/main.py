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
from quant_lab.research.bootstrap_gold import BOOTSTRAP_GATE_VERSION
from quant_lab.risk.permissions import evaluate_live_permission


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


KNOWN_DATASETS = [
    "market_bar",
    "feature_value",
    "feature_coverage_daily",
    "feature_anomaly_daily",
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
            return permission.model_copy(
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
