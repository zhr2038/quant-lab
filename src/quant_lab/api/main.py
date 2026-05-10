"""Read-only FastAPI surface for quant-lab strategy consumers.

This API is read-only and must not mutate strategy state or exchange state.
"""

import os
from datetime import datetime
from pathlib import Path

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
from quant_lab.costs.model import CostBucket, estimate_cost_bps
from quant_lab.data.lake import read_market_bars
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
    ) -> CostEstimate:
        _ = quantile
        buckets = [
            CostBucket(
                bucket_id=f"{symbol}-default",
                symbol=symbol,
                regime=regime,
                min_notional_usdt=0,
                max_notional_usdt=100_000,
                cost_bps=4.2,
            )
        ]
        return estimate_cost_bps(symbol, regime, notional_usdt, buckets)

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
        evidence = AlphaEvidence.example_live_ready().model_copy(
            update={"alpha_id": f"{strategy}-demo-alpha", "version": version}
        )
        gate_decision = evaluate_alpha_gate(evidence)
        return evaluate_live_permission(
            strategy=strategy,
            version=version,
            gate_decisions=[gate_decision],
            cost_health={"status": "ok", "cost_model_version": evidence.cost_model_version},
            data_health={"status": "ok"},
        )

    return app


app = create_app()
