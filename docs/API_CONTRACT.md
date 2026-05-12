# API Contract

This document describes the implemented v0.1 FastAPI surface. All public
strategy-facing routes under `/v1` are GET-only. The service publishes
read-only research state; V5/V7 continue to own execution, order state,
reconcile, balances, and kill-switch behavior.

`quant-lab` must not expose order placement, cancellation, amendment, funds
transfer, withdrawal, private trading mutation, or account mutation endpoints.

## Source Policy

- `quant-lab` directly ingests OKX public market data.
- OKX public data does not require an API key.
- Optional OKX read-only private GET ingestion may be enabled for real
  fills/bills/order-history cost evidence.
- Any optional OKX private key must have Read permission only.
- Trade permission and Withdraw permission are forbidden.
- The V5 reports importer is optional legacy tooling, not the primary source.

## Authentication

All `/v1/*` routes support optional bearer-token protection. When
`QUANT_LAB_API_TOKEN` is set in the API process environment, clients must send:

```text
Authorization: Bearer <token>
```

`/v1/health` uses the same rule, except localhost health checks may be allowed
when `QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH=true` (the default). Set
`QUANT_LAB_HEALTH_ALLOW_LOCAL_UNAUTH=false` to require the bearer token for
health checks as well.

Do not write `QUANT_LAB_API_TOKEN` to lake files, logs, expert exports, or V5
bundles.

## GET /v1/health

Request params: none.

Example response:

```json
{
  "status": "ok",
  "service": "quant-lab",
  "mode": "read-only"
}
```

## GET /v1/catalog/datasets

Lists known v0.1 dataset names.

Request params: none.

Example response:

```json
{
  "datasets": [
    "market_bar",
    "feature_value",
    "feature_coverage_daily",
    "feature_anomaly_daily",
    "cost_bucket_daily",
    "cost_health_daily",
    "alpha_evidence",
    "gate_decision",
    "risk_permission",
    "v5_quant_lab_mode_daily",
    "v5_quant_lab_enforcement_daily"
  ]
}
```

## GET /v1/market/bars

Returns closed normalized market bars from `lake/silver/market_bar`.

Request params:

- `venue` required string, for example `okx`.
- `symbol` required string, for example `BTC-USDT`.
- `timeframe` required string, for example `1H`.
- `start` required UTC datetime.
- `end` required UTC datetime.

Example response:

```json
[
  {
    "venue": "okx",
    "symbol": "BTC-USDT",
    "market_type": "SPOT",
    "timeframe": "1H",
    "ts": "2026-05-10T01:00:00Z",
    "open": 100.0,
    "high": 110.0,
    "low": 90.0,
    "close": 105.0,
    "volume": 12.0,
    "quote_volume": 1260.0,
    "source": "okx_public_rest",
    "ingest_ts": "2026-05-10T02:00:00Z",
    "is_closed": true
  }
]
```

## GET /v1/costs/example

Returns a deterministic example cost estimate.

Request params: none.

Example response:

```json
{
  "symbol": "BTCUSDT",
  "regime": "normal",
  "notional_usdt": 10000.0,
  "cost_bps": 4.2,
  "fallback_level": "NONE",
  "bucket_id": "btc-default"
}
```

## GET /v1/costs/estimate

Returns a read-only lake-backed cost estimate from
`lake/gold/cost_bucket_daily`. If no calibrated rows exist, the endpoint returns
an explicit `global_default` fallback and does not claim actual OKX costs.

Request params:

- `symbol` required string, for example `BTC-USDT`.
- `regime` required string, for example `normal`.
- `notional_usdt` required positive number.
- `quantile` optional string, default `p75`; one of `p50`, `p75`, `p90`.
- `notional_bucket` optional string, for explicit bucket lookup.

Example response:

```json
{
  "symbol": "BTC-USDT",
  "regime": "normal",
  "notional_usdt": 10000.0,
  "quantile": "p75",
  "fee_bps": 1.0,
  "slippage_bps": 0.0,
  "spread_bps": 1.5,
  "total_cost_bps": 2.5,
  "cost_bps": 2.5,
  "fallback_level": "PUBLIC_SPREAD_PROXY",
  "source": "public_spread_proxy",
  "sample_count": 120,
  "cost_model_version": "cost_bucket_daily:2026-05-11",
  "bucket_id": "2026-05-11:BTC-USDT:public_proxy:spread_proxy:all"
}
```

## GET /v1/features/latest

Returns the latest read-only feature values from `lake/gold/feature_value`.

Request params:

- `feature_set` required string, for example `core`.
- `feature_version` optional string.
- `symbols` optional comma-separated symbols.
- `timeframe` optional string.
- `feature_names` optional comma-separated feature names.

Example response:

```json
{
  "feature_set": "core",
  "feature_version": "v0.1",
  "timeframe": "1H",
  "created_at": "2026-05-11T00:00:00Z",
  "rows": [
    {
      "symbol": "BTC-USDT",
      "ts": "2026-05-11T00:00:00Z",
      "features": {
        "close_return_24": 0.012,
        "rolling_volatility_24": 0.004
      }
    }
  ],
  "warnings": []
}
```

## GET /v1/gates/example

Returns a deterministic example gate decision.

Request params: none.

Example response:

```json
{
  "alpha_id": "example-alpha",
  "version": "v1",
  "gate_version": "default-v0.1",
  "status": "LIVE_READY",
  "passed": true,
  "reasons": ["all_default_gates_passed"],
  "metrics": {
    "coverage": 0.99,
    "ic_tstat": 3.1,
    "edge_cost_ratio": 2.4,
    "oos_sharpe": 1.2,
    "paper_days": 21
  },
  "next_action": "eligible_for_strategy_consumer_review",
  "created_at": "2026-05-10T00:00:00Z"
}
```

## GET /v1/gates/decision/{alpha_id}

Returns the latest lake-backed gate decision from `lake/gold/gate_decision`.
If no row exists for the alpha, the API returns a conservative `QUARANTINE`
response with reason `missing_gate_decision`.

Path params:

- `alpha_id` required string.

Allowed statuses: `DEAD`, `QUARANTINE`, `PAPER_READY`, `LIVE_READY`.

## GET /v1/research/alpha/{alpha_id}

Returns the latest alpha evidence and gate decision for an alpha.

Path params:

- `alpha_id` required string.

Example response:

```json
{
  "alpha_id": "v5.core.momentum",
  "evidence": {
    "alpha_id": "v5.core.momentum",
    "version": "v0.1",
    "coverage": 0.98,
    "ic_mean": 0.03,
    "ic_tstat": 2.4,
    "edge_cost_ratio": 1.8,
    "paper_days": 0
  },
  "gate_decision": {
    "alpha_id": "v5.core.momentum",
    "status": "PAPER_READY",
    "reasons": ["needs_paper_observation"]
  },
  "warnings": []
}
```

## GET /v1/risk/live-permission

Returns the current read-only risk permission for a strategy/version pair. The
API reads `lake/gold/risk_permission` when published and fresh, otherwise it
computes a conservative response from gate, cost, market-data, and V5 telemetry
health. Published permissions older than
`QUANT_LAB_RISK_PERMISSION_TTL_SECONDS` (default `300`) are recomputed. If the
recomputed permission is more conservative than the published row, the API
returns the recomputed permission. It is a research permission signal, not an
execution command.

Request params:

- `strategy` required string, for example `v5`.
- `version` required string, for example `v1`.

Example response:

```json
{
  "strategy": "v5",
  "version": "v1",
  "permission": "ALLOW",
  "allowed_modes": ["paper", "live_canary"],
  "max_gross_exposure": 0.25,
  "max_single_weight": 0.05,
  "cost_model_version": "costs-v1",
  "gate_version": "default-v0.1",
  "reasons": ["all_required_alpha_gates_live_ready"],
  "created_at": "2026-05-10T00:00:00Z"
}
```

Allowed permissions: `ALLOW`, `SELL_ONLY`, `ABORT`.

## GET /v1/risk/live-permission-detail

Returns the same `RiskPermission` plus API-side audit metadata such as cache
freshness, data health, cost health, gate summary, and V5 telemetry summary.
Use this endpoint for operations diagnostics; strategy consumers that only need
the permission contract can keep using `/v1/risk/live-permission`.

Request params:

- `strategy` required string, for example `v5`.
- `version` required string, for example `v1`.

Example response:

```json
{
  "permission": {
    "strategy": "v5",
    "version": "v1",
    "permission": "SELL_ONLY",
    "allowed_modes": ["sell_only"],
    "max_gross_exposure": 0.0,
    "max_single_weight": 0.0,
    "cost_model_version": "costs-v1",
    "gate_version": "default-v0.1",
    "reasons": ["cost_health_missing"],
    "created_at": "2026-05-10T00:00:00Z"
  },
  "permission_source": "recomputed",
  "permission_freshness_seconds": 420,
  "published_permission_stale": true,
  "data_health": {"status": "ok"},
  "cost_health": {"status": "missing"},
  "gate_summary": {"total": 1, "status_counts": {"LIVE_READY": 1}},
  "v5_telemetry_summary": {"status": "ok", "reasons": []}
}
```

All public `/v1` routes are GET-only.
