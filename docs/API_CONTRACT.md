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
    "cost_bucket_daily",
    "alpha_evidence",
    "gate_decision",
    "risk_permission"
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

Returns a read-only cost estimate. In v0.1 this endpoint uses a deterministic
default bucket in the API layer; calibrated lake-backed lookup is a follow-up.

Request params:

- `symbol` required string, for example `BTC-USDT`.
- `regime` required string, for example `normal`.
- `notional_usdt` required positive number.
- `quantile` optional string, default `p75`; accepted for client compatibility.

Example response:

```json
{
  "symbol": "BTC-USDT",
  "regime": "normal",
  "notional_usdt": 10000.0,
  "cost_bps": 4.2,
  "fallback_level": "NONE",
  "bucket_id": "BTC-USDT-default"
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

Returns the current gate decision shape for an alpha. In v0.1 this is a
deterministic demo response using the requested `alpha_id`; lake-backed
`gold/gate_decision` reads are a follow-up.

Path params:

- `alpha_id` required string.

Allowed statuses: `DEAD`, `QUARANTINE`, `PAPER_READY`, `LIVE_READY`.

## GET /v1/risk/live-permission

Returns the current read-only risk permission for a strategy/version pair. This
is a research permission signal, not an execution command.

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

## Planned But Not Exposed In v0.1

The following strategy-facing contracts remain planned follow-ups and are not
currently registered FastAPI routes:

- `GET /v1/features/latest`
- `GET /v1/research/alpha/{alpha_id}`
