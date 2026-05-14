# V5 Quant Lab Integration Contract

Contract version: `v5_quant_lab_contract.v0.1`

This contract defines the read-only integration between V5 and quant-lab.
quant-lab publishes research, cost, gate, and risk permission data. V5 remains
responsible for execution, reconcile, balances, and kill-switch behavior.

## Shared Rules

- All timestamps are UTC ISO8601 with `Z` or `+00:00`.
- Symbols are normalized to OKX-style `BASE-QUOTE`.
- Examples: `BNB/USDT`, `BNB-USDT`, `BNBUSDT`, and `OKX:BNB-USDT` all normalize to `BNB-USDT`.
- Strategy-facing quant-lab API calls are `GET` only.
- `global_default` cost estimates must include a degraded reason and may not be treated as actual cost.
- Stale or expired risk permissions are not enforceable.

The machine-readable source of truth is
`contracts/v5_quant_lab_contract.yaml`.

## Interfaces

The contract covers:

- `GET /v1/costs/estimate` request and response.
- `GET /v1/risk/live-permission` response.
- V5 `quant_lab_requests` events.
- V5 `quant_lab_fallback` events.
- V5 `quant_lab_cost_usage` events.
- V5 trade/fill summary rows used for cost backfill.

## V5 Trade/Fill Summary

`trades.csv` rows are normalized into `silver/v5_trade_event` and can feed the
daily cost model as `mixed_actual_proxy` when fee evidence is present but
arrival/mid slippage is unavailable. Required cost-backfill fields are
`ts_utc`, `symbol`, `normalized_symbol`, `side`, `action`, `qty`, `price`,
`notional_usdt`, `fee`, `fee_ccy`, `fee_usdt`, `run_id`, and `strategy_id`.
`trade_id`, `order_id`, and `slippage_usdt` are optional. Missing optional
fields must not cause the whole row to be dropped.

## Runtime Version References

The Python runtime exposes the same versions from
`quant_lab.contracts.v5_quant_lab`. Cost estimates, risk permissions, V5
telemetry rows, and daily expert export manifest/provenance must carry these
versions so V5 and quant-lab can detect contract drift.
