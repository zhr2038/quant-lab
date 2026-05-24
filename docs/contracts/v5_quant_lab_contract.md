# V5 Quant Lab Integration Contract

Contract version: `v5.quant_lab.telemetry.v2`

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
- Strategy opportunity advisory rows exposed through
  `GET /v1/strategy-opportunity-advisory`.

## V5 Trade/Fill Summary

`trades.csv` rows are normalized into `silver/v5_trade_event` and can feed the
daily cost model as `mixed_actual_proxy` when fee evidence is present but
arrival/mid slippage is unavailable. Required cost-backfill fields are
`ts_utc`, `symbol`, `normalized_symbol`, `side`, `action`, `qty`, `price`,
`notional_usdt`, `fee`, `fee_ccy`, `fee_usdt`, `run_id`, and `strategy_id`.
`trade_id`, `order_id`, and `slippage_usdt` are optional. Missing optional
fields must not cause the whole row to be dropped.

## Strategy Opportunity Advisory

`GET /v1/strategy-opportunity-advisory`,
`GET /v1/strategy_opportunity_advisory`, and
`GET /v1/reports/strategy-opportunity-advisory` return the same read-only
advisory rows. Each row must include freshness and provenance metadata:

- `as_of_ts`, `generated_at`, `expires_at`
- `contract_version = v5.quant_lab.telemetry.v2`
- `schema_version = strategy_opportunity_advisory.v0.1`
- `quant_lab_git_commit` when a git commit can be resolved
- `source_version`, falling back to the quant-lab package version when git is unavailable

Each advisory row must explicitly expose V5-consumable intent fields:

- `advisory_intent` (`research_only`, `paper_shadow`, or future `live_command`)
- `would_block_if_enabled`
- `would_enter`
- `no_sample_reason`

These fields are advisory metadata only. quant-lab is not the V5 live
commander: `risk_permission=ABORT` means quant-lab's research/advisory layer
does not recommend live promotion, not that V5 must stop its local live
controller. `cost_trusted_for_live=false` is likewise an advisory risk flag, not
a strategy-facing hard block.
`PAPER_READY` rows must keep `max_live_notional_usdt = 0`; only
`LIVE_SMALL_READY` with system safety approval may ever expose live notional.

Alpha Factory and expanded-universe rows may additionally expose
`source_module`, `template_family`, `candidate_id`, `promotion_state`,
`alpha_factory_score`, `universe_type`, `cost_quality_score`, and
`paper_ready_block_reasons`.

## Runtime Version References

The Python runtime exposes the same versions from
`quant_lab.contracts.v5_quant_lab`. Cost estimates, risk permissions, V5
telemetry rows, and daily expert export manifest/provenance must carry these
versions so V5 and quant-lab can detect contract drift.
