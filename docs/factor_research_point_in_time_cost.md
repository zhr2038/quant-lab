# Factor Research v2 point-in-time cost policy

Factor Research v2 separates three questions that require different evidence windows:

1. Signal validity uses the locked historical market window and does not fail merely because
   execution-cost history is shorter.
2. Long-only portfolio validity uses a locked 30-day point-in-time cost window. At least 80% of
   the eligible symbol/hour samples must have a causal cost estimate.
3. Deployment readiness requires at least 80% trusted `actual` or `mixed_actual_proxy` coverage.
   Public spread proxies and bootstrap probes never satisfy this gate.

## Causal public proxy

The research-only fallback is reconstructed from immutable
`silver/orderbook_spread_1m` observations included in the signed research snapshot:

- observations are grouped by symbol and UTC day;
- duplicate public channels within one minute use the maximum spread;
- a day needs at least 60 distinct minute observations;
- the daily estimate is the p75 spread and becomes available only at the next UTC midnight;
- the round-trip research cost is floored at 30 bps to cover missing fees and slippage;
- the estimate expires after 36 hours;
- it is tagged `public_spread_proxy`, `eligible_for_live_cost_coverage=false`, and
  `cost_reconstructed_from_spread=true`.

Published `cost_bucket_daily` evidence takes precedence only when it was materialized no more
than 36 hours after its observation day closed. A late historical row is excluded instead of
overwriting the causal proxy. Timely trusted fill evidence has the highest precedence, followed
by timely public rows, bootstrap rows, and the reconstructed proxy.

The snapshot contains only the cost evaluation window plus the prior source day. It also binds
all projected files, hashes, row counts, and time bounds. The NAS worker remains read-only and
cannot publish Gold, promote a strategy, modify risk permission, or create an order.

## Interpretation

`cost_coverage >= 0.80` means the locked portfolio calculation has a conservative causal cost.
It does not mean production execution cost is known. `trusted_cost_coverage >= 0.80` is the
separate deployment prerequisite. The Web v2 Factor Research view exposes both values and labels
proxy-only results as `RESEARCH_PROXY_ONLY_DEPLOYMENT_BLOCKED`.
