# Production Safety

## Invariants

- `research_only=true`
- `live_order_effect=none`
- `automatic_promotion=false`
- `max_live_notional_usdt=0`
- V5 repository and production runtime are read-only for this work
- no Paper ACK/Tracker mutation
- no risk-permission, canary, enforce, capital, position, or order change

Cloud and worker validators reject unknown decisions, missing outputs, stale or
superseded snapshots, commit mismatches, path traversal, symlinks, signature/hash
errors, non-zero notional, and any live effect. Anti-leakage accepts only an exact
all-PASS check set with zero violations.

The request service is guarded by both
`QUANT_LAB_NAS_RESEARCH_ENABLED=1` and
`QUANT_LAB_NAS_FACTOR_RESEARCH_ENABLED=1`. The example configuration sets both
the global and Factor Research switches to disabled. No local automatic fallback
exists.
