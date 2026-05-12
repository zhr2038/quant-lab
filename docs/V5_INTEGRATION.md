# V5/V7 Integration

`quant-lab` is a read-only research middle layer. It provides cost estimates,
gate decisions, and risk permissions to V5/V7, but it does not place orders,
cancel orders, amend orders, reconcile exchange state, manage balances, or
replace V5/V7 kill-switch logic.

V5/V7 remain responsible for all execution behavior.

## Client Usage

Install `quant-lab` in the V5/V7 runtime environment or vendor
`quant_lab.client.QuantLabClient` as a small read-only dependency.

```python
import os

from quant_lab.client import QuantLabClient, QuantLabUnavailable

client = QuantLabClient(
    base_url="http://127.0.0.1:8027",
    timeout_seconds=1.0,
    api_token=os.environ.get("QUANT_LAB_API_TOKEN"),
)

try:
    cost = client.estimate_cost(
        symbol="BTC-USDT",
        regime="normal",
        notional_usdt=10_000,
        quantile="p75",
    )
except QuantLabUnavailable:
    if config.allow_local_cost_fallback:
        cost = local_cost_model.estimate("BTC-USDT", "normal", 10_000)
        decision_audit["cost_source"] = "quant_lab_unavailable_local_fallback"
    else:
        decision_audit["cost_source"] = "quant_lab_unavailable_abort"
        return "ABORT"
else:
    decision_audit["cost_source"] = "quant_lab"
    decision_audit["cost_bps"] = cost.cost_bps
    decision_audit["cost_fallback_level"] = cost.fallback_level
```

When qyun2 sets `QUANT_LAB_API_TOKEN`, V5/V7 must send it as a bearer token.
Store it only in the process environment or a root-owned runtime config file;
do not write it to V5 bundles, logs, lake files, or expert exports.

Fallbacks must be explicit. If V5 uses a local cost model because quant-lab is
unavailable, it must write this exact audit value:

```python
decision_audit["cost_source"] = "quant_lab_unavailable_local_fallback"
```

## Live Permission

V5/V7 should request live permission before entering paper or live modes.
Permission is a research/risk signal only; V5/V7 still decide and execute their
own state transitions.

```python
from quant_lab.client import QuantLabPermissionError, QuantLabUnavailable

try:
    permission = client.get_live_permission(strategy="v5", version=STRATEGY_VERSION)
except (QuantLabUnavailable, QuantLabPermissionError):
    permission_value = operator_policy.default_when_quant_lab_unavailable
    if permission_value not in {"ABORT", "SELL_ONLY"}:
        permission_value = "ABORT"
else:
    permission_value = permission.permission

if permission_value == "ABORT":
    decision_audit["quant_lab_permission"] = "ABORT"
    return "ABORT"

if permission_value == "SELL_ONLY":
    decision_audit["quant_lab_permission"] = "SELL_ONLY"
    risk_mode = "SELL_ONLY"
    return risk_mode

if permission_value == "ALLOW":
    if target_mode not in permission.allowed_modes:
        decision_audit["quant_lab_permission"] = "MODE_NOT_ALLOWED"
        return "ABORT"
    decision_audit["quant_lab_permission"] = "ALLOW"
    decision_audit["max_gross_exposure"] = permission.max_gross_exposure
    decision_audit["max_single_weight"] = permission.max_single_weight
```

If live permission cannot be read, the default must be conservative:

- `ABORT`, or
- `SELL_ONLY` if the V5 operator policy explicitly allows that degraded mode.

## Gate Decision

Gate decisions are read from quant-lab and used as evidence. V5/V7 must not
write gold gate tables.

```python
decision = client.get_gate_decision(alpha_id="mean-reversion-btc-1h")

if decision.status == "DEAD":
    return "ABORT"
if decision.status == "QUARANTINE":
    return "SELL_ONLY"
if decision.status == "PAPER_READY":
    target_mode = "paper"
if decision.status == "LIVE_READY":
    target_mode = "live_canary"
```

## Boundary

The client only uses GET requests. It must not expose POST, PUT, PATCH, DELETE,
execution, order, funds transfer, withdrawal, or account mutation behavior.
