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

## Risk Permission And Advisory

V5/V7 can read quant-lab risk permission before entering paper or local live
modes, but quant-lab is not the V5 live commander. Permission is a
research/advisory signal only; V5/V7 still own execution, reconcile,
positions, local kill-switches, and state transitions.

Use the newer fields first:

- `allowed_advisory_modes` tells V5 whether quant-lab has shadow/paper
  opportunities worth displaying or recording.
- `allowed_live_modes` is the only quant-lab field that may ever be interpreted
  as live escalation advice. In the current production contract it is normally
  empty.
- `live_block_reasons` explains why quant-lab is not recommending live
  promotion, for example `quant_lab_live_command_not_allowed` or
  `v5_local_live_not_controlled_by_quant_lab`.

Do not treat `permission=ABORT` as an instruction that V5 must stop local live
trading. It means quant-lab is not recommending live promotion from the
research/advisory layer.

```python
from quant_lab.client import QuantLabPermissionError, QuantLabUnavailable

try:
    permission = client.get_live_permission(strategy="v5", version=STRATEGY_VERSION)
except (QuantLabUnavailable, QuantLabPermissionError):
    decision_audit["quant_lab_permission"] = "UNAVAILABLE"
    decision_audit["quant_lab_advisory_modes"] = []
    decision_audit["quant_lab_live_modes"] = []
else:
    decision_audit["quant_lab_permission"] = permission.permission
    decision_audit["quant_lab_permission_status"] = permission.permission_status
    decision_audit["quant_lab_advisory_modes"] = permission.allowed_advisory_modes
    decision_audit["quant_lab_live_modes"] = permission.allowed_live_modes
    decision_audit["quant_lab_live_block_reasons"] = permission.live_block_reasons

    if target_mode in {"shadow", "paper"}:
        if target_mode not in permission.allowed_advisory_modes:
            decision_audit["quant_lab_advisory"] = "NOT_RECOMMENDED"
        else:
            decision_audit["quant_lab_advisory"] = "RECOMMENDED"

    if target_mode.startswith("live") and target_mode not in permission.allowed_live_modes:
        decision_audit["quant_lab_live_advisory"] = "NOT_RECOMMENDED"
```

If quant-lab cannot be read, V5 should record the failure and use its own
operator-approved local degraded policy. quant-lab unavailability must not cause
V5 to invent a live approval.

## Gate Decision

Gate decisions are read from quant-lab and used as evidence. The generic
`v5.core.momentum` gate is a research baseline, not the global V5 live gate.
V5/V7 must not write gold gate tables.

```python
decision = client.get_gate_decision(alpha_id="mean-reversion-btc-1h")

if decision.status == "DEAD":
    return "ABORT"
if decision.status == "QUARANTINE":
    return "SELL_ONLY"
if decision.status == "PAPER_READY":
    target_mode = "paper"
if decision.status == "LIVE_READY":
    target_mode = "paper_or_manual_live_review"
```

## Boundary

The client only uses GET requests. It must not expose POST, PUT, PATCH, DELETE,
execution, order, funds transfer, withdrawal, or account mutation behavior.
