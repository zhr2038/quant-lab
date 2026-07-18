"""Layered decision schema and v2 safety rules."""

from __future__ import annotations

VALID = {"PASS", "FAIL", "INCONCLUSIVE"}
REPLAYABILITY = {"REPLAYABLE", "PARTIALLY_REPLAYABLE", "NOT_REPLAYABLE"}
DASHBOARD_REQUIRED_LABELS = (
    "Signal Validity",
    "Portfolio Validity",
    "Deployment Readiness",
    "POST_HOC",
    "FORWARD ONLY",
    "PARTIALLY_REPLAYABLE",
    "LIVE ALPHA FROZEN",
    "CONFIRMATORY",
    "EXPLORATORY",
)


def validate_layered_decisions(payload: dict) -> None:
    """Fail closed when a v2 decision violates the three-layer contract."""
    for name in ("current_v5", "rev_xs_20d", "funding_fade"):
        item = payload[name]
        for field in ("signal_validity", "portfolio_validity", "deployment_readiness"):
            if item[field] not in VALID:
                raise ValueError(f"{name}.{field} has invalid value: {item[field]}")
    low_vol = payload["low_vol_20d"]
    for field in ("signal_validity", "locked_portfolio_validity", "deployment_readiness"):
        if low_vol[field] not in VALID:
            raise ValueError(f"low_vol_20d.{field} has invalid value: {low_vol[field]}")
    post_hoc = payload["low_vol_btc_trend"]
    if post_hoc.get("hypothesis_type") != "POST_HOC_HYPOTHESIS":
        raise ValueError("low_vol_btc_trend must be POST_HOC_HYPOTHESIS")
    if post_hoc.get("parameters_locked") is not True:
        raise ValueError("post-hoc parameters must be locked")
    for field in ("portfolio_validity", "deployment_readiness"):
        if post_hoc[field] != "INCONCLUSIVE":
            raise ValueError(f"post-hoc {field} cannot exceed INCONCLUSIVE")
    if payload["current_v5"]["replayability"] not in REPLAYABILITY:
        raise ValueError("invalid current_v5.replayability")
    # Audit v2 never authorizes deployment.
    for name, item in payload.items():
        if isinstance(item, dict) and item.get("deployment_readiness") == "PASS":
            raise ValueError(f"{name} cannot receive deployment PASS in audit v2")


def validate_dashboard_contract(content: str) -> None:
    """Fail when a rendered v2 dashboard omits a mandatory audit boundary."""
    missing = [label for label in DASHBOARD_REQUIRED_LABELS if label not in content]
    if missing:
        raise ValueError(f"dashboard is missing required labels: {', '.join(missing)}")
