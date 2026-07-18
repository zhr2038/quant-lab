"""Cross-format consistency gates for Audit v2.1 reports."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping


def _extract(pattern: str, text: str, label: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"missing {label} in rendered report")
    return float(match.group(1))


def validate_v21_report_contract(
    *,
    status: Mapping,
    performance: Mapping,
    dashboard_html: str,
    forward_markdown: str,
    decisions: Mapping,
) -> dict:
    expected = float(status["forward_available_days"])
    values = {
        "status_json": expected,
        "performance_csv": float(performance["forward_available_days"]),
        "dashboard_html": _extract(
            r'data-forward-available-days="([0-9.]+)"',
            dashboard_html,
            "dashboard forward days",
        ),
        "forward_markdown": _extract(
            r"Forward available days \| `([0-9.]+)`",
            forward_markdown,
            "Markdown forward days",
        ),
        "final_decisions_json": float(
            decisions["forward_v21"]["available_days"]
        ),
    }
    if max(values.values()) - min(values.values()) >= 1e-9:
        raise ValueError(f"forward day values differ across outputs: {values}")
    required_dashboard_labels = (
        "Forward Runner",
        "Funding 时间窗口",
        "INVALIDATED",
        "PRODUCTION ALPHA · FROZEN",
        "已实现净复合",
        "未实现净标记",
        "BTC 基准",
        "动态币池基准",
        "Parameter lock",
        "INCONCLUSIVE",
    )
    missing = [label for label in required_dashboard_labels if label not in dashboard_html]
    if missing:
        raise ValueError(f"dashboard contract labels missing: {missing}")
    forbidden = {
        decisions["forward_v21"].get("portfolio_validity"),
        decisions["forward_v21"].get("deployment_readiness"),
    } & {"PASS", "LIVE", "LIVE_SMALL"}
    if forbidden:
        raise ValueError(f"forward decision was automatically promoted: {forbidden}")
    if decisions.get("live_or_live_small_permitted") is not False:
        raise ValueError("live status is not explicitly forbidden")
    if decisions["current_v5"].get("production_alpha") != "FROZEN":
        raise ValueError("production Alpha freeze disappeared")
    if not all(math.isfinite(value) and value >= 0.0 for value in values.values()):
        raise ValueError("forward available days must be finite and non-negative")
    return {
        "forward_available_days": values,
        "required_dashboard_labels": list(required_dashboard_labels),
        "final_decision_constraints": {
            "no_forward_pass": True,
            "no_live": True,
            "production_alpha_frozen": True,
        },
        "consistent": True,
    }
