from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA_VERSION = "v5_trade_outcome_attribution.v0.1"
V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA = {
    "schema_version": pl.Utf8,
    "sample_id": pl.Utf8,
    "event_id": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "sample_type": pl.Utf8,
    "net_bps": pl.Float64,
    "entry_signal_quality": pl.Utf8,
    "exit_quality": pl.Utf8,
    "execution_quality": pl.Utf8,
    "market_tailwind": pl.Boolean,
    "cost_underestimated": pl.Boolean,
    "profit_lock_contribution": pl.Boolean,
    "would_have_been_blocked_by_quant_lab": pl.Boolean,
    "attribution": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def build_v5_trade_outcome_attribution(
    samples: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if samples.is_empty():
        return pl.DataFrame(schema=V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA)
    rows = [_attribution_row(row, created) for row in samples.to_dicts()]
    return _frame(rows, V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA)


def _attribution_row(sample: dict[str, Any], created: datetime) -> dict[str, Any]:
    net_bps = _float(sample.get("net_bps"))
    actual_all_in = _float(sample.get("actual_all_in_bps"))
    selected_cost = _float(sample.get("cost_bps"))
    cost_underestimated = (
        actual_all_in is not None
        and selected_cost is not None
        and actual_all_in > max(selected_cost + 10.0, selected_cost * 1.5)
    )
    profit_lock = "profit" in _text(sample.get("exit_reason")).lower()
    entry_signal_quality = _quality_from_net(net_bps)
    exit_quality = "PASS" if profit_lock or (net_bps is not None and net_bps > 0.0) else "UNKNOWN"
    execution_quality = _execution_quality(
        actual_all_in=actual_all_in,
        selected_cost=selected_cost,
        cost_underestimated=cost_underestimated,
    )
    reasons = []
    if entry_signal_quality == "PASS":
        reasons.append("entry_signal_after_cost_positive")
    elif entry_signal_quality == "FAIL":
        reasons.append("entry_signal_after_cost_negative")
    if profit_lock:
        reasons.append("profit_lock_exit")
    if cost_underestimated:
        reasons.append("actual_cost_above_selected_cost")
    if sample.get("quant_lab_false_block_candidate"):
        reasons.append("quant_lab_false_block_candidate")
    return {
        "schema_version": V5_TRADE_OUTCOME_ATTRIBUTION_SCHEMA_VERSION,
        "sample_id": _text(sample.get("sample_id")),
        "event_id": _text(sample.get("event_id")),
        "decision_ts": _timestamp(sample.get("decision_ts")),
        "symbol": _text(sample.get("symbol")),
        "strategy_candidate": _text(sample.get("strategy_candidate")),
        "sample_type": _text(sample.get("sample_type")),
        "net_bps": net_bps,
        "entry_signal_quality": entry_signal_quality,
        "exit_quality": exit_quality,
        "execution_quality": execution_quality,
        "market_tailwind": None,
        "cost_underestimated": cost_underestimated,
        "profit_lock_contribution": profit_lock,
        "would_have_been_blocked_by_quant_lab": bool(sample.get("quant_lab_would_block")),
        "attribution": ";".join(reasons) or "pending_or_unattributed",
        "created_at": created,
        "source": "quant_lab.trade_learning.attribution",
    }


def _quality_from_net(net_bps: float | None) -> str:
    if net_bps is None:
        return "UNKNOWN"
    return "PASS" if net_bps > 0.0 else "FAIL"


def _execution_quality(
    *,
    actual_all_in: float | None,
    selected_cost: float | None,
    cost_underestimated: bool,
) -> str:
    if actual_all_in is None or selected_cost is None:
        return "UNKNOWN"
    if cost_underestimated:
        return "WARN"
    return "PASS"


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row").select(
        [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    )


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None
