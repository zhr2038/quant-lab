"""Immutable forward-paper accounting for Alpha Audit v2.1.

The module contains only deterministic research accounting.  It does not know
about exchange credentials, order APIs, production positions, or live gates.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

RUNNER_VERSION = "quant_lab_low_vol_forward.v2.1"
STRATEGY_ID = "low_vol_btc_trend_top3_score_120h_v21"
HOLDING_HOURS = 120
ENTRY_PRICE_RULE = "next_bar_close"
EXIT_PRICE_RULE = "scheduled_exit_or_next_available_close"
ENTRY_FEE_BPS = 10.0
ENTRY_SLIPPAGE_BPS = 5.0
EXIT_FEE_BPS = 10.0
EXIT_SLIPPAGE_BPS = 5.0

DECISION_SCHEMA = {
    "decision_id": pl.Utf8,
    "strategy_id": pl.Utf8,
    "decision_ts": pl.Datetime("us", "UTC"),
    "feature_cutoff_ts": pl.Datetime("us", "UTC"),
    "entry_ts": pl.Datetime("us", "UTC"),
    "available_data_cutoff": pl.Datetime("us", "UTC"),
    "btc_trend_state": pl.Utf8,
    "universe": pl.Utf8,
    "factor_values": pl.Utf8,
    "ranked_symbols": pl.Utf8,
    "selected_symbols": pl.Utf8,
    "rejected_symbols": pl.Utf8,
    "target_weights": pl.Utf8,
    "cash_weight": pl.Float64,
    "invested_weight": pl.Float64,
    "decision_status": pl.Utf8,
    "decision_delay_bars": pl.Int64,
    "entry_price_rule": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "code_commit": pl.Utf8,
    "snapshot_id": pl.Utf8,
    "hypothesis_type": pl.Utf8,
    "parameters_locked": pl.Boolean,
    "runner_version": pl.Utf8,
}

TRADE_SCHEMA = {
    "trade_id": pl.Utf8,
    "decision_id": pl.Utf8,
    "symbol": pl.Utf8,
    "target_weight": pl.Float64,
    "entry_ts": pl.Datetime("us", "UTC"),
    "entry_price": pl.Float64,
    "scheduled_exit_ts": pl.Datetime("us", "UTC"),
    "actual_exit_ts": pl.Datetime("us", "UTC"),
    "exit_price": pl.Float64,
    "exit_delay_bars": pl.Int64,
    "mark_ts": pl.Datetime("us", "UTC"),
    "mark_price": pl.Float64,
    "entry_fee_bps": pl.Float64,
    "entry_slippage_bps": pl.Float64,
    "exit_fee_bps": pl.Float64,
    "exit_slippage_bps": pl.Float64,
    "gross_return": pl.Float64,
    "net_return": pl.Float64,
    "unrealized_gross_return": pl.Float64,
    "unrealized_net_return": pl.Float64,
    "weighted_gross_contribution": pl.Float64,
    "weighted_net_contribution": pl.Float64,
    "status": pl.Utf8,
    "invalidation_reason": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "code_commit": pl.Utf8,
    "snapshot_id": pl.Utf8,
    "runner_version": pl.Utf8,
}

EVENT_SCHEMA = {
    "event_id": pl.Utf8,
    "event_type": pl.Utf8,
    "event_ts": pl.Datetime("us", "UTC"),
    "decision_id": pl.Utf8,
    "trade_id": pl.Utf8,
    "payload_sha256": pl.Utf8,
    "payload_json": pl.Utf8,
    "parameter_lock_hash": pl.Utf8,
    "code_commit": pl.Utf8,
    "runner_version": pl.Utf8,
    "immutable": pl.Boolean,
}

EQUITY_SCHEMA = {
    "timestamp": pl.Datetime("us", "UTC"),
    "strategy_equity": pl.Float64,
    "btc_equity": pl.Float64,
    "equal_weight_universe_equity": pl.Float64,
    "cash_equity": pl.Float64,
    "drawdown": pl.Float64,
    "gross_exposure": pl.Float64,
    "cash_weight": pl.Float64,
    "open_trade_count": pl.Int64,
}

IMMUTABLE_DECISION_FIELDS = tuple(DECISION_SCHEMA)
IMMUTABLE_TRADE_FIELDS = (
    "trade_id",
    "decision_id",
    "symbol",
    "target_weight",
    "entry_ts",
    "entry_price",
    "scheduled_exit_ts",
    "entry_fee_bps",
    "entry_slippage_bps",
    "exit_fee_bps",
    "exit_slippage_bps",
    "parameter_lock_hash",
    "code_commit",
    "snapshot_id",
    "runner_version",
)


def empty_frame(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=dict(schema))


def utc(value: datetime | str) -> datetime:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest_payload(value: Any) -> str:
    return sha256_text(canonical_json(value))


def parameter_lock_digest(lock: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in lock.items() if key != "sha256"}
    return digest_payload(payload)


def validate_parameter_lock(lock: Mapping[str, Any]) -> None:
    expected = {
        "strategy_id": STRATEGY_ID,
        "hypothesis_type": "POST_HOC_HYPOTHESIS",
        "parameters_locked": True,
        "factor": "low_vol_20d",
        "factor_lookback_hours": 480,
        "top_n": 3,
        "weighting": "score",
        "rebalance_hours": HOLDING_HOURS,
        "holding_hours": HOLDING_HOURS,
        "rebalance_anchor_rule": "first_full_hour_strictly_after_cutoff",
        "btc_trend_filter": True,
        "btc_trend_lookback_hours": 1440,
        "decision_delay_bars": 1,
        "entry_price_rule": ENTRY_PRICE_RULE,
        "exit_price_rule": EXIT_PRICE_RULE,
        "round_trip_cost_bps": 30.0,
        "runner_version": RUNNER_VERSION,
    }
    mismatches = {
        key: {"expected": value, "actual": lock.get(key)}
        for key, value in expected.items()
        if lock.get(key) != value
    }
    if mismatches:
        raise ValueError(f"parameter lock violates frozen hypothesis: {mismatches}")
    if lock.get("sha256") != parameter_lock_digest(lock):
        raise ValueError("parameter lock sha256 mismatch")


def stable_decision_id(
    strategy_id: str,
    decision_ts: datetime,
    snapshot_id: str,
    parameter_lock_hash: str,
) -> str:
    return sha256_text(
        "|".join(
            [strategy_id, utc(decision_ts).isoformat(), snapshot_id, parameter_lock_hash]
        )
    )


def stable_trade_id(decision_id: str, symbol: str, entry_ts: datetime) -> str:
    return sha256_text(
        "|".join([decision_id, str(symbol), utc(entry_ts).isoformat()])
    )


def cost_adjusted_return(
    gross_return: float, entry_cost_bps: float, exit_cost_bps: float
) -> float:
    return (
        (1.0 - float(entry_cost_bps) / 10_000.0)
        * (1.0 + float(gross_return))
        * (1.0 - float(exit_cost_bps) / 10_000.0)
        - 1.0
    )


def entry_mark_return(gross_return: float, entry_cost_bps: float) -> float:
    return (1.0 - float(entry_cost_bps) / 10_000.0) * (
        1.0 + float(gross_return)
    ) - 1.0


def compounded_return(period_returns: Iterable[float]) -> float:
    equity = 1.0
    for value in period_returns:
        equity *= 1.0 + float(value)
    return equity - 1.0


def arithmetic_sum_return(period_returns: Iterable[float]) -> float:
    return float(sum(float(value) for value in period_returns))


def forward_available_days(
    forward_start_cutoff: datetime, available_market_data_cutoff: datetime
) -> float:
    return max(
        0.0,
        (utc(available_market_data_cutoff) - utc(forward_start_cutoff)).total_seconds()
        / 86400.0,
    )


def validate_new_cutoff(
    forward_v21_start_cutoff: datetime, runner_fix_completed_at: datetime
) -> None:
    if utc(forward_v21_start_cutoff) < utc(runner_fix_completed_at):
        raise ValueError("forward v2.1 cutoff predates the completed runner fix")


def select_next_bar_close(
    bars: pl.DataFrame, symbol: str, feature_cutoff_ts: datetime
) -> tuple[datetime, float] | None:
    """Return exactly the next completed 1h close; never use the signal bar."""
    expected = utc(feature_cutoff_ts) + timedelta(hours=1)
    rows = bars.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("ts") == expected)
        & pl.col("close").is_finite()
    )
    if rows.is_empty():
        return None
    row = rows.sort("ts").head(1).to_dicts()[0]
    return utc(row["ts"]), float(row["close"])


def performance_counts(
    decisions: pl.DataFrame, trades: pl.DataFrame
) -> dict[str, int]:
    decision_count = decisions.height
    cash_decisions = (
        decisions.filter(pl.col("decision_status") == "CASH").height
        if not decisions.is_empty()
        else 0
    )
    entry_count = trades.height
    open_count = (
        trades.filter(pl.col("status") == "OPEN").height
        if not trades.is_empty()
        else 0
    )
    closed_count = (
        trades.filter(pl.col("status") == "CLOSED").height
        if not trades.is_empty()
        else 0
    )
    closed_periods = 0
    if closed_count:
        closed_periods = trades.group_by("decision_id").agg(
            pl.col("status").eq("CLOSED").all().alias("all_closed")
        ).filter(pl.col("all_closed")).height
    return {
        "decision_count": decision_count,
        "cash_decision_count": cash_decisions,
        "entry_count": entry_count,
        "open_trade_count": open_count,
        "closed_trade_count": closed_count,
        "completed_independent_period_count": int(closed_periods),
    }


def forward_review_status(
    *,
    available_days: float,
    completed_periods: int,
    entry_count: int,
    data_coverage: float,
    runner_error_count: int,
    unhandled_delay_count: int,
    strategy_net_return: float,
    excess_vs_btc: float,
    excess_vs_universe: float,
    drawdown_within_lock: bool,
    concentration_within_lock: bool,
) -> str:
    ready = all(
        (
            available_days >= 30.0,
            completed_periods >= 6,
            entry_count >= 12,
            data_coverage >= 0.95,
            runner_error_count == 0,
            unhandled_delay_count == 0,
            strategy_net_return > 0.0,
            excess_vs_btc > 0.0,
            excess_vs_universe > 0.0,
            drawdown_within_lock,
            concentration_within_lock,
        )
    )
    return "PAPER_REVIEW_READY" if ready else "INCONCLUSIVE_FORWARD_SAMPLE_INSUFFICIENT"


def _normalize_compare(value: Any) -> Any:
    if isinstance(value, datetime):
        return utc(value).isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _assert_same_fields(
    old: Mapping[str, Any], new: Mapping[str, Any], fields: Sequence[str], label: str
) -> None:
    changed = [
        field
        for field in fields
        if _normalize_compare(old.get(field)) != _normalize_compare(new.get(field))
    ]
    if changed:
        raise ValueError(f"immutable {label} fields changed: {changed}")


def merge_immutable_decisions(
    existing: pl.DataFrame, new: pl.DataFrame
) -> pl.DataFrame:
    if existing.is_empty():
        return new.sort("decision_ts") if not new.is_empty() else empty_frame(DECISION_SCHEMA)
    if new.is_empty():
        return existing.sort("decision_ts")
    rows = {row["decision_id"]: row for row in existing.iter_rows(named=True)}
    for row in new.iter_rows(named=True):
        prior = rows.get(row["decision_id"])
        if prior is not None:
            _assert_same_fields(prior, row, IMMUTABLE_DECISION_FIELDS, "decision")
        else:
            rows[row["decision_id"]] = row
    return pl.DataFrame(list(rows.values()), schema=DECISION_SCHEMA).sort("decision_ts")


def merge_trade_states(existing: pl.DataFrame, updates: pl.DataFrame) -> pl.DataFrame:
    if existing.is_empty():
        return (
            updates.sort(["entry_ts", "symbol"])
            if not updates.is_empty()
            else empty_frame(TRADE_SCHEMA)
        )
    if updates.is_empty():
        return existing.sort(["entry_ts", "symbol"])
    rows = {row["trade_id"]: row for row in existing.iter_rows(named=True)}
    for row in updates.iter_rows(named=True):
        prior = rows.get(row["trade_id"])
        if prior is None:
            rows[row["trade_id"]] = row
            continue
        _assert_same_fields(prior, row, IMMUTABLE_TRADE_FIELDS, "trade")
        if prior.get("status") == "CLOSED":
            _assert_same_fields(prior, row, tuple(TRADE_SCHEMA), "closed trade")
            continue
        if prior.get("status") == "INVALIDATED":
            _assert_same_fields(prior, row, tuple(TRADE_SCHEMA), "invalidated trade")
            continue
        rows[row["trade_id"]] = row
    return pl.DataFrame(list(rows.values()), schema=TRADE_SCHEMA).sort(
        ["entry_ts", "symbol"]
    )


def make_event(
    *,
    event_type: str,
    event_ts: datetime,
    decision_id: str,
    trade_id: str = "",
    payload: Mapping[str, Any] | None = None,
    parameter_lock_hash: str,
    code_commit: str,
) -> dict[str, Any]:
    normalized_payload = dict(payload or {})
    payload_json = canonical_json(normalized_payload)
    payload_sha256 = sha256_text(payload_json)
    timestamp = utc(event_ts)
    event_id = sha256_text(
        "|".join(
            [
                event_type,
                decision_id,
                trade_id,
                timestamp.isoformat(),
                payload_sha256,
                parameter_lock_hash,
            ]
        )
    )
    return {
        "event_id": event_id,
        "event_type": event_type,
        "event_ts": timestamp,
        "decision_id": decision_id,
        "trade_id": trade_id,
        "payload_sha256": payload_sha256,
        "payload_json": payload_json,
        "parameter_lock_hash": parameter_lock_hash,
        "code_commit": code_commit,
        "runner_version": RUNNER_VERSION,
        "immutable": True,
    }


def append_only_events(existing: pl.DataFrame, new: pl.DataFrame) -> pl.DataFrame:
    if existing.is_empty():
        return (
            new.sort(["event_ts", "event_id"])
            if not new.is_empty()
            else empty_frame(EVENT_SCHEMA)
        )
    if new.is_empty():
        return existing.sort(["event_ts", "event_id"])
    rows = {row["event_id"]: row for row in existing.iter_rows(named=True)}
    for row in new.iter_rows(named=True):
        prior = rows.get(row["event_id"])
        if prior is not None:
            _assert_same_fields(prior, row, tuple(EVENT_SCHEMA), "event")
        else:
            rows[row["event_id"]] = row
    return pl.DataFrame(list(rows.values()), schema=EVENT_SCHEMA).sort(
        ["event_ts", "event_id"]
    )


def _symbol_bars(
    bars: pl.DataFrame, symbol: str, available_cutoff: datetime
) -> pl.DataFrame:
    return bars.filter(
        (pl.col("symbol") == symbol)
        & (pl.col("ts") <= utc(available_cutoff))
        & pl.col("close").is_finite()
    ).sort("ts")


def resolve_trade_state(
    trade: Mapping[str, Any], bars: pl.DataFrame, available_cutoff: datetime
) -> dict[str, Any]:
    """Mark or close one trade without ever repricing an existing close."""
    current = dict(trade)
    if current.get("status") in {"CLOSED", "INVALIDATED"}:
        return current
    available = utc(available_cutoff)
    symbol_bars = _symbol_bars(bars, str(current["symbol"]), available)
    after_entry = symbol_bars.filter(pl.col("ts") >= utc(current["entry_ts"]))
    if after_entry.is_empty():
        return current
    mark = after_entry.tail(1).to_dicts()[0]
    entry_price = float(current["entry_price"])
    entry_cost = float(current["entry_fee_bps"]) + float(
        current["entry_slippage_bps"]
    )
    exit_cost = float(current["exit_fee_bps"]) + float(
        current["exit_slippage_bps"]
    )
    scheduled_exit = utc(current["scheduled_exit_ts"])
    current.update(
        {
            "mark_ts": mark["ts"],
            "mark_price": float(mark["close"]),
            "unrealized_gross_return": float(mark["close"]) / entry_price - 1.0,
        }
    )
    current["unrealized_net_return"] = entry_mark_return(
        current["unrealized_gross_return"], entry_cost
    )
    if available < scheduled_exit:
        current.update(
            {
                "actual_exit_ts": None,
                "exit_price": None,
                "exit_delay_bars": None,
                "gross_return": None,
                "net_return": None,
                "weighted_gross_contribution": None,
                "weighted_net_contribution": None,
                "status": "OPEN",
            }
        )
        return current
    exit_rows = after_entry.filter(pl.col("ts") >= scheduled_exit)
    if exit_rows.is_empty():
        current["status"] = "OPEN"
        return current
    exit_row = exit_rows.head(1).to_dicts()[0]
    gross = float(exit_row["close"]) / entry_price - 1.0
    net = cost_adjusted_return(gross, entry_cost, exit_cost)
    weight = float(current["target_weight"])
    current.update(
        {
            "actual_exit_ts": exit_row["ts"],
            "exit_price": float(exit_row["close"]),
            "exit_delay_bars": int(
                round((exit_row["ts"] - scheduled_exit).total_seconds() / 3600.0)
            ),
            "mark_ts": exit_row["ts"],
            "mark_price": float(exit_row["close"]),
            "gross_return": gross,
            "net_return": net,
            "unrealized_gross_return": None,
            "unrealized_net_return": None,
            "weighted_gross_contribution": weight * gross,
            "weighted_net_contribution": weight * net,
            "status": "CLOSED",
        }
    )
    return current


def portfolio_period_returns(
    trades: Sequence[Mapping[str, Any]], cash_weight: float, *, realized: bool
) -> dict[str, float]:
    gross_factor = float(cash_weight)
    net_factor = float(cash_weight)
    entry_cost = 0.0
    exit_cost = 0.0
    for trade in trades:
        weight = float(trade["target_weight"])
        if realized:
            if trade.get("status") != "CLOSED" or trade.get("gross_return") is None:
                raise ValueError("realized period contains an open trade")
            gross = float(trade["gross_return"])
            net = float(trade["net_return"])
            exit_cost += weight * (
                float(trade["exit_fee_bps"]) + float(trade["exit_slippage_bps"])
            ) / 10_000.0
        else:
            if trade.get("unrealized_gross_return") is None:
                raise ValueError("open period has no mark")
            gross = float(trade["unrealized_gross_return"])
            net = float(trade["unrealized_net_return"])
        entry_cost += weight * (
            float(trade["entry_fee_bps"]) + float(trade["entry_slippage_bps"])
        ) / 10_000.0
        gross_factor += weight * (1.0 + gross)
        net_factor += weight * (1.0 + net)
    return {
        "gross_return": gross_factor - 1.0,
        "net_return": net_factor - 1.0,
        "entry_cost_return": entry_cost,
        "exit_cost_return": exit_cost,
    }


def _latest_price(
    bars_by_symbol: Mapping[str, list[tuple[datetime, float]]],
    symbol: str,
    timestamp: datetime,
    fallback: float,
) -> float:
    rows = bars_by_symbol.get(symbol, [])
    output = fallback
    for ts, price in rows:
        if ts > timestamp:
            break
        output = price
    return output


def _legs_by_decision(legs: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for row in legs.iter_rows(named=True):
        output.setdefault(str(row["decision_id"]), []).append(row)
    return output


def build_cohort_equity(
    decisions: pl.DataFrame,
    legs: pl.DataFrame,
    bars: pl.DataFrame,
    timeline: Sequence[datetime],
) -> list[dict[str, float | int | datetime]]:
    """Build non-overlapping 120h cohort equity with cash preserved."""
    if decisions.is_empty():
        return [
            {
                "timestamp": utc(ts),
                "equity": 1.0,
                "gross_exposure": 0.0,
                "cash_weight": 1.0,
                "open_trade_count": 0,
            }
            for ts in timeline
        ]
    ordered = decisions.sort("entry_ts").to_dicts()
    leg_groups = _legs_by_decision(legs)
    invested = [row for row in ordered if float(row.get("invested_weight", 0.0)) > 0]
    for left, right in zip(invested, invested[1:], strict=False):
        if utc(right["entry_ts"]) < utc(left["entry_ts"]) + timedelta(
            hours=HOLDING_HOURS
        ):
            raise ValueError("overlapping invested cohorts are not allowed by the lock")
    bars_by_symbol: dict[str, list[tuple[datetime, float]]] = {}
    for row in bars.select(["symbol", "ts", "close"]).sort(
        ["symbol", "ts"]
    ).iter_rows(named=True):
        bars_by_symbol.setdefault(str(row["symbol"]), []).append(
            (utc(row["ts"]), float(row["close"]))
        )
    output: list[dict[str, float | int | datetime]] = []
    for raw_ts in timeline:
        ts = utc(raw_ts)
        equity = 1.0
        active_exposure = 0.0
        active_cash = 1.0
        open_count = 0
        for decision in ordered:
            entry_ts = utc(decision["entry_ts"])
            if ts < entry_ts:
                break
            cohort_legs = leg_groups.get(str(decision["decision_id"]), [])
            if not cohort_legs:
                continue
            closed_by_ts = all(
                leg.get("status") == "CLOSED"
                and leg.get("actual_exit_ts") is not None
                and utc(leg["actual_exit_ts"]) <= ts
                for leg in cohort_legs
            )
            factor = float(decision["cash_weight"])
            if closed_by_ts:
                for leg in cohort_legs:
                    factor += float(leg["target_weight"]) * (
                        1.0 + float(leg["net_return"])
                    )
                equity *= factor
                continue
            open_weight = 0.0
            for leg in cohort_legs:
                if (
                    leg.get("status") == "CLOSED"
                    and leg.get("actual_exit_ts") is not None
                    and utc(leg["actual_exit_ts"]) <= ts
                ):
                    factor += float(leg["target_weight"]) * (
                        1.0 + float(leg["net_return"])
                    )
                    continue
                price = _latest_price(
                    bars_by_symbol,
                    str(leg["symbol"]),
                    ts,
                    float(leg["entry_price"]),
                )
                gross = price / float(leg["entry_price"]) - 1.0
                entry_cost = float(leg["entry_fee_bps"]) + float(
                    leg["entry_slippage_bps"]
                )
                marked_net = entry_mark_return(gross, entry_cost)
                factor += float(leg["target_weight"]) * (1.0 + marked_net)
                open_weight += float(leg["target_weight"])
            equity *= factor
            active_exposure = open_weight
            active_cash = max(0.0, 1.0 - open_weight)
            open_count = sum(
                1
                for leg in cohort_legs
                if not (
                    leg.get("status") == "CLOSED"
                    and leg.get("actual_exit_ts") is not None
                    and utc(leg["actual_exit_ts"]) <= ts
                )
            )
            break
        output.append(
            {
                "timestamp": ts,
                "equity": equity,
                "gross_exposure": active_exposure,
                "cash_weight": active_cash,
                "open_trade_count": open_count,
            }
        )
    return output


def atomic_write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.write_parquet(partial, compression="zstd")
    os.replace(partial, path)


def atomic_write_csv(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    frame.write_csv(partial)
    os.replace(partial, path)


def atomic_write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
