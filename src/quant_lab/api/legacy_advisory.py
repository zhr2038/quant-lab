"""Frozen historical advisory read contracts; no research producers or scheduler."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.symbols import normalize_symbol

PORTFOLIO_ADVISORY_OVERRIDE_STATUSES = {
    "KILL",
    "SHADOW",
    "DOWNGRADED_FROM_PAPER",
    "PAUSED",
    "BASELINE_ONLY",
}

KNOWN_RESEARCH_PORTFOLIO_IDS: dict[str, tuple[str, str]] = {
    "ETH_F3_DOMINANT_ENTRY_PAPER_V1": ("v5.f3_dominant_entry", "ETH-USDT"),
    "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1": (
        "v5.sol_protect_alpha6_low_exception",
        "SOL-USDT",
    ),
    "SOL_F4_VOLUME_EXPANSION_PAPER_V1": ("v5.f4_volume_expansion_entry", "SOL-USDT"),
}


def portfolio_status_overrides_by_identifier(
    research_portfolio: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, Any]]:
    overrides: dict[tuple[str, str], dict[str, Any]] = {}
    if research_portfolio.is_empty():
        return overrides
    frame = dedupe_research_portfolio_status(research_portfolio)
    rows = frame.to_dicts()
    latest_as_of_date = _latest_as_of_date(rows)
    if latest_as_of_date:
        rows = [
            row for row in rows if str(row.get("as_of_date") or "").strip() == latest_as_of_date
        ]
    for row in rows:
        status = str(row.get("status") or "").strip().upper()
        if status not in PORTFOLIO_ADVISORY_OVERRIDE_STATUSES:
            continue
        for key in _portfolio_override_keys(row):
            current = overrides.get(key)
            if current is None or _row_time(row) >= _row_time(current):
                overrides[key] = row
    return overrides


def portfolio_override_for_row(
    row: dict[str, Any],
    overrides: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    symbol = normalize_symbol(row.get("symbol") or row.get("v5_symbol")) or "UNKNOWN"
    identifiers = [
        row.get("strategy_id"),
        row.get("research_id"),
        row.get("strategy_candidate"),
        row.get("candidate_name"),
    ]
    for identifier in identifiers:
        override = portfolio_override_for_identifier_symbol(
            overrides,
            str(identifier or "").strip(),
            symbol,
        )
        if override is not None:
            return override
    return None


def portfolio_override_for_identifier_symbol(
    overrides: dict[tuple[str, str], dict[str, Any]],
    identifier: str,
    symbol: str,
) -> dict[str, Any] | None:
    if not identifier:
        return None
    normalized = normalize_symbol(symbol) or "UNKNOWN"
    return (
        overrides.get((identifier, normalized))
        or overrides.get((identifier, "UNKNOWN"))
        or overrides.get((identifier, ""))
    )


def portfolio_overridden_decision_mode(
    *,
    decision: str,
    recommended_mode: str,
    portfolio_override: dict[str, Any] | None,
) -> tuple[str, str, list[str]]:
    normalized_decision = str(decision or "RESEARCH_ONLY").strip().upper()
    normalized_mode = str(recommended_mode or "").strip().lower()
    if portfolio_override is None:
        return (
            normalized_decision,
            normalized_mode or advisory_recommended_mode(normalized_decision),
            [],
        )
    status = str(portfolio_override.get("status") or "").strip().upper()
    if status == "KILL":
        return "KILL", "none", ["research_portfolio_kill"]
    if status == "SHADOW":
        if normalized_decision in {"PAPER_READY", "LIVE_SMALL_READY"} or normalized_mode in {
            "paper",
            "live_small",
        }:
            return "KEEP_SHADOW", "shadow", ["research_portfolio_shadow"]
        return (
            normalized_decision,
            normalized_mode or advisory_recommended_mode(normalized_decision),
            [],
        )
    if status == "DOWNGRADED_FROM_PAPER":
        if normalized_decision in {"PAPER_READY", "LIVE_SMALL_READY"} or normalized_mode in {
            "paper",
            "live_small",
        }:
            return "KEEP_SHADOW", "shadow", ["downgraded_from_paper"]
        return (
            normalized_decision,
            normalized_mode or advisory_recommended_mode(normalized_decision),
            ["downgraded_from_paper"],
        )
    if status == "PAUSED":
        return "RESEARCH_ONLY", "research", ["research_paused"]
    if status == "BASELINE_ONLY":
        return "RESEARCH_ONLY", "research", ["baseline_only"]
    return (
        normalized_decision,
        normalized_mode or advisory_recommended_mode(normalized_decision),
        [],
    )


def advisory_recommended_mode(decision: str) -> str:
    normalized = str(decision or "RESEARCH_ONLY").strip().upper()
    if normalized == "KILL":
        return "none"
    if normalized == "PAPER_READY":
        return "paper"
    if normalized == "LIVE_SMALL_READY":
        return "live_small"
    if normalized in {"KEEP_SHADOW", "REGIME_SHADOW"}:
        return "shadow"
    return "research"


def _portfolio_override_keys(row: dict[str, Any]) -> set[tuple[str, str]]:
    candidate, symbol = _portfolio_candidate_symbol(row)
    research_id = str(row.get("research_id") or "").strip()
    strategy_id = str(row.get("strategy_id") or "").strip()
    identifiers = {value for value in [candidate, research_id, strategy_id] if value}
    symbols = {symbol}
    if symbol == "UNKNOWN":
        symbols.update({"", "UNKNOWN"})
    return {
        (identifier, candidate_symbol) for identifier in identifiers for candidate_symbol in symbols
    }


def _portfolio_candidate_symbol(row: dict[str, Any]) -> tuple[str, str]:
    candidate = str(row.get("strategy_candidate") or "").strip()
    symbol = normalize_symbol(row.get("symbol")) or "UNKNOWN"
    research_id = str(row.get("research_id") or "").strip().upper()
    if research_id in KNOWN_RESEARCH_PORTFOLIO_IDS:
        return KNOWN_RESEARCH_PORTFOLIO_IDS[research_id]
    return (candidate, symbol)


def _latest_as_of_date(rows: list[dict[str, Any]]) -> str:
    values = [
        str(row.get("as_of_date") or "").strip()
        for row in rows
        if str(row.get("as_of_date") or "").strip()
    ]
    return max(values) if values else ""


def _row_time(row: dict[str, Any]) -> datetime:
    value = row.get("created_at") or row.get("generated_at") or row.get("as_of_ts")
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.min.replace(tzinfo=UTC)


STATUS_SCHEMA: dict[str, Any] = {
    "schema_version": pl.Utf8,
    "as_of_date": pl.Utf8,
    "research_id": pl.Utf8,
    "module": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "status": pl.Utf8,
    "action": pl.Utf8,
    "reason": pl.Utf8,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "p25_net_bps": pl.Float64,
    "paper_days": pl.Int64,
    "entry_day_count": pl.Int64,
    "paper_negative_streak": pl.Int64,
    "downgrade_reason": pl.Utf8,
    "latest_paper_trend": pl.Utf8,
    "priority": pl.Utf8,
    "cost_source_mix": pl.Utf8,
    "last_review_date": pl.Utf8,
    "next_review_date": pl.Utf8,
    "recommended_new_research_slots": pl.Int64,
    "freed_research_slots": pl.Int64,
    "active_research_count": pl.Int64,
    "killed_research_count": pl.Int64,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def dedupe_research_portfolio_status(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=STATUS_SCHEMA)
    normalized = _normalize_status_frame(frame)
    normalized = normalized.sort(["as_of_date", "research_id", "created_at"])
    normalized = normalized.unique(
        subset=["as_of_date", "research_id"],
        keep="last",
        maintain_order=True,
    )
    return normalized.sort(["as_of_date", "research_id"], descending=[True, False])


def _normalize_status_frame(frame: pl.DataFrame) -> pl.DataFrame:
    normalized = frame
    if "as_of_date" not in normalized.columns:
        fallback = (
            pl.col("last_review_date").cast(pl.Utf8)
            if "last_review_date" in normalized.columns
            else pl.lit("")
        )
        normalized = normalized.with_columns(fallback.alias("as_of_date"))
    if "created_at" in normalized.columns:
        normalized = normalized.with_columns(
            pl.col("created_at")
            .cast(pl.Datetime(time_zone="UTC"), strict=False)
            .alias("created_at")
        )
    else:
        normalized = normalized.with_columns(
            pl.lit(datetime.min.replace(tzinfo=UTC), dtype=pl.Datetime(time_zone="UTC")).alias(
                "created_at"
            )
        )
    for column, dtype in STATUS_SCHEMA.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return normalized.select(list(STATUS_SCHEMA))


ALPHA_FACTORY_CANDIDATES = frozenset(
    [
        "v5.alt_basket_vs_btc_shadow",
        "v5.alt_impulse_shadow",
        "v5.btc_strict_probe_exit_policy_review",
        "v5.eth_f3_exit_policy_review",
        "v5.expanded_relative_strength_rotation_shadow",
        "v5.expanded_relative_strength_top1_shadow",
        "v5.expanded_relative_strength_top3_shadow",
        "v5.futures_downtrend_short_proxy_shadow",
        "v5.futures_risk_off_hedge_proxy_shadow",
        "v5.pair_trade_eth_btc_shadow",
        "v5.pair_trade_sol_eth_shadow",
        "v5.sol_paper_exit_policy_review",
    ]
)
BOOTSTRAP_GATE_VERSION = "bootstrap.quarantine.v1"
