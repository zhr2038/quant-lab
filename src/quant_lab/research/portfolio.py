from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps

RESEARCH_PORTFOLIO_STATUS_DATASET = Path("gold") / "research_portfolio_status"
SOURCE_NAME = "research.portfolio_pruning.v0.1"
SCHEMA_VERSION = "research_portfolio_status.v0.1"

CLOSED_RESEARCH_CANDIDATES: frozenset[str] = frozenset(
    {
        "v5.btc_broad_leadership",
        "v5.btc_leadership_blocked_relaxed",
        "v5.btc_alpha6_factor",
        "v5.btc_leadership_alpha6_low_blocked",
        "v5.btc_leadership_f5_low_blocked",
        "v5.btc_leadership_no_breakout_blocked",
        "v5.multi_position_k1",
        "v5.multi_position_k2",
        "v5.multi_position_k3",
        "v5.portfolio_trend_following",
        "v5.portfolio_trend_following_SOL",
        "v5.pullback_reversal_v1",
        "v5.pullback_reversal_shadow",
    }
)

CLOSED_RESEARCH_REASON = "research_closed_by_operator_after_negative_or_low_quality_evidence"

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


class ResearchPortfolioBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    rows_written: int = Field(ge=0)
    status_counts: dict[str, int] = Field(default_factory=dict)
    action_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_research_portfolio_status(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> ResearchPortfolioBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    frame = build_research_portfolio_status(root, as_of_date=day)
    dataset_path = root / RESEARCH_PORTFOLIO_STATUS_DATASET
    existing = _remove_as_of_date(read_parquet_dataset(dataset_path), day.isoformat())
    combined = pl.concat(
        [current for current in [existing, frame] if not current.is_empty()],
        how="diagonal_relaxed",
    )
    combined = dedupe_research_portfolio_status(combined)
    write_parquet_dataset(combined, dataset_path)
    rows_written = combined.height
    return ResearchPortfolioBuildResult(
        as_of_date=day.isoformat(),
        rows_written=rows_written,
        status_counts=_counts(frame, "status"),
        action_counts=_counts(frame, "action"),
        warnings=[] if rows_written else ["research_portfolio_status_empty"],
    )


def build_research_portfolio_status(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> pl.DataFrame:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    created_at = datetime.now(UTC)

    strategy_evidence = read_parquet_dataset(root / "gold" / "strategy_evidence")
    alpha_board = read_parquet_dataset(root / "gold" / "alpha_discovery_board")
    paper_daily = read_parquet_dataset(root / "gold" / "paper_strategy_daily")
    missed_low = read_parquet_dataset(root / "gold" / "v5_missed_low_audit")
    late_chase = read_parquet_dataset(root / "gold" / "v5_late_entry_chase_shadow")
    pullback = read_parquet_dataset(root / "gold" / "v5_pullback_reversal_shadow")
    expanded = read_parquet_dataset(root / "gold" / "expanded_crypto_universe_shadow")

    evidence_rows = [
        *(_rows(strategy_evidence)),
        *(_rows(alpha_board)),
    ]
    paper_rows = _rows(paper_daily)
    eth_f3_status, eth_f3_action, eth_f3_reason = _eth_f3_portfolio_state(paper_rows)

    rows = [
        _status_row(
            research_id="v5.core.momentum",
            module="baseline",
            strategy_candidate="v5.core.momentum",
            status="BASELINE_ONLY",
            action="TRACK_AS_RESEARCH_BASELINE",
            reason="generic_momentum_baseline_not_global_strategy_gate",
            metrics=_empty_metrics(),
            day=day,
            created_at=created_at,
        ),
        _status_row(
            research_id="v5.entry_quality_missed_low_audit",
            module="entry_quality",
            strategy_candidate="v5.entry_quality_missed_low_audit",
            status="ACTIVE" if not missed_low.is_empty() else "PAUSED",
            action="ACTIVE_DIAGNOSTIC" if not missed_low.is_empty() else "PAUSED_TO_WEEKLY",
            reason=(
                "diagnostic_audit_has_recent_open_long_samples"
                if not missed_low.is_empty()
                else "no_recent_open_long_samples"
            ),
            metrics=_frame_metrics(missed_low, fallback_sample_count=missed_low.height),
            day=day,
            created_at=created_at,
            review_days=1 if not missed_low.is_empty() else 7,
        ),
        _status_row(
            research_id="v5.late_entry_chase_guard_shadow",
            module="entry_quality",
            strategy_candidate="v5.late_entry_chase_guard_shadow",
            status="SHADOW",
            action="CONTINUE_SHADOW",
            reason="symbol_thresholds_are_research_only_no_hard_guard",
            metrics=_frame_metrics(late_chase, fallback_sample_count=late_chase.height),
            day=day,
            created_at=created_at,
        ),
        _status_row(
            research_id="v5.pullback_reversal_v1",
            module="entry_quality",
            strategy_candidate="v5.pullback_reversal_shadow",
            status="KILL",
            action="CLOSED_RESEARCH_V1",
            reason="historical_pullback_reversal_v1_negative_expectancy_and_large_mae",
            metrics=_frame_metrics(pullback, fallback_sample_count=pullback.height),
            day=day,
            created_at=created_at,
            review_days=7,
        ),
        _candidate_row(
            research_id="ETH_F3_DOMINANT_ENTRY_PAPER_V1",
            module="paper_strategy",
            candidate="v5.f3_dominant_entry",
            status=eth_f3_status,
            action=eth_f3_action,
            reason=eth_f3_reason,
            evidence=evidence_rows,
            paper=paper_rows,
            day=day,
            created_at=created_at,
            symbol="ETH-USDT",
        ),
        _candidate_row(
            research_id="SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            module="paper_strategy",
            candidate="v5.sol_protect_alpha6_low_exception",
            status="PAPER",
            action="CONTINUE_PAPER",
            reason="sol_protect_exception_is_paper_ready_but_not_live_validated",
            evidence=evidence_rows,
            paper=paper_rows,
            day=day,
            created_at=created_at,
            symbol="SOL-USDT",
        ),
        _candidate_row(
            research_id="SOL_F4_VOLUME_EXPANSION_PAPER_V1",
            module="paper_strategy",
            candidate="v5.f4_volume_expansion_entry",
            status="PAPER",
            action="CONTINUE_PAPER",
            reason="sol_f4_volume_expansion_is_paper_ready_but_not_live_validated",
            evidence=evidence_rows,
            paper=paper_rows,
            day=day,
            created_at=created_at,
            symbol="SOL-USDT",
        ),
        _candidate_row(
            research_id="v5.alt_impulse_shadow",
            module="regime_shadow",
            candidate="v5.alt_impulse_shadow",
            status="SHADOW",
            action="REGIME_SHADOW",
            reason="alt_impulse_is_regime_dependent_and_not_live_validated",
            evidence=evidence_rows,
            paper=paper_rows,
            day=day,
            created_at=created_at,
        ),
        _status_row(
            research_id="v5.expanded_crypto_universe_shadow",
            module="universe_research",
            strategy_candidate="v5.expanded_crypto_universe_shadow",
            status="SHADOW" if not expanded.is_empty() else "PAUSED",
            action="PAPER_RESEARCH" if not expanded.is_empty() else "WAIT_FOR_UNIVERSE_OUTPUT",
            reason=(
                "expanded_universe_shadow_collecting_replacement_candidates"
                if not expanded.is_empty()
                else "expanded_universe_shadow_missing"
            ),
            metrics=_frame_metrics(expanded, fallback_sample_count=expanded.height),
            day=day,
            created_at=created_at,
        ),
    ]
    rows.extend(_known_kill_rows(evidence_rows, paper_rows, day=day, created_at=created_at))
    rows.extend(_data_driven_rows(evidence_rows, paper_rows, day=day, created_at=created_at))

    rows = _dedupe_rows(rows)
    summary = _summary_counts(rows)
    rows = [{**row, **summary} for row in rows]
    return pl.DataFrame(rows, schema=STATUS_SCHEMA, orient="row")


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


def _remove_as_of_date(frame: pl.DataFrame, as_of_date: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    normalized = _normalize_status_frame(frame)
    return normalized.filter(pl.col("as_of_date") != as_of_date)


def research_portfolio_summary_md(
    frame: pl.DataFrame,
    *,
    as_of_date: str | date | None = None,
) -> str:
    day = (
        _parse_day(as_of_date).isoformat()
        if as_of_date is not None
        else _latest_as_of_date(frame)
    )
    current = _latest_day_frame(frame, day)
    lines = [
        f"# Research Portfolio Summary - {day or 'unknown'}",
        "",
        "This summary is read-only. It does not change V5 live configuration or risk permission.",
        "",
    ]
    sections = [
        (
            "CLOSE_RESEARCH",
            current.filter(
                (pl.col("status") == "KILL") | pl.col("action").str.starts_with("CLOSE")
            ),
        ),
        ("CONTINUE_PAPER", current.filter(pl.col("status") == "PAPER")),
        (
            "CONTINUE_SHADOW",
            current.filter((pl.col("status") == "SHADOW") | (pl.col("action") == "REGIME_SHADOW")),
        ),
        ("ACTIVE_DIAGNOSTIC", current.filter(pl.col("action") == "ACTIVE_DIAGNOSTIC")),
        ("BASELINE_ONLY", current.filter(pl.col("status") == "BASELINE_ONLY")),
    ]
    for title, section in sections:
        lines.extend([f"## {title}", ""])
        if section.is_empty():
            lines.extend(["- none", ""])
            continue
        for row in section.sort(["status", "research_id"]).to_dicts():
            lines.append(_summary_line(row, include_metrics=title == "CLOSE_RESEARCH"))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def is_closed_research_candidate(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text in CLOSED_RESEARCH_CANDIDATES


def _known_kill_rows(
    evidence: list[dict[str, Any]],
    paper: list[dict[str, Any]],
    *,
    day: date,
    created_at: datetime,
) -> list[dict[str, Any]]:
    items = [
        ("v5.btc_broad_leadership", "v5.btc_leadership_blocked_relaxed"),
        ("v5.btc_alpha6_factor", "v5.btc_leadership_alpha6_low_blocked"),
        ("v5.btc_leadership_f5_low", "v5.btc_leadership_f5_low_blocked"),
        ("v5.btc_leadership_no_breakout", "v5.btc_leadership_no_breakout_blocked"),
        ("v5.multi_position_k1", "v5.multi_position_k1"),
        ("v5.multi_position_k2", "v5.multi_position_k2"),
        ("v5.multi_position_k3", "v5.multi_position_k3"),
        ("v5.portfolio_trend_following", "v5.portfolio_trend_following"),
        ("v5.portfolio_trend_following_SOL", "v5.portfolio_trend_following"),
    ]
    return [
        _candidate_row(
            research_id=research_id,
            module="candidate_pruning",
            candidate=candidate,
            status="KILL",
            action="CLOSED_RESEARCH",
            reason=CLOSED_RESEARCH_REASON,
            evidence=evidence,
            paper=paper,
            day=day,
            created_at=created_at,
            symbol="SOL-USDT" if research_id.endswith("_SOL") else None,
            review_days=7,
        )
        for research_id, candidate in items
    ]


def _eth_f3_portfolio_state(paper: list[dict[str, Any]]) -> tuple[str, str, str]:
    rows = [
        row
        for row in paper
        if str(row.get("proposal_id") or "") == "ETH_USDT_F3_DOMINANT_ENTRY_PAPER_V1"
        or (
            str(row.get("strategy_candidate") or "") == "v5.f3_dominant_entry"
            and str(row.get("symbol") or "") == "ETH-USDT"
        )
    ]
    if not rows:
        return (
            "PAPER",
            "CONTINUE_PAPER",
            "eth_f3_wait_for_48h_paper_labels_no_live",
        )
    latest = max(rows, key=_portfolio_row_time)
    decision = str(latest.get("latest_board_decision") or "").strip().upper()
    reasons = " ".join(
        [
            str(latest.get("live_block_reason") or ""),
            str(latest.get("decision_reasons") or ""),
        ]
    )
    if decision == "KEEP_SHADOW" or "eth_f3_48h_paper_pnl_negative" in reasons:
        return (
            "SHADOW",
            "CONTINUE_SHADOW",
            "eth_f3_48h_negative_keep_shadow_no_live",
        )
    return (
        "PAPER",
        "CONTINUE_PAPER",
        "eth_f3_continue_paper_until_48h_sample_count_30_and_14_days_validated",
    )


def _portfolio_row_time(row: dict[str, Any]) -> datetime:
    for field in ["created_at", "as_of_ts", "as_of_date", "bundle_ts", "ingest_ts"]:
        parsed = _created_at_key(row.get(field))
        if parsed != datetime.min.replace(tzinfo=UTC):
            return parsed
    return datetime.min.replace(tzinfo=UTC)


def _data_driven_rows(
    evidence: list[dict[str, Any]],
    paper: list[dict[str, Any]],
    *,
    day: date,
    created_at: datetime,
) -> list[dict[str, Any]]:
    fixed = {
        "v5.core.momentum",
        "v5.entry_quality_missed_low_audit",
        "v5.late_entry_chase_guard_shadow",
        "v5.pullback_reversal_shadow",
        "v5.f3_dominant_entry",
        "v5.sol_protect_alpha6_low_exception",
        "v5.f4_volume_expansion_entry",
        "v5.alt_impulse_shadow",
        "v5.btc_leadership_blocked_relaxed",
        "v5.btc_leadership_alpha6_low_blocked",
        "v5.btc_leadership_f5_low_blocked",
        "v5.btc_leadership_no_breakout_blocked",
        "v5.multi_position_k1",
        "v5.multi_position_k2",
        "v5.multi_position_k3",
        "v5.portfolio_trend_following",
    }
    candidates = sorted(
        {
            str(row.get("strategy_candidate") or "")
            for row in evidence
            if str(row.get("strategy_candidate") or "")
            and row.get("strategy_candidate") not in fixed
            and not is_closed_research_candidate(row.get("strategy_candidate"))
        }
    )
    rows = []
    for candidate in candidates:
        metrics = _metrics_for(evidence, [candidate])
        status, action, reason = _status_from_metrics(metrics)
        rows.append(
            _status_row(
                research_id=candidate,
                module="candidate_research",
                strategy_candidate=candidate,
                status=status,
                action=action,
                reason=reason,
                metrics=_with_paper(metrics, paper, candidate),
                day=day,
                created_at=created_at,
                review_days=7 if status in {"KILL", "PAUSED"} else 1,
            )
        )
    return rows


def _candidate_row(
    *,
    research_id: str,
    module: str,
    candidate: str,
    status: str,
    action: str,
    reason: str,
    evidence: list[dict[str, Any]],
    paper: list[dict[str, Any]],
    day: date,
    created_at: datetime,
    symbol: str | None = None,
    review_days: int = 1,
) -> dict[str, Any]:
    metrics = _metrics_for(evidence, [candidate], symbol=symbol)
    return _status_row(
        research_id=research_id,
        module=module,
        strategy_candidate=candidate,
        status=status,
        action=action,
        reason=reason,
        metrics=_with_paper(metrics, paper, candidate, research_id=research_id),
        day=day,
        created_at=created_at,
        review_days=review_days,
    )


def _status_row(
    *,
    research_id: str,
    module: str,
    strategy_candidate: str,
    status: str,
    action: str,
    reason: str,
    metrics: dict[str, Any],
    day: date,
    created_at: datetime,
    review_days: int = 1,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of_date": day.isoformat(),
        "research_id": research_id,
        "module": module,
        "strategy_candidate": strategy_candidate,
        "status": status,
        "action": action,
        "reason": reason,
        "sample_count": int(metrics.get("sample_count") or 0),
        "complete_sample_count": int(metrics.get("complete_sample_count") or 0),
        "avg_net_bps": _float_or_none(metrics.get("avg_net_bps")),
        "win_rate": _float_or_none(metrics.get("win_rate")),
        "p25_net_bps": _float_or_none(metrics.get("p25_net_bps")),
        "paper_days": int(metrics.get("paper_days") or 0),
        "entry_day_count": int(metrics.get("entry_day_count") or 0),
        "cost_source_mix": str(metrics.get("cost_source_mix") or "{}"),
        "last_review_date": day.isoformat(),
        "next_review_date": (day + timedelta(days=review_days)).isoformat(),
        "recommended_new_research_slots": 0,
        "freed_research_slots": 0,
        "active_research_count": 0,
        "killed_research_count": 0,
        "created_at": created_at,
        "source": SOURCE_NAME,
    }


def _metrics_for(
    rows: list[dict[str, Any]],
    candidates: list[str],
    *,
    symbol: str | None = None,
) -> dict[str, Any]:
    selected = []
    normalized_candidates = {candidate.lower() for candidate in candidates}
    for row in rows:
        candidate = str(row.get("strategy_candidate") or "").lower()
        if candidate not in normalized_candidates:
            continue
        if symbol and str(row.get("symbol") or "").upper() != symbol.upper():
            continue
        selected.append(row)
    if not selected:
        return _empty_metrics()

    sample_count = max(_int(row.get("sample_count")) for row in selected)
    complete_count = max(_int(row.get("complete_sample_count")) for row in selected)
    avg_net = _mean(row.get("avg_net_bps") for row in selected)
    win_rate = _mean(row.get("win_rate") for row in selected)
    p25 = _min_observed(row.get("p25_net_bps") for row in selected)
    cost_mix = _combine_cost_sources(row.get("cost_source_mix") for row in selected)
    return {
        "sample_count": sample_count,
        "complete_sample_count": complete_count,
        "avg_net_bps": avg_net,
        "win_rate": win_rate,
        "p25_net_bps": p25,
        "cost_source_mix": safe_json_dumps(cost_mix),
    }


def _frame_metrics(frame: pl.DataFrame, *, fallback_sample_count: int = 0) -> dict[str, Any]:
    if frame.is_empty():
        return _empty_metrics()
    rows = _rows(frame)
    sample_counts = [_int(row.get("sample_count")) for row in rows]
    return {
        "sample_count": max([*sample_counts, fallback_sample_count]),
        "complete_sample_count": max(
            [_int(row.get("complete_sample_count")) for row in rows] + [0]
        ),
        "avg_net_bps": _mean(row.get("avg_net_bps") for row in rows),
        "win_rate": _mean(row.get("win_rate") for row in rows),
        "p25_net_bps": _min_observed(row.get("p25_net_bps") for row in rows),
        "cost_source_mix": safe_json_dumps(
            _combine_cost_sources(row.get("cost_source_mix") for row in rows)
        ),
    }


def _with_paper(
    metrics: dict[str, Any],
    paper_rows: list[dict[str, Any]],
    candidate: str,
    *,
    research_id: str | None = None,
) -> dict[str, Any]:
    result = dict(metrics)
    keys = {candidate.lower()}
    if research_id:
        keys.add(research_id.lower())
    selected = [
        row
        for row in paper_rows
        if str(row.get("strategy_candidate") or "").lower() in keys
        or str(row.get("proposal_id") or "").lower() in keys
        or str(row.get("strategy_id") or "").lower() in keys
    ]
    if selected:
        result["paper_days"] = max(_int(row.get("paper_days")) for row in selected)
        result["entry_day_count"] = max(_int(row.get("entry_day_count")) for row in selected)
    return result


def _status_from_metrics(metrics: dict[str, Any]) -> tuple[str, str, str]:
    if _kill_condition(metrics):
        return (
            "KILL",
            "CLOSE_RESEARCH",
            "complete_samples_sufficient_and_negative_after_cost_edge",
        )
    if (
        int(metrics.get("sample_count") or 0) < 10
        or int(metrics.get("complete_sample_count") or 0) < 5
    ):
        return "PAUSED", "WAIT_FOR_MORE_SAMPLES", "insufficient_samples"
    cost_mix = str(metrics.get("cost_source_mix") or "")
    if "global_default" in cost_mix or "cost_not_requested_no_order" in cost_mix:
        return "PAUSED", "IMPROVE_COST_QUALITY", "cost_source_not_trusted"
    if _float(metrics.get("avg_net_bps")) and (_float(metrics.get("avg_net_bps")) or 0.0) > 0:
        return "SHADOW", "CONTINUE_SHADOW", "positive_edge_needs_more_validation"
    return "PAUSED", "REVIEW_WEEKLY", "no_clear_positive_edge"


def _kill_condition(metrics: dict[str, Any]) -> bool:
    return (
        int(metrics.get("complete_sample_count") or 0) >= 30
        and (_float(metrics.get("avg_net_bps")) or 0.0) < 0
        and (_float(metrics.get("win_rate")) or 0.0) < 0.45
        and (_float(metrics.get("p25_net_bps")) or 0.0) < -50
    )


def _summary_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    active_count = sum(1 for row in rows if row["status"] in {"ACTIVE", "SHADOW", "PAPER"})
    killed_count = sum(1 for row in rows if row["status"] == "KILL")
    freed_slots = sum(1 for row in rows if row["status"] in {"KILL", "PAUSED"})
    return {
        "recommended_new_research_slots": freed_slots,
        "freed_research_slots": freed_slots,
        "active_research_count": active_count,
        "killed_research_count": killed_count,
    }


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("as_of_date") or row.get("last_review_date") or ""),
            str(row["research_id"]),
        )
        previous = by_id.get(key)
        if previous is None or _created_at_key(row.get("created_at")) >= _created_at_key(
            previous.get("created_at")
        ):
            by_id[key] = row
    return [by_id[key] for key in sorted(by_id)]


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


def _latest_as_of_date(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return ""
    normalized = _normalize_status_frame(frame)
    dates = [
        str(value)
        for value in normalized.get_column("as_of_date").drop_nulls().to_list()
        if str(value)
    ]
    return max(dates) if dates else ""


def _latest_day_frame(frame: pl.DataFrame, day: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=STATUS_SCHEMA)
    normalized = dedupe_research_portfolio_status(frame)
    if not day:
        day = _latest_as_of_date(normalized)
    if day:
        normalized = normalized.filter(pl.col("as_of_date") == day)
    return normalized


def _summary_line(row: dict[str, Any], *, include_metrics: bool) -> str:
    base = (
        f"- {row.get('research_id')}: {row.get('status')} / {row.get('action')} - "
        f"{row.get('reason')}"
    )
    if not include_metrics:
        return base
    metrics = (
        f"sample={_int(row.get('sample_count'))}, "
        f"complete={_int(row.get('complete_sample_count'))}, "
        f"avg_net_bps={_display_metric(row.get('avg_net_bps'))}, "
        f"win_rate={_display_metric(row.get('win_rate'))}, "
        f"p25_net_bps={_display_metric(row.get('p25_net_bps'))}, "
        f"cost_source_mix={row.get('cost_source_mix') or '{}'}"
    )
    return f"{base}; {metrics}"


def _display_metric(value: Any) -> str:
    parsed = _float(value)
    if parsed is None:
        return "n/a"
    return f"{parsed:.4g}"


def _created_at_key(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return datetime.min.replace(tzinfo=UTC)
    return datetime.min.replace(tzinfo=UTC)


def _combine_cost_sources(values: Any) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for value in values:
        if value is None:
            continue
        if isinstance(value, dict):
            for key, count in value.items():
                counter[str(key)] += _int(count) or 1
            continue
        text = str(value).strip()
        if not text:
            continue
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError:
            loaded = text
        if isinstance(loaded, dict):
            for key, count in loaded.items():
                counter[str(key)] += _int(count) or 1
        elif isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict):
                    key = item.get("cost_source") or item.get("source")
                    if key:
                        counter[str(key)] += _int(item.get("count")) or 1
                elif item:
                    counter[str(item)] += 1
        elif loaded:
            counter[str(loaded)] += 1
    return dict(sorted(counter.items()))


def _empty_metrics() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "complete_sample_count": 0,
        "avg_net_bps": None,
        "win_rate": None,
        "p25_net_bps": None,
        "paper_days": 0,
        "entry_day_count": 0,
        "cost_source_mix": "{}",
    }


def _rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return [] if frame.is_empty() else frame.to_dicts()


def _counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    if frame.is_empty() or column not in frame.columns:
        return {}
    return {
        str(row[column]): int(row["count"])
        for row in frame.group_by(column).len(name="count").sort(column).to_dicts()
    }


def _mean(values: Any) -> float | None:
    observed = [_float(value) for value in values]
    clean = [value for value in observed if value is not None]
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def _min_observed(values: Any) -> float | None:
    clean = [value for value in (_float(value) for value in values) if value is not None]
    return min(clean) if clean else None


def _float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    return _float(value)


def _int(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _parse_day(value: str | date | None) -> date:
    if value is None:
        return datetime.now(UTC).date()
    if isinstance(value, date):
        return value
    if str(value).lower() == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(str(value))
