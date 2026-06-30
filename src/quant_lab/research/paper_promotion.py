from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

PAPER_STRATEGY_REGISTRY_DATASET = Path("gold") / "paper_strategy_registry"
PAPER_STRATEGY_PROMOTION_GATE_DATASET = Path("gold") / "paper_strategy_promotion_gate"
PAPER_STRATEGY_PROPOSAL_ACK_DATASET = Path("silver") / "v5_paper_strategy_proposal_ack"
PAPER_STRATEGY_RUNS_DATASET = Path("gold") / "paper_strategy_runs"
PAPER_STRATEGY_DAILY_DATASET = Path("gold") / "paper_strategy_daily"
STRATEGY_OPPORTUNITY_ADVISORY_DATASET = Path("gold") / "strategy_opportunity_advisory"

PAPER_PIPELINE_SOURCE = "research.paper_strategy_promotion.v0.1"
PAPER_PIPELINE_SCHEMA_VERSION = "paper_strategy_pipeline.v1"
MIN_PAPER_DAYS = 14
MIN_CLOSED_ENTRIES = 20
MIN_ENTRY_DAY_COUNT = 7
MIN_ARRIVAL_MID_COVERAGE = 0.8
MIN_P25_AFTER_COST_BPS = -30.0
TRUSTED_PAPER_COST_SOURCES = {"actual_fills", "mixed_actual_proxy", "bootstrap_cost_probe"}
BLOCKED_PAPER_COST_SOURCES = {
    "fallback_not_live_safe",
    "public_spread_proxy",
    "public_microstructure_proxy",
    "global_default",
    "local_estimate",
    "conservative_shadow_cost",
}

PAPER_STRATEGY_REGISTRY_SCHEMA = {
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "status": pl.Utf8,
    "paper_start_at": pl.Utf8,
    "rules_locked": pl.Boolean,
    "accepted": pl.Boolean,
    "reject_reason": pl.Utf8,
    "first_seen_at": pl.Utf8,
    "accepted_at": pl.Utf8,
    "expires_at": pl.Utf8,
    "source_pack_sha256": pl.Utf8,
    "source_v5_bundle_sha256": pl.Utf8,
    "paper_only": pl.Boolean,
    "max_live_notional_usdt": pl.Float64,
    "live_order_effect": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}

PAPER_STRATEGY_PROMOTION_GATE_SCHEMA = {
    "strategy_id": pl.Utf8,
    "strategy_version": pl.Utf8,
    "proposal_id": pl.Utf8,
    "proposal_hash": pl.Utf8,
    "paper_tracker_id": pl.Utf8,
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "lifecycle_state": pl.Utf8,
    "accepted": pl.Boolean,
    "paper_tracker_created": pl.Boolean,
    "paper_runs": pl.Int64,
    "paper_days": pl.Int64,
    "closed_entries": pl.Int64,
    "entry_day_count": pl.Int64,
    "arrival_mid_coverage": pl.Float64,
    "mean_after_cost_bps": pl.Float64,
    "median_after_cost_bps": pl.Float64,
    "p25_after_cost_bps": pl.Float64,
    "recent_7d_mean_bps": pl.Float64,
    "recent_7d_hit_rate": pl.Float64,
    "cost_source": pl.Utf8,
    "cost_trusted_for_paper": pl.Boolean,
    "backtest_paper_conflict": pl.Boolean,
    "paper_ready": pl.Boolean,
    "block_reason": pl.Utf8,
    "created_at": pl.Utf8,
    "source": pl.Utf8,
    "schema_version": pl.Utf8,
}


class PaperStrategyPipelineResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    paper_strategy_registry: int = Field(ge=0)
    paper_strategy_promotion_gate: int = Field(ge=0)


@dataclass(frozen=True)
class _StrategyKey:
    proposal_id: str
    paper_tracker_id: str
    strategy_candidate: str
    symbol: str


def build_and_publish_paper_strategy_pipeline(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> PaperStrategyPipelineResult:
    root = Path(lake_root)
    day = _resolve_as_of_date(as_of_date)
    frames = build_paper_strategy_pipeline_frames(
        proposals=read_parquet_dataset(root / STRATEGY_OPPORTUNITY_ADVISORY_DATASET),
        proposal_ack=read_parquet_dataset(root / PAPER_STRATEGY_PROPOSAL_ACK_DATASET),
        runs=read_parquet_dataset(root / PAPER_STRATEGY_RUNS_DATASET),
        daily=read_parquet_dataset(root / PAPER_STRATEGY_DAILY_DATASET),
    )
    registry = frames["paper_strategy_registry"]
    gate = frames["paper_strategy_promotion_gate"]
    write_parquet_dataset(registry, root / PAPER_STRATEGY_REGISTRY_DATASET)
    write_parquet_dataset(gate, root / PAPER_STRATEGY_PROMOTION_GATE_DATASET)
    return PaperStrategyPipelineResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        paper_strategy_registry=registry.height,
        paper_strategy_promotion_gate=gate.height,
    )


def build_paper_strategy_pipeline_frames(
    *,
    proposals: pl.DataFrame | None = None,
    proposal_ack: pl.DataFrame | None = None,
    runs: pl.DataFrame | None = None,
    daily: pl.DataFrame | None = None,
    created_at: datetime | None = None,
) -> dict[str, pl.DataFrame]:
    created = (created_at or datetime.now(UTC)).isoformat()
    proposal_rows = _proposal_rows(proposals if proposals is not None else pl.DataFrame())
    ack_rows = _latest_rows_by_strategy(
        proposal_ack if proposal_ack is not None else pl.DataFrame()
    )
    run_rows = (runs if runs is not None else pl.DataFrame()).to_dicts()
    daily_rows = _latest_rows_by_strategy(daily if daily is not None else pl.DataFrame())

    keys = _collect_strategy_keys(proposal_rows, ack_rows, run_rows, daily_rows)
    registry_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for key in sorted(
        keys,
        key=lambda item: (item.symbol, item.proposal_id, item.paper_tracker_id),
    ):
        proposal = _match_strategy_row(proposal_rows, key)
        ack = _match_strategy_row(ack_rows, key)
        daily_row = _match_strategy_row(daily_rows, key)
        matched_runs = [row for row in run_rows if _row_matches_key(row, key)]
        registry = _registry_row(
            key=key,
            proposal=proposal,
            ack=ack,
            daily=daily_row,
            runs=matched_runs,
            created_at=created,
        )
        gate = _promotion_gate_row(
            registry=registry,
            daily=daily_row,
            runs=matched_runs,
            created_at=created,
        )
        registry_rows.append(registry)
        gate_rows.append(gate)
    return {
        "paper_strategy_registry": _frame(registry_rows, PAPER_STRATEGY_REGISTRY_SCHEMA),
        "paper_strategy_promotion_gate": _frame(
            gate_rows,
            PAPER_STRATEGY_PROMOTION_GATE_SCHEMA,
        ),
    }


def _proposal_rows(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    rows = []
    for row in frame.to_dicts():
        recommended = _text(row.get("recommended_mode")).lower()
        decision = _text(row.get("decision")).upper()
        proposal_id = _text(row.get("proposal_id")) or _text(row.get("strategy_id"))
        if not proposal_id and decision not in {"PAPER_READY", "LIVE_SMALL_READY"}:
            continue
        if recommended not in {"", "paper", "paper_only"} and decision != "PAPER_READY":
            continue
        rows.append(row)
    return rows


def _latest_rows_by_strategy(frame: pl.DataFrame) -> list[dict[str, Any]]:
    if frame.is_empty():
        return []
    best: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in frame.to_dicts():
        key = _row_key_tuple(row)
        if key == ("", "", "", ""):
            continue
        current = best.get(key)
        if current is None or _row_sort_ts(row) >= _row_sort_ts(current):
            best[key] = row
    return list(best.values())


def _collect_strategy_keys(
    proposals: list[dict[str, Any]],
    ack_rows: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
) -> set[_StrategyKey]:
    keys: set[_StrategyKey] = set()
    for row in [*proposals, *ack_rows, *run_rows, *daily_rows]:
        proposal_id, paper_tracker_id, strategy_candidate, symbol = _row_key_tuple(row)
        if not any([proposal_id, paper_tracker_id, strategy_candidate, symbol]):
            continue
        keys.add(
            _StrategyKey(
                proposal_id=proposal_id,
                paper_tracker_id=paper_tracker_id,
                strategy_candidate=strategy_candidate,
                symbol=symbol,
            )
        )
    return keys


def _registry_row(
    *,
    key: _StrategyKey,
    proposal: Mapping[str, Any],
    ack: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    accepted = _bool(ack.get("accepted"))
    reject_reason = _text(ack.get("reject_reason"))
    proposal_id = (
        key.proposal_id
        or _text(ack.get("proposal_id"))
        or _text(proposal.get("proposal_id"))
    )
    explicit_paper_tracker_id = (
        key.paper_tracker_id
        or _text(ack.get("paper_tracker_id"))
        or _text(daily.get("paper_tracker_id"))
    )
    paper_tracker_id = explicit_paper_tracker_id or (
        proposal_id if accepted and (daily or runs) else ""
    )
    strategy_id = paper_tracker_id or proposal_id or _text(daily.get("strategy_id"))
    paper_only = _paper_only(ack, proposal)
    max_live_notional = _float(
        ack.get("max_live_notional_usdt")
        or ack.get("advisory_max_live_notional_usdt")
        or proposal.get("max_live_notional_usdt")
    )
    if max_live_notional is None and paper_only:
        max_live_notional = 0.0
    status = _registry_status(
        accepted=accepted,
        reject_reason=reject_reason,
        paper_tracker_id=paper_tracker_id,
        daily=daily,
        runs=runs,
    )
    return {
        "strategy_id": strategy_id,
        "strategy_version": _strategy_version(strategy_id or proposal_id),
        "proposal_id": proposal_id,
        "proposal_hash": _text(ack.get("proposal_hash")) or _proposal_hash(proposal),
        "paper_tracker_id": paper_tracker_id,
        "symbol": key.symbol or _symbol(ack) or _symbol(proposal) or _symbol(daily),
        "strategy_candidate": (
            key.strategy_candidate
            or _text(ack.get("strategy_candidate"))
            or _text(proposal.get("strategy_candidate"))
            or _text(daily.get("strategy_candidate"))
        ),
        "status": status,
        "paper_start_at": _paper_start_at(ack=ack, daily=daily, runs=runs),
        "rules_locked": bool(accepted and paper_only and paper_tracker_id),
        "accepted": accepted,
        "reject_reason": reject_reason,
        "first_seen_at": _first_seen_at(proposal=proposal, ack=ack, runs=runs, daily=daily),
        "accepted_at": _text(ack.get("accepted_at")) or _text(ack.get("ingest_ts")),
        "expires_at": _text(ack.get("expires_at")),
        "source_pack_sha256": _text(ack.get("source_pack_sha256")),
        "source_v5_bundle_sha256": _text(ack.get("source_v5_bundle_sha256"))
        or _text(ack.get("bundle_sha256")),
        "paper_only": paper_only,
        "max_live_notional_usdt": max_live_notional,
        "live_order_effect": _text(ack.get("live_order_effect")) or "paper_only_no_live_order",
        "created_at": created_at,
        "source": PAPER_PIPELINE_SOURCE,
        "schema_version": PAPER_PIPELINE_SCHEMA_VERSION,
    }


def _promotion_gate_row(
    *,
    registry: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
    created_at: str,
) -> dict[str, Any]:
    values = _paper_pnl_values(runs)
    recent_values = _recent_paper_pnl_values(runs)
    paper_days = _int(daily.get("paper_days")) or _int(daily.get("heartbeat_day_count")) or 0
    closed_entries = (
        _int(daily.get("paper_pnl_observed_count"))
        or _int(daily.get("cumulative_paper_pnl_observed_count"))
        or len(values)
    )
    entry_day_count = _int(daily.get("entry_day_count")) or 0
    arrival_mid_coverage = _float(daily.get("arrival_mid_coverage")) or 0.0
    mean_bps = _float(daily.get("avg_paper_pnl_bps")) or _mean(values)
    median_bps = _median(values)
    p25_bps = _percentile(values, 0.25)
    recent_mean = _mean(recent_values)
    recent_hit_rate = _hit_rate(recent_values)
    cost_sources = _cost_sources(daily.get("cost_source_mix"))
    cost_trusted = bool(cost_sources & TRUSTED_PAPER_COST_SOURCES) and not bool(
        cost_sources & BLOCKED_PAPER_COST_SOURCES
    )
    accepted = bool(registry.get("accepted"))
    paper_tracker_created = bool(_text(registry.get("paper_tracker_id")))
    block_reasons = _gate_block_reasons(
        accepted=accepted,
        reject_reason=_text(registry.get("reject_reason")),
        paper_tracker_created=paper_tracker_created,
        paper_days=paper_days,
        closed_entries=closed_entries,
        entry_day_count=entry_day_count,
        arrival_mid_coverage=arrival_mid_coverage,
        mean_bps=mean_bps,
        median_bps=median_bps,
        p25_bps=p25_bps,
        recent_mean=recent_mean,
        cost_trusted=cost_trusted,
        daily_block_reasons=_json_list(daily.get("live_block_reason")),
    )
    paper_ready = not block_reasons
    return {
        "strategy_id": _text(registry.get("strategy_id")),
        "strategy_version": _text(registry.get("strategy_version")),
        "proposal_id": _text(registry.get("proposal_id")),
        "proposal_hash": _text(registry.get("proposal_hash")),
        "paper_tracker_id": _text(registry.get("paper_tracker_id")),
        "symbol": _text(registry.get("symbol")),
        "strategy_candidate": _text(registry.get("strategy_candidate")),
        "lifecycle_state": "PAPER_READY" if paper_ready else _text(registry.get("status")),
        "accepted": accepted,
        "paper_tracker_created": paper_tracker_created,
        "paper_runs": len(runs),
        "paper_days": paper_days,
        "closed_entries": closed_entries,
        "entry_day_count": entry_day_count,
        "arrival_mid_coverage": arrival_mid_coverage,
        "mean_after_cost_bps": mean_bps,
        "median_after_cost_bps": median_bps,
        "p25_after_cost_bps": p25_bps,
        "recent_7d_mean_bps": recent_mean,
        "recent_7d_hit_rate": recent_hit_rate,
        "cost_source": safe_json_dumps(sorted(cost_sources)),
        "cost_trusted_for_paper": cost_trusted,
        "backtest_paper_conflict": False,
        "paper_ready": paper_ready,
        "block_reason": safe_json_dumps(block_reasons),
        "created_at": created_at,
        "source": PAPER_PIPELINE_SOURCE,
        "schema_version": PAPER_PIPELINE_SCHEMA_VERSION,
    }


def _gate_block_reasons(
    *,
    accepted: bool,
    reject_reason: str,
    paper_tracker_created: bool,
    paper_days: int,
    closed_entries: int,
    entry_day_count: int,
    arrival_mid_coverage: float,
    mean_bps: float | None,
    median_bps: float | None,
    p25_bps: float | None,
    recent_mean: float | None,
    cost_trusted: bool,
    daily_block_reasons: list[str],
) -> list[str]:
    reasons: set[str] = set()
    if not accepted:
        reasons.add("proposal_not_acked")
    if reject_reason:
        reasons.add(f"v5_rejected:{reject_reason}")
    if not paper_tracker_created:
        reasons.add("paper_tracker_missing")
    if paper_days < MIN_PAPER_DAYS:
        reasons.add("insufficient_paper_days")
    if closed_entries < MIN_CLOSED_ENTRIES:
        reasons.add("insufficient_closed_entries")
    if entry_day_count < MIN_ENTRY_DAY_COUNT:
        reasons.add("insufficient_entry_day_count")
    if arrival_mid_coverage < MIN_ARRIVAL_MID_COVERAGE:
        reasons.add("insufficient_arrival_mid_coverage")
    if mean_bps is None or mean_bps <= 0.0:
        reasons.add("mean_after_cost_not_positive")
    if median_bps is None or median_bps <= 0.0:
        reasons.add("median_after_cost_not_positive")
    if p25_bps is None or p25_bps < MIN_P25_AFTER_COST_BPS:
        reasons.add("p25_after_cost_too_weak")
    if recent_mean is None or recent_mean < 0.0:
        reasons.add("recent_7d_mean_negative_or_missing")
    if not cost_trusted:
        reasons.add("cost_not_trusted_for_paper")
    reasons.update(_paper_block_reason_subset(daily_block_reasons))
    return sorted(reasons)


def _paper_block_reason_subset(reasons: list[str]) -> set[str]:
    ignored = {
        "no_live_slippage_coverage",
        "cost_source_not_actual_or_mixed",
        "paper_active_but_no_entries_yet",
    }
    return {reason for reason in reasons if reason and reason not in ignored}


def _registry_status(
    *,
    accepted: bool,
    reject_reason: str,
    paper_tracker_id: str,
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
) -> str:
    if reject_reason and not accepted:
        return "REJECTED_BY_V5"
    if not accepted:
        return "PROPOSED_AWAITING_ACK"
    if not paper_tracker_id:
        return "ACKED_TRACKER_MISSING"
    if bool(daily.get("live_eligible")):
        return "PAPER_REVIEW"
    if daily or runs:
        return "PAPER_TRACKING"
    return "ACKED"


def _match_strategy_row(rows: list[Mapping[str, Any]], key: _StrategyKey) -> Mapping[str, Any]:
    for row in rows:
        if _row_matches_key(row, key):
            return row
    return {}


def _row_matches_key(row: Mapping[str, Any], key: _StrategyKey) -> bool:
    aliases = {
        _text(row.get("proposal_id")),
        _text(row.get("paper_tracker_id")),
        _text(row.get("strategy_id")),
    }
    key_aliases = {key.proposal_id, key.paper_tracker_id}
    if aliases & key_aliases - {""}:
        return True
    return (
        bool(key.strategy_candidate)
        and bool(key.symbol)
        and _text(row.get("strategy_candidate")) == key.strategy_candidate
        and _symbol(row) == key.symbol
    )


def _row_key_tuple(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    proposal_id = _text(row.get("proposal_id")) or _text(row.get("strategy_id"))
    paper_tracker_id = _text(row.get("paper_tracker_id")) or (
        _text(row.get("strategy_id")) if proposal_id != _text(row.get("strategy_id")) else ""
    )
    return (
        proposal_id,
        paper_tracker_id,
        _text(row.get("strategy_candidate")),
        _symbol(row),
    )


def _paper_start_at(
    *,
    ack: Mapping[str, Any],
    daily: Mapping[str, Any],
    runs: list[dict[str, Any]],
) -> str:
    run_times = [_row_sort_ts(row) for row in runs if _row_sort_ts(row)]
    if run_times:
        return min(run_times)
    return (
        _text(daily.get("paper_start_at"))
        or _text(ack.get("accepted_at"))
        or _text(ack.get("ingest_ts"))
    )


def _first_seen_at(
    *,
    proposal: Mapping[str, Any],
    ack: Mapping[str, Any],
    runs: list[dict[str, Any]],
    daily: Mapping[str, Any],
) -> str:
    values = [
        _text(proposal.get("created_at")),
        _text(proposal.get("generated_at")),
        _text(ack.get("first_seen_at")),
        _text(ack.get("ingest_ts")),
        _text(daily.get("created_at")),
        *[_row_sort_ts(row) for row in runs],
    ]
    values = [value for value in values if value]
    return min(values) if values else ""


def _paper_only(ack: Mapping[str, Any], proposal: Mapping[str, Any]) -> bool:
    live_effect = _text(ack.get("live_order_effect")).lower()
    mode = _text(ack.get("recommended_mode") or proposal.get("recommended_mode")).lower()
    max_notional = _float(
        ack.get("max_live_notional_usdt")
        or ack.get("advisory_max_live_notional_usdt")
        or proposal.get("max_live_notional_usdt")
    )
    return (
        "paper" in live_effect
        or mode in {"paper", "paper_only"}
        or max_notional == 0.0
        or max_notional is None
    )


def _paper_pnl_values(rows: list[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for key in (
            "paper_pnl_bps",
            "paper_pnl_bps_4h",
            "paper_pnl_bps_8h",
            "paper_pnl_bps_12h",
            "paper_pnl_bps_24h",
            "paper_pnl_bps_48h",
            "paper_pnl_bps_72h",
        ):
            value = _float(row.get(key))
            if value is not None:
                values.append(value)
    return values


def _recent_paper_pnl_values(rows: list[Mapping[str, Any]]) -> list[float]:
    dated: list[tuple[date, float]] = []
    for row in rows:
        row_date = _row_date(row)
        if row_date is None:
            continue
        for value in _paper_pnl_values([row]):
            dated.append((row_date, value))
    if not dated:
        return []
    latest = max(day for day, _value in dated)
    cutoff = latest - timedelta(days=6)
    return [value for day, value in dated if day >= cutoff]


def _row_date(row: Mapping[str, Any]) -> date | None:
    text = _text(row.get("as_of_date")) or _text(row.get("ts_utc"))[:10]
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _cost_sources(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, list):
        items = value
    else:
        text = str(value).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {
                part.strip().lower()
                for part in text.replace(";", ",").split(",")
                if part.strip()
            }
        items = parsed if isinstance(parsed, list) else [parsed]
    sources = set()
    for item in items:
        if isinstance(item, dict):
            source = item.get("cost_source") or item.get("source")
        else:
            source = item
        text = str(source or "").strip().lower()
        if text:
            sources.add(text)
    return sources


def _json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    text = str(value).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [part.strip() for part in text.replace(";", ",").split(",") if part.strip()]
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if parsed else []


def _proposal_hash(row: Mapping[str, Any]) -> str:
    if not row:
        return ""
    payload = {
        key: row.get(key)
        for key in sorted(
            {
                "proposal_id",
                "strategy_candidate",
                "symbol",
                "entry_conditions",
                "recommended_mode",
                "suggested_horizon",
                "cost_source_mix",
                "required_paper_days",
                "required_slippage_coverage",
            }
        )
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _strategy_version(strategy_id: str) -> str:
    if not strategy_id:
        return ""
    marker = "_V"
    if marker not in strategy_id:
        return "V1"
    suffix = strategy_id.rsplit(marker, 1)[-1]
    return f"V{suffix}" if suffix else "V1"


def _resolve_as_of_date(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and str(value).lower() != "auto":
        return date.fromisoformat(str(value)[:10])
    return datetime.now(UTC).date()


def _row_sort_ts(row: Mapping[str, Any]) -> str:
    for key in (
        "accepted_at",
        "created_at",
        "generated_at",
        "as_of_ts",
        "ts_utc",
        "as_of_date",
        "ingest_ts",
        "bundle_ts",
    ):
        value = _text(row.get(key))
        if value:
            return value
    return ""


def _symbol(row: Mapping[str, Any]) -> str:
    value = _text(row.get("symbol") or row.get("v5_symbol") or row.get("normalized_symbol"))
    return normalize_symbol(value) if value else ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan", "nat"} else text


def _float(value: Any) -> float | None:
    text = _text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    return text in {"1", "true", "yes", "y", "accepted"}


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    return None if not values else float(median(values))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q)))
    return ordered[index]


def _hit_rate(values: list[float]) -> float | None:
    return None if not values else sum(1 for value in values if value > 0.0) / len(values)


def _frame(rows: list[dict[str, Any]], schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, orient="row")
