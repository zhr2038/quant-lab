from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.second_stage_alpha_factory import (
    SECOND_STAGE_CANDIDATES,
    SECOND_STAGE_SUMMARY_DATASET,
    build_and_publish_second_stage_alpha_factory,
)
from quant_lab.research.strategy_evidence import (
    SUMMARY_SCHEMA,
    normalize_strategy_evidence_decisions,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "research.alpha_factory.v0.1"
SCHEMA_VERSION = "alpha_factory.v0.1"
MAX_DAILY_CANDIDATES = 200

ALPHA_FACTORY_CANDIDATE_DATASET = Path("gold") / "alpha_factory_candidate"
ALPHA_FACTORY_RESULT_DATASET = Path("gold") / "alpha_factory_result"
ALPHA_FACTORY_PROMOTION_QUEUE_DATASET = Path("gold") / "alpha_factory_promotion_queue"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"

STRATEGY_TEMPLATE_BY_CANDIDATE = {
    "v5.expanded_relative_strength_top1_shadow": "expanded_relative_strength",
    "v5.expanded_relative_strength_top3_shadow": "expanded_relative_strength",
    "v5.expanded_relative_strength_rotation_shadow": "expanded_relative_strength",
    "v5.btc_strict_probe_exit_policy_review": "exit_policy_review",
    "v5.eth_f3_exit_policy_review": "exit_policy_review",
    "v5.sol_paper_exit_policy_review": "exit_policy_review",
    "v5.futures_risk_off_hedge_shadow": "futures_hedge_shadow",
    "v5.futures_downtrend_short_shadow": "futures_hedge_shadow",
    "v5.pair_trade_eth_btc_shadow": "pair_market_neutral_shadow",
    "v5.pair_trade_sol_eth_shadow": "pair_market_neutral_shadow",
    "v5.alt_basket_vs_btc_shadow": "pair_market_neutral_shadow",
    "v5.alt_impulse_shadow": "alt_impulse_regime",
}
STRATEGY_TEMPLATES = (
    "expanded_relative_strength",
    "exit_policy_review",
    "futures_hedge_shadow",
    "pair_market_neutral_shadow",
    "alt_impulse_regime",
)
ALPHA_FACTORY_CANDIDATES = frozenset([*SECOND_STAGE_CANDIDATES, "v5.alt_impulse_shadow"])

CANDIDATE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "template_name": pl.Utf8,
    "candidate_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "parameter_json": pl.Utf8,
    "whitelist_json": pl.Utf8,
    "blacklist_json": pl.Utf8,
    "source_dataset": pl.Utf8,
    "candidate_state": pl.Utf8,
    "max_live_notional_usdt": pl.Float64,
    "safety_mode": pl.Utf8,
    "source": pl.Utf8,
}

RESULT_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "template_name": pl.Utf8,
    "candidate_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "regime_state": pl.Utf8,
    "horizon_hours": pl.Int64,
    "sample_count": pl.Int64,
    "complete_sample_count": pl.Int64,
    "avg_net_bps": pl.Float64,
    "median_net_bps": pl.Float64,
    "p25_net_bps": pl.Float64,
    "win_rate": pl.Float64,
    "cost_source_mix": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "start_ts": pl.Datetime(time_zone="UTC"),
    "end_ts": pl.Datetime(time_zone="UTC"),
    "max_live_notional_usdt": pl.Float64,
    "source": pl.Utf8,
}

PROMOTION_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "template_name": pl.Utf8,
    "candidate_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "horizon_hours": pl.Int64,
    "promotion_state": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "action": pl.Utf8,
    "reasons": pl.Utf8,
    "max_live_notional_usdt": pl.Float64,
    "manual_live_approval_required": pl.Boolean,
    "source": pl.Utf8,
}


class AlphaFactoryBuildResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    candidate_rows: int = Field(ge=0)
    result_rows: int = Field(ge=0)
    promotion_rows: int = Field(ge=0)
    strategy_evidence_rows: int = Field(ge=0)
    strategy_evidence_sample_rows: int = Field(ge=0)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_alpha_factory(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    lookback_days: int = 30,
    max_candidates: int = MAX_DAILY_CANDIDATES,
) -> AlphaFactoryBuildResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    generated_at = datetime.now(UTC)

    second_stage = build_and_publish_second_stage_alpha_factory(
        root,
        as_of_date=day,
        lookback_days=lookback_days,
    )
    summary = _alpha_factory_source_summary(root, day)
    candidates = build_alpha_factory_candidates(
        summary,
        as_of_date=day,
        generated_at=generated_at,
        max_candidates=max_candidates,
    )
    results = build_alpha_factory_results(
        summary,
        as_of_date=day,
        generated_at=generated_at,
        max_candidates=max_candidates,
    )
    promotion = build_alpha_factory_promotion_queue(results, generated_at=generated_at)

    write_parquet_dataset(candidates, root / ALPHA_FACTORY_CANDIDATE_DATASET)
    write_parquet_dataset(results, root / ALPHA_FACTORY_RESULT_DATASET)
    write_parquet_dataset(promotion, root / ALPHA_FACTORY_PROMOTION_QUEUE_DATASET)
    evidence_rows = _publish_alpha_factory_results_to_strategy_evidence(root, results, day)

    return AlphaFactoryBuildResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        candidate_rows=candidates.height,
        result_rows=results.height,
        promotion_rows=promotion.height,
        strategy_evidence_rows=evidence_rows,
        strategy_evidence_sample_rows=read_parquet_dataset(
            root / "gold" / "strategy_evidence_sample"
        ).height,
        decision_counts=_decision_counts(results),
        warnings=list(second_stage.warnings)
        + ([] if candidates.height <= max_candidates else ["alpha_factory_candidate_cap_applied"]),
    )


def build_alpha_factory_candidates(
    summary: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
    max_candidates: int = MAX_DAILY_CANDIDATES,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    rows = []
    for row in _ranked_summary_rows(summary, as_of_date=as_of_date, max_candidates=max_candidates):
        candidate = str(row.get("strategy_candidate") or "")
        template = _template_name(candidate)
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "template_name": template,
                "candidate_id": _candidate_id(row),
                "strategy_candidate": candidate,
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "regime_state": str(row.get("regime_state") or "UNKNOWN"),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "parameter_json": safe_json_dumps(_parameter_payload(row, template)),
                "whitelist_json": safe_json_dumps([normalize_symbol(row.get("symbol"))]),
                "blacklist_json": safe_json_dumps([]),
                "source_dataset": str(row.get("source_dataset") or "gold/strategy_evidence"),
                "candidate_state": "RESEARCH",
                "max_live_notional_usdt": 0.0,
                "safety_mode": "paper_shadow_only",
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=CANDIDATE_SCHEMA, orient="row") if rows else pl.DataFrame(
        schema=CANDIDATE_SCHEMA
    )


def build_alpha_factory_results(
    summary: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime | None = None,
    max_candidates: int = MAX_DAILY_CANDIDATES,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    rows = []
    for row in _ranked_summary_rows(summary, as_of_date=as_of_date, max_candidates=max_candidates):
        decision, reasons = alpha_factory_decision(
            sample_count=_int(row.get("sample_count")) or 0,
            complete_sample_count=_int(row.get("complete_sample_count")) or 0,
            avg_net_bps=_float(row.get("avg_net_bps")),
            p25_net_bps=_float(row.get("p25_net_bps")),
            win_rate=_float(row.get("win_rate")),
        )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "template_name": _template_name(row.get("strategy_candidate")),
                "candidate_id": _candidate_id(row),
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "regime_state": str(row.get("regime_state") or "UNKNOWN"),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "sample_count": _int(row.get("sample_count")) or 0,
                "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
                "avg_net_bps": _float(row.get("avg_net_bps")),
                "median_net_bps": _float(row.get("median_net_bps")),
                "p25_net_bps": _float(row.get("p25_net_bps")),
                "win_rate": _float(row.get("win_rate")),
                "cost_source_mix": str(row.get("cost_source_mix") or "{}"),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "recommended_mode": _recommended_mode(decision),
                "start_ts": _parse_dt(row.get("start_ts")),
                "end_ts": _parse_dt(row.get("end_ts")),
                "max_live_notional_usdt": 0.0,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=RESULT_SCHEMA, orient="row") if rows else pl.DataFrame(
        schema=RESULT_SCHEMA
    )


def build_alpha_factory_promotion_queue(
    results: pl.DataFrame,
    *,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    generated = generated_at or datetime.now(UTC)
    rows = []
    for row in results.to_dicts() if not results.is_empty() else []:
        decision = str(row.get("decision") or "RESEARCH").upper()
        reasons = _json_list(row.get("decision_reasons"))
        rows.append(
            {
                "as_of_date": str(row.get("as_of_date") or ""),
                "generated_at": generated,
                "schema_version": SCHEMA_VERSION,
                "template_name": str(row.get("template_name") or ""),
                "candidate_id": str(row.get("candidate_id") or ""),
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "promotion_state": decision,
                "recommended_mode": _recommended_mode(decision),
                "action": _promotion_action(decision),
                "reasons": safe_json_dumps(
                    _dedupe_text([*reasons, "alpha_factory_live_disabled"])
                ),
                "max_live_notional_usdt": 0.0,
                "manual_live_approval_required": True,
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=PROMOTION_SCHEMA, orient="row") if rows else pl.DataFrame(
        schema=PROMOTION_SCHEMA
    )


def alpha_factory_decision(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    win_rate: float | None,
) -> tuple[str, list[str]]:
    if complete_sample_count < 10:
        return "RESEARCH", ["insufficient_complete_samples"]
    if sample_count >= 30 and (avg_net_bps or 0.0) < 0.0 and (win_rate or 0.0) < 0.45:
        return "KILL", ["negative_after_cost_edge", "win_rate_below_threshold"]
    if (
        complete_sample_count >= 30
        and win_rate is not None
        and win_rate > 0.55
        and p25_net_bps is not None
        and p25_net_bps > -50.0
    ):
        return "PAPER_READY", ["paper_ready_thresholds_met", "live_disabled"]
    if complete_sample_count >= 10 and avg_net_bps is not None and avg_net_bps > 0.0:
        return "KEEP_SHADOW", ["positive_after_cost_edge", "collect_more_samples"]
    return "RESEARCH", ["edge_not_confirmed"]


def alpha_factory_daily_md(
    *,
    candidates: pl.DataFrame,
    results: pl.DataFrame,
    promotion_queue: pl.DataFrame,
    as_of_date: str | date | None = None,
) -> str:
    day = str(as_of_date or _latest_as_of_date(results) or _latest_as_of_date(candidates) or "auto")
    lines = [
        f"# Alpha Factory Daily - {day}",
        "",
        "Alpha Factory is read-only. It can only produce research, shadow, or paper candidates.",
        "It never changes V5 live symbols and never creates live orders.",
        "",
        f"- candidates: {candidates.height}",
        f"- results: {results.height}",
        f"- promotion queue: {promotion_queue.height}",
        "- max_live_notional_usdt: 0",
        "",
        "## Decision Counts",
        "",
    ]
    counts = _decision_counts(results)
    if counts:
        for decision, count in sorted(counts.items()):
            lines.append(f"- {decision}: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Templates", ""])
    for template in STRATEGY_TEMPLATES:
        frame = _filter_template(results, template)
        lines.append(f"- {template}: {frame.height} rows")
    return "\n".join(lines).rstrip() + "\n"


def _publish_alpha_factory_results_to_strategy_evidence(
    root: Path,
    results: pl.DataFrame,
    day: date,
) -> int:
    dataset = root / STRATEGY_EVIDENCE_DATASET
    existing = read_parquet_dataset(dataset)
    summary = _results_to_strategy_evidence(results)
    if existing.is_empty():
        write_parquet_dataset(summary, dataset)
        return summary.height
    retained = existing
    if "as_of_date" in retained.columns and "strategy_candidate" in retained.columns:
        retained = retained.filter(
            ~(
                (pl.col("as_of_date").cast(pl.Utf8) == day.isoformat())
                & pl.col("strategy_candidate")
                .cast(pl.Utf8)
                .is_in(sorted(ALPHA_FACTORY_CANDIDATES))
            )
        )
    combined = pl.concat([retained, summary], how="diagonal_relaxed")
    combined = normalize_strategy_evidence_decisions(combined)
    write_parquet_dataset(combined, dataset)
    return combined.height


def _results_to_strategy_evidence(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame(schema=SUMMARY_SCHEMA)
    rows = []
    for row in results.to_dicts():
        rows.append(
            {
                "strategy": "v5",
                "evidence_version": SCHEMA_VERSION,
                "as_of_date": str(row.get("as_of_date") or ""),
                "strategy_candidate": str(row.get("strategy_candidate") or ""),
                "candidate_name": str(row.get("strategy_candidate") or ""),
                "symbol": normalize_symbol(row.get("symbol")) or "UNKNOWN",
                "regime_state": str(row.get("regime_state") or "UNKNOWN"),
                "horizon_hours": _int(row.get("horizon_hours")) or 0,
                "sample_count": _int(row.get("sample_count")) or 0,
                "complete_sample_count": _int(row.get("complete_sample_count")) or 0,
                "avg_net_bps": _float(row.get("avg_net_bps")),
                "median_net_bps": _float(row.get("median_net_bps")),
                "p25_net_bps": _float(row.get("p25_net_bps")),
                "win_rate": _float(row.get("win_rate")),
                "cost_source_mix": str(row.get("cost_source_mix") or "{}"),
                "decision": _strategy_evidence_decision(str(row.get("decision") or "")),
                "decision_reasons": safe_json_dumps(
                    _dedupe_text(
                        [
                            *_json_list(row.get("decision_reasons")),
                            "alpha_factory_live_disabled",
                            "max_live_notional_zero",
                        ]
                    )
                ),
                "start_ts": _parse_dt(row.get("start_ts")),
                "end_ts": _parse_dt(row.get("end_ts")),
                "created_at": _parse_dt(row.get("generated_at")) or datetime.now(UTC),
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA, orient="row")


def _ranked_summary_rows(
    summary: pl.DataFrame,
    *,
    as_of_date: date,
    max_candidates: int,
) -> list[dict[str, Any]]:
    if summary.is_empty():
        return []
    rows = [
        row
        for row in summary.to_dicts()
        if str(row.get("as_of_date") or "")[:10] == as_of_date.isoformat()
        and str(row.get("strategy_candidate") or "") in ALPHA_FACTORY_CANDIDATES
    ]
    rows.sort(
        key=lambda row: (
            _int(row.get("complete_sample_count")) or 0,
            _float(row.get("avg_net_bps")) or -1e9,
            _float(row.get("win_rate")) or 0.0,
            str(row.get("strategy_candidate") or ""),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    return rows[: max(max_candidates, 1)]


def _candidate_id(row: dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("as_of_date") or ""),
            str(row.get("strategy_candidate") or ""),
            normalize_symbol(row.get("symbol")) or "UNKNOWN",
            str(row.get("regime_state") or "UNKNOWN"),
            str(_int(row.get("horizon_hours")) or 0),
        ]
    )


def _template_name(candidate: Any) -> str:
    return STRATEGY_TEMPLATE_BY_CANDIDATE.get(str(candidate or ""), "alpha_factory")


def _alpha_factory_source_summary(root: Path, day: date) -> pl.DataFrame:
    second_stage = read_parquet_dataset(root / SECOND_STAGE_SUMMARY_DATASET)
    second_stage = _with_source_dataset(
        second_stage,
        "gold/second_stage_alpha_factory_summary",
    )
    strategy_evidence = read_parquet_dataset(root / STRATEGY_EVIDENCE_DATASET)
    alt_impulse = pl.DataFrame()
    if not strategy_evidence.is_empty() and "strategy_candidate" in strategy_evidence.columns:
        alt_impulse = strategy_evidence.filter(
            (pl.col("strategy_candidate").cast(pl.Utf8) == "v5.alt_impulse_shadow")
            & (pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10) == day.isoformat())
        )
        alt_impulse = _with_source_dataset(alt_impulse, "gold/strategy_evidence")
    frames = [frame for frame in [second_stage, alt_impulse] if not frame.is_empty()]
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _with_source_dataset(frame: pl.DataFrame, source_dataset: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "source_dataset" in frame.columns:
        return frame.with_columns(pl.lit(source_dataset).alias("source_dataset"))
    return frame.with_columns(pl.lit(source_dataset).alias("source_dataset"))


def _parameter_payload(row: dict[str, Any], template: str) -> dict[str, Any]:
    return {
        "template": template,
        "horizon_hours": _int(row.get("horizon_hours")) or 0,
        "regime_state": str(row.get("regime_state") or "UNKNOWN"),
        "dry_run_first": True,
        "batch_scan": True,
        "paper_only_by_default": True,
    }


def _recommended_mode(decision: str) -> str:
    if decision == "KILL":
        return "none"
    if decision == "PAPER_READY":
        return "paper"
    if decision == "KEEP_SHADOW":
        return "shadow"
    return "research"


def _promotion_action(decision: str) -> str:
    if decision == "KILL":
        return "CLOSE_RESEARCH"
    if decision == "PAPER_READY":
        return "QUEUE_FOR_PAPER_REVIEW"
    if decision == "KEEP_SHADOW":
        return "CONTINUE_SHADOW"
    return "COLLECT_MORE_SAMPLES"


def _strategy_evidence_decision(factory_decision: str) -> str:
    if factory_decision == "RESEARCH":
        return "RESEARCH_ONLY"
    return factory_decision


def _filter_template(frame: pl.DataFrame, template: str) -> pl.DataFrame:
    if frame.is_empty() or "template_name" not in frame.columns:
        return pl.DataFrame()
    return frame.filter(pl.col("template_name") == template)


def _latest_as_of_date(frame: pl.DataFrame) -> str | None:
    if frame.is_empty() or "as_of_date" not in frame.columns:
        return None
    values = [
        str(value)[:10]
        for value in frame.get_column("as_of_date").drop_nulls().to_list()
        if str(value).strip()
    ]
    return max(values) if values else None


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    text = str(value or "auto").strip().lower()
    if not text or text == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(text[:10])


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and parsed not in {float("inf"), float("-inf")} else None


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return int(parsed) if parsed is not None else None


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if not isinstance(value, str):
        return [str(value)]
    try:
        import json

        parsed = json.loads(value)
    except Exception:
        return [value] if value else []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return [str(parsed)] if parsed is not None else []


def _dedupe_text(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _decision_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "decision" not in frame.columns:
        return {}
    return {
        str(row["decision"]): int(row["count"])
        for row in frame.group_by("decision").len(name="count").to_dicts()
    }
