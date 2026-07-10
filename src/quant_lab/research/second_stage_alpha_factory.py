from __future__ import annotations

import statistics
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import (
    count_parquet_rows,
    read_parquet_dataset,
    read_parquet_lazy,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
from quant_lab.research.strategy_evidence import (
    EVIDENCE_VERSION,
    SAMPLE_SCHEMA,
    SUMMARY_SCHEMA,
    normalize_strategy_evidence_decisions,
    normalize_strategy_evidence_samples,
    summarize_strategy_evidence,
)
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "research.second_stage_alpha_factory.v0.1"
SCHEMA_VERSION = "second_stage_alpha_factory.v0.1"

SECOND_STAGE_SAMPLE_DATASET = Path("gold") / "second_stage_alpha_factory_sample"
SECOND_STAGE_SUMMARY_DATASET = Path("gold") / "second_stage_alpha_factory_summary"
EXPANDED_RELATIVE_STRENGTH_DECISION_SAMPLE_DATASET = (
    Path("gold") / "expanded_relative_strength_decision_sample"
)
EXIT_POLICY_REVIEW_SAMPLE_DATASET = Path("gold") / "exit_policy_review_sample"
EXIT_POLICY_REVIEW_SUMMARY_DATASET = Path("gold") / "exit_policy_review_summary"
STRATEGY_EVIDENCE_SAMPLE_DATASET = Path("gold") / "strategy_evidence_sample"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"

MARKET_BAR_DATASET = Path("silver") / "market_bar"
EXPANDED_QUALITY_DATASET = Path("gold") / "expanded_universe_quality"
COST_BUCKET_DAILY_DATASET = Path("gold") / "cost_bucket_daily"
MARKET_REGIME_DATASET = Path("gold") / "market_regime_daily"
BTC_EXIT_REVIEW_DATASET = Path("gold") / "btc_probe_exit_policy_review"
PAPER_RUN_DATASET = Path("gold") / "paper_strategy_runs"

RELATIVE_STRENGTH_CANDIDATES = (
    "v5.expanded_relative_strength_top1_shadow",
    "v5.expanded_relative_strength_top3_shadow",
    "v5.expanded_relative_strength_rotation_shadow",
)
FUTURES_SHADOW_CANDIDATES = (
    "v5.futures_risk_off_hedge_proxy_shadow",
    "v5.futures_downtrend_short_proxy_shadow",
)
TRUE_FUTURES_SHADOW_CANDIDATES = (
    "v5.futures_risk_off_hedge_shadow",
    "v5.futures_downtrend_short_shadow",
)
EXIT_POLICY_CANDIDATES = (
    "v5.btc_strict_probe_exit_policy_review",
    "v5.eth_f3_exit_policy_review",
    "v5.sol_paper_exit_policy_review",
)
PAIR_SHADOW_CANDIDATES = (
    "v5.pair_trade_eth_btc_shadow",
    "v5.pair_trade_sol_eth_shadow",
    "v5.alt_basket_vs_btc_shadow",
)
SECOND_STAGE_CANDIDATES = frozenset(
    [
        *RELATIVE_STRENGTH_CANDIDATES,
        *FUTURES_SHADOW_CANDIDATES,
        *EXIT_POLICY_CANDIDATES,
        *PAIR_SHADOW_CANDIDATES,
    ]
)

EXIT_POLICY_REVIEW_SAMPLE_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "strategy_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "source_entry_id": pl.Utf8,
    "symbol": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "actual_exit_net_bps": pl.Float64,
    "fixed_hold_4h_net_bps": pl.Float64,
    "fixed_hold_8h_net_bps": pl.Float64,
    "fixed_hold_12h_net_bps": pl.Float64,
    "fixed_hold_24h_net_bps": pl.Float64,
    "fixed_hold_48h_net_bps": pl.Float64,
    "best_alternative_exit_policy": pl.Utf8,
    "best_alternative_net_bps": pl.Float64,
    "delta_vs_actual_bps": pl.Float64,
    "mfe_bps": pl.Float64,
    "mae_bps": pl.Float64,
    "exit_reason": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "source": pl.Utf8,
}

EXIT_POLICY_REVIEW_SUMMARY_SCHEMA: dict[str, Any] = {
    "as_of_date": pl.Utf8,
    "generated_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "strategy_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "sample_count": pl.Int64,
    "actual_exit_count": pl.Int64,
    "stop_loss_too_early_count": pl.Int64,
    "hold_24h_better_than_actual_count": pl.Int64,
    "avg_delta_hold24h_vs_actual": pl.Float64,
    "avg_delta_best_vs_actual_bps": pl.Float64,
    "best_alternative_exit_policy": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "recommended_mode": pl.Utf8,
    "source": pl.Utf8,
}

EXPANDED_RELATIVE_STRENGTH_DECISION_SAMPLE_SCHEMA: dict[str, Any] = {
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "lookback_hours": pl.Int64,
    "lookback_return_bps": pl.Float64,
    "top_k": pl.Int64,
    "selected_rank": pl.Int64,
    "selected": pl.Boolean,
    "symbol_quality_score": pl.Float64,
    "spread_bps": pl.Float64,
    "btc_corr": pl.Float64,
    "volume_24h_usdt": pl.Float64,
    "selection_reason": pl.Utf8,
    "anti_leakage_check": pl.Utf8,
    "label_horizon_hours": pl.Int64,
    "future_net_bps": pl.Float64,
    "label_status": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "schema_version": pl.Utf8,
    "source": pl.Utf8,
}

DEFAULT_HORIZONS = (4, 8, 12, 24, 48)
EXIT_POLICY_HORIZONS = (4, 8, 12, 24, 48)
DEFAULT_LOOKBACK_DAYS = 30
CONSERVATIVE_ROUNDTRIP_COST_BPS = 30.0
RELATIVE_STRENGTH_LOOKBACK_HOURS = (4, 8, 12, 24)
RELATIVE_STRENGTH_SELECTIONS = (
    ("v5.expanded_relative_strength_top1_shadow", 1),
    ("v5.expanded_relative_strength_top3_shadow", 3),
    ("v5.expanded_relative_strength_rotation_shadow", 5),
)
RELATIVE_STRENGTH_FILTER = {
    "min_quality_score": 75.0,
    "max_spread_bps": 5.0,
    "btc_corr_max": 0.7,
    "min_24h_volume_usdt": 5_000_000.0,
}


class SecondStageAlphaFactoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    sample_rows: int = Field(ge=0)
    summary_rows: int = Field(ge=0)
    expanded_relative_strength_decision_sample_rows: int = Field(ge=0)
    strategy_evidence_sample_rows: int = Field(ge=0)
    strategy_evidence_rows: int = Field(ge=0)
    exit_policy_review_sample_rows: int = Field(ge=0)
    exit_policy_review_summary_rows: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    decision_counts: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


def build_and_publish_second_stage_alpha_factory(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> SecondStageAlphaFactoryResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    samples, decision_samples, warnings = _build_second_stage_alpha_factory_tables(
        root,
        as_of_date=day,
        lookback_days=lookback_days,
    )
    summaries = _cap_second_stage_decisions(
        summarize_strategy_evidence(samples, as_of_date=day)
    )
    sample_frame = normalize_strategy_evidence_samples(samples)
    summary_frame = _summary_frame(summaries)

    write_parquet_dataset(sample_frame, root / SECOND_STAGE_SAMPLE_DATASET)
    write_parquet_dataset(summary_frame, root / SECOND_STAGE_SUMMARY_DATASET)
    write_parquet_dataset(
        decision_samples,
        root / EXPANDED_RELATIVE_STRENGTH_DECISION_SAMPLE_DATASET,
    )
    exit_policy_samples, exit_policy_summary = build_exit_policy_review_tables(
        root,
        as_of_date=day,
        lookback_days=lookback_days,
        generated_at=datetime.now(UTC),
    )
    write_parquet_dataset(exit_policy_samples, root / EXIT_POLICY_REVIEW_SAMPLE_DATASET)
    write_parquet_dataset(exit_policy_summary, root / EXIT_POLICY_REVIEW_SUMMARY_DATASET)
    sample_rows = _publish_second_stage_samples(root, sample_frame)
    evidence_rows = _publish_second_stage_summaries(root, summary_frame, day)

    return SecondStageAlphaFactoryResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        sample_rows=sample_frame.height,
        summary_rows=summary_frame.height,
        expanded_relative_strength_decision_sample_rows=decision_samples.height,
        strategy_evidence_sample_rows=sample_rows,
        strategy_evidence_rows=evidence_rows,
        exit_policy_review_sample_rows=exit_policy_samples.height,
        exit_policy_review_summary_rows=exit_policy_summary.height,
        candidate_count=len(
            {
                str(row.get("strategy_candidate") or "")
                for row in summary_frame.to_dicts()
                if str(row.get("strategy_candidate") or "")
            }
        ),
        decision_counts=_decision_counts(summary_frame),
        warnings=warnings,
    )


def build_second_stage_alpha_factory_samples(
    lake_root: str | Path,
    *,
    as_of_date: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[pl.DataFrame, list[str]]:
    samples, _, warnings = _build_second_stage_alpha_factory_tables(
        lake_root,
        as_of_date=as_of_date,
        lookback_days=lookback_days,
    )
    return samples, warnings


def _build_second_stage_alpha_factory_tables(
    lake_root: str | Path,
    *,
    as_of_date: date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    root = Path(lake_root)
    generated_at = datetime.now(UTC)
    lookback_start = datetime.combine(
        as_of_date - timedelta(days=max(lookback_days, 1)),
        time.min,
        tzinfo=UTC,
    )
    label_end = datetime.combine(as_of_date + timedelta(days=1), time.min, tzinfo=UTC)
    market_start = lookback_start - timedelta(hours=max(DEFAULT_HORIZONS))
    market_end = label_end + timedelta(hours=max(DEFAULT_HORIZONS))

    market_bars = _read_recent_dataset(
        root / MARKET_BAR_DATASET,
        timestamp_columns=("ts", "created_at", "ingest_ts"),
        start=market_start,
        end=market_end,
    )
    quality = _latest_day_dataset(read_parquet_dataset(root / EXPANDED_QUALITY_DATASET), as_of_date)
    cost_bucket = read_parquet_dataset(root / COST_BUCKET_DAILY_DATASET)
    regime = _current_regime(read_parquet_dataset(root / MARKET_REGIME_DATASET))
    btc_exit = _read_recent_dataset(
        root / BTC_EXIT_REVIEW_DATASET,
        timestamp_columns=("entry_ts", "created_at", "as_of_date"),
        start=lookback_start,
        end=label_end,
    )
    paper_runs = _read_recent_dataset(
        root / PAPER_RUN_DATASET,
        timestamp_columns=("ts_utc", "created_at", "as_of_date"),
        start=lookback_start,
        end=label_end,
    )

    warnings: list[str] = []
    if market_bars.is_empty():
        warnings.append("market_bar_empty")
    if quality.is_empty():
        warnings.append("expanded_universe_quality_empty")

    cost_context = _cost_context(cost_bucket)
    bars_by_symbol = _bars_by_symbol(market_bars)
    relative_strength_rows, decision_rows = _expanded_relative_strength_tables(
        quality=quality,
        bars_by_symbol=bars_by_symbol,
        cost_context=cost_context,
        generated_at=generated_at,
        regime_state=regime,
    )
    rows: list[dict[str, Any]] = []
    rows.extend(relative_strength_rows)
    rows.extend(
        _futures_short_shadow_samples(
            bars_by_symbol=bars_by_symbol,
            cost_context=cost_context,
            generated_at=generated_at,
            regime_state=regime,
            as_of_date=as_of_date,
        )
    )
    rows.extend(
        _exit_policy_review_samples(
            btc_exit=btc_exit,
            paper_runs=paper_runs,
            generated_at=generated_at,
            regime_state=regime,
        )
    )
    rows.extend(
        _pair_market_neutral_samples(
            bars_by_symbol=bars_by_symbol,
            cost_context=cost_context,
            generated_at=generated_at,
            regime_state=regime,
            as_of_date=as_of_date,
        )
    )
    rows = _dedupe_rows(rows)
    decision_frame = (
        pl.DataFrame(
            decision_rows,
            schema=EXPANDED_RELATIVE_STRENGTH_DECISION_SAMPLE_SCHEMA,
            orient="row",
        )
        if decision_rows
        else pl.DataFrame(schema=EXPANDED_RELATIVE_STRENGTH_DECISION_SAMPLE_SCHEMA)
    )
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA), decision_frame, warnings
    return (
        normalize_strategy_evidence_samples(
            pl.DataFrame(rows, schema=SAMPLE_SCHEMA, orient="row")
        ),
        decision_frame,
        warnings,
    )


def build_exit_policy_review_tables(
    lake_root: str | Path,
    *,
    as_of_date: str | date,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    generated_at: datetime | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    generated = generated_at or datetime.now(UTC)
    lookback_start = datetime.combine(
        day - timedelta(days=max(lookback_days, 1)),
        time.min,
        tzinfo=UTC,
    )
    label_end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=UTC)
    btc_exit = _read_recent_dataset(
        root / BTC_EXIT_REVIEW_DATASET,
        timestamp_columns=("entry_ts", "created_at", "as_of_date"),
        start=lookback_start,
        end=label_end,
    )
    paper_runs = _read_recent_dataset(
        root / PAPER_RUN_DATASET,
        timestamp_columns=("ts_utc", "created_at", "as_of_date"),
        start=lookback_start,
        end=label_end,
    )
    samples = build_exit_policy_review_samples(
        btc_exit=btc_exit,
        paper_runs=paper_runs,
        as_of_date=day,
        generated_at=generated,
    )
    summary = build_exit_policy_review_summary(samples, as_of_date=day, generated_at=generated)
    return samples, summary


def build_exit_policy_review_samples(
    *,
    btc_exit: pl.DataFrame,
    paper_runs: pl.DataFrame,
    as_of_date: date,
    generated_at: datetime,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in btc_exit.to_dicts() if not btc_exit.is_empty() else []:
        entry_ts = _parse_dt(row.get("entry_ts") or row.get("ts_utc") or row.get("created_at"))
        if entry_ts is None:
            continue
        source_entry_id = str(row.get("run_id") or row.get("trade_id") or entry_ts.isoformat())
        rows.append(
            _exit_policy_sample_row(
                as_of_date=as_of_date,
                generated_at=generated_at,
                strategy_id="BTC_STRICT_PROBE_EXIT_POLICY_REVIEW",
                strategy_candidate="v5.btc_strict_probe_exit_policy_review",
                source_entry_id=source_entry_id,
                symbol="BTC-USDT",
                entry_ts=entry_ts,
                actual_exit_net_bps=_first_float(
                    row,
                    ["actual_exit_net_bps", "realized_net_bps", "exit_net_bps", "net_bps"],
                ),
                fixed_hold=_fixed_hold_values(row),
                mfe_bps=_float(row.get("mfe_bps")),
                mae_bps=_float(row.get("mae_bps")),
                exit_reason=str(row.get("exit_reason") or ""),
            )
        )
    for row in paper_runs.to_dicts() if not paper_runs.is_empty() else []:
        strategy_id = str(row.get("strategy_id") or row.get("proposal_id") or "").upper()
        if "ETH" in strategy_id and "F3" in strategy_id:
            candidate = "v5.eth_f3_exit_policy_review"
            symbol = "ETH-USDT"
            strategy = "ETH_F3_EXIT_POLICY_REVIEW"
        elif "SOL" in strategy_id and "PROTECT" in strategy_id:
            candidate = "v5.sol_paper_exit_policy_review"
            symbol = "SOL-USDT"
            strategy = "SOL_PAPER_EXIT_POLICY_REVIEW"
        else:
            continue
        entry_ts = _parse_dt(row.get("ts_utc") or row.get("created_at") or row.get("as_of_date"))
        if entry_ts is None:
            continue
        source_entry_id = str(
            row.get("run_id") or row.get("source_entry_id") or entry_ts.isoformat()
        )
        rows.append(
            _exit_policy_sample_row(
                as_of_date=as_of_date,
                generated_at=generated_at,
                strategy_id=strategy,
                strategy_candidate=candidate,
                source_entry_id=source_entry_id,
                symbol=symbol,
                entry_ts=entry_ts,
                actual_exit_net_bps=_first_float(row, ["actual_exit_net_bps", "paper_pnl_bps"]),
                fixed_hold={
                    horizon: _float(row.get(f"paper_pnl_bps_{horizon}h"))
                    for horizon in EXIT_POLICY_HORIZONS
                },
                mfe_bps=_first_float(
                    row,
                    ["mfe_bps", *[f"mfe_bps_{horizon}h" for horizon in EXIT_POLICY_HORIZONS]],
                ),
                mae_bps=_first_float(
                    row,
                    ["mae_bps", *[f"mae_bps_{horizon}h" for horizon in EXIT_POLICY_HORIZONS]],
                ),
                exit_reason=str(row.get("exit_reason") or "paper_entry_horizon_review"),
            )
        )
    if not rows:
        return pl.DataFrame(schema=EXIT_POLICY_REVIEW_SAMPLE_SCHEMA)
    return pl.DataFrame(rows, schema=EXIT_POLICY_REVIEW_SAMPLE_SCHEMA, orient="row")


def build_exit_policy_review_summary(
    samples: pl.DataFrame,
    *,
    as_of_date: date,
    generated_at: datetime,
) -> pl.DataFrame:
    if samples.is_empty():
        return pl.DataFrame(schema=EXIT_POLICY_REVIEW_SUMMARY_SCHEMA)
    rows: list[dict[str, Any]] = []
    for (strategy_id, candidate, symbol), group in samples.group_by(
        ["strategy_id", "strategy_candidate", "symbol"],
        maintain_order=True,
    ):
        records = group.to_dicts()
        actual_count = sum(
            1 for row in records if _float(row.get("actual_exit_net_bps")) is not None
        )
        stop_loss_too_early = [
            row
            for row in records
            if "stop" in str(row.get("exit_reason") or "").lower()
            and (_float(row.get("delta_vs_actual_bps")) or 0.0) > 0.0
        ]
        hold24_better = [
            row
            for row in records
            if _float(row.get("fixed_hold_24h_net_bps")) is not None
            and _float(row.get("actual_exit_net_bps")) is not None
            and (_float(row.get("fixed_hold_24h_net_bps")) or 0.0)
            > (_float(row.get("actual_exit_net_bps")) or 0.0)
        ]
        delta24 = [
            (_float(row.get("fixed_hold_24h_net_bps")) or 0.0)
            - (_float(row.get("actual_exit_net_bps")) or 0.0)
            for row in hold24_better
        ]
        best_deltas = [
            value
            for row in records
            if (value := _float(row.get("delta_vs_actual_bps"))) is not None
        ]
        decision, reasons = _exit_policy_summary_decision(
            stop_loss_too_early_count=len(stop_loss_too_early),
            avg_delta_hold24h_vs_actual=_mean(delta24),
            actual_exit_count=actual_count,
        )
        rows.append(
            {
                "as_of_date": as_of_date.isoformat(),
                "generated_at": generated_at,
                "schema_version": SCHEMA_VERSION,
                "strategy_id": str(strategy_id),
                "strategy_candidate": str(candidate),
                "symbol": normalize_symbol(symbol) or "UNKNOWN",
                "sample_count": len(records),
                "actual_exit_count": actual_count,
                "stop_loss_too_early_count": len(stop_loss_too_early),
                "hold_24h_better_than_actual_count": len(hold24_better),
                "avg_delta_hold24h_vs_actual": _mean(delta24),
                "avg_delta_best_vs_actual_bps": _mean(best_deltas),
                "best_alternative_exit_policy": _most_common(
                    [
                        str(row.get("best_alternative_exit_policy") or "")
                        for row in records
                        if str(row.get("best_alternative_exit_policy") or "")
                    ]
                ),
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "recommended_mode": "shadow" if decision == "REVIEW_EXIT_POLICY" else "research",
                "source": SOURCE_NAME,
            }
        )
    return pl.DataFrame(rows, schema=EXIT_POLICY_REVIEW_SUMMARY_SCHEMA, orient="row")


def _expanded_relative_strength_tables(
    *,
    quality: pl.DataFrame,
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    cost_context: dict[str, dict[str, Any]],
    generated_at: datetime,
    regime_state: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    quality_context = _quality_context_by_symbol(quality)
    if not quality_context or not bars_by_symbol:
        return [], []
    bar_times_by_symbol = _bar_times_by_symbol(bars_by_symbol)
    output: list[dict[str, Any]] = []
    decision_output: list[dict[str, Any]] = []

    decision_times = sorted(
        {
            ts
            for rows in bars_by_symbol.values()
            for ts in [_parse_dt(row.get("ts")) for row in rows]
            if ts is not None
        }
    )
    for decision_ts in decision_times:
        for lookback_hours in RELATIVE_STRENGTH_LOOKBACK_HOURS:
            ranked = _relative_strength_ranking(
                bars_by_symbol=bars_by_symbol,
                bar_times_by_symbol=bar_times_by_symbol,
                quality_context=quality_context,
                decision_ts=decision_ts,
                lookback_hours=lookback_hours,
            )
            if not ranked:
                continue
            for candidate, top_k in RELATIVE_STRENGTH_SELECTIONS:
                for selected_rank, selection in enumerate(ranked, start=1):
                    symbol = selection["symbol"]
                    current_index = _int(selection.get("current_index"))
                    if current_index is None:
                        continue
                    bars = bars_by_symbol.get(symbol, [])
                    bar_times = bar_times_by_symbol.get(symbol, [])
                    if current_index < 0 or current_index >= len(bars):
                        continue
                    current_bar = bars[current_index]
                    for horizon in DEFAULT_HORIZONS:
                        future_index = _future_bar_index_at_or_after(
                            bar_times,
                            current_index,
                            horizon,
                        )
                        future_bar = bars[future_index] if future_index is not None else None
                        entry_close = _float(current_bar.get("close"))
                        label_close = _float(future_bar.get("close")) if future_bar else None
                        gross_bps = (
                            (label_close / entry_close - 1.0) * 10_000.0
                            if entry_close and label_close
                            else None
                        )
                        cost_bps = max(
                            _cost_bps(cost_context, symbol),
                            CONSERVATIVE_ROUNDTRIP_COST_BPS,
                        )
                        net_bps = gross_bps - cost_bps if gross_bps is not None else None
                        label_ts = _parse_dt(future_bar.get("ts")) if future_bar else None
                        anti_leakage = _relative_strength_anti_leakage_check(
                            decision_ts=decision_ts,
                            lookback_start_ts=selection.get("lookback_start_ts"),
                            label_ts=label_ts,
                        )
                        selected = selected_rank <= top_k
                        decision_output.append(
                            _relative_strength_decision_sample_row(
                                decision_ts=decision_ts,
                                selection=selection,
                                lookback_hours=lookback_hours,
                                top_k=top_k,
                                selected_rank=selected_rank,
                                selected=selected,
                                horizon=horizon,
                                future_net_bps=net_bps,
                                label_status="complete" if label_ts is not None else "pending",
                                anti_leakage=anti_leakage,
                                generated_at=generated_at,
                            )
                        )
                        if not selected or anti_leakage != "pass":
                            continue
                        source_key = (
                            f"{decision_ts.isoformat()}|lookback={lookback_hours}|"
                            f"top_k={top_k}|rank={selected_rank}"
                        )
                        output.append(
                            _sample_row(
                                candidate=candidate,
                                symbol=symbol,
                                source_type=(
                                    "second_stage_expanded_relative_strength_decision_time"
                                ),
                                source_key=source_key,
                                ts_utc=decision_ts,
                                horizon_hours=horizon,
                                decision_ts=decision_ts,
                                label_ts=label_ts,
                                entry_close=entry_close,
                                label_close=label_close,
                                gross_bps=gross_bps,
                                net_bps=net_bps,
                                mfe_bps=_path_mfe_bps_between_indexes(
                                    bars,
                                    current_index,
                                    future_index,
                                ),
                                mae_bps=_path_mae_bps_between_indexes(
                                    bars,
                                    current_index,
                                    future_index,
                                ),
                                win=net_bps is not None and net_bps > 0.0,
                                label_status="complete" if label_ts is not None else "pending",
                                label_reason="decision_time_relative_strength_shadow_only",
                                cost_bps=cost_bps,
                                cost_source=str(
                                    cost_context.get(symbol, {}).get("source")
                                    or cost_context.get(symbol, {}).get("cost_source")
                                    or "conservative_default"
                                ),
                                final_score=selection.get("quality_score"),
                                regime_state=regime_state,
                                generated_at=generated_at,
                                rank_score_bps=selection.get("rank_score_bps"),
                                rank_lookback_hours=lookback_hours,
                                selected_rank=selected_rank,
                                top_k=top_k,
                                selection_reason=_relative_strength_selection_reason(),
                                anti_leakage_check=anti_leakage,
                            )
                        )
    return output, decision_output


def _relative_strength_decision_sample_row(
    *,
    decision_ts: datetime,
    selection: dict[str, Any],
    lookback_hours: int,
    top_k: int,
    selected_rank: int,
    selected: bool,
    horizon: int,
    future_net_bps: float | None,
    label_status: str,
    anti_leakage: str,
    generated_at: datetime,
) -> dict[str, Any]:
    return {
        "decision_ts": decision_ts,
        "symbol": normalize_symbol(selection.get("symbol")) or "UNKNOWN",
        "lookback_hours": int(lookback_hours),
        "lookback_return_bps": _float(selection.get("rank_score_bps")),
        "top_k": int(top_k),
        "selected_rank": int(selected_rank),
        "selected": bool(selected),
        "symbol_quality_score": _float(selection.get("quality_score")),
        "spread_bps": _float(selection.get("spread_bps")),
        "btc_corr": _float(selection.get("btc_corr")),
        "volume_24h_usdt": _float(selection.get("volume_24h_usdt")),
        "selection_reason": _relative_strength_selection_reason(),
        "anti_leakage_check": anti_leakage,
        "label_horizon_hours": int(horizon),
        "future_net_bps": future_net_bps,
        "label_status": label_status,
        "created_at": generated_at,
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE_NAME,
    }


def _futures_short_shadow_samples(
    *,
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    cost_context: dict[str, dict[str, Any]],
    generated_at: datetime,
    regime_state: str,
    as_of_date: date,
) -> list[dict[str, Any]]:
    btc_bars = bars_by_symbol.get("BTC-USDT", [])
    if not btc_bars:
        return []
    output: list[dict[str, Any]] = []
    for candidate in FUTURES_SHADOW_CANDIDATES:
        for horizon in DEFAULT_HORIZONS:
            for idx, bar in enumerate(btc_bars):
                ts_utc = _parse_dt(bar.get("ts"))
                if ts_utc is None or ts_utc.date() > as_of_date:
                    continue
                future = _future_bar(btc_bars, idx, horizon)
                if future is None:
                    continue
                entry = _float(bar.get("close"))
                future_close = _float(future.get("close"))
                if entry is None or future_close is None or entry <= 0:
                    continue
                gross = -((future_close / entry - 1.0) * 10_000.0)
                cost = max(_cost_bps(cost_context, "BTC-USDT"), CONSERVATIVE_ROUNDTRIP_COST_BPS)
                net = gross - cost
                output.append(
                    _sample_row(
                        candidate=candidate,
                        symbol="BTC-USDT",
                        source_type="second_stage_futures_short_spot_inverse_proxy",
                        source_key=f"{candidate}:{ts_utc.isoformat()}:{horizon}",
                        ts_utc=ts_utc,
                        horizon_hours=horizon,
                        decision_ts=ts_utc,
                        label_ts=_parse_dt(future.get("ts")),
                        entry_close=entry,
                        label_close=future_close,
                        gross_bps=gross,
                        net_bps=net,
                        mfe_bps=None,
                        mae_bps=None,
                        win=net > 0,
                        label_status="complete",
                        label_reason=(
                            "spot_inverse_proxy_shadow_only;"
                            "futures_data_missing;"
                            "funding_not_observable"
                        ),
                        cost_bps=cost,
                        cost_source="conservative_roundtrip_proxy",
                        regime_state=regime_state,
                        generated_at=generated_at,
                        futures_data_available=False,
                        funding_available=False,
                        funding_cost_bps=None,
                        mark_price_source="BTC-USDT spot close inverse proxy",
                        liquidation_buffer_pct=None,
                        proxy_warning=(
                            "spot inverse proxy only; no perp mark, funding, "
                            "open_interest, liquidation, or borrow constraints observed"
                        ),
                    )
                )
    return output


def _exit_policy_review_samples(
    *,
    btc_exit: pl.DataFrame,
    paper_runs: pl.DataFrame,
    generated_at: datetime,
    regime_state: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in btc_exit.to_dicts() if not btc_exit.is_empty() else []:
        source_key = str(row.get("run_id") or row.get("trade_id") or row.get("entry_ts") or "")
        entry_ts = _parse_dt(row.get("entry_ts") or row.get("ts_utc") or row.get("created_at"))
        if entry_ts is None:
            continue
        for horizon in DEFAULT_HORIZONS:
            value = _first_float(
                row,
                [
                    f"would_hold_{horizon}h_net_bps",
                    f"would_have_held_{horizon}h_net_bps",
                    f"fixed_hold_{horizon}h_net_bps",
                ],
            )
            if value is None:
                continue
            rows.append(
                _sample_row(
                    candidate="v5.btc_strict_probe_exit_policy_review",
                    symbol="BTC-USDT",
                    source_type="second_stage_exit_policy_review",
                    source_key=f"btc_exit:{source_key}:{horizon}",
                    ts_utc=entry_ts,
                    horizon_hours=horizon,
                    decision_ts=entry_ts,
                    label_ts=entry_ts + timedelta(hours=horizon),
                    entry_close=_float(row.get("entry_px") or row.get("entry_price")),
                    label_close=_float(row.get("label_close") or row.get("exit_px")),
                    gross_bps=value,
                    net_bps=value,
                    mfe_bps=_float(row.get("mfe_bps")),
                    mae_bps=_float(row.get("mae_bps")),
                    win=value > 0,
                    label_status="complete",
                    label_reason=str(row.get("exit_reason") or "exit_policy_review"),
                    cost_bps=0.0,
                    cost_source="actual_exit_review",
                    regime_state=str(row.get("regime_state") or regime_state),
                    generated_at=generated_at,
                )
            )
    for row in paper_runs.to_dicts() if not paper_runs.is_empty() else []:
        strategy_id = str(row.get("strategy_id") or row.get("proposal_id") or "").upper()
        if "ETH" in strategy_id and "F3" in strategy_id:
            candidate = "v5.eth_f3_exit_policy_review"
            symbol = "ETH-USDT"
        elif "SOL" in strategy_id and "PROTECT" in strategy_id:
            candidate = "v5.sol_paper_exit_policy_review"
            symbol = "SOL-USDT"
        else:
            continue
        ts_utc = _parse_dt(row.get("ts_utc") or row.get("created_at") or row.get("as_of_date"))
        if ts_utc is None:
            continue
        for horizon in DEFAULT_HORIZONS:
            value = _float(row.get(f"paper_pnl_bps_{horizon}h"))
            if value is None:
                continue
            rows.append(
                _sample_row(
                    candidate=candidate,
                    symbol=symbol,
                    source_type="second_stage_exit_policy_review",
                    source_key=f"{strategy_id}:{row.get('run_id') or ts_utc.isoformat()}:{horizon}",
                    ts_utc=ts_utc,
                    horizon_hours=horizon,
                    decision_ts=ts_utc,
                    label_ts=ts_utc + timedelta(hours=horizon),
                    entry_close=None,
                    label_close=None,
                    gross_bps=value,
                    net_bps=value,
                    mfe_bps=_float(row.get(f"mfe_bps_{horizon}h") or row.get("mfe_bps")),
                    mae_bps=_float(row.get(f"mae_bps_{horizon}h") or row.get("mae_bps")),
                    win=value > 0,
                    label_status="complete",
                    label_reason="paper_exit_policy_review_shadow_only",
                    cost_bps=0.0,
                    cost_source=str(row.get("cost_source") or "paper_telemetry"),
                    regime_state=str(row.get("regime_state") or regime_state),
                    generated_at=generated_at,
                )
            )
    return rows


def _fixed_hold_values(row: dict[str, Any]) -> dict[int, float | None]:
    return {
        horizon: _first_float(
            row,
            [
                f"fixed_hold_{horizon}h_net_bps",
                f"would_hold_{horizon}h_net_bps",
                f"would_have_held_{horizon}h_net_bps",
            ],
        )
        for horizon in EXIT_POLICY_HORIZONS
    }


def _exit_policy_sample_row(
    *,
    as_of_date: date,
    generated_at: datetime,
    strategy_id: str,
    strategy_candidate: str,
    source_entry_id: str,
    symbol: str,
    entry_ts: datetime,
    actual_exit_net_bps: float | None,
    fixed_hold: dict[int, float | None],
    mfe_bps: float | None,
    mae_bps: float | None,
    exit_reason: str,
) -> dict[str, Any]:
    alternatives = {
        f"fixed_hold_{horizon}h": value
        for horizon, value in fixed_hold.items()
        if value is not None
    }
    best_policy = ""
    best_value = None
    if alternatives:
        best_policy, best_value = max(alternatives.items(), key=lambda item: item[1])
    delta = (
        best_value - actual_exit_net_bps
        if best_value is not None and actual_exit_net_bps is not None
        else None
    )
    decision, reasons = _exit_policy_sample_decision(
        actual_exit_net_bps=actual_exit_net_bps,
        fixed_hold_24h=fixed_hold.get(24),
        delta_vs_actual_bps=delta,
        exit_reason=exit_reason,
    )
    return {
        "as_of_date": as_of_date.isoformat(),
        "generated_at": generated_at,
        "schema_version": SCHEMA_VERSION,
        "strategy_id": strategy_id,
        "strategy_candidate": strategy_candidate,
        "source_entry_id": source_entry_id,
        "symbol": normalize_symbol(symbol) or "UNKNOWN",
        "entry_ts": entry_ts,
        "actual_exit_net_bps": actual_exit_net_bps,
        "fixed_hold_4h_net_bps": fixed_hold.get(4),
        "fixed_hold_8h_net_bps": fixed_hold.get(8),
        "fixed_hold_12h_net_bps": fixed_hold.get(12),
        "fixed_hold_24h_net_bps": fixed_hold.get(24),
        "fixed_hold_48h_net_bps": fixed_hold.get(48),
        "best_alternative_exit_policy": best_policy,
        "best_alternative_net_bps": best_value,
        "delta_vs_actual_bps": delta,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "exit_reason": exit_reason,
        "decision": decision,
        "decision_reasons": safe_json_dumps(reasons),
        "recommended_mode": "shadow" if decision == "REVIEW_EXIT_POLICY" else "research",
        "source": SOURCE_NAME,
    }


def _exit_policy_sample_decision(
    *,
    actual_exit_net_bps: float | None,
    fixed_hold_24h: float | None,
    delta_vs_actual_bps: float | None,
    exit_reason: str,
) -> tuple[str, list[str]]:
    reasons = ["research_only_no_live_exit_change"]
    if actual_exit_net_bps is None:
        return "RESEARCH_ONLY", [*reasons, "actual_exit_not_observable"]
    if delta_vs_actual_bps is None:
        return "RESEARCH_ONLY", [*reasons, "alternative_exit_not_observable"]
    if (
        "stop" in str(exit_reason or "").lower()
        and fixed_hold_24h is not None
        and fixed_hold_24h > actual_exit_net_bps
    ):
        return "REVIEW_EXIT_POLICY", [
            *reasons,
            "stop_loss_exit_underperformed_hold_24h",
        ]
    if delta_vs_actual_bps > 0.0:
        return "REVIEW_EXIT_POLICY", [*reasons, "alternative_exit_better_than_actual"]
    return "RESEARCH_ONLY", [*reasons, "actual_exit_not_worse_than_alternatives"]


def _exit_policy_summary_decision(
    *,
    stop_loss_too_early_count: int,
    avg_delta_hold24h_vs_actual: float | None,
    actual_exit_count: int,
) -> tuple[str, list[str]]:
    reasons = ["research_only_no_live_exit_change"]
    if actual_exit_count <= 0:
        return "RESEARCH_ONLY", [*reasons, "actual_exit_not_observable"]
    if stop_loss_too_early_count > 0 and (avg_delta_hold24h_vs_actual or 0.0) > 0.0:
        return "REVIEW_EXIT_POLICY", [
            *reasons,
            "hold_24h_better_than_actual_observed",
            "requires_more_samples_before_v5_change",
        ]
    return "RESEARCH_ONLY", [*reasons, "no_systematic_exit_policy_issue_confirmed"]


def _pair_market_neutral_samples(
    *,
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    cost_context: dict[str, dict[str, Any]],
    generated_at: datetime,
    regime_state: str,
    as_of_date: date,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    configs = (
        ("v5.pair_trade_eth_btc_shadow", "ETH-BTC", ("ETH-USDT",), ("BTC-USDT",)),
        ("v5.pair_trade_sol_eth_shadow", "SOL-ETH", ("SOL-USDT",), ("ETH-USDT",)),
        (
            "v5.alt_basket_vs_btc_shadow",
            "ALT-BASKET-BTC",
            ("ETH-USDT", "SOL-USDT", "BNB-USDT"),
            ("BTC-USDT",),
        ),
    )
    for candidate, synthetic_symbol, long_symbols, short_symbols in configs:
        common_length = min(
            [len(bars_by_symbol.get(symbol, [])) for symbol in [*long_symbols, *short_symbols]]
            or [0]
        )
        if common_length <= max(DEFAULT_HORIZONS):
            continue
        for horizon in DEFAULT_HORIZONS:
            for idx in range(0, common_length - horizon):
                base_bar = bars_by_symbol[long_symbols[0]][idx]
                ts_utc = _parse_dt(base_bar.get("ts"))
                if ts_utc is None or ts_utc.date() > as_of_date:
                    continue
                long_return = _basket_return_bps(bars_by_symbol, long_symbols, idx, horizon)
                short_return = _basket_return_bps(bars_by_symbol, short_symbols, idx, horizon)
                if long_return is None or short_return is None:
                    continue
                gross = long_return - short_return
                pair_symbols = [*long_symbols, *short_symbols]
                pair_cost_bps = sum(_cost_bps(cost_context, symbol) for symbol in pair_symbols)
                cost = max(pair_cost_bps, CONSERVATIVE_ROUNDTRIP_COST_BPS)
                net = gross - cost
                output.append(
                    _sample_row(
                        candidate=candidate,
                        symbol=synthetic_symbol,
                        source_type="second_stage_pair_market_neutral",
                        source_key=f"{candidate}:{ts_utc.isoformat()}:{horizon}",
                        ts_utc=ts_utc,
                        horizon_hours=horizon,
                        decision_ts=ts_utc,
                        label_ts=ts_utc + timedelta(hours=horizon),
                        entry_close=None,
                        label_close=None,
                        gross_bps=gross,
                        net_bps=net,
                        mfe_bps=None,
                        mae_bps=None,
                        win=net > 0,
                        label_status="complete",
                        label_reason="pair_market_neutral_shadow_only",
                        cost_bps=cost,
                        cost_source="conservative_pair_cost_proxy",
                        regime_state=regime_state,
                        generated_at=generated_at,
                    )
                )
    return output


def _sample_row(
    *,
    candidate: str,
    symbol: str,
    source_type: str,
    source_key: str,
    ts_utc: datetime | None,
    horizon_hours: int,
    decision_ts: datetime | None,
    label_ts: datetime | None,
    entry_close: float | None,
    label_close: float | None,
    gross_bps: float | None,
    net_bps: float | None,
    mfe_bps: float | None,
    mae_bps: float | None,
    win: bool | None,
    label_status: str,
    label_reason: str,
    cost_bps: float,
    cost_source: str,
    generated_at: datetime,
    regime_state: str = "UNKNOWN",
    final_score: float | None = None,
    rank_score_bps: float | None = None,
    rank_lookback_hours: int | None = None,
    selected_rank: int | None = None,
    top_k: int | None = None,
    selection_reason: str = "",
    anti_leakage_check: str = "",
    futures_data_available: bool | None = None,
    funding_available: bool | None = None,
    funding_cost_bps: float | None = None,
    mark_price_source: str = "",
    liquidation_buffer_pct: float | None = None,
    proxy_warning: str = "",
) -> dict[str, Any]:
    ts = ts_utc or generated_at
    normalized_symbol = normalize_symbol(symbol) or "UNKNOWN"
    event_key = f"{source_type}:{candidate}:{normalized_symbol}:{source_key}:{horizon_hours}"
    return {
        "strategy": "v5",
        "evidence_version": EVIDENCE_VERSION,
        "as_of_date": ts.date().isoformat(),
        "candidate_id": event_key,
        "run_id": "",
        "ts_utc": ts,
        "symbol": normalized_symbol,
        "strategy_candidate": candidate,
        "candidate_name": candidate,
        "source_type": source_type,
        "sample_count": 1,
        "complete_sample_count": 1 if str(label_status).lower() == "complete" else 0,
        "regime_state": regime_state or "UNKNOWN",
        "horizon_hours": int(horizon_hours),
        "decision_ts": decision_ts,
        "label_ts": label_ts,
        "entry_close": entry_close,
        "label_close": label_close,
        "gross_bps": gross_bps,
        "net_bps_after_cost": net_bps,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "win": win,
        "label_status": label_status,
        "label_reason": label_reason,
        "cost_bps": cost_bps,
        "cost_source": cost_source or "unknown",
        "block_reason": "second_stage_shadow_only",
        "final_decision": "SHADOW_RESEARCH",
        "final_score": final_score,
        "rank_score_bps": rank_score_bps,
        "rank_lookback_hours": rank_lookback_hours,
        "selected_rank": selected_rank,
        "top_k": top_k,
        "selection_reason": selection_reason,
        "anti_leakage_check": anti_leakage_check,
        "futures_data_available": futures_data_available,
        "funding_available": funding_available,
        "funding_cost_bps": funding_cost_bps,
        "mark_price_source": mark_price_source,
        "liquidation_buffer_pct": liquidation_buffer_pct,
        "proxy_warning": proxy_warning,
        "expected_edge_bps": net_bps,
        "required_edge_bps": cost_bps,
        "alpha6_score": None,
        "alpha6_side": "",
        "protect_level": "",
        "risk_level": "",
        "btc_trend_state": "",
        "broad_market_positive_count": None,
        "funding_state": "not_observable",
        "volatility_bucket": "",
        "source_path_inside_bundle": "",
        "source_event_key": event_key,
        "source_bundle_ts": None,
        "created_at": generated_at,
        "source": SOURCE_NAME,
    }


def _cap_second_stage_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    capped: list[dict[str, Any]] = []
    for row in rows:
        candidate = str(row.get("strategy_candidate") or "")
        if candidate in SECOND_STAGE_CANDIDATES:
            reasons = _json_list(row.get("decision_reasons"))
            reasons.extend(["second_stage_factory_read_only", "max_live_notional_zero"])
            row = {
                **row,
                "decision_reasons": safe_json_dumps(_dedupe_text(reasons)),
            }
        if candidate in SECOND_STAGE_CANDIDATES and row.get("decision") == "LIVE_SMALL_READY":
            reasons = _json_list(row.get("decision_reasons"))
            reasons.extend(["second_stage_factory_live_disabled", "shadow_or_paper_only"])
            row = {
                **row,
                "decision": "PAPER_READY",
                "decision_reasons": safe_json_dumps(_dedupe_text(reasons)),
            }
        elif candidate in {*FUTURES_SHADOW_CANDIDATES, *PAIR_SHADOW_CANDIDATES}:
            decision = str(row.get("decision") or "")
            if decision == "PAPER_READY":
                reasons = _json_list(row.get("decision_reasons"))
                reasons.extend(["shadow_only", "not_live_validated"])
                row = {
                    **row,
                    "decision": "KEEP_SHADOW",
                    "decision_reasons": safe_json_dumps(_dedupe_text(reasons)),
                }
        if candidate in FUTURES_SHADOW_CANDIDATES:
            reasons = _json_list(row.get("decision_reasons"))
            reasons.extend(
                [
                    "spot_inverse_proxy_only",
                    "futures_data_missing",
                    "funding_not_observable",
                ]
            )
            decision = str(row.get("decision") or "")
            row = {
                **row,
                "decision": "KEEP_SHADOW" if decision == "PAPER_READY" else decision,
                "decision_reasons": safe_json_dumps(_dedupe_text(reasons)),
            }
        capped.append(row)
    return capped


def _publish_second_stage_samples(root: Path, frame: pl.DataFrame) -> int:
    dataset = root / STRATEGY_EVIDENCE_SAMPLE_DATASET
    if frame.is_empty():
        return count_parquet_rows(dataset)
    return upsert_parquet_dataset(
        frame,
        dataset,
        key_columns=[
            "strategy",
            "source_type",
            "candidate_id",
            "symbol",
            "strategy_candidate",
            "horizon_hours",
            "source_event_key",
        ],
        streaming_upsert=True,
    )


def _publish_second_stage_summaries(root: Path, frame: pl.DataFrame, day: date) -> int:
    dataset = root / STRATEGY_EVIDENCE_DATASET
    existing = read_parquet_dataset(dataset)
    normalized = normalize_strategy_evidence_decisions(frame)
    if existing.is_empty():
        write_parquet_dataset(normalized, dataset)
        return normalized.height
    retained = existing
    if "as_of_date" in retained.columns and "strategy_candidate" in retained.columns:
        retained = retained.filter(
            ~(
                (pl.col("as_of_date").cast(pl.Utf8) == day.isoformat())
                & pl.col("strategy_candidate").cast(pl.Utf8).is_in(sorted(SECOND_STAGE_CANDIDATES))
            )
        )
    combined = pl.concat([retained, normalized], how="diagonal_relaxed")
    combined = normalize_strategy_evidence_decisions(combined)
    write_parquet_dataset(combined, dataset)
    return combined.height


def _summary_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=SUMMARY_SCHEMA)
    return pl.DataFrame(rows, schema=SUMMARY_SCHEMA, orient="row")


def _read_recent_dataset(
    dataset_path: Path,
    *,
    timestamp_columns: tuple[str, ...],
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    try:
        lazy = read_parquet_lazy(dataset_path)
        columns = lazy.collect_schema().names()
    except Exception:
        return pl.DataFrame()
    timestamp_column = next((column for column in timestamp_columns if column in columns), None)
    if timestamp_column is None:
        try:
            return lazy.collect()
        except Exception:
            return read_parquet_dataset(dataset_path)
    try:
        expr = pl.col(timestamp_column).cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)
        return lazy.filter(expr.is_between(start, end, closed="left")).collect()
    except Exception:
        frame = read_parquet_dataset(dataset_path)
        if frame.is_empty() or timestamp_column not in frame.columns:
            return frame
        return frame.filter(
            pl.col(timestamp_column)
            .cast(pl.Utf8)
            .str.to_datetime(time_zone="UTC", strict=False)
            .is_between(start, end, closed="left")
        )


def _latest_day_dataset(frame: pl.DataFrame, as_of_date: date) -> pl.DataFrame:
    if frame.is_empty() or "as_of_date" not in frame.columns:
        return frame
    text = pl.col("as_of_date").cast(pl.Utf8).str.slice(0, 10)
    eligible = frame.filter(text <= as_of_date.isoformat())
    if eligible.is_empty():
        return frame
    latest = eligible.select(text.max().alias("latest")).item(0, "latest")
    return eligible.filter(text == str(latest))


def _current_regime(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "UNKNOWN"
    rows = frame.to_dicts()
    rows.sort(key=lambda row: str(row.get("as_of_date") or row.get("created_at") or ""))
    return str(rows[-1].get("current_regime") or rows[-1].get("regime_state") or "UNKNOWN")


def _bars_by_symbol(frame: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if frame.is_empty():
        return {}
    for row in frame.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        ts = _parse_dt(row.get("ts"))
        if not symbol or ts is None:
            continue
        row = dict(row)
        row["ts"] = ts
        by_symbol[symbol].append(row)
    for rows in by_symbol.values():
        rows.sort(key=lambda row: _parse_dt(row.get("ts")) or datetime.min.replace(tzinfo=UTC))
    return dict(by_symbol)


def _bar_times_by_symbol(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
) -> dict[str, list[datetime]]:
    return {
        symbol: [ts for row in rows if (ts := _parse_dt(row.get("ts"))) is not None]
        for symbol, rows in bars_by_symbol.items()
    }


def _future_bar(
    rows: list[dict[str, Any]],
    index: int,
    horizon_hours: int,
) -> dict[str, Any] | None:
    if index < 0 or index >= len(rows):
        return None
    start_ts = _parse_dt(rows[index].get("ts"))
    if start_ts is None:
        return None
    target = start_ts + timedelta(hours=horizon_hours)
    for row in rows[index + 1 :]:
        ts = _parse_dt(row.get("ts"))
        if ts is not None and ts >= target:
            return row
    return None


def _basket_return_bps(
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    symbols: tuple[str, ...],
    index: int,
    horizon_hours: int,
) -> float | None:
    values: list[float] = []
    for symbol in symbols:
        rows = bars_by_symbol.get(symbol, [])
        if index >= len(rows):
            return None
        future = _future_bar(rows, index, horizon_hours)
        if future is None:
            return None
        entry = _float(rows[index].get("close"))
        close = _float(future.get("close"))
        if entry is None or close is None or entry <= 0:
            return None
        values.append((close / entry - 1.0) * 10_000.0)
    return statistics.fmean(values) if values else None


def _cost_context(frame: pl.DataFrame) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    if frame.is_empty() or "symbol" not in frame.columns:
        return output
    rows = frame.to_dicts()
    rows.sort(key=lambda row: str(row.get("day") or row.get("created_at") or ""))
    for row in rows:
        symbol = normalize_symbol(row.get("symbol"))
        if symbol:
            output[symbol] = row
    return output


def _cost_bps(cost_context: dict[str, dict[str, Any]], symbol: str) -> float:
    row = cost_context.get(normalize_symbol(symbol), {})
    for key in [
        "roundtrip_all_in_cost_bps",
        "total_cost_bps_p75",
        "selected_total_cost_bps",
        "cost_bps",
    ]:
        value = _float(row.get(key))
        if value is not None:
            return max(value, 0.0)
    return CONSERVATIVE_ROUNDTRIP_COST_BPS


def _quality_score_by_symbol(frame: pl.DataFrame) -> dict[str, float]:
    if frame.is_empty():
        return {}
    output: dict[str, float] = {}
    for row in frame.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        value = _float(row.get("symbol_quality_score") or row.get("quality_score"))
        if symbol and value is not None:
            output[symbol] = value
    return output


def _quality_context_by_symbol(frame: pl.DataFrame) -> dict[str, dict[str, float]]:
    if frame.is_empty():
        return {}
    output: dict[str, dict[str, float]] = {}
    for row in frame.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        quality = _float(row.get("symbol_quality_score") or row.get("quality_score"))
        if not symbol or quality is None:
            continue
        output[symbol] = {
            "quality_score": quality,
            "spread_bps": _first_float(
                row,
                [
                    "avg_spread_bps",
                    "spread_bps_p75",
                    "spread_bps",
                    "avg_spread",
                ],
            )
            or 0.0,
            "btc_corr": _first_float(
                row,
                ["btc_correlation", "btc_corr", "correlation_to_btc"],
            )
            or 0.0,
            "volume_24h_usdt": _first_float(
                row,
                [
                    "volume_24h_usdt",
                    "quote_volume_24h",
                    "quote_volume_24h_usdt",
                    "volume_usdt_24h",
                ],
            )
            or 0.0,
        }
    return output


def _relative_strength_ranking(
    *,
    bars_by_symbol: dict[str, list[dict[str, Any]]],
    bar_times_by_symbol: dict[str, list[datetime]],
    quality_context: dict[str, dict[str, float]],
    decision_ts: datetime,
    lookback_hours: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol, quality in quality_context.items():
        if not _passes_relative_strength_filter(quality):
            continue
        bars = bars_by_symbol.get(symbol, [])
        bar_times = bar_times_by_symbol.get(symbol, [])
        current_index = _bar_index_at_or_before_times(bar_times, decision_ts)
        if current_index is None:
            continue
        lookback_index = _bar_index_at_or_before_times(
            bar_times,
            decision_ts - timedelta(hours=lookback_hours),
        )
        if lookback_index is None or lookback_index >= current_index:
            continue
        current_close = _float(bars[current_index].get("close"))
        lookback_close = _float(bars[lookback_index].get("close"))
        if current_close is None or lookback_close is None or lookback_close <= 0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "rank_score_bps": (current_close / lookback_close - 1.0) * 10_000.0,
                "quality_score": quality["quality_score"],
                "spread_bps": quality["spread_bps"],
                "btc_corr": quality["btc_corr"],
                "volume_24h_usdt": quality["volume_24h_usdt"],
                "lookback_start_ts": _parse_dt(bars[lookback_index].get("ts")),
                "current_index": current_index,
                "lookback_index": lookback_index,
            }
        )
    rows.sort(
        key=lambda row: (
            float(row.get("rank_score_bps") or -1e9),
            float(row.get("quality_score") or 0.0),
            str(row.get("symbol") or ""),
        ),
        reverse=True,
    )
    return rows


def _passes_relative_strength_filter(quality: dict[str, float]) -> bool:
    return (
        quality.get("quality_score", 0.0) >= RELATIVE_STRENGTH_FILTER["min_quality_score"]
        and quality.get("spread_bps", 0.0) <= RELATIVE_STRENGTH_FILTER["max_spread_bps"]
        and abs(quality.get("btc_corr", 0.0)) <= RELATIVE_STRENGTH_FILTER["btc_corr_max"]
        and quality.get("volume_24h_usdt", 0.0)
        >= RELATIVE_STRENGTH_FILTER["min_24h_volume_usdt"]
    )


def _bar_index_at_or_before(
    bars: list[dict[str, Any]],
    ts: datetime,
) -> int | None:
    selected: int | None = None
    for index, row in enumerate(bars):
        row_ts = _parse_dt(row.get("ts"))
        if row_ts is None:
            continue
        if row_ts <= ts:
            selected = index
        else:
            break
    return selected


def _bar_index_at_or_before_times(
    times: list[datetime],
    ts: datetime,
) -> int | None:
    index = bisect_right(times, ts) - 1
    return index if index >= 0 else None


def _future_bar_index_at_or_after(
    times: list[datetime],
    index: int,
    horizon_hours: int,
) -> int | None:
    if index < 0 or index >= len(times):
        return None
    target = times[index] + timedelta(hours=horizon_hours)
    future_index = bisect_left(times, target, index + 1)
    return future_index if future_index < len(times) else None


def _path_mfe_bps_between_indexes(
    bars: list[dict[str, Any]],
    index: int,
    future_index: int | None,
) -> float | None:
    entry = _float(bars[index].get("close")) if 0 <= index < len(bars) else None
    if entry is None or entry <= 0 or future_index is None or future_index <= index:
        return None
    clean = [
        value
        for value in (
            _float(row.get("high") or row.get("close"))
            for row in bars[index + 1 : future_index + 1]
        )
        if value is not None
    ]
    return ((max(clean) / entry - 1.0) * 10_000.0) if clean else None


def _path_mae_bps_between_indexes(
    bars: list[dict[str, Any]],
    index: int,
    future_index: int | None,
) -> float | None:
    entry = _float(bars[index].get("close")) if 0 <= index < len(bars) else None
    if entry is None or entry <= 0 or future_index is None or future_index <= index:
        return None
    clean = [
        value
        for value in (
            _float(row.get("low") or row.get("close"))
            for row in bars[index + 1 : future_index + 1]
        )
        if value is not None
    ]
    return ((min(clean) / entry - 1.0) * 10_000.0) if clean else None


def _path_mfe_bps(
    bars: list[dict[str, Any]],
    index: int,
    horizon_hours: int,
) -> float | None:
    entry = _float(bars[index].get("close")) if 0 <= index < len(bars) else None
    if entry is None or entry <= 0:
        return None
    future = _future_bar(bars, index, horizon_hours)
    future_ts = _parse_dt(future.get("ts")) if future else None
    if future_ts is None:
        return None
    highs = [
        _float(row.get("high") or row.get("close"))
        for row in bars[index + 1 :]
        if (_parse_dt(row.get("ts")) or datetime.max.replace(tzinfo=UTC)) <= future_ts
    ]
    clean = [value for value in highs if value is not None]
    return ((max(clean) / entry - 1.0) * 10_000.0) if clean else None


def _path_mae_bps(
    bars: list[dict[str, Any]],
    index: int,
    horizon_hours: int,
) -> float | None:
    entry = _float(bars[index].get("close")) if 0 <= index < len(bars) else None
    if entry is None or entry <= 0:
        return None
    future = _future_bar(bars, index, horizon_hours)
    future_ts = _parse_dt(future.get("ts")) if future else None
    if future_ts is None:
        return None
    lows = [
        _float(row.get("low") or row.get("close"))
        for row in bars[index + 1 :]
        if (_parse_dt(row.get("ts")) or datetime.max.replace(tzinfo=UTC)) <= future_ts
    ]
    clean = [value for value in lows if value is not None]
    return ((min(clean) / entry - 1.0) * 10_000.0) if clean else None


def _relative_strength_anti_leakage_check(
    *,
    decision_ts: datetime,
    lookback_start_ts: Any,
    label_ts: datetime | None,
) -> str:
    lookback_ts = _parse_dt(lookback_start_ts)
    if lookback_ts is None or lookback_ts > decision_ts:
        return "fail"
    if label_ts is None or label_ts <= decision_ts:
        return "fail"
    return "pass"


def _relative_strength_selection_reason() -> str:
    return (
        "decision_time_relative_strength_rank;"
        f"min_quality_score={RELATIVE_STRENGTH_FILTER['min_quality_score']};"
        f"max_spread_bps={RELATIVE_STRENGTH_FILTER['max_spread_bps']};"
        f"btc_corr_max={RELATIVE_STRENGTH_FILTER['btc_corr_max']};"
        f"min_24h_volume_usdt={RELATIVE_STRENGTH_FILTER['min_24h_volume_usdt']}"
    )


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
        return datetime.combine(value, time.min, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 10 and text[4] == "-":
        return datetime.combine(date.fromisoformat(text), time.min, tzinfo=UTC)
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
    number = _float(value)
    return int(number) if number is not None else None


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


def _mean(values: Any) -> float | None:
    clean = [value for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


def _most_common(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        text = str(value).strip()
        if text:
            counts[text] = counts.get(text, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


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
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keyed: dict[tuple[str, str, str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("source_type") or ""),
            str(row.get("strategy_candidate") or ""),
            str(row.get("symbol") or ""),
            int(row.get("horizon_hours") or 0),
            str(row.get("source_event_key") or row.get("candidate_id") or ""),
        )
        keyed[key] = row
    return list(keyed.values())


def _decision_counts(frame: pl.DataFrame) -> dict[str, int]:
    if frame.is_empty() or "decision" not in frame.columns:
        return {}
    return {
        str(row["decision"]): int(row["count"])
        for row in frame.group_by("decision").len(name="count").to_dicts()
    }
