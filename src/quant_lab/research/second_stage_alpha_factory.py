from __future__ import annotations

import statistics
from collections import defaultdict
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import (
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
STRATEGY_EVIDENCE_SAMPLE_DATASET = Path("gold") / "strategy_evidence_sample"
STRATEGY_EVIDENCE_DATASET = Path("gold") / "strategy_evidence"

MARKET_BAR_DATASET = Path("silver") / "market_bar"
EXPANDED_LABEL_DATASET = Path("gold") / "expanded_universe_candidate_label"
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

DEFAULT_HORIZONS = (4, 8, 12, 24, 48)
DEFAULT_LOOKBACK_DAYS = 30
CONSERVATIVE_ROUNDTRIP_COST_BPS = 30.0


class SecondStageAlphaFactoryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: str
    as_of_date: str
    sample_rows: int = Field(ge=0)
    summary_rows: int = Field(ge=0)
    strategy_evidence_sample_rows: int = Field(ge=0)
    strategy_evidence_rows: int = Field(ge=0)
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
    samples, warnings = build_second_stage_alpha_factory_samples(
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
    sample_rows = _publish_second_stage_samples(root, sample_frame)
    evidence_rows = _publish_second_stage_summaries(root, summary_frame, day)

    return SecondStageAlphaFactoryResult(
        lake_root=str(root),
        as_of_date=day.isoformat(),
        sample_rows=sample_frame.height,
        summary_rows=summary_frame.height,
        strategy_evidence_sample_rows=sample_rows,
        strategy_evidence_rows=evidence_rows,
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
    expanded_labels = _read_recent_dataset(
        root / EXPANDED_LABEL_DATASET,
        timestamp_columns=("ts_utc", "decision_ts", "generated_at"),
        start=lookback_start,
        end=label_end,
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
    if expanded_labels.is_empty():
        warnings.append("expanded_universe_candidate_label_empty")
    if quality.is_empty():
        warnings.append("expanded_universe_quality_empty")

    cost_context = _cost_context(cost_bucket)
    bars_by_symbol = _bars_by_symbol(market_bars)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _expanded_relative_strength_samples(
            expanded_labels=expanded_labels,
            quality=quality,
            generated_at=generated_at,
            regime_state=regime,
        )
    )
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
    if not rows:
        return pl.DataFrame(schema=SAMPLE_SCHEMA), warnings
    return normalize_strategy_evidence_samples(
        pl.DataFrame(rows, schema=SAMPLE_SCHEMA, orient="row")
    ), warnings


def _expanded_relative_strength_samples(
    *,
    expanded_labels: pl.DataFrame,
    quality: pl.DataFrame,
    generated_at: datetime,
    regime_state: str,
) -> list[dict[str, Any]]:
    if expanded_labels.is_empty():
        return []
    labels = [
        row
        for row in expanded_labels.to_dicts()
        if _int(row.get("horizon_hours")) in DEFAULT_HORIZONS
        and str(row.get("label_status") or "").lower() == "complete"
        and normalize_symbol(row.get("symbol"))
    ]
    if not labels:
        return []
    quality_by_symbol = _quality_score_by_symbol(quality)
    groups: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in labels:
        groups[int(row.get("horizon_hours") or 0)][normalize_symbol(row.get("symbol"))].append(row)

    output: list[dict[str, Any]] = []
    for horizon, by_symbol in groups.items():
        ranked = sorted(
            by_symbol.items(),
            key=lambda item: (
                _mean(_float(row.get("net_bps_after_cost")) for row in item[1]) or -1e9,
                quality_by_symbol.get(item[0], 0.0),
            ),
            reverse=True,
        )
        top1 = {symbol for symbol, _rows in ranked[:1]}
        top3 = {symbol for symbol, _rows in ranked[:3]}
        selections = (
            ("v5.expanded_relative_strength_top1_shadow", top1),
            ("v5.expanded_relative_strength_top3_shadow", top3),
            ("v5.expanded_relative_strength_rotation_shadow", top3),
        )
        for candidate, selected_symbols in selections:
            for symbol in selected_symbols:
                for label in by_symbol.get(symbol, []):
                    output.append(
                        _sample_row(
                            candidate=candidate,
                            symbol=symbol,
                            source_type="second_stage_expanded_relative_strength",
                            source_key=str(label.get("candidate_id") or ""),
                            ts_utc=_parse_dt(label.get("ts_utc")),
                            horizon_hours=horizon,
                            decision_ts=_parse_dt(label.get("decision_ts") or label.get("ts_utc")),
                            label_ts=_parse_dt(label.get("label_ts")),
                            entry_close=_float(label.get("entry_close")),
                            label_close=_float(label.get("label_close")),
                            gross_bps=_float(label.get("gross_bps")),
                            net_bps=_float(label.get("net_bps_after_cost")),
                            mfe_bps=_float(label.get("mfe_bps")),
                            mae_bps=_float(label.get("mae_bps")),
                            win=_bool(label.get("win")),
                            label_status="complete",
                            label_reason="expanded_relative_strength_shadow_only",
                            cost_bps=_float(label.get("cost_bps")) or 0.0,
                            cost_source=str(label.get("cost_source") or "unknown"),
                            final_score=quality_by_symbol.get(symbol),
                            regime_state=str(label.get("regime_state") or regime_state),
                            generated_at=generated_at,
                        )
                    )
    return output


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
                        source_type="second_stage_futures_short_proxy",
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
                        label_reason="futures_proxy_shadow_only;funding_not_observable",
                        cost_bps=cost,
                        cost_source="conservative_roundtrip_proxy",
                        regime_state=regime_state,
                        generated_at=generated_at,
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
        capped.append(row)
    return capped


def _publish_second_stage_samples(root: Path, frame: pl.DataFrame) -> int:
    if frame.is_empty():
        return read_parquet_dataset(root / STRATEGY_EVIDENCE_SAMPLE_DATASET).height
    return upsert_parquet_dataset(
        frame,
        root / STRATEGY_EVIDENCE_SAMPLE_DATASET,
        key_columns=[
            "strategy",
            "source_type",
            "candidate_id",
            "symbol",
            "strategy_candidate",
            "horizon_hours",
            "source_event_key",
        ],
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
