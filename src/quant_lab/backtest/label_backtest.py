from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import polars as pl

from quant_lab.backtest.datasets import (
    HORIZONS,
    coerce_dt,
    first_float,
    first_value,
    float_or_none,
    iso_utc,
    normalize_strategy_symbol,
    rows,
)
from quant_lab.backtest.metrics import frame_with_schema, recent_7d_avg_net_bps, summarize_net_bps

BACKTEST_LABEL_SUMMARY_FIELDS = [
    "strategy_id",
    "symbol",
    "regime",
    "horizon_hours",
    "dedupe_before_rows",
    "dedupe_after_rows",
    "duplicate_rate",
    "sample_count",
    "complete_sample_count",
    "avg_net_bps",
    "median_net_bps",
    "p25_net_bps",
    "p10_net_bps",
    "win_rate",
    "max_loss_bps",
    "cost_model",
    "data_leakage_check",
    "recommendation",
]

LABEL_SOURCE_DATASETS = (
    "strategy_opportunity_advisory",
    "v5_candidate_label",
    "risk_on_multi_buy_shadow",
    "final_score_vs_alpha6_conflict",
    "v5_final_score_vs_alpha6_conflict",
    "bnb_strong_alpha6_bypass_shadow",
    "v5_bnb_strong_alpha6_bypass_shadow",
    "expanded_universe_candidate_label",
    "expanded_universe_candidate_maturity",
    "post_impulse_overextension_shadow",
    "bottom_zone_reversal_shadow",
)


def build_label_backtest_summary(
    frames: dict[str, pl.DataFrame],
    *,
    default_cost_model: str = "conservative_p75_or_default_30bps",
) -> pl.DataFrame:
    samples: list[dict[str, Any]] = []
    for dataset_name in LABEL_SOURCE_DATASETS:
        samples.extend(_samples_from_frame(dataset_name, frames.get(dataset_name, pl.DataFrame())))
    samples.extend(_expanded_hype_wld_samples(frames))
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[
            (
                str(sample.get("strategy_id") or "UNKNOWN"),
                normalize_strategy_symbol(sample.get("symbol")),
                str(sample.get("regime") or "ALL"),
                int(sample.get("horizon_hours") or 0),
            )
        ].append(sample)

    rows_out: list[dict[str, Any]] = []
    for (strategy_id, symbol, regime, horizon), values in sorted(grouped.items()):
        numeric = summarize_net_bps([row.get("net_bps") for row in values])
        recent_7d = recent_7d_avg_net_bps(values)
        cost_models = sorted(
            {
                str(row.get("cost_model") or default_cost_model)
                for row in values
                if str(row.get("cost_model") or "").strip()
            }
        )
        leakage_checks = sorted(
            {
                str(row.get("data_leakage_check") or "pass_visible_at_decision_time")
                for row in values
            }
        )
        rows_out.append(
            {
                "strategy_id": strategy_id,
                "symbol": symbol,
                "regime": regime,
                "horizon_hours": horizon,
                "dedupe_before_rows": _max_int(values, "_dedupe_before_rows"),
                "dedupe_after_rows": _max_int(values, "_dedupe_after_rows"),
                "duplicate_rate": _max_float(values, "_duplicate_rate"),
                "sample_count": len(values),
                "complete_sample_count": numeric["complete_sample_count"],
                "avg_net_bps": numeric["avg_net_bps"],
                "median_net_bps": numeric["median_net_bps"],
                "p25_net_bps": numeric["p25_net_bps"],
                "p10_net_bps": numeric["p10_net_bps"],
                "win_rate": numeric["win_rate"],
                "max_loss_bps": numeric["max_loss_bps"],
                "cost_model": ";".join(cost_models) or default_cost_model,
                "data_leakage_check": ";".join(leakage_checks),
                "recommendation": _recommendation(
                    sample_count=len(values),
                    complete_sample_count=int(numeric["complete_sample_count"] or 0),
                    avg_net_bps=float_or_none(numeric["avg_net_bps"]),
                    p25_net_bps=float_or_none(numeric["p25_net_bps"]),
                    win_rate=float_or_none(numeric["win_rate"]),
                    recent_7d_avg_net_bps=recent_7d,
                    duplicate_rate=_max_float(values, "_duplicate_rate"),
                ),
            }
        )
    return frame_with_schema(rows_out, BACKTEST_LABEL_SUMMARY_FIELDS)


def label_summary_md(summary: pl.DataFrame) -> str:
    lines = [
        "# Backtest Label Summary",
        "",
        "Read-only label backtest using existing future-label telemetry.",
        "It does not replay live orders and does not change V5 live behavior.",
        "",
        f"- summary_rows: {summary.height}",
    ]
    if summary.is_empty():
        lines.append("- status: no observable labeled strategy samples")
        return "\n".join(lines) + "\n"
    rows_sorted = sorted(
        rows(summary),
        key=lambda row: (
            -int(row.get("complete_sample_count") or 0),
            str(row.get("strategy_id") or ""),
            str(row.get("symbol") or ""),
        ),
    )[:12]
    lines.append("")
    lines.append("Top labeled groups:")
    for row in rows_sorted:
        lines.append(
            "- "
            f"{row.get('strategy_id')} {row.get('symbol')} h={row.get('horizon_hours')} "
            f"complete={row.get('complete_sample_count')} avg={row.get('avg_net_bps')} "
            f"p25={row.get('p25_net_bps')} win={row.get('win_rate')} "
            f"recommendation={row.get('recommendation')}"
        )
    return "\n".join(lines) + "\n"


def _samples_from_frame(dataset_name: str, frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    input_rows = rows(frame)
    deduped_rows, dedupe_meta = _dedupe_label_source_rows(dataset_name, input_rows)
    for row in deduped_rows:
        strategy_id = _strategy_id(dataset_name, row)
        symbol = normalize_strategy_symbol(first_value(row, ("symbol", "would_buy_symbol")))
        selected_symbols = str(
            first_value(row, ("selected_symbols", "would_buy_symbols")) or ""
        ).strip()
        if symbol == "UNKNOWN" and selected_symbols:
            symbol = "MULTI"
        regime = str(
            first_value(
                row,
                (
                    "regime",
                    "regime_state",
                    "current_regime",
                    "risk_level",
                    "market_regime",
                ),
            )
            or "ALL"
        )
        decision_ts = coerce_dt(
            first_value(row, ("decision_ts", "ts_utc", "entry_ts", "as_of_ts", "generated_at"))
        )
        cost_model = str(
            first_value(row, ("cost_model", "cost_model_version", "cost_source")) or ""
        )
        explicit_horizon = float_or_none(
            first_value(row, ("horizon_hours", "suggested_horizon_hours", "primary_horizon_hours"))
        )
        horizons = (
            (int(explicit_horizon),)
            if explicit_horizon and explicit_horizon > 0
            else HORIZONS
        )
        for horizon in horizons:
            value = _horizon_value(row, horizon)
            if value is None and dataset_name == "bottom_zone_reversal_shadow":
                value = first_float(row, (f"return_{horizon}h_bps",))
            if value is None and not explicit_horizon and not _has_horizon_marker(row, horizon):
                continue
            output.append(
                {
                    "source_dataset": dataset_name,
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "regime": regime or "ALL",
                    "horizon_hours": int(horizon),
                    "net_bps": value,
                    "decision_ts": iso_utc(decision_ts),
                    "_decision_ts": decision_ts,
                    "cost_model": cost_model,
                    "data_leakage_check": _leakage_check(row, decision_ts),
                    "_dedupe_before_rows": dedupe_meta["dedupe_before_rows"],
                    "_dedupe_after_rows": dedupe_meta["dedupe_after_rows"],
                    "_duplicate_rate": dedupe_meta["duplicate_rate"],
                }
            )
    return output


def _dedupe_label_source_rows(
    dataset_name: str,
    input_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not input_rows:
        return [], {
            "dedupe_before_rows": 0,
            "dedupe_after_rows": 0,
            "duplicate_rate": 0.0,
        }
    keys = _dedupe_keys_for_dataset(dataset_name)
    if not keys:
        after_rows = len(input_rows)
        return input_rows, {
            "dedupe_before_rows": len(input_rows),
            "dedupe_after_rows": after_rows,
            "duplicate_rate": 0.0,
        }
    by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for index, row in enumerate(input_rows):
        key = _dedupe_key(row, keys, fallback_index=index)
        current = by_key.get(key)
        if current is None or _row_freshness_ts(row) >= _row_freshness_ts(current):
            by_key[key] = row
    deduped = list(by_key.values())
    before_rows = len(input_rows)
    after_rows = len(deduped)
    duplicate_rate = (before_rows - after_rows) / before_rows if before_rows else 0.0
    return deduped, {
        "dedupe_before_rows": before_rows,
        "dedupe_after_rows": after_rows,
        "duplicate_rate": round(duplicate_rate, 6),
    }


def _dedupe_keys_for_dataset(dataset_name: str) -> tuple[str, ...]:
    if dataset_name in {"final_score_vs_alpha6_conflict", "v5_final_score_vs_alpha6_conflict"}:
        return ("run_id", "symbol", "ts_utc")
    if dataset_name in {"bnb_strong_alpha6_bypass_shadow", "v5_bnb_strong_alpha6_bypass_shadow"}:
        return ("run_id", "symbol", "ts_utc", "strategy_id")
    return ()


def _dedupe_key(
    row: dict[str, Any],
    keys: tuple[str, ...],
    *,
    fallback_index: int,
) -> tuple[str, ...]:
    values: list[str] = []
    for key in keys:
        if key == "symbol":
            values.append(normalize_strategy_symbol(row.get("symbol")))
        elif key == "strategy_id":
            values.append(str(first_value(row, ("strategy_id", "strategy_candidate")) or ""))
        elif key == "ts_utc":
            values.append(
                iso_utc(
                    first_value(
                        row,
                        ("ts_utc", "decision_ts", "entry_ts", "as_of_ts", "generated_at"),
                    )
                )
            )
        else:
            values.append(str(row.get(key) or ""))
    if all(not value for value in values):
        values.append(f"row_index:{fallback_index}")
    return tuple(values)


def _row_freshness_ts(row: dict[str, Any]) -> datetime:
    return (
        coerce_dt(
            first_value(row, ("bundle_ts", "ingest_ts", "generated_at", "as_of_ts", "ts_utc"))
        )
        or datetime.min.replace(tzinfo=UTC)
    )


def _expanded_hype_wld_samples(frames: dict[str, pl.DataFrame]) -> list[dict[str, Any]]:
    labels = rows(frames.get("expanded_universe_candidate_label", pl.DataFrame()))
    maturity_rows = rows(frames.get("expanded_universe_candidate_maturity", pl.DataFrame()))
    ready_by_symbol: dict[str, list[Any]] = defaultdict(list)
    for row in maturity_rows:
        state = str(
            first_value(
                row,
                ("expanded_universe_maturity_state", "maturity_state", "decision"),
            )
            or ""
        ).upper()
        if state != "PAPER_READY":
            continue
        symbol = normalize_strategy_symbol(row.get("symbol"))
        ts = coerce_dt(
            first_value(row, ("as_of_ts", "generated_at", "paper_ready_ts", "decision_ts"))
        )
        ready_by_symbol[symbol].append(ts)

    output: list[dict[str, Any]] = []
    for row in labels:
        symbol = normalize_strategy_symbol(row.get("symbol"))
        if symbol not in {"HYPE-USDT", "WLD-USDT"}:
            continue
        decision_ts = coerce_dt(first_value(row, ("decision_ts", "ts_utc", "label_ts")))
        ready_points = ready_by_symbol.get(symbol, [])
        if ready_points and decision_ts is not None:
            if not any(point is None or point <= decision_ts for point in ready_points):
                continue
        strategy_id = f"{symbol.split('-', 1)[0]}_EXPANDED_UNIVERSE_BACKTEST"
        for horizon in (4, 8, 12, 24):
            output.append(
                {
                    "source_dataset": "expanded_universe_candidate_label",
                    "strategy_id": strategy_id,
                    "symbol": symbol,
                    "regime": str(first_value(row, ("regime_state", "risk_level")) or "ALL"),
                    "horizon_hours": horizon,
                    "net_bps": _horizon_value(row, horizon),
                    "decision_ts": iso_utc(decision_ts),
                    "_decision_ts": decision_ts,
                    "cost_model": str(
                        first_value(row, ("cost_model", "cost_source"))
                        or "conservative_p75"
                    ),
                    "data_leakage_check": "pass_maturity_state_visible_at_decision_time",
                    "_dedupe_before_rows": 0,
                    "_dedupe_after_rows": 0,
                    "_duplicate_rate": 0.0,
                }
            )
    return output


def _max_int(values: list[dict[str, Any]], field: str) -> int:
    observed = [int(float_or_none(row.get(field)) or 0) for row in values]
    return max(observed) if observed else 0


def _max_float(values: list[dict[str, Any]], field: str) -> float:
    observed = [float_or_none(row.get(field)) for row in values]
    observed = [value for value in observed if value is not None]
    return round(max(observed), 6) if observed else 0.0


def _strategy_id(dataset_name: str, row: dict[str, Any]) -> str:
    if dataset_name in {"final_score_vs_alpha6_conflict", "v5_final_score_vs_alpha6_conflict"}:
        return "FINAL_SCORE_ALPHA6_CONFLICT_BACKTEST"
    if dataset_name in {"bnb_strong_alpha6_bypass_shadow", "v5_bnb_strong_alpha6_bypass_shadow"}:
        return "BNB_STRONG_ALPHA6_BYPASS_BACKTEST"
    if dataset_name == "risk_on_multi_buy_shadow":
        return "RISK_ON_MULTI_BUY_BACKTEST"
    if dataset_name == "bottom_zone_reversal_shadow":
        return "BOTTOM_ZONE_PROBE_BACKTEST"
    explicit = first_value(
        row,
        (
            "strategy_id",
            "strategy_candidate",
            "source_strategy_candidate",
            "candidate_name",
            "experiment_name",
            "strategy",
        ),
    )
    if explicit:
        return str(explicit)
    return dataset_name


def _horizon_value(row: dict[str, Any], horizon: int) -> float | None:
    return first_float(
        row,
        (
            f"future_{horizon}h_net_bps",
            f"paper_pnl_bps_{horizon}h",
            f"label_{horizon}h_net_bps",
            f"net_bps_{horizon}h",
            f"{horizon}h_net_bps",
            f"future_net_bps_{horizon}h",
            f"avg_net_bps_{horizon}h",
        ),
    )


def _has_horizon_marker(row: dict[str, Any], horizon: int) -> bool:
    text = "|".join(str(key).lower() for key in row.keys())
    return f"{horizon}h" in text


def _leakage_check(row: dict[str, Any], decision_ts: Any) -> str:
    label_ts = coerce_dt(first_value(row, ("label_ts", "future_label_ts")))
    if decision_ts is not None and label_ts is not None and label_ts < decision_ts:
        return "fail_label_before_decision_ts"
    return "pass_visible_at_decision_time"


def _recommendation(
    *,
    sample_count: int,
    complete_sample_count: int,
    avg_net_bps: float | None,
    p25_net_bps: float | None,
    win_rate: float | None,
    recent_7d_avg_net_bps: float | None,
    duplicate_rate: float | None = None,
) -> str:
    if duplicate_rate is not None and duplicate_rate > 0.05:
        return "QUARANTINE_DUPLICATE_LABELS"
    if complete_sample_count <= 0:
        return "RESEARCH_ONLY_PENDING_LABELS"
    if sample_count >= 30 and (avg_net_bps or 0.0) > 0:
        if (
            complete_sample_count >= 50
            and (p25_net_bps is not None and p25_net_bps > -50)
            and (win_rate is not None and win_rate > 0.55)
            and (recent_7d_avg_net_bps is None or recent_7d_avg_net_bps >= 0)
        ):
            return "PAPER_CANDIDATE_REVIEW"
        return "KEEP_SHADOW"
    if avg_net_bps is not None and avg_net_bps <= 0:
        return "KILL_OR_KEEP_RESEARCH"
    return "RESEARCH_ONLY_COLLECT_MORE_SAMPLES"
