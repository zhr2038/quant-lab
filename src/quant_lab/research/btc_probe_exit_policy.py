from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "quant_lab.btc_probe_exit_policy_review"
SCHEMA_VERSION = "btc_probe_exit_policy_review.v0.1"
DEFAULT_MIN_SAMPLE_COUNT = 10
DEFAULT_ROUNDTRIP_COST_BPS = 30.0
HORIZONS = (8, 12, 24, 48)

V5_ROUNDTRIP_DATASET = Path("silver") / "v5_roundtrip"
V5_TRADE_EVENT_DATASET = Path("silver") / "v5_trade_event"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
BTC_PROBE_EXIT_POLICY_REVIEW_DATASET = Path("gold") / "btc_probe_exit_policy_review"
BTC_PROBE_EXIT_POLICY_SUMMARY_DATASET = Path("gold") / "btc_probe_exit_policy_summary"

REVIEW_SCHEMA: dict[str, Any] = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "run_id": pl.Utf8,
    "roundtrip_id": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "exit_ts": pl.Datetime(time_zone="UTC"),
    "entry_px": pl.Float64,
    "exit_px": pl.Float64,
    "actual_exit_net_bps": pl.Float64,
    "would_hold_8h_net_bps": pl.Float64,
    "would_hold_12h_net_bps": pl.Float64,
    "would_hold_24h_net_bps": pl.Float64,
    "would_hold_48h_net_bps": pl.Float64,
    "mae_bps": pl.Float64,
    "mfe_bps": pl.Float64,
    "exit_reason": pl.Utf8,
    "probe_hold_hours": pl.Float64,
    "selected_roundtrip_cost_bps": pl.Float64,
    "label_source": pl.Utf8,
    "exit_policy_signal": pl.Utf8,
    "status": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}

SUMMARY_SCHEMA: dict[str, Any] = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "sample_count": pl.Int64,
    "probe_stop_loss_count": pl.Int64,
    "premature_stop_loss_count": pl.Int64,
    "avg_actual_exit_net_bps": pl.Float64,
    "avg_would_hold_8h_net_bps": pl.Float64,
    "avg_would_hold_12h_net_bps": pl.Float64,
    "avg_would_hold_24h_net_bps": pl.Float64,
    "avg_would_hold_48h_net_bps": pl.Float64,
    "avg_mae_bps": pl.Float64,
    "avg_mfe_bps": pl.Float64,
    "actual_vs_24h_delta_bps": pl.Float64,
    "actual_vs_48h_delta_bps": pl.Float64,
    "status": pl.Utf8,
    "decision": pl.Utf8,
    "decision_reasons": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class BtcProbeExitPolicyReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    review_rows: int = Field(ge=0)
    summary_rows: int = Field(ge=0)
    status: str
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Context:
    as_of_date: date
    generated_at: datetime
    generated_from_bundle_id: str
    git_commit: str | None


def build_and_publish_btc_probe_exit_policy_review(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> BtcProbeExitPolicyReviewResult:
    root = Path(lake_root)
    ctx = _Context(
        as_of_date=_parse_day(as_of_date),
        generated_at=datetime.now(UTC),
        generated_from_bundle_id=_latest_bundle_id(root),
        git_commit=_git_commit(),
    )
    roundtrips = read_parquet_dataset(root / V5_ROUNDTRIP_DATASET)
    market = read_parquet_dataset(root / MARKET_BAR_DATASET)
    review = build_btc_probe_exit_policy_review(
        roundtrips=roundtrips,
        market_bars=market,
        ctx=ctx,
        min_sample_count=min_sample_count,
    )
    summary = build_btc_probe_exit_policy_summary(
        review,
        ctx=ctx,
        min_sample_count=min_sample_count,
    )
    write_parquet_dataset(review, root / BTC_PROBE_EXIT_POLICY_REVIEW_DATASET)
    write_parquet_dataset(summary, root / BTC_PROBE_EXIT_POLICY_SUMMARY_DATASET)
    status = (
        str(summary["status"][0])
        if not summary.is_empty() and "status" in summary.columns
        else "RESEARCH_ONLY"
    )
    warnings = [] if review.height else ["btc_strict_probe_roundtrip_missing"]
    return BtcProbeExitPolicyReviewResult(
        as_of_date=ctx.as_of_date.isoformat(),
        review_rows=review.height,
        summary_rows=summary.height,
        status=status,
        warnings=warnings,
    )


def build_btc_probe_exit_policy_review(
    *,
    roundtrips: pl.DataFrame,
    market_bars: pl.DataFrame,
    ctx: _Context,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> pl.DataFrame:
    if roundtrips.is_empty():
        return pl.DataFrame(schema=REVIEW_SCHEMA)
    bars = _market_bar_index(market_bars)
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, ...]] = set()
    for row in roundtrips.to_dicts():
        payload = _payload(row)
        if not _is_btc_strict_probe_roundtrip(row, payload):
            continue
        entry_ts = _parse_datetime(_field(row, payload, "entry_ts", "open_ts", "entry_time"))
        exit_ts = _parse_datetime(
            _field(row, payload, "exit_ts", "close_ts", "closed_at", "ts_utc", "ts")
        )
        entry_px = _float_or_none(_field(row, payload, "entry_px", "entry_price", "open_price"))
        exit_px = _float_or_none(_field(row, payload, "exit_px", "exit_price", "close_price"))
        actual = _float_or_none(
            _field(
                row,
                payload,
                "actual_exit_net_bps",
                "net_bps",
                "realized_net_bps",
                "pnl_bps",
                "roundtrip_net_bps",
            )
        )
        exit_reason = str(_field(row, payload, "exit_reason", "close_reason", "reason") or "")
        if entry_ts is None or (actual is None and not exit_reason.strip()):
            continue
        cost_bps = _float_or_none(
            _field(
                row,
                payload,
                "selected_roundtrip_cost_bps",
                "roundtrip_all_in_cost_bps",
                "roundtrip_cost_bps",
                "cost_bps",
            )
        )
        if cost_bps is None:
            cost_bps = DEFAULT_ROUNDTRIP_COST_BPS
        roundtrip_id = str(
            _field(row, payload, "roundtrip_id", "trade_id", "position_id")
            or _source_event_key(row)
        )
        dedupe_key = _roundtrip_dedupe_key(
            row=row,
            payload=payload,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            entry_px=entry_px,
            exit_px=exit_px,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        hold_values: dict[int, float | None] = {}
        label_sources: set[str] = set()
        for horizon in HORIZONS:
            value = _float_or_none(
                _field(
                    row,
                    payload,
                    f"would_hold_{horizon}h_net_bps",
                    f"would_have_held_{horizon}h_net_bps",
                    f"hold_{horizon}h_net_bps",
                    f"label_{horizon}h_net_bps",
                )
            )
            if value is None and entry_ts is not None and entry_px is not None:
                value = _would_hold_from_market(
                    bars=bars,
                    entry_ts=entry_ts,
                    entry_px=entry_px,
                    horizon_hours=horizon,
                    side=str(_field(row, payload, "side", "entry_side", "direction") or "buy"),
                    cost_bps=cost_bps,
                )
                if value is not None:
                    label_sources.add("market_bar_conservative_cost")
            else:
                label_sources.add("v5_roundtrip_field")
            hold_values[horizon] = value
        mae_bps = _float_or_none(
            _field(
                row,
                payload,
                "mae_bps",
                "max_adverse_bps",
                "adverse_excursion_bps",
                "label_24h_mae_bps",
                "label_48h_mae_bps",
                "mae_bps_24h",
                "mae_bps_48h",
                "paper_mae_bps_24h",
                "paper_mae_bps_48h",
            )
        )
        mfe_bps = _float_or_none(
            _field(
                row,
                payload,
                "mfe_bps",
                "max_favorable_bps",
                "favorable_excursion_bps",
                "label_24h_mfe_bps",
                "label_48h_mfe_bps",
                "mfe_bps_24h",
                "mfe_bps_48h",
                "paper_mfe_bps_24h",
                "paper_mfe_bps_48h",
            )
        )
        if (mae_bps is None or mfe_bps is None) and entry_ts is not None and entry_px is not None:
            market_mae, market_mfe = _mae_mfe_from_market(
                bars=bars,
                entry_ts=entry_ts,
                entry_px=entry_px,
                horizon_hours=max(HORIZONS),
                side=str(_field(row, payload, "side", "entry_side", "direction") or "buy"),
            )
            mae_bps = mae_bps if mae_bps is not None else market_mae
            mfe_bps = mfe_bps if mfe_bps is not None else market_mfe
        signal = _exit_policy_signal(
            actual_exit_net_bps=actual,
            would_hold_24h_net_bps=hold_values[24],
            exit_reason=exit_reason,
        )
        rows.append(
            _common(ctx)
            | {
                "as_of_date": ctx.as_of_date.isoformat(),
                "strategy_candidate": "v5.btc_leadership_probe_strict",
                "symbol": "BTC-USDT",
                "run_id": str(_field(row, payload, "run_id") or row.get("run_id") or ""),
                "roundtrip_id": roundtrip_id,
                "entry_ts": entry_ts,
                "exit_ts": exit_ts,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "actual_exit_net_bps": actual,
                "would_hold_8h_net_bps": hold_values[8],
                "would_hold_12h_net_bps": hold_values[12],
                "would_hold_24h_net_bps": hold_values[24],
                "would_hold_48h_net_bps": hold_values[48],
                "mae_bps": mae_bps,
                "mfe_bps": mfe_bps,
                "exit_reason": exit_reason,
                "probe_hold_hours": _hold_hours(entry_ts, exit_ts)
                or _float_or_none(_field(row, payload, "probe_hold_hours", "hold_hours")),
                "selected_roundtrip_cost_bps": cost_bps,
                "label_source": "+".join(sorted(label_sources)) or "not_observable",
                "exit_policy_signal": signal,
                "status": "RESEARCH_ONLY",
                "created_at": ctx.generated_at,
                "source": SOURCE_NAME,
            }
        )
    if not rows:
        return pl.DataFrame(schema=REVIEW_SCHEMA)
    frame = pl.DataFrame(rows, schema=REVIEW_SCHEMA, orient="row")
    sample_count = frame.height
    status = "RESEARCH_ONLY" if sample_count < min_sample_count else "REVIEW"
    return frame.with_columns(pl.lit(status).alias("status"))


def build_btc_probe_exit_policy_summary(
    review: pl.DataFrame,
    *,
    ctx: _Context,
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT,
) -> pl.DataFrame:
    rows = review.to_dicts() if not review.is_empty() else []
    sample_count = len(rows)
    stop_loss_rows = [
        row for row in rows if "stop_loss" in str(row.get("exit_reason") or "").lower()
    ]
    premature = [
        row for row in rows if row.get("exit_policy_signal") == "possible_stop_loss_too_early"
    ]
    avg_actual = _mean(row.get("actual_exit_net_bps") for row in rows)
    avg_8h = _mean(row.get("would_hold_8h_net_bps") for row in rows)
    avg_12h = _mean(row.get("would_hold_12h_net_bps") for row in rows)
    avg_24h = _mean(row.get("would_hold_24h_net_bps") for row in rows)
    avg_48h = _mean(row.get("would_hold_48h_net_bps") for row in rows)
    avg_mae = _mean(row.get("mae_bps") for row in rows)
    avg_mfe = _mean(row.get("mfe_bps") for row in rows)
    status = "RESEARCH_ONLY" if sample_count < min_sample_count else "REVIEW"
    reasons = []
    if sample_count < min_sample_count:
        reasons.append("insufficient_btc_strict_probe_roundtrips")
    if premature:
        reasons.append("probe_stop_loss_may_be_too_early")
    if not rows:
        reasons.append("no_btc_strict_probe_roundtrip")
    decision = "do_not_change_live_exit_policy"
    return pl.DataFrame(
        [
            _common(ctx)
            | {
                "as_of_date": ctx.as_of_date.isoformat(),
                "strategy_candidate": "v5.btc_leadership_probe_strict",
                "symbol": "BTC-USDT",
                "sample_count": sample_count,
                "probe_stop_loss_count": len(stop_loss_rows),
                "premature_stop_loss_count": len(premature),
                "avg_actual_exit_net_bps": avg_actual,
                "avg_would_hold_8h_net_bps": avg_8h,
                "avg_would_hold_12h_net_bps": avg_12h,
                "avg_would_hold_24h_net_bps": avg_24h,
                "avg_would_hold_48h_net_bps": avg_48h,
                "avg_mae_bps": avg_mae,
                "avg_mfe_bps": avg_mfe,
                "actual_vs_24h_delta_bps": (
                    avg_24h - avg_actual if avg_24h is not None and avg_actual is not None else None
                ),
                "actual_vs_48h_delta_bps": (
                    avg_48h - avg_actual if avg_48h is not None and avg_actual is not None else None
                ),
                "status": status,
                "decision": decision,
                "decision_reasons": safe_json_dumps(reasons),
                "created_at": ctx.generated_at,
                "source": SOURCE_NAME,
            }
        ],
        schema=SUMMARY_SCHEMA,
        orient="row",
    )


def btc_probe_exit_policy_summary_md(summary: pl.DataFrame, review: pl.DataFrame) -> str:
    if summary.is_empty():
        return (
            "# BTC Strict Probe Exit Policy Review\n\n"
            "- status: RESEARCH_ONLY\n"
            "- reason: no summary rows\n"
        )
    row = summary.to_dicts()[0]
    lines = [
        "# BTC Strict Probe Exit Policy Review",
        "",
        f"- status: {row.get('status')}",
        f"- decision: {row.get('decision')}",
        f"- sample_count: {row.get('sample_count')}",
        f"- probe_stop_loss_count: {row.get('probe_stop_loss_count')}",
        f"- premature_stop_loss_count: {row.get('premature_stop_loss_count')}",
        f"- avg_actual_exit_net_bps: {_fmt(row.get('avg_actual_exit_net_bps'))}",
        f"- avg_would_hold_8h_net_bps: {_fmt(row.get('avg_would_hold_8h_net_bps'))}",
        f"- avg_would_hold_12h_net_bps: {_fmt(row.get('avg_would_hold_12h_net_bps'))}",
        f"- avg_would_hold_24h_net_bps: {_fmt(row.get('avg_would_hold_24h_net_bps'))}",
        f"- avg_would_hold_48h_net_bps: {_fmt(row.get('avg_would_hold_48h_net_bps'))}",
        f"- avg_mae_bps: {_fmt(row.get('avg_mae_bps'))}",
        f"- avg_mfe_bps: {_fmt(row.get('avg_mfe_bps'))}",
        f"- actual_vs_24h_delta_bps: {_fmt(row.get('actual_vs_24h_delta_bps'))}",
        f"- actual_vs_48h_delta_bps: {_fmt(row.get('actual_vs_48h_delta_bps'))}",
        f"- decision_reasons: {row.get('decision_reasons')}",
        "",
        "This is read-only research. It does not modify V5 exit policy.",
    ]
    if not review.is_empty():
        signals = review.select(pl.col("exit_policy_signal")).to_series().to_list()
        counts = {signal: signals.count(signal) for signal in sorted(set(signals))}
        lines.extend(["", f"- exit_policy_signal_counts: {safe_json_dumps(counts)}"])
    return "\n".join(lines) + "\n"


def _common(ctx: _Context) -> dict[str, Any]:
    return {
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "quant_lab_git_commit": ctx.git_commit or "not_observable",
        "source_version": _source_version("btc_probe_exit_policy", ctx.git_commit),
        "generated_at_utc": ctx.generated_at,
        "generated_from_bundle_id": ctx.generated_from_bundle_id,
    }


def _is_btc_strict_probe_roundtrip(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    symbol_value = _field(row, payload, "normalized_symbol", "symbol", "v5_symbol", "inst_id")
    if normalize_symbol(str(symbol_value or "")) != "BTC-USDT":
        return False
    haystack = " ".join(
        str(_field(row, payload, field) or "")
        for field in [
            "strategy_candidate",
            "candidate_name",
            "probe_name",
            "probe_type",
            "entry_reason",
            "strategy_id",
            "source_type",
        ]
    ).lower()
    if "v5.btc_leadership_probe_strict" in haystack or "btc_leadership_probe" in haystack:
        return True
    return "btc" in haystack and "strict" in haystack and "probe" in haystack


def _roundtrip_dedupe_key(
    *,
    row: dict[str, Any],
    payload: dict[str, Any],
    entry_ts: datetime | None,
    exit_ts: datetime | None,
    entry_px: float | None,
    exit_px: float | None,
) -> tuple[str, ...]:
    return (
        str(_field(row, payload, "run_id") or row.get("run_id") or ""),
        "" if entry_ts is None else entry_ts.isoformat(),
        "" if exit_ts is None else exit_ts.isoformat(),
        "" if entry_px is None else f"{entry_px:.8f}",
        "" if exit_px is None else f"{exit_px:.8f}",
        str(_field(row, payload, "exit_reason", "close_reason", "reason") or ""),
    )


def _exit_policy_signal(
    *,
    actual_exit_net_bps: float | None,
    would_hold_24h_net_bps: float | None,
    exit_reason: str,
) -> str:
    reason = exit_reason.lower()
    if actual_exit_net_bps is None or would_hold_24h_net_bps is None:
        return "insufficient_label"
    if "stop_loss" in reason and actual_exit_net_bps < 0.0 and would_hold_24h_net_bps > 0.0:
        return "possible_stop_loss_too_early"
    if "stop_loss" in reason and would_hold_24h_net_bps <= actual_exit_net_bps:
        return "stop_loss_helped"
    if would_hold_24h_net_bps > actual_exit_net_bps:
        return "hold_24h_would_improve"
    return "actual_exit_not_worse_than_24h_hold"


def _market_bar_index(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if market_bars.is_empty() or "close" not in market_bars.columns:
        return {}
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {}
    for row in market_bars.to_dicts():
        symbol = normalize_symbol(str(row.get("symbol") or row.get("inst_id") or ""))
        ts = _parse_datetime(row.get("ts"))
        close = _float_or_none(row.get("close"))
        if not symbol or ts is None or close is None:
            continue
        high = _float_or_none(row.get("high"))
        low = _float_or_none(row.get("low"))
        rows_by_symbol.setdefault(symbol, []).append(
            {
                "ts": ts,
                "close": close,
                "high": high if high is not None else close,
                "low": low if low is not None else close,
            }
        )
    for rows in rows_by_symbol.values():
        rows.sort(key=lambda item: item["ts"])
    return rows_by_symbol


def _would_hold_from_market(
    *,
    bars: dict[str, list[dict[str, Any]]],
    entry_ts: datetime,
    entry_px: float,
    horizon_hours: int,
    side: str,
    cost_bps: float,
) -> float | None:
    future_ts = entry_ts + timedelta(hours=horizon_hours)
    future_close = None
    for bar in bars.get("BTC-USDT", []):
        if bar["ts"] >= future_ts:
            future_close = float(bar["close"])
            break
    if future_close is None or entry_px <= 0:
        return None
    side_text = side.lower()
    if side_text in {"sell", "short"}:
        gross = (entry_px / future_close - 1.0) * 10_000.0
    else:
        gross = (future_close / entry_px - 1.0) * 10_000.0
    return gross - cost_bps


def _mae_mfe_from_market(
    *,
    bars: dict[str, list[dict[str, Any]]],
    entry_ts: datetime,
    entry_px: float,
    horizon_hours: int,
    side: str,
) -> tuple[float | None, float | None]:
    if entry_px <= 0:
        return (None, None)
    end_ts = entry_ts + timedelta(hours=horizon_hours)
    window = [bar for bar in bars.get("BTC-USDT", []) if entry_ts < bar["ts"] <= end_ts]
    if not window:
        return (None, None)
    side_text = side.lower()
    max_high = max(float(bar["high"]) for bar in window)
    min_low = min(float(bar["low"]) for bar in window)
    if side_text in {"sell", "short"}:
        mfe = (entry_px / min_low - 1.0) * 10_000.0 if min_low > 0 else None
        mae = (entry_px / max_high - 1.0) * 10_000.0 if max_high > 0 else None
    else:
        mfe = (max_high / entry_px - 1.0) * 10_000.0
        mae = (min_low / entry_px - 1.0) * 10_000.0
    return (mae, mfe)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("raw_payload_json")
    if not value:
        return {}
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _field(row: dict[str, Any], payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        value = row.get(name)
        if _observable(value):
            return value
        value = payload.get(name)
        if _observable(value):
            return value
    return None


def _observable(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() not in {"", "none", "null", "nan", "not_observable"}


def _float_or_none(value: Any) -> float | None:
    if not _observable(value):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not _observable(value):
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).strip()
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _hold_hours(entry_ts: datetime | None, exit_ts: datetime | None) -> float | None:
    if entry_ts is None or exit_ts is None:
        return None
    return (exit_ts - entry_ts).total_seconds() / 3600.0


def _mean(values: Any) -> float | None:
    observed = [_float_or_none(value) for value in values]
    numbers = [value for value in observed if value is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _source_event_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(field) or "")
        for field in ["source_path_inside_bundle", "source_file", "row_index", "run_id"]
    )


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    return "None" if number is None else f"{number:.4f}"


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value and value != "auto":
        return date.fromisoformat(str(value))
    return datetime.now(UTC).date()


def _latest_bundle_id(root: Path) -> str:
    for dataset in [V5_ROUNDTRIP_DATASET, V5_TRADE_EVENT_DATASET]:
        frame = read_parquet_dataset(root / dataset)
        if frame.is_empty():
            continue
        candidates = []
        for row in frame.to_dicts():
            bundle_ts = _parse_datetime(row.get("bundle_ts") or row.get("ingest_ts"))
            bundle_name = str(row.get("bundle_name") or row.get("source_bundle") or "")
            if bundle_ts is not None:
                candidates.append((bundle_ts, bundle_name or bundle_ts.isoformat()))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
    return "not_observable"


def _git_commit() -> str | None:
    root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            cwd=root,
            text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def _source_version(component: str, git_commit: str | None) -> str:
    return f"{component}:{git_commit or __version__}"
