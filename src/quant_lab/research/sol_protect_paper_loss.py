from __future__ import annotations

import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab import __version__
from quant_lab.contracts.v5_quant_lab import V5_QUANT_LAB_CONTRACT_VERSION
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.paper_tracking import latest_v5_paper_frame
from quant_lab.strategy_telemetry.sanitize import safe_json_dumps
from quant_lab.symbols import normalize_symbol

SOURCE_NAME = "quant_lab.sol_protect_paper_loss_attribution"
SCHEMA_VERSION = "sol_protect_paper_loss_attribution.v0.1"
SUMMARY_SCHEMA_VERSION = "sol_protect_paper_loss_summary.v0.1"
PROPOSAL_ID = "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1"
STRATEGY_CANDIDATE = "v5.sol_protect_alpha6_low_exception"
SYMBOL = "SOL-USDT"
HORIZONS = (4, 8, 12, 24)

V5_PAPER_STRATEGY_RUN_DATASET = Path("silver") / "v5_paper_strategy_run"
MARKET_BAR_DATASET = Path("silver") / "market_bar"
ATTRIBUTION_DATASET = Path("gold") / "sol_protect_paper_loss_attribution"
SUMMARY_DATASET = Path("gold") / "sol_protect_paper_loss_summary"

ATTRIBUTION_SCHEMA: dict[str, Any] = {
    "contract_version": pl.Utf8,
    "schema_version": pl.Utf8,
    "quant_lab_git_commit": pl.Utf8,
    "source_version": pl.Utf8,
    "generated_at_utc": pl.Datetime(time_zone="UTC"),
    "generated_from_bundle_id": pl.Utf8,
    "as_of_date": pl.Utf8,
    "proposal_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "strategy_id": pl.Utf8,
    "run_id": pl.Utf8,
    "entry_ts": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "entry_px": pl.Float64,
    "alpha6_score": pl.Float64,
    "alpha6_side": pl.Utf8,
    "f4_volume_expansion": pl.Float64,
    "f5_rsi_trend_confirm": pl.Float64,
    "risk_level": pl.Utf8,
    "btc_trend_state": pl.Utf8,
    "market_regime": pl.Utf8,
    "entry_position_in_24h_range": pl.Float64,
    "paper_pnl_4h": pl.Float64,
    "paper_pnl_8h": pl.Float64,
    "paper_pnl_12h": pl.Float64,
    "paper_pnl_24h": pl.Float64,
    "mae_bps": pl.Float64,
    "mfe_bps": pl.Float64,
    "mae_bps_4h": pl.Float64,
    "mae_bps_8h": pl.Float64,
    "mae_bps_12h": pl.Float64,
    "mae_bps_24h": pl.Float64,
    "mfe_bps_4h": pl.Float64,
    "mfe_bps_8h": pl.Float64,
    "mfe_bps_12h": pl.Float64,
    "mfe_bps_24h": pl.Float64,
    "loss_horizons": pl.Utf8,
    "primary_loss_bps": pl.Float64,
    "attribution_tags": pl.Utf8,
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
    "proposal_id": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "symbol": pl.Utf8,
    "entry_count": pl.Int64,
    "loss_entry_count": pl.Int64,
    "loss_rate": pl.Float64,
    "avg_alpha6_score_loss": pl.Float64,
    "avg_f4_volume_expansion_loss": pl.Float64,
    "avg_f5_rsi_trend_confirm_loss": pl.Float64,
    "avg_entry_position_in_24h_range_loss": pl.Float64,
    "avg_paper_pnl_4h_loss": pl.Float64,
    "avg_paper_pnl_8h_loss": pl.Float64,
    "avg_paper_pnl_12h_loss": pl.Float64,
    "avg_paper_pnl_24h_loss": pl.Float64,
    "avg_mae_bps_loss": pl.Float64,
    "avg_mfe_bps_loss": pl.Float64,
    "risk_level_mix_loss": pl.Utf8,
    "btc_trend_state_mix_loss": pl.Utf8,
    "market_regime_mix_loss": pl.Utf8,
    "alpha6_side_mix_loss": pl.Utf8,
    "common_loss_tags": pl.Utf8,
    "diagnosis": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


class SolProtectPaperLossAttributionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    attribution_rows: int = Field(ge=0)
    summary_rows: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class _Context:
    as_of_date: date
    generated_at: datetime
    generated_from_bundle_id: str
    git_commit: str


def build_and_publish_sol_protect_paper_loss_attribution(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> SolProtectPaperLossAttributionResult:
    root = Path(lake_root)
    ctx = _Context(
        as_of_date=_parse_day(as_of_date),
        generated_at=datetime.now(UTC),
        generated_from_bundle_id=_latest_bundle_id(root),
        git_commit=_git_commit(),
    )
    raw_runs = latest_v5_paper_frame(read_parquet_dataset(root / V5_PAPER_STRATEGY_RUN_DATASET))
    market_bars = read_parquet_dataset(root / MARKET_BAR_DATASET)
    attribution = build_sol_protect_paper_loss_attribution(
        paper_runs=raw_runs,
        market_bars=market_bars,
        ctx=ctx,
    )
    summary = build_sol_protect_paper_loss_summary(attribution, ctx=ctx)
    write_parquet_dataset(attribution, root / ATTRIBUTION_DATASET)
    write_parquet_dataset(summary, root / SUMMARY_DATASET)
    warnings = []
    if attribution.is_empty():
        warnings.append("sol_protect_paper_entry_missing")
    return SolProtectPaperLossAttributionResult(
        as_of_date=ctx.as_of_date.isoformat(),
        attribution_rows=attribution.height,
        summary_rows=summary.height,
        warnings=warnings,
    )


def build_sol_protect_paper_loss_attribution(
    *,
    paper_runs: pl.DataFrame,
    market_bars: pl.DataFrame,
    ctx: _Context,
) -> pl.DataFrame:
    if paper_runs.is_empty():
        return pl.DataFrame(schema=ATTRIBUTION_SCHEMA)
    bar_index = _market_bar_index(market_bars)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw_row in paper_runs.to_dicts():
        row = _flatten_row(raw_row)
        if not _is_sol_protect_paper_entry(row):
            continue
        entry_ts = _parse_datetime(_field(row, "entry_ts", "ts_utc", "ts", "created_at"))
        if entry_ts is None:
            continue
        symbol = normalize_symbol(_field(row, "symbol", "normalized_symbol") or SYMBOL)
        entry_px = _float(
            _field(
                row,
                "entry_px",
                "entry_price",
                "estimated_fill_px",
                "estimated_fill_price",
                "arrival_mid",
                "mid_at_arrival",
                "current_px",
                "price",
            )
        )
        key = (
            str(_field(row, "run_id") or ""),
            entry_ts.isoformat(),
            str(_field(row, "strategy_id", "proposal_id") or PROPOSAL_ID),
        )
        if key in seen:
            continue
        seen.add(key)
        entry_position = _float(_field(row, "entry_position_in_24h_range"))
        if entry_position is None:
            entry_position = _entry_position_in_24h_range(
                bar_index.get(symbol, []),
                entry_ts=entry_ts,
                entry_px=entry_px,
            )
        pnl = {h: _pnl(row, h) for h in HORIZONS}
        mae = {h: _mae(row, h) for h in HORIZONS}
        mfe = {h: _mfe(row, h) for h in HORIZONS}
        loss_horizons = [f"{h}h" for h, value in pnl.items() if value is not None and value < 0]
        primary_loss = _primary_loss(pnl)
        row_out = {
            "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "quant_lab_git_commit": ctx.git_commit,
            "source_version": f"{SOURCE_NAME}:{__version__}",
            "generated_at_utc": ctx.generated_at,
            "generated_from_bundle_id": _generated_from_bundle(row, ctx),
            "as_of_date": _as_of_date(row, entry_ts, ctx),
            "proposal_id": _field(row, "proposal_id", "strategy_id") or PROPOSAL_ID,
            "strategy_candidate": _field(row, "strategy_candidate") or STRATEGY_CANDIDATE,
            "strategy_id": _field(row, "strategy_id", "proposal_id") or PROPOSAL_ID,
            "run_id": _field(row, "run_id"),
            "entry_ts": entry_ts,
            "symbol": symbol,
            "entry_px": entry_px,
            "alpha6_score": _float(_field(row, "alpha6_score")),
            "alpha6_side": _field(row, "alpha6_side"),
            "f4_volume_expansion": _float(_field(row, "f4_volume_expansion")),
            "f5_rsi_trend_confirm": _float(_field(row, "f5_rsi_trend_confirm")),
            "risk_level": _field(row, "risk_level"),
            "btc_trend_state": _field(row, "btc_trend_state", "btc_trend"),
            "market_regime": _field(row, "market_regime", "regime_state"),
            "entry_position_in_24h_range": entry_position,
            "paper_pnl_4h": pnl[4],
            "paper_pnl_8h": pnl[8],
            "paper_pnl_12h": pnl[12],
            "paper_pnl_24h": pnl[24],
            "mae_bps": _aggregate_mae(mae),
            "mfe_bps": _aggregate_mfe(mfe),
            "mae_bps_4h": mae[4],
            "mae_bps_8h": mae[8],
            "mae_bps_12h": mae[12],
            "mae_bps_24h": mae[24],
            "mfe_bps_4h": mfe[4],
            "mfe_bps_8h": mfe[8],
            "mfe_bps_12h": mfe[12],
            "mfe_bps_24h": mfe[24],
            "loss_horizons": ",".join(loss_horizons),
            "primary_loss_bps": primary_loss,
            "attribution_tags": safe_json_dumps(
                _attribution_tags(
                    entry_position=entry_position,
                    f4=_float(_field(row, "f4_volume_expansion")),
                    f5=_float(_field(row, "f5_rsi_trend_confirm")),
                    risk_level=_field(row, "risk_level"),
                    btc_trend_state=_field(row, "btc_trend_state", "btc_trend"),
                    market_regime=_field(row, "market_regime", "regime_state"),
                    mae_bps=_aggregate_mae(mae),
                    loss_horizons=loss_horizons,
                )
            ),
            "created_at": ctx.generated_at,
            "source": SOURCE_NAME,
        }
        rows.append(row_out)
    if not rows:
        return pl.DataFrame(schema=ATTRIBUTION_SCHEMA)
    return pl.DataFrame(rows, schema=ATTRIBUTION_SCHEMA, orient="row").sort("entry_ts")


def build_sol_protect_paper_loss_summary(
    attribution: pl.DataFrame,
    *,
    ctx: _Context,
) -> pl.DataFrame:
    if attribution.is_empty():
        return pl.DataFrame(schema=SUMMARY_SCHEMA)
    rows = attribution.to_dicts()
    loss_rows = [
        row
        for row in rows
        if any((_float(row.get(f"paper_pnl_{h}h")) or 0.0) < 0 for h in HORIZONS)
    ]
    source_rows = loss_rows or rows
    tag_counter: Counter[str] = Counter()
    for row in source_rows:
        for tag in _json_list(row.get("attribution_tags")):
            tag_counter[str(tag)] += 1
    diagnosis = _diagnosis(tag_counter, loss_rows)
    row = {
        "contract_version": V5_QUANT_LAB_CONTRACT_VERSION,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "quant_lab_git_commit": ctx.git_commit,
        "source_version": f"{SOURCE_NAME}:{__version__}",
        "generated_at_utc": ctx.generated_at,
        "generated_from_bundle_id": _summary_bundle_id(rows, ctx),
        "as_of_date": ctx.as_of_date.isoformat(),
        "proposal_id": PROPOSAL_ID,
        "strategy_candidate": STRATEGY_CANDIDATE,
        "symbol": SYMBOL,
        "entry_count": len(rows),
        "loss_entry_count": len(loss_rows),
        "loss_rate": len(loss_rows) / len(rows) if rows else 0.0,
        "avg_alpha6_score_loss": _mean(row.get("alpha6_score") for row in source_rows),
        "avg_f4_volume_expansion_loss": _mean(
            row.get("f4_volume_expansion") for row in source_rows
        ),
        "avg_f5_rsi_trend_confirm_loss": _mean(
            row.get("f5_rsi_trend_confirm") for row in source_rows
        ),
        "avg_entry_position_in_24h_range_loss": _mean(
            row.get("entry_position_in_24h_range") for row in source_rows
        ),
        "avg_paper_pnl_4h_loss": _mean(row.get("paper_pnl_4h") for row in source_rows),
        "avg_paper_pnl_8h_loss": _mean(row.get("paper_pnl_8h") for row in source_rows),
        "avg_paper_pnl_12h_loss": _mean(row.get("paper_pnl_12h") for row in source_rows),
        "avg_paper_pnl_24h_loss": _mean(row.get("paper_pnl_24h") for row in source_rows),
        "avg_mae_bps_loss": _mean(row.get("mae_bps") for row in source_rows),
        "avg_mfe_bps_loss": _mean(row.get("mfe_bps") for row in source_rows),
        "risk_level_mix_loss": _mix(source_rows, "risk_level"),
        "btc_trend_state_mix_loss": _mix(source_rows, "btc_trend_state"),
        "market_regime_mix_loss": _mix(source_rows, "market_regime"),
        "alpha6_side_mix_loss": _mix(source_rows, "alpha6_side"),
        "common_loss_tags": safe_json_dumps(dict(tag_counter.most_common())),
        "diagnosis": diagnosis,
        "created_at": ctx.generated_at,
        "source": SOURCE_NAME,
    }
    return pl.DataFrame([row], schema=SUMMARY_SCHEMA, orient="row")


def sol_protect_paper_loss_summary_md(
    summary: pl.DataFrame,
    attribution: pl.DataFrame,
) -> str:
    lines = [
        "# SOL Protect Paper Loss Attribution",
        "",
        "This report is read-only and does not change V5 live configuration.",
        "",
    ]
    if summary.is_empty():
        lines.extend(
            [
                "No SOL protect paper entries were available for attribution.",
                "",
                "- next_action: wait_for_v5_paper_strategy_run_telemetry",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"
    row = summary.to_dicts()[0]
    lines.extend(
        [
            "## Summary",
            "",
            f"- as_of_date: {row.get('as_of_date')}",
            f"- entry_count: {_int(row.get('entry_count'))}",
            f"- loss_entry_count: {_int(row.get('loss_entry_count'))}",
            f"- loss_rate: {_display_pct(row.get('loss_rate'))}",
            f"- diagnosis: {row.get('diagnosis')}",
            "",
            "## Loss Entry Common Features",
            "",
            f"- avg_alpha6_score: {_display_number(row.get('avg_alpha6_score_loss'))}",
            "- avg_f4_volume_expansion: "
            f"{_display_number(row.get('avg_f4_volume_expansion_loss'))}",
            "- avg_f5_rsi_trend_confirm: "
            f"{_display_number(row.get('avg_f5_rsi_trend_confirm_loss'))}",
            "- avg_entry_position_in_24h_range: "
            f"{_display_number(row.get('avg_entry_position_in_24h_range_loss'))}",
            f"- avg_24h_pnl_bps: {_display_number(row.get('avg_paper_pnl_24h_loss'))}",
            f"- avg_mae_bps: {_display_number(row.get('avg_mae_bps_loss'))}",
            f"- risk_level_mix: {row.get('risk_level_mix_loss') or '{}'}",
            f"- btc_trend_state_mix: {row.get('btc_trend_state_mix_loss') or '{}'}",
            f"- market_regime_mix: {row.get('market_regime_mix_loss') or '{}'}",
            f"- common_loss_tags: {row.get('common_loss_tags') or '{}'}",
            "",
        ]
    )
    if not attribution.is_empty():
        worst = sorted(
            attribution.to_dicts(),
            key=lambda item: _float(item.get("primary_loss_bps")) or 0.0,
        )[:5]
        lines.extend(["## Worst Entries", ""])
        for item in worst:
            lines.append(
                "- "
                f"{item.get('entry_ts')} "
                f"run={item.get('run_id') or 'n/a'} "
                f"24h={_display_number(item.get('paper_pnl_24h'))}bps "
                f"pos24h={_display_number(item.get('entry_position_in_24h_range'))} "
                f"tags={item.get('attribution_tags') or '[]'}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _is_sol_protect_paper_entry(row: dict[str, Any]) -> bool:
    symbol = normalize_symbol(_field(row, "symbol", "normalized_symbol") or "")
    if symbol != SYMBOL:
        return False
    identity = " ".join(
        str(_field(row, key) or "")
        for key in ["proposal_id", "strategy_id", "strategy_candidate", "experiment_name"]
    ).lower()
    if not (
        PROPOSAL_ID.lower() in identity
        or STRATEGY_CANDIDATE.lower() in identity
        or "sol_protect_alpha6_low_exception" in identity
    ):
        return False
    would_enter = _bool(_field(row, "would_enter", "enter"))
    has_pnl = any(_pnl(row, horizon) is not None for horizon in HORIZONS)
    return would_enter is True or has_pnl


def _flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    payload = _payload(row)
    flattened = dict(row)
    for key, value in payload.items():
        if key not in flattened or _empty(flattened.get(key)):
            flattened[key] = value
    return flattened


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_payload_json")
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _field(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if not _empty(value):
            return str(value)
    return ""


def _empty(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"none", "null", "nan", "not_observable"}


def _pnl(row: dict[str, Any], horizon: int) -> float | None:
    return _float(
        _field(
            row,
            f"paper_pnl_bps_{horizon}h",
            f"paper_pnl_{horizon}h",
            f"label_{horizon}h_net_bps",
            f"net_bps_{horizon}h",
            f"forward_{horizon}h_net_bps",
        )
    )


def _mae(row: dict[str, Any], horizon: int) -> float | None:
    return _float(
        _field(
            row,
            f"mae_bps_{horizon}h",
            f"paper_mae_bps_{horizon}h",
            f"label_{horizon}h_mae_bps",
            f"mae_{horizon}h_bps",
        )
    )


def _mfe(row: dict[str, Any], horizon: int) -> float | None:
    return _float(
        _field(
            row,
            f"mfe_bps_{horizon}h",
            f"paper_mfe_bps_{horizon}h",
            f"label_{horizon}h_mfe_bps",
            f"mfe_{horizon}h_bps",
        )
    )


def _aggregate_mae(values: dict[int, float | None]) -> float | None:
    observed = [value for value in values.values() if value is not None]
    return min(observed) if observed else None


def _aggregate_mfe(values: dict[int, float | None]) -> float | None:
    observed = [value for value in values.values() if value is not None]
    return max(observed) if observed else None


def _primary_loss(pnl: dict[int, float | None]) -> float | None:
    losses = [value for value in pnl.values() if value is not None and value < 0]
    return min(losses) if losses else None


def _attribution_tags(
    *,
    entry_position: float | None,
    f4: float | None,
    f5: float | None,
    risk_level: str,
    btc_trend_state: str,
    market_regime: str,
    mae_bps: float | None,
    loss_horizons: list[str],
) -> list[str]:
    tags: list[str] = []
    if loss_horizons:
        tags.append("paper_loss_observed")
    if entry_position is not None and entry_position >= 0.7:
        tags.append("entry_high_in_24h_range")
    if f4 is not None and f4 < 0.5:
        tags.append("weak_f4_volume_confirmation")
    if f5 is not None and f5 < 0.45:
        tags.append("weak_f5_rsi_confirmation")
    if risk_level.strip().lower() in {"high", "risk_high", "protect"}:
        tags.append("elevated_risk_level")
    if btc_trend_state.strip().lower() in {"down", "trend_down", "bear", "dumping"}:
        tags.append("btc_trend_unfavorable")
    if market_regime.strip().lower() in {"risk_off", "trend_down", "high_vol"}:
        tags.append("unfavorable_market_regime")
    if mae_bps is not None and mae_bps <= -120:
        tags.append("large_adverse_excursion")
    return tags


def _diagnosis(tags: Counter[str], loss_rows: list[dict[str, Any]]) -> str:
    if not loss_rows:
        return "no_loss_entries_observed"
    if not tags:
        return "losses_observed_without_common_feature"
    top = [tag for tag, _count in tags.most_common(3)]
    return "loss_common_features:" + ",".join(top)


def _market_bar_index(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if market_bars.is_empty():
        return {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in market_bars.to_dicts():
        symbol = normalize_symbol(row.get("symbol") or "")
        ts = _parse_datetime(row.get("ts"))
        if symbol and ts is not None:
            item = dict(row)
            item["_ts"] = ts
            rows.setdefault(symbol, []).append(item)
    for symbol_rows in rows.values():
        symbol_rows.sort(key=lambda item: item["_ts"])
    return rows


def _entry_position_in_24h_range(
    bars: list[dict[str, Any]],
    *,
    entry_ts: datetime,
    entry_px: float | None,
) -> float | None:
    if entry_px is None:
        return None
    start = entry_ts - timedelta(hours=24)
    window = [row for row in bars if start <= row["_ts"] <= entry_ts]
    highs = [_float(row.get("high")) for row in window]
    lows = [_float(row.get("low")) for row in window]
    high_values = [value for value in highs if value is not None]
    low_values = [value for value in lows if value is not None]
    if not high_values or not low_values:
        return None
    high = max(high_values)
    low = min(low_values)
    if high <= low:
        return None
    return (entry_px - low) / (high - low)


def _as_of_date(row: dict[str, Any], entry_ts: datetime, ctx: _Context) -> str:
    value = _field(row, "as_of_date", "paper_date", "date")
    if value:
        return value[:10]
    return entry_ts.date().isoformat() if entry_ts else ctx.as_of_date.isoformat()


def _generated_from_bundle(row: dict[str, Any], ctx: _Context) -> str:
    return (
        _field(row, "bundle_name", "bundle_sha256", "source_bundle_id", "bundle_ts")
        or ctx.generated_from_bundle_id
    )


def _summary_bundle_id(rows: list[dict[str, Any]], ctx: _Context) -> str:
    values = [
        str(row.get("generated_from_bundle_id") or "")
        for row in rows
        if str(row.get("generated_from_bundle_id") or "").strip()
    ]
    return max(values) if values else ctx.generated_from_bundle_id


def _latest_bundle_id(root: Path) -> str:
    candidates: list[str] = []
    for dataset in [
        root / V5_PAPER_STRATEGY_RUN_DATASET,
        root / "silver" / "v5_candidate_event",
        root / "gold" / "strategy_health_daily",
    ]:
        frame = read_parquet_dataset(dataset)
        if frame.is_empty():
            continue
        for column in ["bundle_name", "bundle_sha256", "bundle_ts", "latest_bundle_ts"]:
            if column in frame.columns:
                candidates.extend(
                    str(value)
                    for value in frame.get_column(column).drop_nulls().to_list()
                    if str(value).strip()
                )
    return max(candidates) if candidates else "not_observable"


def _parse_day(value: str | date | None) -> date:
    if isinstance(value, date):
        return value
    if value is None or str(value).strip().lower() == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(str(value)[:10])


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if parsed != parsed:
        return None
    return parsed


def _bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _mean(values: Any) -> float | None:
    clean = [value for value in (_float(value) for value in values) if value is not None]
    return float(sum(clean) / len(clean)) if clean else None


def _mix(rows: list[dict[str, Any]], column: str) -> str:
    counter = Counter(
        str(row.get(column) or "missing")
        for row in rows
        if str(row.get(column) or "").strip()
    )
    return safe_json_dumps(dict(counter.most_common()))


def _json_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _int(value: Any) -> int:
    parsed = _float(value)
    return int(parsed) if parsed is not None else 0


def _display_number(value: Any) -> str:
    parsed = _float(value)
    return "n/a" if parsed is None else f"{parsed:.4g}"


def _display_pct(value: Any) -> str:
    parsed = _float(value)
    return "n/a" if parsed is None else f"{parsed * 100:.2f}%"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "not_observable"
    return result.stdout.strip() or "not_observable"
