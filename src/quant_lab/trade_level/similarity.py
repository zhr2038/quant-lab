from __future__ import annotations

import statistics
from datetime import UTC, datetime, timedelta
from typing import Any

import polars as pl

TRADE_LEVEL_SIMILARITY_SCHEMA_VERSION = "trade_level_similarity_outcome.v0.2"
TRADE_LEVEL_SIMILARITY_SCHEMA = {
    "schema_version": pl.Utf8,
    "event_id": pl.Utf8,
    "decision_ts": pl.Datetime(time_zone="UTC"),
    "symbol": pl.Utf8,
    "strategy_candidate": pl.Utf8,
    "regime": pl.Utf8,
    "risk_level": pl.Utf8,
    "similar_sample_count": pl.Int64,
    "similar_mean_after_cost_bps": pl.Float64,
    "similar_median_after_cost_bps": pl.Float64,
    "similar_p25_after_cost_bps": pl.Float64,
    "similar_hit_rate": pl.Float64,
    "similar_max_adverse_bps": pl.Float64,
    "recent_7d_similar_mean": pl.Float64,
    "similarity_key": pl.Utf8,
    "created_at": pl.Datetime(time_zone="UTC"),
    "source": pl.Utf8,
}


def build_trade_level_similarity_outcome(
    events: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    created_at: datetime | None = None,
) -> pl.DataFrame:
    created = created_at or datetime.now(UTC)
    if events.is_empty():
        return pl.DataFrame(schema=TRADE_LEVEL_SIMILARITY_SCHEMA)
    labels_by_event = {str(row.get("event_id") or ""): row for row in labels.to_dicts()}
    event_rows = sorted(
        events.to_dicts(),
        key=lambda row: (
            _timestamp(row.get("decision_ts")) or datetime.min.replace(tzinfo=UTC),
            _text(row.get("event_id")),
        ),
    )
    rows: list[dict[str, Any]] = []
    for event in event_rows:
        event_ts = _timestamp(event.get("decision_ts"))
        selected_prior: list[
            tuple[dict[str, Any], dict[str, Any], float, int, datetime]
        ] = []
        for prior_event in event_rows:
            if not _is_prior_similar(event, prior_event, event_ts):
                continue
            prior_label = labels_by_event.get(str(prior_event.get("event_id") or ""))
            value, horizon, available_at = select_causal_similarity_outcome(
                prior_label,
                current_decision_ts=event_ts,
            )
            if (
                prior_label is not None
                and value is not None
                and horizon is not None
                and available_at is not None
            ):
                selected_prior.append(
                    (prior_event, prior_label, value, horizon, available_at)
                )
        values = [item[2] for item in selected_prior]
        recent_cutoff = event_ts - timedelta(days=7) if event_ts else None
        recent_values = [
            value
            for prior_event, _label, value, _horizon, _available_at in selected_prior
            if recent_cutoff is not None
            and (
                _timestamp(prior_event.get("decision_ts"))
                or datetime.min.replace(tzinfo=UTC)
            )
            >= recent_cutoff
        ]
        adverse = [
            _float(label.get(f"label_{horizon}h_mae_bps"))
            for _prior_event, label, _value, horizon, _available_at in selected_prior
        ]
        adverse = [value for value in adverse if value is not None]
        rows.append(
            {
                "schema_version": TRADE_LEVEL_SIMILARITY_SCHEMA_VERSION,
                "event_id": str(event.get("event_id") or ""),
                "decision_ts": event_ts,
                "symbol": _text(event.get("symbol")),
                "strategy_candidate": _text(event.get("strategy_candidate")),
                "regime": _text(event.get("regime")),
                "risk_level": _text(event.get("risk_level")),
                "similar_sample_count": len(values),
                "similar_mean_after_cost_bps": _mean(values),
                "similar_median_after_cost_bps": _median(values),
                "similar_p25_after_cost_bps": _quantile(values, 0.25),
                "similar_hit_rate": _hit_rate(values),
                "similar_max_adverse_bps": min(adverse) if adverse else None,
                "recent_7d_similar_mean": _mean(recent_values),
                "similarity_key": _similarity_key(event),
                "created_at": created,
                "source": "quant_lab.trade_level.similarity",
            }
        )
    return _similarity_frame(rows)


def _is_prior_similar(
    event: dict[str, Any],
    prior: dict[str, Any],
    event_ts: datetime | None,
) -> bool:
    prior_ts = _timestamp(prior.get("decision_ts"))
    if event_ts is None or prior_ts is None or prior_ts >= event_ts:
        return False
    event_id = _text(event.get("event_id"))
    prior_event_id = _text(prior.get("event_id"))
    if not event_id or not prior_event_id or event_id == prior_event_id:
        return False
    symbol_match = _text(event.get("symbol")) == _text(prior.get("symbol"))
    strategy_match = _text(event.get("strategy_candidate")) == _text(
        prior.get("strategy_candidate")
    )
    regime_match = _text(event.get("regime")) == _text(prior.get("regime"))
    risk_match = _text(event.get("risk_level")) == _text(prior.get("risk_level"))
    bucket_match = _alpha_bucket(event) == _alpha_bucket(prior) and _rank_bucket(
        event
    ) == _rank_bucket(prior)
    return symbol_match and (strategy_match or regime_match) and (risk_match or bucket_match)


def _similarity_key(row: dict[str, Any]) -> str:
    return "|".join(
        [
            _text(row.get("symbol")) or "UNKNOWN_SYMBOL",
            _text(row.get("strategy_candidate")) or "UNKNOWN_STRATEGY",
            _text(row.get("regime")) or "UNKNOWN_REGIME",
            _text(row.get("risk_level")) or "UNKNOWN_RISK",
            _alpha_bucket(row),
            _rank_bucket(row),
            _cost_bucket(row),
        ]
    )


def select_causal_similarity_outcome(
    label: dict[str, Any] | None,
    *,
    current_decision_ts: datetime | None,
) -> tuple[float | None, int | None, datetime | None]:
    if not label:
        return None, None, None
    current_ts = _timestamp(current_decision_ts)
    if current_ts is None:
        return None, None, None
    label_decision_ts = _timestamp(label.get("decision_ts"))
    for horizon in (24, 8, 4):
        field = f"label_{horizon}h_after_cost_bps"
        value = _float(label.get(field))
        if value is None:
            continue
        available_at = _timestamp(label.get(f"label_{horizon}h_available_at"))
        if available_at is None and label_decision_ts is not None:
            available_at = label_decision_ts + timedelta(hours=horizon)
        if available_at is not None and available_at <= current_ts:
            return value, horizon, available_at
    return None, None, None


def _alpha_bucket(row: dict[str, Any]) -> str:
    score = _float(row.get("alpha6_score"))
    if score is None:
        return "alpha_missing"
    if score >= 0.95:
        return "alpha_ge_0_95"
    if score >= 0.85:
        return "alpha_ge_0_85"
    if score >= 0.75:
        return "alpha_ge_0_75"
    return "alpha_lt_0_75"


def _rank_bucket(row: dict[str, Any]) -> str:
    rank = _int(row.get("rank"))
    if rank == 1:
        return "rank_1"
    if rank is not None and rank <= 3:
        return "rank_2_3"
    return "rank_other"


def _cost_bucket(row: dict[str, Any]) -> str:
    cost = _float(row.get("cost_bps"))
    if cost is None:
        return "cost_missing"
    if cost <= 20:
        return "cost_le_20"
    if cost <= 50:
        return "cost_le_50"
    return "cost_gt_50"


def _similarity_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=TRADE_LEVEL_SIMILARITY_SCHEMA)
    return pl.DataFrame(rows, schema=TRADE_LEVEL_SIMILARITY_SCHEMA, orient="row").select(
        [
            pl.col(name).cast(dtype, strict=False).alias(name)
            for name, dtype in TRADE_LEVEL_SIMILARITY_SCHEMA.items()
        ]
    )


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * q)))
    return ordered[index]


def _hit_rate(values: list[float]) -> float | None:
    return sum(1 for value in values if value > 0.0) / len(values) if values else None


def _timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if value in (None, ""):
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"none", "null", "nan"} else text


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _int(value: Any) -> int | None:
    number = _float(value)
    return int(number) if number is not None else None
