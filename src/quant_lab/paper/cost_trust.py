from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class _TrustRank(IntEnum):
    BLOCK = 0
    PAPER_ONLY = 1
    CANARY = 2
    SCALE_READY = 3


class StrategyCostTrust(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy_id: str
    cost_trust_level: str
    actual_sample_count: int = Field(ge=0)
    mixed_sample_count: int = Field(ge=0)
    proxy_sample_count: int = Field(ge=0)
    sample_age_seconds: float | None = Field(default=None, ge=0)
    coverage_dimensions: list[str] = Field(default_factory=list)
    missing_dimensions: list[str] = Field(default_factory=list)
    evaluated_condition_count: int = Field(ge=0)
    required_condition_count: int = Field(ge=0)
    research_cost_usable: bool = False
    paper_cost_usable: bool = False
    canary_cost_usable: bool = False
    live_cost_usable: bool = False
    source: str = "paper.strategy_cost_trust.v1"


REQUIRED_COST_DIMENSIONS = (
    "symbol",
    "notional_bucket",
    "market_regime",
    "liquidity_role",
    "order_leg",
    "spread_bucket",
    "volatility_bucket",
)


def evaluate_strategy_cost_trust(
    *,
    strategy_id: str,
    required_conditions: Iterable[Mapping[str, Any]],
    observations: Iterable[Mapping[str, Any]],
    now: datetime | None = None,
    max_age_seconds: float = 7 * 24 * 60 * 60,
) -> StrategyCostTrust:
    current = (now or datetime.now(UTC)).astimezone(UTC)
    required = [dict(item) for item in required_conditions]
    observed = [dict(item) for item in observations]
    source_counts: Counter[str] = Counter()
    ages: list[float] = []
    condition_levels: list[_TrustRank] = []
    covered_dimensions: set[str] = set()
    missing_dimensions: set[str] = set()

    for condition in required:
        missing_dimensions.update(
            dimension
            for dimension in REQUIRED_COST_DIMENSIONS
            if not str(condition.get(dimension) or "").strip()
        )
        matches = [row for row in observed if _matches(condition, row)]
        if not matches:
            condition_levels.append(_TrustRank.BLOCK)
            missing_dimensions.update(
                dimension
                for dimension in REQUIRED_COST_DIMENSIONS
                if str(condition.get(dimension) or "").strip()
            )
            continue
        covered_dimensions.update(REQUIRED_COST_DIMENSIONS)
        fresh_matches: list[Mapping[str, Any]] = []
        fresh_exact_matches: list[Mapping[str, Any]] = []
        for row in matches:
            source = str(row.get("cost_source") or row.get("source") or "proxy").lower()
            source_counts[_source_class(source)] += max(_to_int(row.get("sample_count")), 1)
            age = _observation_age_seconds(row, current)
            if age is not None:
                ages.append(age)
            if age is not None and age <= max_age_seconds:
                fresh_matches.append(row)
                if _matches_exact(condition, row):
                    fresh_exact_matches.append(row)
        condition_levels.append(
            _condition_level(fresh_matches, fresh_exact_matches)
            if fresh_matches
            else _TrustRank.PAPER_ONLY
        )

    overall = min(condition_levels, default=_TrustRank.BLOCK)
    return StrategyCostTrust(
        strategy_id=strategy_id,
        cost_trust_level=overall.name,
        actual_sample_count=source_counts["actual"],
        mixed_sample_count=source_counts["mixed"],
        proxy_sample_count=source_counts["proxy"],
        sample_age_seconds=max(ages) if ages else None,
        coverage_dimensions=sorted(covered_dimensions - missing_dimensions),
        missing_dimensions=sorted(missing_dimensions),
        evaluated_condition_count=len(condition_levels),
        required_condition_count=len(required),
        research_cost_usable=overall >= _TrustRank.PAPER_ONLY,
        paper_cost_usable=overall >= _TrustRank.PAPER_ONLY,
        canary_cost_usable=overall >= _TrustRank.CANARY,
        live_cost_usable=overall >= _TrustRank.SCALE_READY,
    )


def build_strategy_cost_trust_frame(
    proposals: pl.DataFrame,
    cost_buckets: pl.DataFrame,
    *,
    paper_runtime_cost_evidence: pl.DataFrame | None = None,
    now: datetime | None = None,
) -> pl.DataFrame:
    columns = {
        "strategy_id": pl.Utf8,
        "cost_trust_level": pl.Utf8,
        "actual_sample_count": pl.Int64,
        "mixed_sample_count": pl.Int64,
        "proxy_sample_count": pl.Int64,
        "sample_age_seconds": pl.Float64,
        "coverage_dimensions": pl.Utf8,
        "missing_dimensions": pl.Utf8,
        "evaluated_condition_count": pl.Int64,
        "required_condition_count": pl.Int64,
        "research_cost_usable": pl.Boolean,
        "paper_cost_usable": pl.Boolean,
        "canary_cost_usable": pl.Boolean,
        "live_cost_usable": pl.Boolean,
        "source": pl.Utf8,
        "created_at": pl.Datetime(time_zone="UTC"),
    }
    if proposals.is_empty():
        return pl.DataFrame(schema=columns)
    observations = [_normalize_cost_observation(row) for row in cost_buckets.to_dicts()]
    if paper_runtime_cost_evidence is not None and not paper_runtime_cost_evidence.is_empty():
        observations.extend(
            _normalize_paper_runtime_cost_observation(row)
            for row in paper_runtime_cost_evidence.to_dicts()
            if _to_int(row.get("closed_trade_count")) > 0
            and _to_int(row.get("cost_observed_count")) > 0
        )
    created = (now or datetime.now(UTC)).astimezone(UTC)
    rows = []
    for proposal in proposals.to_dicts():
        strategy_id = str(proposal.get("strategy_id") or proposal.get("proposal_id") or "")
        notional = _to_float(proposal.get("paper_notional_usdt"))
        required = _proposal_required_conditions(proposal, notional=notional)
        result = evaluate_strategy_cost_trust(
            strategy_id=strategy_id,
            required_conditions=required,
            observations=observations,
            now=created,
        )
        row = result.model_dump()
        row["coverage_dimensions"] = ",".join(result.coverage_dimensions)
        row["missing_dimensions"] = ",".join(result.missing_dimensions)
        row["created_at"] = created
        rows.append(row)
    return (
        pl.DataFrame(rows, infer_schema_length=None)
        .cast(columns, strict=False)
        .select(list(columns))
    )


def _condition_level(
    rows: list[Mapping[str, Any]],
    exact_rows: list[Mapping[str, Any]],
) -> _TrustRank:
    if not rows:
        return _TrustRank.BLOCK
    local = Counter()
    for row in exact_rows:
        source = _source_class(str(row.get("cost_source") or row.get("source") or "proxy"))
        local[source] += max(_to_int(row.get("sample_count")), 1)
    if local["actual"] >= 30:
        return _TrustRank.SCALE_READY
    if local["actual"] >= 10 or (local["actual"] >= 5 and local["mixed"] >= 10):
        return _TrustRank.CANARY
    if rows:
        return _TrustRank.PAPER_ONLY
    return _TrustRank.BLOCK


def _matches(required: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    for dimension in REQUIRED_COST_DIMENSIONS:
        expected = str(required.get(dimension) or "").strip().lower()
        actual = str(observed.get(dimension) or "").strip().lower()
        if (
            expected
            and expected not in {"*", "any", "all"}
            and actual not in {"*", "any", "all"}
            and expected != actual
        ):
            return False
    return True


def _matches_exact(required: Mapping[str, Any], observed: Mapping[str, Any]) -> bool:
    for dimension in REQUIRED_COST_DIMENSIONS:
        expected = str(required.get(dimension) or "").strip().lower()
        actual = str(observed.get(dimension) or "").strip().lower()
        if expected in {"", "*", "any", "all"}:
            continue
        if actual in {"", "*", "any", "all"} or expected != actual:
            return False
    return True


def _source_class(source: str) -> str:
    value = source.lower()
    if "bootstrap" in value or "probe" in value:
        return "proxy"
    if "mixed" in value:
        return "mixed"
    if "actual" in value or "fill" in value:
        return "actual"
    return "proxy"


def _proposal_required_conditions(
    proposal: Mapping[str, Any],
    *,
    notional: float,
) -> list[dict[str, str]]:
    regimes = _entry_rule_regimes(proposal.get("entry_rule")) or ["any"]
    return [
        {
            "symbol": _normalize_symbol(proposal.get("symbol")),
            "notional_bucket": _notional_bucket(notional),
            "market_regime": regime.lower(),
            "liquidity_role": "taker",
            "order_leg": leg,
            "spread_bucket": "any",
            "volatility_bucket": "any",
        }
        for regime in regimes
        for leg in ("entry", "exit")
    ]


def _entry_rule_regimes(value: Any) -> list[str]:
    payload = value
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return []
    if not isinstance(payload, Mapping):
        return []
    output: list[str] = []
    if str(payload.get("operator") or "") == "regime_in":
        output.extend(str(item) for item in (payload.get("values") or []) if str(item))
    for child in payload.get("children") or []:
        output.extend(_entry_rule_regimes(child))
    return list(dict.fromkeys(output))


def _observation_age_seconds(row: Mapping[str, Any], now: datetime) -> float | None:
    raw = row.get("observed_at") or row.get("as_of_ts") or row.get("updated_at")
    if raw in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max((now - parsed.astimezone(UTC)).total_seconds(), 0.0)


def _to_int(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", "-").replace("_", "-")


def _notional_bucket(notional: float) -> str:
    if notional <= 20:
        return "0_20"
    if notional <= 100:
        return "20_100"
    if notional <= 1000:
        return "100_1000"
    return "1000_plus"


def _normalize_cost_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    minimum = _to_float(row.get("min_notional_usdt"))
    maximum = _to_float(row.get("max_notional_usdt"))
    representative = maximum or minimum
    return {
        "symbol": _normalize_symbol(row.get("symbol")),
        "notional_bucket": str(row.get("notional_bucket") or _notional_bucket(representative)),
        "market_regime": str(row.get("market_regime") or row.get("regime") or "any").lower(),
        "liquidity_role": str(row.get("liquidity_role") or row.get("maker_taker") or "any").lower(),
        "order_leg": str(row.get("order_leg") or row.get("entry_exit") or "any").lower(),
        "spread_bucket": str(row.get("spread_bucket") or "any").lower(),
        "volatility_bucket": str(row.get("volatility_bucket") or "any").lower(),
        "cost_source": row.get("cost_source") or row.get("source") or "proxy",
        "sample_count": row.get("sample_count") or row.get("n") or 0,
        "observed_at": row.get("observed_at") or row.get("as_of_ts") or row.get("created_at"),
    }


def _normalize_paper_runtime_cost_observation(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "symbol": _normalize_symbol(row.get("symbol")),
        "notional_bucket": "any",
        "market_regime": "any",
        "liquidity_role": "any",
        "order_leg": "any",
        "spread_bucket": "any",
        "volatility_bucket": "any",
        "cost_source": row.get("cost_source") or "configured_conservative_paper",
        "sample_count": row.get("closed_trade_count") or row.get("cost_observed_count") or 0,
        "observed_at": row.get("observed_at")
        or row.get("as_of_ts")
        or row.get("created_at")
        or row.get("ingest_ts"),
    }
