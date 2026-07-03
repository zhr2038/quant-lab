import logging
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from quant_lab.contracts.models import CostEstimate, FillEvent
from quant_lab.costs.probe import (
    build_cost_probe_fill_bill_match,
    cost_probe_private_fill_count_by_symbol,
    cost_probe_terminal_fill_count_by_symbol,
)
from quant_lab.data.lake import read_parquet_dataset, read_parquet_lazy
from quant_lab.symbols import normalize_optional_symbol, normalize_symbol

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_COST_BPS = 25.0
SUPPORTED_COST_QUANTILES = {"p50", "p75", "p90"}
COST_BUCKET_STALE_SECONDS = 36 * 60 * 60
MIN_TRUSTED_COST_SAMPLE_COUNT = 30
CONFIG_FEE_BPS = 10.0
CONFIG_SLIPPAGE_BPS = 2.0
CONFIG_DELAY_COST_BPS = 0.0
PUBLIC_PROXY_UNCERTAINTY_BUFFER_BPS = 2.0
SMALL_SAMPLE_UNCERTAINTY_BUFFER_BPS = 3.0
STALE_BUCKET_UNCERTAINTY_BUFFER_BPS = 5.0
DEFAULT_LIVE_UNIVERSE_SYMBOLS = ("BNB-USDT", "BTC-USDT", "ETH-USDT", "SOL-USDT")
LIVE_UNIVERSE_COST_COVERAGE_FIELDS = [
    "generated_at",
    "symbol",
    "latest_source",
    "latest_actual_or_mixed_source",
    "effective_cost_source",
    "cost_evidence_tier",
    "anchored_mixed_proxy_candidate",
    "actual_or_mixed_direct",
    "mixed_proxy_eligible",
    "stale_actual_or_mixed",
    "fee_fresh",
    "spread_fresh",
    "slippage_fresh",
    "actual_or_mixed_covered",
    "sample_count",
    "actual_fill_count",
    "mixed_fill_count",
    "proxy_sample_count",
    "cost_probe_fill_count",
    "strategy_live_fill_count",
    "private_fill_count",
    "live_cost_sample_count",
    "sample_origin_mix",
    "eligible_for_live_cost_coverage",
    "fee_bps_p75",
    "spread_bps_p75",
    "slippage_bps_p75",
    "total_cost_bps_p75",
    "latest_created_at",
    "latest_actual_or_mixed_created_at",
    "latest_actual_or_mixed_age_sec",
    "coverage_reason",
    "actual_or_mixed_cost_coverage_live_universe",
    "target_coverage",
    "coverage_status",
    "live_order_effect",
]
COST_BOOTSTRAP_READINESS_FIELDS = [
    "generated_at",
    "symbol",
    "bootstrap_state",
    "cost_evidence_tier",
    "cost_trust_level",
    "actual_fill_count",
    "mixed_fill_count",
    "cost_probe_fill_count",
    "strategy_live_fill_count",
    "private_fill_count",
    "bill_matched_count",
    "matched_bill_count",
    "fee_available",
    "slippage_available",
    "spread_available",
    "sample_count",
    "live_cost_sample_count",
    "trusted_sample_count",
    "latest_fill_ts",
    "latest_probe_ts",
    "latest_probe_fill_ts",
    "latest_bill_ts",
    "fill_match_status",
    "bill_match_status",
    "fee_match_status",
    "fee_match_diff_usdt",
    "latest_cost_source",
    "roundtrip_cost_p50_bps",
    "roundtrip_cost_p75_bps",
    "roundtrip_cost_p90_bps",
    "trusted_for_paper",
    "trusted_for_live",
    "actual_or_mixed_bootstrap_covered",
    "actual_or_mixed_trusted_covered",
    "actual_or_mixed_bootstrap_coverage_live_universe",
    "actual_or_mixed_trusted_coverage_live_universe",
    "target_trusted_coverage",
    "coverage_status",
    "live_order_effect",
    "next_action",
]


def evaluate_live_universe_cost_coverage(
    cost_bucket_daily: pl.DataFrame | None,
    *,
    live_symbols: Iterable[str] = DEFAULT_LIVE_UNIVERSE_SYMBOLS,
    target_coverage: float = 0.50,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    """Evaluate actual/mixed coverage for the live universe.

    A public proxy row can count as mixed only when at least one live symbol has
    direct actual/mixed cost evidence to anchor the fee/slippage side of the model.
    """

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    symbols = sorted({normalize_symbol(symbol) for symbol in live_symbols if symbol})
    cost_rows = _normalized_cost_rows(cost_bucket_daily)
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {
        symbol: [row for row in cost_rows if _row_symbol(row) == symbol]
        for symbol in symbols
    }
    direct_symbols = {
        symbol
        for symbol, symbol_rows in rows_by_symbol.items()
        if _latest_fresh_actual_or_mixed_row(symbol_rows, reference_time=generated) is not None
    }
    has_live_actual_anchor = bool(direct_symbols)

    output: list[dict[str, Any]] = []
    covered_count = 0
    for symbol in symbols:
        symbol_rows = rows_by_symbol.get(symbol, [])
        latest = _latest_cost_row_for_coverage(symbol_rows)
        latest_actual = _latest_actual_or_mixed_row(symbol_rows)
        fresh_actual = _latest_fresh_actual_or_mixed_row(
            symbol_rows,
            reference_time=generated,
        )
        latest_proxy = _latest_public_proxy_row(symbol_rows)
        direct = fresh_actual is not None
        stale_actual_or_mixed = (
            _row_is_stale(latest_actual, reference_time=generated)
            if latest_actual is not None
            else False
        )
        latest_is_bootstrap_probe = (
            latest is not None and _cost_source(latest) == "bootstrap_cost_probe"
        )
        anchored_mixed_proxy_candidate = (
            not direct
            and not latest_is_bootstrap_probe
            and latest_proxy is not None
            and has_live_actual_anchor
        )
        covered = direct
        covered_count += int(covered)
        fee_fresh = _component_fresh(fresh_actual, "fee_bps_p75", reference_time=generated)
        slippage_fresh = _component_fresh(
            fresh_actual,
            "slippage_bps_p75",
            reference_time=generated,
        )
        spread_fresh = _component_fresh(
            latest_proxy or fresh_actual,
            "spread_bps_p75",
            reference_time=generated,
        )
        effective_source = (
            _cost_source(fresh_actual)
            if fresh_actual is not None
            else _cost_source(latest)
        )
        source_row = (
            fresh_actual
            or (latest if latest_is_bootstrap_probe else None)
            or latest_proxy
            or latest_actual
            or latest
            or {}
        )
        output.append(
            {
                "generated_at": generated.isoformat().replace("+00:00", "Z"),
                "symbol": symbol,
                "latest_source": _cost_source(latest) or "missing",
                "latest_actual_or_mixed_source": _cost_source(latest_actual) or "missing",
                "effective_cost_source": effective_source or "missing",
                "cost_evidence_tier": _cost_evidence_tier(
                    direct=direct,
                    has_live_actual_anchor=has_live_actual_anchor,
                    latest=latest,
                    latest_actual=latest_actual,
                    latest_proxy=latest_proxy,
                    stale_actual_or_mixed=stale_actual_or_mixed,
                ),
                "anchored_mixed_proxy_candidate": anchored_mixed_proxy_candidate,
                "actual_or_mixed_direct": direct,
                "mixed_proxy_eligible": anchored_mixed_proxy_candidate,
                "stale_actual_or_mixed": stale_actual_or_mixed,
                "latest_actual_or_mixed_stale": stale_actual_or_mixed,
                "fee_fresh": fee_fresh,
                "spread_fresh": spread_fresh,
                "slippage_fresh": slippage_fresh,
                "actual_or_mixed_covered": covered,
                "sample_count": _int_value(source_row.get("sample_count")),
                "latest_sample_count": _int_value(source_row.get("sample_count")),
                "actual_fill_count": _int_value(source_row.get("actual_fill_count")),
                "mixed_fill_count": _int_value(source_row.get("mixed_fill_count")),
                "proxy_sample_count": _coverage_proxy_sample_count(source_row),
                "cost_probe_fill_count": _int_value(source_row.get("cost_probe_fill_count")),
                "strategy_live_fill_count": _int_value(
                    source_row.get("strategy_live_fill_count")
                ),
                "private_fill_count": _int_value(source_row.get("private_fill_count")),
                "live_cost_sample_count": _live_cost_sample_count(source_row),
                "sample_origin_mix": _sample_origin_mix_value(source_row),
                "eligible_for_live_cost_coverage": _row_live_cost_eligible(source_row),
                "fee_bps_p75": _float_value(source_row, "fee_bps_p75"),
                "spread_bps_p75": _float_value(source_row, "spread_bps_p75"),
                "slippage_bps_p75": _float_value(source_row, "slippage_bps_p75"),
                "total_cost_bps_p75": _float_value(source_row, "total_cost_bps_p75"),
                "latest_created_at": _coverage_ts(source_row),
                "latest_actual_or_mixed_created_at": _coverage_ts(latest_actual or {}),
                "latest_actual_or_mixed_age_sec": _coverage_age_sec(
                    latest_actual or {},
                    reference_time=generated,
                ),
                "coverage_reason": _coverage_reason(
                    direct=direct,
                    anchored_mixed_proxy_candidate=anchored_mixed_proxy_candidate,
                    has_live_actual_anchor=has_live_actual_anchor,
                    latest_actual=latest_actual,
                    stale_actual_or_mixed=stale_actual_or_mixed,
                    latest_proxy=latest_proxy,
                    latest=latest,
                ),
                "actual_or_mixed_cost_coverage_live_universe": None,
                "target_coverage": target_coverage,
                "coverage_status": "UNKNOWN",
                "live_order_effect": "read_only_no_live_order",
            }
        )

    denominator = max(len(symbols), 1)
    coverage_rate = covered_count / denominator
    status = "PASS" if coverage_rate >= target_coverage else "WARNING"
    for row in output:
        row["actual_or_mixed_cost_coverage_live_universe"] = coverage_rate
        row["coverage_status"] = status
    return {
        "rows": output,
        "coverage_rate": coverage_rate,
        "target_coverage": target_coverage,
        "coverage_status": status,
        "covered_symbols": sorted(
            str(row["symbol"]) for row in output if bool(row["actual_or_mixed_covered"])
        ),
        "direct_symbols": sorted(
            str(row["symbol"]) for row in output if bool(row["actual_or_mixed_direct"])
        ),
        "mixed_proxy_symbols": sorted(
            str(row["symbol"]) for row in output if bool(row["mixed_proxy_eligible"])
        ),
        "stale_actual_or_mixed_symbols": sorted(
            str(row["symbol"]) for row in output if bool(row["stale_actual_or_mixed"])
        ),
        "missing_symbols": sorted(
            str(row["symbol"]) for row in output if row["latest_source"] == "missing"
        ),
        "proxy_only_symbols": sorted(
            str(row["symbol"])
            for row in output
            if row["latest_source"] in {"public_spread_proxy", "public_proxy"}
            and not bool(row["actual_or_mixed_covered"])
        ),
        "detail_by_symbol": {str(row["symbol"]): row for row in output},
    }


def build_live_universe_cost_coverage(
    cost_bucket_daily: pl.DataFrame | None,
    *,
    live_symbols: Iterable[str] = DEFAULT_LIVE_UNIVERSE_SYMBOLS,
    target_coverage: float = 0.50,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    """Build a transparent actual/mixed coverage report for the live universe."""

    evaluation = evaluate_live_universe_cost_coverage(
        cost_bucket_daily,
        live_symbols=live_symbols,
        target_coverage=target_coverage,
        generated_at=generated_at,
    )
    output = list(evaluation["rows"])
    if not output:
        return pl.DataFrame(schema={field: pl.Utf8 for field in LIVE_UNIVERSE_COST_COVERAGE_FIELDS})
    return pl.DataFrame(output, infer_schema_length=None).select(
        LIVE_UNIVERSE_COST_COVERAGE_FIELDS
    )


def build_cost_bootstrap_readiness(
    cost_bucket_daily: pl.DataFrame | None,
    *,
    v5_order_lifecycle: pl.DataFrame | None = None,
    v5_cost_probe_order_events: pl.DataFrame | None = None,
    v5_cost_probe_roundtrip_events: pl.DataFrame | None = None,
    okx_private_readonly_fills: pl.DataFrame | None = None,
    okx_private_readonly_bills: pl.DataFrame | None = None,
    live_symbols: Iterable[str] = DEFAULT_LIVE_UNIVERSE_SYMBOLS,
    target_trusted_coverage: float = 0.50,
    min_trusted_sample_count: int = MIN_TRUSTED_COST_SAMPLE_COUNT,
    generated_at: datetime | None = None,
) -> pl.DataFrame:
    """Explain cost bootstrap state without weakening live coverage gates."""

    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    symbols = sorted({normalize_symbol(symbol) for symbol in live_symbols if symbol})
    cost_rows = _normalized_cost_rows(cost_bucket_daily)
    rows_by_symbol: dict[str, list[dict[str, Any]]] = {
        symbol: [row for row in cost_rows if _row_symbol(row) == symbol]
        for symbol in symbols
    }
    lifecycle_by_symbol = _rows_by_symbol(v5_order_lifecycle)
    private_fills_by_symbol = _rows_by_symbol(okx_private_readonly_fills)
    cost_probe_private_fills_by_symbol = cost_probe_private_fill_count_by_symbol(
        okx_private_readonly_fills,
        v5_cost_probe_order_events,
        v5_cost_probe_roundtrip_events,
    )
    cost_probe_event_fills_by_symbol = cost_probe_terminal_fill_count_by_symbol(
        v5_cost_probe_order_events,
        v5_cost_probe_roundtrip_events,
    )
    cost_probe_fill_bill_match_by_symbol = _rows_by_symbol(
        build_cost_probe_fill_bill_match(
            v5_cost_probe_order_events,
            v5_cost_probe_roundtrip_events,
            okx_private_readonly_fills,
            okx_private_readonly_bills,
            generated_at=generated,
        )
    )
    cost_probe_roundtrips_by_symbol = _rows_by_symbol(v5_cost_probe_roundtrip_events)
    bills_by_symbol = _rows_by_symbol(okx_private_readonly_bills)

    output: list[dict[str, Any]] = []
    bootstrap_covered = 0
    trusted_covered = 0
    for symbol in symbols:
        symbol_rows = rows_by_symbol.get(symbol, [])
        latest = _latest_cost_row_for_coverage(symbol_rows)
        latest_actual = _latest_actual_or_mixed_row(symbol_rows)
        latest_bootstrap = _latest_bootstrap_cost_probe_row(symbol_rows)
        fresh_actual = _latest_fresh_actual_or_mixed_row(
            symbol_rows,
            reference_time=generated,
        )
        latest_proxy = _latest_public_proxy_row(symbol_rows)
        fresh_bootstrap = latest_bootstrap is not None and not _row_is_stale(
            latest_bootstrap,
            reference_time=generated,
        )
        source_row = (
            fresh_actual
            or (latest_bootstrap if fresh_bootstrap else None)
            or latest_actual
            or latest_proxy
            or latest
            or {}
        )
        lifecycle_rows = lifecycle_by_symbol.get(symbol, [])
        filled_lifecycle = [row for row in lifecycle_rows if _lifecycle_row_filled(row)]
        fresh_filled_lifecycle = [
            row
            for row in filled_lifecycle
            if _event_row_is_fresh(row, reference_time=generated)
        ]
        cost_probe_rows = [
            row for row in fresh_filled_lifecycle if _lifecycle_row_is_cost_probe(row)
        ]
        strategy_live_rows = [
            row for row in fresh_filled_lifecycle if not _lifecycle_row_is_cost_probe(row)
        ]
        private_fill_rows = private_fills_by_symbol.get(symbol, [])
        cost_probe_private_fill_count = min(
            len(private_fill_rows),
            cost_probe_private_fills_by_symbol.get(symbol, 0),
        )
        non_probe_private_fill_count = max(
            len(private_fill_rows) - cost_probe_private_fill_count,
            0,
        )
        bill_rows = bills_by_symbol.get(symbol, [])
        probe_roundtrip_rows = cost_probe_roundtrips_by_symbol.get(symbol, [])
        latest_probe_roundtrip = _latest_probe_roundtrip_row(probe_roundtrip_rows)
        fill_bill_match_rows = cost_probe_fill_bill_match_by_symbol.get(symbol, [])
        latest_fill_bill_match = _latest_probe_fill_bill_match_row(fill_bill_match_rows)

        strategy_live_fill_count = max(
            _int_value(source_row.get("strategy_live_fill_count")),
            len(strategy_live_rows),
        )
        private_fill_count = max(
            _int_value(source_row.get("private_fill_count")),
            non_probe_private_fill_count,
        )
        actual_fill_count = max(
            _int_value(source_row.get("actual_fill_count")),
            strategy_live_fill_count,
            private_fill_count,
        )
        mixed_fill_count = _int_value(source_row.get("mixed_fill_count"))
        cost_probe_fill_count = max(
            _int_value(source_row.get("cost_probe_fill_count")),
            _int_value(latest_bootstrap.get("cost_probe_fill_count") if latest_bootstrap else 0),
            len(cost_probe_rows),
            cost_probe_event_fills_by_symbol.get(symbol, 0),
            cost_probe_private_fill_count,
        )
        live_cost_sample_count = _live_cost_sample_count(
            source_row,
            strategy_live_fill_count=strategy_live_fill_count,
            private_fill_count=private_fill_count,
        )
        bill_matched_count = (
            _probe_fill_bill_matched_count(fill_bill_match_rows)
            if fill_bill_match_rows
            else len(bill_rows)
        )
        sample_count = max(
            _int_value(source_row.get("sample_count")),
            live_cost_sample_count + cost_probe_fill_count,
        )
        fee_available = (
            _positive_cost_value(source_row, "fee_bps_p75")
            or _any_positive(filled_lifecycle, "fee_bps", "fee_usdt", "fee")
            or bool(bill_rows)
        )
        slippage_available = (
            _positive_cost_value(source_row, "slippage_bps_p75")
            or _any_positive(
                filled_lifecycle,
                "arrival_slippage_bps",
                "delay_cost_bps",
                "slippage_bps",
            )
            or _any_row_has_all(filled_lifecycle, "arrival_mid", "avg_fill_px")
        )
        spread_available = (
            _positive_cost_value(source_row, "spread_bps_p75")
            or _any_positive(
                filled_lifecycle,
                "arrival_spread_bps",
                "spread_bps",
                "spread_bps_at_decision",
            )
            or _any_row_has_all(filled_lifecycle, "arrival_bid", "arrival_ask")
        )
        direct_fresh = fresh_actual is not None
        stale_actual = latest_actual is not None and _row_is_stale(
            latest_actual,
            reference_time=generated,
        )
        cost_probe_bootstrap_ready = (
            (fresh_bootstrap and cost_probe_fill_count >= 2)
            or cost_probe_event_fills_by_symbol.get(symbol, 0) >= 2
            or (latest_bootstrap is None and len(cost_probe_rows) > 0)
        )
        bootstrap_ready = direct_fresh or cost_probe_bootstrap_ready or private_fill_count > 0
        trusted_sample_count = (
            live_cost_sample_count
            if direct_fresh
            and fee_available
            and slippage_available
            and spread_available
            else 0
        )
        trusted_live = trusted_sample_count >= min_trusted_sample_count
        paper_ready = (
            bootstrap_ready
            or (_is_public_proxy_source(_cost_source(latest_proxy)) if latest_proxy else False)
            or (
                bool(latest)
                and _cost_source(latest) not in {"", "global_default"}
                and not _row_is_stale(latest, reference_time=generated)
            )
        )
        bootstrap_covered += int(bootstrap_ready)
        trusted_covered += int(trusted_live)
        state = _cost_bootstrap_state(
            latest=latest,
            latest_actual=latest_actual,
            latest_proxy=latest_proxy,
            stale_actual=stale_actual,
            trusted_live=trusted_live,
            actual_fill_count=actual_fill_count,
            mixed_fill_count=mixed_fill_count,
            cost_probe_fill_count=cost_probe_fill_count,
            sample_count=sample_count,
            min_trusted_sample_count=min_trusted_sample_count,
        )
        output.append(
            {
                "generated_at": generated.isoformat().replace("+00:00", "Z"),
                "symbol": symbol,
                "bootstrap_state": state,
                "cost_evidence_tier": _cost_bootstrap_evidence_tier(state),
                "cost_trust_level": _cost_bootstrap_trust_level(
                    state,
                    trusted_for_live=trusted_live,
                    trusted_for_paper=paper_ready,
                ),
                "actual_fill_count": actual_fill_count,
                "mixed_fill_count": mixed_fill_count,
                "cost_probe_fill_count": cost_probe_fill_count,
                "strategy_live_fill_count": strategy_live_fill_count,
                "private_fill_count": private_fill_count,
                "bill_matched_count": bill_matched_count,
                "matched_bill_count": bill_matched_count,
                "fee_available": fee_available,
                "slippage_available": slippage_available,
                "spread_available": spread_available,
                "sample_count": sample_count,
                "live_cost_sample_count": live_cost_sample_count,
                "trusted_sample_count": trusted_sample_count,
                "latest_fill_ts": _latest_row_ts(
                    [*filled_lifecycle, *private_fill_rows],
                    "last_fill_ts",
                    "fill_ts",
                    "ts_utc",
                    "ts",
                    "timestamp",
                    "trade_ts",
                    "ingest_ts",
                ),
                "latest_probe_ts": _latest_row_ts(
                    probe_roundtrip_rows,
                    "event_ts",
                    "closed_at",
                    "generated_at",
                    "ingest_ts",
                ),
                "latest_probe_fill_ts": _latest_row_ts(
                    [*probe_roundtrip_rows, *cost_probe_rows],
                    "closed_at",
                    "event_ts",
                    "last_fill_ts",
                    "fill_ts",
                    "ts_utc",
                    "ingest_ts",
                ),
                "latest_bill_ts": _latest_row_ts(
                    [*bill_rows, *fill_bill_match_rows],
                    "ts_utc",
                    "ts",
                    "timestamp",
                    "bill_ts",
                    "generated_at",
                    "ingest_ts",
                ),
                "fill_match_status": _probe_fill_match_status(
                    cost_probe_fill_count,
                    latest_probe_roundtrip,
                ),
                "bill_match_status": _probe_bill_match_status(
                    latest_probe_roundtrip,
                    bill_matched_count,
                    latest_fill_bill_match,
                ),
                "fee_match_status": _probe_fee_match_status(
                    latest_probe_roundtrip,
                    latest_fill_bill_match,
                ),
                "fee_match_diff_usdt": _probe_fee_match_diff_usdt(
                    latest_probe_roundtrip,
                    latest_fill_bill_match,
                ),
                "latest_cost_source": _cost_source(source_row) or _cost_source(latest) or "missing",
                "roundtrip_cost_p50_bps": _roundtrip_cost_value(source_row, "p50"),
                "roundtrip_cost_p75_bps": _roundtrip_cost_value(source_row, "p75"),
                "roundtrip_cost_p90_bps": _roundtrip_cost_value(source_row, "p90"),
                "trusted_for_paper": paper_ready,
                "trusted_for_live": trusted_live,
                "actual_or_mixed_bootstrap_covered": bootstrap_ready,
                "actual_or_mixed_trusted_covered": trusted_live,
                "actual_or_mixed_bootstrap_coverage_live_universe": None,
                "actual_or_mixed_trusted_coverage_live_universe": None,
                "target_trusted_coverage": target_trusted_coverage,
                "coverage_status": "UNKNOWN",
                "live_order_effect": "read_only_no_live_order",
                "next_action": _cost_bootstrap_next_action(
                    state,
                    fee_available=fee_available,
                    slippage_available=slippage_available,
                    spread_available=spread_available,
                    bill_matched_count=bill_matched_count,
                ),
            }
        )

    denominator = max(len(symbols), 1)
    bootstrap_rate = bootstrap_covered / denominator
    trusted_rate = trusted_covered / denominator
    status = "PASS" if trusted_rate >= target_trusted_coverage else "WARNING"
    for row in output:
        row["actual_or_mixed_bootstrap_coverage_live_universe"] = bootstrap_rate
        row["actual_or_mixed_trusted_coverage_live_universe"] = trusted_rate
        row["coverage_status"] = status
    if not output:
        return pl.DataFrame(schema={field: pl.Utf8 for field in COST_BOOTSTRAP_READINESS_FIELDS})
    return pl.DataFrame(output, infer_schema_length=None).select(
        COST_BOOTSTRAP_READINESS_FIELDS
    )


class CostBucket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bucket_id: str | None = None
    symbol: str | None = None
    regime: str | None = None
    min_notional_usdt: float = Field(default=0, ge=0)
    max_notional_usdt: float | None = Field(default=None, gt=0)
    cost_bps: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_range(self) -> "CostBucket":
        if self.max_notional_usdt is not None and self.max_notional_usdt < self.min_notional_usdt:
            raise ValueError("max_notional_usdt must be greater than or equal to min_notional_usdt")
        return self

    @model_validator(mode="before")
    @classmethod
    def normalize_bucket_symbol(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("symbol") is not None:
            normalized = dict(data)
            normalized["symbol"] = normalize_optional_symbol(normalized.get("symbol"))
            return normalized
        return data

    def includes_notional(self, notional_usdt: float) -> bool:
        if notional_usdt < self.min_notional_usdt:
            return False
        return self.max_notional_usdt is None or notional_usdt <= self.max_notional_usdt


class CostBucketDaily(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    day: str = Field(min_length=10)
    symbol: str = Field(min_length=1)
    regime: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    notional_bucket: str = Field(min_length=1)
    sample_count: int = Field(ge=0)
    fee_bps_p50: float = Field(ge=0)
    fee_bps_p75: float = Field(ge=0)
    fee_bps_p90: float = Field(ge=0)
    slippage_bps_p50: float = Field(ge=0)
    slippage_bps_p75: float = Field(ge=0)
    slippage_bps_p90: float = Field(ge=0)
    spread_bps_p50: float = Field(ge=0)
    spread_bps_p75: float = Field(ge=0)
    spread_bps_p90: float = Field(ge=0)
    spread_source: str = Field(default="unavailable", min_length=1)
    total_cost_bps_p50: float = Field(ge=0)
    total_cost_bps_p75: float = Field(ge=0)
    total_cost_bps_p90: float = Field(ge=0)
    roundtrip_cost_p50_bps: float = Field(default=0.0, ge=0)
    roundtrip_cost_p75_bps: float = Field(default=0.0, ge=0)
    roundtrip_cost_p90_bps: float = Field(default=0.0, ge=0)
    fallback_level: str = Field(min_length=1)
    source: str = Field(min_length=1)
    cost_source: str | None = None
    actual_fill_count: int = Field(default=0, ge=0)
    mixed_fill_count: int = Field(default=0, ge=0)
    proxy_sample_count: int = Field(default=0, ge=0)
    cost_probe_fill_count: int = Field(default=0, ge=0)
    strategy_live_fill_count: int = Field(default=0, ge=0)
    private_fill_count: int = Field(default=0, ge=0)
    sample_origin_mix: str = Field(default="unknown", min_length=1)
    eligible_for_live_cost_coverage: bool = True
    cost_model_version: str = Field(default="cost_bucket_daily.v0.1", min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="before")
    @classmethod
    def normalize_daily_symbol(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("symbol") not in {None, "GLOBAL"}:
            normalized = dict(data)
            normalized["symbol"] = normalize_symbol(normalized.get("symbol"))
            normalized.setdefault("cost_source", normalized.get("source"))
            return normalized
        if isinstance(data, dict):
            normalized = dict(data)
            normalized.setdefault("cost_source", normalized.get("source"))
            return normalized
        return data


def _normalize_buckets(buckets: Iterable[CostBucket | Mapping[str, Any]]) -> list[CostBucket]:
    return [
        bucket if isinstance(bucket, CostBucket) else CostBucket.model_validate(bucket)
        for bucket in buckets
    ]


def _choose_bucket(
    symbol: str, regime: str, notional_usdt: float, buckets: list[CostBucket]
) -> tuple[CostBucket | None, str]:
    notional_matches = [bucket for bucket in buckets if bucket.includes_notional(notional_usdt)]

    tiers = [
        (
            "NONE",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol == symbol and bucket.regime == regime
            ],
        ),
        (
            "REGIME_FALLBACK",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol == symbol and bucket.regime is None
            ],
        ),
        (
            "SYMBOL_FALLBACK",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol is None and bucket.regime == regime
            ],
        ),
        (
            "GLOBAL_BUCKET_FALLBACK",
            [
                bucket
                for bucket in notional_matches
                if bucket.symbol is None and bucket.regime is None
            ],
        ),
    ]
    for fallback_level, candidates in tiers:
        if candidates:
            return candidates[0], fallback_level
    return None, "DEFAULT_FALLBACK"


def estimate_cost_bps(
    symbol: str,
    regime: str,
    notional_usdt: float,
    buckets: Iterable[CostBucket | Mapping[str, Any]],
) -> CostEstimate:
    if notional_usdt <= 0:
        raise ValueError("notional_usdt must be positive")

    requested_symbol = normalize_symbol(symbol)
    normalized = _normalize_buckets(buckets)
    bucket, fallback_level = _choose_bucket(requested_symbol, regime, notional_usdt, normalized)

    if bucket is None:
        logger.warning(
            "No cost bucket matched; using explicit default fallback",
            extra={
                "symbol": requested_symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "fallback_level": fallback_level,
            },
        )
        return CostEstimate(
            symbol=requested_symbol,
            regime=regime,
            notional_usdt=notional_usdt,
            quantile="p75",
            fee_bps=0.0,
            slippage_bps=0.0,
            spread_bps=0.0,
            total_cost_bps=DEFAULT_FALLBACK_COST_BPS,
            cost_bps=DEFAULT_FALLBACK_COST_BPS,
            fallback_level=fallback_level,
            source="global_default",
            sample_count=0,
            cost_model_version="legacy_cost_bucket_v0",
            bucket_id=None,
        )

    if fallback_level != "NONE":
        logger.warning(
            "Cost bucket fallback used",
            extra={
                "symbol": requested_symbol,
                "regime": regime,
                "notional_usdt": notional_usdt,
                "bucket_id": bucket.bucket_id,
                "fallback_level": fallback_level,
            },
        )

    return CostEstimate(
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile="p75",
        fee_bps=0.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        total_cost_bps=bucket.cost_bps,
        cost_bps=bucket.cost_bps,
        fallback_level=fallback_level,
        source="configured_cost_bucket",
        sample_count=0,
        cost_model_version="legacy_cost_bucket_v0",
        bucket_id=bucket.bucket_id,
    )


def estimate_cost_from_lake(
    lake_root: str | Path,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str = "p75",
    notional_bucket: str | None = None,
) -> CostEstimate:
    requested_symbol = normalize_symbol(symbol)
    try:
        rows, dataset_has_rows = _cost_bucket_rows_for_symbol(Path(lake_root), requested_symbol)
    except Exception:
        logger.warning("Cost bucket daily read failed; using global default", exc_info=True)
        return _global_default_estimate(
            requested_symbol,
            regime,
            notional_usdt,
            quantile,
            fallback_reason="service_unavailable",
            degraded_reason="global_default_cost",
        )
    if not rows:
        return _global_default_estimate(
            requested_symbol,
            regime,
            notional_usdt,
            quantile,
            fallback_reason="symbol_missing" if dataset_has_rows else "service_unavailable",
            degraded_reason="global_default_cost",
        )
    return estimate_cost_from_cost_bucket_daily_rows(
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        rows=rows,
        notional_bucket=notional_bucket,
    )


def estimate_cost_from_cost_bucket_table_rows(
    *,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str = "p75",
    rows: Iterable[Mapping[str, Any]],
    dataset_has_rows: bool,
    notional_bucket: str | None = None,
) -> CostEstimate:
    requested_symbol = normalize_symbol(symbol)
    row_list = list(rows)
    if not row_list:
        return _global_default_estimate(
            requested_symbol,
            regime,
            notional_usdt,
            quantile,
            fallback_reason="service_unavailable",
            degraded_reason="global_default_cost",
        )
    filtered = [
        row
        for row in row_list
        if _cost_row_matches_symbol(row, requested_symbol)
    ]
    if not filtered:
        return _global_default_estimate(
            requested_symbol,
            regime,
            notional_usdt,
            quantile,
            fallback_reason="symbol_missing" if dataset_has_rows else "service_unavailable",
            degraded_reason="global_default_cost",
        )
    return estimate_cost_from_cost_bucket_daily_rows(
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        rows=filtered,
        notional_bucket=notional_bucket,
    )


def _cost_bucket_rows_for_symbol(
    lake_root: Path, normalized_symbol: str
) -> tuple[list[dict[str, Any]], bool]:
    dataset_path = lake_root / "gold" / "cost_bucket_daily"
    try:
        lazy = read_parquet_lazy(dataset_path)
        columns = set(lazy.collect_schema().names())
    except Exception:
        df = read_parquet_dataset(dataset_path)
        if df.is_empty():
            return [], False
        if "symbol" not in df.columns and "normalized_symbol" not in df.columns:
            return [], True
        filtered = df.filter(_eager_cost_symbol_filter(df, normalized_symbol))
        return filtered.to_dicts(), True

    if "symbol" not in columns and "normalized_symbol" not in columns:
        dataset_has_rows = _lazy_row_count(lazy) > 0
        return [], dataset_has_rows

    filtered = lazy.filter(_lazy_cost_symbol_filter(columns, normalized_symbol))
    rows = filtered.collect().to_dicts()
    if rows:
        return rows, True
    return [], _lazy_row_count(lazy) > 0


def _lazy_cost_symbol_filter(columns: set[str], normalized_symbol: str) -> pl.Expr:
    lookup_values = _cost_symbol_lookup_values(normalized_symbol)
    global_values = {"", "GLOBAL", "ALL", "*"}
    expressions: list[pl.Expr] = []
    for column in ("symbol", "normalized_symbol"):
        if column not in columns:
            continue
        normalized_column = (
            pl.col(column).cast(pl.Utf8, strict=False).str.to_uppercase().fill_null("")
        )
        expressions.append(normalized_column.is_in(sorted(lookup_values | global_values)))
    return _or_expressions(expressions)


def _eager_cost_symbol_filter(df: pl.DataFrame, normalized_symbol: str) -> pl.Expr:
    return _lazy_cost_symbol_filter(set(df.columns), normalized_symbol)


def _cost_symbol_lookup_values(normalized_symbol: str) -> set[str]:
    symbol = normalize_symbol(normalized_symbol)
    values = {symbol}
    if "-" in symbol:
        values.add(symbol.replace("-", "/"))
        values.add(symbol.replace("-", "_"))
        values.add(symbol.replace("-", ""))
    values.update({f"OKX:{value}" for value in list(values)})
    return {value.upper() for value in values if value}


def _cost_row_matches_symbol(row: Mapping[str, Any], normalized_symbol: str) -> bool:
    lookup_values = _cost_symbol_lookup_values(normalized_symbol)
    global_values = {"", "GLOBAL", "ALL", "*"}
    for column in ("symbol", "normalized_symbol"):
        value = str(row.get(column) or "").strip().upper()
        if value in lookup_values or value in global_values:
            return True
    return False


def _or_expressions(expressions: list[pl.Expr]) -> pl.Expr:
    if not expressions:
        return pl.lit(False)
    combined = expressions[0]
    for expression in expressions[1:]:
        combined = combined | expression
    return combined


def _lazy_row_count(lazy: pl.LazyFrame) -> int:
    try:
        return int(lazy.select(pl.len().alias("rows")).collect().item(0, "rows") or 0)
    except Exception:
        return 0


def estimate_cost_from_cost_bucket_daily_rows(
    *,
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str,
    rows: Iterable[Mapping[str, Any]],
    notional_bucket: str | None = None,
) -> CostEstimate:
    if notional_usdt <= 0:
        raise ValueError("notional_usdt must be positive")
    if quantile not in SUPPORTED_COST_QUANTILES:
        raise ValueError("quantile must be one of p50, p75, p90")

    requested_symbol = normalize_symbol(symbol)
    normalized_rows = [_normalize_cost_row(row) for row in rows]
    if not normalized_rows:
        return _global_default_estimate(
            requested_symbol,
            regime,
            notional_usdt,
            quantile,
            fallback_reason="service_unavailable",
            degraded_reason="global_default_cost",
        )

    tiered = _rank_cost_bucket_rows(
        rows=normalized_rows,
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        notional_bucket=notional_bucket,
    )
    if not tiered:
        fallback_reason = (
            "no_matching_regime"
            if any(_row_symbol(row) == requested_symbol for row in normalized_rows)
            else "symbol_missing"
        )
        return _global_default_estimate(
            requested_symbol,
            regime,
            notional_usdt,
            quantile,
            fallback_reason=fallback_reason,
            degraded_reason="global_default_cost",
        )

    row, fallback_level = tiered[0]
    row_fallback_level = str(row.get("fallback_level") or "")
    if row_fallback_level.upper() == "GLOBAL_DEFAULT":
        fallback_level = "GLOBAL_DEFAULT"
    elif row_fallback_level and row_fallback_level not in {
        "NONE",
        "actual_okx_fills_and_bills",
    }:
        fallback_level = (
            row_fallback_level
            if fallback_level == "NONE"
            else f"{fallback_level};{row_fallback_level}"
        )
    observed_fee_bps = _float_value(row, f"fee_bps_{quantile}")
    observed_slippage_bps = _float_value(row, f"slippage_bps_{quantile}")
    observed_spread_bps = _float_value(row, f"spread_bps_{quantile}")
    total_cost_bps = _float_value(row, f"total_cost_bps_{quantile}")
    if total_cost_bps == 0:
        total_cost_bps = observed_fee_bps + observed_slippage_bps + observed_spread_bps

    bucket_id = _cost_bucket_id(row)
    source = str(row.get("source") or "cost_bucket_daily")
    stale = _row_is_stale(row) and not _is_bootstrap_cost_probe_source(source)
    sample_count = int(row.get("sample_count") or 0)
    fallback_reason = _fallback_reason(fallback_level, stale=stale, source=source)
    components = _all_in_cost_components(
        source=source,
        fallback_level=fallback_level,
        spread_source=str(row.get("spread_source") or ""),
        observed_fee_bps=observed_fee_bps,
        observed_slippage_bps=observed_slippage_bps,
        observed_spread_bps=observed_spread_bps,
        sample_count=sample_count,
        live_cost_sample_count=_live_cost_sample_count(row),
        stale=stale,
    )
    return CostEstimate(
        symbol=requested_symbol,
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        requested_quantile=quantile,
        fee_bps=components["fee_bps"],
        slippage_bps=components["slippage_bps"],
        spread_bps=components["spread_bps"],
        total_cost_bps=total_cost_bps,
        cost_bps=total_cost_bps,
        fallback_level=fallback_level,
        source=source,
        sample_count=sample_count,
        live_cost_sample_count=components["live_cost_sample_count"],
        trusted_live_sample_count=components["trusted_live_sample_count"],
        cost_model_version=str(
            row.get("cost_model_version") or f"cost_bucket_daily:{row.get('day', 'unknown')}"
        ),
        bucket_id=bucket_id,
        requested_regime=regime,
        matched_regime=str(row.get("regime") or "unknown"),
        cost_source=source,
        total_cost_bps_p50=_float_value(row, "total_cost_bps_p50"),
        total_cost_bps_p75=_float_value(row, "total_cost_bps_p75"),
        total_cost_bps_p90=_float_value(row, "total_cost_bps_p90"),
        selected_total_cost_bps=total_cost_bps,
        fallback_reason=fallback_reason,
        degraded_reason="cost_bucket_stale" if stale else "none",
        degraded_cost_model=_estimate_degraded(
            source=source,
            fallback_level=fallback_level,
            fallback_reason=fallback_reason,
            degraded_reason="cost_bucket_stale" if stale else "none",
        ),
        as_of_ts=_row_as_of_ts(row),
        fee_source=components["fee_source"],
        spread_source=components["spread_source"],
        slippage_source=components["slippage_source"],
        delay_cost_bps=components["delay_cost_bps"],
        delay_cost_source=components["delay_cost_source"],
        uncertainty_buffer_bps=components["uncertainty_buffer_bps"],
        one_way_all_in_cost_bps=components["one_way_all_in_cost_bps"],
        roundtrip_all_in_cost_bps=components["roundtrip_all_in_cost_bps"],
        cost_quality=components["cost_quality"],
        cost_trusted_for_paper=components["cost_trusted_for_paper"],
        cost_trusted_for_live=components["cost_trusted_for_live"],
    )


def build_cost_bucket_daily_inputs(
    fill_events: Iterable[FillEvent | Mapping[str, Any]],
    regime: str = "realized",
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}

    for raw_event in fill_events:
        event = (
            raw_event if isinstance(raw_event, FillEvent) else FillEvent.model_validate(raw_event)
        )
        notional = abs(event.fill_price * event.fill_size)
        if notional <= 0:
            continue
        symbol = normalize_symbol(event.inst_id)
        key = (symbol, event.ts.date().isoformat())
        bucket = grouped.setdefault(
            key,
            {
                "symbol": symbol,
                "cost_day": event.ts.date().isoformat(),
                "regime": regime,
                "notional_usdt": 0.0,
                "fee_abs": 0.0,
                "source": event.source,
            },
        )
        bucket["notional_usdt"] += notional
        bucket["fee_abs"] += abs(event.fee)

    rows: list[dict[str, Any]] = []
    for bucket in grouped.values():
        notional = bucket["notional_usdt"]
        fee_abs = bucket["fee_abs"]
        rows.append(
            {
                **bucket,
                "cost_bps": (fee_abs / notional) * 10_000 if notional else 0.0,
            }
        )
    return sorted(rows, key=lambda row: (row["symbol"], row["cost_day"], row["regime"]))


def cost_bucket_daily_to_cost_buckets(
    rows: Iterable[CostBucketDaily | Mapping[str, Any]],
    percentile: str = "p50",
) -> list[CostBucket]:
    cost_column = f"total_cost_bps_{percentile}"
    buckets: list[CostBucket] = []
    for raw_row in rows:
        row = (
            raw_row
            if isinstance(raw_row, CostBucketDaily)
            else CostBucketDaily.model_validate(raw_row)
        )
        if not hasattr(row, cost_column):
            raise ValueError(f"Unsupported cost percentile: {percentile}")
        min_notional, max_notional = _notional_bucket_bounds(row.notional_bucket)
        buckets.append(
            CostBucket(
                bucket_id=(
                    f"{row.day}:{row.symbol}:{row.regime}:{row.event_type}:{row.notional_bucket}"
                ),
                symbol=row.symbol if row.symbol != "GLOBAL" else None,
                regime=row.regime if row.regime != "global_default" else None,
                min_notional_usdt=min_notional,
                max_notional_usdt=max_notional,
                cost_bps=float(getattr(row, cost_column)),
            )
        )
    return buckets


def _notional_bucket_bounds(notional_bucket: str) -> tuple[float, float | None]:
    match notional_bucket:
        case "0-1k":
            return 0.0, 1_000.0
        case "1k-10k":
            return 1_000.0, 10_000.0
        case "10k-100k":
            return 10_000.0, 100_000.0
        case "100k+":
            return 100_000.0, None
        case "all":
            return 0.0, None
        case _:
            return 0.0, None


def _rank_cost_bucket_rows(
    *,
    rows: list[dict[str, Any]],
    symbol: str,
    regime: str,
    notional_usdt: float,
    notional_bucket: str | None,
) -> list[tuple[dict[str, Any], str]]:
    ranked: list[tuple[int, str, dict[str, Any]]] = []
    requested_regime = regime.lower()
    fresh_symbol_candidate_available = any(
        _row_symbol(candidate) == symbol and not _row_is_stale(candidate)
        for candidate in rows
    )
    for row in rows:
        row_symbol = _row_symbol(row)
        row_regime = str(row.get("regime") or "")
        normalized_row_regime = row_regime.lower()
        row_bucket = str(row.get("notional_bucket") or "")
        notional_match = _row_matches_notional(row_bucket, notional_usdt, notional_bucket)

        source = str(row.get("source") or "")

        if row_symbol == symbol and normalized_row_regime == requested_regime and notional_match:
            tier, fallback = 0, "NONE"
        elif row_symbol == symbol and normalized_row_regime == requested_regime:
            tier, fallback = 1, "NOTIONAL_BUCKET_FALLBACK"
        elif row_symbol == symbol and _is_actual_or_mixed_source(source) and notional_match:
            tier, fallback = 2, "REGIME_FALLBACK"
        elif row_symbol == symbol and _is_actual_or_mixed_source(source):
            tier, fallback = 3, "REGIME_FALLBACK"
        elif row_symbol == symbol and _is_bootstrap_cost_probe_source(source) and notional_match:
            tier, fallback = 4, "REGIME_FALLBACK"
        elif row_symbol == symbol and _is_bootstrap_cost_probe_source(source):
            tier, fallback = 5, "REGIME_FALLBACK"
        elif row_symbol == symbol and _is_public_proxy_source(source) and notional_match:
            tier, fallback = 6, "REGIME_FALLBACK"
        elif row_symbol == symbol and _is_public_proxy_source(source):
            tier, fallback = 7, "REGIME_FALLBACK"
        elif row_symbol == symbol and _is_global_regime(row_regime) and notional_match:
            tier, fallback = 8, "REGIME_FALLBACK"
        elif row_symbol == symbol and notional_match:
            tier, fallback = 9, "REGIME_FALLBACK"
        elif row_symbol == symbol:
            tier, fallback = 10, "REGIME_AND_NOTIONAL_BUCKET_FALLBACK"
        elif (
            _is_global_symbol(row_symbol)
            and normalized_row_regime == requested_regime
            and notional_match
        ):
            tier, fallback = 11, "SYMBOL_FALLBACK"
        elif _is_global_symbol(row_symbol) and _is_global_regime(row_regime):
            tier, fallback = 12, "GLOBAL_BUCKET_FALLBACK"
        else:
            continue
        if (
            fresh_symbol_candidate_available
            and _row_symbol(row) == symbol
            and _row_is_stale(row)
            and not _is_bootstrap_cost_probe_source(source)
        ):
            tier += 10
        ranked.append((tier, fallback, row))

    return [
        (row, fallback)
        for _tier, fallback, row in sorted(
            ranked,
            key=lambda item: (
                item[0],
                _source_priority(str(item[2].get("source") or "")),
                _day_sort_value(item[2]),
                -int(item[2].get("sample_count") or 0),
            ),
        )
    ]


def _row_matches_notional(
    row_bucket: str,
    notional_usdt: float,
    requested_bucket: str | None,
) -> bool:
    if requested_bucket:
        return row_bucket == requested_bucket
    min_notional, max_notional = _notional_bucket_bounds(row_bucket)
    if notional_usdt < min_notional:
        return False
    return max_notional is None or notional_usdt <= max_notional


def _is_global_symbol(symbol: str) -> bool:
    return symbol.upper() in {"", "GLOBAL", "ALL", "*"}


def _is_global_regime(regime: str) -> bool:
    return regime.lower() in {"", "global", "global_default", "all", "*"}


def _float_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value or 0.0)


def _cost_bucket_id(row: Mapping[str, Any]) -> str:
    return ":".join(
        str(row.get(part) or "unknown")
        for part in ["day", "symbol", "regime", "event_type", "notional_bucket"]
    )


def _day_sort_value(row: Mapping[str, Any]) -> int:
    digits = "".join(character for character in str(row.get("day") or "") if character.isdigit())
    return -int(digits or 0)


def _global_default_estimate(
    symbol: str,
    regime: str,
    notional_usdt: float,
    quantile: str,
    *,
    fallback_reason: str = "symbol_missing",
    degraded_reason: str = "global_default_cost",
) -> CostEstimate:
    components = _global_default_components()
    return CostEstimate(
        symbol=normalize_symbol(symbol),
        regime=regime,
        notional_usdt=notional_usdt,
        quantile=quantile,
        requested_quantile=quantile,
        fee_bps=components["fee_bps"],
        slippage_bps=components["slippage_bps"],
        spread_bps=components["spread_bps"],
        total_cost_bps=DEFAULT_FALLBACK_COST_BPS,
        cost_bps=DEFAULT_FALLBACK_COST_BPS,
        fallback_level="GLOBAL_DEFAULT",
        source="global_default",
        sample_count=0,
        cost_model_version="global_default_v0",
        bucket_id=None,
        requested_regime=regime,
        matched_regime="global_default",
        cost_source="global_default",
        total_cost_bps_p50=DEFAULT_FALLBACK_COST_BPS,
        total_cost_bps_p75=DEFAULT_FALLBACK_COST_BPS,
        total_cost_bps_p90=DEFAULT_FALLBACK_COST_BPS,
        selected_total_cost_bps=DEFAULT_FALLBACK_COST_BPS,
        fallback_reason=fallback_reason,
        degraded_reason=degraded_reason,
        degraded_cost_model=True,
        as_of_ts=datetime.now(UTC),
        fee_source=components["fee_source"],
        spread_source=components["spread_source"],
        slippage_source=components["slippage_source"],
        delay_cost_bps=components["delay_cost_bps"],
        delay_cost_source=components["delay_cost_source"],
        uncertainty_buffer_bps=components["uncertainty_buffer_bps"],
        one_way_all_in_cost_bps=components["one_way_all_in_cost_bps"],
        roundtrip_all_in_cost_bps=components["roundtrip_all_in_cost_bps"],
        cost_quality="global_default",
        cost_trusted_for_paper=False,
        cost_trusted_for_live=False,
    )


def _normalize_cost_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    if not _is_global_symbol(str(normalized.get("symbol") or "")):
        normalized["symbol"] = normalize_symbol(normalized.get("symbol"))
    return normalized


def _row_symbol(row: Mapping[str, Any]) -> str:
    raw = str(row.get("symbol") or "")
    return raw if _is_global_symbol(raw) else normalize_symbol(raw)


def _source_priority(source: str) -> int:
    normalized = source.lower()
    if normalized in {"actual_okx_fills_and_bills", "actual_fills", "mixed_actual_proxy"}:
        return 0
    if normalized == "actual_okx_fills_fee_missing":
        return 1
    if normalized == "bootstrap_cost_probe":
        return 2
    if normalized in {"public_spread_proxy", "public_proxy"}:
        return 3
    if normalized == "global_default":
        return 4
    return 5


def _is_actual_or_mixed_source(source: str) -> bool:
    return source.lower() in {
        "actual_okx_fills_and_bills",
        "actual_fills",
        "mixed_actual_proxy",
        "actual_okx_fills_fee_missing",
    }


def _is_public_proxy_source(source: str) -> bool:
    return source.lower() in {"public_spread_proxy", "public_proxy"}


def _is_bootstrap_cost_probe_source(source: str) -> bool:
    return source.lower() == "bootstrap_cost_probe"


def _estimate_degraded(
    *,
    source: str,
    fallback_level: str,
    fallback_reason: str,
    degraded_reason: str,
) -> bool:
    return (
        source in {"global_default", "public_spread_proxy"}
        or fallback_level not in {"", "NONE", "actual_okx_fills_and_bills"}
        or fallback_reason not in {"", "NONE"}
        or degraded_reason not in {"", "none"}
    )


def _all_in_cost_components(
    *,
    source: str,
    fallback_level: str,
    spread_source: str,
    observed_fee_bps: float,
    observed_slippage_bps: float,
    observed_spread_bps: float,
    sample_count: int,
    live_cost_sample_count: int,
    stale: bool,
) -> dict[str, Any]:
    normalized_source = source.lower()
    normalized_fallback = fallback_level.upper()
    actual_or_mixed = _is_actual_or_mixed_source(normalized_source)
    trusted_live_sample_count = live_cost_sample_count if actual_or_mixed else 0
    quality_sample_count = live_cost_sample_count if actual_or_mixed else sample_count
    fee_is_actual = actual_or_mixed and observed_fee_bps > 0
    slippage_is_actual = (
        actual_or_mixed
        and observed_slippage_bps > 0
        and "SLIPPAGE_UNKNOWN" not in normalized_fallback
    )

    fee_bps = observed_fee_bps if fee_is_actual else CONFIG_FEE_BPS
    slippage_bps = observed_slippage_bps if slippage_is_actual else CONFIG_SLIPPAGE_BPS
    spread_bps = observed_spread_bps

    uncertainty_buffer = 0.0
    if _is_public_proxy_source(normalized_source):
        uncertainty_buffer += PUBLIC_PROXY_UNCERTAINTY_BUFFER_BPS
    if quality_sample_count < MIN_TRUSTED_COST_SAMPLE_COUNT:
        uncertainty_buffer += SMALL_SAMPLE_UNCERTAINTY_BUFFER_BPS
    if stale:
        uncertainty_buffer += STALE_BUCKET_UNCERTAINTY_BUFFER_BPS

    one_way = fee_bps + spread_bps + slippage_bps + CONFIG_DELAY_COST_BPS + uncertainty_buffer
    return {
        "fee_bps": fee_bps,
        "fee_source": "actual_fills_bills" if fee_is_actual else "config_fee_bps",
        "spread_bps": spread_bps,
        "spread_source": _normalized_spread_source(
            spread_source,
            spread_bps=spread_bps,
            stale=stale,
            source=normalized_source,
        ),
        "slippage_bps": slippage_bps,
        "slippage_source": (
            "v5_order_lifecycle_arrival_mid" if slippage_is_actual else "config_slippage_bps"
        ),
        "delay_cost_bps": CONFIG_DELAY_COST_BPS,
        "delay_cost_source": "config_delay_bps",
        "uncertainty_buffer_bps": uncertainty_buffer,
        "one_way_all_in_cost_bps": one_way,
        "roundtrip_all_in_cost_bps": one_way * 2.0,
        "live_cost_sample_count": live_cost_sample_count,
        "trusted_live_sample_count": trusted_live_sample_count,
        "cost_quality": _cost_quality(
            source=normalized_source,
            sample_count=quality_sample_count,
            stale=stale,
        ),
        "cost_trusted_for_paper": normalized_source != "global_default" and not stale,
        "cost_trusted_for_live": (
            normalized_source
            in {"actual_fills", "actual_okx_fills_and_bills", "mixed_actual_proxy"}
            and live_cost_sample_count >= MIN_TRUSTED_COST_SAMPLE_COUNT
            and not stale
        ),
    }


def _normalized_spread_source(
    value: str,
    *,
    spread_bps: float,
    stale: bool,
    source: str,
) -> str:
    if stale or spread_bps <= 0:
        return "unavailable"
    normalized = str(value or "").strip().lower()
    if normalized in {
        "actual_arrival_book",
        "actual_order_lifecycle_arrival_book",
        "v5_order_lifecycle_arrival_book",
        "spread_at_decision",
    }:
        return "actual_arrival_book"
    if normalized in {
        "fresh_public_orderbook_p75",
        "fresh_orderbook_p75",
        "public_orderbook_p75",
        "public_spread_proxy",
        "spread_proxy",
    }:
        return "fresh_public_orderbook_p75"
    if normalized == "unavailable":
        return "unavailable"
    if _is_public_proxy_source(source):
        return "fresh_public_orderbook_p75"
    return "fresh_public_orderbook_p75"


def _global_default_components() -> dict[str, Any]:
    fee_bps = CONFIG_FEE_BPS
    spread_bps = 5.0
    slippage_bps = 5.0
    uncertainty_buffer = DEFAULT_FALLBACK_COST_BPS - (
        fee_bps + spread_bps + slippage_bps + CONFIG_DELAY_COST_BPS
    )
    one_way = DEFAULT_FALLBACK_COST_BPS
    return {
        "fee_bps": fee_bps,
        "fee_source": "config_fee_bps",
        "spread_bps": spread_bps,
        "spread_source": "global_default_config",
        "slippage_bps": slippage_bps,
        "slippage_source": "config_slippage_bps",
        "delay_cost_bps": CONFIG_DELAY_COST_BPS,
        "delay_cost_source": "config_delay_bps",
        "uncertainty_buffer_bps": max(uncertainty_buffer, 0.0),
        "one_way_all_in_cost_bps": one_way,
        "roundtrip_all_in_cost_bps": one_way * 2.0,
    }


def _cost_quality(*, source: str, sample_count: int, stale: bool) -> str:
    if stale:
        return "stale"
    if source == "global_default":
        return "global_default"
    if source == "bootstrap_cost_probe":
        return "bootstrap_cost_probe"
    if source in {"public_spread_proxy", "public_proxy"}:
        return "public_proxy_only"
    if sample_count < MIN_TRUSTED_COST_SAMPLE_COUNT:
        return "small_sample"
    if source in {"mixed_actual_proxy", "actual_okx_fills_fee_missing"}:
        return "mixed_actual_proxy"
    if source in {"actual_fills", "actual_okx_fills_and_bills"}:
        return "actual"
    return "unknown"


def _fallback_reason(fallback_level: str, *, stale: bool, source: str) -> str:
    if stale:
        return "cost_bucket_stale"
    if fallback_level == "NONE":
        return "NONE"
    normalized = fallback_level.upper()
    if "REGIME_FALLBACK" in normalized:
        return "no_matching_regime"
    if "SYMBOL_FALLBACK" in normalized or "GLOBAL_BUCKET_FALLBACK" in normalized:
        return "symbol_missing"
    if source == "global_default":
        return "symbol_missing"
    return fallback_level


def _row_as_of_ts(row: Mapping[str, Any]) -> datetime | None:
    for key in ("created_at", "as_of_ts"):
        value = row.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if value:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed.astimezone(UTC)
    day = row.get("day")
    if day:
        try:
            return datetime.fromisoformat(str(day)).replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def _row_is_stale(
    row: Mapping[str, Any],
    *,
    reference_time: datetime | None = None,
) -> bool:
    as_of_ts = _row_explicit_as_of_ts(row)
    if as_of_ts is None:
        return False
    reference = (reference_time or datetime.now(UTC)).astimezone(UTC)
    age_seconds = (reference - as_of_ts.astimezone(UTC)).total_seconds()
    return age_seconds > COST_BUCKET_STALE_SECONDS


def _event_row_is_fresh(
    row: Mapping[str, Any],
    *,
    reference_time: datetime,
) -> bool:
    timestamp = _row_ts(
        row,
        "last_fill_ts",
        "fill_ts",
        "ts_utc",
        "ts",
        "timestamp",
        "trade_ts",
        "submit_ts",
        "decision_ts",
        "ingest_ts",
    )
    if timestamp is None:
        return True
    age_seconds = (reference_time - timestamp.astimezone(UTC)).total_seconds()
    return age_seconds <= COST_BUCKET_STALE_SECONDS


def _row_explicit_as_of_ts(row: Mapping[str, Any]) -> datetime | None:
    for key in ("created_at", "as_of_ts"):
        value = row.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=UTC)
        if value:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except ValueError:
                continue
            return parsed.astimezone(UTC)
    return None


def _normalized_cost_rows(frame: pl.DataFrame | None) -> list[dict[str, Any]]:
    if frame is None or frame.is_empty():
        return []
    return [_normalize_cost_row(row) for row in frame.to_dicts()]


def _latest_cost_row_for_coverage(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=_coverage_sort_key)[-1]


def _latest_actual_or_mixed_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if _is_actual_or_mixed_source(_cost_source(row))
        and _row_live_cost_eligible(row)
    ]
    return _latest_cost_row_for_coverage(candidates)


def _latest_bootstrap_cost_probe_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if _cost_source(row) == "bootstrap_cost_probe"]
    return _latest_cost_row_for_coverage(candidates)


def _latest_fresh_actual_or_mixed_row(
    rows: list[dict[str, Any]],
    *,
    reference_time: datetime | None = None,
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if _is_actual_or_mixed_source(_cost_source(row))
        and _row_live_cost_eligible(row)
        and not _row_is_stale(row, reference_time=reference_time)
    ]
    return _latest_cost_row_for_coverage(candidates)


def _latest_public_proxy_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in rows if _is_public_proxy_source(_cost_source(row))]
    return _latest_cost_row_for_coverage(candidates)


def _coverage_sort_key(row: Mapping[str, Any]) -> tuple[datetime, int]:
    ts = _row_as_of_ts(row) or datetime.min.replace(tzinfo=UTC)
    return ts, _int_value(row.get("sample_count"))


def _cost_source(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return ""
    return str(row.get("source") or row.get("cost_source") or "").strip().lower()


def _sample_origin_mix_value(row: Mapping[str, Any] | None) -> str:
    if row is None:
        return ""
    return str(row.get("sample_origin_mix") or row.get("cost_sample_origin_mix") or "").strip()


def _live_cost_sample_count(
    row: Mapping[str, Any] | None,
    *,
    strategy_live_fill_count: int | None = None,
    private_fill_count: int | None = None,
) -> int:
    if row is None:
        return max((strategy_live_fill_count or 0) + (private_fill_count or 0), 0)

    source = _cost_source(row)
    strategy_count = (
        max(strategy_live_fill_count, 0)
        if strategy_live_fill_count is not None
        else _int_value(row.get("strategy_live_fill_count"))
    )
    private_count = (
        max(private_fill_count, 0)
        if private_fill_count is not None
        else _int_value(row.get("private_fill_count"))
    )
    live_origin_count = strategy_count + private_count
    actual_or_mixed_count = _int_value(row.get("actual_fill_count")) + _int_value(
        row.get("mixed_fill_count")
    )
    live_count = max(live_origin_count, actual_or_mixed_count)
    if live_count > 0:
        return live_count

    if not _is_actual_or_mixed_source(source):
        return 0
    if _bool_or_none(row.get("eligible_for_live_cost_coverage")) is False:
        return 0
    if _sample_origin_mix_value(row).lower() == "cost_probe_only":
        return 0
    if _int_value(row.get("cost_probe_fill_count")) > 0:
        return 0
    return _int_value(row.get("sample_count"))


def _coverage_proxy_sample_count(row: Mapping[str, Any] | None) -> int:
    if row is None:
        return 0
    source = _cost_source(row)
    origin_mix = _sample_origin_mix_value(row).lower()
    if source == "bootstrap_cost_probe" or origin_mix == "cost_probe_only":
        return 0
    if _is_actual_or_mixed_source(source) and "proxy" not in origin_mix:
        return 0
    return _int_value(row.get("proxy_sample_count"))


def _row_live_cost_eligible(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    source = _cost_source(row)
    if not _is_actual_or_mixed_source(source):
        return False
    explicit = _bool_or_none(row.get("eligible_for_live_cost_coverage"))
    if explicit is False:
        return False
    origin_mix = _sample_origin_mix_value(row).lower()
    if origin_mix == "cost_probe_only":
        return False
    cost_probe_count = _int_value(row.get("cost_probe_fill_count"))
    live_count = max(
        _int_value(row.get("strategy_live_fill_count")),
        _int_value(row.get("private_fill_count")),
        _int_value(row.get("actual_fill_count")),
        _int_value(row.get("mixed_fill_count")),
    )
    return not (cost_probe_count > 0 and live_count <= 0)


def _coverage_ts(row: Mapping[str, Any]) -> str:
    ts = _row_as_of_ts(row)
    return ts.isoformat().replace("+00:00", "Z") if ts is not None else ""


def _coverage_age_sec(
    row: Mapping[str, Any],
    *,
    reference_time: datetime,
) -> float | None:
    ts = _row_as_of_ts(row)
    if ts is None:
        return None
    return max((reference_time - ts.astimezone(UTC)).total_seconds(), 0.0)


def _component_fresh(
    row: Mapping[str, Any] | None,
    column: str,
    *,
    reference_time: datetime,
) -> bool:
    if row is None or not row:
        return False
    if _row_is_stale(row, reference_time=reference_time):
        return False
    return _float_value(row, column) is not None


def _cost_evidence_tier(
    *,
    direct: bool,
    has_live_actual_anchor: bool,
    latest: Mapping[str, Any] | None,
    latest_actual: Mapping[str, Any] | None,
    latest_proxy: Mapping[str, Any] | None,
    stale_actual_or_mixed: bool,
) -> str:
    if direct:
        return "strict_direct_actual_or_mixed"
    if latest is not None and _cost_source(latest) == "bootstrap_cost_probe":
        return "bootstrap_cost_probe_not_counted"
    if latest_proxy is not None and has_live_actual_anchor:
        return "anchored_proxy_candidate_not_counted"
    if latest_proxy is not None:
        return "proxy_only_not_counted"
    if latest_actual is not None and stale_actual_or_mixed:
        return "stale_direct_not_counted"
    if latest_actual is not None:
        return "actual_or_mixed_not_fresh"
    if latest is not None and _cost_source(latest) == "bootstrap_cost_probe":
        return "bootstrap_cost_probe_not_counted"
    return "missing"


def _coverage_reason(
    *,
    direct: bool,
    anchored_mixed_proxy_candidate: bool,
    has_live_actual_anchor: bool,
    latest_actual: Mapping[str, Any] | None,
    stale_actual_or_mixed: bool,
    latest_proxy: Mapping[str, Any] | None,
    latest: Mapping[str, Any] | None,
) -> str:
    if direct:
        return "direct_actual_or_mixed_cost"
    if latest is not None and _cost_source(latest) == "bootstrap_cost_probe":
        return "bootstrap_cost_probe_not_live_coverage"
    if anchored_mixed_proxy_candidate:
        if latest_actual is not None and stale_actual_or_mixed:
            return "stale_actual_or_mixed_with_anchored_proxy_not_counted"
        return "anchored_proxy_candidate_not_counted"
    if latest_actual is not None and stale_actual_or_mixed:
        if latest_proxy is not None:
            return "stale_actual_or_mixed_no_fresh_live_anchor"
        return "stale_actual_or_mixed_no_fresh_proxy"
    if latest is not None and _cost_source(latest) == "bootstrap_cost_probe":
        return "bootstrap_cost_probe_not_live_coverage"
    if latest_proxy is not None:
        if not has_live_actual_anchor:
            return "public_proxy_only_no_live_actual_anchor"
        return "public_proxy_only_no_symbol_actual_anchor"
    if latest is not None:
        return "cost_row_not_actual_or_mixed"
    return "missing_cost_row"


def _int_value(value: Any) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _rows_by_symbol(frame: pl.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if frame is None or frame.is_empty():
        return grouped
    for row in frame.to_dicts():
        symbol = _symbol_from_any_row(row)
        if not symbol:
            continue
        grouped.setdefault(symbol, []).append(dict(row))
    return grouped


def _symbol_from_any_row(row: Mapping[str, Any]) -> str:
    for key in ("normalized_symbol", "symbol", "inst_id", "instId", "instrument_id"):
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return normalize_symbol(value)
        except Exception:
            continue
    return ""


def _lifecycle_row_filled(row: Mapping[str, Any]) -> bool:
    if _int_value(row.get("fill_count")) > 0:
        return True
    for key in ("filled_qty", "acc_fill_sz", "fill_sz", "qty"):
        if _positive_value(row.get(key)):
            return True
    state = str(row.get("order_state") or row.get("state") or "").strip().lower()
    return state in {"filled", "fully_filled", "partially_filled", "partial_fill"}


def _lifecycle_row_is_cost_probe(row: Mapping[str, Any]) -> bool:
    values = [
        row.get("execution_purpose"),
        row.get("cost_sample_origin"),
        row.get("strategy_candidate"),
        row.get("live_order_effect"),
    ]
    return any("cost_probe" in str(value or "").strip().lower() for value in values)


def _positive_cost_value(row: Mapping[str, Any], key: str) -> bool:
    return _positive_value(row.get(key))


def _positive_value(value: Any) -> bool:
    try:
        return abs(float(value)) > 0
    except (TypeError, ValueError):
        return False


def _any_positive(rows: Iterable[Mapping[str, Any]], *keys: str) -> bool:
    return any(_positive_value(row.get(key)) for row in rows for key in keys)


def _any_row_has_all(rows: Iterable[Mapping[str, Any]], *keys: str) -> bool:
    for row in rows:
        if all(row.get(key) not in {None, "", "null", "None"} for key in keys):
            return True
    return False


def _latest_row_ts(rows: Iterable[Mapping[str, Any]], *keys: str) -> str:
    latest: datetime | None = None
    for row in rows:
        parsed = _row_ts(row, *keys)
        if parsed is None:
            continue
        if latest is None or parsed > latest:
            latest = parsed
    return latest.isoformat().replace("+00:00", "Z") if latest is not None else ""


def _latest_probe_roundtrip_row(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    latest: Mapping[str, Any] | None = None
    latest_ts: datetime | None = None
    for row in rows:
        status = str(row.get("roundtrip_status") or row.get("status") or "").strip().lower()
        if status not in {"closed", "closed_flat", "completed"}:
            continue
        parsed = _row_ts(row, "event_ts", "closed_at", "generated_at", "ingest_ts")
        if latest is None or (
            parsed is not None and (latest_ts is None or parsed > latest_ts)
        ):
            latest = row
            latest_ts = parsed
    return latest


def _latest_probe_fill_bill_match_row(
    rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    latest: Mapping[str, Any] | None = None
    latest_ts: datetime | None = None
    for row in rows:
        parsed = _row_ts(row, "generated_at", "event_ts", "ts_utc", "ts", "ingest_ts")
        if latest is None or (
            parsed is not None and (latest_ts is None or parsed > latest_ts)
        ):
            latest = row
            latest_ts = parsed
    return latest


def _probe_fill_bill_matched_count(rows: Iterable[Mapping[str, Any]]) -> int:
    bill_ids: set[str] = set()
    pass_rows = 0
    for row in rows:
        if str(row.get("bill_match_status") or "").strip().upper() == "PASS":
            pass_rows += 1
        for key in ("entry_bill_id", "exit_bill_id"):
            for bill_id in str(row.get(key) or "").replace(",", ";").split(";"):
                bill_id = bill_id.strip()
                if bill_id:
                    bill_ids.add(bill_id)
    if bill_ids:
        return len(bill_ids)
    return pass_rows * 2


def _probe_fill_match_status(
    cost_probe_fill_count: int,
    latest_probe_roundtrip: Mapping[str, Any] | None,
) -> str:
    if latest_probe_roundtrip is not None:
        entry_filled = _positive_value(latest_probe_roundtrip.get("entry_filled_qty"))
        exit_filled = _positive_value(latest_probe_roundtrip.get("exit_filled_qty"))
        if entry_filled and exit_filled:
            return "entry_exit_fill_observed"
        if _bool_or_none(latest_probe_roundtrip.get("execution_completed")) is True:
            return "terminal_roundtrip_fill_incomplete"
    if cost_probe_fill_count >= 2:
        return "terminal_event_fills_observed"
    if cost_probe_fill_count > 0:
        return "partial_fill_observed"
    return "not_observed"


def _probe_bill_match_status(
    latest_probe_roundtrip: Mapping[str, Any] | None,
    bill_matched_count: int,
    latest_fill_bill_match: Mapping[str, Any] | None = None,
) -> str:
    if latest_fill_bill_match is not None:
        value = str(latest_fill_bill_match.get("bill_match_status") or "").strip()
        if value:
            return value
    if latest_probe_roundtrip is not None:
        value = str(latest_probe_roundtrip.get("bill_match_status") or "").strip()
        if value:
            return value
    return "bill_observed" if bill_matched_count > 0 else "bill_not_observed"


def _probe_fee_match_status(
    latest_probe_roundtrip: Mapping[str, Any] | None,
    latest_fill_bill_match: Mapping[str, Any] | None = None,
) -> str:
    if latest_fill_bill_match is not None:
        value = str(latest_fill_bill_match.get("bill_match_status") or "").strip().upper()
        if value == "PASS":
            return "fill_bill_fee_match"
        if value and value != "BILL_NOT_OBSERVED":
            return value.lower()
    if latest_probe_roundtrip is not None:
        value = str(latest_probe_roundtrip.get("fee_match_status") or "").strip()
        if value:
            return value
    return "fee_match_not_observed"


def _probe_fee_match_diff_usdt(
    latest_probe_roundtrip: Mapping[str, Any] | None,
    latest_fill_bill_match: Mapping[str, Any] | None = None,
) -> Any:
    if latest_fill_bill_match is not None:
        value = latest_fill_bill_match.get("fee_diff_usdt")
        if value not in {None, ""}:
            return value
    if latest_probe_roundtrip is None:
        return ""
    return latest_probe_roundtrip.get("fee_match_diff_usdt") or ""


def _row_ts(row: Mapping[str, Any], *keys: str) -> datetime | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        if value in {None, "", "null", "None"}:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed.astimezone(UTC) if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _roundtrip_cost_value(row: Mapping[str, Any], quantile: str) -> float:
    if not row:
        return 0.0
    explicit = _float_value(row, f"roundtrip_cost_{quantile}_bps")
    if explicit > 0:
        return explicit
    total = _float_value(row, f"total_cost_bps_{quantile}")
    return total * 2.0 if total > 0 else 0.0


def _cost_bootstrap_state(
    *,
    latest: Mapping[str, Any] | None,
    latest_actual: Mapping[str, Any] | None,
    latest_proxy: Mapping[str, Any] | None,
    stale_actual: bool,
    trusted_live: bool,
    actual_fill_count: int,
    mixed_fill_count: int,
    cost_probe_fill_count: int,
    sample_count: int,
    min_trusted_sample_count: int,
) -> str:
    if trusted_live:
        return "ACTUAL_FILLS_TRUSTED"
    if (latest_actual is not None and not stale_actual) or actual_fill_count > 0:
        if mixed_fill_count > 0 or _cost_source(latest_actual) == "mixed_actual_proxy":
            return "MIXED_ACTUAL_PROXY_AVAILABLE"
        if sample_count < min_trusted_sample_count:
            return "ACTUAL_FILLS_SMALL_SAMPLE"
        return "ACTUAL_FILLS_SMALL_SAMPLE"
    if cost_probe_fill_count >= 2:
        return "BOOTSTRAP_PROBE_AVAILABLE"
    if stale_actual:
        return "STALE"
    if cost_probe_fill_count > 0:
        return "BOOTSTRAP_PROBE_AVAILABLE"
    if latest_proxy is not None:
        return "PUBLIC_PROXY_ONLY"
    if latest is not None and _cost_source(latest) not in {"", "global_default"}:
        return "ACTUAL_FILLS_SMALL_SAMPLE"
    return "NO_COST_DATA"


def _cost_bootstrap_evidence_tier(state: str) -> str:
    return {
        "NO_COST_DATA": "global_default",
        "PUBLIC_PROXY_ONLY": "public_spread_proxy",
        "BOOTSTRAP_PROBE_AVAILABLE": "bootstrap_cost_probe",
        "MIXED_ACTUAL_PROXY_AVAILABLE": "mixed_actual_proxy",
        "ACTUAL_FILLS_SMALL_SAMPLE": "actual_fills_small_sample",
        "ACTUAL_FILLS_TRUSTED": "actual_fills_trusted",
        "STALE": "stale_actual_or_mixed",
        "BROKEN_RECONCILE": "broken_reconcile",
    }.get(state, "unknown")


def _cost_bootstrap_trust_level(
    state: str,
    *,
    trusted_for_live: bool,
    trusted_for_paper: bool,
) -> str:
    if trusted_for_live:
        return "live_small_review_candidate"
    if trusted_for_paper:
        return "paper_or_shadow_only"
    if state == "PUBLIC_PROXY_ONLY":
        return "diagnostic_only"
    return "not_trusted"


def _cost_bootstrap_next_action(
    state: str,
    *,
    fee_available: bool,
    slippage_available: bool,
    spread_available: bool,
    bill_matched_count: int,
) -> str:
    if state == "NO_COST_DATA":
        return "run okx read-only backfill; if still empty, review V5 cost_probe dry-run plan"
    if state == "PUBLIC_PROXY_ONLY":
        return "collect actual/mixed fee and slippage samples; public proxy is diagnostic only"
    if state == "BOOTSTRAP_PROBE_AVAILABLE":
        missing = _missing_components(
            fee_available=fee_available,
            slippage_available=slippage_available,
            spread_available=spread_available,
            bill_matched_count=bill_matched_count,
        )
        if not missing:
            return (
                "BOOTSTRAP_COMPLETE_BILL_MATCHED; keep live coverage disabled "
                "until trusted live samples and strategy evidence are present"
            )
        return f"bootstrap sample present; resolve {missing} before trusted live review"
    if state == "MIXED_ACTUAL_PROXY_AVAILABLE":
        return "continue read-only backfill and increase samples before live-small review"
    if state == "ACTUAL_FILLS_SMALL_SAMPLE":
        return "increase actual/mixed sample count before trusted coverage"
    if state == "ACTUAL_FILLS_TRUSTED":
        return "eligible for manual live-small cost review; strategy evidence still required"
    if state == "STALE":
        return "refresh actual/mixed cost anchors; stale rows do not count as live coverage"
    if state == "BROKEN_RECONCILE":
        return "fix lifecycle/fill/bill reconciliation before using samples"
    return "inspect cost bucket source and lifecycle evidence"


def _missing_components(
    *,
    fee_available: bool,
    slippage_available: bool,
    spread_available: bool,
    bill_matched_count: int,
) -> str:
    missing: list[str] = []
    if not fee_available:
        missing.append("fee")
    if not slippage_available:
        missing.append("slippage")
    if not spread_available:
        missing.append("spread")
    if bill_matched_count <= 0:
        missing.append("bill_match")
    return ",".join(missing)
