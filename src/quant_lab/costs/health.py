import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset

COST_HEALTH_DAILY_DATASET = Path("gold") / "cost_health_daily"
ACTUAL_SOURCE = "actual_okx_fills_and_bills"
ACTUAL_FILLS_SOURCE = "actual_fills"
MIXED_ACTUAL_PROXY_SOURCE = "mixed_actual_proxy"
FEE_ONLY_SOURCE = "actual_okx_fills_fee_missing"
PROXY_SOURCE = "public_spread_proxy"
DEFAULT_SOURCE = "global_default"
ACTUAL_SOURCES = {ACTUAL_SOURCE, ACTUAL_FILLS_SOURCE}
ACTUAL_OR_MIXED_SOURCES = {ACTUAL_SOURCE, ACTUAL_FILLS_SOURCE, MIXED_ACTUAL_PROXY_SOURCE}

COST_HEALTH_DAILY_SCHEMA = {
    "day": pl.Utf8,
    "status": pl.Utf8,
    "cost_model_version": pl.Utf8,
    "actual_rows": pl.Int64,
    "mixed_rows": pl.Int64,
    "proxy_rows": pl.Int64,
    "global_default_rows": pl.Int64,
    "fallback_ratio": pl.Float64,
    "hard_fallback_count": pl.Int64,
    "hard_fallback_ratio": pl.Float64,
    "soft_fallback_count": pl.Int64,
    "soft_fallback_ratio": pl.Float64,
    "proxy_only_count": pl.Int64,
    "global_default_count": pl.Int64,
    "symbols_with_actual_cost": pl.Utf8,
    "symbols_with_mixed_cost": pl.Utf8,
    "symbols_with_proxy_only": pl.Utf8,
    "symbols_proxy_only": pl.Utf8,
    "symbols_missing_cost": pl.Utf8,
    "actual_sample_count_by_symbol": pl.Utf8,
    "data_quality_checks_json": pl.Utf8,
    "min_sample_count": pl.Int64,
    "api_global_default_count": pl.Int64,
    "api_symbol_proxy_hit_count": pl.Int64,
    "api_regime_fallback_count": pl.Int64,
    "api_degraded_cost_count": pl.Int64,
    "api_cost_usage_rows": pl.Int64,
    "warnings_json": pl.Utf8,
    "created_at": pl.Utf8,
}


class CostHealthDaily(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    day: str
    status: str
    cost_model_version: str
    actual_rows: int = Field(ge=0)
    mixed_rows: int = Field(default=0, ge=0)
    proxy_rows: int = Field(ge=0)
    global_default_rows: int = Field(ge=0)
    fallback_ratio: float = Field(ge=0, le=1)
    hard_fallback_count: int = Field(default=0, ge=0)
    hard_fallback_ratio: float = Field(default=0.0, ge=0, le=1)
    soft_fallback_count: int = Field(default=0, ge=0)
    soft_fallback_ratio: float = Field(default=0.0, ge=0, le=1)
    proxy_only_count: int = Field(default=0, ge=0)
    global_default_count: int = Field(default=0, ge=0)
    symbols_with_actual_cost: list[str] = Field(default_factory=list)
    symbols_with_mixed_cost: list[str] = Field(default_factory=list)
    symbols_with_proxy_only: list[str] = Field(default_factory=list)
    symbols_proxy_only: list[str] = Field(default_factory=list)
    symbols_missing_cost: list[str] = Field(default_factory=list)
    actual_sample_count_by_symbol: dict[str, int] = Field(default_factory=dict)
    data_quality_checks_json: str = "{}"
    min_sample_count: int = Field(ge=1)
    api_global_default_count: int = Field(default=0, ge=0)
    api_symbol_proxy_hit_count: int = Field(default=0, ge=0)
    api_regime_fallback_count: int = Field(default=0, ge=0)
    api_degraded_cost_count: int = Field(default=0, ge=0)
    api_cost_usage_rows: int = Field(default=0, ge=0)
    warnings_json: str = "[]"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def build_cost_health_daily(
    cost_rows: pl.DataFrame,
    *,
    day: str,
    min_sample_count: int,
    expected_symbols: list[str] | None = None,
    private_fill_rows: int = 0,
    private_bill_rows: int = 0,
    v5_trade_rows: int = 0,
    v5_order_lifecycle_rows: int = 0,
    v5_lifecycle_zero_fill_count: int = 0,
    v5_lifecycle_missing_cost_count: int = 0,
    fee_bps_missing_count: int = 0,
    api_global_default_count: int = 0,
    api_symbol_proxy_hit_count: int = 0,
    api_regime_fallback_count: int = 0,
    api_degraded_cost_count: int = 0,
    api_cost_usage_rows: int = 0,
) -> CostHealthDaily:
    expected = set(expected_symbols or [])
    if cost_rows.is_empty():
        warnings = ["cost_bucket_daily empty"]
        return CostHealthDaily(
            day=day,
            status="CRITICAL",
            cost_model_version=f"cost_bucket_daily:{day}",
            actual_rows=0,
            mixed_rows=0,
            proxy_rows=0,
            global_default_rows=0,
            fallback_ratio=1.0,
            hard_fallback_count=1,
            hard_fallback_ratio=1.0,
            soft_fallback_count=0,
            soft_fallback_ratio=0.0,
            proxy_only_count=0,
            global_default_count=0,
            symbols_missing_cost=sorted(expected),
            data_quality_checks_json=_json(
                _data_quality_checks(
                    private_fill_rows=private_fill_rows,
                    private_bill_rows=private_bill_rows,
                    v5_trade_rows=v5_trade_rows,
                    v5_order_lifecycle_rows=v5_order_lifecycle_rows,
                    v5_lifecycle_zero_fill_count=v5_lifecycle_zero_fill_count,
                    v5_lifecycle_missing_cost_count=v5_lifecycle_missing_cost_count,
                    actual_rows_count=0,
                    actual_or_mixed_rows=0,
                    fee_bps_missing_count=fee_bps_missing_count,
                    actual_symbols=set(),
                    mixed_symbols=set(),
                    expected_symbols=expected,
                )
            ),
            min_sample_count=min_sample_count,
            api_global_default_count=api_global_default_count,
            api_symbol_proxy_hit_count=api_symbol_proxy_hit_count,
            api_regime_fallback_count=api_regime_fallback_count,
            api_degraded_cost_count=api_degraded_cost_count,
            api_cost_usage_rows=api_cost_usage_rows,
            warnings_json=_json(warnings),
        )

    rows = cost_rows.to_dicts()
    actual_rows = [row for row in rows if str(row.get("source")) in ACTUAL_SOURCES]
    mixed_rows = [row for row in rows if str(row.get("source")) == MIXED_ACTUAL_PROXY_SOURCE]
    actual_or_mixed_rows = [*actual_rows, *mixed_rows]
    trusted_actual_rows = [
        row
        for row in actual_or_mixed_rows
        if str(row.get("source")) in {ACTUAL_SOURCE, ACTUAL_FILLS_SOURCE}
        and int(row.get("sample_count") or 0) >= min_sample_count
    ]
    proxy_rows = [row for row in rows if str(row.get("source")) == PROXY_SOURCE]
    global_rows = [row for row in rows if str(row.get("source")) == DEFAULT_SOURCE]
    fallback_count = sum(1 for row in rows if _is_fallback_row(row))
    fallback_ratio = fallback_count / len(rows)
    hard_fallback_count = sum(1 for row in rows if _is_hard_fallback_row(row))
    soft_fallback_count = sum(1 for row in rows if _is_soft_fallback_row(row))
    hard_fallback_ratio = hard_fallback_count / len(rows)
    soft_fallback_ratio = soft_fallback_count / len(rows)

    actual_symbols = {
        str(row.get("symbol"))
        for row in actual_rows
        if str(row.get("source")) in {ACTUAL_SOURCE, ACTUAL_FILLS_SOURCE}
    }
    mixed_symbols = {str(row.get("symbol")) for row in mixed_rows}
    proxy_symbols = {str(row.get("symbol")) for row in proxy_rows}
    global_symbols = {str(row.get("symbol")) for row in global_rows}
    known_symbols = actual_symbols | mixed_symbols | proxy_symbols | global_symbols
    missing = expected.difference(known_symbols)
    proxy_only = proxy_symbols.difference(actual_symbols | mixed_symbols)
    data_quality_checks = _data_quality_checks(
        private_fill_rows=private_fill_rows,
        private_bill_rows=private_bill_rows,
        v5_trade_rows=v5_trade_rows,
        v5_order_lifecycle_rows=v5_order_lifecycle_rows,
        v5_lifecycle_zero_fill_count=v5_lifecycle_zero_fill_count,
        v5_lifecycle_missing_cost_count=v5_lifecycle_missing_cost_count,
        actual_rows_count=len(actual_rows),
        actual_or_mixed_rows=len(actual_or_mixed_rows),
        fee_bps_missing_count=fee_bps_missing_count,
        actual_symbols=actual_symbols,
        mixed_symbols=mixed_symbols,
        expected_symbols=expected,
    )

    warnings: list[str] = []
    if fallback_ratio > 0.8:
        warnings.append("fallback_ratio_gt_0.8")
    elif fallback_ratio > 0.5:
        warnings.append("fallback_ratio_gt_0.5")
    if hard_fallback_ratio > 0.25:
        warnings.append("hard_fallback_ratio_gt_0.25")
    if soft_fallback_ratio > 0.5:
        warnings.append("soft_fallback_ratio_gt_0.5")
    if len(proxy_rows) == len(rows):
        warnings.append("all_rows_public_spread_proxy")
    if len(global_rows) == len(rows):
        warnings.append("all_rows_global_default")
    if missing:
        warnings.append("symbols_missing_cost")
    for name, passed in data_quality_checks.items():
        if passed is False:
            warnings.append(name)

    all_proxy = len(proxy_rows) == len(rows)
    all_global = len(global_rows) == len(rows)
    status = "OK"
    if (
        all_global
        or (hard_fallback_ratio > 0.25 and not actual_or_mixed_rows)
        or (missing and not actual_or_mixed_rows)
        or data_quality_checks.get("private_fills_present_but_actual_cost_zero") is False
        or data_quality_checks.get("trades_present_but_not_in_cost_model") is False
        or data_quality_checks.get("lifecycle_present_but_not_in_actual_cost") is False
        or data_quality_checks.get("filled_order_missing_lifecycle_cost") is False
    ):
        status = "CRITICAL"
    elif all_proxy or soft_fallback_ratio > 0.5 or not trusted_actual_rows:
        status = "WARNING"

    return CostHealthDaily(
        day=day,
        status=status,
        cost_model_version=_cost_model_version(rows, day),
        actual_rows=len(actual_rows),
        mixed_rows=len(mixed_rows),
        proxy_rows=len(proxy_rows),
        global_default_rows=len(global_rows),
        fallback_ratio=fallback_ratio,
        hard_fallback_count=hard_fallback_count,
        hard_fallback_ratio=hard_fallback_ratio,
        soft_fallback_count=soft_fallback_count,
        soft_fallback_ratio=soft_fallback_ratio,
        proxy_only_count=len(proxy_only),
        global_default_count=len(global_rows),
        symbols_with_actual_cost=sorted(actual_symbols),
        symbols_with_mixed_cost=sorted(mixed_symbols),
        symbols_with_proxy_only=sorted(proxy_only),
        symbols_proxy_only=sorted(proxy_only),
        symbols_missing_cost=sorted(missing),
        actual_sample_count_by_symbol=_actual_sample_count_by_symbol(actual_or_mixed_rows),
        data_quality_checks_json=_json(data_quality_checks),
        min_sample_count=min_sample_count,
        api_global_default_count=api_global_default_count,
        api_symbol_proxy_hit_count=api_symbol_proxy_hit_count,
        api_regime_fallback_count=api_regime_fallback_count,
        api_degraded_cost_count=api_degraded_cost_count,
        api_cost_usage_rows=api_cost_usage_rows,
        warnings_json=_json(warnings),
    )


def summarize_cost_api_usage(cost_usage: pl.DataFrame) -> dict[str, int]:
    if cost_usage.is_empty():
        return {
            "api_global_default_count": 0,
            "api_symbol_proxy_hit_count": 0,
            "api_regime_fallback_count": 0,
            "api_degraded_cost_count": 0,
            "api_cost_usage_rows": 0,
        }

    global_default = 0
    symbol_proxy_hit = 0
    regime_fallback = 0
    degraded = 0
    rows = cost_usage.to_dicts()
    for row in rows:
        payload = _payload(row.get("raw_payload_json"))
        source = _lower_text(
            _first_value(row, payload, ["cost_source", "source", "response.cost_source"])
        )
        fallback_level = _upper_text(
            _first_value(row, payload, ["fallback_level", "response.fallback_level"])
        )
        if source == DEFAULT_SOURCE or fallback_level == "GLOBAL_DEFAULT":
            global_default += 1
        if source in {PROXY_SOURCE, "public_proxy"}:
            symbol_proxy_hit += 1
        if "REGIME_FALLBACK" in fallback_level:
            regime_fallback += 1
        if _truthy(
            _first_value(
                row,
                payload,
                ["degraded_cost_model", "response.degraded_cost_model"],
            )
        ):
            degraded += 1

    return {
        "api_global_default_count": global_default,
        "api_symbol_proxy_hit_count": symbol_proxy_hit,
        "api_regime_fallback_count": regime_fallback,
        "api_degraded_cost_count": degraded,
        "api_cost_usage_rows": len(rows),
    }


def publish_cost_health_daily(lake_root: str | Path, row: CostHealthDaily) -> int:
    frame = cost_health_daily_frame([row])
    return upsert_parquet_dataset(
        frame,
        Path(lake_root) / COST_HEALTH_DAILY_DATASET,
        key_columns=["day", "cost_model_version"],
    )


def read_cost_health_daily(lake_root: str | Path, day: str | None = None) -> dict[str, Any]:
    df = read_parquet_dataset(Path(lake_root) / COST_HEALTH_DAILY_DATASET)
    if df.is_empty():
        return {"status": "missing", "rows": 0, "warnings": ["cost_health_daily missing"]}
    filtered = df
    if day and "day" in filtered.columns:
        filtered = filtered.filter(pl.col("day") == day)
    if filtered.is_empty():
        return {"status": "missing", "rows": 0, "warnings": ["cost_health_daily day missing"]}
    row = filtered.sort("day").tail(1).to_dicts()[0]
    return {
        **row,
        "rows": filtered.height,
        "warnings": _loads(row.get("warnings_json")),
        "symbols_with_actual_cost": _loads(row.get("symbols_with_actual_cost")),
        "symbols_with_mixed_cost": _loads(row.get("symbols_with_mixed_cost")),
        "symbols_with_proxy_only": _loads(row.get("symbols_with_proxy_only")),
        "symbols_proxy_only": _loads(row.get("symbols_proxy_only")),
        "symbols_missing_cost": _loads(row.get("symbols_missing_cost")),
        "actual_sample_count_by_symbol": _loads(row.get("actual_sample_count_by_symbol")),
        "data_quality_checks": _loads(row.get("data_quality_checks_json")),
    }


def cost_health_daily_frame(rows: list[CostHealthDaily]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                **row.model_dump(mode="json"),
                "symbols_with_actual_cost": _json(row.symbols_with_actual_cost),
                "symbols_with_mixed_cost": _json(row.symbols_with_mixed_cost),
                "symbols_with_proxy_only": _json(row.symbols_with_proxy_only),
                "symbols_proxy_only": _json(row.symbols_proxy_only),
                "symbols_missing_cost": _json(row.symbols_missing_cost),
                "actual_sample_count_by_symbol": _json(row.actual_sample_count_by_symbol),
            }
            for row in rows
        ],
        schema=COST_HEALTH_DAILY_SCHEMA,
        orient="row",
    )


def _payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _first_value(row: dict[str, Any], payload: dict[str, Any], fields: list[str]) -> Any:
    for field in fields:
        value = _nested_value(row, field)
        if value not in {None, ""}:
            return value
        value = _nested_value(payload, field)
        if value not in {None, ""}:
            return value
    return None


def _nested_value(source: dict[str, Any], field: str) -> Any:
    current: Any = source
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _lower_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _upper_text(value: Any) -> str:
    return str(value or "").strip().upper()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_fallback_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "")
    fallback = str(row.get("fallback_level") or "")
    if source in {PROXY_SOURCE, DEFAULT_SOURCE, FEE_ONLY_SOURCE}:
        return True
    return fallback not in {"", "NONE"}


def _is_hard_fallback_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or row.get("cost_source") or "").strip().lower()
    rendered = _fallback_text(row)
    return source == DEFAULT_SOURCE or any(
        token in rendered
        for token in [
            "GLOBAL_DEFAULT",
            "SYMBOL_MISSING",
            "SERVICE_UNAVAILABLE",
            "STALE_COST_BUCKET",
        ]
    )


def _is_soft_fallback_row(row: dict[str, Any]) -> bool:
    if _is_hard_fallback_row(row):
        return False
    rendered = _fallback_text(row)
    return any(
        token in rendered
        for token in ["SAMPLE_TOO_SMALL", "SLIPPAGE_UNKNOWN", "SPREAD_PROXY"]
    )


def _fallback_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(column) or "").strip().upper()
        for column in ["source", "cost_source", "fallback_level", "fallback_reason"]
    )


def _actual_sample_count_by_symbol(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        symbol = str(row.get("symbol") or "")
        if not symbol or symbol == "GLOBAL":
            continue
        sample_count = max(
            int(row.get("sample_count") or 0),
            int(row.get("actual_fill_count") or 0),
            int(row.get("mixed_fill_count") or 0),
        )
        counts[symbol] = max(counts.get(symbol, 0), sample_count)
    return counts


def _data_quality_checks(
    *,
    private_fill_rows: int,
    private_bill_rows: int,
    v5_trade_rows: int,
    v5_order_lifecycle_rows: int,
    v5_lifecycle_zero_fill_count: int,
    v5_lifecycle_missing_cost_count: int,
    actual_rows_count: int,
    actual_or_mixed_rows: int,
    fee_bps_missing_count: int,
    actual_symbols: set[str],
    mixed_symbols: set[str],
    expected_symbols: set[str],
) -> dict[str, bool | str]:
    actual_or_mixed = actual_symbols | mixed_symbols
    fill_like_rows = private_fill_rows + v5_trade_rows + v5_order_lifecycle_rows
    fee_missing_rate = (
        "n/a" if fill_like_rows == 0 else f"{fee_bps_missing_count}/{fill_like_rows}"
    )
    return {
        "private_fills_present_but_actual_cost_zero": not (
            private_fill_rows > 0 and actual_or_mixed_rows == 0
        ),
        "trades_present_but_not_in_cost_model": not (
            v5_trade_rows > 0 and actual_or_mixed_rows == 0
        ),
        "lifecycle_present_but_not_in_actual_cost": not (
            v5_order_lifecycle_rows > 0 and actual_rows_count == 0
        ),
        "filled_order_missing_lifecycle_cost": v5_lifecycle_missing_cost_count == 0,
        "fill_count_zero_for_filled_order": v5_lifecycle_zero_fill_count == 0,
        "bills_present_but_fee_bps_missing": not (
            private_bill_rows > 0 and fee_bps_missing_count > 0
        ),
        "fee_missing_rate": fee_missing_rate,
        "actual_cost_symbol_coverage": (
            "n/a"
            if fill_like_rows == 0
            else f"{len(actual_or_mixed)}/{len(expected_symbols or actual_or_mixed)}"
        ),
    }


def _cost_model_version(rows: list[dict[str, Any]], day: str) -> str:
    values = sorted(
        {str(row.get("cost_model_version") or "") for row in rows if row.get("cost_model_version")}
    )
    return "+".join(values) if values else f"cost_bucket_daily:{day}"


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
