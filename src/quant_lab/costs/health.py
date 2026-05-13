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
ACTUAL_SOURCES = {ACTUAL_SOURCE, ACTUAL_FILLS_SOURCE, MIXED_ACTUAL_PROXY_SOURCE}

COST_HEALTH_DAILY_SCHEMA = {
    "day": pl.Utf8,
    "status": pl.Utf8,
    "cost_model_version": pl.Utf8,
    "actual_rows": pl.Int64,
    "proxy_rows": pl.Int64,
    "global_default_rows": pl.Int64,
    "fallback_ratio": pl.Float64,
    "symbols_with_actual_cost": pl.Utf8,
    "symbols_with_proxy_only": pl.Utf8,
    "symbols_missing_cost": pl.Utf8,
    "min_sample_count": pl.Int64,
    "warnings_json": pl.Utf8,
    "created_at": pl.Utf8,
}


class CostHealthDaily(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    day: str
    status: str
    cost_model_version: str
    actual_rows: int = Field(ge=0)
    proxy_rows: int = Field(ge=0)
    global_default_rows: int = Field(ge=0)
    fallback_ratio: float = Field(ge=0, le=1)
    symbols_with_actual_cost: list[str] = Field(default_factory=list)
    symbols_with_proxy_only: list[str] = Field(default_factory=list)
    symbols_missing_cost: list[str] = Field(default_factory=list)
    min_sample_count: int = Field(ge=1)
    warnings_json: str = "[]"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


def build_cost_health_daily(
    cost_rows: pl.DataFrame,
    *,
    day: str,
    min_sample_count: int,
    expected_symbols: list[str] | None = None,
) -> CostHealthDaily:
    expected = set(expected_symbols or [])
    if cost_rows.is_empty():
        warnings = ["cost_bucket_daily empty"]
        return CostHealthDaily(
            day=day,
            status="CRITICAL",
            cost_model_version=f"cost_bucket_daily:{day}",
            actual_rows=0,
            proxy_rows=0,
            global_default_rows=0,
            fallback_ratio=1.0,
            symbols_missing_cost=sorted(expected),
            min_sample_count=min_sample_count,
            warnings_json=_json(warnings),
        )

    rows = cost_rows.to_dicts()
    actual_rows = [
        row
        for row in rows
        if str(row.get("source")) in ACTUAL_SOURCES
        and int(row.get("sample_count") or 0) >= min_sample_count
    ]
    proxy_rows = [row for row in rows if str(row.get("source")) == PROXY_SOURCE]
    global_rows = [row for row in rows if str(row.get("source")) == DEFAULT_SOURCE]
    fallback_count = sum(1 for row in rows if _is_fallback_row(row))
    fallback_ratio = fallback_count / len(rows)

    actual_symbols = {str(row.get("symbol")) for row in actual_rows}
    proxy_symbols = {str(row.get("symbol")) for row in proxy_rows}
    global_symbols = {str(row.get("symbol")) for row in global_rows}
    known_symbols = actual_symbols | proxy_symbols | global_symbols
    missing = expected.difference(known_symbols)
    proxy_only = proxy_symbols.difference(actual_symbols)

    warnings: list[str] = []
    if fallback_ratio > 0.8:
        warnings.append("fallback_ratio_gt_0.8")
    elif fallback_ratio > 0.5:
        warnings.append("fallback_ratio_gt_0.5")
    if len(proxy_rows) == len(rows):
        warnings.append("all_rows_public_spread_proxy")
    if len(global_rows) == len(rows):
        warnings.append("all_rows_global_default")
    if missing:
        warnings.append("symbols_missing_cost")

    all_proxy = len(proxy_rows) == len(rows)
    all_global = len(global_rows) == len(rows)
    status = "OK"
    if all_global or (fallback_ratio > 0.8 and not all_proxy) or missing and not actual_rows:
        status = "CRITICAL"
    elif all_proxy or fallback_ratio > 0.5 or not actual_rows:
        status = "WARNING"

    return CostHealthDaily(
        day=day,
        status=status,
        cost_model_version=_cost_model_version(rows, day),
        actual_rows=len(actual_rows),
        proxy_rows=len(proxy_rows),
        global_default_rows=len(global_rows),
        fallback_ratio=fallback_ratio,
        symbols_with_actual_cost=sorted(actual_symbols),
        symbols_with_proxy_only=sorted(proxy_only),
        symbols_missing_cost=sorted(missing),
        min_sample_count=min_sample_count,
        warnings_json=_json(warnings),
    )


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
        "symbols_with_proxy_only": _loads(row.get("symbols_with_proxy_only")),
        "symbols_missing_cost": _loads(row.get("symbols_missing_cost")),
    }


def cost_health_daily_frame(rows: list[CostHealthDaily]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                **row.model_dump(mode="json"),
                "symbols_with_actual_cost": _json(row.symbols_with_actual_cost),
                "symbols_with_proxy_only": _json(row.symbols_with_proxy_only),
                "symbols_missing_cost": _json(row.symbols_missing_cost),
            }
            for row in rows
        ],
        schema=COST_HEALTH_DAILY_SCHEMA,
        orient="row",
    )


def _is_fallback_row(row: dict[str, Any]) -> bool:
    source = str(row.get("source") or "")
    fallback = str(row.get("fallback_level") or "")
    if source in {PROXY_SOURCE, DEFAULT_SOURCE, FEE_ONLY_SOURCE}:
        return True
    return fallback not in {"", "NONE"}


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
