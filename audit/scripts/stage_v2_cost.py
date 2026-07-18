# ruff: noqa: E501
"""Read-only v2 execution-cost coverage and distribution audit."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl


def _table(path: Path, table: str) -> pl.DataFrame:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return pl.read_database(f'SELECT * FROM "{table}"', connection)
    finally:
        connection.close()


def _first(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _quantile_summary(values: np.ndarray, rng: np.random.Generator) -> dict:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "sample_count": 0,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p50_ci_95_low": None,
            "p50_ci_95_high": None,
        }
    boot = np.asarray(
        [np.quantile(rng.choice(clean, size=clean.size, replace=True), 0.50) for _ in range(500)]
    )
    return {
        "sample_count": int(clean.size),
        "p50": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "p90": float(np.quantile(clean, 0.90)),
        "p95": float(np.quantile(clean, 0.95)),
        "p50_ci_95_low": float(np.quantile(boot, 0.025)),
        "p50_ci_95_high": float(np.quantile(boot, 0.975)),
    }


def main() -> None:
    root = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2"))
    v1_root = Path(os.environ.get("AUDIT_V1_ROOT", "/home/hr/quant-alpha-audit"))
    state = root / "state/v5_current"
    artifacts = root / "artifacts"
    reports = root / "reports"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    fills = _table(state / "fills.sqlite", "fills")
    orders = _table(state / "orders.sqlite", "orders")
    fill_columns, order_columns = fills.columns, orders.columns
    inst_col = _first(fill_columns, ["inst_id", "symbol"])
    px_col = _first(fill_columns, ["fill_px", "price", "px"])
    size_col = _first(fill_columns, ["fill_sz", "size", "sz", "qty"])
    fee_col = _first(fill_columns, ["fee"])
    fee_ccy_col = _first(fill_columns, ["fee_ccy", "fee_currency"])
    side_col = _first(fill_columns, ["side"])
    exec_col = _first(fill_columns, ["exec_type", "liquidity", "maker_taker"])
    fill_ts_col = _first(fill_columns, ["ts_ms", "fill_ts_ms", "timestamp"])
    fill_key = _first(fill_columns, ["cl_ord_id", "client_order_id", "ord_id", "order_id"])
    order_key = _first(order_columns, ["cl_ord_id", "client_order_id", "ord_id", "order_id"])
    if not all([inst_col, px_col, size_col, fill_ts_col]):
        raise SystemExit(f"fills schema missing required columns: {fill_columns}")
    expressions = [
        pl.col(inst_col).cast(pl.Utf8).alias("symbol"),
        pl.col(px_col).cast(pl.Float64, strict=False).alias("fill_price"),
        pl.col(size_col).cast(pl.Float64, strict=False).alias("fill_size"),
        pl.col(fill_ts_col).cast(pl.Int64, strict=False).alias("fill_ts_ms"),
        pl.col(side_col).cast(pl.Utf8).str.to_lowercase().alias("side")
        if side_col
        else pl.lit("unknown").alias("side"),
        pl.col(exec_col).cast(pl.Utf8).str.to_lowercase().alias("maker_taker")
        if exec_col
        else pl.lit("unknown").alias("maker_taker"),
        pl.col(fee_col).cast(pl.Float64, strict=False).alias("fee")
        if fee_col
        else pl.lit(None).cast(pl.Float64).alias("fee"),
        pl.col(fee_ccy_col).cast(pl.Utf8).str.to_uppercase().alias("fee_currency")
        if fee_ccy_col
        else pl.lit(None).cast(pl.Utf8).alias("fee_currency"),
        pl.col(fill_key).cast(pl.Utf8).alias("order_key")
        if fill_key
        else pl.lit(None).cast(pl.Utf8).alias("order_key"),
    ]
    panel = fills.select(expressions).with_columns(
        (pl.col("fill_price") * pl.col("fill_size")).alias("notional_usdt"),
        pl.from_epoch("fill_ts_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("fill_ts"),
        pl.col("symbol").str.split("-").list.first().alias("base_currency"),
    ).with_columns(
        pl.when(pl.col("fee_currency").is_in(["USDT", "USD"]))
        .then(pl.col("fee").abs())
        .when(pl.col("fee_currency") == pl.col("base_currency").str.to_uppercase())
        .then(pl.col("fee").abs() * pl.col("fill_price"))
        .otherwise(None)
        .alias("fee_usdt")
    ).with_columns(
        (pl.col("fee_usdt") / pl.col("notional_usdt") * 10_000).alias("fee_bps")
    )

    requested_col = _first(order_columns, ["requested_price", "requested_px", "px", "price"])
    order_ts_col = _first(order_columns, ["created_ts_ms", "c_time", "ctime", "created_at", "ts_ms"])
    bid_col = _first(order_columns, ["bid", "best_bid", "bid_px"])
    ask_col = _first(order_columns, ["ask", "best_ask", "ask_px"])
    order_type_col = _first(order_columns, ["ord_type", "order_type", "type"])
    state_col = _first(order_columns, ["state", "status"])
    if fill_key and order_key:
        order_select = [pl.col(order_key).cast(pl.Utf8).alias("order_key")]
        for source, alias, dtype in (
            (requested_col, "requested_price", pl.Float64),
            (order_ts_col, "order_ts_ms", pl.Int64),
            (bid_col, "bid_snapshot", pl.Float64),
            (ask_col, "ask_snapshot", pl.Float64),
            (order_type_col, "order_type", pl.Utf8),
            (state_col, "order_state", pl.Utf8),
        ):
            order_select.append(
                pl.col(source).cast(dtype, strict=False).alias(alias)
                if source
                else pl.lit(None).cast(dtype).alias(alias)
            )
        order_frame = orders.select(order_select).unique("order_key", keep="last")
        panel = panel.join(order_frame, on="order_key", how="left")
    else:
        panel = panel.with_columns(
            pl.lit(None).cast(pl.Float64).alias("requested_price"),
            pl.lit(None).cast(pl.Int64).alias("order_ts_ms"),
            pl.lit(None).cast(pl.Float64).alias("bid_snapshot"),
            pl.lit(None).cast(pl.Float64).alias("ask_snapshot"),
            pl.lit(None).cast(pl.Utf8).alias("order_type"),
            pl.lit(None).cast(pl.Utf8).alias("order_state"),
        )
    panel = panel.with_columns(
        pl.when((pl.col("requested_price") > 0) & (pl.col("side") == "buy"))
        .then((pl.col("fill_price") / pl.col("requested_price") - 1.0) * 10_000)
        .when((pl.col("requested_price") > 0) & (pl.col("side") == "sell"))
        .then((pl.col("requested_price") / pl.col("fill_price") - 1.0) * 10_000)
        .otherwise(None)
        .alias("arrival_slippage_bps"),
        pl.when((pl.col("bid_snapshot") > 0) & (pl.col("ask_snapshot") > 0))
        .then(
            (pl.col("ask_snapshot") - pl.col("bid_snapshot"))
            / ((pl.col("ask_snapshot") + pl.col("bid_snapshot")) / 2.0)
            * 10_000
        )
        .otherwise(None)
        .alias("spread_bps"),
        (pl.col("fill_ts_ms") - pl.col("order_ts_ms")).cast(pl.Float64).alias("order_latency_ms"),
    ).with_columns(
        (pl.col("fee_bps") + pl.col("arrival_slippage_bps").fill_null(0.0).clip(lower_bound=0.0)).alias("observed_cost_bps")
    )
    counts = panel.group_by("order_key").agg(pl.len().alias("fill_count_for_order"))
    panel = panel.join(counts, on="order_key", how="left")

    bars = pl.read_parquet(v1_root / "data/silver/bars_1h.parquet").filter(
        pl.col("symbol").is_in(panel["symbol"].unique().implode())
    ).sort(["symbol", "ts"])
    features = bars.with_columns(
        pl.col("close").pct_change().rolling_std(24).over("symbol").alias("volatility_24h"),
        pl.col("quote_volume").rolling_mean(24).over("symbol").alias("liquidity_24h"),
    ).select(["symbol", "ts", "volatility_24h", "liquidity_24h"])
    panel = panel.sort(["symbol", "fill_ts"]).join_asof(
        features,
        left_on="fill_ts",
        right_on="ts",
        by="symbol",
        strategy="backward",
        check_sortedness=False,
    )
    vol_values = panel["volatility_24h"].drop_nulls()
    liq_values = panel["liquidity_24h"].drop_nulls()
    n50 = float(panel["notional_usdt"].quantile(0.5))
    n90 = float(panel["notional_usdt"].quantile(0.9))
    v50 = float(vol_values.quantile(0.5)) if vol_values.len() else 0.0
    v90 = float(vol_values.quantile(0.9)) if vol_values.len() else 0.0
    l50 = float(liq_values.quantile(0.5)) if liq_values.len() else 0.0
    l90 = float(liq_values.quantile(0.9)) if liq_values.len() else 0.0
    panel = panel.with_columns(
        pl.when(pl.col("notional_usdt") <= n50).then(pl.lit("small")).when(pl.col("notional_usdt") <= n90).then(pl.lit("medium")).otherwise(pl.lit("large")).alias("notional_bucket"),
        pl.when(pl.col("volatility_24h") <= v50).then(pl.lit("low")).when(pl.col("volatility_24h") <= v90).then(pl.lit("medium")).otherwise(pl.lit("high")).alias("volatility_bucket"),
        pl.when(pl.col("liquidity_24h") <= l50).then(pl.lit("low")).when(pl.col("liquidity_24h") <= l90).then(pl.lit("medium")).otherwise(pl.lit("high")).alias("liquidity_bucket"),
        pl.col("fill_ts").dt.strftime("%Y-%m").alias("time_period"),
    )

    coverage = panel.group_by("symbol").agg(
        pl.len().alias("fill_count"),
        (pl.col("side") == "buy").sum().alias("buy_fill_count"),
        (pl.col("side") == "sell").sum().alias("sell_fill_count"),
        pl.col("maker_taker").filter(pl.col("maker_taker").str.contains("maker")).len().alias("maker_count"),
        pl.col("maker_taker").filter(pl.col("maker_taker").str.contains("taker")).len().alias("taker_count"),
        pl.col("fee_bps").is_not_null().sum().alias("fee_coverage_count"),
        pl.col("arrival_slippage_bps").is_not_null().sum().alias("requested_price_coverage_count"),
        pl.col("spread_bps").is_not_null().sum().alias("bid_ask_coverage_count"),
        pl.col("order_latency_ms").is_not_null().sum().alias("latency_coverage_count"),
        (pl.col("fill_count_for_order") > 1).sum().alias("partial_fill_rows"),
        pl.col("notional_usdt").sum().alias("notional_usdt"),
        pl.col("fill_ts").min().alias("first_fill_ts"),
        pl.col("fill_ts").max().alias("last_fill_ts"),
    ).sort("fill_count", descending=True)
    coverage.write_csv(artifacts / "cost_coverage_by_symbol.csv")

    rng = np.random.default_rng(20260718)
    distribution_rows: list[dict] = []
    dimensions = [
        ("overall", None),
        ("symbol", "symbol"),
        ("side", "side"),
        ("maker_taker", "maker_taker"),
        ("notional", "notional_bucket"),
        ("volatility", "volatility_bucket"),
        ("liquidity", "liquidity_bucket"),
        ("time_period", "time_period"),
    ]
    for dimension, column in dimensions:
        groups = [("ALL", panel)] if column is None else panel.partition_by(column, as_dict=True)
        if column is not None:
            groups = [
                ((key[0] if isinstance(key, tuple) else key), frame)
                for key, frame in groups.items()
            ]
        for bucket, group in groups:
            for metric in ("fee_bps", "arrival_slippage_bps", "spread_bps", "order_latency_ms", "observed_cost_bps"):
                summary = _quantile_summary(group[metric].to_numpy(), rng)
                distribution_rows.append(
                    {
                        "dimension": dimension,
                        "bucket": str(bucket),
                        "metric": metric,
                        **summary,
                    }
                )
    distribution = pl.DataFrame(distribution_rows)
    distribution.write_csv(artifacts / "cost_distribution_v2.csv")

    covered_symbols = coverage.height
    requested_coverage = int(panel["arrival_slippage_bps"].is_not_null().sum())
    direct_cost_adequate = covered_symbols >= 20 and requested_coverage >= 200
    metadata = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": "read-only copy of current qyun reports/fills.sqlite and orders.sqlite",
        "fill_schema_columns": fill_columns,
        "order_schema_columns": order_columns,
        "fill_rows": panel.height,
        "covered_symbols": covered_symbols,
        "audit_universe_symbols": 93,
        "symbol_coverage": covered_symbols / 93.0,
        "requested_price_coverage_rows": requested_coverage,
        "bid_ask_coverage_rows": int(panel["spread_bps"].is_not_null().sum()),
        "latency_coverage_rows": int(panel["order_latency_ms"].is_not_null().sum()),
        "direct_cost_adequate": direct_cost_adequate,
        "scenario_policy": "optimistic/base/stress collapse to the same conservative 15bps one-way / 30bps roundtrip floor because broad-universe arrival-cost evidence is insufficient",
        "scenario_one_way_bps": {"optimistic": 15.0, "base": 15.0, "stress": 15.0},
        "production_mutations": 0,
        "secrets_read": False,
    }
    (artifacts / "cost_model_v2_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
