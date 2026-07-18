# ruff: noqa: E501
"""Conservative capacity diagnostics; never assume frictionless close fills."""

from __future__ import annotations

import math
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.paths import (  # noqa: E402
    ARTIFACTS,
    DATA_GOLD,
    DATA_SILVER,
    REPORTS,
    snapshot_dir,
)

NOTIONALS = [10.0, 100.0, 1_000.0, 10_000.0]
UNIVERSES = ["top10", "top20", "top50"]


def _actual_fill_notionals() -> list[float]:
    path = snapshot_dir() / "v5" / "db" / "fills.sqlite"
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute("SELECT fill_px, fill_sz FROM fills").fetchall()
    finally:
        connection.close()
    values = []
    for price, size in rows:
        try:
            value = float(price) * float(size)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    return values


def main() -> None:
    bars = pl.read_parquet(DATA_SILVER / "bars_1h.parquet").with_columns(
        pl.col("ts").dt.date().alias("date")
    )
    memberships = pl.read_parquet(DATA_GOLD / "dynamic_universe.parquet")
    base_cost = (
        pl.read_csv(ARTIFACTS / "cost_model_summary.csv")
        .filter(pl.col("scenario") == "base")
        .row(0, named=True)
    )
    actual_notionals = _actual_fill_notionals()
    actual_median = (
        sorted(actual_notionals)[len(actual_notionals) // 2] if actual_notionals else None
    )
    actual_p90 = (
        sorted(actual_notionals)[int(0.90 * (len(actual_notionals) - 1))]
        if actual_notionals
        else None
    )

    rows: list[dict] = []
    for universe in UNIVERSES:
        eligible = bars.join(
            memberships.filter(pl.col("universe") == universe).select(["date", "symbol"]),
            on=["date", "symbol"],
            how="inner",
        ).filter(pl.col("quote_volume") > 0)
        for notional in NOTIONALS:
            participation = eligible.select(
                (notional / pl.col("quote_volume")).alias("participation")
            )["participation"]
            p50 = float(participation.quantile(0.50))
            p75 = float(participation.quantile(0.75))
            p90 = float(participation.quantile(0.90))
            within_limit = float((participation <= 0.05).mean())
            # Uncalibrated square-root impact is a conservative diagnostic only;
            # it is never used to award PASS.
            additional_impact_bps = 10.0 * math.sqrt(max(p90, 0.0))
            if p90 > 0.01:
                liquidity_risk = "HIGH"
            elif p90 > 0.001:
                liquidity_risk = "MEDIUM"
            else:
                liquidity_risk = "LOW"
            rows.append(
                {
                    "universe": universe,
                    "notional_usdt": notional,
                    "min_order_amount_usdt": 10.0,
                    "min_order_check": "PASS" if notional >= 10.0 else "FAIL",
                    "quantity_precision_check": "INCONCLUSIVE_NOT_SNAPSHOTTED",
                    "price_precision_check": "INCONCLUSIVE_NOT_SNAPSHOTTED",
                    "participation_rate_p50": p50,
                    "participation_rate_p75": p75,
                    "participation_rate_p90": p90,
                    "participation_rate_max": float(participation.max()),
                    "within_production_5pct_hourly_volume_rate": within_limit,
                    "taker_probability": 1.0,
                    "taker_probability_basis": "all snapshot V5 orders are market orders; conservative assumption",
                    "partial_fill_risk": liquidity_risk,
                    "liquidity_insufficiency_risk": liquidity_risk,
                    "delayed_execution_risk": liquidity_risk,
                    "base_one_way_cost_bps": float(base_cost["one_way_cost_bps"]),
                    "uncalibrated_size_impact_bps": additional_impact_bps,
                    "diagnostic_one_way_cost_bps": float(base_cost["one_way_cost_bps"])
                    + additional_impact_bps,
                    "actual_fill_notional_median_usdt": actual_median,
                    "actual_fill_notional_p90_usdt": actual_p90,
                    "capacity_decision": "INCONCLUSIVE",
                    "decision_reason": "instrument precision and size-dependent slippage/partial-fill evidence are not available across the historical universe",
                }
            )
    frame = pl.DataFrame(rows)
    frame.write_csv(ARTIFACTS / "capacity_results.csv")
    lines = [
        "# Capacity Report",
        "",
        f"- generated_at: {datetime.now(UTC).isoformat()}",
        f"- actual fill notionals: n={len(actual_notionals)}, median={actual_median}, p90={actual_p90} USDT.",
        "- Production minimum trade value is 10 USDT and max hourly volume participation is 5%.",
        "- All snapshot orders are market orders, so taker probability is conservatively set to 1.0.",
        "- Quantity/price precision snapshots and partial-fill/latency curves are missing for the broad historical universe.",
        "- Square-root size impact is diagnostic and uncalibrated; it cannot award a PASS.",
        "- Formal capacity verdict for every requested size: INCONCLUSIVE.",
        "",
        "| Universe | Notional | p50 participation | p90 participation | <=5% coverage | Liquidity risk | Diagnostic one-way bps | Verdict |",
        "|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for row in frame.iter_rows(named=True):
        lines.append(
            f"| {row['universe']} | {row['notional_usdt']:.0f} | {row['participation_rate_p50']:.6f} | "
            f"{row['participation_rate_p90']:.6f} | {row['within_production_5pct_hourly_volume_rate']:.3f} | "
            f"{row['liquidity_insufficiency_risk']} | {row['diagnostic_one_way_cost_bps']:.2f} | {row['capacity_decision']} |"
        )
    (REPORTS / "capacity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"capacity rows={frame.height}")


if __name__ == "__main__":
    main()
