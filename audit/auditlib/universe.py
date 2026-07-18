"""Dynamic liquidity universe (audit spec section 7).

At each decision date the universe contains the top-N symbols by trailing
30-day mean daily quote volume, restricted to symbols that:
- have bar data at that date (listed long enough: >= min_age_days)
- meet minimum activity (mean daily quote volume >= min_quote_volume)
- have acceptable recent data completeness (<= max_missing_rate over the
  trailing window)

No forward-looking inputs: the ranking at date D only uses bars with
ts < D. Limitations (recorded in outputs):
- the backfill candidate set itself was selected from currently-listed
  symbols, which introduces a survivorship tilt for coins that died before
  the window; this bias is documented, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import polars as pl


@dataclass(frozen=True)
class UniverseSpec:
    name: str
    top_n: int
    min_age_days: int = 30
    min_daily_quote_volume: float = 1_000_000.0
    lookback_days: int = 30
    max_missing_rate: float = 0.20


UNIVERSES = {
    "top10": UniverseSpec("top10", 10),
    "top20": UniverseSpec("top20", 20),
    "top50": UniverseSpec("top50", 50, min_daily_quote_volume=500_000.0),
}


def build_daily_universe(bars: pl.DataFrame, spec: UniverseSpec) -> pl.DataFrame:
    """Return [date, symbol, rank, mean_qv_30d, age_days, missing_rate].

    bars: [symbol, ts(datetime UTC), quote_volume, close]
    """
    daily = (
        bars.with_columns(pl.col("ts").dt.date().alias("date"))
        .group_by(["symbol", "date"])
        .agg(
            [
                pl.col("quote_volume").sum().alias("qv"),
                pl.len().alias("bars"),
            ]
        )
        .sort(["symbol", "date"])
    )
    candidates: list[dict] = []
    for symbol_key, frame in daily.group_by("symbol", maintain_order=True):
        symbol = symbol_key[0] if isinstance(symbol_key, tuple) else symbol_key
        rows = frame.sort("date").to_dicts()
        if not rows:
            continue
        listing_date = rows[0]["date"]
        for row in rows:
            decision_date = row["date"]
            window_start = decision_date - timedelta(days=spec.lookback_days)
            # Strictly causal: the decision-date volume is never part of its own rank.
            history = [r for r in rows if window_start <= r["date"] < decision_date]
            if not history:
                continue
            age_days = (decision_date - listing_date).days
            bars_in_window = sum(int(r["bars"]) for r in history)
            missing_rate = max(0.0, 1.0 - bars_in_window / (24.0 * spec.lookback_days))
            mean_qv = sum(float(r["qv"]) for r in history) / len(history)
            if (
                age_days >= spec.min_age_days
                and mean_qv >= spec.min_daily_quote_volume
                and missing_rate <= spec.max_missing_rate
            ):
                candidates.append(
                    {
                        "date": decision_date,
                        "symbol": str(symbol),
                        "mean_qv_30d": mean_qv,
                        "age_days": age_days,
                        "missing_rate": missing_rate,
                    }
                )

    if not candidates:
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "symbol": pl.Utf8,
                "rank": pl.UInt32,
                "mean_qv_30d": pl.Float64,
                "age_days": pl.Int64,
                "missing_rate": pl.Float64,
            }
        )
    return (
        pl.DataFrame(candidates)
        .sort(["date", "mean_qv_30d", "symbol"], descending=[False, True, False])
        .with_columns(
            pl.col("mean_qv_30d").rank("ordinal", descending=True).over("date").alias("rank")
        )
        .filter(pl.col("rank") <= spec.top_n)
        .select(["date", "symbol", "rank", "mean_qv_30d", "age_days", "missing_rate"])
        .sort(["date", "rank", "symbol"])
    )
