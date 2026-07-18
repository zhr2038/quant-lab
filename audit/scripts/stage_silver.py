"""Stage: build unified silver datasets from raw backfill partitions.

Outputs:
- data/silver/bars_1h.parquet      [symbol, ts, open, high, low, close, volume, quote_volume]
- data/silver/funding.parquet      [symbol, funding_ts, funding_rate]
- artifacts/data_quality_summary.csv + reports/data_quality_report.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from audit.auditlib.data_quality import run_data_quality  # noqa: E402
from audit.auditlib.paths import CANDLES_DIR, DATA_SILVER, FUNDING_DIR  # noqa: E402


def main() -> None:
    # candles -> unified bars
    parts = []
    for inst_dir in sorted(CANDLES_DIR.iterdir()):
        if not inst_dir.is_dir():
            continue
        sym = inst_dir.name.replace("inst_id=", "")
        files = sorted(inst_dir.rglob("*.parquet"))
        if not files:
            continue
        df = (
            pl.scan_parquet([str(f) for f in files])
            .filter(pl.col("confirm") == "1")
            .select(
                [
                    pl.lit(sym).alias("symbol"),
                    pl.from_epoch("ts", time_unit="ms").dt.replace_time_zone("UTC").alias("ts"),
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "quote_volume",
                ]
            )
            .collect()
        )
        parts.append(df)
    bars = pl.concat(parts).sort(["symbol", "ts"])
    # safety dedupe
    bars = bars.unique(subset=["symbol", "ts"], keep="last")
    DATA_SILVER.mkdir(parents=True, exist_ok=True)
    bars.write_parquet(DATA_SILVER / "bars_1h.parquet", compression="zstd")
    print(
        f"bars_1h: {bars.height:,} rows, {bars['symbol'].n_unique()} symbols, "
        f"{bars['ts'].min()} .. {bars['ts'].max()}"
    )

    # funding -> unified, mapped to spot symbol names
    fparts = []
    for inst_dir in sorted(FUNDING_DIR.iterdir()):
        if not inst_dir.is_dir():
            continue
        swap = inst_dir.name.replace("inst_id=", "")
        spot = swap.replace("-SWAP", "")
        files = sorted(inst_dir.rglob("*.parquet"))
        if not files:
            continue
        df = (
            pl.scan_parquet([str(f) for f in files])
            .select(
                [
                    pl.lit(spot).alias("symbol"),
                    pl.from_epoch("funding_time", time_unit="ms")
                    .dt.replace_time_zone("UTC")
                    .alias("funding_ts"),
                    "funding_rate",
                ]
            )
            .collect()
        )
        fparts.append(df)
    if fparts:
        funding = (
            pl.concat(fparts)
            .unique(subset=["symbol", "funding_ts"], keep="last")
            .sort(["symbol", "funding_ts"])
        )
    else:
        funding = pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "funding_ts": pl.Datetime("us", "UTC"),
                "funding_rate": pl.Float64,
            }
        )
    funding.write_parquet(DATA_SILVER / "funding.parquet", compression="zstd")
    print(f"funding: {funding.height:,} rows, {funding['symbol'].n_unique()} symbols")

    dq, fdq, execution_dq = run_data_quality()
    print(
        "data_quality_summary written:",
        dq.height,
        "candle rows;",
        0 if fdq.is_empty() else fdq.height,
        "funding rows;",
        execution_dq.height,
        "execution sources",
    )


if __name__ == "__main__":
    main()
