# ruff: noqa: E501
"""Build causal universes and frozen factor signal partitions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.factors import (  # noqa: E402
    funding_fade,
    low_vol_480,
    rev_xs_480,
    v5_alpha6_static_signal,
    v5_raw_factors,
)
from audit.auditlib.paths import ARTIFACTS, DATA_GOLD, DATA_SILVER  # noqa: E402
from audit.auditlib.signal_chain import build_current_signal_chain  # noqa: E402
from audit.auditlib.universe import UNIVERSES, build_daily_universe  # noqa: E402


def main() -> None:
    bars = pl.read_parquet(DATA_SILVER / "bars_1h.parquet")
    funding = pl.read_parquet(DATA_SILVER / "funding.parquet")
    DATA_GOLD.mkdir(parents=True, exist_ok=True)
    signal_dir = DATA_GOLD / "signals"
    signal_dir.mkdir(parents=True, exist_ok=True)

    universe_frames = []
    for name, spec in UNIVERSES.items():
        frame = build_daily_universe(bars, spec).with_columns(pl.lit(name).alias("universe"))
        universe_frames.append(frame)
        print(f"universe {name}: {frame.height:,} memberships over {frame['date'].n_unique()} days")
    universes = pl.concat(universe_frames).sort(["date", "universe", "rank"])
    universes.write_parquet(DATA_GOLD / "dynamic_universe.parquet", compression="zstd")

    raw = v5_raw_factors(bars)
    raw.write_parquet(DATA_GOLD / "v5_raw_factors.parquet", compression="zstd")
    signals = {
        "v5.alpha6_static_proxy": v5_alpha6_static_signal(raw).select(
            ["symbol", "feature_ts", "signal"]
        ),
        "core.rev_xs_480": rev_xs_480(bars),
        "core.low_vol_480": low_vol_480(bars),
        "core.funding_fade_480": funding_fade(bars.select(["symbol", "ts"]), funding).filter(
            pl.col("signal").is_not_null()
        ),
    }
    manifest = []
    for factor_id, frame in signals.items():
        path = signal_dir / f"{factor_id.replace('.', '__')}.parquet"
        frame.sort(["symbol", "feature_ts"]).write_parquet(path, compression="zstd")
        manifest.append(
            {
                "factor_id": factor_id,
                "rows": frame.height,
                "symbols": frame["symbol"].n_unique() if frame.height else 0,
                "first_feature_ts": str(frame["feature_ts"].min()) if frame.height else None,
                "last_feature_ts": str(frame["feature_ts"].max()) if frame.height else None,
                "path": str(path),
            }
        )
        print(f"signal {factor_id}: {frame.height:,} rows")
    (ARTIFACTS / "feature_manifest.json").write_text(
        json.dumps(
            {
                "universes": {
                    name: {
                        "top_n": spec.top_n,
                        "lookback_days": spec.lookback_days,
                        "min_age_days": spec.min_age_days,
                        "min_daily_quote_volume": spec.min_daily_quote_volume,
                        "max_missing_rate": spec.max_missing_rate,
                        "causal_rule": "decision date D uses only dates D-lookback <= d < D",
                    }
                    for name, spec in UNIVERSES.items()
                },
                "survivorship_limitation": "The seed set contains current-listed instruments; delisted-before-snapshot instruments cannot be reconstructed.",
                "signals": manifest,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    build_current_signal_chain()


if __name__ == "__main__":
    main()
