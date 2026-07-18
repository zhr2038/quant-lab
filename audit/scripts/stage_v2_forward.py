# ruff: noqa: E501
"""Forward-only runner for the single frozen low_vol + BTC trend hypothesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.factors import low_vol_480
from audit.auditlib.forward_paper import (
    FORWARD_CONFIG,
    upsert_by_decision_timestamp,
    validate_forward_timestamps,
)
from audit.auditlib.portfolio_backtest import _capped_weights
from audit.auditlib.universe import UNIVERSES, build_daily_universe

OKX_HISTORY = "https://www.okx.com/api/v5/market/history-candles"
DEFAULT_CUTOFF = datetime(2026, 7, 17, 23, tzinfo=UTC)


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.astimezone(UTC)


def _fetch_symbol(symbol: str, cutoff: datetime, as_of: datetime) -> list[dict]:
    rows: list[dict] = []
    cursor = int(as_of.timestamp() * 1000)
    cutoff_ms = int(cutoff.timestamp() * 1000)
    for _attempt_page in range(20):
        params = {"instId": symbol, "bar": "1H", "limit": "100", "after": str(cursor)}
        response = None
        for retry in range(5):
            try:
                response = requests.get(OKX_HISTORY, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
                if str(payload.get("code")) != "0":
                    raise RuntimeError(f"OKX {payload.get('code')}: {payload.get('msg')}")
                break
            except Exception:
                if retry == 4:
                    raise
                time.sleep(2**retry)
        assert response is not None
        data = response.json().get("data") or []
        if not data:
            break
        oldest = min(int(item[0]) for item in data)
        for item in data:
            ts_ms = int(item[0])
            if cutoff_ms < ts_ms <= int(as_of.timestamp() * 1000) and str(item[8]) == "1":
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": datetime.fromtimestamp(ts_ms / 1000, tz=UTC),
                        "open": float(item[1]),
                        "high": float(item[2]),
                        "low": float(item[3]),
                        "close": float(item[4]),
                        "volume": float(item[5]),
                        "quote_volume": float(item[7]),
                    }
                )
        if oldest <= cutoff_ms or len(data) < 100:
            break
        cursor = oldest - 1
        time.sleep(0.13)
    return rows


def _load_or_fetch(root: Path, v1_root: Path, cutoff: datetime, as_of: datetime) -> pl.DataFrame:
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    cache = state / "forward_market_bars.parquet"
    existing = pl.read_parquet(cache) if cache.exists() else pl.DataFrame()
    base = pl.scan_parquet(v1_root / "data/silver/bars_1h.parquet").select("symbol").unique().collect()
    rows: list[dict] = []
    for index, symbol in enumerate(base["symbol"].sort().to_list(), start=1):
        rows.extend(_fetch_symbol(symbol, cutoff, as_of))
        if index % 20 == 0:
            print(f"forward public candles: {index}/{base.height} symbols")
        time.sleep(0.13)
    fresh = pl.DataFrame(rows) if rows else pl.DataFrame()
    frames = [frame for frame in (existing, fresh) if not frame.is_empty()]
    if not frames:
        return pl.DataFrame(
            schema={
                "symbol": pl.Utf8,
                "ts": pl.Datetime("us", "UTC"),
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "quote_volume": pl.Float64,
            }
        )
    combined = pl.concat(frames, how="vertical_relaxed").unique(
        subset=["symbol", "ts"], keep="last"
    ).sort(["symbol", "ts"])
    combined.write_parquet(cache, compression="zstd")
    return combined


def _report(root: Path, cutoff: datetime, as_of: datetime) -> None:
    decisions_path = root / "artifacts/forward_paper_decisions.parquet"
    performance_path = root / "artifacts/forward_paper_performance.csv"
    decisions = pl.read_parquet(decisions_path) if decisions_path.exists() else pl.DataFrame()
    performance = pl.read_csv(performance_path) if performance_path.exists() else pl.DataFrame()
    completed = (
        int(performance["completed_independent_trades"].max())
        if performance.height and "completed_independent_trades" in performance.columns
        else 0
    )
    latest_data = (
        str(decisions["available_data_cutoff"].max()) if decisions.height else "none"
    )
    sample_days = max(0.0, (as_of - cutoff).total_seconds() / 86400.0)
    status = (
        "INCONCLUSIVE: forward sample insufficient"
        if sample_days < 14 or completed < 3
        else "INCONCLUSIVE: forward observation complete but no PASS is granted automatically"
    )
    lines = [
        "# Low-vol + BTC Trend Forward Paper",
        "",
        f"- Status: **{status}**",
        "- Hypothesis: **POST_HOC_HYPOTHESIS / FORWARD ONLY**.",
        f"- Strict v1 cutoff: `{cutoff.isoformat()}`; requested as-of: `{as_of.isoformat()}`.",
        f"- Available forward data cutoff: `{latest_data}`.",
        f"- Elapsed forward calendar days: {sample_days:.3f}; recorded decisions: {decisions.height}; completed independent 120h trades: {completed}.",
        "- Frozen parameters: low_vol_20d, v1 dynamic Top20, Top3, score weights, 120h, BTC 60d trend ON, 15bps one-way cost.",
        "- Old blind/OOS data is not counted as forward evidence. No order is submitted; all fills are theoretical paper fills.",
        "",
        "## Reproduction",
        "",
        f"`./scripts/run_low_vol_forward_paper.sh --start-after {cutoff.isoformat()} --as-of {as_of.isoformat()} --resume`",
        "",
        "Artifacts: `forward_paper_decisions.parquet`, `forward_paper_positions.parquet`, and `forward_paper_performance.csv`.",
    ]
    (root / "reports/forward_paper_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-after", default=DEFAULT_CUTOFF.isoformat())
    parser.add_argument("--as-of", default=datetime.now(UTC).replace(microsecond=0).isoformat())
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    cutoff, as_of = _utc(args.start_after), _utc(args.as_of)
    if as_of <= cutoff:
        raise SystemExit("--as-of must be strictly later than --start-after")
    root = Path(os.environ.get("AUDIT_ROOT", "/home/hr/quant-alpha-audit-v2"))
    v1_root = Path(os.environ.get("AUDIT_V1_ROOT", "/home/hr/quant-alpha-audit"))
    (root / "artifacts").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    if args.report_only:
        _report(root, cutoff, as_of)
        return
    decision_path = root / "artifacts/forward_paper_decisions.parquet"
    if decision_path.exists() and not args.resume and not args.dry_run:
        raise SystemExit("forward state exists; use --resume or --report-only")
    forward = _load_or_fetch(root, v1_root, cutoff, as_of)
    if forward.is_empty():
        print("No completed public candles strictly after v1 cutoff.")
        if not args.dry_run:
            pl.DataFrame(
                schema={"decision_timestamp": pl.Datetime("us", "UTC")}
            ).write_parquet(decision_path)
            pl.DataFrame(
                schema={"decision_timestamp": pl.Datetime("us", "UTC"), "symbol": pl.Utf8}
            ).write_parquet(root / "artifacts/forward_paper_positions.parquet")
            pl.DataFrame(
                [
                    {
                        "as_of": as_of.isoformat(),
                        "forward_calendar_days": (as_of - cutoff).total_seconds() / 86400.0,
                        "decision_count": 0,
                        "completed_independent_trades": 0,
                        "portfolio_net_return": 0.0,
                        "benchmark_net_return": 0.0,
                        "conclusion": "INCONCLUSIVE_FORWARD_SAMPLE_INSUFFICIENT",
                    }
                ]
            ).write_csv(root / "artifacts/forward_paper_performance.csv")
            _report(root, cutoff, as_of)
        return
    history = pl.read_parquet(v1_root / "data/silver/bars_1h.parquet")
    combined = pl.concat([history, forward], how="vertical_relaxed").unique(
        subset=["symbol", "ts"], keep="last"
    ).sort(["symbol", "ts"])
    validate_forward_timestamps(forward["ts"], cutoff)
    signals = low_vol_480(combined)
    universe = build_daily_universe(combined, UNIVERSES["top20"])
    btc = (
        combined.filter(pl.col("symbol") == "BTC-USDT")
        .sort("ts")
        .with_columns(pl.col("close").rolling_mean(1440).alias("btc_ma_60d"))
        .select(["ts", "close", "btc_ma_60d"])
    )
    all_ts = forward["ts"].unique().sort().to_list()
    first_decision = min(all_ts)
    decision_times = [
        ts
        for ts in all_ts
        if ts > cutoff
        and ts <= as_of
        and int((ts - first_decision).total_seconds() / 3600) % 120 == 0
    ]
    decision_rows: list[dict] = []
    position_rows: list[dict] = []
    latest_ts = forward["ts"].max()
    config_digest = hashlib.sha256(
        json.dumps(FORWARD_CONFIG, sort_keys=True).encode("utf-8")
    ).hexdigest()
    for decision_ts in decision_times:
        feature_ts = decision_ts - timedelta(hours=1)
        membership = universe.filter(pl.col("date") == decision_ts.date()).sort("rank")
        values = (
            signals.filter(pl.col("feature_ts") == feature_ts)
            .join(membership.select(["symbol", "rank"]), on="symbol", how="inner")
            .sort(["signal", "symbol"], descending=[True, False])
        )
        btc_row = btc.filter(pl.col("ts") == feature_ts)
        btc_up = bool(
            btc_row.height
            and btc_row["btc_ma_60d"][0] is not None
            and btc_row["close"][0] >= btc_row["btc_ma_60d"][0]
        )
        target = values.head(3) if btc_up and values.height >= 3 else values.head(0)
        weights = (
            _capped_weights(target["signal"].to_numpy(), 3, 0.50, "score")
            if target.height == 3
            else np.asarray([], dtype=float)
        )
        decision_prices = combined.filter(pl.col("ts") == decision_ts).select(
            ["symbol", pl.col("close").alias("decision_price")]
        )
        latest_prices = combined.filter(pl.col("ts") == latest_ts).select(
            ["symbol", pl.col("close").alias("latest_price")]
        )
        selected = (
            target.with_columns(pl.Series("target_weight", weights))
            .join(decision_prices, on="symbol", how="left")
            .join(latest_prices, on="symbol", how="left")
            .with_columns(
                (
                    pl.col("target_weight")
                    * (pl.col("latest_price") / pl.col("decision_price") - 1.0)
                ).alias("gross_contribution")
            )
        )
        for row in selected.iter_rows(named=True):
            position_rows.append(
                {
                    "decision_timestamp": decision_ts,
                    "available_data_cutoff": latest_ts,
                    **row,
                    "paper_only": True,
                    "live_order_effect": "none",
                }
            )
        gross_mark = float(selected["gross_contribution"].sum()) if selected.height else 0.0
        one_way_cost = (1.0 if selected.height else 0.0) * 15.0 / 10_000.0
        decision_rows.append(
            {
                "decision_timestamp": decision_ts,
                "available_data_cutoff": latest_ts,
                "feature_timestamp": feature_ts,
                "btc_trend_up": btc_up,
                "universe_json": json.dumps(membership["symbol"].to_list()),
                "factor_values_json": json.dumps(values.select(["symbol", "signal"]).to_dicts()),
                "ranking_json": json.dumps(values["symbol"].to_list()),
                "target_positions_json": json.dumps(selected.select(["symbol", "target_weight"]).to_dicts()),
                "theoretical_orders_json": json.dumps(selected.select(["symbol", "target_weight", "decision_price"]).to_dicts()),
                "simulated_fills_json": json.dumps(selected.select(["symbol", "decision_price"]).to_dicts()),
                "fees_return": 0.001 if selected.height else 0.0,
                "slippage_return": 0.0005 if selected.height else 0.0,
                "marked_gross_return": gross_mark,
                "marked_net_return": gross_mark - one_way_cost,
                "benchmark_net_return": 0.0,
                "data_missing_json": json.dumps([] if values.height >= 3 else ["insufficient ranked universe"]),
                "data_latency_seconds": max(0.0, (as_of - latest_ts).total_seconds()),
                "anomalies_json": json.dumps([]),
                "hypothesis_type": "POST_HOC_HYPOTHESIS",
                "config_digest": config_digest,
                "paper_only": True,
                "live_order_effect": "none",
            }
        )
    new_decisions = pl.DataFrame(decision_rows) if decision_rows else pl.DataFrame(
        schema={"decision_timestamp": pl.Datetime("us", "UTC")}
    )
    new_positions = pl.DataFrame(position_rows) if position_rows else pl.DataFrame(
        schema={"decision_timestamp": pl.Datetime("us", "UTC"), "symbol": pl.Utf8}
    )
    if args.dry_run:
        print(f"dry-run decisions={new_decisions.height} positions={new_positions.height}")
        return
    existing_decisions = pl.read_parquet(decision_path) if decision_path.exists() else pl.DataFrame()
    decisions = upsert_by_decision_timestamp(existing_decisions, new_decisions)
    decisions.write_parquet(decision_path, compression="zstd")
    position_path = root / "artifacts/forward_paper_positions.parquet"
    existing_positions = pl.read_parquet(position_path) if position_path.exists() else pl.DataFrame()
    positions = pl.concat([existing_positions, new_positions], how="diagonal_relaxed").unique(
        subset=["decision_timestamp", "symbol"], keep="last"
    ) if not existing_positions.is_empty() and not new_positions.is_empty() else (
        new_positions if existing_positions.is_empty() else existing_positions
    )
    positions.sort(["decision_timestamp", "symbol"]).write_parquet(position_path, compression="zstd")
    completed = sum(
        1 for ts in decisions["decision_timestamp"].to_list() if latest_ts >= ts + timedelta(hours=120)
    )
    pl.DataFrame(
        [
            {
                "as_of": as_of.isoformat(),
                "available_data_cutoff": latest_ts.isoformat(),
                "forward_calendar_days": (latest_ts - cutoff).total_seconds() / 86400.0,
                "decision_count": decisions.height,
                "completed_independent_trades": completed,
                "portfolio_net_return": float(decisions["marked_net_return"].sum()) if "marked_net_return" in decisions.columns else 0.0,
                "benchmark_net_return": float(decisions["benchmark_net_return"].sum()) if "benchmark_net_return" in decisions.columns else 0.0,
                "conclusion": "INCONCLUSIVE_FORWARD_SAMPLE_INSUFFICIENT"
                if (latest_ts - cutoff) < timedelta(days=14) or completed < 3
                else "INCONCLUSIVE_NO_AUTOMATIC_PASS",
            }
        ]
    ).write_csv(root / "artifacts/forward_paper_performance.csv")
    _report(root, cutoff, as_of)
    print(f"forward decisions={decisions.height} positions={positions.height} completed={completed}")


if __name__ == "__main__":
    main()
