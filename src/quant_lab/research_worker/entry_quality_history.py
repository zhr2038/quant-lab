from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as datetime_time
from pathlib import Path

import polars as pl

from quant_lab.research.entry_quality import (
    COST_BUCKET_DAILY_DATASET,
    ENTRY_QUALITY_SYMBOLS,
    MARKET_BAR_DATASET,
    PULLBACK_HORIZON_HOURS,
    V5_CANDIDATE_EVENT_DATASET,
    V5_CANDIDATE_LABEL_DATASET,
    V5_ORDER_LIFECYCLE_DATASET,
    V5_TRADE_EVENT_DATASET,
    EntryQualityHistoryArtifacts,
    compute_entry_quality_history,
)
from quant_lab.research_plane.contracts import ResearchSnapshotManifest, ResearchTask
from quant_lab.symbols import normalize_symbol

DATASET_COLUMNS = {
    str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): (
        "run_id",
        "trade_id",
        "order_id",
        "lifecycle_id",
        "source_event_key",
        "ts_utc",
        "ts",
        "entry_ts",
        "normalized_symbol",
        "symbol",
        "side",
        "action",
        "intent",
        "final_decision",
        "price",
        "fill_px",
        "fill_price",
        "entry_reason",
        "probe_type",
        "exit_reason",
        "realized_net_bps",
        "net_bps",
        "pnl_bps",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
    ),
    str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (
        "run_id",
        "trade_id",
        "order_id",
        "lifecycle_id",
        "source_event_key",
        "ts_utc",
        "last_fill_ts",
        "submit_ts",
        "decision_ts",
        "normalized_symbol",
        "symbol",
        "side",
        "action",
        "intent",
        "final_decision",
        "avg_fill_px",
        "fill_px",
        "entry_px",
        "entry_reason",
        "probe_type",
        "exit_reason",
        "realized_net_bps",
        "net_bps",
        "total_realized_cost_bps",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
    ),
    str(MARKET_BAR_DATASET).replace("\\", "/"): (
        "symbol",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
    ),
    str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): (
        "candidate_id",
        "run_id",
        "source_event_key",
        "ts_utc",
        "ts",
        "symbol",
        "normalized_symbol",
        "strategy_candidate",
        "entry_close",
        "candidate_px",
        "current_px",
        "price",
        "close",
        "last_price",
        "regime_state",
        "risk_level",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "estimated_spread_bps",
        "spread_bps",
    ),
    str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
        "candidate_id",
        "horizon_hours",
        "net_bps_after_cost",
        "decision_ts",
        "label_ts",
        "label_end_ts",
        "label_status",
    ),
    str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): (
        "symbol",
        "as_of_date",
        "roundtrip_all_in_cost_bps",
        "one_way_all_in_cost_bps",
        "total_cost_bps_p75",
        "selected_total_cost_bps",
    ),
}

TIME_COLUMNS = {
    str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): ("ts_utc", "ts", "entry_ts"),
    str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (
        "ts_utc",
        "last_fill_ts",
        "submit_ts",
        "decision_ts",
    ),
    str(MARKET_BAR_DATASET).replace("\\", "/"): ("ts",),
    str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): ("ts_utc", "ts"),
    str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
        "decision_ts",
        "label_ts",
        "label_end_ts",
    ),
    str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): ("as_of_date",),
}


@dataclass(frozen=True)
class EntryQualityWorkerComputeResult:
    artifacts: EntryQualityHistoryArtifacts
    input_rows: dict[str, int]
    stage_seconds: dict[str, float]


def compute_entry_quality_history_from_snapshot(
    snapshot_root: str | Path,
    manifest: ResearchSnapshotManifest,
    task: ResearchTask,
) -> EntryQualityWorkerComputeResult:
    root = Path(snapshot_root)
    start_dt = datetime.combine(task.parameters.start_date, datetime_time.min, tzinfo=UTC)
    end_dt = datetime.combine(
        task.parameters.end_date + timedelta(days=1), datetime_time.min, tzinfo=UTC
    )
    ranges = {
        str(V5_TRADE_EVENT_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/"): (start_dt, end_dt),
        str(MARKET_BAR_DATASET).replace("\\", "/"): (
            start_dt - timedelta(hours=24),
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        ),
        str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/"): (
            start_dt,
            end_dt + timedelta(hours=max(PULLBACK_HORIZON_HOURS)),
        ),
        str(COST_BUCKET_DAILY_DATASET).replace("\\", "/"): (start_dt, end_dt),
    }
    paths_by_dataset: dict[str, list[Path]] = {name: [] for name in manifest.datasets}
    for reference in manifest.files:
        paths_by_dataset.setdefault(reference.dataset_name, []).append(
            root / "files" / reference.relative_path
        )
    frames: dict[str, pl.DataFrame] = {}
    stage_seconds: dict[str, float] = {}
    input_rows: dict[str, int] = {}
    market_dataset = str(MARKET_BAR_DATASET).replace("\\", "/")
    ordered_datasets = [dataset for dataset in manifest.datasets if dataset != market_dataset]
    for dataset in ordered_datasets:
        started = time.perf_counter()
        frame = _scan_projected_dataset(
            paths_by_dataset.get(dataset, []),
            columns=DATASET_COLUMNS[dataset],
            time_columns=TIME_COLUMNS[dataset],
            start_dt=ranges[dataset][0],
            end_dt=ranges[dataset][1],
        )
        frames[dataset] = frame
        input_rows[dataset] = frame.height
        stage_seconds[f"load:{dataset}"] = time.perf_counter() - started

    market_symbols = _required_market_symbols(frames)
    started = time.perf_counter()
    market = _scan_projected_dataset(
        paths_by_dataset.get(market_dataset, []),
        columns=DATASET_COLUMNS[market_dataset],
        time_columns=TIME_COLUMNS[market_dataset],
        start_dt=ranges[market_dataset][0],
        end_dt=ranges[market_dataset][1],
        symbols=market_symbols,
        timeframe="1H",
    )
    frames[market_dataset] = market
    input_rows[market_dataset] = market.height
    stage_seconds[f"load:{market_dataset}"] = time.perf_counter() - started

    candidate_dataset = str(V5_CANDIDATE_EVENT_DATASET).replace("\\", "/")
    label_dataset = str(V5_CANDIDATE_LABEL_DATASET).replace("\\", "/")
    candidates = frames[candidate_dataset]
    labels = frames[label_dataset]
    if (
        not candidates.is_empty()
        and not labels.is_empty()
        and "candidate_id" in candidates.columns
        and "candidate_id" in labels.columns
    ):
        candidate_ids = candidates.get_column("candidate_id").drop_nulls().unique().to_list()
        labels = labels.filter(pl.col("candidate_id").is_in(candidate_ids))
        frames[label_dataset] = labels
        input_rows[label_dataset] = labels.height

    compute_started = time.perf_counter()
    artifacts = compute_entry_quality_history(
        trades=frames[str(V5_TRADE_EVENT_DATASET).replace("\\", "/")],
        lifecycles=frames[str(V5_ORDER_LIFECYCLE_DATASET).replace("\\", "/")],
        market_bars=frames[str(MARKET_BAR_DATASET).replace("\\", "/")],
        candidates=candidates,
        labels=labels,
        costs=frames[str(COST_BUCKET_DAILY_DATASET).replace("\\", "/")],
        start_date=task.parameters.start_date,
        end_date=task.parameters.end_date,
        mode=task.parameters.mode,
        cost_mode=task.parameters.cost_mode,
        window_hours=task.parameters.window_hours,
        generated_from_bundle_id=task.selected_v5_bundle_id,
    )
    stage_seconds["compute"] = time.perf_counter() - compute_started
    return EntryQualityWorkerComputeResult(
        artifacts=artifacts,
        input_rows=input_rows,
        stage_seconds=stage_seconds,
    )


def _scan_projected_dataset(
    paths: list[Path],
    *,
    columns: tuple[str, ...],
    time_columns: tuple[str, ...],
    start_dt: datetime,
    end_dt: datetime,
    symbols: set[str] | None = None,
    timeframe: str | None = None,
) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    scans: list[pl.LazyFrame] = []
    for path in sorted(paths):
        schema = pl.read_parquet_schema(path)
        selected = [column for column in columns if column in schema]
        lazy = pl.scan_parquet(path).select(selected)
        time_column = next((column for column in time_columns if column in schema), None)
        if time_column is not None:
            parsed = (
                pl.col(time_column).cast(pl.Utf8).str.to_datetime(time_zone="UTC", strict=False)
            )
            lazy = lazy.filter((parsed >= start_dt) & (parsed < end_dt))
        if timeframe is not None and "timeframe" in schema:
            lazy = lazy.filter(
                pl.col("timeframe").cast(pl.Utf8).str.to_uppercase() == timeframe.upper()
            )
        if symbols and "symbol" in schema:
            lazy = lazy.filter(
                pl.col("symbol")
                .cast(pl.Utf8)
                .str.to_uppercase()
                .is_in(sorted(_symbol_variants(symbols)))
            )
        scans.append(lazy)
    if not scans:
        return pl.DataFrame()
    return pl.concat(scans, how="diagonal_relaxed").collect(engine="streaming")


def _required_market_symbols(frames: dict[str, pl.DataFrame]) -> set[str]:
    symbols = set(ENTRY_QUALITY_SYMBOLS)
    for frame in frames.values():
        if frame.is_empty():
            continue
        for column in ("normalized_symbol", "symbol"):
            if column not in frame.columns:
                continue
            for value in frame.get_column(column).drop_nulls().to_list():
                normalized = normalize_symbol(value)
                if normalized:
                    symbols.add(normalized)
    symbols.add("BTC-USDT")
    return symbols


def _symbol_variants(symbols: set[str]) -> set[str]:
    variants: set[str] = set()
    for symbol in symbols:
        normalized = normalize_symbol(symbol)
        if not normalized:
            continue
        variants.update(
            {
                normalized,
                normalized.replace("-", "/"),
                normalized.replace("-", "_"),
                normalized.replace("-", ""),
            }
        )
    return variants
