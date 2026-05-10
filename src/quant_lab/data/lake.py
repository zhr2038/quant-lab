from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.contracts.models import MarketBar, require_utc

MARKET_BAR_DATASET = Path("silver") / "market_bar"
MARKET_BAR_PRIMARY_KEY = ["venue", "symbol", "timeframe", "ts"]
MARKET_BAR_SCHEMA = {
    "venue": pl.Utf8,
    "symbol": pl.Utf8,
    "market_type": pl.Utf8,
    "timeframe": pl.Utf8,
    "ts": pl.Datetime(time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "quote_volume": pl.Float64,
    "source": pl.Utf8,
    "ingest_ts": pl.Datetime(time_zone="UTC"),
    "is_closed": pl.Boolean,
}


class LakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lake_root: Path
    created_by: str = Field(default="quant-lab", min_length=1)


def write_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    partition_by: str | Sequence[str] | None = None,
) -> Path:
    path = Path(dataset_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sorted_df = _sort_dataframe(df)
    _remove_existing_dataset(path)

    if partition_by:
        path.mkdir(parents=True, exist_ok=True)
        sorted_df.write_parquet(path, partition_by=partition_by, mkdir=True)
        return path

    path.mkdir(parents=True, exist_ok=True)
    sorted_df.write_parquet(path / "data.parquet")
    return path


def read_parquet_dataset(dataset_path: str | Path) -> pl.DataFrame:
    files = _parquet_files(dataset_path)
    if not files:
        return pl.DataFrame()
    return pl.read_parquet([str(path) for path in files])


def upsert_parquet_dataset(
    df: pl.DataFrame,
    dataset_path: str | Path,
    key_columns: Sequence[str],
) -> int:
    existing_df = read_parquet_dataset(dataset_path)
    frames = [frame for frame in [existing_df, df] if not frame.is_empty()]
    combined = pl.concat(frames, how="diagonal_relaxed") if frames else df
    if not combined.is_empty():
        available_keys = [column for column in key_columns if column in combined.columns]
        if available_keys:
            combined = combined.unique(subset=available_keys, keep="last", maintain_order=True)
    write_parquet_dataset(combined, dataset_path)
    return combined.height


def validate_market_bars(records: Sequence[MarketBar | dict]) -> list[MarketBar]:
    validated = [
        record if isinstance(record, MarketBar) else MarketBar(**record) for record in records
    ]
    seen_keys: set[tuple[str, str, str, datetime]] = set()
    duplicate_keys: list[tuple[str, str, str, datetime]] = []

    for record in validated:
        key = (record.venue, record.symbol, record.timeframe, record.ts)
        if key in seen_keys:
            duplicate_keys.append(key)
        seen_keys.add(key)

    if duplicate_keys:
        rendered = ", ".join(
            f"{venue}/{symbol}/{timeframe}/{ts.isoformat()}"
            for venue, symbol, timeframe, ts in duplicate_keys
        )
        raise ValueError(f"duplicate market_bar primary key: {rendered}")

    return validated


def market_bars_to_polars(records: Sequence[MarketBar | dict]) -> pl.DataFrame:
    validated = validate_market_bars(records)
    rows = [record.model_dump() for record in validated]
    return pl.DataFrame(rows, schema=MARKET_BAR_SCHEMA, orient="row")


def write_market_bars(lake_root: str | Path, records: Sequence[MarketBar | dict]) -> int:
    dataset_path = Path(lake_root) / MARKET_BAR_DATASET
    new_df = market_bars_to_polars(records)
    if new_df.is_empty():
        write_parquet_dataset(new_df, dataset_path)
        return 0
    return upsert_parquet_dataset(new_df, dataset_path, key_columns=MARKET_BAR_PRIMARY_KEY)


def read_market_bars(
    lake_root: str | Path,
    venue: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[MarketBar]:
    start_utc = require_utc(start)
    end_utc = require_utc(end)
    if end_utc < start_utc:
        raise ValueError("end must be greater than or equal to start")

    dataset_path = Path(lake_root) / MARKET_BAR_DATASET
    df = read_parquet_dataset(dataset_path)
    if df.is_empty():
        return []

    normalized = _normalize_market_bar_frame(df)
    filtered = (
        normalized.filter(
            (pl.col("venue") == venue)
            & (pl.col("symbol") == symbol)
            & (pl.col("timeframe") == timeframe)
            & (pl.col("ts") >= start_utc)
            & (pl.col("ts") <= end_utc)
        )
        .sort("ts")
        .select(list(MARKET_BAR_SCHEMA))
    )
    return validate_market_bars(filtered.to_dicts())


def read_parquet_lazy(path: str | Path) -> pl.LazyFrame:
    files = _parquet_files(path)
    if files:
        return pl.scan_parquet([str(file_path) for file_path in files])
    return pl.scan_parquet(str(path))


def scan_parquet_with_duckdb(path: str | Path) -> duckdb.DuckDBPyRelation:
    files = _parquet_files(path)
    if not files:
        raise FileNotFoundError(f"No Parquet files found under dataset path: {path}")
    connection = duckdb.connect(database=":memory:", read_only=False)
    return connection.read_parquet([str(file) for file in files])


def query_dataset_sql(lake_root: str | Path, dataset_name: str, sql: str) -> pl.DataFrame:
    dataset_path = Path(lake_root) / dataset_name
    files = _parquet_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No Parquet files found for dataset: {dataset_name}")

    query = sql.strip()
    if ";" in query.rstrip(";"):
        raise ValueError("query_dataset_sql accepts a single read-only SELECT statement")
    query = query.rstrip(";")
    if not query.lower().startswith("select"):
        raise ValueError("query_dataset_sql only accepts read-only SELECT statements")

    connection = duckdb.connect(database=":memory:", read_only=False)
    connection.read_parquet([str(path) for path in files]).create_view("dataset")
    return pl.from_arrow(connection.execute(query).to_arrow_table())


def _parquet_files(dataset_path: str | Path) -> list[Path]:
    path = Path(dataset_path)
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if not path.exists():
        return []
    return sorted(path.rglob("*.parquet"))


def _sort_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    preferred = [
        "source_path",
        "run_id",
        "venue",
        "symbol",
        "timeframe",
        "ts",
        "cost_day",
        "bucket_index",
        "dataset",
        "ingest_ts",
    ]
    sort_columns = [column for column in preferred if column in df.columns]
    if not sort_columns:
        sort_columns = sorted(df.columns)
    try:
        return df.sort(sort_columns)
    except Exception:
        return df


def _normalize_market_bar_frame(df: pl.DataFrame) -> pl.DataFrame:
    normalized = df
    if "quote_volume" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(None, dtype=pl.Float64).alias("quote_volume"))
    if "is_closed" not in normalized.columns:
        normalized = normalized.with_columns(pl.lit(True).alias("is_closed"))

    return normalized.with_columns(
        [
            _datetime_column(normalized, "ts"),
            _datetime_column(normalized, "ingest_ts"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.col("quote_volume").cast(pl.Float64),
            pl.col("is_closed").cast(pl.Boolean),
        ]
    )


def _datetime_column(df: pl.DataFrame, column: str) -> pl.Expr:
    expression = pl.col(column)
    if df.schema.get(column) == pl.String:
        return expression.str.to_datetime(time_zone="UTC", strict=False).alias(column)
    return expression.cast(pl.Datetime(time_zone="UTC")).alias(column)


def _remove_existing_dataset(path: Path) -> None:
    if path.is_file():
        path.unlink()
        return
    if path.is_dir():
        for parquet_file in path.rglob("*.parquet"):
            parquet_file.unlink()
        for child in sorted(path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
