from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset

DATASET_PATHS = {
    "market_bar": Path("silver") / "market_bar",
    "feature_value": Path("gold") / "feature_value",
    "cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "alpha_evidence": Path("gold") / "alpha_evidence",
    "gate_decision": Path("gold") / "gate_decision",
    "risk_permission": Path("gold") / "risk_permission",
    "strategy_health_daily": Path("gold") / "strategy_health_daily",
    "trade_print": Path("silver") / "trade_print",
    "orderbook_snapshot": Path("silver") / "orderbook_snapshot",
    "okx_public_ws": Path("bronze") / "okx_public_ws",
    "okx_private_readonly_fills": Path("bronze") / "okx_private_readonly" / "fills_history",
    "okx_private_readonly_bills": Path("bronze") / "okx_private_readonly" / "bills",
    "decision_audit": Path("silver") / "decision_audit",
}

CORE_DIAGNOSTIC_DATASETS = {
    "silver/market_bar": Path("silver") / "market_bar",
    "gold/cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "gold/gate_decision": Path("gold") / "gate_decision",
    "gold/risk_permission": Path("gold") / "risk_permission",
    "gold/strategy_health_daily": Path("gold") / "strategy_health_daily",
}

MARKET_BOOTSTRAP_COMMAND = (
    "qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT "
    "--lake-root {lake_root} --history --limit 100"
)
V5_TELEMETRY_SYNC_COMMAND = (
    "qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml"
)
EXPERT_PACK_MISSING_MESSAGE = "qlab export-daily 尚未实现或尚未运行。"
DISPLAY_LIMIT = 500


@dataclass(frozen=True)
class DatasetState:
    name: str
    path: Path
    rows: int
    exists: bool
    parquet_file_count: int = 0
    warning: str | None = None


def read_dataset(lake_root: str | Path, dataset_name: str) -> pl.DataFrame:
    dataset_path = dataset_path_for(lake_root, dataset_name)
    return read_parquet_dataset(dataset_path)


def dataset_path_for(lake_root: str | Path, dataset_name: str) -> Path:
    relative = DATASET_PATHS.get(dataset_name)
    if relative is None:
        relative = Path(dataset_name)
    return Path(lake_root) / relative


def dataset_states(lake_root: str | Path) -> list[DatasetState]:
    states: list[DatasetState] = []
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(lake_root, name)
        try:
            df = read_parquet_dataset(path)
        except Exception as exc:
            states.append(
                DatasetState(
                    name=name,
                    path=path,
                    rows=0,
                    exists=path.exists(),
                    warning=str(exc),
                )
            )
            continue
        states.append(
            DatasetState(
                name=name,
                path=path,
                rows=df.height,
                exists=path.exists(),
                parquet_file_count=_parquet_file_count(path),
            )
        )
    return states


def dashboard_overview(lake_root: str | Path) -> dict[str, Any]:
    diagnostics = lake_diagnostics(lake_root)
    market_health = data_health_summary(lake_root)
    costs = cost_model_summary(lake_root)
    gates = alpha_gate_summary(lake_root)
    consumers = strategy_consumer_summary(lake_root)
    experts = expert_export_summary(default_exports_root(lake_root))

    warnings = [
        *diagnostics["warnings"],
        *market_health["warnings"],
        *costs["warnings"],
        *gates["warnings"],
        *consumers["warnings"],
        *experts["warnings"],
    ]
    status = "OK"
    if warnings:
        status = "WARNING"
    if market_health["schema_violation_count"] > 0 or market_health["unclosed_bar_count"] > 0:
        status = "CRITICAL"

    return {
        "status": status,
        "v5_permission": consumers["permissions"].get("v5", "UNKNOWN"),
        "v7_permission": consumers["permissions"].get("v7", "UNKNOWN"),
        "okx_public_rest_status": "OK" if market_health["latest_market_bar_ts"] else "WARNING",
        "okx_public_ws_status": "OK" if okx_ws_latest(lake_root) else "WARNING",
        "okx_readonly_status": readonly_private_status(lake_root),
        "latest_market_bar_ts": market_health["latest_market_bar_ts"],
        "missing_bar_ratio": market_health["missing_bar_ratio"],
        "cost_fallback_ratio": costs["fallback_ratio"],
        "alpha_gate_counts": gates["counts"],
        "latest_expert_pack": experts["latest_pack"],
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def lake_diagnostics(lake_root: str | Path) -> dict[str, Any]:
    root = Path(lake_root)
    parquet_count = _parquet_file_count(root)
    latest_market_bar_ts: datetime | None = None
    warnings: list[str] = []
    dataset_rows: list[dict[str, Any]] = []
    dataset_row_counts: dict[str, int] = {}

    if not root.exists():
        warnings.append(f"lake_root does not exist: {root}")
    if parquet_count == 0:
        warnings.append(f"No Parquet files found under lake_root: {root}")

    for dataset_name, relative_path in CORE_DIAGNOSTIC_DATASETS.items():
        path = root / relative_path
        row_count = 0
        warning = None
        try:
            df = read_parquet_dataset(path)
            row_count = df.height
            if dataset_name == "silver/market_bar" and not df.is_empty():
                latest_market_bar_ts = _max_datetime(_normalize_market_frame(df), "ts")
        except Exception as exc:
            warning = f"failed to read {dataset_name}: {exc}"
            warnings.append(warning)

        file_count = _parquet_file_count(path)
        exists = path.exists()
        dataset_row_counts[dataset_name] = row_count
        if not exists or row_count == 0:
            warnings.append(f"{dataset_name} dataset is missing or empty at {path}")

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "exists": exists,
                "parquet_file_count": file_count,
                "rows": row_count,
                "path": str(path),
                "warning": warning or "",
            }
        )

    if dataset_row_counts.get("silver/market_bar", 0) == 0:
        warnings.append(
            "market_bar is empty. Suggested command: "
            f"{MARKET_BOOTSTRAP_COMMAND.format(lake_root=root)}"
        )

    if dataset_row_counts.get("gold/strategy_health_daily", 0) == 0:
        warnings.append(
            "V5 telemetry is empty. Suggested command: " f"{V5_TELEMETRY_SYNC_COMMAND}"
        )

    experts = expert_export_summary(default_exports_root(root))
    if not experts["latest_pack"]:
        warnings.append(EXPERT_PACK_MISSING_MESSAGE)

    return {
        "lake_root": str(root),
        "lake_root_exists": root.exists(),
        "parquet_file_count": parquet_count,
        "latest_market_bar_ts": latest_market_bar_ts,
        "datasets": pl.DataFrame(dataset_rows),
        "warnings": _dedupe_strings(warnings),
    }


def data_health_summary(lake_root: str | Path) -> dict[str, Any]:
    warnings: list[str] = []
    market = read_dataset(lake_root, "market_bar")
    if market.is_empty():
        return {
            "latest_per_symbol": pl.DataFrame(),
            "missing_bars": pl.DataFrame(),
            "duplicate_bar_count": 0,
            "unclosed_bar_count": 0,
            "schema_violations": ["market_bar dataset is missing or empty"],
            "schema_violation_count": 1,
            "stale_datasets": _stale_dataset_rows(lake_root),
            "latest_market_bar_ts": None,
            "missing_bar_ratio": 0.0,
            "warnings": ["market_bar dataset is missing or empty"],
        }

    market = _normalize_market_frame(market)
    schema_violations = market_bar_schema_violations(market)
    duplicate_count = duplicate_market_bar_count(market)
    unclosed_count = unclosed_market_bar_count(market)
    latest_per_symbol = latest_market_bars(market)
    missing_bars = missing_bar_table(market)
    missing_ratio = _missing_ratio(missing_bars, market.height)
    latest_ts = _max_datetime(market, "ts")

    if duplicate_count:
        warnings.append(f"duplicate market_bar primary keys: {duplicate_count}")
    if unclosed_count:
        warnings.append(f"unclosed market bars: {unclosed_count}")
    warnings.extend(schema_violations)

    return {
        "latest_per_symbol": latest_per_symbol,
        "missing_bars": missing_bars,
        "duplicate_bar_count": duplicate_count,
        "unclosed_bar_count": unclosed_count,
        "schema_violations": schema_violations,
        "schema_violation_count": len(schema_violations),
        "stale_datasets": _stale_dataset_rows(lake_root),
        "latest_market_bar_ts": latest_ts,
        "missing_bar_ratio": missing_ratio,
        "warnings": warnings,
    }


def okx_collector_summary(lake_root: str | Path) -> dict[str, Any]:
    market = read_dataset(lake_root, "market_bar")
    ws_raw = read_dataset(lake_root, "okx_public_ws")
    trades = read_dataset(lake_root, "trade_print")
    books = read_dataset(lake_root, "orderbook_snapshot")

    latest_rest = _max_datetime(_normalize_optional_time(market, "ingest_ts"), "ingest_ts")
    latest_ws = _max_datetime(_normalize_optional_time(ws_raw, "received_at"), "received_at")

    rows = [
        {
            "collector": "OKX public REST",
            "success_count": market.height,
            "error_count": 0,
            "latest_success_ts": latest_rest,
            "reconnect_count": None,
            "rate_limit_warnings": 0,
            "lag": _lag_label(latest_rest),
        },
        {
            "collector": "OKX public WebSocket",
            "success_count": ws_raw.height,
            "error_count": 0,
            "latest_success_ts": latest_ws,
            "reconnect_count": 0,
            "rate_limit_warnings": 0,
            "lag": _lag_label(latest_ws),
        },
    ]
    warnings = []
    if market.is_empty():
        warnings.append("OKX public REST market_bar dataset is missing or empty")
    if ws_raw.is_empty():
        warnings.append("OKX public WebSocket bronze dataset is missing or empty")

    return {
        "collectors": pl.DataFrame(rows),
        "trade_print_rows": trades.height,
        "orderbook_snapshot_rows": books.height,
        "warnings": warnings,
    }


def market_regime_summary(lake_root: str | Path) -> dict[str, Any]:
    market = read_dataset(lake_root, "market_bar")
    books = read_dataset(lake_root, "orderbook_snapshot")
    trades = read_dataset(lake_root, "trade_print")
    if market.is_empty():
        return {
            "regimes": pl.DataFrame(),
            "spread_bps": pl.DataFrame(),
            "trade_activity": pl.DataFrame(),
            "abnormal_symbols": pl.DataFrame(),
            "warnings": ["market_bar dataset is missing or empty"],
        }

    market = _normalize_market_frame(market)
    regimes = (
        market.sort(["symbol", "timeframe", "ts"])
        .with_columns(
            ((pl.col("close") / pl.col("close").shift(1).over(["symbol", "timeframe"])) - 1.0)
            .abs()
            .alias("abs_return")
        )
        .group_by(["symbol", "timeframe"])
        .agg(
            pl.col("abs_return").mean().fill_null(0.0).alias("mean_abs_return"),
            pl.col("volume").mean().fill_null(0.0).alias("avg_volume"),
            pl.col("ts").max().alias("latest_ts"),
        )
        .with_columns(
            pl.when(pl.col("mean_abs_return") > 0.03)
            .then(pl.lit("HIGH_VOL"))
            .when(pl.col("mean_abs_return") > 0.01)
            .then(pl.lit("NORMAL"))
            .otherwise(pl.lit("LOW_VOL"))
            .alias("volatility_regime")
        )
        .sort(["symbol", "timeframe"])
    )

    return {
        "regimes": redact_frame(regimes),
        "spread_bps": orderbook_spread_table(books),
        "trade_activity": trade_activity_table(trades),
        "abnormal_symbols": regimes.filter(pl.col("mean_abs_return") > 0.03),
        "warnings": [],
    }


def cost_model_summary(lake_root: str | Path) -> dict[str, Any]:
    costs = read_dataset(lake_root, "cost_bucket_daily")
    if costs.is_empty():
        return {
            "costs": pl.DataFrame(),
            "fallback_ratio": 0.0,
            "warnings": ["cost_bucket_daily dataset is missing or empty"],
        }

    fallback_ratio = 0.0
    if "fallback_level" in costs.columns and costs.height:
        fallback_rows = costs.filter(
            ~pl.col("fallback_level").is_in(["actual_okx_fills_and_bills", "NONE"])
        ).height
        fallback_ratio = fallback_rows / costs.height

    return {
        "costs": redact_frame(costs).head(DISPLAY_LIMIT),
        "fallback_ratio": fallback_ratio,
        "warnings": [],
    }


def alpha_gate_summary(lake_root: str | Path) -> dict[str, Any]:
    gates = read_dataset(lake_root, "gate_decision")
    evidence = read_dataset(lake_root, "alpha_evidence")
    warnings = []
    if gates.is_empty():
        warnings.append("gate_decision dataset is missing or empty")
    counts: dict[str, int] = {}
    if "status" in gates.columns:
        counts = {
            row["status"]: row["count"]
            for row in gates.group_by("status").len(name="count").sort("status").to_dicts()
        }
    joined = gates
    if not gates.is_empty() and not evidence.is_empty():
        join_keys = [
            key
            for key in ["alpha_id", "version"]
            if key in gates.columns and key in evidence.columns
        ]
        if join_keys:
            joined = gates.join(evidence, on=join_keys, how="left", suffix="_evidence")
    return {
        "gates": redact_frame(joined).head(DISPLAY_LIMIT),
        "counts": counts,
        "warnings": warnings,
    }


def strategy_consumer_summary(lake_root: str | Path) -> dict[str, Any]:
    permissions = read_dataset(lake_root, "risk_permission")
    audits = read_dataset(lake_root, "decision_audit")
    warnings = []
    if permissions.is_empty():
        warnings.append("risk_permission dataset is missing or empty")

    latest_by_strategy: dict[str, str] = {}
    if not permissions.is_empty() and {"strategy", "permission"}.issubset(permissions.columns):
        sort_column = "created_at" if "created_at" in permissions.columns else "strategy"
        for row in permissions.sort(sort_column).to_dicts():
            latest_by_strategy[str(row["strategy"]).lower()] = str(row["permission"])

    return {
        "permissions": latest_by_strategy,
        "permission_rows": redact_frame(permissions).head(DISPLAY_LIMIT),
        "fallback_rows": _fallback_rows(audits),
        "warnings": warnings,
    }


def v5_telemetry_summary(lake_root: str | Path) -> dict[str, Any]:
    health = read_parquet_dataset(Path(lake_root) / "gold" / "strategy_health_daily")
    gate = read_parquet_dataset(Path(lake_root) / "gold" / "v5_gate_compliance_daily")
    if health.is_empty():
        return {
            "latest": {},
            "health_rows": pl.DataFrame(),
            "gate_compliance_rows": gate,
            "warnings": ["strategy_health_daily dataset is missing or empty"],
        }
    sort_column = "date" if "date" in health.columns else health.columns[0]
    latest = health.sort(sort_column).tail(1).to_dicts()[0]
    return {
        "latest": latest,
        "health_rows": health.sort(sort_column, descending=True).head(DISPLAY_LIMIT),
        "gate_compliance_rows": gate.head(DISPLAY_LIMIT),
        "warnings": [],
    }


def expert_export_summary(exports_root: str | Path) -> dict[str, Any]:
    root = Path(exports_root)
    packs = sorted(root.glob("quant_lab_expert_pack_*.zip")) if root.exists() else []
    if not packs:
        return {
            "latest_pack": None,
            "packs": pl.DataFrame(),
            "manifest_summary": {},
            "data_quality_summary": {},
            "expert_questions": [],
            "warnings": [f"no expert packs found under {root}"],
        }

    latest = packs[-1]
    manifest = _read_json_from_zip(latest, "manifest.json")
    data_quality = _read_json_from_zip(latest, "data_quality.json")
    questions = _read_text_from_zip(latest, "expert_questions.md").splitlines()
    pack_rows = [
        {
            "path": str(path),
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
        }
        for path in packs
    ]
    return {
        "latest_pack": str(latest),
        "packs": pl.DataFrame(pack_rows),
        "manifest_summary": manifest,
        "data_quality_summary": data_quality,
        "expert_questions": [line for line in questions if line.strip()][:20],
        "warnings": [],
    }


def default_exports_root(lake_root: str | Path) -> Path:
    root = Path(lake_root)
    if root.name == "lake":
        return root.parent / "exports"
    return root / "exports"


def _parquet_file_count(path: str | Path) -> int:
    root = Path(path)
    try:
        if root.is_file():
            return 1 if root.suffix == ".parquet" else 0
        if not root.exists():
            return 0
        return sum(1 for _ in root.rglob("*.parquet"))
    except OSError:
        return 0


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def latest_market_bars(market: pl.DataFrame) -> pl.DataFrame:
    if market.is_empty():
        return pl.DataFrame()
    return (
        market.group_by(["symbol", "timeframe"])
        .agg(pl.col("ts").max().alias("latest_ts"), pl.len().alias("rows"))
        .sort(["symbol", "timeframe"])
    )


def duplicate_market_bar_count(market: pl.DataFrame) -> int:
    keys = ["venue", "symbol", "timeframe", "ts"]
    if not set(keys).issubset(market.columns):
        return 0
    return market.group_by(keys).len(name="count").filter(pl.col("count") > 1).height


def unclosed_market_bar_count(market: pl.DataFrame) -> int:
    if "is_closed" not in market.columns:
        return 0
    return market.filter(pl.col("is_closed") == False).height  # noqa: E712


def market_bar_schema_violations(market: pl.DataFrame) -> list[str]:
    required = {
        "venue",
        "symbol",
        "market_type",
        "timeframe",
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "ingest_ts",
    }
    violations = [f"missing column: {column}" for column in sorted(required - set(market.columns))]
    if violations:
        return violations
    invalid_rows = market.filter(
        (pl.col("open") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.col("low"))
        | (pl.col("volume") < 0)
    ).height
    if invalid_rows:
        violations.append(f"invalid OHLCV rows: {invalid_rows}")
    return violations


def missing_bar_table(market: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    if market.is_empty() or not {"symbol", "timeframe", "ts"}.issubset(market.columns):
        return pl.DataFrame(rows)
    for group in market.group_by(["symbol", "timeframe"], maintain_order=True):
        keys, group_df = group
        symbol, timeframe = keys
        step = _timeframe_delta(str(timeframe))
        if step is None or group_df.height < 2:
            continue
        timestamps = sorted(set(group_df["ts"].to_list()))
        expected = int((timestamps[-1] - timestamps[0]) / step) + 1
        missing = max(expected - len(timestamps), 0)
        if missing:
            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "expected_bars": expected,
                    "actual_bars": len(timestamps),
                    "missing_bars": missing,
                }
            )
    return pl.DataFrame(rows)


def orderbook_spread_table(books: pl.DataFrame) -> pl.DataFrame:
    if books.is_empty() or not {"asks_json", "bids_json"}.issubset(books.columns):
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in books.to_dicts():
        ask = _best_price(row.get("asks_json"))
        bid = _best_price(row.get("bids_json"))
        if ask is None or bid is None or ask <= bid:
            continue
        mid = (ask + bid) / 2.0
        rows.append(
            {
                "symbol": row.get("symbol"),
                "channel": row.get("channel"),
                "ts": row.get("ts"),
                "spread_bps": ((ask - bid) / mid) * 10_000,
            }
        )
    return pl.DataFrame(rows)


def trade_activity_table(trades: pl.DataFrame) -> pl.DataFrame:
    if trades.is_empty() or "symbol" not in trades.columns:
        return pl.DataFrame()
    aggregations = [pl.len().alias("trade_count")]
    if "size" in trades.columns:
        aggregations.append(pl.col("size").cast(pl.Float64, strict=False).sum().alias("size_sum"))
    if "ts" in trades.columns:
        aggregations.append(pl.col("ts").max().alias("latest_trade_ts"))
    return trades.group_by("symbol").agg(aggregations).sort("symbol")


def redact_frame(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    expressions = []
    for column in df.columns:
        if _sensitive_column(column):
            expressions.append(pl.lit("[REDACTED]").alias(column))
        else:
            expressions.append(
                pl.col(column).map_elements(redact_text, return_dtype=df.schema[column])
            )
    try:
        return df.with_columns(expressions)
    except Exception:
        return df


def redact_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    if _sensitive_text(lowered):
        return "[REDACTED]"
    return value


def okx_ws_latest(lake_root: str | Path) -> datetime | None:
    ws_raw = read_dataset(lake_root, "okx_public_ws")
    return _max_datetime(_normalize_optional_time(ws_raw, "received_at"), "received_at")


def readonly_private_status(lake_root: str | Path) -> str:
    fills = read_dataset(lake_root, "okx_private_readonly_fills")
    bills = read_dataset(lake_root, "okx_private_readonly_bills")
    if fills.is_empty() and bills.is_empty():
        return "NOT_CONFIGURED"
    return "OK"


def _fallback_rows(audits: pl.DataFrame) -> pl.DataFrame:
    if audits.is_empty():
        return pl.DataFrame()
    text_columns = [column for column in audits.columns if audits.schema[column] == pl.String]
    if not text_columns:
        return pl.DataFrame()
    predicate = None
    for column in text_columns:
        expression = pl.col(column).str.contains("fallback", literal=True)
        predicate = expression if predicate is None else predicate | expression
    return redact_frame(audits.filter(predicate)) if predicate is not None else pl.DataFrame()


def _stale_dataset_rows(lake_root: str | Path) -> pl.DataFrame:
    rows = []
    for state in dataset_states(lake_root):
        if not state.exists or state.rows == 0:
            rows.append(
                {
                    "dataset": state.name,
                    "rows": state.rows,
                    "status": "missing",
                    "path": str(state.path),
                }
            )
    return pl.DataFrame(rows)


def _normalize_market_frame(df: pl.DataFrame) -> pl.DataFrame:
    normalized = _normalize_optional_time(df, "ts")
    normalized = _normalize_optional_time(normalized, "ingest_ts")
    return normalized


def _normalize_optional_time(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if df.is_empty() or column not in df.columns:
        return df
    if df.schema[column] == pl.String:
        return df.with_columns(pl.col(column).str.to_datetime(time_zone="UTC", strict=False))
    try:
        return df.with_columns(pl.col(column).cast(pl.Datetime(time_zone="UTC")).alias(column))
    except Exception:
        return df


def _max_datetime(df: pl.DataFrame, column: str) -> datetime | None:
    if df.is_empty() or column not in df.columns:
        return None
    value = df.select(pl.col(column).max()).item()
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return None


def _missing_ratio(missing_bars: pl.DataFrame, actual_rows: int) -> float:
    if missing_bars.is_empty() or "missing_bars" not in missing_bars.columns:
        return 0.0
    missing = float(missing_bars["missing_bars"].sum())
    denominator = actual_rows + missing
    return missing / denominator if denominator else 0.0


def _lag_label(timestamp: datetime | None) -> str:
    if timestamp is None:
        return "unknown"
    lag = datetime.now(UTC) - timestamp
    if lag < timedelta(hours=2):
        return "fresh"
    if lag < timedelta(hours=24):
        return "delayed"
    return "stale"


def _timeframe_delta(timeframe: str) -> timedelta | None:
    match timeframe:
        case "1m":
            return timedelta(minutes=1)
        case "3m":
            return timedelta(minutes=3)
        case "5m":
            return timedelta(minutes=5)
        case "15m":
            return timedelta(minutes=15)
        case "30m":
            return timedelta(minutes=30)
        case "1H":
            return timedelta(hours=1)
        case "2H":
            return timedelta(hours=2)
        case "4H":
            return timedelta(hours=4)
        case "6H":
            return timedelta(hours=6)
        case "12H":
            return timedelta(hours=12)
        case "1D" | "1Dutc":
            return timedelta(days=1)
        case _:
            return None


def _best_price(raw_value: Any) -> float | None:
    if raw_value is None:
        return None
    try:
        values = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        return float(values[0][0]) if values else None
    except Exception:
        return None


def _read_json_from_zip(path: Path, member: str) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                payload = json.loads(handle.read().decode("utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_text_from_zip(path: Path, member: str) -> str:
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member) as handle:
                return handle.read().decode("utf-8")
    except Exception:
        return ""


def _sensitive_column(column: str) -> bool:
    lowered = column.lower()
    return (
        "secret" in lowered
        or "passphrase" in lowered
        or "credential" in lowered
        or "token" in lowered
        or ("key" in lowered and ("api" in lowered or "access" in lowered))
    )


def _sensitive_text(lowered: str) -> bool:
    return (
        "ok-access-" in lowered
        or "passphrase=" in lowered
        or "secret=" in lowered
        or "credential=" in lowered
    )
