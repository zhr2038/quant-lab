from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import invalid_parquet_files, read_parquet_dataset

DATASET_PATHS = {
    "market_bar": Path("silver") / "market_bar",
    "feature_value": Path("gold") / "feature_value",
    "feature_coverage_daily": Path("gold") / "feature_coverage_daily",
    "feature_anomaly_daily": Path("gold") / "feature_anomaly_daily",
    "cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "cost_health_daily": Path("gold") / "cost_health_daily",
    "alpha_evidence": Path("gold") / "alpha_evidence",
    "gate_decision": Path("gold") / "gate_decision",
    "risk_permission": Path("gold") / "risk_permission",
    "strategy_health_daily": Path("gold") / "strategy_health_daily",
    "okx_public_ws_health": Path("bronze") / "collector_health" / "okx_public_ws",
    "v5_execution_quality_daily": Path("gold") / "v5_execution_quality_daily",
    "v5_gate_compliance_daily": Path("gold") / "v5_gate_compliance_daily",
    "v5_missed_opportunity_daily": Path("gold") / "v5_missed_opportunity_daily",
    "v5_config_health_daily": Path("gold") / "v5_config_health_daily",
    "v5_issue_summary_daily": Path("gold") / "v5_issue_summary_daily",
    "v5_quant_lab_mode_daily": Path("gold") / "v5_quant_lab_mode_daily",
    "v5_quant_lab_enforcement_daily": Path("gold") / "v5_quant_lab_enforcement_daily",
    "v5_quant_lab_usage": Path("silver") / "v5_quant_lab_usage",
    "v5_quant_lab_compliance": Path("silver") / "v5_quant_lab_compliance",
    "v5_quant_lab_cost_usage": Path("silver") / "v5_quant_lab_cost_usage",
    "v5_quant_lab_fallback": Path("silver") / "v5_quant_lab_fallback",
    "trade_print": Path("silver") / "trade_print",
    "orderbook_snapshot": Path("silver") / "orderbook_snapshot",
    "okx_public_ws": Path("bronze") / "okx_public_ws",
    "okx_private_readonly_fills": Path("bronze") / "okx_private_readonly" / "fills_history",
    "okx_private_readonly_bills": Path("bronze") / "okx_private_readonly" / "bills",
    "decision_audit": Path("silver") / "decision_audit",
}

DATASET_TIMESTAMP_COLUMNS: dict[str, tuple[str, ...]] = {
    "market_bar": ("ts",),
    "okx_public_ws": ("received_at",),
    "trade_print": ("ts", "ingest_ts"),
    "orderbook_snapshot": ("ts", "ingest_ts"),
    "cost_bucket_daily": ("created_at", "day"),
    "cost_health_daily": ("created_at", "day"),
    "gate_decision": ("created_at",),
    "risk_permission": ("created_at",),
    "strategy_health_daily": ("latest_bundle_ts", "date"),
    "v5_execution_quality_daily": ("latest_bundle_ts", "date"),
    "v5_gate_compliance_daily": ("latest_bundle_ts", "date"),
    "v5_missed_opportunity_daily": ("latest_bundle_ts", "date"),
    "v5_config_health_daily": ("latest_bundle_ts", "date"),
    "v5_issue_summary_daily": ("latest_bundle_ts", "date"),
    "v5_quant_lab_mode_daily": ("latest_bundle_ts", "created_at", "date"),
    "v5_quant_lab_enforcement_daily": ("latest_bundle_ts", "created_at", "date"),
    "okx_private_readonly_fills": ("ingest_ts",),
    "okx_private_readonly_bills": ("ingest_ts",),
    "feature_value": ("created_at", "ts"),
    "feature_coverage_daily": ("created_at", "max_ts", "day"),
    "feature_anomaly_daily": ("created_at", "example_ts", "day"),
    "alpha_evidence": ("created_at", "end_ts"),
    "okx_public_ws_health": ("last_message_at", "updated_at", "started_at"),
    "decision_audit": ("ingest_ts", "loaded_at"),
    "v5_quant_lab_usage": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_compliance": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_cost_usage": ("ingest_ts", "bundle_ts"),
    "v5_quant_lab_fallback": ("ingest_ts", "bundle_ts"),
}

CORE_DIAGNOSTIC_DATASETS = {
    "silver/market_bar": Path("silver") / "market_bar",
    "gold/cost_bucket_daily": Path("gold") / "cost_bucket_daily",
    "gold/cost_health_daily": Path("gold") / "cost_health_daily",
    "gold/gate_decision": Path("gold") / "gate_decision",
    "gold/risk_permission": Path("gold") / "risk_permission",
    "gold/strategy_health_daily": Path("gold") / "strategy_health_daily",
    "gold/v5_gate_compliance_daily": Path("gold") / "v5_gate_compliance_daily",
}

MARKET_BOOTSTRAP_COMMAND = (
    "qlab okx-fetch-candles --inst-id BTC-USDT --bar 1H --market-type SPOT "
    "--lake-root {lake_root} --history --limit 100"
)
V5_TELEMETRY_SYNC_COMMAND = (
    "qlab sync-v5-telemetry --config /etc/quant-lab/v5_telemetry_remote.yaml"
)
OKX_WS_COLLECT_COMMAND = (
    "qlab okx-ws-collect-universe --symbols BTC-USDT,ETH-USDT,SOL-USDT,BNB-USDT "
    "--channels tickers,trades,books5 --market-type SPOT --lake-root {lake_root} "
    "--flush-interval-seconds 10 --flush-max-messages 100"
)
EXPERT_PACK_MISSING_MESSAGE = "qlab export-daily 尚未实现或尚未运行。"
BOOTSTRAP_GOLD_HEALTH_COMMAND = (
    "qlab bootstrap-gold-health --lake-root {lake_root} "
    "--strategy v5 --version bootstrap --day auto"
)
BOOTSTRAP_PLACEHOLDER_WARNING = "bootstrap conservative placeholder, not live-ready evidence"
DISPLAY_LIMIT = 500


@dataclass(frozen=True)
class DatasetState:
    name: str
    path: Path
    rows: int
    exists: bool
    parquet_file_count: int = 0
    warning: str | None = None


@dataclass(frozen=True)
class DatasetSnapshot:
    rows: int
    exists: bool
    parquet_file_count: int
    freshness: dict[str, Any]
    warning: str | None = None


def read_dataset(lake_root: str | Path, dataset_name: str) -> pl.DataFrame:
    df, _warning = read_dataset_with_warning(lake_root, dataset_name)
    return df


def read_dataset_with_warning(
    lake_root: str | Path,
    dataset_name: str,
) -> tuple[pl.DataFrame, str | None]:
    dataset_path = dataset_path_for(lake_root, dataset_name)
    return _read_parquet_dataset_with_warning(dataset_path, dataset_name)


def dataset_path_for(lake_root: str | Path, dataset_name: str) -> Path:
    relative = DATASET_PATHS.get(dataset_name)
    if relative is None:
        relative = Path(dataset_name)
    return Path(lake_root) / relative


def dataset_freshness_payload(
    dataset_name: str,
    df: pl.DataFrame,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    if df.is_empty():
        return {
            "freshness_seconds": None,
            "freshness_status": "missing",
            "timestamp_column": None,
            "latest_timestamp": None,
        }
    latest, column = latest_dataset_timestamp(dataset_name, df)
    if latest is None:
        return {
            "freshness_seconds": None,
            "freshness_status": "unknown",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    reference_time = now.astimezone(UTC) if now else datetime.now(UTC)
    seconds = max(int((reference_time - latest).total_seconds()), 0)
    if seconds <= 6 * 60 * 60:
        status = "fresh"
    elif seconds <= 24 * 60 * 60:
        status = "delayed"
    else:
        status = "stale"
    return {
        "freshness_seconds": seconds,
        "freshness_status": status,
        "timestamp_column": column,
        "latest_timestamp": latest.isoformat(),
    }


def latest_dataset_timestamp(
    dataset_name: str,
    df: pl.DataFrame,
) -> tuple[datetime | None, str | None]:
    if df.is_empty():
        return None, None
    columns = DATASET_TIMESTAMP_COLUMNS.get(dataset_name, ("ts", "created_at", "ingest_ts"))
    seen_timestamp_column: str | None = None
    for column in columns:
        if column not in df.columns:
            continue
        seen_timestamp_column = column
        values = [_coerce_timestamp(value) for value in df[column].to_list()]
        parsed = [value for value in values if value is not None]
        if parsed:
            return max(parsed), column
    return None, seen_timestamp_column


def _read_parquet_dataset_with_warning(
    dataset_path: str | Path,
    dataset_label: str,
) -> tuple[pl.DataFrame, str | None]:
    path = Path(dataset_path)
    invalid_files = invalid_parquet_files(path)
    try:
        df = read_parquet_dataset(path)
    except Exception as exc:
        return pl.DataFrame(), (
            f"{dataset_label} 数据集读取失败，已按空数据处理：{path}；错误：{exc}"
        )
    if invalid_files:
        rendered = ", ".join(str(file_path) for file_path in invalid_files[:5])
        suffix = "" if len(invalid_files) <= 5 else f" (+{len(invalid_files) - 5} more)"
        return df, (
            f"{dataset_label} invalid parquet files ignored: {rendered}{suffix}"
        )
    return df, None


def _dataset_snapshot(
    lake_root: str | Path,
    dataset_name: str,
    *,
    timestamp_columns: tuple[str, ...] | None = None,
    now: datetime | None = None,
) -> DatasetSnapshot:
    path = dataset_path_for(lake_root, dataset_name)
    invalid_files = invalid_parquet_files(path)
    files = _valid_parquet_files(path, invalid_files=invalid_files)
    warning = _invalid_parquet_warning(dataset_name, invalid_files)
    if not files:
        return DatasetSnapshot(
            rows=0,
            exists=path.exists(),
            parquet_file_count=0,
            freshness=_freshness_payload(None, None, is_empty=True, now=now),
            warning=warning,
        )

    try:
        lazy = pl.scan_parquet([str(file_path) for file_path in files])
        schema = lazy.collect_schema()
        rows = int(lazy.select(pl.len().alias("rows")).collect().item())
        latest, column = _latest_lazy_timestamp(
            dataset_name,
            lazy,
            schema,
            timestamp_columns=timestamp_columns,
        )
    except Exception as exc:
        return DatasetSnapshot(
            rows=0,
            exists=path.exists(),
            parquet_file_count=len(files),
            freshness=_freshness_payload(None, None, is_empty=True, now=now),
            warning=f"{dataset_name} metadata read failed: {exc}",
        )

    return DatasetSnapshot(
        rows=rows,
        exists=path.exists(),
        parquet_file_count=len(files),
        freshness=_freshness_payload(latest, column, is_empty=rows == 0, now=now),
        warning=warning,
    )


def _latest_lazy_timestamp(
    dataset_name: str,
    lazy: pl.LazyFrame,
    schema: pl.Schema,
    *,
    timestamp_columns: tuple[str, ...] | None = None,
) -> tuple[datetime | None, str | None]:
    columns = timestamp_columns or DATASET_TIMESTAMP_COLUMNS.get(
        dataset_name,
        ("ts", "created_at", "ingest_ts"),
    )
    seen_timestamp_column: str | None = None
    for column in columns:
        if column not in schema:
            continue
        seen_timestamp_column = column
        value = lazy.select(pl.col(column).max().alias(column)).collect().item()
        latest = _coerce_timestamp(value)
        if latest is not None:
            return latest, column
    return None, seen_timestamp_column


def _snapshot_latest_timestamp(snapshot: DatasetSnapshot) -> datetime | None:
    value = snapshot.freshness.get("latest_timestamp")
    return _coerce_timestamp(value)


def _freshness_payload(
    latest: datetime | None,
    column: str | None,
    *,
    is_empty: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    if is_empty:
        return {
            "freshness_seconds": None,
            "freshness_status": "missing",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    if latest is None:
        return {
            "freshness_seconds": None,
            "freshness_status": "unknown",
            "timestamp_column": column,
            "latest_timestamp": None,
        }
    reference_time = now.astimezone(UTC) if now else datetime.now(UTC)
    seconds = max(int((reference_time - latest).total_seconds()), 0)
    if seconds <= 6 * 60 * 60:
        status = "fresh"
    elif seconds <= 24 * 60 * 60:
        status = "delayed"
    else:
        status = "stale"
    return {
        "freshness_seconds": seconds,
        "freshness_status": status,
        "timestamp_column": column,
        "latest_timestamp": latest.isoformat(),
    }


def _valid_parquet_files(path: Path, *, invalid_files: list[Path] | None = None) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        candidates = [path]
    elif path.exists():
        candidates = sorted(path.rglob("*.parquet"))
    else:
        candidates = []
    invalid = set(invalid_files if invalid_files is not None else invalid_parquet_files(path))
    return [file_path for file_path in candidates if file_path not in invalid]


def _invalid_parquet_warning(dataset_label: str, invalid_files: list[Path]) -> str | None:
    if not invalid_files:
        return None
    rendered = ", ".join(str(file_path) for file_path in invalid_files[:5])
    suffix = "" if len(invalid_files) <= 5 else f" (+{len(invalid_files) - 5} more)"
    return f"{dataset_label} invalid parquet files ignored: {rendered}{suffix}"


def dataset_states(lake_root: str | Path) -> list[DatasetState]:
    states: list[DatasetState] = []
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(lake_root, name)
        snapshot = _dataset_snapshot(lake_root, name)
        states.append(
            DatasetState(
                name=name,
                path=path,
                rows=snapshot.rows,
                exists=snapshot.exists,
                parquet_file_count=snapshot.parquet_file_count,
                warning=snapshot.warning,
            )
        )
    return states


def dashboard_overview(lake_root: str | Path) -> dict[str, Any]:
    diagnostics = lake_diagnostics(lake_root)
    market_health = data_health_summary(lake_root)
    costs = cost_model_summary(lake_root)
    gates = alpha_gate_summary(lake_root)
    consumers = strategy_consumer_summary(lake_root)
    collectors = okx_collector_summary(lake_root)
    experts = expert_export_summary(default_exports_root(lake_root))

    warnings = [
        *diagnostics["warnings"],
        *market_health["warnings"],
        *costs["warnings"],
        *gates["warnings"],
        *consumers["warnings"],
        *collectors["warnings"],
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
        "okx_public_ws_status": collectors["okx_public_ws_status"],
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
    suggested_commands: list[dict[str, str]] = []
    dataset_rows: list[dict[str, Any]] = []
    dataset_row_counts: dict[str, int] = {}

    if not root.exists():
        warnings.append(f"Lake 根目录不存在：{root}")
    if parquet_count == 0:
        warnings.append(f"Lake 根目录下没有 Parquet 文件：{root}")

    for dataset_name, relative_path in CORE_DIAGNOSTIC_DATASETS.items():
        path = root / relative_path
        row_count = 0
        warning = None
        df, warning = _read_parquet_dataset_with_warning(path, dataset_name)
        canonical_name = _canonical_dataset_name(dataset_name)
        freshness = dataset_freshness_payload(canonical_name, df)
        row_count = df.height
        if warning:
            warnings.append(warning)
        if dataset_name == "silver/market_bar" and not df.is_empty():
            latest_market_bar_ts = _max_datetime(_normalize_market_frame(df), "ts")

        file_count = _parquet_file_count(path)
        exists = path.exists()
        dataset_row_counts[dataset_name] = row_count
        if not exists or row_count == 0:
            warnings.append(f"{dataset_name} 数据集缺失或为空：{path}")
        elif _is_bootstrap_placeholder(df):
            warning = BOOTSTRAP_PLACEHOLDER_WARNING
            warnings.append(f"{dataset_name}: {warning}")

        dataset_rows.append(
            {
                "dataset": dataset_name,
                "exists": exists,
                "parquet_file_count": file_count,
                "rows": row_count,
                "path": str(path),
                "freshness_seconds": freshness["freshness_seconds"],
                "freshness_status": freshness["freshness_status"],
                "timestamp_column": freshness["timestamp_column"] or "",
                "latest_timestamp": freshness["latest_timestamp"] or "",
                "warning": warning or "",
            }
        )

    if dataset_row_counts.get("silver/market_bar", 0) == 0:
        command = MARKET_BOOTSTRAP_COMMAND.format(lake_root=root)
        warnings.append(
            "market_bar 为空。建议命令："
            f"{MARKET_BOOTSTRAP_COMMAND.format(lake_root=root)}"
        )

        suggested_commands.append({"purpose": "backfill market_bar", "command": command})

    gold_health_missing = any(
        dataset_row_counts.get(dataset_name, 0) == 0
        for dataset_name in [
            "gold/cost_bucket_daily",
            "gold/gate_decision",
            "gold/risk_permission",
        ]
    )
    if gold_health_missing:
        command = BOOTSTRAP_GOLD_HEALTH_COMMAND.format(lake_root=root)
        warnings.append(f"gold health bootstrap suggested: {command}")
        suggested_commands.append({"purpose": "bootstrap gold health", "command": command})

    ws_health, ws_health_warning = _read_parquet_dataset_with_warning(
        root / DATASET_PATHS["okx_public_ws_health"],
        "okx_public_ws_health",
    )
    if ws_health_warning:
        warnings.append(ws_health_warning)
    if ws_health.is_empty():
        command = OKX_WS_COLLECT_COMMAND.format(lake_root=root)
        warnings.append(
            f"OKX public WebSocket collector health is empty. Suggested command: {command}"
        )
        suggested_commands.append({"purpose": "run OKX public WebSocket", "command": command})

    if dataset_row_counts.get("gold/strategy_health_daily", 0) == 0:
        warnings.append(
            "V5 遥测为空。建议命令：" f"{V5_TELEMETRY_SYNC_COMMAND}"
        )

        suggested_commands.append(
            {"purpose": "sync V5 telemetry", "command": V5_TELEMETRY_SYNC_COMMAND}
        )

    experts = expert_export_summary(default_exports_root(root))
    if not experts["latest_pack"]:
        warnings.append(EXPERT_PACK_MISSING_MESSAGE)
        suggested_commands.append(
            {
                "purpose": "create expert pack",
                "command": (
                    "qlab export-daily --date YYYY-MM-DD "
                    f"--lake-root {root} --out-dir {default_exports_root(root)}"
                ),
            }
        )

    return {
        "lake_root": str(root),
        "lake_root_exists": root.exists(),
        "parquet_file_count": parquet_count,
        "latest_market_bar_ts": latest_market_bar_ts,
        "latest_v5_bundle_ts": _latest_v5_bundle_ts(root),
        "latest_expert_pack": experts["latest_pack"],
        "datasets": pl.DataFrame(dataset_rows),
        "suggested_commands": pl.DataFrame(suggested_commands),
        "warnings": _dedupe_strings(warnings),
    }


def data_health_summary(lake_root: str | Path) -> dict[str, Any]:
    warnings: list[str] = []
    market, market_warning = read_dataset_with_warning(lake_root, "market_bar")
    if market_warning:
        warnings.append(market_warning)
    if market.is_empty():
        return {
            "latest_per_symbol": pl.DataFrame(),
            "missing_bars": pl.DataFrame(),
            "duplicate_bar_count": 0,
            "unclosed_bar_count": 0,
            "schema_violations": ["market_bar 数据集缺失或为空"],
            "schema_violation_count": 1,
            "stale_datasets": _stale_dataset_rows(lake_root),
            "latest_market_bar_ts": None,
            "missing_bar_ratio": 0.0,
            "warnings": [*warnings, "market_bar 数据集缺失或为空"],
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
        warnings.append(f"market_bar 主键重复：{duplicate_count}")
    if unclosed_count:
        warnings.append(f"未闭合 market_bar：{unclosed_count}")
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
    market_snapshot = _dataset_snapshot(
        lake_root,
        "market_bar",
        timestamp_columns=("ingest_ts",),
    )
    ws_raw_snapshot = _dataset_snapshot(lake_root, "okx_public_ws")
    trades_snapshot = _dataset_snapshot(lake_root, "trade_print")
    books_snapshot = _dataset_snapshot(lake_root, "orderbook_snapshot")
    ws_health, ws_health_warning = read_dataset_with_warning(lake_root, "okx_public_ws_health")

    latest_rest = _snapshot_latest_timestamp(market_snapshot)
    latest_ws = _snapshot_latest_timestamp(ws_raw_snapshot)
    latest_health = _latest_collector_health(ws_health)
    if latest_health:
        latest_ws = _parse_datetime(latest_health.get("last_message_at")) or latest_ws

    rows = [
        {
            "collector": "OKX 公共 REST",
            "success_count": market_snapshot.rows,
            "error_count": 0,
            "latest_success_ts": latest_rest,
            "reconnect_count": None,
            "rate_limit_warnings": 0,
            "lag": _lag_label(latest_rest),
        },
        {
            "collector": "OKX 公共 WebSocket",
            "success_count": int(latest_health.get("messages_read") or ws_raw_snapshot.rows)
            if latest_health
            else ws_raw_snapshot.rows,
            "error_count": int(latest_health.get("error_count") or 0) if latest_health else 0,
            "latest_success_ts": latest_ws,
            "reconnect_count": int(latest_health.get("reconnect_count") or 0)
            if latest_health
            else 0,
            "rate_limit_warnings": 0,
            "lag": _lag_label(latest_ws),
            "status": str(latest_health.get("status") or "UNKNOWN") if latest_health else "UNKNOWN",
        },
    ]
    warnings = [
        warning
        for warning in [
            market_snapshot.warning,
            ws_raw_snapshot.warning,
            ws_health_warning,
            trades_snapshot.warning,
            books_snapshot.warning,
        ]
        if warning
    ]
    if market_snapshot.rows == 0:
        warnings.append("OKX 公共 REST market_bar 数据集缺失或为空")
    if ws_raw_snapshot.rows == 0:
        warnings.append("OKX 公共 WebSocket bronze 数据集缺失或为空")

    if ws_health.is_empty():
        warnings.append("OKX public WebSocket collector_health is missing or empty")

    return {
        "collectors": pl.DataFrame(rows),
        "collector_health": redact_frame(ws_health).head(DISPLAY_LIMIT),
        "okx_public_ws_status": str(latest_health.get("status") or "UNKNOWN")
        if latest_health
        else "WARNING",
        "trade_print_rows": trades_snapshot.rows,
        "orderbook_snapshot_rows": books_snapshot.rows,
        "warnings": warnings,
    }


def market_regime_summary(lake_root: str | Path) -> dict[str, Any]:
    market, market_warning = read_dataset_with_warning(lake_root, "market_bar")
    books, books_warning = read_dataset_with_warning(lake_root, "orderbook_snapshot")
    trades, trades_warning = read_dataset_with_warning(lake_root, "trade_print")
    warnings = [warning for warning in [market_warning, books_warning, trades_warning] if warning]
    if market.is_empty():
        return {
            "regimes": pl.DataFrame(),
            "spread_bps": pl.DataFrame(),
            "trade_activity": pl.DataFrame(),
            "abnormal_symbols": pl.DataFrame(),
            "warnings": [*warnings, "market_bar 数据集缺失或为空"],
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
            .then(pl.lit("高波动"))
            .when(pl.col("mean_abs_return") > 0.01)
            .then(pl.lit("正常"))
            .otherwise(pl.lit("低波动"))
            .alias("volatility_regime")
        )
        .sort(["symbol", "timeframe"])
    )

    return {
        "regimes": redact_frame(regimes),
        "spread_bps": orderbook_spread_table(books),
        "trade_activity": trade_activity_table(trades),
        "abnormal_symbols": regimes.filter(pl.col("mean_abs_return") > 0.03),
        "warnings": warnings,
    }


def cost_model_summary(lake_root: str | Path) -> dict[str, Any]:
    costs, costs_warning = read_dataset_with_warning(lake_root, "cost_bucket_daily")
    health, health_warning = read_dataset_with_warning(lake_root, "cost_health_daily")
    warnings = [warning for warning in [costs_warning, health_warning] if warning]
    if costs.is_empty():
        return {
            "costs": pl.DataFrame(),
            "cost_health": redact_frame(health).head(DISPLAY_LIMIT),
            "actual_rows": 0,
            "proxy_rows": 0,
            "global_default_rows": 0,
            "fallback_ratio": None,
            "fallback_ratio_status": "N/A",
            "warnings": [*warnings, "cost_bucket_daily 数据集缺失或为空"],
        }

    fallback_ratio = 0.0
    if "fallback_level" in costs.columns and costs.height:
        fallback_rows = costs.filter(
            ~pl.col("fallback_level").is_in(["actual_okx_fills_and_bills", "NONE"])
        ).height
        fallback_ratio = fallback_rows / costs.height

    latest_health = health.sort("day").tail(1).to_dicts()[0] if not health.is_empty() else {}
    return {
        "costs": redact_frame(costs).head(DISPLAY_LIMIT),
        "cost_health": redact_frame(health).head(DISPLAY_LIMIT),
        "actual_rows": int(latest_health.get("actual_rows") or 0),
        "proxy_rows": int(latest_health.get("proxy_rows") or 0),
        "global_default_rows": int(latest_health.get("global_default_rows") or 0),
        "symbols_with_actual_cost": latest_health.get("symbols_with_actual_cost"),
        "symbols_with_proxy_only": latest_health.get("symbols_with_proxy_only"),
        "fallback_ratio": fallback_ratio,
        "fallback_ratio_status": "OK" if fallback_ratio <= 0.25 else "WARNING",
        "warnings": warnings,
    }


def alpha_gate_summary(lake_root: str | Path) -> dict[str, Any]:
    gates, gates_warning = read_dataset_with_warning(lake_root, "gate_decision")
    evidence, evidence_warning = read_dataset_with_warning(lake_root, "alpha_evidence")
    warnings = [warning for warning in [gates_warning, evidence_warning] if warning]
    if gates.is_empty():
        warnings.append("gate_decision 数据集缺失或为空")
    if evidence.is_empty():
        warnings.append("alpha_evidence research evidence not generated yet")
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


def feature_summary(lake_root: str | Path) -> dict[str, Any]:
    features, feature_warning = read_dataset_with_warning(lake_root, "feature_value")
    coverage, coverage_warning = read_dataset_with_warning(lake_root, "feature_coverage_daily")
    anomalies, anomaly_warning = read_dataset_with_warning(lake_root, "feature_anomaly_daily")
    warnings = [
        warning
        for warning in [feature_warning, coverage_warning, anomaly_warning]
        if warning
    ]
    if features.is_empty():
        warnings.append("feature_value dataset is missing or empty")
    latest = _latest_features_table(features)
    null_ratio = 0.0
    if not features.is_empty() and "value" in features.columns:
        null_ratio = float(features["value"].null_count()) / features.height
    display_features = features.tail(DISPLAY_LIMIT) if not features.is_empty() else features
    return {
        "features": redact_frame(display_features),
        "coverage": redact_frame(coverage),
        "anomalies": redact_frame(anomalies),
        "latest": redact_frame(latest),
        "null_ratio": null_ratio,
        "top_anomalies": _top_feature_anomalies(anomalies),
        "warnings": warnings,
    }


def strategy_consumer_summary(lake_root: str | Path) -> dict[str, Any]:
    permissions, permissions_warning = read_dataset_with_warning(lake_root, "risk_permission")
    audits, audits_warning = read_dataset_with_warning(lake_root, "decision_audit")
    warnings = [warning for warning in [permissions_warning, audits_warning] if warning]
    if permissions.is_empty():
        warnings.append("risk_permission 数据集缺失或为空")

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
    health, health_warning = read_dataset_with_warning(lake_root, "strategy_health_daily")
    gate, gate_warning = read_dataset_with_warning(lake_root, "v5_gate_compliance_daily")
    mode, mode_warning = read_dataset_with_warning(lake_root, "v5_quant_lab_mode_daily")
    enforcement, enforcement_warning = read_dataset_with_warning(
        lake_root,
        "v5_quant_lab_enforcement_daily",
    )
    warnings = [
        warning
        for warning in [health_warning, gate_warning, mode_warning, enforcement_warning]
        if warning
    ]
    if health.is_empty():
        return {
            "latest": {},
            "health_rows": pl.DataFrame(),
            "gate_compliance_rows": gate,
            "quant_lab_mode_rows": mode,
            "quant_lab_enforcement_rows": enforcement,
            "warnings": [*warnings, "strategy_health_daily 数据集缺失或为空"],
        }
    sort_column = "date" if "date" in health.columns else health.columns[0]
    latest = health.sort(sort_column).tail(1).to_dicts()[0]
    return {
        "latest": latest,
        "health_rows": health.sort(sort_column, descending=True).head(DISPLAY_LIMIT),
        "gate_compliance_rows": gate.head(DISPLAY_LIMIT),
        "quant_lab_mode_rows": mode.head(DISPLAY_LIMIT),
        "quant_lab_enforcement_rows": enforcement.head(DISPLAY_LIMIT),
        "warnings": warnings,
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
            "warnings": [f"未在目录下找到专家包：{root}"],
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


def _latest_v5_bundle_ts(lake_root: str | Path) -> datetime | None:
    health, _health_warning = read_dataset_with_warning(lake_root, "strategy_health_daily")
    if not health.is_empty() and "latest_bundle_ts" in health.columns:
        normalized = _normalize_optional_time(health, "latest_bundle_ts")
        return _max_datetime(normalized, "latest_bundle_ts")
    manifest, _manifest_warning = _read_parquet_dataset_with_warning(
        Path(lake_root) / "bronze/strategy_telemetry/v5/bundle_manifest",
        "bronze/strategy_telemetry/v5/bundle_manifest",
    )
    return _max_datetime(_normalize_optional_time(manifest, "bundle_ts"), "bundle_ts")


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


def _is_bootstrap_placeholder(df: pl.DataFrame) -> bool:
    if df.is_empty():
        return False
    if "source" in df.columns:
        source_values = {str(value) for value in df["source"].drop_nulls().to_list()}
        if "bootstrap_gold_health" in source_values or source_values == {"global_default"}:
            return True
    if "gate_version" in df.columns:
        gate_versions = {str(value) for value in df["gate_version"].drop_nulls().to_list()}
        if any(version.startswith("bootstrap.") for version in gate_versions):
            return True
    if "fallback_level" in df.columns:
        fallback_values = {str(value) for value in df["fallback_level"].drop_nulls().to_list()}
        if "BOOTSTRAP_CONSERVATIVE" in fallback_values or fallback_values == {"GLOBAL_DEFAULT"}:
            return True
    return False


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
    violations = [f"缺少字段：{column}" for column in sorted(required - set(market.columns))]
    if violations:
        return violations
    invalid_rows = market.filter(
        (pl.col("open") <= 0)
        | (pl.col("close") <= 0)
        | (pl.col("high") < pl.col("low"))
        | (pl.col("volume") < 0)
    ).height
    if invalid_rows:
        violations.append(f"OHLCV 无效行数：{invalid_rows}")
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
    if not rows:
        return pl.DataFrame()
    spread = pl.DataFrame(rows)
    if "ts" not in spread.columns:
        return spread.sort("symbol") if "symbol" in spread.columns else spread
    return (
        spread.with_columns(
            pl.col("ts")
            .map_elements(_coerce_timestamp, return_dtype=pl.Datetime(time_zone="UTC"))
            .alias("ts")
        )
        .sort("ts")
        .group_by(["symbol", "channel"], maintain_order=True)
        .tail(1)
        .sort(["symbol", "channel"])
    )


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
    return _snapshot_latest_timestamp(_dataset_snapshot(lake_root, "okx_public_ws"))


def _latest_collector_health(health: pl.DataFrame) -> dict[str, Any] | None:
    if health.is_empty():
        return None
    sort_column = "updated_at" if "updated_at" in health.columns else health.columns[0]
    return health.sort(sort_column).tail(1).to_dicts()[0]


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None


def readonly_private_status(lake_root: str | Path) -> str:
    fills = _dataset_snapshot(lake_root, "okx_private_readonly_fills")
    bills = _dataset_snapshot(lake_root, "okx_private_readonly_bills")
    if fills.rows == 0 and bills.rows == 0:
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


def _latest_features_table(features: pl.DataFrame) -> pl.DataFrame:
    required = {"symbol", "ts", "feature_name", "value"}
    if features.is_empty() or not required.issubset(features.columns):
        return pl.DataFrame()
    normalized = _normalize_optional_time(features, "ts")
    latest_ts = normalized.group_by("symbol").agg(pl.col("ts").max().alias("ts"))
    return normalized.join(latest_ts, on=["symbol", "ts"], how="inner").sort(
        ["symbol", "feature_name"]
    )


def _top_feature_anomalies(anomalies: pl.DataFrame) -> pl.DataFrame:
    if anomalies.is_empty() or "anomaly_type" not in anomalies.columns:
        return pl.DataFrame()
    count_expr = (
        pl.col("anomaly_count").sum().alias("anomaly_count")
        if "anomaly_count" in anomalies.columns
        else pl.len().alias("anomaly_count")
    )
    return (
        anomalies.group_by(["anomaly_type", "severity"])
        .agg(count_expr)
        .sort("anomaly_count", descending=True)
        .head(20)
    )


def _stale_dataset_rows(lake_root: str | Path) -> pl.DataFrame:
    rows = []
    for name in sorted(DATASET_PATHS):
        path = dataset_path_for(lake_root, name)
        snapshot = _dataset_snapshot(lake_root, name)
        freshness = snapshot.freshness
        status = freshness["freshness_status"]
        if snapshot.rows == 0:
            status = _empty_dataset_status(name)
        if snapshot.warning or status in {"missing", "unknown", "stale"} or snapshot.rows == 0:
            rows.append(
                {
                    "dataset": name,
                    "rows": snapshot.rows,
                    "status": status,
                    "path": str(path),
                    "freshness_seconds": freshness["freshness_seconds"],
                    "timestamp_column": freshness["timestamp_column"] or "",
                    "latest_timestamp": freshness["latest_timestamp"] or "",
                    "warning": snapshot.warning or "",
                }
            )
    return pl.DataFrame(rows)


def _empty_dataset_status(dataset_name: str) -> str:
    if dataset_name == "alpha_evidence":
        return "research evidence not generated yet"
    if dataset_name == "feature_value":
        return "features not published yet"
    if dataset_name == "decision_audit":
        return "legacy optional"
    return "missing"


def _canonical_dataset_name(dataset_name: str) -> str:
    if dataset_name in DATASET_PATHS:
        return dataset_name
    normalized = dataset_name.replace("\\", "/")
    for name, relative_path in DATASET_PATHS.items():
        if str(relative_path).replace("\\", "/") == normalized:
            return name
    return dataset_name


def _coerce_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, date):
        return datetime.combine(value, time.min, tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        try:
            if len(text) == 10:
                parsed_date = date.fromisoformat(text)
                return datetime.combine(parsed_date, time.min, tzinfo=UTC)
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None

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
        return "未知"
    lag = datetime.now(UTC) - timestamp
    if lag < timedelta(hours=2):
        return "新鲜"
    if lag < timedelta(hours=24):
        return "延迟"
    return "过期"


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
