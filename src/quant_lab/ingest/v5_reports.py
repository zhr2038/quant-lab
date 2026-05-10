import json
import logging
import re
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import LakeConfig, read_parquet_dataset, write_parquet_dataset

logger = logging.getLogger(__name__)

DAILY_COST_STATS_PATTERN = re.compile(
    r"daily_cost_stats_(?P<day>\d{4}-\d{2}-\d{2}|\d{8})\.json$"
)


class V5ReportInspection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reports_dir: str
    has_alpha_snapshot: bool
    decision_audit_count: int = Field(ge=0)
    summary_count: int = Field(ge=0)
    cost_stats_file_count: int = Field(ge=0)
    latest_cost_stats_day: date | None
    latest_cost_stats_path: str | None
    warnings: list[str] = Field(default_factory=list)


class PublishResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reports_dir: str
    lake_root: str
    inspection: V5ReportInspection
    bronze_v5_reports_rows: int = Field(ge=0)
    decision_audit_rows: int = Field(ge=0)
    strategy_run_rows: int = Field(ge=0)
    cost_bucket_daily_rows: int = Field(ge=0)
    datasets: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


def inspect_v5_reports(reports_dir: str | Path) -> V5ReportInspection:
    root = _require_reports_dir(reports_dir)
    warnings: list[str] = []

    alpha_snapshot_path = root / "alpha_snapshot.json"
    has_alpha_snapshot = alpha_snapshot_path.is_file()
    if has_alpha_snapshot:
        _load_json_file(alpha_snapshot_path, warnings=warnings)

    decision_audit_paths = _decision_audit_paths(root)
    summary_paths = _summary_paths(root)
    cost_stats_paths = _cost_stats_paths(root)

    for path in [*decision_audit_paths, *summary_paths]:
        _load_json_file(path, warnings=warnings)

    latest_cost_stats_day, latest_cost_stats_path = _latest_cost_stats(cost_stats_paths, warnings)

    return V5ReportInspection(
        reports_dir=str(root),
        has_alpha_snapshot=has_alpha_snapshot,
        decision_audit_count=len(decision_audit_paths),
        summary_count=len(summary_paths),
        cost_stats_file_count=len(cost_stats_paths),
        latest_cost_stats_day=latest_cost_stats_day,
        latest_cost_stats_path=str(latest_cost_stats_path) if latest_cost_stats_path else None,
        warnings=warnings,
    )


def load_latest_v5_cost_stats(reports_dir: str | Path) -> dict[str, Any] | None:
    root = _require_reports_dir(reports_dir)
    _, latest_path = _latest_cost_stats(_cost_stats_paths(root), warnings=[])
    if latest_path is None:
        return None

    payload = _load_json_file(latest_path, warnings=None)
    if payload is None:
        return None
    if not isinstance(payload, dict):
        logger.warning(
            "Latest V5 cost stats payload is not a JSON object",
            extra={"path": str(latest_path), "payload_type": type(payload).__name__},
        )
        return None
    return payload


def iter_v5_decision_audits(reports_dir: str | Path) -> Iterator[dict[str, Any]]:
    root = _require_reports_dir(reports_dir)
    for path in _decision_audit_paths(root):
        payload = _load_json_file(path, warnings=None)
        if payload is None:
            continue
        yield _json_record(path, raw=payload)


def iter_v5_run_summaries(reports_dir: str | Path) -> Iterator[dict[str, Any]]:
    root = _require_reports_dir(reports_dir)
    for path in _summary_paths(root):
        payload = _load_json_file(path, warnings=None)
        if payload is None:
            continue
        yield _json_record(path, raw=payload)


def publish_v5_reports_to_lake(reports_dir: str | Path, lake_root: str | Path) -> PublishResult:
    config = LakeConfig(lake_root=Path(lake_root))
    root = _require_reports_dir(reports_dir)
    inspection = inspect_v5_reports(root)
    ingest_ts = datetime.now(UTC).isoformat()

    datasets = {
        "bronze_v5_reports": config.lake_root / "bronze" / "v5_reports",
        "decision_audit": config.lake_root / "silver" / "decision_audit",
        "strategy_run": config.lake_root / "silver" / "strategy_run",
        "cost_bucket_daily": config.lake_root / "gold" / "cost_bucket_daily",
    }

    bronze_rows = _publish_dataset(
        datasets["bronze_v5_reports"],
        _inspection_frame(inspection, ingest_ts=ingest_ts, created_by=config.created_by),
        key_columns=["reports_dir"],
    )
    decision_audit_rows = _publish_dataset(
        datasets["decision_audit"],
        _json_records_frame(iter_v5_decision_audits(root), ingest_ts=ingest_ts),
        key_columns=["source_path", "run_id"],
    )
    strategy_run_rows = _publish_dataset(
        datasets["strategy_run"],
        _json_records_frame(iter_v5_run_summaries(root), ingest_ts=ingest_ts),
        key_columns=["source_path", "run_id"],
    )
    cost_bucket_daily_rows = _publish_dataset(
        datasets["cost_bucket_daily"],
        _latest_cost_stats_frame(root, inspection=inspection, ingest_ts=ingest_ts),
        key_columns=["cost_day", "source_path", "bucket_index"],
    )

    return PublishResult(
        reports_dir=str(root),
        lake_root=str(config.lake_root),
        inspection=inspection,
        bronze_v5_reports_rows=bronze_rows,
        decision_audit_rows=decision_audit_rows,
        strategy_run_rows=strategy_run_rows,
        cost_bucket_daily_rows=cost_bucket_daily_rows,
        datasets={name: str(path) for name, path in datasets.items()},
        warnings=inspection.warnings,
    )


def _require_reports_dir(reports_dir: str | Path) -> Path:
    root = Path(reports_dir)
    if not root.exists():
        raise FileNotFoundError(f"V5 reports directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"V5 reports path is not a directory: {root}")
    return root


def _decision_audit_paths(root: Path) -> list[Path]:
    return sorted((root / "runs").glob("*/decision_audit.json"))


def _summary_paths(root: Path) -> list[Path]:
    return sorted((root / "runs").glob("*/summary.json"))


def _cost_stats_paths(root: Path) -> list[Path]:
    return sorted((root / "cost_stats").glob("daily_cost_stats_*.json"))


def _latest_cost_stats(
    paths: list[Path], warnings: list[str] | None
) -> tuple[date | None, Path | None]:
    latest_day: date | None = None
    latest_path: Path | None = None

    for path in paths:
        payload = _load_json_file(path, warnings=warnings)
        parsed_day = _cost_stats_day(path, payload=payload, warnings=warnings)
        if parsed_day is None:
            continue
        if latest_day is None or parsed_day > latest_day:
            latest_day = parsed_day
            latest_path = path

    return latest_day, latest_path


def _cost_stats_day(path: Path, payload: Any, warnings: list[str] | None) -> date | None:
    match = DAILY_COST_STATS_PATTERN.match(path.name)
    if match:
        return _parse_cost_stats_day(match.group("day"), path=path, warnings=warnings)

    if not isinstance(payload, dict):
        _warn(
            f"Could not derive cost stats day from non-object JSON payload: {path}",
            warnings=warnings,
        )
        return None

    raw_day = payload.get("day") or payload.get("date")
    if not raw_day:
        _warn(f"Cost stats file has no day/date field: {path}", warnings=warnings)
        return None
    return _parse_cost_stats_day(str(raw_day), path=path, warnings=warnings)


def _parse_cost_stats_day(raw_day: str, path: Path, warnings: list[str] | None) -> date | None:
    try:
        if re.fullmatch(r"\d{8}", raw_day):
            return datetime.strptime(raw_day, "%Y%m%d").date()
        return date.fromisoformat(raw_day)
    except ValueError:
        _warn(f"Cost stats day is not parseable for {path}: {raw_day}", warnings=warnings)
        return None


def _load_json_file(path: Path, warnings: list[str] | None) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _warn(f"Could not read JSON file {path}: {exc}", warnings=warnings)
    except json.JSONDecodeError as exc:
        _warn(f"Invalid JSON in {path}: {exc.msg}", warnings=warnings)
    return None


def _json_record(path: Path, raw: Any) -> dict[str, Any]:
    return {
        "run_id": _run_id_from_path(path),
        "source_path": str(path),
        "loaded_at": datetime.now(UTC),
        "raw": raw,
    }


def _run_id_from_path(path: Path) -> str | None:
    parent = path.parent
    if parent.name and parent.parent.name == "runs":
        return parent.name
    return None


def _warn(message: str, warnings: list[str] | None) -> None:
    logger.warning(message)
    if warnings is not None:
        warnings.append(message)


def _publish_dataset(dataset_path: Path, new_df: pl.DataFrame, key_columns: list[str]) -> int:
    existing_df = read_parquet_dataset(dataset_path)
    frames = [frame for frame in [existing_df, new_df] if not frame.is_empty()]
    if frames:
        combined = pl.concat(frames, how="diagonal_relaxed")
        existing_keys = [column for column in key_columns if column in combined.columns]
        if existing_keys:
            combined = combined.unique(subset=existing_keys, keep="last", maintain_order=True)
    else:
        combined = new_df

    write_parquet_dataset(combined, dataset_path)
    return combined.height


def _inspection_frame(
    inspection: V5ReportInspection,
    ingest_ts: str,
    created_by: str,
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "reports_dir": inspection.reports_dir,
                "ingest_ts": ingest_ts,
                "created_by": created_by,
                "has_alpha_snapshot": inspection.has_alpha_snapshot,
                "decision_audit_count": inspection.decision_audit_count,
                "summary_count": inspection.summary_count,
                "cost_stats_file_count": inspection.cost_stats_file_count,
                "latest_cost_stats_day": _date_to_string(inspection.latest_cost_stats_day),
                "latest_cost_stats_path": inspection.latest_cost_stats_path,
                "warnings_json": _json_dumps(inspection.warnings),
            }
        ],
        schema={
            "reports_dir": pl.Utf8,
            "ingest_ts": pl.Utf8,
            "created_by": pl.Utf8,
            "has_alpha_snapshot": pl.Boolean,
            "decision_audit_count": pl.Int64,
            "summary_count": pl.Int64,
            "cost_stats_file_count": pl.Int64,
            "latest_cost_stats_day": pl.Utf8,
            "latest_cost_stats_path": pl.Utf8,
            "warnings_json": pl.Utf8,
        },
        orient="row",
    )


def _json_records_frame(records: Iterator[dict[str, Any]], ingest_ts: str) -> pl.DataFrame:
    rows = [
        {
            "run_id": record["run_id"],
            "source_path": record["source_path"],
            "ingest_ts": ingest_ts,
            "raw_json": _json_dumps(record["raw"]),
        }
        for record in records
    ]
    return pl.DataFrame(
        rows,
        schema={
            "run_id": pl.Utf8,
            "source_path": pl.Utf8,
            "ingest_ts": pl.Utf8,
            "raw_json": pl.Utf8,
        },
        orient="row",
    )


def _latest_cost_stats_frame(
    reports_dir: Path,
    inspection: V5ReportInspection,
    ingest_ts: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    payload = load_latest_v5_cost_stats(reports_dir)
    has_latest_cost_stats = (
        payload is not None
        and inspection.latest_cost_stats_day
        and inspection.latest_cost_stats_path
    )
    if has_latest_cost_stats:
        buckets = payload.get("buckets")
        if isinstance(buckets, list) and buckets:
            rows = [
                _cost_bucket_row(
                    bucket=bucket,
                    bucket_index=bucket_index,
                    source_path=inspection.latest_cost_stats_path,
                    cost_day=inspection.latest_cost_stats_day,
                    ingest_ts=ingest_ts,
                )
                for bucket_index, bucket in enumerate(buckets)
            ]
        else:
            rows = [
                _cost_bucket_row(
                    bucket=payload,
                    bucket_index=0,
                    source_path=inspection.latest_cost_stats_path,
                    cost_day=inspection.latest_cost_stats_day,
                    ingest_ts=ingest_ts,
                )
            ]

    return pl.DataFrame(
        rows,
        schema={
            "cost_day": pl.Utf8,
            "bucket_index": pl.Int64,
            "source_path": pl.Utf8,
            "ingest_ts": pl.Utf8,
            "bucket_id": pl.Utf8,
            "symbol": pl.Utf8,
            "regime": pl.Utf8,
            "cost_bps": pl.Float64,
            "raw_json": pl.Utf8,
        },
        orient="row",
    )


def _cost_bucket_row(
    bucket: Any,
    bucket_index: int,
    source_path: str,
    cost_day: date,
    ingest_ts: str,
) -> dict[str, Any]:
    bucket_payload = bucket if isinstance(bucket, dict) else {"value": bucket}
    return {
        "cost_day": _date_to_string(cost_day),
        "bucket_index": bucket_index,
        "source_path": source_path,
        "ingest_ts": ingest_ts,
        "bucket_id": _optional_string(bucket_payload.get("bucket_id")),
        "symbol": _optional_string(bucket_payload.get("symbol")),
        "regime": _optional_string(bucket_payload.get("regime")),
        "cost_bps": _optional_float(bucket_payload.get("cost_bps")),
        "raw_json": _json_dumps(bucket_payload),
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _date_to_string(value: date | None) -> str | None:
    return value.isoformat() if value else None


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
