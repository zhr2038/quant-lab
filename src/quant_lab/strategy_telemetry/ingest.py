from __future__ import annotations

import csv
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.data.lake import read_parquet_dataset, upsert_parquet_dataset
from quant_lab.strategy_telemetry.bundle import (
    compute_sha256,
    inspect_v5_bundle,
    parse_bundle_ts,
    safe_extract_v5_bundle,
    validate_v5_bundle,
)
from quant_lab.strategy_telemetry.models import (
    BundleLimits,
    V5BundleIngestResult,
    V5InboxIngestResult,
    utc_now,
)
from quant_lab.strategy_telemetry.sanitize import (
    redact_extracted_bundle,
    redact_json_like,
    safe_json_dumps,
    scan_for_secrets,
)

SCHEMA_VERSION = "v5-telemetry-v0.1"

BRONZE_DATASETS = {
    "bundle_manifest": Path("bronze/strategy_telemetry/v5/bundle_manifest"),
    "secret_scan": Path("bronze/strategy_telemetry/v5/secret_scan"),
    "raw_file_index": Path("bronze/strategy_telemetry/v5/raw_file_index"),
}

SILVER_DATASETS = {
    "v5_run_summary": Path("silver/v5_run_summary"),
    "v5_decision_audit": Path("silver/v5_decision_audit"),
    "v5_equity_point": Path("silver/v5_equity_point"),
    "v5_trade_event": Path("silver/v5_trade_event"),
    "v5_roundtrip": Path("silver/v5_roundtrip"),
    "v5_router_decision": Path("silver/v5_router_decision"),
    "v5_open_position": Path("silver/v5_open_position"),
    "v5_state_snapshot": Path("silver/v5_state_snapshot"),
    "v5_issue": Path("silver/v5_issue"),
    "v5_config_audit": Path("silver/v5_config_audit"),
    "v5_high_score_blocked_target": Path("silver/v5_high_score_blocked_target"),
    "v5_high_score_blocked_outcome": Path("silver/v5_high_score_blocked_outcome"),
    "v5_skipped_candidate_outcome": Path("silver/v5_skipped_candidate_outcome"),
    "v5_shadow_outcome": Path("silver/v5_shadow_outcome"),
    "v5_probe_diagnostic": Path("silver/v5_probe_diagnostic"),
    "v5_quant_lab_usage": Path("silver/v5_quant_lab_usage"),
    "v5_quant_lab_request": Path("silver/v5_quant_lab_request"),
    "v5_quant_lab_compliance": Path("silver/v5_quant_lab_compliance"),
    "v5_quant_lab_cost_usage": Path("silver/v5_quant_lab_cost_usage"),
    "v5_quant_lab_fallback": Path("silver/v5_quant_lab_fallback"),
}


def archive_v5_bundle(
    bundle_path: Path,
    restricted_archive_dir: Path,
    redacted_archive_dir: Path,
    bundle_sha256: str,
    bundle_day: str,
) -> tuple[Path, Path]:
    restricted_root = Path(restricted_archive_dir) / bundle_day / bundle_sha256
    redacted_root = Path(redacted_archive_dir) / bundle_day / bundle_sha256
    restricted_root.mkdir(parents=True, exist_ok=True)
    redacted_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle_path, restricted_root / "raw_bundle.tar.gz")
    return restricted_root, redacted_root


def ingest_v5_bundle(
    bundle_path: Path,
    lake_root: Path,
    restricted_archive_dir: Path,
    redacted_archive_dir: Path,
    strategy: str = "v5",
    limits: BundleLimits | None = None,
) -> V5BundleIngestResult:
    effective_limits = limits or BundleLimits()
    validation = validate_v5_bundle(bundle_path, effective_limits)
    if validation.rejected or validation.sha256 is None:
        empty_scan = scan_for_secrets("")
        return V5BundleIngestResult(
            strategy=strategy,
            bundle_path=str(bundle_path),
            bundle_sha256=validation.sha256 or "",
            bundle_name=Path(bundle_path).name,
            bundle_ts=parse_bundle_ts(Path(bundle_path).name),
            validation=validation,
            secret_scan=empty_scan,
            restricted_archive_path="",
            redacted_archive_path="",
            warnings=validation.reasons,
        )

    bundle_sha256 = validation.sha256
    inspection = inspect_v5_bundle(bundle_path)
    bundle_day = (inspection.bundle_ts or datetime.now(UTC)).date().isoformat()
    restricted_root, redacted_root = archive_v5_bundle(
        bundle_path,
        restricted_archive_dir,
        redacted_archive_dir,
        bundle_sha256,
        bundle_day,
    )

    if _already_ingested(lake_root, bundle_sha256):
        secret_scan = scan_for_secrets("")
        return V5BundleIngestResult(
            strategy=strategy,
            bundle_path=str(bundle_path),
            bundle_sha256=bundle_sha256,
            bundle_name=Path(bundle_path).name,
            bundle_ts=inspection.bundle_ts,
            skipped=True,
            validation=validation,
            secret_scan=secret_scan,
            restricted_archive_path=str(restricted_root),
            redacted_archive_path=str(redacted_root),
            warnings=["bundle sha256 already ingested"],
        )

    ingest_ts = utc_now()
    with tempfile.TemporaryDirectory(prefix="quant_lab_v5_bundle_") as temp_name:
        extracted_dir = Path(temp_name) / "extracted"
        safe_extract_v5_bundle(bundle_path, extracted_dir, effective_limits)
        secret_scan = scan_for_secrets(extracted_dir)
        redaction = redact_extracted_bundle(extracted_dir, redacted_root / "redacted_files")

        metadata = _metadata(
            strategy=strategy,
            bundle_sha256=bundle_sha256,
            bundle_name=Path(bundle_path).name,
            bundle_ts=inspection.bundle_ts,
            ingest_ts=ingest_ts,
        )
        _write_archive_json(
            redacted_root,
            "bundle_manifest.json",
            _manifest_payload(inspection, metadata),
        )
        _write_archive_json(redacted_root, "validation.json", validation.model_dump(mode="json"))
        _write_archive_json(redacted_root, "secret_scan.json", secret_scan.model_dump(mode="json"))
        _write_archive_json(
            redacted_root,
            "redaction_report.json",
            redaction.model_dump(mode="json"),
        )
        _write_archive_json(redacted_root, "provenance.json", metadata)

        bronze_rows = _write_bronze(lake_root, inspection, validation, secret_scan, metadata)
        silver_rows, warnings = _write_silver(lake_root, redacted_root / "redacted_files", metadata)

    from quant_lab.strategy_telemetry.analyze import analyze_v5_telemetry

    analysis = analyze_v5_telemetry(lake_root, date=bundle_day)
    return V5BundleIngestResult(
        strategy=strategy,
        bundle_path=str(bundle_path),
        bundle_sha256=bundle_sha256,
        bundle_name=Path(bundle_path).name,
        bundle_ts=inspection.bundle_ts,
        validation=validation,
        secret_scan=secret_scan,
        restricted_archive_path=str(restricted_root),
        redacted_archive_path=str(redacted_root),
        bronze_rows=bronze_rows,
        silver_rows=silver_rows,
        gold_rows={"strategy_health_daily": 1 if analysis else 0},
        warnings=warnings,
    )


def ingest_v5_inbox(
    inbox_dir: Path,
    lake_root: Path,
    restricted_archive_dir: Path,
    redacted_archive_dir: Path,
    strategy: str = "v5",
    limits: BundleLimits | None = None,
) -> V5InboxIngestResult:
    processed: list[V5BundleIngestResult] = []
    skipped: list[str] = []
    for bundle_path in sorted(Path(inbox_dir).glob("v5_live_followup_bundle_*.tar.gz")):
        sha256 = compute_sha256(bundle_path)
        if _already_ingested(lake_root, sha256):
            skipped.append(str(bundle_path))
            continue
        processed.append(
            ingest_v5_bundle(
                bundle_path=bundle_path,
                lake_root=lake_root,
                restricted_archive_dir=restricted_archive_dir,
                redacted_archive_dir=redacted_archive_dir,
                strategy=strategy,
                limits=limits,
            )
        )
    return V5InboxIngestResult(
        strategy=strategy,
        inbox_dir=str(inbox_dir),
        processed=processed,
        skipped_files=skipped,
    )


def _write_bronze(
    lake_root: Path,
    inspection,
    validation,
    secret_scan,
    metadata: dict[str, Any],
) -> dict[str, int]:
    manifest_row = {
        **metadata,
        "file_count": inspection.file_count,
        "total_uncompressed_size_bytes": inspection.total_uncompressed_size_bytes,
        "detected_files_json": json.dumps(inspection.detected_files, sort_keys=True),
    }
    secret_row = {
        **metadata,
        "scanned_files": secret_scan.scanned_files,
        "high_severity_count": secret_scan.high_severity_count,
        "medium_severity_count": secret_scan.medium_severity_count,
        "low_severity_count": secret_scan.low_severity_count,
        "redaction_required": secret_scan.redaction_required,
        "findings_json": secret_scan.model_dump_json(),
    }
    file_rows = [
        {
            **metadata,
            "source_path_inside_bundle": path,
            "detected": True,
        }
        for path in inspection.detected_files
    ]
    return {
        "bundle_manifest": _upsert_rows(
            lake_root / BRONZE_DATASETS["bundle_manifest"],
            [manifest_row],
            ["bundle_sha256"],
        ),
        "secret_scan": _upsert_rows(
            lake_root / BRONZE_DATASETS["secret_scan"],
            [secret_row],
            ["bundle_sha256"],
        ),
        "raw_file_index": _upsert_rows(
            lake_root / BRONZE_DATASETS["raw_file_index"],
            file_rows,
            ["bundle_sha256", "source_path_inside_bundle"],
        ),
    }


def _write_silver(
    lake_root: Path,
    redacted_files_dir: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, int], list[str]]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SILVER_DATASETS}
    warnings: list[str] = []
    for file_path in sorted(path for path in redacted_files_dir.rglob("*") if path.is_file()):
        relative = file_path.relative_to(redacted_files_dir).as_posix()
        try:
            _append_file_rows(rows, file_path, relative, metadata)
        except Exception as exc:
            warnings.append(f"failed to parse {relative}: {exc}")
            rows["v5_issue"].append(
                _base_row(metadata, relative, None, 0)
                | {
                    "severity": "medium",
                    "issue_type": "parse_error",
                    "message": str(exc),
                    "raw_payload_json": "{}",
                }
            )
    counts = {
        name: _upsert_rows(
            lake_root / dataset,
            dataset_rows,
            ["strategy", "bundle_sha256", "source_path_inside_bundle", "row_index"],
        )
        for name, dataset in SILVER_DATASETS.items()
        if (dataset_rows := rows[name])
    }
    return counts, warnings


def _append_file_rows(
    rows: dict[str, list[dict[str, Any]]],
    file_path: Path,
    relative: str,
    metadata: dict[str, Any],
) -> None:
    logical = _logical_bundle_path(relative)
    run_id = run_id_from_path(logical)
    if logical.endswith("/summary.json") or logical == "summaries/window_summary.json":
        payload = _read_json(file_path)
        rows["v5_run_summary"].append(
            _json_row(metadata, relative, payload, run_id)
        )
        return
    if logical.endswith("/decision_audit.json"):
        payload = _read_json(file_path)
        rows["v5_decision_audit"].append(
            _json_row(metadata, relative, payload, run_id)
        )
        return
    if logical.endswith("/equity.jsonl"):
        rows["v5_equity_point"].extend(_jsonl_rows(metadata, relative, file_path))
        return
    if logical == "raw/quant_lab/quant_lab_usage.jsonl":
        rows["v5_quant_lab_usage"].extend(_jsonl_rows(metadata, relative, file_path))
        return
    if logical == "raw/quant_lab/quant_lab_requests.jsonl":
        rows["v5_quant_lab_request"].extend(_jsonl_rows(metadata, relative, file_path))
        return
    if logical.endswith("/trades.csv"):
        rows["v5_trade_event"].extend(_csv_rows(metadata, relative, file_path))
        return
    if logical.startswith("raw/state/") and logical.endswith(".json"):
        state_type = Path(logical).stem
        payload = _read_json(file_path)
        rows["v5_state_snapshot"].append(
            _json_row(metadata, relative, payload, None)
            | {
                "state_type": state_type,
                "ok": _json_bool(payload, "ok"),
                "enabled": _json_bool(payload, "enabled"),
                "level": str(
                    payload.get("current_level")
                    or payload.get("level")
                    or payload.get("risk_level")
                    or ""
                ),
            }
        )
        return
    if logical == "summaries/issues_to_fix.json":
        rows["v5_issue"].extend(_issue_rows(metadata, relative, _read_json(file_path)))
        return
    csv_mapping = {
        "summaries/router_decisions.csv": "v5_router_decision",
        "summaries/trades_roundtrips.csv": "v5_roundtrip",
        "summaries/open_positions.csv": "v5_open_position",
        "summaries/config_runtime_consumption_audit.csv": "v5_config_audit",
        "summaries/high_score_blocked_targets.csv": "v5_high_score_blocked_target",
        "summaries/skipped_candidate_maturity_audit.csv": "v5_skipped_candidate_outcome",
        "summaries/probe_diagnostics.csv": "v5_probe_diagnostic",
        "summaries/quant_lab_compliance.csv": "v5_quant_lab_compliance",
        "summaries/quant_lab_cost_usage.csv": "v5_quant_lab_cost_usage",
        "summaries/quant_lab_fallbacks.csv": "v5_quant_lab_fallback",
    }
    if logical.startswith("summaries/high_score_blocked_outcomes"):
        rows["v5_high_score_blocked_outcome"].extend(_csv_rows(metadata, relative, file_path))
    elif logical.startswith("summaries/alt_impulse_shadow"):
        rows["v5_shadow_outcome"].extend(_csv_rows(metadata, relative, file_path))
    elif logical in csv_mapping:
        rows[csv_mapping[logical]].extend(_csv_rows(metadata, relative, file_path))


def _json_row(
    metadata: dict[str, Any],
    relative: str,
    payload: dict[str, Any],
    run_id: str | None,
    row_index: int = 0,
) -> dict[str, Any]:
    return _base_row(metadata, relative, run_id, row_index) | {
        "raw_payload_json": safe_json_dumps(payload),
    }


def _jsonl_rows(metadata: dict[str, Any], relative: str, file_path: Path) -> list[dict[str, Any]]:
    rows = []
    run_id = run_id_from_path(_logical_bundle_path(relative))
    for index, line in enumerate(file_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        payload = redact_json_like(json.loads(line))
        rows.append(_json_row(metadata, relative, payload, run_id, index))
    return rows


def _csv_rows(metadata: dict[str, Any], relative: str, file_path: Path) -> list[dict[str, Any]]:
    rows = []
    run_id = run_id_from_path(_logical_bundle_path(relative))
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, raw_row in enumerate(reader):
            safe_row = redact_json_like(dict(raw_row))
            rows.append(
                _base_row(metadata, relative, run_id, index)
                | {key: str(value) for key, value in safe_row.items()}
                | {"raw_payload_json": safe_json_dumps(safe_row)}
            )
    return rows


def _issue_rows(
    metadata: dict[str, Any],
    relative: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    issues = payload.get("issues", payload if isinstance(payload, list) else [])
    if isinstance(issues, dict):
        issues = [issues]
    rows = []
    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            issue = {"message": str(issue)}
        rows.append(
            _base_row(metadata, relative, None, index)
            | {
                "severity": str(issue.get("severity") or issue.get("level") or "medium").lower(),
                "issue_type": str(issue.get("type") or issue.get("issue_type") or "unknown"),
                "message": str(issue.get("message") or issue.get("description") or ""),
                "raw_payload_json": safe_json_dumps(issue),
            }
        )
    return rows


def _base_row(
    metadata: dict[str, Any],
    relative: str,
    run_id: str | None,
    row_index: int,
) -> dict[str, Any]:
    return {
        **metadata,
        "source_path_inside_bundle": relative,
        "run_id": run_id,
        "row_index": row_index,
    }


def _metadata(
    strategy: str,
    bundle_sha256: str,
    bundle_name: str,
    bundle_ts: datetime | None,
    ingest_ts: datetime,
) -> dict[str, Any]:
    return {
        "strategy": strategy,
        "bundle_sha256": bundle_sha256,
        "bundle_name": bundle_name,
        "bundle_ts": bundle_ts,
        "ingest_ts": ingest_ts,
        "schema_version": SCHEMA_VERSION,
    }


def _manifest_payload(inspection, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        **metadata,
        "path": inspection.path,
        "file_count": inspection.file_count,
        "detected_files": inspection.detected_files,
        "total_uncompressed_size_bytes": inspection.total_uncompressed_size_bytes,
    }


def _upsert_rows(dataset_path: Path, rows: list[dict[str, Any]], keys: list[str]) -> int:
    if not rows:
        return read_parquet_dataset(dataset_path).height
    df = pl.DataFrame(_json_safe_rows(rows))
    return upsert_parquet_dataset(df, dataset_path, key_columns=keys)


def _json_safe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    safe_rows = []
    for row in rows:
        safe_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            }
        )
    return safe_rows


def _already_ingested(lake_root: Path, bundle_sha256: str) -> bool:
    existing = read_parquet_dataset(lake_root / BRONZE_DATASETS["bundle_manifest"])
    return not existing.is_empty() and "bundle_sha256" in existing.columns and bundle_sha256 in set(
        existing["bundle_sha256"].to_list()
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    safe_payload = redact_json_like(payload)
    return safe_payload if isinstance(safe_payload, dict) else {"value": safe_payload}


def _json_bool(payload: dict[str, Any], key: str) -> bool | None:
    value = payload.get(key)
    return value if isinstance(value, bool) else None


def _write_archive_json(root: Path, name: str, payload: dict[str, Any]) -> None:
    (root / name).write_text(
        json.dumps(
            redact_json_like(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )


def run_id_from_path(relative: str) -> str | None:
    parts = relative.split("/")
    if "recent_runs" in parts:
        index = parts.index("recent_runs")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _logical_bundle_path(relative: str) -> str:
    parts = [part for part in relative.split("/") if part]
    if len(parts) > 1 and parts[0].startswith("v5_live_followup_bundle_"):
        return "/".join(parts[1:])
    return relative
