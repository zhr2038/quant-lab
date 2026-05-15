from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.contracts.v5_quant_lab import V5_TELEMETRY_DATASET_SCHEMA_VERSION
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
from quant_lab.symbols import normalize_symbol

SCHEMA_VERSION = V5_TELEMETRY_DATASET_SCHEMA_VERSION

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
    "v5_candidate_event": Path("silver/v5_candidate_event"),
}

GOLD_DATASETS = {
    "v5_candidate_label": Path("gold/v5_candidate_label"),
    "v5_candidate_quality_daily": Path("gold/v5_candidate_quality_daily"),
}

QUANT_LAB_USAGE_PATHS = {
    "raw/reports/quant_lab_usage.jsonl",
    "raw/quant_lab/quant_lab_usage.jsonl",
    "reports/quant_lab_usage.jsonl",
}
QUANT_LAB_REQUEST_PATHS = {
    "raw/reports/quant_lab_requests.jsonl",
    "raw/quant_lab/quant_lab_requests.jsonl",
    "reports/quant_lab_requests.jsonl",
}
EVENT_KEY_DATASETS = {"v5_quant_lab_request", "v5_quant_lab_fallback"}
EVENT_KEY_METADATA_FIELDS = {
    "strategy",
    "bundle_sha256",
    "bundle_name",
    "bundle_ts",
    "ingest_ts",
    "schema_version",
    "source_path_inside_bundle",
    "row_index",
    "source_count",
    "first_seen_bundle_ts",
    "last_seen_bundle_ts",
}
CANDIDATE_EVENT_SCHEMA_VERSION = "v5.candidate_snapshot.v1"
CANDIDATE_LABEL_SCHEMA_VERSION = "v5.candidate_label.v1"
CANDIDATE_LABEL_HORIZONS_HOURS = (4, 8, 12, 24, 48, 72, 120)


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
        candidate_gold_rows = _write_candidate_gold(lake_root)

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
        gold_rows={"strategy_health_daily": 1 if analysis else 0, **candidate_gold_rows},
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
    counts: dict[str, int] = {}
    for name, dataset in SILVER_DATASETS.items():
        dataset_rows = rows[name]
        if not dataset_rows:
            continue
        dataset_path = lake_root / dataset
        if name in EVENT_KEY_DATASETS:
            counts[name] = _upsert_event_rows(dataset_path, dataset_rows)
        elif name == "v5_candidate_event":
            counts[name] = _upsert_rows(
                dataset_path,
                dataset_rows,
                ["strategy", "candidate_id"],
            )
        else:
            counts[name] = _upsert_rows(
                dataset_path,
                dataset_rows,
                ["strategy", "bundle_sha256", "source_path_inside_bundle", "row_index"],
            )
    return counts, warnings


def _write_candidate_gold(lake_root: Path) -> dict[str, int]:
    candidates = read_parquet_dataset(lake_root / SILVER_DATASETS["v5_candidate_event"])
    if candidates.is_empty():
        return {}
    market_bars = read_parquet_dataset(lake_root / "silver/market_bar")
    label_rows = _candidate_label_rows(candidates, market_bars)
    quality_rows = [_candidate_quality_row(candidates, label_rows)]
    counts: dict[str, int] = {}
    if label_rows:
        counts["v5_candidate_label"] = _upsert_rows(
            lake_root / GOLD_DATASETS["v5_candidate_label"],
            label_rows,
            ["strategy", "candidate_id", "horizon_hours"],
        )
    counts["v5_candidate_quality_daily"] = _upsert_rows(
        lake_root / GOLD_DATASETS["v5_candidate_quality_daily"],
        quality_rows,
        ["strategy", "date"],
    )
    return counts


def _candidate_label_rows(
    candidates: pl.DataFrame,
    market_bars: pl.DataFrame,
) -> list[dict[str, Any]]:
    bars_by_symbol = _market_bars_by_symbol(market_bars)
    rows: list[dict[str, Any]] = []
    for candidate in candidates.to_dicts():
        candidate_id = _clean_text(candidate.get("candidate_id"))
        if not candidate_id:
            continue
        symbol = normalize_symbol(
            candidate.get("normalized_symbol") or candidate.get("symbol") or ""
        )
        candidate_ts = _parse_utc_dt(candidate.get("ts_utc") or candidate.get("ts"))
        bars = bars_by_symbol.get(symbol, [])
        cost_bps = _numeric(candidate.get("cost_bps")) or 0.0
        side_multiplier = _candidate_side_multiplier(candidate)
        for horizon in CANDIDATE_LABEL_HORIZONS_HOURS:
            label = _candidate_label_for_horizon(
                bars=bars,
                candidate_ts=candidate_ts,
                horizon_hours=horizon,
                side_multiplier=side_multiplier,
                cost_bps=cost_bps,
            )
            rows.append(
                {
                    "strategy": candidate.get("strategy") or "v5",
                    "candidate_label_schema_version": CANDIDATE_LABEL_SCHEMA_VERSION,
                    "candidate_id": candidate_id,
                    "run_id": candidate.get("run_id"),
                    "ts_utc": candidate.get("ts_utc"),
                    "symbol": symbol,
                    "strategy_candidate": candidate.get("strategy_candidate"),
                    "block_reason": candidate.get("block_reason"),
                    "final_decision": candidate.get("final_decision"),
                    "cost_bps": cost_bps,
                    "cost_source": candidate.get("cost_source"),
                    "horizon_hours": horizon,
                    **label,
                    "source_candidate_bundle_sha256": candidate.get("bundle_sha256"),
                    "source_path_inside_bundle": candidate.get("source_path_inside_bundle"),
                    "created_at": datetime.now(UTC),
                }
            )
    return rows


def _market_bars_by_symbol(market_bars: pl.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if market_bars.is_empty():
        return {}
    required = {"symbol", "ts", "close"}
    if not required.issubset(set(market_bars.columns)):
        return {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in market_bars.to_dicts():
        symbol = normalize_symbol(row.get("symbol"))
        ts = _parse_utc_dt(row.get("ts"))
        close = _numeric(row.get("close"))
        if not symbol or ts is None or close is None:
            continue
        rows.setdefault(symbol, []).append({"ts": ts, "close": close})
    for symbol, values in rows.items():
        rows[symbol] = sorted(values, key=lambda item: item["ts"])
    return rows


def _candidate_label_for_horizon(
    *,
    bars: list[dict[str, Any]],
    candidate_ts: datetime | None,
    horizon_hours: int,
    side_multiplier: float,
    cost_bps: float,
) -> dict[str, Any]:
    if candidate_ts is None or not bars:
        return _empty_candidate_label(horizon_hours, "insufficient_market_bar")
    start = _first_bar_at_or_after(bars, candidate_ts)
    future = _first_bar_at_or_after(bars, candidate_ts + timedelta(hours=horizon_hours))
    if start is None or future is None:
        return _empty_candidate_label(horizon_hours, "insufficient_market_bar")
    start_close = float(start["close"])
    future_close = float(future["close"])
    if start_close <= 0:
        return _empty_candidate_label(horizon_hours, "invalid_start_price")
    horizon_ts = candidate_ts + timedelta(hours=horizon_hours)
    path = [bar for bar in bars if candidate_ts <= bar["ts"] <= horizon_ts]
    gross_bps = ((future_close / start_close) - 1.0) * 10_000.0 * side_multiplier
    path_returns = [
        ((float(bar["close"]) / start_close) - 1.0) * 10_000.0 * side_multiplier
        for bar in path
    ]
    mfe_bps = max(path_returns) if path_returns else gross_bps
    mae_bps = min(path_returns) if path_returns else gross_bps
    net_bps = gross_bps - float(cost_bps or 0.0)
    return {
        "decision_ts": start["ts"],
        "label_ts": future["ts"],
        "gross_bps": gross_bps,
        "net_bps_after_cost": net_bps,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "win": net_bps > 0,
        "label_status": "complete",
    }


def _empty_candidate_label(horizon_hours: int, status: str) -> dict[str, Any]:
    return {
        "decision_ts": None,
        "label_ts": None,
        "gross_bps": None,
        "net_bps_after_cost": None,
        "mfe_bps": None,
        "mae_bps": None,
        "win": None,
        "label_status": status,
    }


def _first_bar_at_or_after(
    bars: list[dict[str, Any]],
    ts: datetime,
) -> dict[str, Any] | None:
    for bar in bars:
        if bar["ts"] >= ts:
            return bar
    return None


def _candidate_side_multiplier(candidate: dict[str, Any]) -> float:
    side = _clean_text(candidate.get("alpha6_side") or "").lower()
    decision = _clean_text(candidate.get("final_decision") or "").lower()
    if side == "sell" or "sell" in decision or "close" in decision:
        return -1.0
    return 1.0


def _parse_utc_dt(value: Any) -> datetime | None:
    normalized = _normalize_event_time(value)
    if not normalized:
        return None
    try:
        parsed = datetime.fromisoformat(str(normalized).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(UTC)


def _candidate_quality_row(
    candidates: pl.DataFrame,
    label_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = candidates.to_dicts()
    total = len(rows)
    completed = sum(1 for row in label_rows if row.get("label_status") == "complete")
    expected_labels = total * len(CANDIDATE_LABEL_HORIZONS_HOURS)
    required_feature_fields = [
        "final_score",
        "f4_volume_expansion",
        "f5_rsi_trend_confirm",
        "alpha6_score",
        "expected_edge_bps",
    ]
    feature_values = 0
    for row in rows:
        feature_values += sum(1 for field in required_feature_fields if _clean_text(row.get(field)))
    feature_total = max(total * len(required_feature_fields), 1)
    cost_covered = sum(1 for row in rows if _clean_text(row.get("cost_source")))
    bundle_ts = max(
        (_parse_utc_dt(row.get("bundle_ts")) for row in rows),
        default=None,
        key=lambda value: value or datetime.min.replace(tzinfo=UTC),
    )
    return {
        "strategy": "v5",
        "date": (bundle_ts or datetime.now(UTC)).date().isoformat(),
        "candidate_event_rows": total,
        "candidate_run_count": len(
            {_clean_text(row.get("run_id")) for row in rows if _clean_text(row.get("run_id"))}
        ),
        "feature_completeness": feature_values / feature_total,
        "label_completeness": completed / max(expected_labels, 1),
        "cost_source_coverage": cost_covered / max(total, 1),
        "label_horizons_json": json.dumps(list(CANDIDATE_LABEL_HORIZONS_HOURS)),
        "created_at": datetime.now(UTC),
    }


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
    if logical in QUANT_LAB_USAGE_PATHS:
        rows["v5_quant_lab_usage"].extend(_jsonl_rows(metadata, relative, file_path))
        return
    if logical in QUANT_LAB_REQUEST_PATHS:
        request_rows = _enrich_event_rows(
            _jsonl_rows(metadata, relative, file_path),
            default_event_type="request",
        )
        rows["v5_quant_lab_request"].extend(request_rows)
        rows["v5_quant_lab_fallback"].extend(_request_fallback_rows(request_rows))
        return
    if logical.endswith("/trades.csv"):
        rows["v5_trade_event"].extend(_v5_trade_rows(metadata, relative, file_path))
        return
    if logical.endswith("/candidate_snapshot.csv") or logical == "candidate_snapshot.csv":
        rows["v5_candidate_event"].extend(_candidate_event_rows(metadata, relative, file_path))
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
    }
    if logical.startswith("summaries/high_score_blocked_outcomes"):
        rows["v5_high_score_blocked_outcome"].extend(_csv_rows(metadata, relative, file_path))
    elif logical.startswith("summaries/alt_impulse_shadow"):
        rows["v5_shadow_outcome"].extend(_csv_rows(metadata, relative, file_path))
    elif logical == "summaries/quant_lab_fallbacks.csv":
        rows["v5_quant_lab_fallback"].extend(_fallback_csv_rows(metadata, relative, file_path))
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
            safe_row = _normalize_csv_symbol_fields(safe_row)
            rows.append(
                _base_row(metadata, relative, run_id, index)
                | {key: str(value) for key, value in safe_row.items()}
                | {"raw_payload_json": safe_json_dumps(safe_row)}
            )
    return rows


def _v5_trade_rows(
    metadata: dict[str, Any],
    relative: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _csv_rows(metadata, relative, file_path):
        payload = _loads_payload(row.get("raw_payload_json"))
        symbol_value = _clean_text(
            _first_value(
                row,
                payload,
                ["normalized_symbol", "symbol", "inst_id", "instId", "instrument", "pair"],
            )
        )
        normalized_symbol = normalize_symbol(symbol_value) if symbol_value else ""
        price = _numeric(_first_value(row, payload, ["price", "fill_price", "fill_px", "px"]))
        qty = _numeric(
            _first_value(row, payload, ["qty", "quantity", "size", "fill_size", "fill_sz", "sz"])
        )
        notional = _numeric(
            _first_value(row, payload, ["notional_usdt", "notional", "quote_notional"])
        )
        if notional is None and price is not None and qty is not None:
            notional = abs(price * qty)
        fee = _numeric(_first_value(row, payload, ["fee", "commission", "fee_abs"]))
        fee_ccy = _clean_text(
            _first_value(row, payload, ["fee_ccy", "fee_currency", "commission_asset"])
        )
        fee_usdt = _numeric(_first_value(row, payload, ["fee_usdt", "fee_abs_usdt"]))
        if fee_usdt is None:
            fee_usdt = _trade_fee_usdt(
                fee=fee,
                fee_ccy=fee_ccy,
                symbol=normalized_symbol,
                price=price,
            )
        ts_utc = _normalize_event_time(
            _first_value(row, payload, ["ts_utc", "ts", "timestamp", "time", "trade_ts"])
        )
        side = _clean_text(_first_value(row, payload, ["side", "order_side"])).lower()
        action = _clean_text(_first_value(row, payload, ["action", "intent"])).lower()
        rows.append(
            row
            | {
                "strategy_id": _clean_text(
                    _first_value(row, payload, ["strategy_id", "strategyId", "strategy"])
                    or row.get("strategy")
                ),
                "ts_utc": ts_utc,
                "symbol": normalized_symbol or symbol_value,
                "normalized_symbol": normalized_symbol,
                "side": side,
                "action": action,
                "qty": "" if qty is None else str(qty),
                "price": "" if price is None else str(price),
                "notional_usdt": "" if notional is None else str(abs(notional)),
                "fee": "" if fee is None else str(fee),
                "fee_ccy": fee_ccy,
                "fee_usdt": "" if fee_usdt is None else str(abs(fee_usdt)),
                "slippage_usdt": str(_trade_slippage_usdt(row, payload) or ""),
                "order_id": _clean_text(_first_value(row, payload, ["order_id", "ordId"])),
                "trade_id": _clean_text(_first_value(row, payload, ["trade_id", "tradeId"])),
                "raw_payload_json": safe_json_dumps(
                    {
                        **payload,
                        "normalized_symbol": normalized_symbol,
                        "ts_utc": ts_utc,
                        "notional_usdt": notional,
                        "fee_usdt": None if fee_usdt is None else abs(fee_usdt),
                    }
                ),
            }
        )
    return rows


def _candidate_event_rows(
    metadata: dict[str, Any],
    relative: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _csv_rows(metadata, relative, file_path):
        payload = _loads_payload(row.get("raw_payload_json"))
        run_id = _clean_text(_first_value(row, payload, ["run_id"]) or row.get("run_id"))
        symbol_value = _clean_text(_first_value(row, payload, ["symbol", "normalized_symbol"]))
        normalized_symbol = normalize_symbol(symbol_value) if symbol_value else ""
        strategy_candidate = _clean_text(
            _first_value(row, payload, ["strategy_candidate", "strategy_id", "strategy"])
            or "portfolio"
        )
        candidate_id = _clean_text(_first_value(row, payload, ["candidate_id"]))
        if not candidate_id:
            candidate_id = _candidate_id(
                run_id,
                normalized_symbol or symbol_value,
                strategy_candidate,
            )
        ts_utc = _normalize_event_time(_first_value(row, payload, ["ts_utc", "ts", "timestamp"]))
        event = row | {
            "event_type": "candidate_event",
            "candidate_event_schema_version": CANDIDATE_EVENT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "run_id": run_id or row.get("run_id"),
            "ts_utc": ts_utc,
            "symbol": normalized_symbol or symbol_value,
            "normalized_symbol": normalized_symbol,
            "strategy_candidate": strategy_candidate,
            "candidate_quality_key": _candidate_quality_key(
                run_id,
                normalized_symbol,
                strategy_candidate,
            ),
            "raw_payload_json": safe_json_dumps(
                {
                    **payload,
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "ts_utc": ts_utc,
                    "symbol": normalized_symbol or symbol_value,
                    "normalized_symbol": normalized_symbol,
                    "strategy_candidate": strategy_candidate,
                }
            ),
        }
        rows.append(event)
    return rows


def _candidate_id(run_id: str, symbol: str, strategy_candidate: str) -> str:
    material = "|".join(
        [
            str(run_id or "").strip(),
            str(symbol or "").strip().upper(),
            str(strategy_candidate or "portfolio").strip(),
        ]
    )
    return "cand_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _candidate_quality_key(run_id: str, symbol: str, strategy_candidate: str) -> str:
    return "|".join(
        [
            str(run_id or "").strip(),
            str(symbol or "").strip().upper(),
            str(strategy_candidate or "portfolio").strip(),
        ]
    )


def _trade_fee_usdt(
    *,
    fee: float | None,
    fee_ccy: str,
    symbol: str,
    price: float | None,
) -> float | None:
    if fee is None:
        return None
    normalized_ccy = fee_ccy.upper().strip()
    fee_abs = abs(fee)
    if not normalized_ccy or normalized_ccy in {"USDT", "USDC", "USD"}:
        return fee_abs
    base, _, quote = symbol.partition("-")
    if normalized_ccy == quote:
        return fee_abs
    if normalized_ccy == base and price is not None:
        return fee_abs * price
    return fee_abs


def _trade_slippage_usdt(row: dict[str, Any], payload: dict[str, Any]) -> float | None:
    return _numeric(_first_value(row, payload, ["slippage_usdt", "realized_slippage_usdt"]))


def _normalize_csv_symbol_fields(row: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(row)
    for key in ["symbol", "inst_id", "instId", "instrument", "pair"]:
        value = normalized.get(key)
        if value:
            symbol = normalize_symbol(value)
            normalized["symbol"] = symbol
            normalized["normalized_symbol"] = symbol
            break
    return normalized


def _fallback_csv_rows(
    metadata: dict[str, Any],
    relative: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    return [
        _with_event_key(row | {"event_type": "request"}, default_event_type="request")
        for row in _csv_rows(metadata, relative, file_path)
        if _is_fallback_row(row)
    ]


def _request_fallback_rows(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fallback_rows: list[dict[str, Any]] = []
    for row in request_rows:
        if not _is_fallback_row(row):
            continue
        payload = _loads_payload(row.get("raw_payload_json"))
        fallback_rows.append(
            _with_event_key(
                row
                | {
                    "event_type": "request",
                    "actual_fallback": True,
                    "diagnosis": _fallback_diagnosis(row, payload),
                    "degraded_reason": _fallback_diagnosis(row, payload),
                },
                default_event_type="request",
            )
        )
    return fallback_rows


def _enrich_event_rows(
    rows: list[dict[str, Any]],
    *,
    default_event_type: str,
) -> list[dict[str, Any]]:
    return [_with_event_key(row, default_event_type=default_event_type) for row in rows]


def _with_event_key(
    row: dict[str, Any],
    *,
    default_event_type: str | None = None,
) -> dict[str, Any]:
    payload = _loads_payload(row.get("raw_payload_json"))
    fields = _event_key_fields(row, payload, default_event_type=default_event_type)
    enriched = dict(row)
    enriched.update(
        {
            "event_id": fields["event_id"],
            "strategy_id": fields["strategy_id"],
            "run_id": fields["run_id"] or row.get("run_id"),
            "ts_utc": fields["ts_utc"],
            "endpoint": fields["endpoint_path"],
            "endpoint_path": fields["endpoint_path"],
            "event_type": fields["event_type"],
            "status_code": fields["status_code"] or row.get("status_code"),
            "error_type": fields["error_type"],
            "fallback_used": fields["fallback_used"],
            "request_id": fields["request_id"],
            "symbol": fields["symbol"],
            "side": fields["side"],
            "intent": fields["intent"],
            "raw_payload_hash": fields["raw_payload_hash"],
            "event_key_fields_json": safe_json_dumps(fields),
            "event_key": _event_key_from_fields(fields),
        }
    )
    return enriched


def _event_key_fields(
    row: dict[str, Any],
    payload: dict[str, Any],
    *,
    default_event_type: str | None,
) -> dict[str, Any]:
    source_path = _logical_bundle_path(str(row.get("source_path_inside_bundle") or ""))
    strategy_id = _clean_text(
        _first_value(row, payload, ["strategy_id", "strategyId", "strategy"])
        or row.get("strategy")
    )
    event_id = _clean_text(
        _first_value(row, payload, ["event_id", "eventId", "source_event_id"])
    )
    run_id = _first_value(row, payload, ["run_id", "runId", "run"])
    ts_utc = _normalize_event_time(
        _first_value(
            row,
            payload,
            [
                "ts_utc",
                "ts",
                "timestamp",
                "created_at",
                "time",
                "request_ts",
                "event_ts",
            ],
        )
    )
    endpoint_path = _clean_text(
        _first_value(
            row,
            payload,
            ["endpoint", "endpoint_path", "path", "url", "route", "api_path", "request_path"],
        )
    )
    if source_path == "summaries/quant_lab_fallbacks.csv":
        event_type = "request"
    else:
        event_type = _clean_text(
            _first_value(row, payload, ["event_type", "type", "kind"])
            or default_event_type
            or ("request" if endpoint_path else "event")
        ).lower()
    error_type = _clean_text(
        _first_value(row, payload, ["error_type", "exception_type", "error", "exception"])
    )
    fallback_used = _parse_bool(
        _first_value(row, payload, ["fallback_used", "used_fallback", "local_fallback"])
    )
    request_id = _clean_text(
        _first_value(row, payload, ["request_id", "trace_id", "id", "uuid"])
    )
    status_code = _status_code(row, payload)
    symbol_value = _clean_text(
        _first_value(
            row,
            payload,
            ["symbol", "normalized_symbol", "inst_id", "instId", "instrument", "pair"],
        )
    )
    symbol = normalize_symbol(symbol_value) if symbol_value else ""
    fields = {
        "event_id": event_id,
        "strategy_id": strategy_id,
        "run_id": _clean_text(run_id),
        "event_type": event_type,
        "endpoint_path": endpoint_path,
        "ts_utc": ts_utc,
        "status_code": "" if status_code is None else str(status_code),
        "error_type": error_type,
        "request_id": request_id,
        "symbol": symbol,
        "side": _clean_text(_first_value(row, payload, ["side", "order_side"])).lower(),
        "intent": _clean_text(
            _first_value(row, payload, ["intent", "action", "router_intent"])
        ).lower(),
        "fallback_used": fallback_used,
    }
    fields["raw_payload_hash"] = _raw_payload_hash(row, payload, fields)
    return fields


def _event_key_from_fields(fields: dict[str, Any]) -> str:
    event_id = str(fields.get("event_id") or "").strip()
    if event_id:
        stable = {
            "strategy_id": str(fields.get("strategy_id") or "").strip(),
            "event_id": event_id,
        }
    else:
        stable = {
            key: value
            for key, value in fields.items()
            if key not in {"event_id", "fallback_used"}
            and value is not None
            and value != ""
        }
        if (
            stable.get("endpoint_path")
            and stable.get("ts_utc")
            and stable.get("error_type")
        ):
            # Summary fallback CSV rows often omit run_id while raw request rows carry it.
            # Endpoint + event time + concrete error are the stable cross-bundle identity.
            stable.pop("run_id", None)
    rendered = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _raw_payload_hash(
    row: dict[str, Any],
    payload: dict[str, Any],
    fields: dict[str, Any],
) -> str:
    stable_event = {
        key: fields.get(key)
        for key in [
            "event_id",
            "strategy_id",
            "event_type",
            "endpoint_path",
            "ts_utc",
            "status_code",
            "error_type",
            "request_id",
            "symbol",
            "side",
            "intent",
        ]
        if fields.get(key) not in {None, ""}
    }
    has_event_identity = any(
        stable_event.get(key) for key in ["event_id", "endpoint_path", "ts_utc", "request_id"]
    )
    if has_event_identity or (
        stable_event.get("status_code") and stable_event.get("error_type")
    ):
        rendered = json.dumps(stable_event, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return _payload_hash(row, payload)


def _payload_hash(row: dict[str, Any], payload: dict[str, Any]) -> str:
    if payload:
        source: Any = payload
    else:
        source = {
            key: value
            for key, value in row.items()
            if key not in EVENT_KEY_METADATA_FIELDS and key != "event_key"
        }
    rendered = json.dumps(source, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _normalize_event_time(value: Any) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, datetime):
        normalized = value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")
    if isinstance(value, (int, float)):
        timestamp = float(value)
        if timestamp > 10_000_000_000:
            timestamp /= 1000.0
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat().replace("+00:00", "Z")
    rendered = str(value).strip()
    try:
        parsed = datetime.fromisoformat(rendered.replace("Z", "+00:00"))
    except ValueError:
        return rendered
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    rendered = str(value).strip()
    return (
        ""
        if rendered.lower()
        in {"none", "null", "nan", "unknown", "not_observable", "not-observable", "n/a", "na"}
        else rendered
    )


def _is_fallback_row(row: dict[str, Any]) -> bool:
    payload = _loads_payload(row.get("raw_payload_json"))
    if _is_successful_request(row, payload):
        return False
    if _truthy(_first_value(row, payload, ["fallback_used", "used_fallback", "local_fallback"])):
        return True
    status_code = _status_code(row, payload)
    if status_code is not None and status_code >= 500:
        return True
    if _actual_error_type(_first_value(row, payload, ["error_type", "exception_type"])):
        return True
    count = _numeric(_first_value(row, payload, ["count", "fallback_count"]))
    if count == 0:
        return False
    if _has_error_indicator(row, payload):
        return True
    action = _first_value(
        row,
        payload,
        ["fail_policy_action", "fail_policy", "action", "fallback_action"],
    )
    if _action_triggered(action):
        return True
    rendered = " ".join(
        str(_first_value(row, payload, [field]) or "").lower()
        for field in ["fallback_reason", "cost_source", "source", "diagnosis", "message"]
    )
    return "local" in rendered and "fallback" in rendered


def _is_successful_request(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    if _truthy(_first_value(row, payload, ["fallback_used", "used_fallback", "local_fallback"])):
        return False
    status_code = _status_code(row, payload)
    success = _parse_bool(_first_value(row, payload, ["success", "ok", "request_ok"]))
    if status_code == 200 and success is not False:
        return True
    return success is True and (status_code is None or 200 <= status_code < 300)


def _status_code(row: dict[str, Any], payload: dict[str, Any]) -> int | None:
    value = _first_value(row, payload, ["status_code", "http_status", "status"])
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _actual_error_type(value: Any) -> bool:
    if not _nonempty_text(value):
        return False
    normalized = str(value).strip().lower()
    if normalized in {"http_200", "200", "request_not_ok"}:
        return False
    return not (
        normalized.startswith("http_4")
        or normalized in {"400", "401", "403", "404", "409", "422", "429"}
    )


def _has_error_indicator(row: dict[str, Any], payload: dict[str, Any]) -> bool:
    rendered = " ".join(
        str(_first_value(row, payload, [field]) or "").lower()
        for field in [
            "error",
            "message",
            "exception",
            "diagnosis",
            "reason",
            "error_type",
            "exception_type",
        ]
    )
    indicators = [
        "timeout",
        "quantlabtimeout",
        "connection",
        "connect",
        "parse",
        "jsondecode",
        "decode",
    ]
    if "http_200" in rendered or "http 200" in rendered:
        rendered = rendered.replace("request_not_ok", "")
    return any(indicator in rendered for indicator in indicators)


def _fallback_diagnosis(row: dict[str, Any], payload: dict[str, Any]) -> str:
    if _truthy(_first_value(row, payload, ["fallback_used", "used_fallback", "local_fallback"])):
        return "fallback_used"
    status_code = _status_code(row, payload)
    if status_code is not None and status_code >= 500:
        return "http_5xx"
    error_type = _first_value(row, payload, ["error_type", "exception_type"])
    if _actual_error_type(error_type):
        return str(error_type)
    if _has_error_indicator(row, payload):
        return "request_error"
    action = _first_value(
        row,
        payload,
        ["fail_policy_action", "fail_policy", "action", "fallback_action"],
    )
    if _action_triggered(action):
        return "fail_policy_action_triggered"
    return "actual_fallback"


def _loads_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _raw_json_payload(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_json")
    if _empty_value(raw):
        raw = payload.get("raw_json")
    return _loads_payload(raw)


def _first_value(row: dict[str, Any], payload: dict[str, Any], keys: list[str]) -> Any:
    raw_payload: dict[str, Any] | None = None
    for key in keys:
        value = row.get(key)
        if _empty_value(value):
            value = payload.get(key)
        if _empty_value(value):
            if raw_payload is None:
                raw_payload = _raw_json_payload(row, payload)
            value = raw_payload.get(key)
        if not _empty_value(value):
            return value
    return None


def _empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _truthy(value: Any) -> bool:
    parsed = _parse_bool(value)
    return parsed is True


def _parse_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _nonempty_text(value: Any) -> bool:
    if value is None:
        return False
    normalized = str(value).strip().lower()
    return normalized not in {"", "none", "null", "ok", "false", "0"}


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _action_triggered(value: Any) -> bool:
    if not _nonempty_text(value):
        return False
    normalized = str(value).strip().lower()
    if normalized in {"none", "no_action", "allow", "ok", "pass"}:
        return False
    return "fallback" in normalized or "trigger" in normalized or "sell_only" in normalized


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


def _upsert_event_rows(dataset_path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return read_parquet_dataset(dataset_path).height
    existing = read_parquet_dataset(dataset_path)
    combined_rows = existing.to_dicts() if not existing.is_empty() else []
    combined_rows.extend(_with_event_key(row) for row in rows)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_row in combined_rows:
        row = _with_event_key(raw_row)
        key = (str(row.get("strategy") or ""), str(row.get("event_key") or ""))
        merged[key] = _merge_event_row(merged.get(key), row)

    df = pl.DataFrame(_json_safe_rows(list(merged.values())))
    return upsert_parquet_dataset(df, dataset_path, key_columns=["strategy", "event_key"])


def _merge_event_row(
    current: dict[str, Any] | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    if current is None:
        seeded = dict(row)
        seeded["source_count"] = _source_count(row)
        seeded["first_seen_bundle_ts"] = _first_seen_bundle_ts(row)
        seeded["last_seen_bundle_ts"] = _last_seen_bundle_ts(row)
        return seeded

    current_count = _source_count(current)
    row_count = _source_count(row)
    current_seen = _last_seen_bundle_ts(current)
    row_seen = _last_seen_bundle_ts(row)
    first_seen = _min_seen_ts(_first_seen_bundle_ts(current), _first_seen_bundle_ts(row))
    last_seen = _max_seen_ts(current_seen, row_seen)

    latest = row if _seen_sort_value(row_seen) >= _seen_sort_value(last_seen) else current
    merged = dict(latest)
    merged["source_count"] = current_count + row_count
    merged["first_seen_bundle_ts"] = first_seen
    merged["last_seen_bundle_ts"] = last_seen
    return merged


def _source_count(row: dict[str, Any]) -> int:
    value = row.get("source_count")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return 1
    return max(parsed, 1)


def _seen_bundle_ts(row: dict[str, Any]) -> Any:
    return row.get("bundle_ts") or row.get("ingest_ts")


def _first_seen_bundle_ts(row: dict[str, Any]) -> Any:
    return row.get("first_seen_bundle_ts") or _seen_bundle_ts(row)


def _last_seen_bundle_ts(row: dict[str, Any]) -> Any:
    return row.get("last_seen_bundle_ts") or _seen_bundle_ts(row)


def _seen_sort_value(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    if value is None or value == "":
        return datetime.min.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _min_seen_ts(left: Any, right: Any) -> Any:
    if _seen_sort_value(right) < _seen_sort_value(left):
        return right
    return left


def _max_seen_ts(left: Any, right: Any) -> Any:
    if _seen_sort_value(right) > _seen_sort_value(left):
        return right
    return left


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
