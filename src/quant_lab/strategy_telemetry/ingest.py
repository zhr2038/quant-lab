from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.contracts.v5_quant_lab import (
    V5_QUANT_LAB_CONTRACT_VERSION,
    V5_TELEMETRY_DATASET_SCHEMA_VERSION,
)
from quant_lab.data.lake import (
    append_parquet_dataset,
    read_parquet_dataset,
    upsert_parquet_dataset,
    write_parquet_dataset,
)
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
    "v5_btc_probe_entry_quality_audit": Path("silver/v5_btc_probe_entry_quality_audit"),
    "v5_quant_lab_usage": Path("silver/v5_quant_lab_usage"),
    "v5_quant_lab_request": Path("silver/v5_quant_lab_request"),
    "v5_quant_lab_compliance": Path("silver/v5_quant_lab_compliance"),
    "v5_quant_lab_cost_usage": Path("silver/v5_quant_lab_cost_usage"),
    "v5_quant_lab_fallback": Path("silver/v5_quant_lab_fallback"),
    "v5_candidate_event": Path("silver/v5_candidate_event"),
    "v5_order_lifecycle": Path("silver/v5_order_lifecycle"),
    "v5_paper_strategy_run": Path("silver/v5_paper_strategy_run"),
    "v5_paper_strategy_daily": Path("silver/v5_paper_strategy_daily"),
    "v5_paper_slippage_coverage": Path("silver/v5_paper_slippage_coverage"),
    "v5_expanded_universe_advisory_reader": Path("silver/v5_expanded_universe_advisory_reader"),
    "v5_expanded_universe_paper_runs": Path("silver/v5_expanded_universe_paper_runs"),
    "v5_expanded_universe_paper_daily": Path("silver/v5_expanded_universe_paper_daily"),
    "v5_bnb_profit_lock_shadow": Path("silver/v5_bnb_profit_lock_shadow"),
    "v5_bnb_negative_expectancy_attribution": Path("silver/v5_bnb_negative_expectancy_attribution"),
    "v5_final_score_vs_alpha6_conflict": Path("silver/v5_final_score_vs_alpha6_conflict"),
    "v5_bnb_strong_alpha6_bypass_shadow": Path("silver/v5_bnb_strong_alpha6_bypass_shadow"),
    "v5_negative_expectancy_attribution": Path("silver/v5_negative_expectancy_attribution"),
    "v5_bnb_paper_strategy_runs": Path("silver/v5_bnb_paper_strategy_runs"),
    "v5_bnb_paper_strategy_daily": Path("silver/v5_bnb_paper_strategy_daily"),
    "v5_negative_expectancy_consistency": Path("silver/v5_negative_expectancy_consistency"),
    "v5_pullback_reversal_shadow": Path("gold/v5_pullback_reversal_shadow"),
    "v5_pullback_reversal_readiness": Path("gold/v5_pullback_reversal_readiness"),
}

QUANT_LAB_USAGE_PATHS = {
    "raw/reports/quant_lab_usage.jsonl",
    "raw/large/reports/quant_lab_usage.jsonl.gz",
    "raw/quant_lab/quant_lab_usage.jsonl",
    "reports/quant_lab_usage.jsonl",
}
QUANT_LAB_REQUEST_PATHS = {
    "raw/reports/quant_lab_requests.jsonl",
    "raw/large/reports/quant_lab_requests.jsonl.gz",
    "raw/quant_lab/quant_lab_requests.jsonl",
    "reports/quant_lab_requests.jsonl",
}
EVENT_KEY_DATASETS = {"v5_quant_lab_request", "v5_quant_lab_fallback"}
STABLE_ROW_KEY_DATASETS = set(SILVER_DATASETS) - EVENT_KEY_DATASETS - {"v5_candidate_event"}
EMPTY_CSV_REFRESH_DATASETS = {
    "v5_expanded_universe_advisory_reader",
    "v5_expanded_universe_paper_runs",
    "v5_expanded_universe_paper_daily",
}
PULLBACK_STABLE_ROW_KEY_DATASETS = {
    "v5_pullback_reversal_shadow",
    "v5_pullback_reversal_readiness",
}
HISTORICAL_OUTCOME_PATH_PREFIXES = (
    "summaries/high_score_blocked_outcomes",
    "summaries/alt_impulse_shadow",
    "summaries/btc_leadership_probe_blocked_outcomes",
    "summaries/multi_position_swing_shadow",
    "summaries/factor_contribution_outcomes_by_factor",
    "summaries/protect_sol_exception_shadow_outcomes",
)
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
    "last_seen_source_count",
    "first_seen_bundle_ts",
    "last_seen_bundle_ts",
}
CANDIDATE_EVENT_SCHEMA_VERSION = "v5.candidate_snapshot.v1"


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
    run_analysis: bool = True,
    refresh_candidate_gold: bool = True,
    include_historical_outcomes: bool = True,
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
        extract_result = safe_extract_v5_bundle(
            bundle_path,
            extracted_dir,
            effective_limits,
            skip_member=None
            if include_historical_outcomes
            else _skip_historical_outcome_member,
        )
        pruned_historical_outcomes = (
            _historical_outcomes_skipped_by_extract(extract_result.warnings)
            if not include_historical_outcomes
            else []
        )
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
        prune_warnings = [
            f"skipped_historical_outcome_file:{path}" for path in pruned_historical_outcomes
        ]
        silver_rows, warnings = _write_silver(
            lake_root,
            redacted_root / "redacted_files",
            metadata,
            include_historical_outcomes=include_historical_outcomes,
        )
        warnings = prune_warnings + warnings
        candidate_gold_rows = (
            _write_candidate_gold(lake_root, bundle_day) if refresh_candidate_gold else {}
        )

    analysis = None
    if run_analysis:
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
    max_bundles: int | None = None,
    max_scan_bundles: int | None = None,
    newest_first: bool = False,
    max_skipped_files_reported: int | None = None,
    run_analysis: bool = True,
    refresh_candidate_gold: bool = True,
    include_historical_outcomes: bool = True,
) -> V5InboxIngestResult:
    processed: list[V5BundleIngestResult] = []
    skipped: list[str] = []
    warnings: list[str] = []
    skipped_total = 0
    existing_sha256s = _ingested_bundle_sha256s(lake_root)
    existing_names = _ingested_bundle_names(lake_root)
    bundle_paths = sorted(
        Path(inbox_dir).glob("v5_live_followup_bundle_*.tar.gz"),
        key=lambda path: path.name,
        reverse=newest_first,
    )
    scanned_bundle_count = len(bundle_paths)
    if max_scan_bundles is not None:
        bundle_paths = bundle_paths[: max(int(max_scan_bundles), 1)]
    for bundle_path in bundle_paths:
        if max_bundles is not None and len(processed) >= max_bundles:
            break
        if bundle_path.name in existing_names:
            skipped_total += 1
            if max_skipped_files_reported is None or len(skipped) < max_skipped_files_reported:
                skipped.append(str(bundle_path))
            continue
        sha256 = compute_sha256(bundle_path)
        if sha256 in existing_sha256s:
            skipped_total += 1
            if max_skipped_files_reported is None or len(skipped) < max_skipped_files_reported:
                skipped.append(str(bundle_path))
            continue
        result = ingest_v5_bundle(
            bundle_path=bundle_path,
            lake_root=lake_root,
            restricted_archive_dir=restricted_archive_dir,
            redacted_archive_dir=redacted_archive_dir,
            strategy=strategy,
            limits=limits,
            run_analysis=run_analysis,
            refresh_candidate_gold=refresh_candidate_gold,
            include_historical_outcomes=include_historical_outcomes,
        )
        processed.append(result)
        existing_sha256s.add(sha256)
        existing_names.add(bundle_path.name)
    if max_skipped_files_reported is not None and skipped_total > len(skipped):
        warnings.append(
            f"skipped_files_truncated:{len(skipped)}_of_{skipped_total}_already_ingested"
        )
    if max_bundles is not None:
        remaining = max(len([path for path in bundle_paths if path.exists()]) - len(processed), 0)
        if remaining:
            warnings.append(f"max_bundles_limit_applied:{max_bundles}")
    if max_scan_bundles is not None and scanned_bundle_count > len(bundle_paths):
        warnings.append(
            f"max_scan_bundles_limit_applied:{len(bundle_paths)}_of_{scanned_bundle_count}"
        )
    return V5InboxIngestResult(
        strategy=strategy,
        inbox_dir=str(inbox_dir),
        processed=processed,
        skipped_files=skipped,
        warnings=warnings,
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
        "raw_file_index": _append_rows(
            lake_root / BRONZE_DATASETS["raw_file_index"],
            file_rows,
        ),
    }


def _write_silver(
    lake_root: Path,
    redacted_files_dir: Path,
    metadata: dict[str, Any],
    *,
    include_historical_outcomes: bool = True,
) -> tuple[dict[str, int], list[str]]:
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in SILVER_DATASETS}
    empty_csv_headers: dict[str, tuple[str, list[str]]] = {}
    warnings: list[str] = []
    for file_path in sorted(path for path in redacted_files_dir.rglob("*") if path.is_file()):
        relative = file_path.relative_to(redacted_files_dir).as_posix()
        logical = _logical_bundle_path(relative)
        if not include_historical_outcomes and _is_historical_outcome_path(logical):
            warnings.append(f"skipped_historical_outcome_file:{logical}")
            continue
        try:
            _append_file_rows(rows, file_path, relative, metadata, empty_csv_headers)
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
            empty_csv = empty_csv_headers.get(name)
            if empty_csv is not None:
                relative, header = empty_csv
                counts[name] = _write_empty_csv_refresh_dataset(
                    lake_root / dataset,
                    metadata,
                    relative,
                    header,
                )
            continue
        dataset_path = lake_root / dataset
        if name in EVENT_KEY_DATASETS:
            counts[name] = _upsert_event_rows(dataset_path, dataset_rows)
        elif name == "v5_candidate_event":
            counts[name] = _upsert_rows(
                dataset_path,
                dataset_rows,
                ["strategy", "candidate_id", "run_id", "ts_utc", "symbol", "strategy_candidate"],
            )
        elif name in STABLE_ROW_KEY_DATASETS:
            counts[name] = _upsert_stable_rows(dataset_path, dataset_rows)
        else:
            counts[name] = _upsert_rows(
                dataset_path,
                dataset_rows,
                ["strategy", "bundle_sha256", "source_path_inside_bundle", "row_index"],
            )
    return counts, warnings


def _is_historical_outcome_path(logical: str) -> bool:
    return logical.startswith(HISTORICAL_OUTCOME_PATH_PREFIXES)


def _skip_historical_outcome_member(member_name: str) -> bool:
    return _is_historical_outcome_path(_logical_bundle_path(member_name))


def _historical_outcomes_skipped_by_extract(warnings: list[str]) -> list[str]:
    skipped: list[str] = []
    for warning in warnings:
        if not warning.startswith("skipped_member:"):
            continue
        logical = _logical_bundle_path(warning.removeprefix("skipped_member:"))
        if _is_historical_outcome_path(logical):
            skipped.append(logical)
    return sorted(set(skipped))


def _prune_historical_outcome_files(extracted_dir: Path) -> list[str]:
    pruned: list[str] = []
    for file_path in sorted(path for path in extracted_dir.rglob("*") if path.is_file()):
        relative = file_path.relative_to(extracted_dir).as_posix()
        logical = _logical_bundle_path(relative)
        if not _is_historical_outcome_path(logical):
            continue
        file_path.unlink()
        pruned.append(logical)
    return pruned


def _write_candidate_gold(lake_root: Path, bundle_day: str) -> dict[str, int]:
    from quant_lab.research.alpha_discovery import build_and_publish_alpha_discovery_board
    from quant_lab.research.candidate_labels import build_and_publish_candidate_labels

    result = build_and_publish_candidate_labels(lake_root, as_of_date=bundle_day)
    board = build_and_publish_alpha_discovery_board(lake_root, as_of_date=bundle_day)
    return {
        "v5_candidate_label": result.candidate_label_rows,
        "v5_candidate_quality_daily": result.candidate_quality_rows,
        "v5_candidate_outcome_summary": result.candidate_outcome_summary_rows,
        "alpha_discovery_board": board.alpha_discovery_board_rows,
    }


def _append_file_rows(
    rows: dict[str, list[dict[str, Any]]],
    file_path: Path,
    relative: str,
    metadata: dict[str, Any],
    empty_csv_headers: dict[str, tuple[str, list[str]]] | None = None,
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
    if logical.endswith("/order_lifecycle.csv") or logical == "order_lifecycle.csv":
        rows["v5_order_lifecycle"].extend(_order_lifecycle_rows(metadata, relative, file_path))
        return
    if logical.endswith("/candidate_snapshot.csv") or logical == "candidate_snapshot.csv":
        rows["v5_candidate_event"].extend(_candidate_event_rows(metadata, relative, file_path))
        return
    if logical == "summaries/btc_probe_entry_quality_audit.csv":
        rows["v5_btc_probe_entry_quality_audit"].extend(
            _btc_probe_entry_quality_audit_rows(metadata, relative, file_path)
        )
        return
    if logical in {
        "reports/pullback_reversal_shadow_outcomes.csv",
        "summaries/pullback_reversal_shadow_outcomes.csv",
    }:
        rows["v5_pullback_reversal_shadow"].extend(
            _pullback_shadow_rows(metadata, relative, file_path)
        )
        return
    if logical in {
        "reports/pullback_reversal_readiness.json",
        "summaries/pullback_reversal_readiness.json",
    }:
        rows["v5_pullback_reversal_readiness"].extend(
            _pullback_readiness_rows(metadata, relative, _read_json(file_path))
        )
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
        "summaries/paper_strategy_runs.csv": "v5_paper_strategy_run",
        "summaries/paper_strategy_daily.csv": "v5_paper_strategy_daily",
        "summaries/paper_slippage_coverage.csv": "v5_paper_slippage_coverage",
        "summaries/expanded_universe_advisory_reader.csv": (
            "v5_expanded_universe_advisory_reader"
        ),
        "summaries/expanded_universe_paper_runs.csv": "v5_expanded_universe_paper_runs",
        "summaries/expanded_universe_paper_daily.csv": "v5_expanded_universe_paper_daily",
        "summaries/bnb_profit_lock_shadow.csv": "v5_bnb_profit_lock_shadow",
        "summaries/bnb_negative_expectancy_attribution.csv": (
            "v5_bnb_negative_expectancy_attribution"
        ),
        "summaries/final_score_vs_alpha6_conflict.csv": "v5_final_score_vs_alpha6_conflict",
        "summaries/bnb_strong_alpha6_bypass_shadow.csv": "v5_bnb_strong_alpha6_bypass_shadow",
        "summaries/negative_expectancy_attribution.csv": "v5_negative_expectancy_attribution",
        "summaries/bnb_paper_strategy_runs.csv": "v5_bnb_paper_strategy_runs",
        "summaries/bnb_paper_strategy_daily.csv": "v5_bnb_paper_strategy_daily",
        "summaries/negative_expectancy_consistency.csv": "v5_negative_expectancy_consistency",
    }
    if logical.startswith("summaries/high_score_blocked_outcomes"):
        rows["v5_high_score_blocked_outcome"].extend(_csv_rows(metadata, relative, file_path))
    elif (
        logical.startswith("summaries/alt_impulse_shadow")
        or logical.startswith("summaries/btc_leadership_probe_blocked_outcomes")
        or logical.startswith("summaries/multi_position_swing_shadow")
        or logical.startswith("summaries/factor_contribution_outcomes_by_factor")
        or logical.startswith("summaries/protect_sol_exception_shadow_outcomes")
    ):
        rows["v5_shadow_outcome"].extend(_csv_rows(metadata, relative, file_path))
    elif logical == "summaries/quant_lab_fallbacks.csv":
        rows["v5_quant_lab_fallback"].extend(_fallback_csv_rows(metadata, relative, file_path))
    elif logical in csv_mapping:
        dataset_name = csv_mapping[logical]
        parsed_rows = _csv_rows(metadata, relative, file_path)
        if parsed_rows:
            rows[dataset_name].extend(parsed_rows)
            return
        if dataset_name in EMPTY_CSV_REFRESH_DATASETS and empty_csv_headers is not None:
            header = _csv_header(file_path)
            if header:
                empty_csv_headers[dataset_name] = (relative, header)


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
    for index, line in enumerate(_read_text_file(file_path).splitlines()):
        if not line.strip():
            continue
        payload = redact_json_like(json.loads(line))
        rows.append(_json_row(metadata, relative, payload, run_id, index))
    return rows


def _read_text_file(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8")


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


def _csv_header(file_path: Path) -> list[str]:
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle).fieldnames or [])


def _btc_probe_entry_quality_audit_rows(
    metadata: dict[str, Any],
    relative: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _csv_rows(metadata, relative, file_path):
        payload = _loads_payload(row.get("raw_payload_json"))
        symbol_value = _clean_text(
            _first_value(row, payload, ["selected_symbol", "symbol", "normalized_symbol"])
        )
        normalized_symbol = normalize_symbol(symbol_value) if symbol_value else ""
        live_order_effect = "none_read_only_v5_bundle_audit"
        enriched_payload = {
            **payload,
            "normalized_symbol": normalized_symbol,
            "live_order_effect": live_order_effect,
        }
        rows.append(
            row
            | {
                "normalized_symbol": normalized_symbol,
                "live_order_effect": live_order_effect,
                "raw_payload_json": safe_json_dumps(enriched_payload),
            }
        )
    return rows


def _pullback_shadow_rows(
    metadata: dict[str, Any],
    relative: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in _csv_rows(metadata, relative, file_path):
        payload = _loads_payload(row.get("raw_payload_json"))
        symbol_value = _clean_text(_first_value(row, payload, ["symbol", "normalized_symbol"]))
        normalized_symbol = normalize_symbol(symbol_value) if symbol_value else ""
        ts_utc = _normalize_event_time(_first_value(row, payload, ["ts_utc", "decision_ts", "ts"]))
        enriched_payload = {
            **payload,
            "symbol": normalized_symbol or symbol_value,
            "ts_utc": ts_utc,
            "contract_version": _first_value(row, payload, ["contract_version"])
            or V5_QUANT_LAB_CONTRACT_VERSION,
            "schema_version": _first_value(row, payload, ["schema_version"]) or SCHEMA_VERSION,
        }
        output.append(
            row
            | {
                "symbol": normalized_symbol or symbol_value,
                "normalized_symbol": normalized_symbol,
                "ts_utc": ts_utc,
                "generated_at_utc": _first_value(row, payload, ["generated_at_utc", "generated_at"])
                or metadata.get("ingest_ts"),
                "contract_version": enriched_payload["contract_version"],
                "schema_version": enriched_payload["schema_version"],
                "source": _first_value(row, payload, ["source"]) or "v5_followup_bundle",
                "raw_payload_json": safe_json_dumps(enriched_payload),
            }
        )
    return output


def _pullback_readiness_rows(
    metadata: dict[str, Any],
    relative: str,
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        raw_rows = [payload]
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw_rows):
        item = raw_row if isinstance(raw_row, dict) else {"value": raw_row}
        safe_item = redact_json_like(dict(item))
        symbol_value = _clean_text(
            _first_value(safe_item, safe_item, ["symbol", "normalized_symbol"])
        )
        normalized_symbol = normalize_symbol(symbol_value) if symbol_value else symbol_value
        rows.append(
            _base_row(metadata, relative, None, index)
            | {
                **{
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in safe_item.items()
                },
                "row_count": safe_item.get("row_count") or payload.get("row_count"),
                "symbol": normalized_symbol,
                "generated_at_utc": safe_item.get("generated_at_utc")
                or safe_item.get("generated_at")
                or metadata.get("ingest_ts"),
                "contract_version": safe_item.get("contract_version")
                or V5_QUANT_LAB_CONTRACT_VERSION,
                "schema_version": safe_item.get("schema_version") or SCHEMA_VERSION,
                "source": safe_item.get("source") or "v5_followup_bundle",
                "raw_payload_json": safe_json_dumps(payload),
            }
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


def _order_lifecycle_rows(
    metadata: dict[str, Any],
    relative: str,
    file_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _csv_rows(metadata, relative, file_path):
        payload = _loads_payload(row.get("raw_payload_json"))
        symbol_value = _clean_text(
            _first_value(row, payload, ["normalized_symbol", "symbol", "inst_id", "instId"])
        )
        normalized_symbol = normalize_symbol(symbol_value) if symbol_value else ""
        side = _clean_text(_first_value(row, payload, ["side", "order_side"])).lower()
        arrival_mid = _numeric(_first_value(row, payload, ["arrival_mid", "mid_px_at_decision"]))
        signal_price = _numeric(_first_value(row, payload, ["signal_price", "decision_price"]))
        avg_fill_px = _numeric(_first_value(row, payload, ["avg_fill_px", "fill_px", "avg_px"]))
        filled_qty = _numeric(
            _first_value(row, payload, ["filled_qty", "fill_qty", "fill_sz", "qty"])
        )
        notional = _numeric(
            _first_value(row, payload, ["notional_usdt", "notional", "requested_notional_usdt"])
        )
        if (
            (notional is None or notional <= 0)
            and avg_fill_px is not None
            and filled_qty is not None
        ):
            notional = abs(avg_fill_px * filled_qty)
        fee_usdt = _numeric(_first_value(row, payload, ["fee_usdt", "fee_abs_usdt"]))
        fee = _numeric(_first_value(row, payload, ["fee", "commission", "fee_abs"]))
        fee_ccy = _clean_text(_first_value(row, payload, ["fee_ccy", "fee_currency"]))
        if fee_usdt is None:
            fee_usdt = _trade_fee_usdt(
                fee=fee,
                fee_ccy=fee_ccy,
                symbol=normalized_symbol,
                price=avg_fill_px,
            )
        spread_bps = _spread_bps_at_decision(row, payload, arrival_mid)
        arrival_slippage_bps = _arrival_slippage_bps(
            side=side,
            avg_fill_px=avg_fill_px,
            arrival_mid=arrival_mid,
        )
        delay_cost_bps = _delay_cost_bps(
            side=side,
            signal_price=signal_price,
            arrival_mid=arrival_mid,
        )
        spread_cost_bps = (max(spread_bps, 0.0) / 2.0) if spread_bps is not None else None
        fee_bps = (
            abs(float(fee_usdt)) / abs(float(notional)) * 10_000.0
            if fee_usdt is not None and notional is not None and abs(float(notional)) > 0
            else None
        )
        total_cost = _sum_cost_parts(delay_cost_bps, arrival_slippage_bps, fee_bps)
        ts_utc = _normalize_event_time(
            _first_value(
                row,
                payload,
                ["ts_utc", "fill_ts", "last_fill_ts", "submit_ts", "decision_ts", "ts"],
            )
        )
        enriched_payload = {
            **payload,
            "symbol": normalized_symbol or symbol_value,
            "normalized_symbol": normalized_symbol,
            "ts_utc": ts_utc,
            "notional_usdt": notional,
            "fee_usdt": fee_usdt,
            "arrival_slippage_bps": arrival_slippage_bps,
            "delay_cost_bps": delay_cost_bps,
            "spread_cost_bps": spread_cost_bps,
            "fee_bps": fee_bps,
            "total_realized_cost_bps": total_cost,
            "realized_total_cost_bps": total_cost,
        }
        rows.append(
            row
            | {
                "event_type": "order_lifecycle",
                "ts_utc": ts_utc,
                "symbol": normalized_symbol or symbol_value,
                "normalized_symbol": normalized_symbol,
                "side": side,
                "notional_usdt": "" if notional is None else str(abs(notional)),
                "fee_usdt": "" if fee_usdt is None else str(abs(fee_usdt)),
                "arrival_slippage_bps": (
                    "" if arrival_slippage_bps is None else str(arrival_slippage_bps)
                ),
                "delay_cost_bps": "" if delay_cost_bps is None else str(delay_cost_bps),
                "spread_bps_at_decision": "" if spread_bps is None else str(spread_bps),
                "spread_cost_bps": "" if spread_cost_bps is None else str(spread_cost_bps),
                "fee_bps": "" if fee_bps is None else str(fee_bps),
                "total_realized_cost_bps": "" if total_cost is None else str(total_cost),
                "realized_total_cost_bps": "" if total_cost is None else str(total_cost),
                "raw_payload_json": safe_json_dumps(enriched_payload),
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
        ts_utc = _normalize_event_time(_first_value(row, payload, ["ts_utc", "ts", "timestamp"]))
        candidate_id = _candidate_text(_first_value(row, payload, ["candidate_id"]))
        if not candidate_id:
            candidate_id = _candidate_id(
                run_id,
                normalized_symbol or symbol_value,
                strategy_candidate,
                ts_utc,
                row.get("row_index"),
            )
        cost_source = _candidate_text(
            _first_value(row, payload, ["cost_source", "cost.source", "cost.cost_source"])
        )
        cost_bps = _candidate_text(_first_value(row, payload, ["cost_bps", "cost.bps", "cost"]))
        selected_total_cost_bps = _candidate_text(
            _first_value(
                row,
                payload,
                [
                    "selected_total_cost_bps",
                    "total_cost_bps",
                    "cost.selected_total_cost_bps",
                    "cost.total_cost_bps",
                ],
            )
        )
        cost_model_version = _candidate_text(
            _first_value(row, payload, ["cost_model_version", "cost.model_version"])
        )
        cost_gate_verified = _candidate_text(
            _first_value(row, payload, ["cost_gate_verified", "cost.gate_verified"])
        )
        would_block_by_cost = _candidate_text(
            _first_value(row, payload, ["would_block_by_cost", "cost.would_block"])
        )
        expected_edge_bps = _candidate_text(
            _first_value(row, payload, ["expected_edge_bps", "edge_bps", "expected_edge"])
        )
        required_edge_bps = _candidate_text(
            _first_value(row, payload, ["required_edge_bps", "required_edge"])
        )
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
            "cost_source": cost_source,
            "cost_bps": cost_bps,
            "selected_total_cost_bps": selected_total_cost_bps,
            "cost_model_version": cost_model_version,
            "cost_gate_verified": cost_gate_verified,
            "would_block_by_cost": would_block_by_cost,
            "expected_edge_bps": expected_edge_bps,
            "required_edge_bps": required_edge_bps,
            "raw_payload_json": safe_json_dumps(
                {
                    **payload,
                    "candidate_id": candidate_id,
                    "run_id": run_id,
                    "ts_utc": ts_utc,
                    "symbol": normalized_symbol or symbol_value,
                    "normalized_symbol": normalized_symbol,
                    "strategy_candidate": strategy_candidate,
                    "cost_source": cost_source,
                    "cost_bps": cost_bps,
                    "selected_total_cost_bps": selected_total_cost_bps,
                    "cost_model_version": cost_model_version,
                    "cost_gate_verified": cost_gate_verified,
                    "would_block_by_cost": would_block_by_cost,
                    "expected_edge_bps": expected_edge_bps,
                    "required_edge_bps": required_edge_bps,
                }
            ),
        }
        rows.append(event)
    return rows


def _candidate_id(
    run_id: str,
    symbol: str,
    strategy_candidate: str,
    ts_utc: str,
    row_index: Any,
) -> str:
    material = "|".join(
        [
            str(run_id or "").strip(),
            str(ts_utc or "").strip(),
            str(symbol or "").strip().upper(),
            str(strategy_candidate or "portfolio").strip(),
            "" if row_index is None else str(row_index).strip(),
        ]
    )
    return "cand_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]


def _candidate_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _arrival_slippage_bps(
    *,
    side: str,
    avg_fill_px: float | None,
    arrival_mid: float | None,
) -> float | None:
    if avg_fill_px is None or arrival_mid is None or arrival_mid <= 0:
        return None
    if side == "sell":
        return (arrival_mid - avg_fill_px) / arrival_mid * 10_000.0
    return (avg_fill_px - arrival_mid) / arrival_mid * 10_000.0


def _delay_cost_bps(
    *,
    side: str,
    signal_price: float | None,
    arrival_mid: float | None,
) -> float | None:
    if signal_price is None or arrival_mid is None or signal_price <= 0:
        return None
    if side == "sell":
        return (signal_price - arrival_mid) / signal_price * 10_000.0
    return (arrival_mid - signal_price) / signal_price * 10_000.0


def _spread_bps_at_decision(
    row: dict[str, Any],
    payload: dict[str, Any],
    arrival_mid: float | None,
) -> float | None:
    explicit = _numeric(
        _first_value(
            row,
            payload,
            [
                "spread_bps_at_decision",
                "arrival_spread_bps",
                "estimated_spread_bps",
                "spread_bps",
            ],
        )
    )
    if explicit is not None:
        return abs(explicit)

    bid = _numeric(_first_value(row, payload, ["arrival_bid", "best_bid", "bid_px", "bid"]))
    ask = _numeric(_first_value(row, payload, ["arrival_ask", "best_ask", "ask_px", "ask"]))
    mid = arrival_mid
    if mid is None and bid is not None and ask is not None:
        mid = (bid + ask) / 2.0
    if bid is not None and ask is not None and mid is not None and mid > 0:
        return abs(ask - bid) / mid * 10_000.0

    generic = _numeric(_first_value(row, payload, ["spread"]))
    if generic is None:
        return None
    unit = str(_first_value(row, payload, ["spread_unit", "spread_units"]) or "").lower()
    if unit in {"price", "quote", "usdt", "absolute", "px"}:
        if mid is None or mid <= 0:
            return None
        return abs(generic) / mid * 10_000.0
    return abs(generic)


def _sum_cost_parts(*parts: float | None) -> float | None:
    observed = [float(part) for part in parts if part is not None]
    if not observed:
        return None
    return sum(observed)


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
            if key not in {"event_id", "fallback_used", "raw_payload_hash"}
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
    del fields
    return _event_payload_conflict_hash(row, payload)


_EVENT_PAYLOAD_IDENTITY_ALIASES = {
    "event_id",
    "eventId",
    "source_event_id",
    "strategy_id",
    "strategyId",
    "strategy",
    "run_id",
    "runId",
    "run",
    "event_type",
    "type",
    "kind",
    "endpoint",
    "endpoint_path",
    "path",
    "url",
    "route",
    "api_path",
    "request_path",
    "ts_utc",
    "ts",
    "timestamp",
    "created_at",
    "time",
    "request_ts",
    "event_ts",
    "status_code",
    "error_type",
    "exception_type",
    "request_id",
    "trace_id",
    "id",
    "uuid",
    "symbol",
    "normalized_symbol",
    "inst_id",
    "instId",
    "instrument",
    "pair",
    "side",
    "order_side",
    "intent",
    "action",
    "router_intent",
    "fallback_used",
    "used_fallback",
    "local_fallback",
    "raw_payload_hash",
    "payload_hash",
    "payload_hashes_json",
    "payload_hash_count",
    "conflicting_duplicate",
    "event_key",
    "event_key_fields_json",
}

_EVENT_PAYLOAD_VOLATILE_FIELDS = {
    "latency_ms",
    "elapsed_ms",
    "duration_ms",
    "request_duration_ms",
    "response_time_ms",
    "roundtrip_ms",
    "timing_ms",
    "duration_seconds",
    "elapsed_seconds",
    "sampled_at",
    "sampled_at_utc",
    "cache_hit",
    "response_cache_hit",
}


def _payload_conflict_key_is_ignored(key: Any) -> bool:
    rendered = str(key)
    return (
        rendered in EVENT_KEY_METADATA_FIELDS
        or rendered in _EVENT_PAYLOAD_IDENTITY_ALIASES
        or rendered in _EVENT_PAYLOAD_VOLATILE_FIELDS
    )


def _event_payload_conflict_hash(row: dict[str, Any], payload: dict[str, Any]) -> str:
    source: Any = payload if payload else row
    if isinstance(source, dict):
        source = {
            key: _normalize_payload_conflict_value(value)
            for key, value in source.items()
            if not _payload_conflict_key_is_ignored(key)
            and not _payload_conflict_value_is_empty(value)
        }
    rendered = json.dumps(source, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _payload_conflict_value_is_empty(value: Any) -> bool:
    if value is None:
        return True
    rendered = str(value).strip().lower()
    return rendered in {"", "not_observable", "not-observable", "none", "null", "nan"}


def _normalize_payload_conflict_value(value: Any) -> Any:
    if isinstance(value, str):
        rendered = value.strip()
        lowered = rendered.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return rendered
    if isinstance(value, list):
        return [_normalize_payload_conflict_value(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _normalize_payload_conflict_value(item)
            for key, item in value.items()
            if not _payload_conflict_key_is_ignored(key)
            and not _payload_conflict_value_is_empty(item)
        }
    return value


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
    df = _dataframe_from_rows(rows)
    return upsert_parquet_dataset(df, dataset_path, key_columns=keys)


def _append_rows(dataset_path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    df = _dataframe_from_rows(rows)
    bundle_sha256 = str(rows[0].get("bundle_sha256") or "batch")
    prefix = f"bundle_{bundle_sha256[:12]}"
    result = append_parquet_dataset(df, dataset_path, file_prefix=prefix)
    return result.rows_written


def _upsert_stable_rows(dataset_path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return read_parquet_dataset(dataset_path).height
    existing = read_parquet_dataset(dataset_path)
    combined_rows = existing.to_dicts() if not existing.is_empty() else []
    combined_rows.extend(rows)
    keyed_rows = [_with_stable_row_key(row) for row in combined_rows]
    df = _dataframe_from_rows(keyed_rows)
    if not df.is_empty():
        candidate_keys = (
            ["strategy", "stable_row_key"]
            if dataset_path.name in PULLBACK_STABLE_ROW_KEY_DATASETS
            else ["strategy", "source_path_inside_bundle", "stable_row_key"]
        )
        key_columns = [
            column
            for column in candidate_keys
            if column in df.columns
        ]
        if key_columns:
            df = df.unique(subset=key_columns, keep="last", maintain_order=True)
    write_parquet_dataset(df, dataset_path)
    return df.height


def _write_empty_csv_refresh_dataset(
    dataset_path: Path,
    metadata: dict[str, Any],
    relative: str,
    header: list[str],
) -> int:
    columns = [
        "strategy",
        "bundle_sha256",
        "bundle_name",
        "bundle_ts",
        "ingest_ts",
        "schema_version",
        "source_path_inside_bundle",
        "run_id",
        "row_index",
        *header,
        "raw_payload_json",
        "stable_row_key",
    ]
    columns = list(dict.fromkeys(str(column) for column in columns if str(column).strip()))
    if not columns:
        columns = list(_base_row(metadata, relative, run_id_from_path(relative), 0).keys())
    write_parquet_dataset(
        pl.DataFrame(schema={column: pl.Utf8 for column in columns}),
        dataset_path,
    )
    return 0


def _with_stable_row_key(row: dict[str, Any]) -> dict[str, Any]:
    seeded = dict(row)
    seeded["stable_row_key"] = _stable_row_key(row)
    return seeded


def _stable_row_key(row: dict[str, Any]) -> str:
    source_path = _logical_bundle_path(str(row.get("source_path_inside_bundle") or ""))
    if source_path in {
        "reports/pullback_reversal_shadow_outcomes.csv",
        "summaries/pullback_reversal_shadow_outcomes.csv",
    }:
        return _pullback_shadow_stable_row_key(row)
    if source_path in {
        "reports/pullback_reversal_readiness.json",
        "summaries/pullback_reversal_readiness.json",
    }:
        return _pullback_readiness_stable_row_key(row)
    payload = row.get("raw_payload_json")
    if isinstance(payload, str) and payload.strip():
        basis: Any = payload.strip()
    else:
        basis = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "bundle_sha256",
                "bundle_name",
                "bundle_ts",
                "ingest_ts",
                "stable_row_key",
            }
        }
    encoded = safe_json_dumps(basis) if not isinstance(basis, str) else basis
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pullback_shadow_stable_row_key(row: dict[str, Any]) -> str:
    payload = _loads_payload(row.get("raw_payload_json"))
    symbol_value = _first_value(row, payload, ["normalized_symbol", "symbol"])
    symbol = normalize_symbol(_clean_text(symbol_value)) if symbol_value else ""
    ts_utc = _normalize_event_time(
        _first_value(row, payload, ["ts_utc", "decision_ts", "ts", "entry_ts"])
    )
    strategy_candidate = _clean_text(
        _first_value(row, payload, ["strategy_candidate", "candidate", "strategy_id"])
    )
    horizon = _clean_text(
        _first_value(
            row,
            payload,
            ["horizon_hours", "label_horizon_hours", "horizon", "label_horizon"],
        )
    )
    candidate_id = _clean_text(
        _first_value(row, payload, ["candidate_id", "source_event_id", "event_id"])
    )
    run_id = _clean_text(_first_value(row, payload, ["run_id", "runId", "run"]))
    row_index = "" if candidate_id else _clean_text(row.get("row_index"))
    basis = {
        "strategy": _clean_text(row.get("strategy")),
        "symbol": symbol,
        "ts_utc": ts_utc,
        "strategy_candidate": strategy_candidate,
        "horizon_hours": horizon,
        "candidate_id": candidate_id,
        "run_id": run_id if not candidate_id else "",
        "row_index": row_index if not candidate_id else "",
    }
    encoded = safe_json_dumps({key: value for key, value in basis.items() if value})
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _pullback_readiness_stable_row_key(row: dict[str, Any]) -> str:
    payload = _loads_payload(row.get("raw_payload_json"))
    symbol_value = _first_value(row, payload, ["normalized_symbol", "symbol"])
    symbol = normalize_symbol(_clean_text(symbol_value)) if symbol_value else ""
    generated_at = _normalize_event_time(
        _first_value(row, payload, ["generated_at_utc", "generated_at", "as_of_ts", "as_of_date"])
    )
    if not generated_at:
        generated_at = _clean_text(_first_value(row, payload, ["as_of_date", "day"]))
    basis = {
        "strategy": _clean_text(row.get("strategy")),
        "symbol": symbol,
        "generated_at_utc": generated_at,
        "contract_version": _clean_text(
            _first_value(row, payload, ["contract_version"]) or row.get("contract_version")
        ),
        "schema_version": _clean_text(
            _first_value(row, payload, ["schema_version"]) or row.get("schema_version")
        ),
        "readiness_item": _clean_text(
            _first_value(row, payload, ["readiness_item", "metric", "name", "strategy_candidate"])
        ),
        "horizon_hours": _clean_text(
            _first_value(row, payload, ["horizon_hours", "label_horizon_hours", "horizon"])
        ),
    }
    encoded = safe_json_dumps({key: value for key, value in basis.items() if value})
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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

    df = _dataframe_from_rows(list(merged.values()))
    return upsert_parquet_dataset(df, dataset_path, key_columns=["strategy", "event_key"])


def _merge_event_row(
    current: dict[str, Any] | None,
    row: dict[str, Any],
) -> dict[str, Any]:
    if current is None:
        seeded = dict(row)
        seeded["source_count"] = _source_count(row)
        seeded["last_seen_source_count"] = seeded["source_count"]
        seeded["first_seen_bundle_ts"] = _first_seen_bundle_ts(row)
        seeded["last_seen_bundle_ts"] = _last_seen_bundle_ts(row)
        hashes = _payload_hash_set(seeded)
        seeded["payload_hashes_json"] = safe_json_dumps(sorted(hashes))
        seeded["payload_hash_count"] = len(hashes)
        seeded["conflicting_duplicate"] = False
        return seeded

    current_count = _source_count(current)
    row_count = _source_count(row)
    current_seen = _last_seen_bundle_ts(current)
    row_seen = _last_seen_bundle_ts(row)
    first_seen = _min_seen_ts(_first_seen_bundle_ts(current), _first_seen_bundle_ts(row))
    last_seen = _max_seen_ts(current_seen, row_seen)
    current_seen_sort = _seen_sort_value(current_seen)
    row_seen_sort = _seen_sort_value(row_seen)
    if row_seen_sort > current_seen_sort:
        last_seen_source_count = row_count
    elif row_seen_sort == current_seen_sort:
        last_seen_source_count = _last_seen_source_count(current) + row_count
    else:
        last_seen_source_count = _last_seen_source_count(current)

    latest = row if row_seen_sort >= _seen_sort_value(last_seen) else current
    merged = dict(latest)
    merged["source_count"] = current_count + row_count
    merged["last_seen_source_count"] = last_seen_source_count
    merged["first_seen_bundle_ts"] = first_seen
    merged["last_seen_bundle_ts"] = last_seen
    hashes = _payload_hash_set(current) | _payload_hash_set(row)
    merged["payload_hashes_json"] = safe_json_dumps(sorted(hashes))
    merged["payload_hash_count"] = len(hashes)
    merged["conflicting_duplicate"] = len(hashes) > 1
    return merged


def _payload_hash_set(row: dict[str, Any]) -> set[str]:
    raw_hashes = row.get("payload_hashes_json")
    hashes: set[str] = set()
    if isinstance(raw_hashes, str) and raw_hashes.strip():
        try:
            parsed = json.loads(raw_hashes)
        except json.JSONDecodeError:
            parsed = []
        if isinstance(parsed, list):
            hashes.update(str(item) for item in parsed if str(item).strip())
    for field in ["raw_payload_hash", "payload_hash"]:
        value = row.get(field)
        if value is not None and str(value).strip():
            hashes.add(str(value).strip())
    if not hashes:
        hashes.add(_stable_row_key(row))
    return hashes


def _source_count(row: dict[str, Any]) -> int:
    value = row.get("source_count")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        return 1
    return max(parsed, 1)


def _last_seen_source_count(row: dict[str, Any]) -> int:
    value = row.get("last_seen_source_count")
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = 0
    if parsed > 0:
        return parsed
    first_seen = _seen_sort_value(_first_seen_bundle_ts(row))
    last_seen = _seen_sort_value(_last_seen_bundle_ts(row))
    if first_seen == last_seen:
        return _source_count(row)
    return 1


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


def _dataframe_from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(_json_safe_rows(rows), infer_schema_length=None)


def _already_ingested(lake_root: Path, bundle_sha256: str) -> bool:
    return bundle_sha256 in _ingested_bundle_sha256s(lake_root)


def _ingested_bundle_sha256s(lake_root: Path) -> set[str]:
    existing = read_parquet_dataset(lake_root / BRONZE_DATASETS["bundle_manifest"])
    if existing.is_empty() or "bundle_sha256" not in existing.columns:
        return set()
    return {str(value) for value in existing["bundle_sha256"].to_list() if value}


def _ingested_bundle_names(lake_root: Path) -> set[str]:
    existing = read_parquet_dataset(lake_root / BRONZE_DATASETS["bundle_manifest"])
    if existing.is_empty() or "bundle_name" not in existing.columns:
        return set()
    return {str(value) for value in existing["bundle_name"].to_list() if value}


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
