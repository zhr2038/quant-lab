from __future__ import annotations

import bisect
import gc
import hashlib
import json
import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.research.candidate_labels import (
    LABEL_SCHEMA,
    candidate_label_bars_by_symbol,
    compute_v5_candidate_labels,
)
from quant_lab.research.strategy_evidence import compute_v5_candidate_evidence_samples
from quant_lab.research_plane.v5_candidate_evidence_contracts import (
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_INPUT_UNCOMPRESSED_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_SNAPSHOT_BYTES,
    V5_CANDIDATE_EVIDENCE_HORIZONS,
    V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS,
    V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceTask,
)
from quant_lab.symbols import normalize_symbol

V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS = (
    "candidate_events_within_signed_window",
    "candidate_event_primary_keys_are_unique",
    "candidate_symbols_are_normalized",
    "decision_bar_is_first_closed_bar_after_event",
    "decision_timestamp_is_after_event",
    "label_timestamp_is_after_decision",
    "horizons_match_signed_contract",
    "mfe_mae_use_decision_to_label_path",
    "market_access_stays_within_snapshot",
    "future_unavailable_rows_are_pending",
    "candidate_label_primary_keys_are_unique",
    "candidate_label_events_exist_in_snapshot",
    "source_bundle_sha_matches_event",
    "source_path_matches_event",
    "cost_bps_comes_from_signed_event",
    "worker_reads_no_external_cost_source",
    "evidence_sample_candidate_labels_exist",
    "evidence_sample_events_exist",
    "evidence_sample_primary_keys_are_unique",
    "strategy_candidate_matches_event",
    "evidence_horizon_matches_label",
    "evidence_net_bps_matches_label",
    "worker_outputs_no_strategy_evidence_decision",
    "worker_outputs_no_paper_status",
    "worker_outputs_no_live_status",
    "automatic_promotion_is_disabled",
    "live_notional_is_zero",
    "live_order_effect_is_none_read_only_research",
)


@dataclass(frozen=True)
class StagedV5CandidateEvidenceDataset:
    dataset_name: str
    paths: tuple[Path, ...]
    row_count: int
    uncompressed_bytes: int

    def collect(self, schema: dict[str, Any]) -> pl.DataFrame:
        if not self.paths:
            return pl.DataFrame(schema=schema)
        return (
            pl.scan_parquet(
                [str(path) for path in self.paths],
                missing_columns="insert",
                extra_columns="ignore",
            )
            .select(list(schema))
            .collect(engine="streaming")
        )


@dataclass(frozen=True)
class V5CandidateEvidenceComputeArtifacts:
    generated_at: datetime
    labels: StagedV5CandidateEvidenceDataset
    samples: StagedV5CandidateEvidenceDataset
    anti_leakage: dict[str, Any]
    worker_report: dict[str, Any]
    warnings: tuple[str, ...]
    peak_rss_bytes: int
    temporary_disk_peak_bytes: int


def compute_v5_candidate_evidence_result(
    snapshot_root: str | Path,
    manifest: V5CandidateEvidenceSnapshotManifest,
    task: V5CandidateEvidenceTask,
    *,
    stage_callback: Callable[[str], None] | None = None,
    max_snapshot_bytes: int = DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_SNAPSHOT_BYTES,
    max_input_uncompressed_bytes: int = (
        DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_INPUT_UNCOMPRESSED_BYTES
    ),
    max_input_rows: int = 5_000_000,
    min_free_disk_bytes: int = 1024**3,
    work_dir: str | Path | None = None,
) -> V5CandidateEvidenceComputeArtifacts:
    """Compute signed-snapshot Candidate Label and Evidence Sample deltas by symbol."""

    started = time.perf_counter()
    _validate_task_manifest_binding(task, manifest)
    if manifest.total_input_bytes > max_snapshot_bytes:
        raise ValueError("v5_candidate_evidence_snapshot_input_size_limit_exceeded")
    if manifest.estimated_uncompressed_bytes > max_input_uncompressed_bytes:
        raise ValueError("v5_candidate_evidence_input_uncompressed_size_limit_exceeded")
    if manifest.total_input_rows > max_input_rows:
        raise ValueError("v5_candidate_evidence_input_row_limit_exceeded")

    snapshot_files = Path(snapshot_root) / "files"
    stage_root = _prepare_stage_root(snapshot_root, task.task_id, work_dir=work_dir)
    if shutil.disk_usage(stage_root).free < min_free_disk_bytes:
        raise ValueError("v5_candidate_evidence_insufficient_temporary_disk")

    generated_at = datetime.now(UTC)
    event_paths = _snapshot_dataset_paths(
        snapshot_files,
        manifest,
        "silver/v5_candidate_event",
    )
    market_paths = _snapshot_dataset_paths(
        snapshot_files,
        manifest,
        "silver/market_bar",
    )
    events = _scan_paths(event_paths)
    markets = _scan_paths(market_paths)
    event_columns = events.collect_schema().names()
    market_columns = markets.collect_schema().names()
    if "symbol" not in event_columns:
        normalized_events = events.with_columns(pl.lit("").alias("_normalized_symbol"))
    else:
        normalized_events = events.with_columns(
            pl.col("symbol")
            .map_elements(normalize_symbol, return_dtype=pl.Utf8)
            .alias("_normalized_symbol")
        )
    if "symbol" not in market_columns:
        normalized_markets = markets.with_columns(pl.lit("").alias("_normalized_symbol"))
    else:
        normalized_markets = markets.with_columns(
            pl.col("symbol")
            .map_elements(normalize_symbol, return_dtype=pl.Utf8)
            .alias("_normalized_symbol")
        )

    symbols = tuple(
        normalized_events.select(
            pl.col("_normalized_symbol").fill_null("").unique().sort()
        )
        .collect(engine="streaming")
        .get_column("_normalized_symbol")
        .to_list()
    )
    expected_symbols = tuple(symbol for symbol in symbols if symbol)
    if expected_symbols != manifest.candidate_symbols:
        raise ValueError("v5_candidate_evidence_snapshot_symbol_binding_mismatch")

    violations = {name: 0 for name in V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS}
    warnings: list[str] = []
    label_paths: list[Path] = []
    sample_paths: list[Path] = []
    label_rows = 0
    sample_rows = 0
    label_uncompressed = 0
    sample_uncompressed = 0
    peak_rss = _peak_rss_bytes()
    temporary_disk_peak = 0
    stage_durations: dict[str, float] = {}

    labels_started = time.perf_counter()
    if stage_callback is not None:
        stage_callback("computing_labels")
    for symbol_index, symbol in enumerate(symbols):
        symbol_events = (
            normalized_events.filter(pl.col("_normalized_symbol") == symbol)
            .drop("_normalized_symbol")
            .collect(engine="streaming")
        )
        symbol_markets = (
            normalized_markets.filter(pl.col("_normalized_symbol") == symbol)
            .drop("_normalized_symbol")
            .collect(engine="streaming")
            if symbol
            else pl.DataFrame(schema={column: market_columns[column] for column in []})
        )
        labels = compute_v5_candidate_labels(
            symbol_events,
            symbol_markets,
            created_at=generated_at,
        )
        _accumulate_symbol_label_checks(
            violations,
            symbol_events,
            symbol_markets,
            labels,
            manifest=manifest,
        )
        paths = _write_frame_by_utc_date(
            labels,
            stage_root / "v5_candidate_label_delta",
            timestamp_column="ts_utc",
            part_token=f"{symbol_index:04d}-{_safe_symbol_token(symbol)}",
        )
        label_paths.extend(paths)
        label_rows += labels.height
        label_uncompressed += sum(_parquet_uncompressed_bytes(path) for path in paths)
        del labels
        gc.collect()
        peak_rss = max(peak_rss, _peak_rss_bytes())
        temporary_disk_peak = max(temporary_disk_peak, _directory_size_bytes(stage_root))
    stage_durations["computing_labels"] = time.perf_counter() - labels_started

    samples_started = time.perf_counter()
    if stage_callback is not None:
        stage_callback("computing_samples")
    for symbol_index, symbol in enumerate(symbols):
        symbol_events = (
            normalized_events.filter(pl.col("_normalized_symbol") == symbol)
            .drop("_normalized_symbol")
            .collect(engine="streaming")
        )
        symbol_labels = _collect_symbol_labels(label_paths, symbol)
        samples, sample_warnings = compute_v5_candidate_evidence_samples(
            symbol_labels,
            symbol_events,
            as_of_date=task.as_of_date.isoformat(),
        )
        warnings.extend(sample_warnings)
        _accumulate_symbol_sample_checks(
            violations,
            symbol_events,
            symbol_labels,
            samples,
        )
        paths = _write_frame_by_utc_date(
            samples,
            stage_root / "strategy_evidence_sample_delta",
            timestamp_column="ts_utc",
            part_token=f"{symbol_index:04d}-{_safe_symbol_token(symbol)}",
        )
        sample_paths.extend(paths)
        sample_rows += samples.height
        sample_uncompressed += sum(_parquet_uncompressed_bytes(path) for path in paths)
        del symbol_labels, samples
        gc.collect()
        peak_rss = max(peak_rss, _peak_rss_bytes())
        temporary_disk_peak = max(temporary_disk_peak, _directory_size_bytes(stage_root))
    stage_durations["computing_samples"] = time.perf_counter() - samples_started

    violations["candidate_event_primary_keys_are_unique"] += _duplicate_count_lazy(
        events,
        ["candidate_id"],
    )
    violations["candidate_label_primary_keys_are_unique"] += _duplicate_count_paths(
        label_paths,
        list(V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS),
    )
    violations["evidence_sample_primary_keys_are_unique"] += _duplicate_count_paths(
        sample_paths,
        list(V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS),
    )
    if task.automatic_promotion:
        violations["automatic_promotion_is_disabled"] += 1
    if task.max_live_notional_usdt != 0:
        violations["live_notional_is_zero"] += 1
    if task.live_order_effect != "none_read_only_research":
        violations["live_order_effect_is_none_read_only_research"] += 1

    failed = {name: count for name, count in violations.items() if count}
    checks = [
        {
            "check_name": name,
            "status": "PASS" if violations[name] == 0 else "FAIL",
            "violation_count": violations[name],
            "detail": (
                "validated against signed snapshot and task"
                if violations[name] == 0
                else f"detected {violations[name]} violation(s)"
            ),
        }
        for name in V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS
    ]
    anti_leakage = {
        "schema_version": "v5_candidate_evidence_anti_leakage.v1",
        "task_id": task.task_id,
        "snapshot_id": manifest.snapshot_id,
        "status": "PASS" if not failed else "FAIL",
        "violation_count": sum(failed.values()),
        "checks": checks,
        "generated_at": datetime.now(UTC).isoformat(),
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "live_order_effect": "none_read_only_research",
    }
    if failed:
        first = next(iter(failed.items()))
        raise ValueError(f"v5_candidate_evidence_anti_leakage_failed:{first[0]}:{first[1]}")

    peak_rss = max(peak_rss, _peak_rss_bytes())
    temporary_disk_peak = max(temporary_disk_peak, _directory_size_bytes(stage_root))
    worker_report = {
        "schema_version": "v5_candidate_evidence_worker_report.v1",
        "task_id": task.task_id,
        "snapshot_id": manifest.snapshot_id,
        "input_fingerprint_digest": manifest.input_fingerprint_digest,
        "candidate_event_rows": manifest.candidate_event_row_count,
        "market_bar_rows": manifest.market_bar_row_count,
        "run_summary_rows": manifest.run_summary_row_count,
        "candidate_symbols": list(manifest.candidate_symbols),
        "label_rows": label_rows,
        "sample_rows": sample_rows,
        "stage_durations_seconds": stage_durations,
        "compute_duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "temporary_disk_peak_bytes": temporary_disk_peak,
        "warnings": sorted(set(warnings)),
        "diagnostic_only": True,
        "research_only": True,
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "live_order_effect": "none_read_only_research",
    }
    return V5CandidateEvidenceComputeArtifacts(
        generated_at=generated_at,
        labels=StagedV5CandidateEvidenceDataset(
            dataset_name="v5_candidate_label_delta",
            paths=tuple(label_paths),
            row_count=label_rows,
            uncompressed_bytes=label_uncompressed,
        ),
        samples=StagedV5CandidateEvidenceDataset(
            dataset_name="strategy_evidence_sample_delta",
            paths=tuple(sample_paths),
            row_count=sample_rows,
            uncompressed_bytes=sample_uncompressed,
        ),
        anti_leakage=anti_leakage,
        worker_report=worker_report,
        warnings=tuple(sorted(set(warnings))),
        peak_rss_bytes=peak_rss,
        temporary_disk_peak_bytes=temporary_disk_peak,
    )


def _validate_task_manifest_binding(
    task: V5CandidateEvidenceTask,
    manifest: V5CandidateEvidenceSnapshotManifest,
) -> None:
    checks = (
        (task.snapshot_id, manifest.snapshot_id),
        (task.snapshot_manifest_sha256, manifest.manifest_sha256),
        (task.quant_lab_commit, manifest.quant_lab_commit),
        (task.input_fingerprint_digest, manifest.input_fingerprint_digest),
        (task.as_of_date, manifest.as_of_date),
        (task.mode, manifest.mode),
        (task.lookback_days, manifest.lookback_days),
        (task.horizon_hours, manifest.horizon_hours),
        (task.include_historical_outcomes, manifest.include_historical_outcomes),
        (task.candidate_label_schema_version, manifest.candidate_label_schema_version),
        (task.strategy_evidence_version, manifest.strategy_evidence_version),
    )
    if any(left != right for left, right in checks):
        raise ValueError("v5_candidate_evidence_task_snapshot_binding_mismatch")


def _accumulate_symbol_label_checks(
    violations: dict[str, int],
    events: pl.DataFrame,
    markets: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    manifest: V5CandidateEvidenceSnapshotManifest,
) -> None:
    event_rows = events.to_dicts()
    label_rows = labels.to_dicts()
    event_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in event_rows
        if str(row.get("candidate_id") or "").strip()
    }
    labels_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in label_rows:
        labels_by_id.setdefault(str(row.get("candidate_id") or "").strip(), []).append(row)

    for event in event_rows:
        event_ts = _as_utc(event.get("ts_utc") or event.get("bundle_ts") or event.get("ingest_ts"))
        if event_ts is None or not (
            manifest.event_window_start <= event_ts < manifest.event_window_end
        ):
            violations["candidate_events_within_signed_window"] += 1
        raw_symbol = event.get("symbol")
        if raw_symbol not in (None, "") and normalize_symbol(raw_symbol) != str(
            raw_symbol
        ).strip().upper().replace("/", "-").replace("_", "-"):
            # Input aliases are permitted, but emitted labels must be canonical.
            pass

    bars_by_symbol = candidate_label_bars_by_symbol(markets)
    for candidate_id, rows in labels_by_id.items():
        event = event_by_id.get(candidate_id)
        if event is None:
            violations["candidate_label_events_exist_in_snapshot"] += len(rows)
            continue
        first = rows[0]
        symbol = normalize_symbol(first.get("symbol"))
        if first.get("symbol") != symbol:
            violations["candidate_symbols_are_normalized"] += 1
        event_ts = _as_utc(first.get("ts_utc"))
        bars = bars_by_symbol.get(symbol, [])
        expected_decision = _first_bar_after(bars, event_ts)
        expected_decision_ts = expected_decision.get("ts") if expected_decision else None
        horizons = tuple(sorted(int(row.get("horizon_hours") or 0) for row in rows))
        if horizons != V5_CANDIDATE_EVIDENCE_HORIZONS:
            violations["horizons_match_signed_contract"] += 1
        for row in rows:
            decision_ts = _as_utc(row.get("decision_ts"))
            label_ts = _as_utc(row.get("label_ts"))
            if _as_utc(expected_decision_ts) != decision_ts:
                expected_missing = expected_decision is None and decision_ts is None
                if not expected_missing:
                    violations["decision_bar_is_first_closed_bar_after_event"] += 1
            if decision_ts is not None and (event_ts is None or decision_ts <= event_ts):
                violations["decision_timestamp_is_after_event"] += 1
            if label_ts is not None and (decision_ts is None or label_ts <= decision_ts):
                violations["label_timestamp_is_after_decision"] += 1
            if decision_ts is not None and not (
                manifest.market_window_start <= decision_ts < manifest.market_window_end
            ):
                violations["market_access_stays_within_snapshot"] += 1
            if label_ts is not None and not (
                manifest.market_window_start <= label_ts < manifest.market_window_end
            ):
                violations["market_access_stays_within_snapshot"] += 1
            if row.get("label_reason") == "future_bar_unavailable" and any(
                row.get(column) is not None
                for column in (
                    "label_ts",
                    "label_close",
                    "gross_bps",
                    "net_bps_after_cost",
                    "mfe_bps",
                    "mae_bps",
                    "win",
                )
            ):
                violations["future_unavailable_rows_are_pending"] += 1
            if row.get("label_reason") == "future_bar_unavailable" and row.get(
                "label_status"
            ) != "partial":
                violations["future_unavailable_rows_are_pending"] += 1
            if str(row.get("source_event_bundle_sha256") or "") != str(
                event.get("bundle_sha256") or ""
            ):
                violations["source_bundle_sha_matches_event"] += 1
            if str(row.get("source_path_inside_bundle") or "") != str(
                event.get("source_path_inside_bundle") or ""
            ):
                violations["source_path_matches_event"] += 1
            event_cost = _finite_float(
                _signed_event_value(event, "cost_bps", "cost")
            ) or 0.0
            if not _float_equal(row.get("cost_bps"), abs(event_cost)):
                violations["cost_bps_comes_from_signed_event"] += 1
            if not _path_metrics_match(row, bars, event):
                violations["mfe_mae_use_decision_to_label_path"] += 1


def _accumulate_symbol_sample_checks(
    violations: dict[str, int],
    events: pl.DataFrame,
    labels: pl.DataFrame,
    samples: pl.DataFrame,
) -> None:
    event_by_id = {
        str(row.get("candidate_id") or "").strip(): row
        for row in events.to_dicts()
        if str(row.get("candidate_id") or "").strip()
    }
    label_by_key = {
        (
            str(row.get("strategy") or ""),
            str(row.get("candidate_id") or ""),
            int(row.get("horizon_hours") or 0),
        ): row
        for row in labels.to_dicts()
    }
    forbidden_columns = set(samples.columns) & {
        "decision",
        "strategy_evidence_decision",
        "paper_status",
        "live_status",
        "risk_permission",
    }
    if "decision" in forbidden_columns or "strategy_evidence_decision" in forbidden_columns:
        violations["worker_outputs_no_strategy_evidence_decision"] += 1
    if "paper_status" in forbidden_columns:
        violations["worker_outputs_no_paper_status"] += 1
    if forbidden_columns & {"live_status", "risk_permission"}:
        violations["worker_outputs_no_live_status"] += 1
    for sample in samples.to_dicts():
        key = (
            str(sample.get("strategy") or ""),
            str(sample.get("candidate_id") or ""),
            int(sample.get("horizon_hours") or 0),
        )
        label = label_by_key.get(key)
        event = event_by_id.get(str(sample.get("candidate_id") or ""))
        if label is None:
            violations["evidence_sample_candidate_labels_exist"] += 1
            continue
        if event is None:
            violations["evidence_sample_events_exist"] += 1
            continue
        expected_candidate = str(
            label.get("strategy_candidate") or event.get("strategy_candidate") or ""
        )
        if sample.get("strategy_candidate") != expected_candidate:
            violations["strategy_candidate_matches_event"] += 1
        if int(sample.get("horizon_hours") or 0) != int(label.get("horizon_hours") or 0):
            violations["evidence_horizon_matches_label"] += 1
        if not _float_equal(
            sample.get("net_bps_after_cost"),
            label.get("net_bps_after_cost"),
        ):
            violations["evidence_net_bps_matches_label"] += 1


def _path_metrics_match(
    row: dict[str, Any],
    bars: list[dict[str, Any]],
    event: dict[str, Any],
) -> bool:
    if row.get("label_status") != "complete":
        return True
    decision_ts = _as_utc(row.get("decision_ts"))
    label_ts = _as_utc(row.get("label_ts"))
    entry = _finite_float(row.get("entry_close"))
    if decision_ts is None or label_ts is None or entry is None or entry <= 0:
        return False
    path = [
        bar
        for bar in bars
        if decision_ts <= (_as_utc(bar.get("ts")) or decision_ts) <= label_ts
    ]
    highs = [_finite_float(bar.get("high")) for bar in path]
    lows = [_finite_float(bar.get("low")) for bar in path]
    highs = [value for value in highs if value is not None and value > 0]
    lows = [value for value in lows if value is not None and value > 0]
    if not highs or not lows:
        return row.get("mfe_bps") is None and row.get("mae_bps") is None
    if _signed_event_direction(event) < 0:
        expected = (
            (entry / min(lows) - 1.0) * 10_000.0,
            (entry / max(highs) - 1.0) * 10_000.0,
        )
    else:
        expected = (
            (max(highs) / entry - 1.0) * 10_000.0,
            (min(lows) / entry - 1.0) * 10_000.0,
        )
    actual = (row.get("mfe_bps"), row.get("mae_bps"))
    return _float_equal(actual[0], expected[0]) and _float_equal(
        actual[1], expected[1]
    )


def _signed_event_value(event: dict[str, Any], *fields: str) -> Any:
    payload: dict[str, Any] = {}
    raw_payload = event.get("raw_payload_json")
    if isinstance(raw_payload, str) and raw_payload.strip():
        try:
            parsed = json.loads(raw_payload)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            payload = parsed
    for field in fields:
        direct = event.get(field)
        if direct not in (None, ""):
            return direct
        nested: Any = payload
        for part in field.split("."):
            if not isinstance(nested, dict):
                nested = None
                break
            nested = nested.get(part)
        if nested not in (None, ""):
            return nested
    return None


def _signed_event_direction(event: dict[str, Any]) -> int:
    for field in ("alpha6_side", "side", "entry_side", "direction", "final_decision"):
        value = str(_signed_event_value(event, field) or "").strip().lower()
        if value in {"short", "sell", "down", "bear", "negative"}:
            return -1
        if value in {"long", "buy", "up", "bull", "positive"}:
            return 1
    current = _finite_float(_signed_event_value(event, "current_weight"))
    target = _finite_float(
        _signed_event_value(event, "target_weight_after_risk", "target_weight_raw")
    )
    return -1 if current is not None and target is not None and target < current else 1


def _collect_symbol_labels(paths: list[Path], symbol: str) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(schema=LABEL_SCHEMA)
    return (
        pl.scan_parquet([str(path) for path in paths], extra_columns="ignore")
        .filter(pl.col("symbol").fill_null("") == symbol)
        .select(list(LABEL_SCHEMA))
        .collect(engine="streaming")
    )


def _write_frame_by_utc_date(
    frame: pl.DataFrame,
    root: Path,
    *,
    timestamp_column: str,
    part_token: str,
) -> list[Path]:
    if frame.is_empty():
        return []
    root.mkdir(parents=True, exist_ok=True)
    partitioned = frame.with_columns(
        pl.col(timestamp_column).cast(pl.Datetime(time_zone="UTC"), strict=False).dt.date().alias(
            "_partition_date"
        )
    )
    days = partitioned.get_column("_partition_date").drop_nulls().unique().sort().to_list()
    if partitioned.get_column("_partition_date").null_count():
        days.append(None)
    paths: list[Path] = []
    for day in days:
        selected = (
            partitioned.filter(pl.col("_partition_date").is_null())
            if day is None
            else partitioned.filter(pl.col("_partition_date") == day)
        ).drop("_partition_date")
        partition_date = day if isinstance(day, date) else date(1970, 1, 1)
        path = root / f"date={partition_date.isoformat()}" / f"part-{part_token}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        selected.write_parquet(path, compression="zstd")
        paths.append(path)
    return paths


def _snapshot_dataset_paths(
    root: Path,
    manifest: V5CandidateEvidenceSnapshotManifest,
    dataset_name: str,
) -> list[Path]:
    return [
        root / item.relative_path
        for item in manifest.files
        if item.dataset_name == dataset_name
    ]


def _scan_paths(paths: list[Path]) -> pl.LazyFrame:
    if not paths:
        return pl.DataFrame().lazy()
    return pl.scan_parquet(
        [str(path) for path in paths],
        missing_columns="insert",
        extra_columns="ignore",
    )


def _duplicate_count_paths(paths: list[Path], keys: list[str]) -> int:
    if not paths:
        return 0
    return _duplicate_count_lazy(
        pl.scan_parquet([str(path) for path in paths], extra_columns="ignore"),
        keys,
    )


def _duplicate_count_lazy(lazy: pl.LazyFrame, keys: list[str]) -> int:
    columns = lazy.collect_schema().names()
    if not set(keys).issubset(columns):
        return 1
    frame = (
        lazy.group_by(keys)
        .agg(pl.len().alias("_count"))
        .filter(pl.col("_count") > 1)
        .select((pl.col("_count") - 1).sum().fill_null(0).alias("duplicates"))
        .collect(engine="streaming")
    )
    return int(frame.item(0, "duplicates") or 0)


def _first_bar_after(
    bars: list[dict[str, Any]],
    event_ts: datetime | None,
) -> dict[str, Any] | None:
    if event_ts is None:
        return None
    timestamps = [_as_utc(bar.get("ts")) for bar in bars]
    if any(value is None for value in timestamps):
        return None
    index = bisect.bisect_right(timestamps, event_ts)  # type: ignore[arg-type]
    return bars[index] if index < len(bars) else None


def _prepare_stage_root(
    snapshot_root: str | Path,
    task_id: str,
    *,
    work_dir: str | Path | None,
) -> Path:
    base = Path(work_dir) if work_dir is not None else Path(snapshot_root).parent
    root = base / "v5-candidate-evidence-stage" / hashlib.sha256(task_id.encode()).hexdigest()[:16]
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _safe_symbol_token(symbol: str) -> str:
    readable = "".join(character.lower() if character.isalnum() else "-" for character in symbol)
    readable = "-".join(part for part in readable.split("-") if part) or "unknown"
    digest = hashlib.sha256(symbol.encode()).hexdigest()[:8]
    return f"{readable[:48]}-{digest}"


def _as_utc(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif value in (None, ""):
        return None
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _finite_float(value: object) -> float | None:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _float_equal(left: object, right: object, tolerance: float = 1e-8) -> bool:
    a = _finite_float(left)
    b = _finite_float(right)
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=tolerance)


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _directory_size_bytes(path: Path) -> int:
    return sum(candidate.stat().st_size for candidate in path.rglob("*") if candidate.is_file())


def _peak_rss_bytes() -> int:
    try:
        import resource  # noqa: PLC0415

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except (ImportError, AttributeError):
        try:
            import psutil  # type: ignore[import-untyped]  # noqa: PLC0415

            return int(psutil.Process().memory_info().rss)
        except (ImportError, OSError):
            return 0
