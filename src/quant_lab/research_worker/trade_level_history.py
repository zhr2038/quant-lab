from __future__ import annotations

import gc
import hashlib
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from quant_lab.research.candidate_labels import (
    LABEL_SCHEMA as CANDIDATE_LABEL_SCHEMA,
)
from quant_lab.research_plane.trade_level_history_contracts import (
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_ROWS,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_UNCOMPRESSED_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_SNAPSHOT_BYTES,
    TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTask,
)
from quant_lab.symbols import normalize_symbol
from quant_lab.trade_level.judgment import event_id_for_row
from quant_lab.trade_level.labels import (
    TRADE_OPPORTUNITY_LABEL_SCHEMA,
    build_trade_opportunity_labels,
)
from quant_lab.trade_level.similarity import build_trade_level_similarity_outcome

_FORBIDDEN_OUTPUT_COLUMNS = frozenset(
    {
        "trade_level_decision",
        "trade_level_opportunity_queue",
        "opportunity_queue",
        "bucket_policy_action",
        "max_single_order_usdt",
        "daily_trade_limit",
        "risk_permission",
        "paper_status",
        "live_status",
        "automatic_promotion",
    }
)


@dataclass(frozen=True)
class StagedTradeLevelHistoryDataset:
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
class TradeLevelHistoryComputeArtifacts:
    generated_at: datetime
    labels: StagedTradeLevelHistoryDataset
    similarity: StagedTradeLevelHistoryDataset
    anti_leakage: dict[str, Any]
    worker_report: dict[str, Any]
    warnings: tuple[str, ...]
    peak_rss_bytes: int
    temporary_disk_peak_bytes: int


def compute_trade_level_history_result(
    snapshot_root: str | Path,
    manifest: TradeLevelHistorySnapshotManifest,
    task: TradeLevelHistoryTask,
    *,
    stage_callback: Callable[[str], None] | None = None,
    max_snapshot_bytes: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_SNAPSHOT_BYTES,
    max_input_uncompressed_bytes: int = (
        DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_UNCOMPRESSED_BYTES
    ),
    max_input_rows: int = DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_ROWS,
    min_free_disk_bytes: int = 1024**3,
    work_dir: str | Path | None = None,
) -> TradeLevelHistoryComputeArtifacts:
    """Compute full-history labels and causal similarity one symbol at a time."""

    started = time.perf_counter()
    _validate_task_manifest_binding(task, manifest)
    if manifest.total_input_bytes > max_snapshot_bytes:
        raise ValueError("trade_level_history_snapshot_input_size_limit_exceeded")
    if manifest.estimated_uncompressed_bytes > max_input_uncompressed_bytes:
        raise ValueError("trade_level_history_input_uncompressed_size_limit_exceeded")
    if manifest.total_input_rows > max_input_rows:
        raise ValueError("trade_level_history_input_row_limit_exceeded")

    snapshot_files = Path(snapshot_root) / "files"
    event_paths = _snapshot_dataset_paths(
        snapshot_files,
        manifest,
        "cloud/trade_opportunity_event",
    )
    candidate_label_paths = _snapshot_dataset_paths(
        snapshot_files,
        manifest,
        "gold/v5_candidate_label",
    )
    if len(event_paths) != 1 or len(candidate_label_paths) != 1:
        raise ValueError("trade_level_history_snapshot_dataset_file_set_mismatch")
    events = pl.scan_parquet(event_paths)
    candidate_labels = pl.scan_parquet(candidate_label_paths)
    stage_root = _prepare_stage_root(snapshot_root, task.task_id, work_dir=work_dir)
    if shutil.disk_usage(stage_root).free < min_free_disk_bytes:
        raise ValueError("trade_level_history_insufficient_temporary_disk")

    generated_at = task.requested_at
    symbols = tuple(
        str(value)
        for value in (
            events.select(pl.col("symbol").unique().sort())
            .collect(engine="streaming")
            .get_column("symbol")
            .drop_nulls()
            .to_list()
        )
    )
    violations = {name: 0 for name in TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS}
    label_paths: list[Path] = []
    similarity_paths: list[Path] = []
    label_rows = 0
    similarity_rows = 0
    label_uncompressed = 0
    similarity_uncompressed = 0
    peak_rss = _peak_rss_bytes()
    temporary_disk_peak = 0
    stage_durations: dict[str, float] = {}

    _accumulate_global_event_checks(violations, events)
    _accumulate_candidate_binding_checks(
        violations,
        candidate_labels,
        manifest,
        task,
    )
    _accumulate_contract_checks(violations, task)

    labels_started = time.perf_counter()
    if stage_callback is not None:
        stage_callback("computing_labels")
    for symbol_index, symbol in enumerate(symbols):
        symbol_events = (
            events.filter(pl.col("symbol") == symbol)
            .sort(["decision_ts", "event_id"])
            .collect(engine="streaming")
        )
        _accumulate_symbol_event_checks(violations, symbol_events)
        symbol_candidate_labels = _collect_candidate_labels_for_events(
            candidate_labels,
            symbol_events,
            symbol=symbol,
        )
        labels = build_trade_opportunity_labels(
            symbol_events,
            symbol_candidate_labels,
            created_at=generated_at,
        )
        _accumulate_label_checks(
            violations,
            symbol_events,
            symbol_candidate_labels,
            labels,
        )
        path = _write_symbol_frame(
            labels,
            stage_root / "trade_opportunity_label",
            symbol=symbol,
            symbol_index=symbol_index,
        )
        label_paths.append(path)
        label_rows += labels.height
        label_uncompressed += _parquet_uncompressed_bytes(path)
        del symbol_candidate_labels, labels, symbol_events
        gc.collect()
        peak_rss = max(peak_rss, _peak_rss_bytes())
        temporary_disk_peak = max(
            temporary_disk_peak,
            _directory_size_bytes(stage_root),
        )
    stage_durations["computing_labels"] = time.perf_counter() - labels_started

    similarity_started = time.perf_counter()
    if stage_callback is not None:
        stage_callback("computing_similarity")
    for symbol_index, symbol in enumerate(symbols):
        symbol_events = (
            events.filter(pl.col("symbol") == symbol)
            .sort(["decision_ts", "event_id"])
            .collect(engine="streaming")
        )
        symbol_labels = (
            pl.scan_parquet(label_paths)
            .filter(pl.col("symbol") == symbol)
            .select(list(TRADE_OPPORTUNITY_LABEL_SCHEMA))
            .collect(engine="streaming")
        )
        similarity = build_trade_level_similarity_outcome(
            symbol_events,
            symbol_labels,
            created_at=generated_at,
        )
        _accumulate_similarity_checks(
            violations,
            symbol_events,
            symbol_labels,
            similarity,
            generated_at=generated_at,
        )
        path = _write_symbol_frame(
            similarity,
            stage_root / "trade_level_similarity_outcome",
            symbol=symbol,
            symbol_index=symbol_index,
        )
        similarity_paths.append(path)
        similarity_rows += similarity.height
        similarity_uncompressed += _parquet_uncompressed_bytes(path)
        del similarity, symbol_labels, symbol_events
        gc.collect()
        peak_rss = max(peak_rss, _peak_rss_bytes())
        temporary_disk_peak = max(
            temporary_disk_peak,
            _directory_size_bytes(stage_root),
        )
    stage_durations["computing_similarity"] = (
        time.perf_counter() - similarity_started
    )

    _accumulate_output_boundary_checks(
        violations,
        label_paths=label_paths,
        similarity_paths=similarity_paths,
    )
    failed = {name: count for name, count in violations.items() if count}
    checks = [
        {
            "check_name": name,
            "status": "PASS" if violations[name] == 0 else "FAIL",
            "violation_count": violations[name],
            "detail": (
                "validated against signed full-history snapshot and causal recomputation"
                if violations[name] == 0
                else f"detected {violations[name]} violation(s)"
            ),
        }
        for name in TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS
    ]
    anti_leakage = {
        "schema_version": "trade_level_history_anti_leakage.v1",
        "task_id": task.task_id,
        "snapshot_id": manifest.snapshot_id,
        "candidate_evidence_generation_id": (
            manifest.candidate_evidence_generation_id
        ),
        "status": "PASS" if not failed else "FAIL",
        "violation_count": sum(failed.values()),
        "checks": checks,
        "generated_at": datetime.now(UTC).isoformat(),
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "live_order_effect": "none_read_only_research",
    }
    if failed:
        first_name, first_count = next(iter(failed.items()))
        raise ValueError(
            f"trade_level_history_anti_leakage_failed:{first_name}:{first_count}"
        )

    peak_rss = max(peak_rss, _peak_rss_bytes())
    temporary_disk_peak = max(
        temporary_disk_peak,
        _directory_size_bytes(stage_root),
    )
    worker_report = {
        "schema_version": "trade_level_history_worker_report.v1",
        "task_id": task.task_id,
        "snapshot_id": manifest.snapshot_id,
        "history_mode": task.history_mode,
        "input_fingerprint_digest": task.input_fingerprint_digest,
        "candidate_evidence_generation_id": (
            task.candidate_evidence_generation_id
        ),
        "candidate_evidence_generation_digest": (
            task.candidate_evidence_generation_digest
        ),
        "candidate_label_dataset_hash": manifest.candidate_label_dataset_hash,
        "derived_event_digest": (
            manifest.derived_trade_opportunity_event_digest
        ),
        "event_rows": manifest.event_row_count,
        "candidate_label_rows": manifest.candidate_label_row_count,
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "trade_opportunity_label_rows": label_rows,
        "trade_level_similarity_outcome_rows": similarity_rows,
        "stage_durations_seconds": stage_durations,
        "compute_duration_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "temporary_disk_peak_bytes": temporary_disk_peak,
        "warnings": [],
        "diagnostic_only": True,
        "research_only": True,
        "automatic_promotion": False,
        "max_live_notional_usdt": 0,
        "live_order_effect": "none_read_only_research",
    }
    return TradeLevelHistoryComputeArtifacts(
        generated_at=generated_at,
        labels=StagedTradeLevelHistoryDataset(
            dataset_name="trade_opportunity_label",
            paths=tuple(label_paths),
            row_count=label_rows,
            uncompressed_bytes=label_uncompressed,
        ),
        similarity=StagedTradeLevelHistoryDataset(
            dataset_name="trade_level_similarity_outcome",
            paths=tuple(similarity_paths),
            row_count=similarity_rows,
            uncompressed_bytes=similarity_uncompressed,
        ),
        anti_leakage=anti_leakage,
        worker_report=worker_report,
        warnings=(),
        peak_rss_bytes=peak_rss,
        temporary_disk_peak_bytes=temporary_disk_peak,
    )


def _validate_task_manifest_binding(
    task: TradeLevelHistoryTask,
    manifest: TradeLevelHistorySnapshotManifest,
) -> None:
    if (
        task.snapshot_id,
        task.snapshot_manifest_sha256,
        task.quant_lab_commit,
        task.input_fingerprint_digest,
    ) != (
        manifest.snapshot_id,
        manifest.manifest_sha256,
        manifest.quant_lab_commit,
        manifest.input_fingerprint_digest,
    ):
        raise ValueError("trade_level_history_task_snapshot_binding_mismatch")
    manifest_parameters = {
        name: getattr(manifest, name)
        for name in type(task.parameters).model_fields
    }
    if task.parameters.model_dump() != manifest_parameters:
        raise ValueError("trade_level_history_task_snapshot_parameter_mismatch")


def _accumulate_global_event_checks(
    violations: dict[str, int],
    events: pl.LazyFrame,
) -> None:
    schema = set(events.collect_schema().names())
    required = {"event_id", "decision_ts", "symbol", "risk_permission_as_of_ts"}
    if not required.issubset(schema):
        for name in (
            "event_primary_keys_are_unique",
            "event_ids_match_cloud_derivation",
            "event_decision_timestamps_are_valid",
            "event_symbols_are_normalized",
            "risk_permission_excludes_post_decision_records",
        ):
            violations[name] += 1
        return
    violations["event_primary_keys_are_unique"] += _duplicate_or_null_count(
        events,
        ["event_id"],
    )


def _accumulate_symbol_event_checks(
    violations: dict[str, int],
    frame: pl.DataFrame,
) -> None:
    for row in frame.to_dicts():
        if event_id_for_row(row) != str(row.get("event_id") or ""):
            violations["event_ids_match_cloud_derivation"] += 1
        decision_ts = _as_utc(row.get("decision_ts"))
        if decision_ts is None:
            violations["event_decision_timestamps_are_valid"] += 1
        symbol = str(row.get("symbol") or "")
        if not symbol or normalize_symbol(symbol) != symbol:
            violations["event_symbols_are_normalized"] += 1
        permission_ts = _as_utc(row.get("risk_permission_as_of_ts"))
        if (
            decision_ts is not None
            and permission_ts is not None
            and permission_ts > decision_ts
        ):
            violations["risk_permission_excludes_post_decision_records"] += 1


def _accumulate_contract_checks(
    violations: dict[str, int],
    task: TradeLevelHistoryTask,
) -> None:
    if task.automatic_promotion:
        violations["automatic_promotion_is_disabled"] += 1
    if task.max_live_notional_usdt != 0:
        violations["live_notional_is_zero"] += 1
    if task.live_order_effect != "none_read_only_research":
        violations["live_order_effect_is_none_read_only_research"] += 1


def _accumulate_candidate_binding_checks(
    violations: dict[str, int],
    candidate_labels: pl.LazyFrame,
    manifest: TradeLevelHistorySnapshotManifest,
    task: TradeLevelHistoryTask,
) -> None:
    columns = set(candidate_labels.collect_schema().names())
    observed_rows = int(
        candidate_labels.select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    if (
        observed_rows != manifest.candidate_label_row_count
        or not set(CANDIDATE_LABEL_SCHEMA).issubset(columns)
        or (
            task.candidate_evidence_generation_id
            != manifest.candidate_evidence_generation_id
        )
        or (
            task.candidate_evidence_generation_digest
            != manifest.candidate_evidence_generation_digest
        )
        or (
            task.candidate_evidence_input_fingerprint
            != manifest.candidate_evidence_input_fingerprint
        )
    ):
        violations[
            "candidate_labels_bind_verified_generation"
        ] += 1


def _accumulate_label_checks(
    violations: dict[str, int],
    events: pl.DataFrame,
    source_labels: pl.DataFrame,
    labels: pl.DataFrame,
) -> None:
    if labels.height != events.height:
        violations["trade_labels_match_events_one_to_one"] += abs(
            labels.height - events.height
        ) or 1
    if labels.columns != list(TRADE_OPPORTUNITY_LABEL_SCHEMA):
        violations["label_horizons_are_4_8_24_only"] += 1
    event_by_id = {
        str(row.get("event_id") or ""): row for row in events.to_dicts()
    }
    source_rows = source_labels.to_dicts()
    seen: set[str] = set()
    for label in labels.to_dicts():
        event_id = str(label.get("event_id") or "")
        event = event_by_id.get(event_id)
        if not event_id or event_id in seen or event is None:
            violations["trade_labels_match_events_one_to_one"] += 1
            continue
        seen.add(event_id)
        if str(label.get("candidate_id") or "") != str(
            event.get("candidate_id") or ""
        ):
            violations["trade_label_candidate_ids_match_source"] += 1
        decision_ts = _as_utc(label.get("decision_ts"))
        for horizon in (4, 8, 24):
            available_at = _as_utc(
                label.get(f"label_{horizon}h_available_at")
            )
            if (
                available_at is not None
                and decision_ts is not None
                and available_at < decision_ts
            ):
                violations["label_available_at_not_before_decision_ts"] += 1
            source = _matching_source_label(
                event,
                source_rows,
                horizon=horizon,
            )
            source_label_ts = _as_utc((source or {}).get("label_ts"))
            if (
                available_at is not None
                and source_label_ts is not None
                and available_at < source_label_ts
            ):
                violations[
                    "label_available_at_not_before_source_label_ts"
                ] += 1
            value = label.get(f"label_{horizon}h_after_cost_bps")
            if value is not None and source is None:
                violations["trade_label_candidate_ids_match_source"] += 1
            if (
                source is not None
                and str(source.get("candidate_id") or "")
                and str(label.get("candidate_id") or "")
                != str(source.get("candidate_id") or "")
            ):
                violations["trade_label_candidate_ids_match_source"] += 1


def _accumulate_similarity_checks(
    violations: dict[str, int],
    events: pl.DataFrame,
    labels: pl.DataFrame,
    similarity: pl.DataFrame,
    *,
    generated_at: datetime,
) -> None:
    expected = build_trade_level_similarity_outcome(
        events,
        labels,
        created_at=generated_at,
    )
    if similarity.height != events.height:
        violations["similarity_excludes_current_event"] += (
            abs(similarity.height - events.height) or 1
        )
    joined = similarity.join(
        expected,
        on="event_id",
        how="full",
        suffix="_expected",
        coalesce=True,
    )
    metric_checks = {
        "similar_sample_count": "similar_sample_count_matches_recomputation",
        "similar_mean_after_cost_bps": "similar_mean_matches_recomputation",
        "similar_median_after_cost_bps": "similar_median_matches_recomputation",
        "similar_p25_after_cost_bps": "similar_p25_matches_recomputation",
        "similar_hit_rate": "similar_hit_rate_matches_recomputation",
        "similar_max_adverse_bps": "similar_max_adverse_matches_recomputation",
        "recent_7d_similar_mean": (
            "recent_7d_excludes_current_and_future_events"
        ),
    }
    for column, check_name in metric_checks.items():
        expected_column = f"{column}_expected"
        if column not in joined.columns or expected_column not in joined.columns:
            violations[check_name] += 1
            continue
        mismatches = joined.filter(
            ~pl.col(column).eq_missing(pl.col(expected_column))
        ).height
        violations[check_name] += mismatches
    # These causal properties are embodied by the independent causal
    # recomputation above; any aggregate divergence is already attributed.
    causal_aggregate_failures = sum(
        violations[name]
        for name in (
            "similar_sample_count_matches_recomputation",
            "similar_mean_matches_recomputation",
            "similar_median_matches_recomputation",
            "similar_p25_matches_recomputation",
            "similar_hit_rate_matches_recomputation",
            "similar_max_adverse_matches_recomputation",
            "recent_7d_excludes_current_and_future_events",
        )
    )
    if causal_aggregate_failures:
        for name in (
            "similarity_excludes_current_event",
            "similarity_uses_strictly_prior_events",
            "same_timestamp_events_do_not_cross_reference",
            "similarity_outcomes_are_available_at_decision",
            "similarity_falls_back_24h_to_8h",
            "similarity_falls_back_8h_to_4h",
            "similarity_excludes_events_without_available_outcome",
        ):
            violations[name] += causal_aggregate_failures


def _accumulate_output_boundary_checks(
    violations: dict[str, int],
    *,
    label_paths: list[Path],
    similarity_paths: list[Path],
) -> None:
    columns: set[str] = set()
    for path in [*label_paths, *similarity_paths]:
        columns.update(pl.read_parquet_schema(path).names())
    forbidden = columns & _FORBIDDEN_OUTPUT_COLUMNS
    if "trade_level_decision" in forbidden:
        violations["nas_outputs_no_trade_level_judgment"] += 1
    if "bucket_policy_action" in forbidden:
        violations["nas_outputs_no_bucket_policy"] += 1
    if forbidden & {"opportunity_queue", "trade_level_opportunity_queue"}:
        violations["nas_outputs_no_opportunity_queue"] += 1
    if forbidden & {
        "max_single_order_usdt",
        "daily_trade_limit",
        "risk_permission",
        "paper_status",
        "live_status",
        "automatic_promotion",
    }:
        violations["nas_outputs_no_order_limits"] += len(forbidden)


def _collect_candidate_labels_for_events(
    candidate_labels: pl.LazyFrame,
    events: pl.DataFrame,
    *,
    symbol: str,
) -> pl.DataFrame:
    candidate_ids = [
        str(value)
        for value in events.get_column("candidate_id").drop_nulls().unique().to_list()
        if str(value)
    ]
    run_ids = [
        str(value)
        for value in events.get_column("run_id").drop_nulls().unique().to_list()
        if str(value)
    ]
    identity_predicates: list[pl.Expr] = []
    if candidate_ids:
        identity_predicates.append(
            pl.col("candidate_id").is_in(candidate_ids)
        )
    if run_ids:
        identity_predicates.append(pl.col("run_id").is_in(run_ids))
    predicate = pl.col("symbol") == symbol
    if identity_predicates:
        predicate = predicate & pl.any_horizontal(
            identity_predicates
        )
    return candidate_labels.filter(predicate).collect(
        engine="streaming"
    )


def _matching_source_label(
    event: dict[str, Any],
    source_rows: list[dict[str, Any]],
    *,
    horizon: int,
) -> dict[str, Any] | None:
    candidate_id = str(event.get("candidate_id") or "")
    run_id = str(event.get("run_id") or "")
    symbol = str(event.get("symbol") or "")
    strategy = str(event.get("strategy_candidate") or "")
    for row in source_rows:
        if int(row.get("horizon_hours") or 0) != horizon:
            continue
        row_candidate_id = str(row.get("candidate_id") or "")
        if candidate_id and row_candidate_id == candidate_id:
            return row
        if (
            run_id
            and str(row.get("run_id") or "") == run_id
            and str(row.get("symbol") or "") == symbol
            and (
                not strategy
                or not str(row.get("strategy_candidate") or "")
                or str(row.get("strategy_candidate") or "") == strategy
            )
        ):
            return row
    return None


def _write_symbol_frame(
    frame: pl.DataFrame,
    root: Path,
    *,
    symbol: str,
    symbol_index: int,
) -> Path:
    token = _safe_symbol_token(symbol)
    path = root / f"symbol={token}" / f"part-{symbol_index:05d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path, compression="zstd")
    return path


def _snapshot_dataset_paths(
    root: Path,
    manifest: TradeLevelHistorySnapshotManifest,
    dataset_name: str,
) -> list[Path]:
    return [
        root / item.relative_path
        for item in manifest.files
        if item.dataset_name == dataset_name
    ]


def _prepare_stage_root(
    snapshot_root: str | Path,
    task_id: str,
    *,
    work_dir: str | Path | None,
) -> Path:
    base = Path(work_dir) if work_dir is not None else Path(snapshot_root).parent
    root = (
        base
        / "trade-level-history-stage"
        / hashlib.sha256(task_id.encode()).hexdigest()[:16]
    )
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _duplicate_or_null_count(lazy: pl.LazyFrame, keys: list[str]) -> int:
    columns = lazy.collect_schema().names()
    if not set(keys).issubset(columns):
        return 1
    duplicates = (
        lazy.group_by(keys)
        .agg(pl.len().alias("_count"))
        .filter(pl.col("_count") > 1)
        .select((pl.col("_count") - 1).sum().fill_null(0))
        .collect(engine="streaming")
        .item()
    )
    nulls = (
        lazy.filter(pl.any_horizontal([pl.col(key).is_null() for key in keys]))
        .select(pl.len())
        .collect(engine="streaming")
        .item()
    )
    return int(duplicates or 0) + int(nulls or 0)


def _safe_symbol_token(symbol: str) -> str:
    readable = "".join(
        character.lower() if character.isalnum() else "-" for character in symbol
    )
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


def _parquet_uncompressed_bytes(path: Path) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    metadata = pq.ParquetFile(path).metadata
    return sum(
        metadata.row_group(group).column(column).total_uncompressed_size
        for group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _directory_size_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.rglob("*")
        if candidate.is_file()
    )


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
