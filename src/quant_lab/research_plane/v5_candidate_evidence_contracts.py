from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

V5_CANDIDATE_EVIDENCE_TASK_TYPE = "v5_candidate_evidence"
V5_CANDIDATE_EVIDENCE_TASK_SCHEMA = "quant_lab_v5_candidate_evidence_task.v1"
V5_CANDIDATE_EVIDENCE_SNAPSHOT_SCHEMA = "quant_lab_v5_candidate_evidence_snapshot.v1"
V5_CANDIDATE_EVIDENCE_RESULT_SCHEMA = "quant_lab_v5_candidate_evidence_result.v1"
V5_CANDIDATE_EVIDENCE_RECEIPT_SCHEMA = "quant_lab_v5_candidate_evidence_receipt.v1"
V5_CANDIDATE_EVIDENCE_FINGERPRINT_SCHEMA = "v5_candidate_evidence_input_fingerprint.v1"
V5_CANDIDATE_EVIDENCE_PROJECTION_VERSION = "v5_candidate_evidence_projection.v1"
V5_CANDIDATE_EVIDENCE_HORIZONS = (4, 8, 12, 24, 48, 72, 120)

DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_SNAPSHOT_BYTES = 512 * 1024**2
DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_INPUT_UNCOMPRESSED_BYTES = 1024**3
DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_BYTES = 256 * 1024**2
DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_UNCOMPRESSED_BYTES = 512 * 1024**2
DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_BYTES = 64 * 1024**2
DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_UNCOMPRESSED_BYTES = 128 * 1024**2
DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_FILE_COUNT = 5_000

V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS = (
    "strategy",
    "candidate_id",
    "horizon_hours",
)
V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS = (
    "strategy",
    "source_type",
    "candidate_id",
    "symbol",
    "strategy_candidate",
    "horizon_hours",
    "source_event_key",
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class V5CandidateEvidenceTaskPayload(_StrictModel):
    as_of_date: date
    mode: Literal["incremental"] = "incremental"
    lookback_days: Literal[8] = 8
    horizon_hours: tuple[int, ...] = V5_CANDIDATE_EVIDENCE_HORIZONS
    include_historical_outcomes: Literal[False] = False
    candidate_label_schema_version: Literal["v5.candidate_label.v1"] = (
        "v5.candidate_label.v1"
    )
    strategy_evidence_version: Literal["strategy-evidence-v0.1"] = (
        "strategy-evidence-v0.1"
    )

    @model_validator(mode="after")
    def validate_payload(self) -> V5CandidateEvidenceTaskPayload:
        _require_exact_horizons(self.horizon_hours)
        return self


class V5CandidateEvidenceSourceFileIdentity(_StrictModel):
    dataset_name: Literal[
        "silver/v5_candidate_event",
        "silver/market_bar",
        "silver/v5_run_summary",
    ]
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    row_count: int = Field(ge=0)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    uncompressed_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_identity(self) -> V5CandidateEvidenceSourceFileIdentity:
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_optional_bounds(self.min_ts, self.max_ts, "source file")
        return self


class V5CandidateEvidenceInputFingerprint(_StrictModel):
    schema_version: Literal["v5_candidate_evidence_input_fingerprint.v1"] = (
        V5_CANDIDATE_EVIDENCE_FINGERPRINT_SCHEMA
    )
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    as_of_date: date
    mode: Literal["incremental"] = "incremental"
    lookback_days: Literal[8] = 8
    horizon_hours: tuple[int, ...] = V5_CANDIDATE_EVIDENCE_HORIZONS
    candidate_label_schema_version: Literal["v5.candidate_label.v1"] = (
        "v5.candidate_label.v1"
    )
    strategy_evidence_version: Literal["strategy-evidence-v0.1"] = (
        "strategy-evidence-v0.1"
    )
    projection_version: Literal["v5_candidate_evidence_projection.v1"] = (
        V5_CANDIDATE_EVIDENCE_PROJECTION_VERSION
    )
    event_window_start: datetime
    event_window_end: datetime
    market_window_start: datetime
    market_window_end: datetime
    candidate_event_digest: str = Field(min_length=64, max_length=64)
    market_bar_digest: str = Field(min_length=64, max_length=64)
    run_summary_digest: str = Field(min_length=64, max_length=64)
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    candidate_event_file_count: int = Field(ge=0)
    market_bar_file_count: int = Field(ge=0)
    run_summary_file_count: int = Field(ge=0)
    candidate_event_row_count: int = Field(ge=0)
    market_bar_row_count: int = Field(ge=0)
    run_summary_row_count: int = Field(ge=0)
    candidate_event_bytes: int = Field(ge=0)
    market_bar_bytes: int = Field(ge=0)
    run_summary_bytes: int = Field(ge=0)
    estimated_uncompressed_bytes: int = Field(ge=0)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_fingerprint(self) -> V5CandidateEvidenceInputFingerprint:
        _require_commit(self.quant_lab_commit)
        _require_exact_horizons(self.horizon_hours)
        for field_name in (
            "candidate_event_digest",
            "market_bar_digest",
            "run_summary_digest",
            "input_fingerprint_digest",
        ):
            _require_sha(getattr(self, field_name), field_name)
        _require_utc(self.event_window_start, "event_window_start")
        _require_utc(self.event_window_end, "event_window_end")
        _require_utc(self.market_window_start, "market_window_start")
        _require_utc(self.market_window_end, "market_window_end")
        _require_utc(self.observed_at, "observed_at")
        if self.event_window_start >= self.event_window_end:
            raise ValueError("v5 candidate evidence event window invalid")
        if self.market_window_start >= self.market_window_end:
            raise ValueError("v5 candidate evidence market window invalid")
        return self


class V5CandidateEvidenceSnapshotFile(_StrictModel):
    dataset_name: Literal[
        "silver/v5_candidate_event",
        "silver/market_bar",
        "silver/v5_run_summary",
    ]
    source_relative_path: str = Field(min_length=1, max_length=1024)
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    uncompressed_bytes: int = Field(ge=0)
    media_type: Literal["application/x-parquet"] = "application/x-parquet"

    @model_validator(mode="after")
    def validate_file(self) -> V5CandidateEvidenceSnapshotFile:
        _require_relative_path(self.source_relative_path)
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_optional_bounds(self.min_ts, self.max_ts, "snapshot file")
        return self


class V5CandidateEvidenceSnapshotManifest(_StrictModel):
    schema_version: Literal["quant_lab_v5_candidate_evidence_snapshot.v1"] = (
        V5_CANDIDATE_EVIDENCE_SNAPSHOT_SCHEMA
    )
    task_type: Literal["v5_candidate_evidence"] = V5_CANDIDATE_EVIDENCE_TASK_TYPE
    snapshot_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    as_of_date: date
    mode: Literal["incremental"] = "incremental"
    lookback_days: Literal[8] = 8
    horizon_hours: tuple[int, ...] = V5_CANDIDATE_EVIDENCE_HORIZONS
    include_historical_outcomes: Literal[False] = False
    candidate_label_schema_version: Literal["v5.candidate_label.v1"] = (
        "v5.candidate_label.v1"
    )
    strategy_evidence_version: Literal["strategy-evidence-v0.1"] = (
        "strategy-evidence-v0.1"
    )
    projection_version: Literal["v5_candidate_evidence_projection.v1"] = (
        V5_CANDIDATE_EVIDENCE_PROJECTION_VERSION
    )
    candidate_event_digest: str = Field(min_length=64, max_length=64)
    market_bar_digest: str = Field(min_length=64, max_length=64)
    run_summary_digest: str = Field(min_length=64, max_length=64)
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    event_window_start: datetime
    event_window_end: datetime
    market_window_start: datetime
    market_window_end: datetime
    min_event_ts: datetime | None = None
    max_event_ts: datetime | None = None
    candidate_symbols: tuple[str, ...] = ()
    candidate_run_ids: tuple[str, ...] = ()
    selected_timeframes: tuple[tuple[str, str], ...] = ()
    candidate_event_row_count: int = Field(ge=0)
    candidate_event_file_count: int = Field(ge=0)
    market_bar_row_count: int = Field(ge=0)
    market_bar_file_count: int = Field(ge=0)
    run_summary_row_count: int = Field(ge=0)
    run_summary_file_count: int = Field(ge=0)
    datasets: tuple[str, ...]
    files: tuple[V5CandidateEvidenceSnapshotFile, ...]
    total_input_bytes: int = Field(ge=0)
    total_input_rows: int = Field(ge=0)
    estimated_uncompressed_bytes: int = Field(ge=0)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> V5CandidateEvidenceSnapshotManifest:
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_utc(self.generated_at, "generated_at")
        _require_commit(self.quant_lab_commit)
        _require_exact_horizons(self.horizon_hours)
        for field_name in (
            "candidate_event_digest",
            "market_bar_digest",
            "run_summary_digest",
            "input_fingerprint_digest",
            "manifest_sha256",
        ):
            _require_sha(getattr(self, field_name), field_name)
        for field_name in (
            "event_window_start",
            "event_window_end",
            "market_window_start",
            "market_window_end",
        ):
            _require_utc(getattr(self, field_name), field_name)
        _require_optional_bounds(self.min_event_ts, self.max_event_ts, "candidate event")
        if self.candidate_symbols != tuple(sorted(set(self.candidate_symbols))):
            raise ValueError("v5 candidate evidence symbols must be sorted and unique")
        if self.candidate_run_ids != tuple(sorted(set(self.candidate_run_ids))):
            raise ValueError("v5 candidate evidence run ids must be sorted and unique")
        if self.selected_timeframes != tuple(sorted(set(self.selected_timeframes))):
            raise ValueError("v5 candidate evidence timeframes must be sorted and unique")
        if any(symbol not in self.candidate_symbols for symbol, _ in self.selected_timeframes):
            raise ValueError("v5 candidate evidence timeframe symbol mismatch")
        expected_datasets = (
            "silver/v5_candidate_event",
            "silver/market_bar",
            "silver/v5_run_summary",
        )
        if self.datasets != expected_datasets:
            raise ValueError("v5 candidate evidence snapshot dataset set mismatch")
        if self.total_input_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("v5 candidate evidence snapshot byte count mismatch")
        if self.total_input_rows != sum(item.row_count for item in self.files):
            raise ValueError("v5 candidate evidence snapshot row count mismatch")
        if self.estimated_uncompressed_bytes != sum(
            item.uncompressed_bytes for item in self.files
        ):
            raise ValueError("v5 candidate evidence snapshot uncompressed count mismatch")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("v5 candidate evidence snapshot paths must be unique")
        return self


class V5CandidateEvidenceTask(V5CandidateEvidenceTaskPayload):
    schema_version: Literal["quant_lab_v5_candidate_evidence_task.v1"] = (
        V5_CANDIDATE_EVIDENCE_TASK_SCHEMA
    )
    task_type: Literal["v5_candidate_evidence"] = V5_CANDIDATE_EVIDENCE_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    previous_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    previous_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    requested_at: datetime
    lease_seconds: int = Field(default=7200, ge=60, le=24 * 60 * 60)
    max_attempts: int = Field(default=3, ge=1, le=10)
    signature_key_id: str = Field(min_length=1, max_length=120)
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task(self) -> V5CandidateEvidenceTask:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_sha(self.input_fingerprint_digest, "input_fingerprint_digest")
        _require_commit(self.quant_lab_commit)
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_utc(self.requested_at, "requested_at")
        _require_identifier(self.signature_key_id, "signature_key_id")
        _require_previous_generation(
            self.previous_generation_id,
            self.previous_generation_digest,
        )
        return self

    @property
    def parameters(self) -> V5CandidateEvidenceTaskPayload:
        return V5CandidateEvidenceTaskPayload(
            as_of_date=self.as_of_date,
            mode=self.mode,
            lookback_days=self.lookback_days,
            horizon_hours=self.horizon_hours,
            include_historical_outcomes=self.include_historical_outcomes,
            candidate_label_schema_version=self.candidate_label_schema_version,
            strategy_evidence_version=self.strategy_evidence_version,
        )


class V5CandidateEvidenceOutputDataset(_StrictModel):
    dataset_name: Literal[
        "v5_candidate_label_delta",
        "strategy_evidence_sample_delta",
    ]
    partition_date: date
    part_number: int = Field(ge=0)
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=1)
    uncompressed_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    partition_identity: str = Field(min_length=64, max_length=64)
    primary_keys: tuple[str, ...]
    publish_mode: Literal["keyed_upsert"] = "keyed_upsert"
    empty_result_semantics: Literal["preserve"] = "preserve"

    @model_validator(mode="after")
    def validate_output(self) -> V5CandidateEvidenceOutputDataset:
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_sha(self.partition_identity, "partition_identity")
        _require_optional_bounds(self.min_ts, self.max_ts, "result partition")
        expected_keys = {
            "v5_candidate_label_delta": V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS,
            "strategy_evidence_sample_delta": (
                V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS
            ),
        }
        if self.primary_keys != expected_keys[self.dataset_name]:
            raise ValueError(f"{self.dataset_name} primary keys mismatch")
        prefix = (
            f"outputs/{self.dataset_name}/date={self.partition_date.isoformat()}/"
        )
        if not self.relative_path.startswith(prefix):
            raise ValueError(f"{self.dataset_name} partition path mismatch")
        if self.row_count == 0 and (self.min_ts is not None or self.max_ts is not None):
            raise ValueError(f"{self.dataset_name} empty partition bounds must be null")
        return self


class V5CandidateEvidenceReportFile(_StrictModel):
    relative_path: Literal[
        "reports/v5_candidate_evidence_worker_report.json",
        "reports/v5_candidate_evidence_anti_leakage.json",
    ]
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=1)
    media_type: Literal["application/json"] = "application/json"

    @model_validator(mode="after")
    def validate_report(self) -> V5CandidateEvidenceReportFile:
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        return self


class V5CandidateEvidenceAntiLeakageCheck(_StrictModel):
    check_name: str = Field(min_length=1, max_length=180)
    status: Literal["PASS"] = "PASS"
    violation_count: Literal[0] = 0
    detail: str = Field(min_length=1, max_length=2000)

    @field_validator("check_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        _require_identifier(value, "check_name")
        return value


class V5CandidateEvidenceResultManifest(_StrictModel):
    schema_version: Literal["quant_lab_v5_candidate_evidence_result.v1"] = (
        V5_CANDIDATE_EVIDENCE_RESULT_SCHEMA
    )
    task_type: Literal["v5_candidate_evidence"] = V5_CANDIDATE_EVIDENCE_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    candidate_event_digest: str = Field(min_length=64, max_length=64)
    market_bar_digest: str = Field(min_length=64, max_length=64)
    run_summary_digest: str = Field(min_length=64, max_length=64)
    previous_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    previous_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    as_of_date: date
    mode: Literal["incremental"] = "incremental"
    lookback_days: Literal[8] = 8
    horizon_hours: tuple[int, ...] = V5_CANDIDATE_EVIDENCE_HORIZONS
    include_historical_outcomes: Literal[False] = False
    candidate_label_schema_version: Literal["v5.candidate_label.v1"] = (
        "v5.candidate_label.v1"
    )
    strategy_evidence_version: Literal["strategy-evidence-v0.1"] = (
        "strategy-evidence-v0.1"
    )
    generation_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    completed_at: datetime
    outputs: tuple[V5CandidateEvidenceOutputDataset, ...] = Field(min_length=2)
    reports: tuple[V5CandidateEvidenceReportFile, ...] = Field(min_length=2, max_length=2)
    anti_leakage_checks: tuple[V5CandidateEvidenceAntiLeakageCheck, ...] = Field(
        min_length=28
    )
    anti_leakage_status: Literal["PASS"] = "PASS"
    anti_leakage_violation_count: Literal[0] = 0
    warnings: tuple[str, ...] = ()
    input_bytes: int = Field(ge=0)
    input_uncompressed_bytes: int = Field(ge=0)
    cache_hit_bytes: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    output_uncompressed_bytes: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    temporary_disk_peak_bytes: int = Field(ge=0)
    compute_duration_seconds: float = Field(ge=0)
    worker_key_id: str = Field(min_length=1, max_length=120)
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    requires_cloud_validation: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> V5CandidateEvidenceResultManifest:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.generation_id, "generation_id")
        _require_commit(self.quant_lab_commit)
        _require_commit(self.worker_commit)
        for field_name in (
            "snapshot_manifest_sha256",
            "input_fingerprint_digest",
            "candidate_event_digest",
            "market_bar_digest",
            "run_summary_digest",
        ):
            _require_sha(getattr(self, field_name), field_name)
        _require_previous_generation(
            self.previous_generation_id,
            self.previous_generation_digest,
        )
        _require_exact_horizons(self.horizon_hours)
        _require_utc(self.generated_at, "generated_at")
        _require_utc(self.completed_at, "completed_at")
        if self.completed_at < self.generated_at:
            raise ValueError("v5 candidate evidence result time order invalid")
        output_names = {item.dataset_name for item in self.outputs}
        if output_names != {
            "v5_candidate_label_delta",
            "strategy_evidence_sample_delta",
        }:
            raise ValueError("v5 candidate evidence output set mismatch")
        paths = [item.relative_path for item in self.outputs]
        if len(paths) != len(set(paths)):
            raise ValueError("v5 candidate evidence output paths must be unique")
        report_paths = [item.relative_path for item in self.reports]
        if set(report_paths) != {
            "reports/v5_candidate_evidence_worker_report.json",
            "reports/v5_candidate_evidence_anti_leakage.json",
        }:
            raise ValueError("v5 candidate evidence report set mismatch")
        check_names = [item.check_name for item in self.anti_leakage_checks]
        if len(check_names) != len(set(check_names)):
            raise ValueError("v5 candidate evidence checks must be unique")
        expected_bytes = sum(item.size_bytes for item in self.outputs) + sum(
            item.size_bytes for item in self.reports
        )
        if self.output_bytes != expected_bytes:
            raise ValueError("v5 candidate evidence output byte count mismatch")
        expected_uncompressed = sum(item.uncompressed_bytes for item in self.outputs) + sum(
            item.size_bytes for item in self.reports
        )
        if self.output_uncompressed_bytes != expected_uncompressed:
            raise ValueError("v5 candidate evidence uncompressed byte count mismatch")
        return self


class V5CandidateEvidenceWorkerReceipt(_StrictModel):
    schema_version: Literal["quant_lab_v5_candidate_evidence_receipt.v1"] = (
        V5_CANDIDATE_EVIDENCE_RECEIPT_SCHEMA
    )
    task_type: Literal["v5_candidate_evidence"] = V5_CANDIDATE_EVIDENCE_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    worker_id: str = Field(min_length=1, max_length=180)
    worker_commit: str = Field(min_length=40, max_length=40)
    state: Literal["completed", "failed"]
    claimed_at: datetime
    completed_at: datetime
    result_manifest_sha256: str = Field(min_length=64, max_length=64)
    output_rows: int = Field(ge=0)
    input_bytes: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    cache_hit_bytes: int = Field(ge=0)
    anti_leakage_status: Literal["PASS"] = "PASS"
    anti_leakage_violation_count: Literal[0] = 0
    error_code: str | None = None
    worker_key_id: str = Field(min_length=1, max_length=120)
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> V5CandidateEvidenceWorkerReceipt:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.worker_id, "worker_id")
        _require_commit(self.worker_commit)
        _require_sha(self.result_manifest_sha256, "result_manifest_sha256")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        if self.completed_at < self.claimed_at:
            raise ValueError("v5 candidate evidence receipt time order invalid")
        return self


def _require_exact_horizons(values: tuple[int, ...]) -> None:
    if values != V5_CANDIDATE_EVIDENCE_HORIZONS:
        raise ValueError("v5 candidate evidence horizons must match the production contract")


def _require_sha(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase sha256")


def _require_commit(value: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("quant_lab_commit must be a full lowercase git sha")


def _require_identifier(value: str, field_name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"{field_name} contains unsafe characters")


def _require_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in value:
        raise ValueError("unsafe relative path")
    if any(part in {"", "."} for part in path.parts):
        raise ValueError("unsafe relative path")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")


def _require_optional_bounds(
    lower: datetime | None,
    upper: datetime | None,
    field_name: str,
) -> None:
    if (lower is None) != (upper is None):
        raise ValueError(f"{field_name} bounds must be complete")
    if lower is None:
        return
    _require_utc(lower, f"{field_name} min_ts")
    _require_utc(upper, f"{field_name} max_ts")
    if lower > upper:
        raise ValueError(f"{field_name} bounds invalid")


def _require_previous_generation(
    generation_id: str | None,
    generation_digest: str | None,
) -> None:
    if (generation_id is None) != (generation_digest is None):
        raise ValueError("previous generation binding must be complete")
    if generation_id is not None:
        _require_identifier(generation_id, "previous_generation_id")
        _require_sha(generation_digest, "previous_generation_digest")
