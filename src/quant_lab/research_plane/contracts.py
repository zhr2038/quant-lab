from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RESEARCH_SNAPSHOT_SCHEMA = "quant_lab_research_snapshot.v1"
RESEARCH_TASK_SCHEMA = "quant_lab_research_task.v1"
RESEARCH_RESULT_SCHEMA = "quant_lab_research_result.v1"
RESEARCH_RECEIPT_SCHEMA = "quant_lab_research_receipt.v1"
RESEARCH_STATUS_SCHEMA = "quant_lab_research_status.v1"
RESEARCH_VALIDATION_SCHEMA = "quant_lab_research_validation.v1"
ENTRY_QUALITY_HISTORY_TASK_TYPE = "entry_quality_history"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchTaskState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    SYNCING = "syncing"
    COMPUTING = "computing"
    VALIDATING_ON_NAS = "validating_on_nas"
    UPLOADING = "uploading"
    VALIDATING_ON_CLOUD = "validating_on_cloud"
    PUBLISHING = "publishing"
    COMPLETED = "completed"
    REJECTED = "rejected"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ResearchDatasetReference(StrictModel):
    dataset_name: str = Field(min_length=1, max_length=180)
    source_relative_path: str = Field(min_length=1, max_length=1024)
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    media_type: Literal["application/x-parquet"] = "application/x-parquet"

    @field_validator("source_relative_path", "relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        _require_safe_relative_path(value)
        return value

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        _require_sha(value, "sha256")
        return value

    @field_validator("min_ts", "max_ts")
    @classmethod
    def validate_optional_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            _require_utc(value, "dataset timestamp")
        return value


class ResearchSnapshotManifest(StrictModel):
    schema_version: Literal["quant_lab_research_snapshot.v1"] = RESEARCH_SNAPSHOT_SCHEMA
    snapshot_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    entry_quality_schema_version: str = Field(min_length=1, max_length=80)
    datasets: list[str]
    files: list[ResearchDatasetReference]
    total_input_bytes: int = Field(ge=0)
    total_input_rows: int = Field(ge=0)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> ResearchSnapshotManifest:
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_utc(self.generated_at, "generated_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.manifest_sha256, "manifest_sha256")
        _require_identifier(self.signature_key_id, "signature_key_id")
        if self.total_input_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("snapshot total_input_bytes mismatch")
        if self.total_input_rows != sum(item.row_count for item in self.files):
            raise ValueError("snapshot total_input_rows mismatch")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("snapshot relative paths must be unique")
        if not self.datasets or len(self.datasets) != len(set(self.datasets)):
            raise ValueError("snapshot datasets must be non-empty and unique")
        if any(item.dataset_name not in self.datasets for item in self.files):
            raise ValueError("snapshot file references undeclared dataset")
        return self


class EntryQualityHistoryTaskParameters(StrictModel):
    start_date: date
    end_date: date
    mode: Literal["full", "recent_7d", "recent_30d", "walk_forward"] = "recent_30d"
    cost_mode: Literal["conservative", "quant_lab"] = "conservative"
    window_hours: int = Field(default=24, ge=1, le=24 * 30)

    @model_validator(mode="after")
    def validate_dates(self) -> EntryQualityHistoryTaskParameters:
        if self.end_date < self.start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        return self


class ResearchTask(StrictModel):
    schema_version: Literal["quant_lab_research_task.v1"] = RESEARCH_TASK_SCHEMA
    task_type: Literal["entry_quality_history"] = ENTRY_QUALITY_HISTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    start_date: date
    end_date: date
    mode: Literal["full", "recent_7d", "recent_30d", "walk_forward"] = "recent_30d"
    cost_mode: Literal["conservative", "quant_lab"] = "conservative"
    window_hours: int = Field(default=24, ge=1, le=24 * 30)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    entry_quality_schema_version: str = Field(min_length=1, max_length=80)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    requested_at: datetime
    lease_seconds: int = Field(default=3600, ge=60, le=24 * 60 * 60)
    max_attempts: int = Field(default=3, ge=1, le=10)
    signature_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task(self) -> ResearchTask:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_utc(self.requested_at, "requested_at")
        _require_identifier(self.signature_key_id, "signature_key_id")
        if self.end_date < self.start_date:
            raise ValueError("end_date must be greater than or equal to start_date")
        return self

    @property
    def parameters(self) -> EntryQualityHistoryTaskParameters:
        return EntryQualityHistoryTaskParameters(
            start_date=self.start_date,
            end_date=self.end_date,
            mode=self.mode,
            cost_mode=self.cost_mode,
            window_hours=self.window_hours,
        )


class ResearchOutputDataset(StrictModel):
    dataset_name: str = Field(min_length=1, max_length=180)
    relative_path: str = Field(min_length=1, max_length=1024)
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    sha256: str = Field(min_length=64, max_length=64)
    row_count: int = Field(ge=0)
    size_bytes: int = Field(ge=0)
    publish_mode: Literal["window_upsert", "window_replace"]
    primary_keys: list[str]
    window_keys: list[str]
    empty_result_semantics: Literal["clear_window", "preserve"]

    @model_validator(mode="after")
    def validate_output(self) -> ResearchOutputDataset:
        _require_safe_relative_path(self.relative_path)
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_sha(self.sha256, "sha256")
        if len(self.primary_keys) != len(set(self.primary_keys)):
            raise ValueError("primary_keys must be unique")
        if len(self.window_keys) != len(set(self.window_keys)):
            raise ValueError("window_keys must be unique")
        return self


class ResearchOutputFile(StrictModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_file(self) -> ResearchOutputFile:
        _require_safe_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        return self


class ResearchResultManifest(StrictModel):
    schema_version: Literal["quant_lab_research_result.v1"] = RESEARCH_RESULT_SCHEMA
    task_type: Literal["entry_quality_history"] = ENTRY_QUALITY_HISTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    entry_quality_schema_version: str = Field(min_length=1, max_length=80)
    start_date: date
    end_date: date
    mode: Literal["full", "recent_7d", "recent_30d", "walk_forward"]
    cost_mode: Literal["conservative", "quant_lab"]
    window_hours: int = Field(ge=1, le=24 * 30)
    generation_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    completed_at: datetime
    outputs: list[ResearchOutputDataset]
    reports: list[ResearchOutputFile]
    anti_leakage_status: Literal["PASS"]
    warnings: list[str] = Field(default_factory=list)
    input_bytes: int = Field(ge=0)
    cache_hit_bytes: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
    compute_duration_seconds: float = Field(ge=0)
    worker_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    requires_cloud_validation: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> ResearchResultManifest:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.generation_id, "generation_id")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_commit(self.worker_commit, "worker_commit")
        _require_utc(self.generated_at, "generated_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        names = [item.dataset_name for item in self.outputs]
        if len(names) != len(set(names)):
            raise ValueError("result output datasets must be unique")
        report_paths = [item.relative_path for item in self.reports]
        if len(report_paths) != len(set(report_paths)):
            raise ValueError("result report paths must be unique")
        expected_output_bytes = sum(item.size_bytes for item in self.outputs) + sum(
            item.size_bytes for item in self.reports
        )
        if self.output_bytes != expected_output_bytes:
            raise ValueError("result output_bytes mismatch")
        return self


class ResearchWorkerReceipt(StrictModel):
    schema_version: Literal["quant_lab_research_receipt.v1"] = RESEARCH_RECEIPT_SCHEMA
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
    anti_leakage_status: str
    error_code: str | None = None
    worker_key_id: str = Field(min_length=1, max_length=120)
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> ResearchWorkerReceipt:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_commit(self.worker_commit, "worker_commit")
        _require_sha(self.result_manifest_sha256, "result_manifest_sha256")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        return self


class ResearchValidationEvent(StrictModel):
    schema_version: Literal["quant_lab_research_validation.v1"] = RESEARCH_VALIDATION_SCHEMA
    task_id: str
    stage: Literal["nas", "cloud"]
    check_name: str
    status: Literal["PASS", "FAIL"]
    detail: str
    observed_at: datetime

    @field_validator("observed_at")
    @classmethod
    def validate_observed_at(cls, value: datetime) -> datetime:
        _require_utc(value, "observed_at")
        return value


class ResearchTaskStatus(StrictModel):
    schema_version: Literal["quant_lab_research_status.v1"] = RESEARCH_STATUS_SCHEMA
    task_id: str
    snapshot_id: str
    task_type: Literal["entry_quality_history"] = ENTRY_QUALITY_HISTORY_TASK_TYPE
    start_date: date
    end_date: date
    mode: str
    cost_mode: str
    state: ResearchTaskState
    worker_id: str | None = None
    requested_at: datetime
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    lease_expires_at: datetime | None = None
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    input_bytes: int = Field(default=0, ge=0)
    downloaded_bytes: int = Field(default=0, ge=0)
    cache_hit_bytes: int = Field(default=0, ge=0)
    output_rows: int = Field(default=0, ge=0)
    anti_leakage_status: str | None = None
    import_status: str | None = None
    last_error: str | None = None
    gold_generation_id: str | None = None
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"

    @model_validator(mode="after")
    def validate_status(self) -> ResearchTaskStatus:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        for field_name in (
            "requested_at",
            "claimed_at",
            "heartbeat_at",
            "completed_at",
            "lease_expires_at",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _require_utc(value, field_name)
        return self


def _require_identifier(value: str, field_name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or len(value) > 180 or any(character not in allowed for character in value):
        raise ValueError(f"unsafe {field_name}")


def _require_sha(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be lowercase sha256")


def _require_commit(value: str, field_name: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a full lowercase git commit")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")


def _require_safe_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("unsafe relative path")
    if "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("unsafe relative path")
