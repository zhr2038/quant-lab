from __future__ import annotations

import re
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EXPORT_SNAPSHOT_SCHEMA = "quant_lab_export_snapshot.v1"
EXPORT_TASK_SCHEMA = "quant_lab_export_task.v1"
EXPORT_RECEIPT_SCHEMA = "quant_lab_export_receipt.v1"
EXPORT_PACK_MANIFEST_SCHEMA = "quant_lab_export_pack_manifest.v1"
EXPORT_VALIDATION_SCHEMA = "quant_lab_export_validation.v1"
EXPORT_INDEX_SCHEMA = "quant_lab_export_index.v1"
SIGNATURE_ALGORITHM = "ed25519"

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,180}$")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExportTaskState(StrEnum):
    SNAPSHOT_PREPARING = "snapshot_preparing"
    PENDING = "pending"
    CLAIMED = "claimed"
    SYNCING = "syncing"
    MATERIALIZING = "materializing"
    VALIDATING_ON_NAS = "validating_on_nas"
    ACCEPTED_ON_NAS = "accepted_on_nas"
    RECEIPT_UPLOADING = "receipt_uploading"
    RECEIPT_RECEIVED = "receipt_received"
    RECEIPT_VERIFIED = "receipt_verified"
    DOWNLOAD_READY = "download_ready"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ExportRequest(StrictModel):
    schema_version: Literal["quant_lab_export_request.v1"] = "quant_lab_export_request.v1"
    request_id: str = Field(min_length=1, max_length=180)
    export_date: date
    export_mode: Literal["cached", "authoritative"]
    requested_at: datetime
    requested_by: str = Field(default="web", min_length=1, max_length=120)

    @model_validator(mode="after")
    def validate_request(self) -> ExportRequest:
        _require_id(self.request_id, "request_id")
        _require_utc(self.requested_at, "requested_at")
        return self


class ExportDatasetReference(StrictModel):
    relative_path: str = Field(min_length=1, max_length=700)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0, le=2**44)
    mtime_ns: int = Field(ge=0)
    row_count: int | None = Field(default=None, ge=0)
    dataset: str = Field(min_length=1, max_length=180)
    media_type: Literal["parquet", "json", "csv", "yaml", "bundle", "other"]

    @field_validator("relative_path")
    @classmethod
    def relative_path_is_safe(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts or normalized.startswith("/"):
            raise ValueError("relative_path must be a safe relative POSIX path")
        if not path.parts or any(part in {"", "."} for part in path.parts):
            raise ValueError("relative_path contains an empty path component")
        return path.as_posix()

    @field_validator("sha256")
    @classmethod
    def sha_is_valid(cls, value: str) -> str:
        normalized = value.lower()
        if not _SHA_RE.fullmatch(normalized):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return normalized


class ExportSnapshotManifest(StrictModel):
    schema_version: Literal[EXPORT_SNAPSHOT_SCHEMA] = EXPORT_SNAPSHOT_SCHEMA
    snapshot_id: str = Field(min_length=1, max_length=180)
    export_date: date
    created_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    quant_lab_current_main_commit: str | None = Field(
        default=None,
        min_length=40,
        max_length=40,
    )
    current_main_production_relationship: Literal[
        "MATCH",
        "MISMATCH",
        "UNOBSERVABLE",
    ] = "UNOBSERVABLE"
    quant_lab_version: str = Field(min_length=1, max_length=80)
    v5_commit: str = Field(min_length=40, max_length=40)
    selected_v5_bundle_name: str = Field(min_length=1, max_length=300)
    selected_v5_bundle_sha256: str = Field(min_length=64, max_length=64)
    acceptance_set_id: str = Field(min_length=1, max_length=180)
    risk_permission_identity: str = Field(min_length=1, max_length=300)
    paper_lifecycle_identity: str = Field(min_length=1, max_length=300)
    proposal_snapshot_id: str | None = Field(default=None, min_length=1, max_length=180)
    proposal_snapshot_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    proposal_content_snapshot_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=180,
    )
    proposal_content_snapshot_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    snapshot_generated_at: datetime | None = None
    v5_observed_proposal_snapshot_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=180,
    )
    v5_observed_proposal_snapshot_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    v5_observed_proposal_content_snapshot_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=180,
    )
    v5_observed_proposal_content_snapshot_sha256: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
    )
    selected_v5_bundle_built_at: datetime | None = None
    environment_fingerprint: str = Field(min_length=64, max_length=64)
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    files: list[ExportDatasetReference] = Field(min_length=1, max_length=20_000)
    total_input_bytes: int = Field(ge=0, le=2**48)
    authoritative_input_snapshot: Literal[True] = True
    manifest_sha256: str = Field(min_length=64, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    signature_algorithm: Literal[SIGNATURE_ALGORITHM] = SIGNATURE_ALGORITHM
    signature: str = Field(min_length=80, max_length=200)

    @model_validator(mode="after")
    def validate_identity(self) -> ExportSnapshotManifest:
        _require_id(self.snapshot_id, "snapshot_id")
        _require_id(self.acceptance_set_id, "acceptance_set_id")
        _require_utc(self.created_at, "created_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        if self.quant_lab_current_main_commit is not None:
            _require_commit(
                self.quant_lab_current_main_commit,
                "quant_lab_current_main_commit",
            )
        _require_commit(self.v5_commit, "v5_commit")
        _require_sha(self.selected_v5_bundle_sha256, "selected_v5_bundle_sha256")
        _require_sha(self.environment_fingerprint, "environment_fingerprint")
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_sha(self.manifest_sha256, "manifest_sha256")
        for value, name in (
            (self.proposal_snapshot_sha256, "proposal_snapshot_sha256"),
            (
                self.proposal_content_snapshot_sha256,
                "proposal_content_snapshot_sha256",
            ),
            (
                self.v5_observed_proposal_snapshot_sha256,
                "v5_observed_proposal_snapshot_sha256",
            ),
            (
                self.v5_observed_proposal_content_snapshot_sha256,
                "v5_observed_proposal_content_snapshot_sha256",
            ),
        ):
            if value is not None:
                _require_sha(value, name)
        if self.snapshot_generated_at is not None:
            _require_utc(self.snapshot_generated_at, "snapshot_generated_at")
        if self.selected_v5_bundle_built_at is not None:
            _require_utc(
                self.selected_v5_bundle_built_at,
                "selected_v5_bundle_built_at",
            )
        acceptance_values = (
            self.quant_lab_current_main_commit,
            self.proposal_content_snapshot_id,
            self.proposal_content_snapshot_sha256,
            self.snapshot_generated_at,
            self.v5_observed_proposal_content_snapshot_id,
            self.v5_observed_proposal_content_snapshot_sha256,
            self.selected_v5_bundle_built_at,
        )
        if any(value is not None for value in acceptance_values) and not all(
            value is not None for value in acceptance_values
        ):
            raise ValueError("snapshot acceptance context must be complete")
        if self.quant_lab_current_main_commit is not None:
            expected_relationship = (
                "MATCH"
                if self.quant_lab_current_main_commit == self.quant_lab_commit
                else "MISMATCH"
            )
            if self.current_main_production_relationship != expected_relationship:
                raise ValueError("current main/production relationship is inconsistent")
        elif self.current_main_production_relationship != "UNOBSERVABLE":
            raise ValueError("current main relationship requires a current main commit")
        if (
            self.proposal_content_snapshot_id is not None
            and (
                self.proposal_content_snapshot_id
                != self.v5_observed_proposal_content_snapshot_id
                or self.proposal_content_snapshot_sha256
                != self.v5_observed_proposal_content_snapshot_sha256
            )
        ):
            raise ValueError("proposal content snapshot identity is inconsistent")
        if (
            self.snapshot_generated_at is not None
            and self.selected_v5_bundle_built_at is not None
            and self.selected_v5_bundle_built_at <= self.snapshot_generated_at
        ):
            raise ValueError("selected V5 bundle must postdate proposal snapshot")
        paths = [item.relative_path for item in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("snapshot contains duplicate relative paths")
        if sum(item.size_bytes for item in self.files) != self.total_input_bytes:
            raise ValueError("total_input_bytes does not match files")
        return self


class ExportTask(StrictModel):
    schema_version: Literal[EXPORT_TASK_SCHEMA] = EXPORT_TASK_SCHEMA
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    export_date: date
    export_mode: Literal["cached", "authoritative"]
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    quant_lab_version: str = Field(min_length=1, max_length=80)
    expected_worker_commit: str = Field(min_length=40, max_length=40)
    report_schema_version: str = Field(min_length=1, max_length=120)
    selected_v5_bundle_sha256: str = Field(min_length=64, max_length=64)
    acceptance_set_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    requested_at: datetime
    lease_seconds: int = Field(default=7200, ge=300, le=86_400)
    max_attempts: int = Field(default=3, ge=1, le=10)
    idempotency_key: str = Field(min_length=64, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    signature_algorithm: Literal[SIGNATURE_ALGORITHM] = SIGNATURE_ALGORITHM
    signature: str = Field(min_length=80, max_length=200)

    @model_validator(mode="after")
    def validate_identity(self) -> ExportTask:
        _require_id(self.task_id, "task_id")
        _require_id(self.snapshot_id, "snapshot_id")
        _require_id(self.acceptance_set_id, "acceptance_set_id")
        _require_utc(self.requested_at, "requested_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_commit(self.expected_worker_commit, "expected_worker_commit")
        _require_sha(self.selected_v5_bundle_sha256, "selected_v5_bundle_sha256")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_sha(self.idempotency_key, "idempotency_key")
        return self


class ExportTaskStatus(StrictModel):
    schema_version: Literal["quant_lab_export_status.v1"] = "quant_lab_export_status.v1"
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    state: ExportTaskState
    requested_at: datetime
    updated_at: datetime
    claimed_at: datetime | None = None
    heartbeat_at: datetime | None = None
    lease_expires_at: datetime | None = None
    worker_id: str | None = Field(default=None, max_length=180)
    attempt: int = Field(default=0, ge=0, le=10)
    max_attempts: int = Field(default=3, ge=1, le=10)
    current_stage: str = Field(default="pending", min_length=1, max_length=120)
    completed_members: int = Field(default=0, ge=0, le=20_000)
    total_members: int = Field(default=0, ge=0, le=20_000)
    input_bytes: int = Field(default=0, ge=0, le=2**48)
    output_bytes: int = Field(default=0, ge=0, le=2**44)
    last_error: str | None = Field(default=None, max_length=4000)
    nas_pack_id: str | None = Field(default=None, max_length=180)
    nas_pack_sha256: str | None = Field(default=None, max_length=64)
    nas_download_path: str | None = Field(default=None, max_length=700)

    @model_validator(mode="after")
    def validate_status(self) -> ExportTaskStatus:
        _require_id(self.task_id, "task_id")
        _require_id(self.snapshot_id, "snapshot_id")
        _require_utc(self.requested_at, "requested_at")
        _require_utc(self.updated_at, "updated_at")
        return self


class ExportPackFile(StrictModel):
    path: str = Field(min_length=1, max_length=700)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0, le=2**44)
    row_count: int | None = Field(default=None, ge=0)
    cache_hit: bool = False

    @field_validator("path")
    @classmethod
    def path_is_safe(cls, value: str) -> str:
        return ExportDatasetReference.relative_path_is_safe(value)

    @field_validator("sha256")
    @classmethod
    def sha_is_valid(cls, value: str) -> str:
        return ExportDatasetReference.sha_is_valid(value)


class ExportPackManifest(StrictModel):
    schema_version: Literal[EXPORT_PACK_MANIFEST_SCHEMA] = EXPORT_PACK_MANIFEST_SCHEMA
    pack_id: str = Field(min_length=1, max_length=180)
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    export_date: date
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    selected_v5_bundle_sha256: str = Field(min_length=64, max_length=64)
    acceptance_set_id: str = Field(min_length=1, max_length=180)
    authoritative_input_snapshot: Literal[True] = True
    files: list[ExportPackFile] = Field(min_length=1, max_length=20_000)


class ExportValidationReport(StrictModel):
    schema_version: Literal[EXPORT_VALIDATION_SCHEMA] = EXPORT_VALIDATION_SCHEMA
    pack_id: str = Field(min_length=1, max_length=180)
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    validated_at: datetime
    valid: bool
    checks: dict[str, bool] = Field(max_length=128)
    failures: list[str] = Field(default_factory=list, max_length=256)
    warnings: list[str] = Field(default_factory=list, max_length=256)
    zip_sha256: str = Field(min_length=64, max_length=64)
    zip_size_bytes: int = Field(ge=0, le=2**44)
    member_count: int = Field(ge=0, le=20_000)
    total_uncompressed_bytes: int = Field(ge=0, le=2**48)
    peak_compression_ratio: float = Field(ge=0, le=100_000)


class ExportWorkerReceipt(StrictModel):
    schema_version: Literal[EXPORT_RECEIPT_SCHEMA] = EXPORT_RECEIPT_SCHEMA
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    worker_id: str = Field(min_length=1, max_length=180)
    worker_commit: str = Field(min_length=40, max_length=40)
    pack_id: str = Field(min_length=1, max_length=180)
    pack_name: str = Field(min_length=1, max_length=500)
    pack_sha256: str = Field(min_length=64, max_length=64)
    pack_size_bytes: int = Field(ge=1, le=2**44)
    pack_manifest_sha256: str = Field(min_length=64, max_length=64)
    pack_state: Literal["accepted"] = "accepted"
    nas_artifact_validated: Literal[True] = True
    validation_report_sha256: str = Field(min_length=64, max_length=64)
    authoritative_input_snapshot: Literal[True] = True
    selected_v5_bundle_sha256: str = Field(min_length=64, max_length=64)
    acceptance_set_id: str = Field(min_length=1, max_length=180)
    download_relative_path: str = Field(min_length=1, max_length=700)
    generated_at: datetime
    accepted_at: datetime
    manifest_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    data_quality_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    expert_questions: list[str] = Field(default_factory=list, max_length=20)
    validation_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    worker_report_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    signature_algorithm: Literal[SIGNATURE_ALGORITHM] = SIGNATURE_ALGORITHM
    signature: str = Field(min_length=80, max_length=200)

    @model_validator(mode="after")
    def validate_identity(self) -> ExportWorkerReceipt:
        for value, field in (
            (self.task_id, "task_id"),
            (self.snapshot_id, "snapshot_id"),
            (self.pack_id, "pack_id"),
            (self.acceptance_set_id, "acceptance_set_id"),
        ):
            _require_id(value, field)
        _require_utc(self.generated_at, "generated_at")
        _require_utc(self.accepted_at, "accepted_at")
        _require_commit(self.worker_commit, "worker_commit")
        for value, field in (
            (self.pack_sha256, "pack_sha256"),
            (self.pack_manifest_sha256, "pack_manifest_sha256"),
            (self.validation_report_sha256, "validation_report_sha256"),
            (self.selected_v5_bundle_sha256, "selected_v5_bundle_sha256"),
        ):
            _require_sha(value, field)
        ExportDatasetReference.relative_path_is_safe(self.download_relative_path)
        return self


class ExportPackIndexEntry(StrictModel):
    schema_version: Literal[EXPORT_INDEX_SCHEMA] = EXPORT_INDEX_SCHEMA
    pack_id: str
    task_id: str | None = Field(default=None, max_length=180)
    pack_name: str
    export_date: date
    generated_at: datetime
    accepted_at: datetime
    pack_sha256: str
    pack_size_bytes: int = Field(ge=1)
    snapshot_id: str
    authoritative_input_snapshot: bool
    nas_artifact_validated: bool
    control_plane_receipt_verified: bool
    download_ready: bool
    download_relative_path: str
    selected_v5_bundle_sha256: str
    acceptance_set_id: str
    worker_id: str
    worker_commit: str
    ai_consumed: bool = False
    pack_state: Literal["accepted"] = "accepted"
    manifest_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    data_quality_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    expert_questions: list[str] = Field(default_factory=list, max_length=20)
    validation_summary: dict[str, object] = Field(default_factory=dict, max_length=64)
    worker_report_summary: dict[str, object] = Field(default_factory=dict, max_length=64)


def _require_id(value: str, field_name: str) -> None:
    if not _ID_RE.fullmatch(value):
        raise ValueError(f"{field_name} contains invalid characters")


def _require_sha(value: str, field_name: str) -> None:
    if not _SHA_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA256")


def _require_commit(value: str, field_name: str) -> None:
    if not _COMMIT_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a full lowercase git SHA")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")
