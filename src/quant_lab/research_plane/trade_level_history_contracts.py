from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TRADE_LEVEL_HISTORY_TASK_TYPE = "trade_level_history"
TRADE_LEVEL_HISTORY_TASK_SCHEMA = "quant_lab_trade_level_history_task.v1"
TRADE_LEVEL_HISTORY_SNAPSHOT_SCHEMA = "quant_lab_trade_level_history_snapshot.v1"
TRADE_LEVEL_HISTORY_RESULT_SCHEMA = "quant_lab_trade_level_history_result.v1"
TRADE_LEVEL_HISTORY_RECEIPT_SCHEMA = "quant_lab_trade_level_history_receipt.v1"
TRADE_LEVEL_HISTORY_FINGERPRINT_SCHEMA = "trade_level_history_input_fingerprint.v1"
TRADE_LEVEL_HISTORY_GENERATION_SCHEMA = "trade_level_history_generation.v1"

TRADE_LEVEL_HISTORY_MODE = "PARITY_FULL"
TRADE_LEVEL_HISTORY_EVENT_SCHEMA = "trade_opportunity_event.v0.3"
TRADE_LEVEL_HISTORY_LABEL_SCHEMA = "trade_opportunity_label.v0.2"
TRADE_LEVEL_HISTORY_SIMILARITY_SCHEMA = "trade_level_similarity_outcome.v0.2"
TRADE_LEVEL_HISTORY_AVAILABILITY_POLICY = "largest_available_horizon"
TRADE_LEVEL_HISTORY_RISK_JOIN_VERSION = "point_in_time.v1"

TRADE_LEVEL_HISTORY_INPUT_DATASETS = (
    "cloud/trade_opportunity_event",
    "gold/v5_candidate_label",
)
TRADE_LEVEL_HISTORY_OUTPUT_DATASETS = (
    "trade_opportunity_label",
    "trade_level_similarity_outcome",
)
TRADE_LEVEL_HISTORY_PRIMARY_KEYS = {
    "trade_opportunity_label": ("event_id",),
    "trade_level_similarity_outcome": ("event_id",),
}
TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS = (
    "event_primary_keys_are_unique",
    "event_ids_match_cloud_derivation",
    "event_decision_timestamps_are_valid",
    "event_symbols_are_normalized",
    "candidate_labels_bind_verified_generation",
    "trade_labels_match_events_one_to_one",
    "trade_label_candidate_ids_match_source",
    "label_horizons_are_4_8_24_only",
    "label_available_at_not_before_source_label_ts",
    "label_available_at_not_before_decision_ts",
    "similarity_excludes_current_event",
    "similarity_uses_strictly_prior_events",
    "same_timestamp_events_do_not_cross_reference",
    "similarity_outcomes_are_available_at_decision",
    "similarity_falls_back_24h_to_8h",
    "similarity_falls_back_8h_to_4h",
    "similarity_excludes_events_without_available_outcome",
    "recent_7d_excludes_current_and_future_events",
    "similar_sample_count_matches_recomputation",
    "similar_mean_matches_recomputation",
    "similar_median_matches_recomputation",
    "similar_p25_matches_recomputation",
    "similar_hit_rate_matches_recomputation",
    "similar_max_adverse_matches_recomputation",
    "risk_permission_excludes_post_decision_records",
    "nas_outputs_no_trade_level_judgment",
    "nas_outputs_no_bucket_policy",
    "nas_outputs_no_opportunity_queue",
    "nas_outputs_no_order_limits",
    "automatic_promotion_is_disabled",
    "live_notional_is_zero",
    "live_order_effect_is_none_read_only_research",
)

DEFAULT_TRADE_LEVEL_HISTORY_MAX_SNAPSHOT_BYTES = 2 * 1024**3
DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_UNCOMPRESSED_BYTES = 4 * 1024**3
DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_BYTES = 1024**3
DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_UNCOMPRESSED_BYTES = 2 * 1024**3
DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_BYTES = 128 * 1024**2
DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_UNCOMPRESSED_BYTES = 256 * 1024**2
DEFAULT_TRADE_LEVEL_HISTORY_MAX_FILE_COUNT = 20_000
DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_ROWS = 10_000_000


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TradeLevelHistoryTaskPayload(_StrictModel):
    as_of_date: date
    history_mode: Literal["PARITY_FULL"] = TRADE_LEVEL_HISTORY_MODE
    candidate_evidence_generation_id: str = Field(min_length=1, max_length=180)
    candidate_evidence_generation_digest: str = Field(min_length=64, max_length=64)
    candidate_evidence_input_fingerprint: str = Field(min_length=64, max_length=64)
    trade_event_schema_version: Literal["trade_opportunity_event.v0.3"] = (
        TRADE_LEVEL_HISTORY_EVENT_SCHEMA
    )
    trade_label_schema_version: Literal["trade_opportunity_label.v0.2"] = (
        TRADE_LEVEL_HISTORY_LABEL_SCHEMA
    )
    similarity_schema_version: Literal["trade_level_similarity_outcome.v0.2"] = (
        TRADE_LEVEL_HISTORY_SIMILARITY_SCHEMA
    )
    similarity_availability_policy: Literal["largest_available_horizon"] = (
        TRADE_LEVEL_HISTORY_AVAILABILITY_POLICY
    )

    @model_validator(mode="after")
    def validate_payload(self) -> TradeLevelHistoryTaskPayload:
        _require_identifier(
            self.candidate_evidence_generation_id,
            "candidate_evidence_generation_id",
        )
        _require_sha(
            self.candidate_evidence_generation_digest,
            "candidate_evidence_generation_digest",
        )
        _require_sha(
            self.candidate_evidence_input_fingerprint,
            "candidate_evidence_input_fingerprint",
        )
        return self


class TradeLevelHistoryInputFingerprint(TradeLevelHistoryTaskPayload):
    schema_version: Literal["trade_level_history_input_fingerprint.v1"] = (
        TRADE_LEVEL_HISTORY_FINGERPRINT_SCHEMA
    )
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    derived_event_digest: str = Field(min_length=64, max_length=64)
    candidate_label_dataset_hash: str = Field(min_length=64, max_length=64)
    event_row_count: int = Field(ge=0)
    candidate_label_row_count: int = Field(ge=0)
    event_min_ts: datetime | None = None
    event_max_ts: datetime | None = None
    candidate_label_min_ts: datetime | None = None
    candidate_label_max_ts: datetime | None = None
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    observed_at: datetime

    @model_validator(mode="after")
    def validate_fingerprint(self) -> TradeLevelHistoryInputFingerprint:
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.derived_event_digest, "derived_event_digest")
        _require_sha(self.candidate_label_dataset_hash, "candidate_label_dataset_hash")
        _require_sha(self.input_fingerprint_digest, "input_fingerprint_digest")
        _require_optional_bounds(self.event_min_ts, self.event_max_ts, "event")
        _require_optional_bounds(
            self.candidate_label_min_ts,
            self.candidate_label_max_ts,
            "candidate label",
        )
        _require_utc(self.observed_at, "observed_at")
        return self


class TradeLevelHistorySnapshotFile(_StrictModel):
    dataset_name: Literal[
        "cloud/trade_opportunity_event",
        "gold/v5_candidate_label",
    ]
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    row_count: int = Field(ge=0)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    uncompressed_bytes: int = Field(ge=0)
    media_type: Literal["application/x-parquet"] = "application/x-parquet"

    @model_validator(mode="after")
    def validate_file(self) -> TradeLevelHistorySnapshotFile:
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_optional_bounds(self.min_ts, self.max_ts, "snapshot file")
        return self


class TradeLevelHistorySnapshotManifest(TradeLevelHistoryTaskPayload):
    schema_version: Literal["quant_lab_trade_level_history_snapshot.v1"] = (
        TRADE_LEVEL_HISTORY_SNAPSHOT_SCHEMA
    )
    task_type: Literal["trade_level_history"] = TRADE_LEVEL_HISTORY_TASK_TYPE
    snapshot_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    candidate_event_digest: str = Field(min_length=64, max_length=64)
    risk_permission_digest: str = Field(min_length=64, max_length=64)
    v5_trade_event_digest: str = Field(min_length=64, max_length=64)
    order_lifecycle_digest: str = Field(min_length=64, max_length=64)
    derived_trade_opportunity_event_digest: str = Field(min_length=64, max_length=64)
    risk_permission_join_version: Literal["point_in_time.v1"] = (
        TRADE_LEVEL_HISTORY_RISK_JOIN_VERSION
    )
    candidate_label_dataset_hash: str = Field(min_length=64, max_length=64)
    candidate_label_row_count: int = Field(ge=0)
    candidate_label_schema: str = Field(min_length=1, max_length=180)
    event_row_count: int = Field(ge=0)
    event_min_ts: datetime | None = None
    event_max_ts: datetime | None = None
    candidate_label_min_ts: datetime | None = None
    candidate_label_max_ts: datetime | None = None
    datasets: tuple[str, ...]
    files: tuple[TradeLevelHistorySnapshotFile, ...]
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
    def validate_manifest(self) -> TradeLevelHistorySnapshotManifest:
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_utc(self.generated_at, "generated_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        for field_name in (
            "input_fingerprint_digest",
            "candidate_event_digest",
            "risk_permission_digest",
            "v5_trade_event_digest",
            "order_lifecycle_digest",
            "derived_trade_opportunity_event_digest",
            "candidate_label_dataset_hash",
            "manifest_sha256",
        ):
            _require_sha(getattr(self, field_name), field_name)
        _require_optional_bounds(self.event_min_ts, self.event_max_ts, "event")
        _require_optional_bounds(
            self.candidate_label_min_ts,
            self.candidate_label_max_ts,
            "candidate label",
        )
        if self.datasets != TRADE_LEVEL_HISTORY_INPUT_DATASETS:
            raise ValueError("trade level history snapshot dataset set mismatch")
        file_names = tuple(item.dataset_name for item in self.files)
        if tuple(sorted(file_names)) != tuple(sorted(TRADE_LEVEL_HISTORY_INPUT_DATASETS)):
            raise ValueError("trade level history snapshot file set mismatch")
        if len(file_names) != len(set(file_names)):
            raise ValueError("trade level history snapshot dataset files must be unique")
        if self.total_input_bytes != sum(item.size_bytes for item in self.files):
            raise ValueError("trade level history snapshot byte count mismatch")
        if self.total_input_rows != sum(item.row_count for item in self.files):
            raise ValueError("trade level history snapshot row count mismatch")
        if self.estimated_uncompressed_bytes != sum(
            item.uncompressed_bytes for item in self.files
        ):
            raise ValueError("trade level history snapshot uncompressed count mismatch")
        if self.event_row_count != next(
            item.row_count
            for item in self.files
            if item.dataset_name == "cloud/trade_opportunity_event"
        ):
            raise ValueError("trade level history event row count mismatch")
        if self.candidate_label_row_count != next(
            item.row_count
            for item in self.files
            if item.dataset_name == "gold/v5_candidate_label"
        ):
            raise ValueError("trade level history candidate label row count mismatch")
        _require_identifier(self.signature_key_id, "signature_key_id")
        return self


class TradeLevelHistoryTask(TradeLevelHistoryTaskPayload):
    schema_version: Literal["quant_lab_trade_level_history_task.v1"] = (
        TRADE_LEVEL_HISTORY_TASK_SCHEMA
    )
    task_type: Literal["trade_level_history"] = TRADE_LEVEL_HISTORY_TASK_TYPE
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
    def validate_task(self) -> TradeLevelHistoryTask:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_sha(self.input_fingerprint_digest, "input_fingerprint_digest")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_previous_generation(
            self.previous_generation_id,
            self.previous_generation_digest,
        )
        _require_utc(self.requested_at, "requested_at")
        _require_identifier(self.signature_key_id, "signature_key_id")
        return self

    @property
    def parameters(self) -> TradeLevelHistoryTaskPayload:
        return TradeLevelHistoryTaskPayload(
            as_of_date=self.as_of_date,
            history_mode=self.history_mode,
            candidate_evidence_generation_id=self.candidate_evidence_generation_id,
            candidate_evidence_generation_digest=self.candidate_evidence_generation_digest,
            candidate_evidence_input_fingerprint=self.candidate_evidence_input_fingerprint,
            trade_event_schema_version=self.trade_event_schema_version,
            trade_label_schema_version=self.trade_label_schema_version,
            similarity_schema_version=self.similarity_schema_version,
            similarity_availability_policy=self.similarity_availability_policy,
        )


class TradeLevelHistoryOutputDataset(_StrictModel):
    dataset_name: Literal[
        "trade_opportunity_label",
        "trade_level_similarity_outcome",
    ]
    partition_symbol: str = Field(min_length=1, max_length=180)
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
    primary_keys: tuple[str, ...] = ("event_id",)
    publish_mode: Literal["replace_full"] = "replace_full"
    empty_result_semantics: Literal["replace_empty"] = "replace_empty"

    @model_validator(mode="after")
    def validate_output(self) -> TradeLevelHistoryOutputDataset:
        _require_identifier(self.partition_symbol, "partition_symbol")
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_sha(self.partition_identity, "partition_identity")
        _require_optional_bounds(self.min_ts, self.max_ts, "result partition")
        if self.primary_keys != TRADE_LEVEL_HISTORY_PRIMARY_KEYS[self.dataset_name]:
            raise ValueError(f"{self.dataset_name} primary keys mismatch")
        prefix = f"outputs/{self.dataset_name}/symbol="
        if not self.relative_path.startswith(prefix) or not self.relative_path.endswith(
            ".parquet"
        ):
            raise ValueError(f"{self.dataset_name} partition path mismatch")
        if self.row_count == 0 and (self.min_ts is not None or self.max_ts is not None):
            raise ValueError(f"{self.dataset_name} empty partition bounds must be null")
        return self


class TradeLevelHistoryReportFile(_StrictModel):
    relative_path: Literal[
        "reports/trade_level_history_worker_report.json",
        "reports/trade_level_history_anti_leakage.json",
    ]
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=1)
    media_type: Literal["application/json"] = "application/json"

    @model_validator(mode="after")
    def validate_report(self) -> TradeLevelHistoryReportFile:
        _require_relative_path(self.relative_path)
        _require_sha(self.sha256, "sha256")
        return self


class TradeLevelHistoryAntiLeakageCheck(_StrictModel):
    check_name: str = Field(min_length=1, max_length=180)
    status: Literal["PASS"] = "PASS"
    violation_count: Literal[0] = 0
    detail: str = Field(min_length=1, max_length=2000)

    @field_validator("check_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        _require_identifier(value, "check_name")
        return value


class TradeLevelHistoryResultManifest(TradeLevelHistoryTaskPayload):
    schema_version: Literal["quant_lab_trade_level_history_result.v1"] = (
        TRADE_LEVEL_HISTORY_RESULT_SCHEMA
    )
    task_type: Literal["trade_level_history"] = TRADE_LEVEL_HISTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    input_fingerprint_digest: str = Field(min_length=64, max_length=64)
    derived_event_digest: str = Field(min_length=64, max_length=64)
    candidate_label_dataset_hash: str = Field(min_length=64, max_length=64)
    previous_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    previous_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    generation_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    completed_at: datetime
    outputs: tuple[TradeLevelHistoryOutputDataset, ...] = Field(min_length=2)
    reports: tuple[TradeLevelHistoryReportFile, ...] = Field(min_length=2, max_length=2)
    anti_leakage_checks: tuple[TradeLevelHistoryAntiLeakageCheck, ...] = Field(min_length=32)
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
    def validate_result(self) -> TradeLevelHistoryResultManifest:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.generation_id, "generation_id")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_commit(self.worker_commit, "worker_commit")
        for field_name in (
            "snapshot_manifest_sha256",
            "input_fingerprint_digest",
            "derived_event_digest",
            "candidate_label_dataset_hash",
        ):
            _require_sha(getattr(self, field_name), field_name)
        _require_previous_generation(
            self.previous_generation_id,
            self.previous_generation_digest,
        )
        _require_utc(self.generated_at, "generated_at")
        _require_utc(self.completed_at, "completed_at")
        if self.completed_at < self.generated_at:
            raise ValueError("trade level history result time order invalid")
        if self.warnings:
            raise ValueError("trade level history result warnings are not publishable")
        output_names = {item.dataset_name for item in self.outputs}
        if output_names != set(TRADE_LEVEL_HISTORY_OUTPUT_DATASETS):
            raise ValueError("trade level history result output set mismatch")
        output_paths = [item.relative_path for item in self.outputs]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("trade level history result output paths must be unique")
        report_paths = {item.relative_path for item in self.reports}
        if report_paths != {
            "reports/trade_level_history_worker_report.json",
            "reports/trade_level_history_anti_leakage.json",
        }:
            raise ValueError("trade level history result report set mismatch")
        check_names = [item.check_name for item in self.anti_leakage_checks]
        if tuple(check_names) != TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS:
            raise ValueError("trade level history anti leakage check set mismatch")
        expected_bytes = sum(item.size_bytes for item in self.outputs) + sum(
            item.size_bytes for item in self.reports
        )
        if self.output_bytes != expected_bytes:
            raise ValueError("trade level history output byte count mismatch")
        expected_uncompressed = sum(item.uncompressed_bytes for item in self.outputs) + sum(
            item.size_bytes for item in self.reports
        )
        if self.output_uncompressed_bytes != expected_uncompressed:
            raise ValueError("trade level history output uncompressed byte count mismatch")
        return self


class TradeLevelHistoryWorkerReceipt(_StrictModel):
    schema_version: Literal["quant_lab_trade_level_history_receipt.v1"] = (
        TRADE_LEVEL_HISTORY_RECEIPT_SCHEMA
    )
    task_type: Literal["trade_level_history"] = TRADE_LEVEL_HISTORY_TASK_TYPE
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
    worker_key_id: str = Field(min_length=1, max_length=120)
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> TradeLevelHistoryWorkerReceipt:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.worker_id, "worker_id")
        _require_commit(self.worker_commit, "worker_commit")
        _require_sha(self.result_manifest_sha256, "result_manifest_sha256")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.completed_at, "completed_at")
        if self.completed_at < self.claimed_at:
            raise ValueError("trade level history receipt time order invalid")
        _require_identifier(self.worker_key_id, "worker_key_id")
        return self


def _require_identifier(value: str, field_name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or any(character not in allowed for character in value):
        raise ValueError(f"{field_name} contains unsafe characters")


def _require_sha(value: str, field_name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a lowercase sha256")


def _require_commit(value: str, field_name: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a full lowercase git sha")


def _require_utc(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")


def _require_optional_bounds(
    minimum: datetime | None,
    maximum: datetime | None,
    label: str,
) -> None:
    if (minimum is None) != (maximum is None):
        raise ValueError(f"{label} bounds must both be set or null")
    if minimum is not None and maximum is not None:
        _require_utc(minimum, f"{label} min")
        _require_utc(maximum, f"{label} max")
        if minimum > maximum:
            raise ValueError(f"{label} bounds are invalid")


def _require_previous_generation(
    generation_id: str | None,
    generation_digest: str | None,
) -> None:
    if (generation_id is None) != (generation_digest is None):
        raise ValueError("previous generation id and digest must both be set or null")
    if generation_id is not None:
        _require_identifier(generation_id, "previous_generation_id")
        _require_sha(generation_digest or "", "previous_generation_digest")


def _require_relative_path(value: str) -> None:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith(("/", "\\")):
        raise ValueError("path must be safe and relative")
