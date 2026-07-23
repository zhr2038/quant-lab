from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from quant_lab.factors.plan import EffectiveFactorPlan
from quant_lab.research_plane.trade_level_history_contracts import (  # noqa: F401
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_FILE_COUNT,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_ROWS,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_INPUT_UNCOMPRESSED_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_PARTITION_UNCOMPRESSED_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_RESULT_UNCOMPRESSED_BYTES,
    DEFAULT_TRADE_LEVEL_HISTORY_MAX_SNAPSHOT_BYTES,
    TRADE_LEVEL_HISTORY_ANTI_LEAKAGE_CHECKS,
    TRADE_LEVEL_HISTORY_AVAILABILITY_POLICY,
    TRADE_LEVEL_HISTORY_EVENT_SCHEMA,
    TRADE_LEVEL_HISTORY_FINGERPRINT_SCHEMA,
    TRADE_LEVEL_HISTORY_GENERATION_SCHEMA,
    TRADE_LEVEL_HISTORY_INPUT_DATASETS,
    TRADE_LEVEL_HISTORY_LABEL_SCHEMA,
    TRADE_LEVEL_HISTORY_MODE,
    TRADE_LEVEL_HISTORY_OUTPUT_DATASETS,
    TRADE_LEVEL_HISTORY_PRIMARY_KEYS,
    TRADE_LEVEL_HISTORY_RECEIPT_SCHEMA,
    TRADE_LEVEL_HISTORY_RESULT_SCHEMA,
    TRADE_LEVEL_HISTORY_RISK_JOIN_VERSION,
    TRADE_LEVEL_HISTORY_SIMILARITY_SCHEMA,
    TRADE_LEVEL_HISTORY_SNAPSHOT_SCHEMA,
    TRADE_LEVEL_HISTORY_TASK_SCHEMA,
    TRADE_LEVEL_HISTORY_TASK_TYPE,
    TradeLevelHistoryAntiLeakageCheck,
    TradeLevelHistoryInputFingerprint,
    TradeLevelHistoryOutputDataset,
    TradeLevelHistoryReportFile,
    TradeLevelHistoryResultManifest,
    TradeLevelHistorySnapshotFile,
    TradeLevelHistorySnapshotManifest,
    TradeLevelHistoryTask,
    TradeLevelHistoryTaskPayload,
    TradeLevelHistoryWorkerReceipt,
)
from quant_lab.research_plane.v5_candidate_evidence_contracts import (  # noqa: F401
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_FILE_COUNT,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_INPUT_UNCOMPRESSED_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_PARTITION_UNCOMPRESSED_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_RESULT_UNCOMPRESSED_BYTES,
    DEFAULT_V5_CANDIDATE_EVIDENCE_MAX_SNAPSHOT_BYTES,
    V5_CANDIDATE_EVIDENCE_FINGERPRINT_SCHEMA,
    V5_CANDIDATE_EVIDENCE_HORIZONS,
    V5_CANDIDATE_EVIDENCE_PROJECTION_VERSION,
    V5_CANDIDATE_EVIDENCE_RECEIPT_SCHEMA,
    V5_CANDIDATE_EVIDENCE_RESULT_SCHEMA,
    V5_CANDIDATE_EVIDENCE_SNAPSHOT_SCHEMA,
    V5_CANDIDATE_EVIDENCE_TASK_SCHEMA,
    V5_CANDIDATE_EVIDENCE_TASK_TYPE,
    V5_CANDIDATE_LABEL_DELTA_PRIMARY_KEYS,
    V5_STRATEGY_EVIDENCE_SAMPLE_DELTA_PRIMARY_KEYS,
    V5CandidateEvidenceAntiLeakageCheck,
    V5CandidateEvidenceInputFingerprint,
    V5CandidateEvidenceOutputDataset,
    V5CandidateEvidenceReportFile,
    V5CandidateEvidenceResultManifest,
    V5CandidateEvidenceSnapshotFile,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceSourceFileIdentity,
    V5CandidateEvidenceTask,
    V5CandidateEvidenceTaskPayload,
    V5CandidateEvidenceWorkerReceipt,
)

RESEARCH_SNAPSHOT_SCHEMA = "quant_lab_research_snapshot.v1"
RESEARCH_TASK_SCHEMA = "quant_lab_research_task.v1"
RESEARCH_RESULT_SCHEMA = "quant_lab_research_result.v1"
RESEARCH_RECEIPT_SCHEMA = "quant_lab_research_receipt.v1"
RESEARCH_STATUS_SCHEMA = "quant_lab_research_status.v1"
RESEARCH_LEASE_SCHEMA = "quant_lab_research_lease.v1"
RESEARCH_VALIDATION_SCHEMA = "quant_lab_research_validation.v1"
ENTRY_QUALITY_HISTORY_TASK_TYPE = "entry_quality_history"
ALPHA_FACTORY_TASK_TYPE = "alpha_factory"
ALPHA_FACTORY_SNAPSHOT_SCHEMA = "quant_lab_alpha_factory_snapshot.v1"
ALPHA_FACTORY_TASK_SCHEMA = "quant_lab_alpha_factory_task.v1"
ALPHA_FACTORY_RESULT_SCHEMA = "quant_lab_alpha_factory_result.v1"
ALPHA_FACTORY_RECEIPT_SCHEMA = "quant_lab_alpha_factory_receipt.v1"
FACTOR_RESEARCH_TASK_TYPE = "factor_research"
FACTOR_RESEARCH_SNAPSHOT_SCHEMA = "quant_lab_factor_research_snapshot.v1"
FACTOR_RESEARCH_TASK_SCHEMA = "quant_lab_factor_research_task.v1"
FACTOR_RESEARCH_RESULT_SCHEMA = "quant_lab_factor_research_result.v1"
FACTOR_RESEARCH_RECEIPT_SCHEMA = "quant_lab_factor_research_receipt.v1"
FACTOR_FACTORY_TASK_TYPE = "factor_factory"
FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1 = "quant_lab_factor_factory_snapshot.v1"
FACTOR_FACTORY_SNAPSHOT_SCHEMA_V2 = "quant_lab_factor_factory_snapshot.v2"
FACTOR_FACTORY_SNAPSHOT_SCHEMA = "quant_lab_factor_factory_snapshot.v3"
FACTOR_FACTORY_TASK_SCHEMA = "quant_lab_factor_factory_task.v1"
FACTOR_FACTORY_RESULT_SCHEMA_V1 = "quant_lab_factor_factory_result.v1"
FACTOR_FACTORY_RESULT_SCHEMA = "quant_lab_factor_factory_result.v2"
FACTOR_FACTORY_RECEIPT_SCHEMA = "quant_lab_factor_factory_receipt.v1"
DEFAULT_RESEARCH_MAX_RESULT_BYTES = 256 * 1024**2
DEFAULT_FACTOR_FACTORY_MAX_SNAPSHOT_BYTES = 2 * 1024**3
DEFAULT_FACTOR_FACTORY_MAX_INPUT_UNCOMPRESSED_BYTES = 4 * 1024**3
DEFAULT_FACTOR_FACTORY_MAX_RESULT_BYTES = 512 * 1024**2
DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_BYTES = 128 * 1024**2
DEFAULT_FACTOR_FACTORY_MAX_VALUE_PARTITION_UNCOMPRESSED_BYTES = 256 * 1024**2
DEFAULT_FACTOR_FACTORY_MAX_FILE_COUNT = 20_000
DEFAULT_FACTOR_FACTORY_MAX_UNCOMPRESSED_BYTES = 1024**3


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ResearchTaskState(StrEnum):
    PENDING = "pending"
    CLAIMED = "claimed"
    SYNCING = "syncing"
    COMPUTING = "computing"
    COMPUTING_VALUES = "computing_values"
    COMPUTING_LABELS = "computing_labels"
    COMPUTING_SIMILARITY = "computing_similarity"
    COMPUTING_SAMPLES = "computing_samples"
    COMPUTING_EVIDENCE = "computing_evidence"
    COMPUTING_CORRELATION = "computing_correlation"
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
    schema_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    uncompressed_bytes: int | None = Field(default=None, ge=0)
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

    @field_validator("schema_fingerprint")
    @classmethod
    def validate_optional_schema_fingerprint(cls, value: str | None) -> str | None:
        if value is not None:
            _require_sha(value, "schema_fingerprint")
        return value


class FactorFactorySourceFileIdentity(StrictModel):
    dataset_name: str = Field(min_length=1, max_length=180)
    relative_path: str = Field(min_length=1, max_length=1024)
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)
    mtime_ns: int = Field(ge=0)
    row_count: int = Field(ge=0)
    min_ts: datetime | None = None
    max_ts: datetime | None = None
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    uncompressed_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        _require_safe_relative_path(value)
        return value

    @field_validator("sha256", "schema_fingerprint")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        _require_sha(value, "source file hash")
        return value

    @field_validator("min_ts", "max_ts")
    @classmethod
    def validate_optional_utc(cls, value: datetime | None) -> datetime | None:
        if value is not None:
            _require_utc(value, "source file timestamp")
        return value


class FactorFactoryInputFingerprint(StrictModel):
    schema_version: Literal["factor_factory_input_fingerprint.v1"] = (
        "factor_factory_input_fingerprint.v1"
    )
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    factor_plan_digest: str = Field(min_length=64, max_length=64)
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    horizon_bars: tuple[int, ...] = Field(min_length=1, max_length=32)
    decision_delay_bars: int = Field(ge=1, le=1000)
    max_factors: int = Field(ge=1, le=200)
    min_samples: int = Field(ge=1)
    top_quantile: float = Field(gt=0.0, le=0.5)
    cost_quantile: Literal["p50", "p75", "p90"]
    feature_input_digest: str = Field(min_length=64, max_length=64)
    market_input_digest: str = Field(min_length=64, max_length=64)
    cost_input_digest: str = Field(min_length=64, max_length=64)
    combined_input_digest: str = Field(min_length=64, max_length=64)
    feature_file_count: int = Field(ge=0)
    market_file_count: int = Field(ge=0)
    cost_file_count: int = Field(ge=0)
    feature_row_count: int = Field(ge=0)
    market_row_count: int = Field(ge=0)
    cost_row_count: int = Field(ge=0)
    feature_min_ts: datetime | None = None
    feature_max_ts: datetime | None = None
    market_min_ts: datetime | None = None
    market_max_ts: datetime | None = None
    observed_at: datetime

    @model_validator(mode="after")
    def validate_fingerprint(self) -> FactorFactoryInputFingerprint:
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        for value, name in (
            (self.factor_plan_digest, "factor_plan_digest"),
            (self.feature_input_digest, "feature_input_digest"),
            (self.market_input_digest, "market_input_digest"),
            (self.cost_input_digest, "cost_input_digest"),
            (self.combined_input_digest, "combined_input_digest"),
        ):
            _require_sha(value, name)
        _require_utc(self.observed_at, "observed_at")
        if tuple(sorted(set(self.horizon_bars))) != self.horizon_bars:
            raise ValueError("factor factory fingerprint horizons invalid")
        for lower_name, upper_name in (
            ("feature_min_ts", "feature_max_ts"),
            ("market_min_ts", "market_max_ts"),
        ):
            lower = getattr(self, lower_name)
            upper = getattr(self, upper_name)
            if (lower is None) != (upper is None):
                raise ValueError(f"factor factory fingerprint {lower_name} bounds incomplete")
            if lower is not None:
                _require_utc(lower, lower_name)
                _require_utc(upper, upper_name)
                if lower > upper:
                    raise ValueError(f"factor factory fingerprint {lower_name} bounds invalid")
        return self


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


class AlphaFactoryTaskParameters(StrictModel):
    as_of_date: date
    lookback_days: int = Field(default=30, ge=1, le=180)
    max_candidates: int = Field(default=200, ge=1, le=200)


class FactorResearchTaskParameters(StrictModel):
    as_of_date: date
    start_date: date
    end_date: date
    max_history_days: int = Field(ge=1, le=1095)
    hypothesis_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    trial_ids: tuple[str, ...] = Field(min_length=1, max_length=54)

    @model_validator(mode="after")
    def validate_parameters(self) -> FactorResearchTaskParameters:
        if self.end_date < self.start_date:
            raise ValueError("factor research end_date precedes start_date")
        if (self.end_date - self.start_date).days > self.max_history_days:
            raise ValueError("factor research window exceeds max_history_days")
        _require_unique_identifiers(self.hypothesis_ids, "hypothesis_ids")
        _require_unique_identifiers(self.trial_ids, "trial_ids")
        return self


class FactorFactoryTaskParameters(StrictModel):
    as_of_date: date
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    horizon_bars: tuple[int, ...] = Field(min_length=1, max_length=32)
    decision_delay_bars: int = Field(ge=1, le=1000)
    max_factors: int = Field(ge=1, le=200)
    min_samples: int = Field(ge=1)
    top_quantile: float = Field(ge=0.01, le=0.5)
    cost_quantile: Literal["p50", "p75", "p90"]
    result_mode: Literal["PARITY_FULL"] = "PARITY_FULL"
    history_mode: Literal["bootstrap_full"] = "bootstrap_full"

    @model_validator(mode="after")
    def validate_parameters(self) -> FactorFactoryTaskParameters:
        if tuple(sorted(set(self.horizon_bars))) != self.horizon_bars:
            raise ValueError("factor factory horizon_bars must be positive, sorted, and unique")
        if any(value <= 0 for value in self.horizon_bars):
            raise ValueError("factor factory horizon_bars must be positive")
        if any(value > 10_000 for value in self.horizon_bars):
            raise ValueError("factor factory horizon_bars exceeds upper bound")
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


class AlphaFactorySnapshotManifest(StrictModel):
    schema_version: Literal["quant_lab_alpha_factory_snapshot.v1"] = ALPHA_FACTORY_SNAPSHOT_SCHEMA
    task_type: Literal["alpha_factory"] = ALPHA_FACTORY_TASK_TYPE
    snapshot_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    alpha_factory_schema_version: str = Field(min_length=1, max_length=80)
    second_stage_schema_version: str = Field(min_length=1, max_length=80)
    template_registry_digest: str = Field(min_length=64, max_length=64)
    factor_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    factor_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    factor_generation_as_of_date: date | None = None
    factor_generation_published_at: datetime | None = None
    hypothesis_registry_digest: str | None = Field(default=None, min_length=64, max_length=64)
    trial_ledger_digest: str | None = Field(default=None, min_length=64, max_length=64)
    factor_generation_fresh: bool | None = None
    factor_generation_hypothesis_ids: tuple[str, ...] | None = None
    source_input_digest: str = Field(min_length=64, max_length=64)
    as_of_date: date
    lookback_days: int = Field(ge=1, le=180)
    max_candidates: int = Field(ge=1, le=200)
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
    def validate_manifest(self) -> AlphaFactorySnapshotManifest:
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_utc(self.generated_at, "generated_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.template_registry_digest, "template_registry_digest")
        _validate_optional_alpha_factor_generation(self)
        _require_sha(self.source_input_digest, "source_input_digest")
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


class AlphaFactoryTask(StrictModel):
    schema_version: Literal["quant_lab_alpha_factory_task.v1"] = ALPHA_FACTORY_TASK_SCHEMA
    task_type: Literal["alpha_factory"] = ALPHA_FACTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    as_of_date: date
    lookback_days: int = Field(default=30, ge=1, le=180)
    max_candidates: int = Field(default=200, ge=1, le=200)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    alpha_factory_schema_version: str = Field(min_length=1, max_length=80)
    second_stage_schema_version: str = Field(min_length=1, max_length=80)
    template_registry_digest: str = Field(min_length=64, max_length=64)
    factor_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    factor_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    factor_generation_as_of_date: date | None = None
    factor_generation_published_at: datetime | None = None
    hypothesis_registry_digest: str | None = Field(default=None, min_length=64, max_length=64)
    trial_ledger_digest: str | None = Field(default=None, min_length=64, max_length=64)
    factor_generation_fresh: bool | None = None
    factor_generation_hypothesis_ids: tuple[str, ...] | None = None
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    requested_at: datetime
    lease_seconds: int = Field(default=7200, ge=60, le=24 * 60 * 60)
    max_attempts: int = Field(default=3, ge=1, le=10)
    signature_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    automatic_promotion: Literal[False] = False
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task(self) -> AlphaFactoryTask:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.template_registry_digest, "template_registry_digest")
        _validate_optional_alpha_factor_generation(self)
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_utc(self.requested_at, "requested_at")
        _require_identifier(self.signature_key_id, "signature_key_id")
        return self

    @property
    def parameters(self) -> AlphaFactoryTaskParameters:
        return AlphaFactoryTaskParameters(
            as_of_date=self.as_of_date,
            lookback_days=self.lookback_days,
            max_candidates=self.max_candidates,
        )


class FactorResearchSnapshotManifest(StrictModel):
    schema_version: Literal["quant_lab_factor_research_snapshot.v1"] = (
        FACTOR_RESEARCH_SNAPSHOT_SCHEMA
    )
    task_type: Literal["factor_research"] = FACTOR_RESEARCH_TASK_TYPE
    snapshot_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    factor_research_schema_version: str = Field(min_length=1, max_length=80)
    hypothesis_registry_digest: str = Field(min_length=64, max_length=64)
    trial_ledger_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    as_of_date: date
    start_date: date
    end_date: date
    max_history_days: int = Field(ge=1, le=1095)
    hypothesis_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    trial_ids: tuple[str, ...] = Field(min_length=1, max_length=54)
    test_count: int = Field(ge=1, le=54)
    datasets: list[str]
    files: list[ResearchDatasetReference]
    total_input_bytes: int = Field(ge=0)
    total_input_rows: int = Field(ge=0)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> FactorResearchSnapshotManifest:
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_utc(self.generated_at, "generated_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.hypothesis_registry_digest, "hypothesis_registry_digest")
        _require_sha(self.trial_ledger_digest, "trial_ledger_digest")
        _require_sha(self.source_input_digest, "source_input_digest")
        _require_sha(self.manifest_sha256, "manifest_sha256")
        _require_identifier(self.signature_key_id, "signature_key_id")
        FactorResearchTaskParameters(
            as_of_date=self.as_of_date,
            start_date=self.start_date,
            end_date=self.end_date,
            max_history_days=self.max_history_days,
            hypothesis_ids=self.hypothesis_ids,
            trial_ids=self.trial_ids,
        )
        if self.test_count != len(self.trial_ids):
            raise ValueError("factor research test_count must equal trial count")
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


class FactorResearchTask(StrictModel):
    schema_version: Literal["quant_lab_factor_research_task.v1"] = FACTOR_RESEARCH_TASK_SCHEMA
    task_type: Literal["factor_research"] = FACTOR_RESEARCH_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    as_of_date: date
    start_date: date
    end_date: date
    max_history_days: int = Field(ge=1, le=1095)
    hypothesis_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    trial_ids: tuple[str, ...] = Field(min_length=1, max_length=54)
    test_count: int = Field(ge=1, le=54)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    factor_research_schema_version: str = Field(min_length=1, max_length=80)
    hypothesis_registry_digest: str = Field(min_length=64, max_length=64)
    trial_ledger_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    requested_at: datetime
    lease_seconds: int = Field(default=7200, ge=60, le=24 * 60 * 60)
    max_attempts: int = Field(default=3, ge=1, le=10)
    signature_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task(self) -> FactorResearchTask:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.hypothesis_registry_digest, "hypothesis_registry_digest")
        _require_sha(self.trial_ledger_digest, "trial_ledger_digest")
        _require_sha(self.source_input_digest, "source_input_digest")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_utc(self.requested_at, "requested_at")
        _require_identifier(self.signature_key_id, "signature_key_id")
        FactorResearchTaskParameters(
            as_of_date=self.as_of_date,
            start_date=self.start_date,
            end_date=self.end_date,
            max_history_days=self.max_history_days,
            hypothesis_ids=self.hypothesis_ids,
            trial_ids=self.trial_ids,
        )
        if self.test_count != len(self.trial_ids):
            raise ValueError("factor research test_count must equal trial count")
        return self

    @property
    def parameters(self) -> FactorResearchTaskParameters:
        return FactorResearchTaskParameters(
            as_of_date=self.as_of_date,
            start_date=self.start_date,
            end_date=self.end_date,
            max_history_days=self.max_history_days,
            hypothesis_ids=self.hypothesis_ids,
            trial_ids=self.trial_ids,
        )


class FactorFactoryCostSnapshotRecord(StrictModel):
    symbol: str = Field(min_length=1, max_length=80)
    cost_date: str | None = Field(default=None, min_length=10, max_length=32)
    cost_model_version: str = Field(min_length=1, max_length=180)
    cost_source: str = Field(min_length=1, max_length=180)
    cost_quantile: Literal["p50", "p75", "p90"]
    cost_bps: float = Field(ge=0.0, le=1_000_000.0)


class FactorFactoryPreviousGeneration(StrictModel):
    schema_version: Literal["factor_factory_generation.v1"]
    generation_id: str = Field(min_length=1, max_length=180)
    generation_digest: str = Field(min_length=64, max_length=64)
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    factor_plan_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    cost_input_digest: str = Field(min_length=64, max_length=64)
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    as_of_date: date
    row_counts: dict[str, int]
    dataset_hashes: dict[str, str]
    published_at: datetime
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0

    @model_validator(mode="after")
    def validate_previous_generation(self) -> FactorFactoryPreviousGeneration:
        for value, name in (
            (self.generation_digest, "generation_digest"),
            (self.factor_plan_digest, "factor_plan_digest"),
            (self.source_input_digest, "source_input_digest"),
            (self.cost_input_digest, "cost_input_digest"),
        ):
            _require_sha(value, name)
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_utc(self.published_at, "published_at")
        if any(value < 0 for value in self.row_counts.values()):
            raise ValueError("factor factory previous generation row count invalid")
        expected_datasets = {
            "factor_definition",
            "factor_value",
            "factor_evidence",
            "factor_candidate",
            "factor_correlation_daily",
        }
        if set(self.row_counts) != expected_datasets or set(self.dataset_hashes) != (
            expected_datasets
        ):
            raise ValueError("factor factory previous generation dataset set invalid")
        for value in self.dataset_hashes.values():
            _require_sha(value, "dataset_hash")
        return self


class FactorFactorySnapshotManifest(StrictModel):
    schema_version: Literal[
        "quant_lab_factor_factory_snapshot.v1",
        "quant_lab_factor_factory_snapshot.v2",
        "quant_lab_factor_factory_snapshot.v3",
    ] = FACTOR_FACTORY_SNAPSHOT_SCHEMA
    task_type: Literal["factor_factory"] = FACTOR_FACTORY_TASK_TYPE
    snapshot_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    as_of_date: date
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    horizon_bars: tuple[int, ...] = Field(min_length=1, max_length=32)
    decision_delay_bars: int = Field(ge=1, le=1000)
    max_factors: int = Field(ge=1, le=200)
    min_samples: int = Field(ge=1)
    top_quantile: float = Field(gt=0.0, le=0.5)
    cost_quantile: Literal["p50", "p75", "p90"]
    result_mode: Literal["PARITY_FULL"] = "PARITY_FULL"
    history_mode: Literal["bootstrap_full"] = "bootstrap_full"
    factor_plan: EffectiveFactorPlan
    factor_plan_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    cost_input_digest: str = Field(min_length=64, max_length=64)
    feature_input_digest: str | None = Field(default=None, min_length=64, max_length=64)
    market_input_digest: str | None = Field(default=None, min_length=64, max_length=64)
    combined_input_digest: str | None = Field(default=None, min_length=64, max_length=64)
    input_fingerprint: FactorFactoryInputFingerprint | None = None
    cost_snapshot: tuple[FactorFactoryCostSnapshotRecord, ...] = ()
    feature_min_ts: datetime | None = None
    feature_max_ts: datetime | None = None
    market_min_ts: datetime | None = None
    market_max_ts: datetime | None = None
    previous_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    previous_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    previous_generation_manifest: FactorFactoryPreviousGeneration | None = None
    source_files: tuple[FactorFactorySourceFileIdentity, ...] = ()
    feature_estimated_uncompressed_bytes: int = Field(default=0, ge=0)
    market_estimated_uncompressed_bytes: int = Field(default=0, ge=0)
    cost_estimated_uncompressed_bytes: int = Field(default=0, ge=0)
    estimated_uncompressed_bytes: int = Field(default=0, ge=0)
    compressed_input_bytes: int | None = Field(default=None, ge=0)
    estimated_uncompressed_input_bytes: int | None = Field(default=None, ge=0)
    feature_compressed_bytes: int | None = Field(default=None, ge=0)
    feature_uncompressed_bytes: int | None = Field(default=None, ge=0)
    market_compressed_bytes: int | None = Field(default=None, ge=0)
    market_uncompressed_bytes: int | None = Field(default=None, ge=0)
    cost_compressed_bytes: int | None = Field(default=None, ge=0)
    cost_uncompressed_bytes: int | None = Field(default=None, ge=0)
    datasets: list[str]
    files: list[ResearchDatasetReference]
    total_input_bytes: int = Field(ge=0)
    total_input_rows: int = Field(ge=0)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    signature_key_id: str = Field(min_length=1, max_length=120)
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_manifest(self) -> FactorFactorySnapshotManifest:
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_utc(self.generated_at, "generated_at")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.factor_plan_digest, "factor_plan_digest")
        _require_sha(self.source_input_digest, "source_input_digest")
        _require_sha(self.cost_input_digest, "cost_input_digest")
        _require_sha(self.manifest_sha256, "manifest_sha256")
        _require_identifier(self.signature_key_id, "signature_key_id")
        FactorFactoryTaskParameters(
            as_of_date=self.as_of_date,
            feature_set=self.feature_set,
            feature_version=self.feature_version,
            factor_version=self.factor_version,
            timeframe=self.timeframe,
            horizon_bars=self.horizon_bars,
            decision_delay_bars=self.decision_delay_bars,
            max_factors=self.max_factors,
            min_samples=self.min_samples,
            top_quantile=self.top_quantile,
            cost_quantile=self.cost_quantile,
            result_mode=self.result_mode,
            history_mode=self.history_mode,
        )
        if self.factor_plan.plan_digest != self.factor_plan_digest:
            raise ValueError("factor factory snapshot plan digest mismatch")
        if self.factor_plan.quant_lab_commit != self.quant_lab_commit:
            raise ValueError("factor factory snapshot plan commit mismatch")
        if (
            self.factor_plan.feature_set != self.feature_set
            or self.factor_plan.feature_version != self.feature_version
            or self.factor_plan.factor_version != self.factor_version
            or self.factor_plan.timeframe != self.timeframe
            or self.factor_plan.max_factors != self.max_factors
        ):
            raise ValueError("factor factory snapshot plan scope mismatch")
        if self.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V1:
            _validate_previous_generation(self)
            if self.previous_generation_manifest is None:
                if self.previous_generation_id is not None:
                    raise ValueError("factor factory previous generation manifest missing")
            elif (
                self.previous_generation_manifest.generation_id != self.previous_generation_id
                or self.previous_generation_manifest.generation_digest
                != self.previous_generation_digest
            ):
                raise ValueError("factor factory previous generation manifest mismatch")
        elif self.schema_version == FACTOR_FACTORY_SNAPSHOT_SCHEMA_V2:
            if any(
                value is not None
                for value in (
                    self.previous_generation_id,
                    self.previous_generation_digest,
                    self.previous_generation_manifest,
                )
            ):
                raise ValueError("factor factory v2 snapshot must not bind previous generation")
            source_paths = [item.relative_path for item in self.source_files]
            if len(source_paths) != len(set(source_paths)):
                raise ValueError("factor factory source file paths must be unique")
            expected_uncompressed = sum(item.uncompressed_bytes for item in self.source_files)
            if expected_uncompressed != self.estimated_uncompressed_bytes:
                raise ValueError("factor factory estimated uncompressed bytes mismatch")
            by_dataset = {
                dataset: sum(
                    item.uncompressed_bytes
                    for item in self.source_files
                    if item.dataset_name == dataset
                )
                for dataset in self.datasets
            }
            if self.feature_estimated_uncompressed_bytes != by_dataset.get("gold/feature_value", 0):
                raise ValueError("factor factory feature uncompressed estimate mismatch")
            if self.market_estimated_uncompressed_bytes != by_dataset.get("silver/market_bar", 0):
                raise ValueError("factor factory market uncompressed estimate mismatch")
            if self.cost_estimated_uncompressed_bytes != by_dataset.get(
                "gold/cost_bucket_daily", 0
            ):
                raise ValueError("factor factory cost uncompressed estimate mismatch")
            if any(item.schema_fingerprint is None for item in self.files):
                raise ValueError("factor factory v2 snapshot file schema fingerprint missing")
            if any(item.uncompressed_bytes is None for item in self.files):
                raise ValueError("factor factory v2 snapshot file uncompressed size missing")
        else:
            if any(
                value is not None
                for value in (
                    self.previous_generation_id,
                    self.previous_generation_digest,
                    self.previous_generation_manifest,
                )
            ):
                raise ValueError("factor factory v3 snapshot must not bind previous generation")
            required_digests = {
                "feature_input_digest": self.feature_input_digest,
                "market_input_digest": self.market_input_digest,
                "combined_input_digest": self.combined_input_digest,
            }
            if any(value is None for value in required_digests.values()):
                raise ValueError("factor factory v3 snapshot input digests missing")
            for name, value in required_digests.items():
                _require_sha(str(value), name)
            if self.input_fingerprint is None:
                raise ValueError("factor factory v3 snapshot fingerprint missing")
            fingerprint = self.input_fingerprint
            if (
                fingerprint.quant_lab_commit != self.quant_lab_commit
                or fingerprint.factor_plan_digest != self.factor_plan_digest
                or fingerprint.feature_input_digest != self.feature_input_digest
                or fingerprint.market_input_digest != self.market_input_digest
                or fingerprint.cost_input_digest != self.cost_input_digest
                or fingerprint.combined_input_digest != self.combined_input_digest
                or fingerprint.feature_set != self.feature_set
                or fingerprint.feature_version != self.feature_version
                or fingerprint.factor_version != self.factor_version
                or fingerprint.timeframe != self.timeframe
                or fingerprint.horizon_bars != self.horizon_bars
                or fingerprint.decision_delay_bars != self.decision_delay_bars
                or fingerprint.max_factors != self.max_factors
                or fingerprint.min_samples != self.min_samples
                or fingerprint.top_quantile != self.top_quantile
                or fingerprint.cost_quantile != self.cost_quantile
            ):
                raise ValueError("factor factory v3 snapshot fingerprint mismatch")
            source_paths = [item.relative_path for item in self.source_files]
            if len(source_paths) != len(set(source_paths)):
                raise ValueError("factor factory source file paths must be unique")
            if any(item.schema_fingerprint is None for item in self.files):
                raise ValueError("factor factory v3 snapshot file schema fingerprint missing")
            if any(item.uncompressed_bytes is None for item in self.files):
                raise ValueError("factor factory v3 snapshot file uncompressed size missing")
            required_capacity = (
                self.compressed_input_bytes,
                self.estimated_uncompressed_input_bytes,
                self.feature_compressed_bytes,
                self.feature_uncompressed_bytes,
                self.market_compressed_bytes,
                self.market_uncompressed_bytes,
                self.cost_compressed_bytes,
                self.cost_uncompressed_bytes,
            )
            if any(value is None for value in required_capacity):
                raise ValueError("factor factory v3 snapshot capacity fields missing")
            compressed_by_dataset = {
                dataset: sum(item.size_bytes for item in self.files if item.dataset_name == dataset)
                for dataset in self.datasets
            }
            uncompressed_by_dataset = {
                dataset: sum(
                    int(item.uncompressed_bytes or 0)
                    for item in self.files
                    if item.dataset_name == dataset
                )
                for dataset in self.datasets
            }
            if self.compressed_input_bytes != sum(compressed_by_dataset.values()):
                raise ValueError("factor factory v3 compressed input bytes mismatch")
            if self.estimated_uncompressed_input_bytes != sum(
                uncompressed_by_dataset.values()
            ):
                raise ValueError("factor factory v3 uncompressed input bytes mismatch")
            capacity_pairs = (
                (self.feature_compressed_bytes, compressed_by_dataset.get("gold/feature_value", 0)),
                (
                    self.feature_uncompressed_bytes,
                    uncompressed_by_dataset.get("gold/feature_value", 0),
                ),
                (self.market_compressed_bytes, compressed_by_dataset.get("silver/market_bar", 0)),
                (
                    self.market_uncompressed_bytes,
                    uncompressed_by_dataset.get("silver/market_bar", 0),
                ),
                (
                    self.cost_compressed_bytes,
                    compressed_by_dataset.get("gold/cost_bucket_daily", 0),
                ),
                (
                    self.cost_uncompressed_bytes,
                    uncompressed_by_dataset.get("gold/cost_bucket_daily", 0),
                ),
            )
            if any(observed != expected for observed, expected in capacity_pairs):
                raise ValueError("factor factory v3 dataset capacity fields mismatch")
            if self.total_input_bytes != self.compressed_input_bytes:
                raise ValueError("factor factory v3 total input bytes mismatch")
            if self.estimated_uncompressed_bytes != self.estimated_uncompressed_input_bytes:
                raise ValueError("factor factory v3 legacy uncompressed alias mismatch")
        cost_symbols = [item.symbol for item in self.cost_snapshot]
        if len(cost_symbols) != len(set(cost_symbols)):
            raise ValueError("factor factory cost snapshot symbols must be unique")
        if any(item.cost_quantile != self.cost_quantile for item in self.cost_snapshot):
            raise ValueError("factor factory cost snapshot quantile mismatch")
        for field_name in ("feature_min_ts", "feature_max_ts", "market_min_ts", "market_max_ts"):
            value = getattr(self, field_name)
            if value is not None:
                _require_utc(value, field_name)
        if (self.feature_min_ts is None) != (self.feature_max_ts is None):
            raise ValueError("factor factory feature bounds must be complete")
        if (self.market_min_ts is None) != (self.market_max_ts is None):
            raise ValueError("factor factory market bounds must be complete")
        if self.feature_min_ts is not None and self.feature_min_ts > self.feature_max_ts:
            raise ValueError("factor factory feature bounds invalid")
        if self.market_min_ts is not None and self.market_min_ts > self.market_max_ts:
            raise ValueError("factor factory market bounds invalid")
        _validate_snapshot_files(
            self.datasets, self.files, self.total_input_bytes, self.total_input_rows
        )
        return self


class FactorFactoryTask(StrictModel):
    schema_version: Literal["quant_lab_factor_factory_task.v1"] = FACTOR_FACTORY_TASK_SCHEMA
    task_type: Literal["factor_factory"] = FACTOR_FACTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    as_of_date: date
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    horizon_bars: tuple[int, ...] = Field(min_length=1, max_length=32)
    decision_delay_bars: int = Field(ge=1, le=1000)
    max_factors: int = Field(ge=1, le=200)
    min_samples: int = Field(ge=1)
    top_quantile: float = Field(gt=0.0, le=0.5)
    cost_quantile: Literal["p50", "p75", "p90"]
    result_mode: Literal["PARITY_FULL"] = "PARITY_FULL"
    history_mode: Literal["bootstrap_full"] = "bootstrap_full"
    factor_plan_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    cost_input_digest: str = Field(min_length=64, max_length=64)
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
    def validate_task(self) -> FactorFactoryTask:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_sha(self.factor_plan_digest, "factor_plan_digest")
        _require_sha(self.source_input_digest, "source_input_digest")
        _require_sha(self.cost_input_digest, "cost_input_digest")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_utc(self.requested_at, "requested_at")
        _require_identifier(self.signature_key_id, "signature_key_id")
        _validate_previous_generation(self)
        FactorFactoryTaskParameters(
            as_of_date=self.as_of_date,
            feature_set=self.feature_set,
            feature_version=self.feature_version,
            factor_version=self.factor_version,
            timeframe=self.timeframe,
            horizon_bars=self.horizon_bars,
            decision_delay_bars=self.decision_delay_bars,
            max_factors=self.max_factors,
            min_samples=self.min_samples,
            top_quantile=self.top_quantile,
            cost_quantile=self.cost_quantile,
            result_mode=self.result_mode,
            history_mode=self.history_mode,
        )
        return self

    @property
    def parameters(self) -> FactorFactoryTaskParameters:
        return FactorFactoryTaskParameters(
            as_of_date=self.as_of_date,
            feature_set=self.feature_set,
            feature_version=self.feature_version,
            factor_version=self.factor_version,
            timeframe=self.timeframe,
            horizon_bars=self.horizon_bars,
            decision_delay_bars=self.decision_delay_bars,
            max_factors=self.max_factors,
            min_samples=self.min_samples,
            top_quantile=self.top_quantile,
            cost_quantile=self.cost_quantile,
            result_mode=self.result_mode,
            history_mode=self.history_mode,
        )


ResearchTaskEnvelope: TypeAlias = Annotated[
    ResearchTask
    | AlphaFactoryTask
    | FactorResearchTask
    | FactorFactoryTask
    | V5CandidateEvidenceTask
    | TradeLevelHistoryTask,
    Field(discriminator="task_type"),
]
RESEARCH_TASK_ADAPTER = TypeAdapter(ResearchTaskEnvelope)


ResearchSnapshotEnvelope: TypeAlias = Annotated[
    ResearchSnapshotManifest
    | AlphaFactorySnapshotManifest
    | FactorResearchSnapshotManifest
    | FactorFactorySnapshotManifest
    | V5CandidateEvidenceSnapshotManifest
    | TradeLevelHistorySnapshotManifest,
    Field(discriminator="schema_version"),
]
RESEARCH_SNAPSHOT_ADAPTER = TypeAdapter(ResearchSnapshotEnvelope)


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


class AlphaFactoryResultManifest(StrictModel):
    schema_version: Literal["quant_lab_alpha_factory_result.v1"] = ALPHA_FACTORY_RESULT_SCHEMA
    task_type: Literal["alpha_factory"] = ALPHA_FACTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    alpha_factory_schema_version: str = Field(min_length=1, max_length=80)
    second_stage_schema_version: str = Field(min_length=1, max_length=80)
    template_registry_digest: str = Field(min_length=64, max_length=64)
    factor_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    factor_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    factor_generation_as_of_date: date | None = None
    factor_generation_published_at: datetime | None = None
    hypothesis_registry_digest: str | None = Field(default=None, min_length=64, max_length=64)
    trial_ledger_digest: str | None = Field(default=None, min_length=64, max_length=64)
    factor_generation_fresh: bool | None = None
    factor_generation_hypothesis_ids: tuple[str, ...] | None = None
    as_of_date: date
    lookback_days: int = Field(ge=1, le=180)
    max_candidates: int = Field(ge=1, le=200)
    generation_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    completed_at: datetime
    outputs: list[ResearchOutputDataset]
    reports: list[ResearchOutputFile]
    anti_leakage_status: Literal["PASS"]
    anti_leakage_violation_count: Literal[0] = 0
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
    automatic_promotion: Literal[False] = False
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> AlphaFactoryResultManifest:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.generation_id, "generation_id")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_sha(self.template_registry_digest, "template_registry_digest")
        _validate_optional_alpha_factor_generation(self)
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


class FactorResearchResultManifest(StrictModel):
    schema_version: Literal["quant_lab_factor_research_result.v1"] = FACTOR_RESEARCH_RESULT_SCHEMA
    task_type: Literal["factor_research"] = FACTOR_RESEARCH_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    selected_v5_bundle_id: str = Field(min_length=1, max_length=512)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    factor_research_schema_version: str = Field(min_length=1, max_length=80)
    hypothesis_registry_digest: str = Field(min_length=64, max_length=64)
    trial_ledger_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    as_of_date: date
    start_date: date
    end_date: date
    max_history_days: int = Field(ge=1, le=1095)
    hypothesis_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    trial_ids: tuple[str, ...] = Field(min_length=1, max_length=54)
    test_count: int = Field(ge=1, le=54)
    multiple_testing_family: str = Field(min_length=1, max_length=180)
    generation_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    completed_at: datetime
    outputs: list[ResearchOutputDataset]
    reports: list[ResearchOutputFile]
    anti_leakage_status: Literal["PASS"]
    anti_leakage_violation_count: Literal[0] = 0
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
    automatic_promotion: Literal[False] = False
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result(self) -> FactorResearchResultManifest:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.generation_id, "generation_id")
        _require_identifier(self.multiple_testing_family, "multiple_testing_family")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_sha(self.hypothesis_registry_digest, "hypothesis_registry_digest")
        _require_sha(self.trial_ledger_digest, "trial_ledger_digest")
        _require_sha(self.source_input_digest, "source_input_digest")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_commit(self.worker_commit, "worker_commit")
        _require_utc(self.generated_at, "generated_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        FactorResearchTaskParameters(
            as_of_date=self.as_of_date,
            start_date=self.start_date,
            end_date=self.end_date,
            max_history_days=self.max_history_days,
            hypothesis_ids=self.hypothesis_ids,
            trial_ids=self.trial_ids,
        )
        if self.test_count != len(self.trial_ids):
            raise ValueError("factor research test_count must equal trial count")
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


class FactorFactoryPartitionReference(StrictModel):
    dataset_name: Literal["factor_value"] = "factor_value"
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    partition_date: date
    part_number: int = Field(ge=0)
    relative_path: str = Field(min_length=1, max_length=1024)
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    sha256: str = Field(min_length=64, max_length=64)
    row_count: int = Field(ge=1)
    size_bytes: int = Field(ge=1)
    uncompressed_bytes: int | None = Field(default=None, ge=1)
    partition_identity: str | None = Field(default=None, min_length=64, max_length=64)
    min_ts: datetime
    max_ts: datetime
    primary_keys: tuple[str, ...] = (
        "factor_id",
        "factor_version",
        "symbol",
        "timeframe",
        "ts",
    )
    publish_mode: Literal["keyed_upsert"] = "keyed_upsert"
    empty_result_semantics: Literal["preserve"] = "preserve"
    state: Literal["changed"] = "changed"

    @model_validator(mode="after")
    def validate_partition(self) -> FactorFactoryPartitionReference:
        _require_safe_relative_path(self.relative_path)
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_sha(self.sha256, "sha256")
        if self.partition_identity is not None:
            _require_sha(self.partition_identity, "partition_identity")
        _require_utc(self.min_ts, "min_ts")
        _require_utc(self.max_ts, "max_ts")
        if self.min_ts > self.max_ts:
            raise ValueError("factor value partition bounds invalid")
        expected_keys = (
            "factor_id",
            "factor_version",
            "symbol",
            "timeframe",
            "ts",
        )
        if self.primary_keys != expected_keys:
            raise ValueError("factor value partition primary keys mismatch")
        expected_prefix = (
            f"outputs/factor_value/factor_version={self.factor_version}/"
            f"timeframe={self.timeframe}/date={self.partition_date.isoformat()}/"
        )
        if not self.relative_path.startswith(expected_prefix):
            raise ValueError("factor value partition path does not match its identity")
        return self


class FactorFactoryOutputDataset(StrictModel):
    dataset_name: Literal[
        "factor_definition_preview",
        "factor_evidence",
        "factor_correlation_daily",
    ]
    relative_path: str = Field(min_length=1, max_length=1024)
    schema_fingerprint: str = Field(min_length=64, max_length=64)
    sha256: str = Field(min_length=64, max_length=64)
    row_count: int = Field(ge=0)
    size_bytes: int = Field(ge=1)
    primary_keys: tuple[str, ...]
    publish_mode: Literal["keyed_upsert"] = "keyed_upsert"
    empty_result_semantics: Literal["preserve"] = "preserve"

    @model_validator(mode="after")
    def validate_output(self) -> FactorFactoryOutputDataset:
        _require_safe_relative_path(self.relative_path)
        _require_sha(self.schema_fingerprint, "schema_fingerprint")
        _require_sha(self.sha256, "sha256")
        expected_keys = {
            "factor_definition_preview": ("factor_id", "factor_version"),
            "factor_evidence": (
                "as_of_date",
                "factor_id",
                "factor_version",
                "timeframe",
                "horizon_bars",
                "decision_delay_bars",
            ),
            "factor_correlation_daily": (
                "as_of_date",
                "factor_id_left",
                "factor_id_right",
                "factor_version",
                "timeframe",
            ),
        }
        if self.primary_keys != expected_keys[self.dataset_name]:
            raise ValueError(f"{self.dataset_name} primary keys mismatch")
        if self.relative_path != f"outputs/{self.dataset_name}.parquet":
            raise ValueError(f"{self.dataset_name} output path mismatch")
        return self


class FactorFactoryAntiLeakageCheck(StrictModel):
    check_name: str = Field(min_length=1, max_length=180)
    status: Literal["PASS"] = "PASS"
    violation_count: Literal[0] = 0
    detail: str = Field(min_length=1, max_length=2000)

    @field_validator("check_name")
    @classmethod
    def validate_check_name(cls, value: str) -> str:
        _require_identifier(value, "check_name")
        return value


class FactorFactoryResultManifest(StrictModel):
    schema_version: Literal[
        "quant_lab_factor_factory_result.v1",
        "quant_lab_factor_factory_result.v2",
    ] = FACTOR_FACTORY_RESULT_SCHEMA
    task_type: Literal["factor_factory"] = FACTOR_FACTORY_TASK_TYPE
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    snapshot_manifest_sha256: str = Field(min_length=64, max_length=64)
    quant_lab_commit: str = Field(min_length=40, max_length=40)
    worker_commit: str = Field(min_length=40, max_length=40)
    factor_plan_digest: str = Field(min_length=64, max_length=64)
    source_input_digest: str = Field(min_length=64, max_length=64)
    cost_input_digest: str = Field(min_length=64, max_length=64)
    previous_generation_id: str | None = Field(default=None, min_length=1, max_length=180)
    previous_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    as_of_date: date
    feature_set: str = Field(min_length=1, max_length=80)
    feature_version: str = Field(min_length=1, max_length=80)
    factor_version: str = Field(min_length=1, max_length=80)
    timeframe: str = Field(min_length=1, max_length=32)
    horizon_bars: tuple[int, ...] = Field(min_length=1, max_length=32)
    decision_delay_bars: int = Field(ge=1, le=1000)
    min_samples: int = Field(ge=1)
    top_quantile: float = Field(ge=0.01, le=0.5)
    cost_quantile: Literal["p50", "p75", "p90"]
    result_mode: Literal["PARITY_FULL"] = "PARITY_FULL"
    history_mode: Literal["bootstrap_full"] = "bootstrap_full"
    generation_id: str = Field(min_length=1, max_length=180)
    generated_at: datetime
    completed_at: datetime
    factor_ids: tuple[str, ...] = Field(max_length=200)
    factor_count: int = Field(ge=0, le=200)
    value_partitions: tuple[FactorFactoryPartitionReference, ...]
    outputs: tuple[FactorFactoryOutputDataset, ...]
    reports: tuple[ResearchOutputFile, ...]
    anti_leakage_checks: tuple[FactorFactoryAntiLeakageCheck, ...] = Field(min_length=20)
    anti_leakage_status: Literal["PASS"] = "PASS"
    anti_leakage_violation_count: Literal[0] = 0
    completed_no_update: bool = False
    no_update_reason: str | None = Field(default=None, min_length=1, max_length=180)
    warnings: tuple[str, ...] = ()
    input_bytes: int = Field(ge=0)
    cache_hit_bytes: int = Field(ge=0)
    downloaded_bytes: int = Field(ge=0)
    output_bytes: int = Field(ge=0)
    peak_rss_bytes: int = Field(ge=0)
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
    def validate_result(self) -> FactorFactoryResultManifest:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.generation_id, "generation_id")
        _require_commit(self.quant_lab_commit, "quant_lab_commit")
        _require_commit(self.worker_commit, "worker_commit")
        _require_sha(self.snapshot_manifest_sha256, "snapshot_manifest_sha256")
        _require_sha(self.factor_plan_digest, "factor_plan_digest")
        _require_sha(self.source_input_digest, "source_input_digest")
        _require_sha(self.cost_input_digest, "cost_input_digest")
        _validate_previous_generation(self)
        _require_utc(self.generated_at, "generated_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        if self.completed_at < self.generated_at:
            raise ValueError("factor factory result completed_at precedes generated_at")
        if tuple(sorted(set(self.factor_ids))) != self.factor_ids:
            raise ValueError("factor factory factor_ids must be sorted and unique")
        if self.factor_count != len(self.factor_ids):
            raise ValueError("factor factory factor_count mismatch")
        if tuple(sorted(set(self.horizon_bars))) != self.horizon_bars or any(
            value <= 0 or value > 10_000 for value in self.horizon_bars
        ):
            raise ValueError("factor factory result horizons invalid")
        if self.schema_version != FACTOR_FACTORY_RESULT_SCHEMA_V1 and any(
            item.partition_identity is None or item.uncompressed_bytes is None
            for item in self.value_partitions
        ):
            raise ValueError("factor factory v2 partition capacity identity missing")
        check_names = [item.check_name for item in self.anti_leakage_checks]
        if len(check_names) != len(set(check_names)):
            raise ValueError("factor factory anti-leakage checks must be unique")
        output_names = [item.dataset_name for item in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError("factor factory output datasets must be unique")
        partition_paths = [item.relative_path for item in self.value_partitions]
        report_paths = [item.relative_path for item in self.reports]
        if len(partition_paths) != len(set(partition_paths)):
            raise ValueError("factor factory partition paths must be unique")
        if len(report_paths) != len(set(report_paths)):
            raise ValueError("factor factory report paths must be unique")
        expected_bytes = (
            sum(item.size_bytes for item in self.value_partitions)
            + sum(item.size_bytes for item in self.outputs)
            + sum(item.size_bytes for item in self.reports)
        )
        if self.output_bytes != expected_bytes:
            raise ValueError("factor factory result output_bytes mismatch")
        if self.completed_no_update:
            if self.no_update_reason is None:
                raise ValueError("completed_no_update requires no_update_reason")
            if self.value_partitions or self.outputs:
                raise ValueError("completed_no_update must not carry publishable outputs")
        else:
            if self.no_update_reason is not None:
                raise ValueError("no_update_reason requires completed_no_update")
            if set(output_names) != {
                "factor_definition_preview",
                "factor_evidence",
                "factor_correlation_daily",
            }:
                raise ValueError("factor factory result output set mismatch")
            if not self.value_partitions:
                raise ValueError("factor factory result requires value partitions")
        return self


ResearchResultEnvelope: TypeAlias = Annotated[
    ResearchResultManifest
    | AlphaFactoryResultManifest
    | FactorResearchResultManifest
    | FactorFactoryResultManifest
    | V5CandidateEvidenceResultManifest
    | TradeLevelHistoryResultManifest,
    Field(discriminator="task_type"),
]
RESEARCH_RESULT_ADAPTER = TypeAdapter(ResearchResultEnvelope)


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


class AlphaFactoryWorkerReceipt(StrictModel):
    schema_version: Literal["quant_lab_alpha_factory_receipt.v1"] = ALPHA_FACTORY_RECEIPT_SCHEMA
    task_type: Literal["alpha_factory"] = ALPHA_FACTORY_TASK_TYPE
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
    anti_leakage_status: Literal["PASS"]
    anti_leakage_violation_count: Literal[0] = 0
    error_code: str | None = None
    worker_key_id: str = Field(min_length=1, max_length=120)
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> AlphaFactoryWorkerReceipt:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.worker_id, "worker_id")
        _require_commit(self.worker_commit, "worker_commit")
        _require_sha(self.result_manifest_sha256, "result_manifest_sha256")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        return self


class FactorResearchWorkerReceipt(StrictModel):
    schema_version: Literal["quant_lab_factor_research_receipt.v1"] = FACTOR_RESEARCH_RECEIPT_SCHEMA
    task_type: Literal["factor_research"] = FACTOR_RESEARCH_TASK_TYPE
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
    anti_leakage_status: Literal["PASS"]
    anti_leakage_violation_count: Literal[0] = 0
    error_code: str | None = None
    worker_key_id: str = Field(min_length=1, max_length=120)
    research_only: Literal[True] = True
    live_order_effect: Literal["none"] = "none"
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> FactorResearchWorkerReceipt:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.worker_id, "worker_id")
        _require_commit(self.worker_commit, "worker_commit")
        _require_sha(self.result_manifest_sha256, "result_manifest_sha256")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        return self


class FactorFactoryWorkerReceipt(StrictModel):
    schema_version: Literal["quant_lab_factor_factory_receipt.v1"] = FACTOR_FACTORY_RECEIPT_SCHEMA
    task_type: Literal["factor_factory"] = FACTOR_FACTORY_TASK_TYPE
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
    anti_leakage_status: Literal["PASS"]
    anti_leakage_violation_count: Literal[0] = 0
    error_code: str | None = None
    worker_key_id: str = Field(min_length=1, max_length=120)
    diagnostic_only: Literal[True] = True
    research_only: Literal[True] = True
    live_order_effect: Literal["none_read_only_research"] = "none_read_only_research"
    max_live_notional_usdt: Literal[0] = 0
    signature: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_receipt(self) -> FactorFactoryWorkerReceipt:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.worker_id, "worker_id")
        _require_commit(self.worker_commit, "worker_commit")
        _require_sha(self.result_manifest_sha256, "result_manifest_sha256")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.completed_at, "completed_at")
        _require_identifier(self.worker_key_id, "worker_key_id")
        if self.completed_at < self.claimed_at:
            raise ValueError("factor factory receipt completed_at precedes claimed_at")
        return self


ResearchReceiptEnvelope: TypeAlias = Annotated[
    ResearchWorkerReceipt
    | AlphaFactoryWorkerReceipt
    | FactorResearchWorkerReceipt
    | FactorFactoryWorkerReceipt
    | V5CandidateEvidenceWorkerReceipt
    | TradeLevelHistoryWorkerReceipt,
    Field(discriminator="schema_version"),
]
RESEARCH_RECEIPT_ADAPTER = TypeAdapter(ResearchReceiptEnvelope)


class ResearchValidationEvent(StrictModel):
    schema_version: Literal["quant_lab_research_validation.v1"] = RESEARCH_VALIDATION_SCHEMA
    task_id: str
    stage: Literal["nas", "cloud"]
    check_name: str
    status: Literal["PASS", "FAIL", "RETRY"]
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
    task_type: Literal[
        "entry_quality_history",
        "alpha_factory",
        "factor_research",
        "factor_factory",
        "v5_candidate_evidence",
        "trade_level_history",
    ] = ENTRY_QUALITY_HISTORY_TASK_TYPE
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
    output_bytes: int = Field(default=0, ge=0)
    peak_rss_bytes: int = Field(default=0, ge=0)
    compute_duration_seconds: float = Field(default=0, ge=0)
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


class ResearchTaskLease(StrictModel):
    schema_version: Literal["quant_lab_research_lease.v1"] = RESEARCH_LEASE_SCHEMA
    task_id: str = Field(min_length=1, max_length=180)
    snapshot_id: str = Field(min_length=1, max_length=180)
    task_type: Literal[
        "entry_quality_history",
        "alpha_factory",
        "factor_research",
        "factor_factory",
        "v5_candidate_evidence",
        "trade_level_history",
    ] = ENTRY_QUALITY_HISTORY_TASK_TYPE
    worker_id: str = Field(min_length=1, max_length=180)
    claimed_at: datetime
    heartbeat_at: datetime
    lease_expires_at: datetime
    sequence: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_lease(self) -> ResearchTaskLease:
        _require_identifier(self.task_id, "task_id")
        _require_identifier(self.snapshot_id, "snapshot_id")
        _require_identifier(self.worker_id, "worker_id")
        _require_utc(self.claimed_at, "claimed_at")
        _require_utc(self.heartbeat_at, "heartbeat_at")
        _require_utc(self.lease_expires_at, "lease_expires_at")
        if self.lease_expires_at <= self.heartbeat_at:
            raise ValueError("lease_expires_at must be after heartbeat_at")
        return self


def _require_identifier(value: str, field_name: str) -> None:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-"
    if not value or len(value) > 180 or any(character not in allowed for character in value):
        raise ValueError(f"unsafe {field_name}")


def _require_unique_identifiers(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")
    for value in values:
        _require_identifier(value, field_name)


def _validate_previous_generation(
    value: FactorFactorySnapshotManifest | FactorFactoryTask | FactorFactoryResultManifest,
) -> None:
    generation_id = value.previous_generation_id
    generation_digest = value.previous_generation_digest
    if (generation_id is None) != (generation_digest is None):
        raise ValueError("previous factor generation binding must be complete")
    if generation_id is not None:
        _require_identifier(generation_id, "previous_generation_id")
        _require_sha(generation_digest, "previous_generation_digest")


def _validate_snapshot_files(
    datasets: list[str],
    files: list[ResearchDatasetReference],
    total_input_bytes: int,
    total_input_rows: int,
) -> None:
    if not datasets or len(datasets) != len(set(datasets)):
        raise ValueError("snapshot datasets must be non-empty and unique")
    if total_input_bytes != sum(item.size_bytes for item in files):
        raise ValueError("snapshot total_input_bytes mismatch")
    if total_input_rows != sum(item.row_count for item in files):
        raise ValueError("snapshot total_input_rows mismatch")
    paths = [item.relative_path for item in files]
    if len(paths) != len(set(paths)):
        raise ValueError("snapshot relative paths must be unique")
    if any(item.dataset_name not in datasets for item in files):
        raise ValueError("snapshot file references undeclared dataset")


def _validate_optional_alpha_factor_generation(value: BaseModel) -> None:
    field_names = (
        "factor_generation_id",
        "factor_generation_digest",
        "factor_generation_as_of_date",
        "factor_generation_published_at",
        "hypothesis_registry_digest",
        "trial_ledger_digest",
        "factor_generation_fresh",
        "factor_generation_hypothesis_ids",
    )
    observed = {field_name: getattr(value, field_name) for field_name in field_names}
    if all(item is None for item in observed.values()):
        return
    missing = [field_name for field_name, item in observed.items() if item is None]
    if missing:
        raise ValueError("alpha factor generation binding must be complete: " + ",".join(missing))
    _require_identifier(str(observed["factor_generation_id"]), "factor_generation_id")
    _require_sha(str(observed["factor_generation_digest"]), "factor_generation_digest")
    _require_sha(str(observed["hypothesis_registry_digest"]), "hypothesis_registry_digest")
    _require_sha(str(observed["trial_ledger_digest"]), "trial_ledger_digest")
    _require_utc(
        observed["factor_generation_published_at"],
        "factor_generation_published_at",
    )
    hypothesis_ids = observed["factor_generation_hypothesis_ids"]
    if not hypothesis_ids:
        raise ValueError("factor_generation_hypothesis_ids must not be empty")
    _require_unique_identifiers(hypothesis_ids, "factor_generation_hypothesis_ids")


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
