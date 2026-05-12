from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_lab.contracts.models import require_utc


class TelemetryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BundleLimits(TelemetryModel):
    max_bundle_size_mb: int = Field(default=512, gt=0)
    max_extracted_size_mb: int = Field(default=2048, gt=0)
    max_file_count: int = Field(default=5000, gt=0)


class PullResult(TelemetryModel):
    strategy: str
    remote_host: str
    remote_bundle_dir: str
    local_inbox_dir: str
    pulled_files: list[str] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    command_summary: list[str]
    started_at: datetime
    finished_at: datetime
    dry_run: bool
    warnings: list[str] = Field(default_factory=list)

    @field_validator("started_at", "finished_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class V5BundleValidationResult(TelemetryModel):
    path: str
    sha256: str | None = None
    size_bytes: int = 0
    valid: bool
    rejected: bool
    reasons: list[str] = Field(default_factory=list)
    warning_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    file_count: int = 0
    total_uncompressed_size_bytes: int = 0
    detected_files: list[str] = Field(default_factory=list)
    started_at: datetime
    finished_at: datetime

    @field_validator("started_at", "finished_at")
    @classmethod
    def timestamps_are_utc(cls, value: datetime) -> datetime:
        return require_utc(value)


class V5BundleInspection(TelemetryModel):
    path: str
    sha256: str
    bundle_name: str
    bundle_ts: datetime | None = None
    detected_files: list[str] = Field(default_factory=list)
    file_count: int
    total_uncompressed_size_bytes: int

    @field_validator("bundle_ts")
    @classmethod
    def timestamp_is_utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None


class SafeExtractResult(TelemetryModel):
    bundle_path: str
    target_dir: str
    extracted_files: list[str] = Field(default_factory=list)
    file_count: int
    total_uncompressed_size_bytes: int
    warnings: list[str] = Field(default_factory=list)


class SecretFinding(TelemetryModel):
    source_path: str
    pattern: str
    severity: str
    line_number: int | None = None


class SecretScanResult(TelemetryModel):
    scanned_files: int
    findings: list[SecretFinding] = Field(default_factory=list)
    high_severity_count: int = 0
    medium_severity_count: int = 0
    low_severity_count: int = 0
    redaction_required: bool = False
    warnings: list[str] = Field(default_factory=list)


class RedactionResult(TelemetryModel):
    source_dir: str
    redacted_dir: str
    copied_files: list[str] = Field(default_factory=list)
    redacted_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class V5BundleIngestResult(TelemetryModel):
    strategy: str
    bundle_path: str
    bundle_sha256: str
    bundle_name: str
    bundle_ts: datetime | None = None
    skipped: bool = False
    validation: V5BundleValidationResult
    secret_scan: SecretScanResult
    restricted_archive_path: str
    redacted_archive_path: str
    bronze_rows: dict[str, int] = Field(default_factory=dict)
    silver_rows: dict[str, int] = Field(default_factory=dict)
    gold_rows: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("bundle_ts")
    @classmethod
    def timestamp_is_utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None


class V5InboxIngestResult(TelemetryModel):
    strategy: str
    inbox_dir: str
    processed: list[V5BundleIngestResult] = Field(default_factory=list)
    skipped_files: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class V5TelemetryAnalysisResult(TelemetryModel):
    strategy: str = "v5"
    date: str
    status: str
    latest_bundle_ts: datetime | None = None
    latest_bundle_sha256: str | None = None
    run_count_72h: int = 0
    decision_audit_count_24h: int = 0
    trade_count_24h: int = 0
    trade_count_72h: int = 0
    roundtrip_count_72h: int = 0
    open_position_count: int = 0
    dust_residual_position_count: int = 0
    kill_switch_enabled: bool | None = None
    reconcile_ok: bool | None = None
    ledger_ok: bool | None = None
    auto_risk_level: str | None = None
    high_issue_count: int = 0
    medium_issue_count: int = 0
    config_not_consumed_count: int = 0
    config_not_consumed_count_unknown: bool = False
    config_not_consumed_top_keys: list[str] = Field(default_factory=list)
    high_score_blocked_count: int = 0
    high_score_blocked_matured_count: int = 0
    high_score_blocked_profitable_count: int = 0
    skipped_candidate_matured_count: int = 0
    router_reason_top: list[dict[str, Any]] = Field(default_factory=list)
    quant_lab_mode: str | None = None
    permission_gate_enforced: bool | None = None
    quant_lab_usage_count: int = 0
    quant_lab_cost_usage_count: int = 0
    quant_lab_fallback_count: int = 0
    quant_lab_actual_violation_count: int = 0
    quant_lab_hypothetical_violation_count: int = 0
    warnings: list[str] = Field(default_factory=list)
    critical_reasons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)

    @field_validator("latest_bundle_ts")
    @classmethod
    def timestamp_is_utc(cls, value: datetime | None) -> datetime | None:
        return require_utc(value) if value is not None else None


def utc_now() -> datetime:
    return datetime.now(UTC)


def path_string(path: str | Path) -> str:
    return str(Path(path))
