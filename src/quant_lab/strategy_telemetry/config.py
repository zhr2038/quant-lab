from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FORBIDDEN_CONFIG_FRAGMENTS = (
    "trade",
    "withdraw",
    "transfer",
    "okx trade",
    "okx withdraw",
)


class V5TelemetryRemoteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: str = "v5"
    remote_host: str
    remote_user: str
    remote_port: int = Field(default=22, gt=0, le=65535)
    remote_bundle_dir: Path
    filename_glob: str = "v5_live_followup_bundle_*.tar.gz"
    ssh_identity_file: Path
    known_hosts_file: Path | None = None
    local_inbox_dir: Path
    restricted_archive_dir: Path
    redacted_archive_dir: Path
    lake_root: Path
    max_bundle_size_mb: int = Field(default=512, gt=0)
    max_extracted_size_mb: int = Field(default=2048, gt=0)
    max_file_count: int = Field(default=5000, gt=0)
    min_stable_age_seconds: int = Field(default=60, ge=0)
    remote_list_timeout_seconds: int = Field(default=30, gt=0)
    rsync_timeout_seconds: int = Field(default=300, gt=0)
    keep_remote_files: bool = True
    dry_run: bool = False

    @field_validator("remote_host", "remote_user", "filename_glob")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("required remote telemetry config value is empty")
        return value

    @model_validator(mode="after")
    def validate_safe_config(self) -> "V5TelemetryRemoteConfig":
        if not self.keep_remote_files:
            raise ValueError("keep_remote_files must remain true for v0.1")
        if not self.filename_glob.endswith(".tar.gz"):
            raise ValueError("filename_glob must target .tar.gz bundles")

        rendered = " ".join(
            [
                self.strategy,
                self.remote_host,
                self.remote_user,
                str(self.remote_bundle_dir),
                self.filename_glob,
            ]
        ).lower()
        for fragment in FORBIDDEN_CONFIG_FRAGMENTS:
            if fragment in rendered:
                raise ValueError(f"forbidden remote telemetry config fragment: {fragment}")
        return self

    @property
    def bundle_limits(self):
        from quant_lab.strategy_telemetry.models import BundleLimits

        return BundleLimits(
            max_bundle_size_mb=self.max_bundle_size_mb,
            max_extracted_size_mb=self.max_extracted_size_mb,
            max_file_count=self.max_file_count,
        )

    def require_identity_file(self) -> None:
        if not self.ssh_identity_file.exists():
            raise FileNotFoundError(f"SSH identity file does not exist: {self.ssh_identity_file}")
        if self.known_hosts_file is not None and not self.known_hosts_file.exists():
            raise FileNotFoundError(f"known_hosts file does not exist: {self.known_hosts_file}")


def load_v5_telemetry_remote_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> V5TelemetryRemoteConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"V5 telemetry config must be a YAML mapping: {config_path}")
    merged = {
        **raw,
        **{key: value for key, value in (overrides or {}).items() if value is not None},
    }
    return V5TelemetryRemoteConfig.model_validate(merged)
