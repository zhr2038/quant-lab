"""Bounded, allowlisted public status. Never execute host commands in HTTP requests."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from quant_lab.decision.storage import read_json

STALE_SECONDS = 180
ROOT = Path("/var/lib/quant-lab/decision")


class PublicModel(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)


class Disk(PublicModel):
    id: Literal["system", "ssd", "hdd"]
    total_bytes: int = Field(gt=0)
    free_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def capacity(self):
        if self.free_bytes > self.total_bytes:
            raise ValueError("invalid disk capacity")
        return self


class Resources(PublicModel):
    cpu_percent: float | None = Field(ge=0, le=100)
    cpu_cores: int = Field(gt=0, le=4096)
    load_1m: float = Field(ge=0)
    memory_total_bytes: int = Field(gt=0)
    memory_available_bytes: int = Field(ge=0)
    swap_total_bytes: int = Field(ge=0)
    swap_used_bytes: int = Field(ge=0)
    uptime_seconds: float = Field(ge=0)
    disks: list[Disk] = Field(min_length=1, max_length=2)

    @model_validator(mode="after")
    def capacity(self):
        if self.memory_available_bytes > self.memory_total_bytes:
            raise ValueError("invalid memory capacity")
        if self.swap_used_bytes > self.swap_total_bytes:
            raise ValueError("invalid swap capacity")
        return self


class Service(PublicModel):
    id: Literal["api", "market", "https", "decision", "backfill", "compaction", "analysis"]
    state: Literal[
        "running", "scheduled", "failed", "stopped", "restarting", "overdue", "missing", "unknown"
    ]
    restart_count: int = Field(ge=0)
    interval_seconds: int = Field(ge=0, le=86400)
    last_finished_at: AwareDatetime | None = None
    runtime_seconds: float | None = Field(default=None, ge=0)
    peak_rss_mib: float | None = Field(default=None, ge=0)


class Container(PublicModel):
    name: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
    state: Literal["created", "running", "paused", "restarting", "removing", "exited", "dead"]
    restart_count: int = Field(ge=0)
    health: Literal["healthy", "unhealthy", "starting"] | None = None


class Snapshot(PublicModel):
    schema_version: Literal["qlab.host_status.v1"]
    host: Literal["qyun2", "nas"]
    observed_at: AwareDatetime
    resources: Resources | None
    services: list[Service] = Field(max_length=12)
    containers: list[Container] = Field(max_length=128)
    errors: list[
        Literal[
            "RESOURCE_COLLECTION_FAILED",
            "SERVICE_COLLECTION_FAILED",
            "CONTAINER_COLLECTION_FAILED",
            "WORKER_COLLECTION_FAILED",
        ]
    ]

    @model_validator(mode="after")
    def completeness(self):
        if self.resources is None and "RESOURCE_COLLECTION_FAILED" not in self.errors:
            raise ValueError("missing resource outcome")
        expected = (
            {"analysis"}
            if self.host == "nas"
            else {"api", "market", "https", "decision", "backfill", "compaction"}
        )
        if not self.errors and {s.id for s in self.services} != expected:
            raise ValueError("missing service outcome")
        if self.resources:
            disks = [disk.id for disk in self.resources.disks]
            if sorted(disks) != (["hdd", "ssd"] if self.host == "nas" else ["system"]):
                raise ValueError("missing disk outcome")
        return self


def host_view(path: Path, host: str, now: datetime) -> dict:
    missing = {
        "host": host,
        "state": "unknown",
        "observed_at": None,
        "age_seconds": None,
        "resources": None,
        "services": [],
        "containers": [],
        "warnings": [],
    }
    try:
        snapshot = Snapshot.model_validate(read_json(path, max_bytes=64 * 1024))
        if snapshot.host != host or snapshot.observed_at > now:
            raise ValueError("invalid snapshot origin or time")
        if any(s.last_finished_at and s.last_finished_at > now for s in snapshot.services):
            raise ValueError("service time is in the future")
    except FileNotFoundError:
        return {**missing, "warnings": ["SNAPSHOT_MISSING"]}
    except (OSError, ValueError, TypeError):
        return {**missing, "warnings": ["SNAPSHOT_INVALID"]}
    age = max(0, int((now - snapshot.observed_at).total_seconds()))
    value = snapshot.model_dump(mode="json", exclude={"errors", "schema_version"})
    warnings = list(snapshot.errors)
    metrics = snapshot.resources
    if metrics:
        if metrics.cpu_percent is None:
            warnings.append("CPU_UNAVAILABLE")
        if metrics.cpu_percent is not None and metrics.cpu_percent >= 90:
            warnings.append("CPU_HIGH")
        if metrics.memory_available_bytes / metrics.memory_total_bytes < 0.1:
            warnings.append("MEMORY_LOW")
        if host == "nas" and metrics.memory_available_bytes < 6 * 1024**3:
            warnings.append("NAS_MEMORY_RESERVE_LOW")
        if any(d.free_bytes / d.total_bytes < 0.1 for d in metrics.disks):
            warnings.append("DISK_LOW")
    if any(s.state not in {"running", "scheduled"} for s in snapshot.services):
        warnings.append("SERVICE_ATTENTION")
    if any(
        c.state != "running" or c.health in {"unhealthy", "starting"} for c in snapshot.containers
    ):
        warnings.append("CONTAINER_ATTENTION")
    state = "warning" if warnings else "ok"
    if age > STALE_SECONDS:
        state = "stale"
        warnings.insert(0, "SNAPSHOT_STALE")
    return {**value, "state": state, "age_seconds": age, "warnings": warnings}


def server_status(root: Path | None = None, now: datetime | None = None) -> dict:
    root = root or Path(os.environ.get("QUANT_LAB_SERVER_STATUS_ROOT", str(ROOT)))
    now = now or datetime.now(UTC)
    return {
        "viewed_at": now.isoformat(),
        "stale_after_seconds": STALE_SECONDS,
        "sample_interval_seconds": 60,
        "hosts": [
            host_view(root / "status/qyun2.json", "qyun2", now),
            host_view(root / "inbox/server-status-nas.json", "nas", now),
        ],
    }
