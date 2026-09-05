"""Small Linux host sampler. Also copied to NAS and run with system Python.

Uses only the standard library; no lake scans, Docker stats stream or daemon.
The API validates and projects these snapshots before making them public.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

CLOUD_UNITS = (
    ("api", "quant-lab-api", 0),
    ("market", "quant-lab-okx-ws", 0),
    ("https", "caddy", 0),
    ("decision", "quant-lab-decision", 300),
    ("backfill", "quant-lab-okx-rest-backfill", 900),
    ("compaction", "quant-lab-lake-compaction", 3600),
)
SCHEMA = "qlab.host_status.v1"


def command(args):
    return subprocess.check_output(args, stderr=subprocess.PIPE, timeout=8, text=True)


def utc_now():
    return datetime.now(UTC)


def cpu_ticks(proc):
    values = [int(v) for v in (proc / "stat").read_text().splitlines()[0].split()[1:9]]
    if len(values) != 8:
        raise ValueError("CPU counters unavailable")
    return sum(values), values[3] + values[4]


def host_resources(host, proc=Path("/proc")):
    first = cpu_ticks(proc)
    time.sleep(0.2)
    last = cpu_ticks(proc)
    delta = last[0] - first[0]
    cpu = round(100 * (1 - (last[1] - first[1]) / delta), 1) if delta > 0 else None
    memory = {}
    for line in (proc / "meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        memory[key] = int(value.split()[0]) * 1024
    total, available = memory["MemTotal"], memory["MemAvailable"]
    disks = []
    for disk_id, path in (
        (("ssd", "/volume1"), ("hdd", "/volume2")) if host == "nas" else (("system", "/"),)
    ):
        usage = shutil.disk_usage(path)
        disks.append({"id": disk_id, "total_bytes": usage.total, "free_bytes": usage.free})
    return {
        "cpu_percent": cpu,
        "cpu_cores": os.cpu_count(),
        "load_1m": os.getloadavg()[0],
        "memory_total_bytes": total,
        "memory_available_bytes": available,
        "swap_total_bytes": memory["SwapTotal"],
        "swap_used_bytes": memory["SwapTotal"] - memory["SwapFree"],
        "uptime_seconds": float((proc / "uptime").read_text().split()[0]),
        "disks": disks,
    }


def unit_states():
    names = [
        name + suffix
        for _, name, period in CLOUD_UNITS
        for suffix in ((".service", ".timer") if period else (".service",))
    ]
    props = "Id,LoadState,ActiveState,SubState,Result,NRestarts,ExecMainStatus"
    props += ",ExecMainExitTimestampMonotonic"
    output = command(["systemctl", "show", "--no-pager", "--property=" + props, *names])
    units = {}
    for block in output.strip().split("\n\n"):
        fields = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "Id" in fields:
            units[fields["Id"]] = fields
    return units


def cloud_services(units, now, uptime):
    rows = []
    for key, name, period in CLOUD_UNITS:
        unit = units.get(name + ".service", {})
        timer = units.get(name + ".timer", {})
        state = "unknown"
        exited = int(unit.get("ExecMainExitTimestampMonotonic") or 0) / 1_000_000
        last_finished_at = (
            datetime.fromtimestamp(now.timestamp() - uptime + exited, UTC).isoformat()
            if 0 < exited <= uptime
            else None
        )
        if unit.get("LoadState") != "loaded":
            state = "missing"
        elif unit.get("ActiveState") == "failed":
            state = "failed"
        elif unit.get("SubState") == "auto-restart":
            state = "restarting"
        elif unit.get("ActiveState") in {"active", "activating", "reloading"}:
            state = "running"
        elif period and timer.get("ActiveState") == "active":
            if unit.get("Result") != "success" or int(unit.get("ExecMainStatus") or 0):
                state = "failed"
            elif not last_finished_at:
                state = "unknown"
            elif uptime - exited > 2 * period + 180:
                state = "overdue"
            else:
                state = "scheduled"
        else:
            state = "stopped"
        rows.append(
            {
                "id": key,
                "state": state,
                "restart_count": int(unit.get("NRestarts") or 0),
                "interval_seconds": period,
                "last_finished_at": last_finished_at,
            }
        )
    return rows


def docker_containers():
    ids = command(["docker", "ps", "-aq"]).split()
    if len(ids) > 128:
        raise ValueError("Container inventory exceeds budget")
    if not ids:
        return []
    fmt = (
        "[{{json .Name}},{{json .State.Status}},{{json .RestartCount}},"
        "{{if .State.Health}}{{json .State.Health.Status}}{{else}}null{{end}}]"
    )
    rows = []
    for line in command(["docker", "inspect", "--format", fmt, *ids]).splitlines():
        name, state, restarts, health = json.loads(line)
        rows.append(
            {"name": name.lstrip("/"), "state": state, "restart_count": restarts, "health": health}
        )
    return sorted(rows, key=lambda item: item["name"])


def nas_worker(root, containers, now, cron=Path("/etc/cron.d/quant-decision")):
    path = root / "archive/worker-status.json"
    status = {}
    if path.exists():
        if path.stat().st_size > 8192:
            raise ValueError("Worker status exceeds budget")
        status = json.loads(path.read_text())
    completed = status.get("at")
    observed = datetime.fromisoformat(completed) if completed else None
    if observed and observed.tzinfo is None:
        raise ValueError("Worker time must have timezone")
    age = (now - observed).total_seconds() if observed else None
    if age is not None and age < -5:
        raise ValueError("Worker time is in the future")
    enabled = any(
        line.strip() and not line.lstrip().startswith("#") and str(root / "run.sh") in line
        for line in cron.read_text().splitlines()
    )
    running = next((c for c in containers if c["name"] == "quant-decision-job"), None)
    if running:
        state = "running" if running["state"] == "running" else "failed"
    elif not enabled:
        state = "stopped"
    elif status.get("status") == "FAILED":
        state = "failed"
    elif age is None or age > 1200:
        state = "overdue"
    elif status.get("status") in {
        "UPLOADED_AWAITING_CLOUD_ACCEPTANCE",
        "UNCHANGED_INPUT_AND_OBSERVATIONS",
    }:
        state = "scheduled"
    else:
        state = "unknown"
    return {
        "id": "analysis",
        "state": state,
        "interval_seconds": 600,
        "restart_count": 0,
        "last_finished_at": completed,
        "runtime_seconds": status.get("runtime_seconds"),
        "peak_rss_mib": status.get("peak_rss_mib"),
    }


def collect(host):
    data = {
        "schema_version": SCHEMA,
        "host": host,
        "observed_at": utc_now().isoformat(),
        "resources": None,
        "services": [],
        "containers": [],
        "errors": [],
    }
    try:
        data["resources"] = host_resources(host)
    except (OSError, ValueError, KeyError):
        data["errors"].append("RESOURCE_COLLECTION_FAILED")
    now = utc_now()
    if host == "qyun2":
        try:
            uptime = float(Path("/proc/uptime").read_text().split()[0])
            data["services"] = cloud_services(unit_states(), now, uptime)
        except (OSError, ValueError, subprocess.SubprocessError):
            data["errors"].append("SERVICE_COLLECTION_FAILED")
    else:
        try:
            data["containers"] = docker_containers()
        except (OSError, ValueError, subprocess.SubprocessError):
            data["errors"].append("CONTAINER_COLLECTION_FAILED")
        try:
            if "CONTAINER_COLLECTION_FAILED" not in data["errors"]:
                data["services"] = [
                    nas_worker(Path("/volume2/quant-lab/decision"), data["containers"], now)
                ]
        except (OSError, ValueError, TypeError, KeyError):
            data["errors"].append("WORKER_COLLECTION_FAILED")
    data["observed_at"] = utc_now().isoformat()
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("host", choices=("qyun2", "nas"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = collect(args.host)
    temporary = args.output.with_suffix(".part")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temporary.chmod(0o640)
    os.replace(temporary, args.output)
    if value["errors"]:
        raise SystemExit(",".join(value["errors"]))


if __name__ == "__main__":
    main()
