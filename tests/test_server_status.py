from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.decision import host_metrics
from quant_lab.decision.server_status import host_view, server_status
from quant_lab.decision.storage import atomic_json

NOW = datetime(2026, 9, 5, 10, 0, tzinfo=UTC)
GIB = 1024**3


def snapshot(host="nas"):
    ids = (
        ["analysis"]
        if host == "nas"
        else ["api", "market", "https", "decision", "backfill", "compaction"]
    )
    return {
        "schema_version": "qlab.host_status.v1",
        "host": host,
        "observed_at": NOW.isoformat(),
        "resources": {
            "cpu_percent": 20,
            "cpu_cores": 6,
            "load_1m": 0.5,
            "memory_total_bytes": 16 * GIB,
            "memory_available_bytes": 10 * GIB,
            "swap_total_bytes": 8 * GIB,
            "swap_used_bytes": 0,
            "uptime_seconds": 10000,
            "disks": [
                {"id": disk, "total_bytes": 1000 * GIB, "free_bytes": 500 * GIB}
                for disk in (["ssd", "hdd"] if host == "nas" else ["system"])
            ],
        },
        "services": [
            {
                "id": key,
                "state": "scheduled",
                "restart_count": 0,
                "interval_seconds": 600,
                "last_finished_at": NOW.isoformat(),
            }
            for key in ids
        ],
        "containers": [{"name": "example", "state": "running", "restart_count": 12, "health": None}]
        if host == "nas"
        else [],
        "errors": [],
    }


def test_public_status_projects_only_safe_fields_and_only_allows_get(tmp_path, monkeypatch):
    data = snapshot()
    # Even if private fields accidentally reach the snapshot, the public API drops them.
    data["credentials"] = {"password": "never-publish-this"}
    data["containers"][0]["env"] = ["SECRET=never-publish-this"]
    data["resources"]["disks"][0]["mountpoint"] = "/private-business-path"
    atomic_json(tmp_path / "inbox/server-status-nas.json", data)
    monkeypatch.setenv("QUANT_LAB_SERVER_STATUS_ROOT", str(tmp_path))
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "test-private-token")
    with TestClient(create_app()) as client:
        response = client.get("/v1/server-status")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert "never-publish-this" not in response.text
        assert "private-business-path" not in response.text
        assert "credentials" not in response.text
        assert client.post("/v1/server-status").status_code == 401
        assert client.get("/v1/server-status/raw").status_code == 401
        assert client.get("/v1/catalog/datasets").status_code == 401
        monkeypatch.setenv("QUANT_LAB_ALLOWED_CLIENT_IPS", "192.0.2.1")
        assert client.get("/v1/server-status").status_code == 403


def test_missing_and_stale_are_not_healthy_and_do_not_break_other_host(tmp_path):
    atomic_json(tmp_path / "inbox/server-status-nas.json", snapshot())
    value = server_status(tmp_path, now=NOW + timedelta(seconds=181))
    assert value["hosts"][0]["state"] == "unknown"
    assert value["hosts"][0]["resources"] is None
    assert value["hosts"][1]["state"] == "stale"
    assert value["hosts"][1]["age_seconds"] == 181
    assert "SNAPSHOT_STALE" in value["hosts"][1]["warnings"]
    # An accumulated restart count alone is not an active restart loop.
    fresh = server_status(tmp_path, now=NOW)["hosts"][1]
    assert fresh["state"] == "ok"


@pytest.mark.parametrize("bad", ["future", "nan", "oversize", "partial", "host", "capacity"])
def test_invalid_samples_are_explicit(tmp_path, bad):
    data = snapshot()
    if bad == "future":
        data["observed_at"] = (NOW + timedelta(seconds=1)).isoformat()
    elif bad == "nan":
        data["resources"]["cpu_percent"] = "nan"
    elif bad == "partial":
        data["services"] = []
    elif bad == "host":
        data["host"] = "qyun2"
    elif bad == "capacity":
        data["resources"]["disks"][0]["free_bytes"] = 2000 * GIB
    path = tmp_path / "sample.json"
    atomic_json(path, data)
    if bad == "oversize":
        path.write_text(" " * (64 * 1024 + 1))
    view = host_view(path, "nas", NOW)
    assert view["state"] == "unknown"
    assert view["warnings"] == ["SNAPSHOT_INVALID"]
    assert view["resources"] is None


def test_resource_pressure_and_restarts_are_visible(tmp_path):
    data = snapshot()
    data["resources"]["memory_available_bytes"] = 5 * GIB
    data["resources"]["cpu_percent"] = 95
    data["resources"]["disks"][0]["free_bytes"] = 50 * GIB
    data["containers"][0]["state"] = "restarting"
    atomic_json(tmp_path / "sample.json", data)
    view = host_view(tmp_path / "sample.json", "nas", NOW)
    assert view["state"] == "warning"
    assert set(view["warnings"]) == {
        "CPU_HIGH",
        "NAS_MEMORY_RESERVE_LOW",
        "DISK_LOW",
        "CONTAINER_ATTENTION",
    }


def test_failed_collection_is_not_a_zero_or_empty_success(tmp_path, monkeypatch):
    def unavailable(*args):
        raise OSError("private diagnostics must not reach the page")

    monkeypatch.setattr(host_metrics, "host_resources", unavailable)
    monkeypatch.setattr(host_metrics, "docker_containers", unavailable)
    data = host_metrics.collect("nas")
    data["observed_at"] = NOW.isoformat()
    atomic_json(tmp_path / "sample.json", data)
    view = host_view(tmp_path / "sample.json", "nas", NOW)
    assert view["state"] == "warning"
    assert view["resources"] is None
    assert view["containers"] == []
    assert "CONTAINER_COLLECTION_FAILED" in view["warnings"]


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("recent", "scheduled"),
        ("failed", "failed"),
        ("disabled", "stopped"),
        ("old", "overdue"),
        ("never", "unknown"),
        ("restarting", "restarting"),
    ],
)
def test_systemd_oneshots_distinguish_idle_failure_and_overdue(mode, expected):
    units = {
        "quant-lab-decision.service": {
            "LoadState": "loaded",
            "ActiveState": "inactive",
            "Result": "success",
            "ExecMainStatus": "0",
            "ExecMainExitTimestampMonotonic": str(9900 * 1_000_000),
        },
        "quant-lab-decision.timer": {"ActiveState": "active"},
    }
    unit = units["quant-lab-decision.service"]
    if mode == "failed":
        unit["Result"] = "exit-code"
    elif mode == "disabled":
        units["quant-lab-decision.timer"]["ActiveState"] = "inactive"
    elif mode == "old":
        unit["ExecMainExitTimestampMonotonic"] = str(1000 * 1_000_000)
    elif mode == "never":
        unit["ExecMainExitTimestampMonotonic"] = "0"
    elif mode == "restarting":
        unit.update(ActiveState="activating", SubState="auto-restart")
    row = next(r for r in host_metrics.cloud_services(units, NOW, 10000) if r["id"] == "decision")
    assert row["state"] == expected


def test_nas_idle_running_failed_and_missing_schedule(tmp_path):
    cron = tmp_path / "cron"
    cron.write_text(f"4-59/10 * * * * root /bin/bash {tmp_path / 'run.sh'}\n")
    path = tmp_path / "archive/worker-status.json"
    data = {"status": "UPLOADED_AWAITING_CLOUD_ACCEPTANCE", "at": NOW.isoformat()}
    atomic_json(path, data)
    assert host_metrics.nas_worker(tmp_path, [], NOW, cron)["state"] == "scheduled"
    overdue = host_metrics.nas_worker(tmp_path, [], NOW + timedelta(minutes=21), cron)
    assert overdue["state"] == "overdue"
    containers = [{"name": "quant-decision-job", "state": "running"}]
    assert host_metrics.nas_worker(tmp_path, containers, NOW, cron)["state"] == "running"
    data["status"] = "FAILED"
    atomic_json(path, data)
    assert host_metrics.nas_worker(tmp_path, [], NOW, cron)["state"] == "failed"
    cron.write_text("")
    assert host_metrics.nas_worker(tmp_path, [], NOW, cron)["state"] == "stopped"
