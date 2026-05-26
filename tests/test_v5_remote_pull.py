import subprocess
from pathlib import Path

from quant_lab.strategy_telemetry.config import V5TelemetryRemoteConfig
from quant_lab.strategy_telemetry.remote_pull import (
    RemoteBundlePuller,
    build_remote_bundle_list_command,
    build_rsync_command,
)


def test_remote_pull_builds_safe_rsync_command(tmp_path):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key", encoding="utf-8")
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("qyun.hrhome.top ssh-ed25519 test", encoding="utf-8")
    config = _config(tmp_path, identity, known_hosts)

    command = build_rsync_command(config)

    assert command[0] == "rsync"
    assert "-e" in command
    assert str(identity) in command[command.index("-e") + 1]
    assert "qyun.hrhome.top" in " ".join(command)
    assert any("v5_live_followup_bundle_*.tar.gz" in part for part in command)
    assert "--ignore-existing" in command
    assert "--partial" in command
    assert "--protect-args" in command
    assert "sshpass" not in " ".join(command).lower()
    assert "password" not in " ".join(command).lower()


def test_remote_pull_dry_run_does_not_execute(tmp_path, monkeypatch):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key", encoding="utf-8")
    config = _config(tmp_path, identity, None).model_copy(update={"dry_run": True})
    executed = False

    def fake_run(*args, **kwargs):
        nonlocal executed
        executed = True
        raise AssertionError("subprocess must not run in dry_run")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = RemoteBundlePuller().pull_bundles(config)

    assert executed is False
    assert result.dry_run is True
    assert result.remote_host == "qyun.hrhome.top"
    assert result.command_summary
    assert result.warnings == ["dry_run: rsync was not executed"]


def test_remote_pull_can_limit_to_newest_stable_files(tmp_path, monkeypatch):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key", encoding="utf-8")
    config = _config(tmp_path, identity, None)
    (config.local_inbox_dir).mkdir(parents=True)
    (config.local_inbox_dir / "v5_live_followup_bundle_20260510T030000Z.tar.gz").write_text(
        "already local",
        encoding="utf-8",
    )
    calls = []

    class Completed:
        def __init__(self, stdout: str = "", returncode: int = 0):
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "ssh":
            assert "head -n 3" in command[-1]
            assert kwargs["timeout"] == config.remote_list_timeout_seconds
            return Completed(
                "\n".join(
                    [
                        "v5_live_followup_bundle_20260510T040000Z.tar.gz",
                        "v5_live_followup_bundle_20260510T030000Z.tar.gz",
                    ]
                )
            )
        assert command[0] == "rsync"
        assert "--files-from=-" in command
        assert kwargs["input"] == "v5_live_followup_bundle_20260510T040000Z.tar.gz\n"
        assert kwargs["timeout"] == config.rsync_timeout_seconds
        return Completed("v5_live_followup_bundle_20260510T040000Z.tar.gz\n")

    monkeypatch.setattr("subprocess.run", fake_run)

    result = RemoteBundlePuller().pull_bundles(config, max_files=3)

    assert len(calls) == 2
    assert result.pulled_files == ["v5_live_followup_bundle_20260510T040000Z.tar.gz"]
    assert result.skipped_files == ["v5_live_followup_bundle_20260510T030000Z.tar.gz"]
    assert any("head -n 3" in part for part in result.command_summary)


def test_remote_pull_remote_list_timeout_returns_warning(tmp_path, monkeypatch):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key", encoding="utf-8")
    config = _config(tmp_path, identity, None).model_copy(
        update={"remote_list_timeout_seconds": 7}
    )

    def fake_run(command, **kwargs):
        assert command[0] == "ssh"
        assert kwargs["timeout"] == 7
        raise subprocess.TimeoutExpired(command, timeout=7)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = RemoteBundlePuller().pull_bundles(config, max_files=2)

    assert result.pulled_files == []
    assert result.warnings == ["remote bundle list timed out after 7s"]


def test_remote_pull_rsync_timeout_returns_warning(tmp_path, monkeypatch):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key", encoding="utf-8")
    config = _config(tmp_path, identity, None).model_copy(
        update={"rsync_timeout_seconds": 11}
    )

    def fake_run(command, **kwargs):
        assert command[0] == "rsync"
        assert kwargs["timeout"] == 11
        raise subprocess.TimeoutExpired(command, timeout=11)

    monkeypatch.setattr("subprocess.run", fake_run)

    result = RemoteBundlePuller().pull_bundles(config)

    assert result.pulled_files == []
    assert result.warnings == ["rsync timed out after 11s"]


def test_remote_bundle_list_command_honors_stable_age(tmp_path):
    identity = tmp_path / "id_ed25519"
    identity.write_text("not-a-real-key", encoding="utf-8")
    config = _config(tmp_path, identity, None)

    command = build_remote_bundle_list_command(config, max_files=2)

    rendered = " ".join(command)
    assert command[0] == "ssh"
    assert "v5readonly@qyun.hrhome.top" in command
    assert "find" in rendered
    assert "age=\"60\"" in rendered
    assert "head -n 2" in rendered


def _config(tmp_path: Path, identity: Path, known_hosts: Path | None) -> V5TelemetryRemoteConfig:
    return V5TelemetryRemoteConfig(
        remote_host="qyun.hrhome.top",
        remote_user="v5readonly",
        remote_bundle_dir=Path("/var/lib/v5/exports/bundles"),
        ssh_identity_file=identity,
        known_hosts_file=known_hosts,
        local_inbox_dir=tmp_path / "inbox",
        restricted_archive_dir=tmp_path / "restricted",
        redacted_archive_dir=tmp_path / "redacted",
        lake_root=tmp_path / "lake",
    )
