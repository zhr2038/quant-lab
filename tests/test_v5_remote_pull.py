from pathlib import Path

from quant_lab.strategy_telemetry.config import V5TelemetryRemoteConfig
from quant_lab.strategy_telemetry.remote_pull import RemoteBundlePuller, build_rsync_command


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
