import subprocess
from datetime import UTC, datetime

from quant_lab.strategy_telemetry.config import V5TelemetryRemoteConfig
from quant_lab.strategy_telemetry.models import PullResult


def build_rsync_command(config: V5TelemetryRemoteConfig) -> list[str]:
    ssh_parts = [
        "ssh",
        "-i",
        str(config.ssh_identity_file),
        "-p",
        str(config.remote_port),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
    ]
    if config.known_hosts_file is not None:
        ssh_parts.extend(["-o", f"UserKnownHostsFile={config.known_hosts_file}"])

    remote = f"{config.remote_user}@{config.remote_host}:{config.remote_bundle_dir}/"
    return [
        "rsync",
        "-av",
        "--ignore-existing",
        "--partial",
        "--protect-args",
        "--prune-empty-dirs",
        f"--include={config.filename_glob}",
        "--exclude=.*",
        "--exclude=.env",
        "--exclude=*",
        "-e",
        " ".join(ssh_parts),
        remote,
        f"{config.local_inbox_dir}/",
    ]


def summarize_command(command: list[str]) -> list[str]:
    return [part if "\n" not in part else part.replace("\n", " ") for part in command]


class RemoteBundlePuller:
    def build_rsync_command(self, config: V5TelemetryRemoteConfig) -> list[str]:
        return build_rsync_command(config)

    def pull_bundles(self, config: V5TelemetryRemoteConfig) -> PullResult:
        started_at = datetime.now(UTC)
        command = self.build_rsync_command(config)
        warnings: list[str] = []
        pulled_files: list[str] = []
        skipped_files: list[str] = []

        if config.dry_run:
            finished_at = datetime.now(UTC)
            return PullResult(
                strategy=config.strategy,
                remote_host=config.remote_host,
                remote_bundle_dir=str(config.remote_bundle_dir),
                local_inbox_dir=str(config.local_inbox_dir),
                pulled_files=[],
                skipped_files=[],
                command_summary=summarize_command(command),
                started_at=started_at,
                finished_at=finished_at,
                dry_run=True,
                warnings=["dry_run: rsync was not executed"],
            )

        config.local_inbox_dir.mkdir(parents=True, exist_ok=True)
        config.require_identity_file()
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        stderr_lines = [line.strip() for line in completed.stderr.splitlines() if line.strip()]
        for line in stdout_lines:
            if line.endswith(".tar.gz") or line.endswith(".tgz"):
                pulled_files.append(line)
        if completed.returncode != 0:
            warnings.append(f"rsync exited with code {completed.returncode}")
            warnings.extend(stderr_lines[:20])

        finished_at = datetime.now(UTC)
        return PullResult(
            strategy=config.strategy,
            remote_host=config.remote_host,
            remote_bundle_dir=str(config.remote_bundle_dir),
            local_inbox_dir=str(config.local_inbox_dir),
            pulled_files=pulled_files,
            skipped_files=skipped_files,
            command_summary=summarize_command(command),
            started_at=started_at,
            finished_at=finished_at,
            dry_run=False,
            warnings=warnings,
        )


def pull_bundles(config: V5TelemetryRemoteConfig) -> PullResult:
    return RemoteBundlePuller().pull_bundles(config)
