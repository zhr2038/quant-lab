import shlex
import subprocess
from datetime import UTC, datetime

from quant_lab.strategy_telemetry.config import V5TelemetryRemoteConfig
from quant_lab.strategy_telemetry.models import PullResult


def _ssh_parts(config: V5TelemetryRemoteConfig) -> list[str]:
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
    return ssh_parts


def _ssh_command(config: V5TelemetryRemoteConfig) -> str:
    return " ".join(shlex.quote(part) for part in _ssh_parts(config))


def build_remote_bundle_list_command(
    config: V5TelemetryRemoteConfig,
    *,
    max_files: int,
) -> list[str]:
    remote_dir = shlex.quote(str(config.remote_bundle_dir))
    filename_glob = shlex.quote(config.filename_glob)
    max_files = max(int(max_files), 1)
    min_age = max(int(config.min_stable_age_seconds), 0)
    script = (
        "now=$(date +%s); "
        f"find {remote_dir} -maxdepth 1 -type f -name {filename_glob} "
        "-printf '%T@ %f\\n' 2>/dev/null | "
        "sort -rn | "
        f"awk -v now=\"$now\" -v age=\"{min_age}\" "
        "'($1 <= now-age) {$1=\"\"; sub(/^ /, \"\"); print}' | "
        f"head -n {max_files}"
    )
    return [*_ssh_parts(config), f"{config.remote_user}@{config.remote_host}", script]


def build_rsync_command(config: V5TelemetryRemoteConfig) -> list[str]:
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
        _ssh_command(config),
        remote,
        f"{config.local_inbox_dir}/",
    ]


def build_limited_rsync_command(config: V5TelemetryRemoteConfig) -> list[str]:
    remote = f"{config.remote_user}@{config.remote_host}:{config.remote_bundle_dir}/"
    return [
        "rsync",
        "-av",
        "--ignore-existing",
        "--partial",
        "--protect-args",
        "--files-from=-",
        "-e",
        _ssh_command(config),
        remote,
        f"{config.local_inbox_dir}/",
    ]


def summarize_command(command: list[str]) -> list[str]:
    return [part if "\n" not in part else part.replace("\n", " ") for part in command]


class RemoteBundlePuller:
    def build_rsync_command(self, config: V5TelemetryRemoteConfig) -> list[str]:
        return build_rsync_command(config)

    def pull_bundles(
        self,
        config: V5TelemetryRemoteConfig,
        *,
        max_files: int | None = None,
    ) -> PullResult:
        started_at = datetime.now(UTC)
        command = (
            build_limited_rsync_command(config)
            if max_files is not None
            else self.build_rsync_command(config)
        )
        warnings: list[str] = []
        pulled_files: list[str] = []
        skipped_files: list[str] = []
        command_summary = summarize_command(command)

        if config.dry_run:
            finished_at = datetime.now(UTC)
            return PullResult(
                strategy=config.strategy,
                remote_host=config.remote_host,
                remote_bundle_dir=str(config.remote_bundle_dir),
                local_inbox_dir=str(config.local_inbox_dir),
                pulled_files=[],
                skipped_files=[],
                command_summary=command_summary,
                started_at=started_at,
                finished_at=finished_at,
                dry_run=True,
                warnings=["dry_run: rsync was not executed"],
            )

        config.local_inbox_dir.mkdir(parents=True, exist_ok=True)
        config.require_identity_file()
        command_input: str | None = None
        if max_files is not None:
            list_command = build_remote_bundle_list_command(config, max_files=max_files)
            command_summary = summarize_command(list_command) + command_summary
            try:
                listed = subprocess.run(
                    list_command,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=config.remote_list_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                finished_at = datetime.now(UTC)
                warnings.append(
                    "remote bundle list timed out after "
                    f"{config.remote_list_timeout_seconds}s"
                )
                return PullResult(
                    strategy=config.strategy,
                    remote_host=config.remote_host,
                    remote_bundle_dir=str(config.remote_bundle_dir),
                    local_inbox_dir=str(config.local_inbox_dir),
                    pulled_files=[],
                    skipped_files=skipped_files,
                    command_summary=command_summary,
                    started_at=started_at,
                    finished_at=finished_at,
                    dry_run=False,
                    warnings=warnings,
                )
            stderr_lines = [line.strip() for line in listed.stderr.splitlines() if line.strip()]
            if listed.returncode != 0:
                warnings.append(f"remote bundle list exited with code {listed.returncode}")
                warnings.extend(stderr_lines[:20])
            remote_files = [
                line.strip()
                for line in listed.stdout.splitlines()
                if line.strip().endswith((".tar.gz", ".tgz"))
            ]
            files_to_pull = []
            for name in remote_files:
                if (config.local_inbox_dir / name).exists():
                    skipped_files.append(name)
                else:
                    files_to_pull.append(name)
            if not files_to_pull:
                finished_at = datetime.now(UTC)
                return PullResult(
                    strategy=config.strategy,
                    remote_host=config.remote_host,
                    remote_bundle_dir=str(config.remote_bundle_dir),
                    local_inbox_dir=str(config.local_inbox_dir),
                    pulled_files=[],
                    skipped_files=skipped_files,
                    command_summary=command_summary,
                    started_at=started_at,
                    finished_at=finished_at,
                    dry_run=False,
                    warnings=warnings,
                )
            command_input = "\n".join(files_to_pull) + "\n"

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                input=command_input,
                timeout=config.rsync_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            finished_at = datetime.now(UTC)
            warnings.append(f"rsync timed out after {config.rsync_timeout_seconds}s")
            return PullResult(
                strategy=config.strategy,
                remote_host=config.remote_host,
                remote_bundle_dir=str(config.remote_bundle_dir),
                local_inbox_dir=str(config.local_inbox_dir),
                pulled_files=pulled_files,
                skipped_files=skipped_files,
                command_summary=command_summary,
                started_at=started_at,
                finished_at=finished_at,
                dry_run=False,
                warnings=warnings,
            )
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
            command_summary=command_summary,
            started_at=started_at,
            finished_at=finished_at,
            dry_run=False,
            warnings=warnings,
        )


def pull_bundles(config: V5TelemetryRemoteConfig) -> PullResult:
    return RemoteBundlePuller().pull_bundles(config)
