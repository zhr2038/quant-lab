from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from quant_lab.decision.contracts import HORIZONS, SYMBOLS, AnalysisResult, HourBar
from quant_lab.decision.engine import build_advice, content_hash, prepare_history
from quant_lab.decision.ledger import Ledger
from quant_lab.decision.pipeline import read_hour_bars
from quant_lab.decision.storage import (
    MAX_INPUT_BYTES,
    MAX_RESULT_BYTES,
    atomic_json,
    load_input,
    load_result,
    read_json,
    result_identity,
)
from quant_lab.export_plane.signatures import (
    load_public_key,
    load_signing_key,
    sha256_file,
    sign_payload,
    verify_payload,
)

CLOUD_ROOT = "/var/lib/quant-lab/decision"


def peak_rss_mib() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class Transport:
    """Fixed-host SSH transport. No public mutation endpoint or HTTP bearer token."""

    def __init__(self, private_root: Path):
        self.ssh = [
            "ssh",
            "-i",
            str(private_root / "id_ed25519"),
            "-o",
            f"UserKnownHostsFile={private_root / 'known_hosts'}",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
        ]
        self.remote = "quant-research@qyun2.hrhome.top"

    def pull(self, archive: Path) -> None:
        import shlex

        archive.mkdir(parents=True, exist_ok=True)
        # No --delete: originals remain on HDD after verified cloud retention.
        self._run(
            [
                "rsync",
                "-rt",
                "--checksum",
                "--safe-links",
                "--max-size=2m",
                "--chmod=F640,D750",
                "-e",
                shlex.join(self.ssh),
                f"{self.remote}:{CLOUD_ROOT}/inputs/",
                str(archive / "inputs") + "/",
            ]
        )
        for name in ("current-input.json", "publication-receipts.json"):
            result = self._run([*self.ssh, self.remote, "cat", f"{CLOUD_ROOT}/{name}"])
            if len(result.stdout) > 2 * 1024 * 1024:
                raise ValueError("transport snapshot exceeds byte budget")
            temporary = archive / (name + ".part")
            temporary.write_bytes(result.stdout)
            read_json(temporary, max_bytes=2 * 1024 * 1024)
            os.replace(temporary, archive / name)

    def push(self, source: Path, *, name: str) -> None:
        import re

        if not re.fullmatch(r"result-[a-f0-9]{64}\.json|archive-ack\.json", name):
            raise ValueError("invalid upload name")
        # sftp batch commands use only constant roots and validated artifact names.
        if any(c in str(source) for c in ('"', "\n", "\r")):
            raise ValueError("unsupported local transport path")
        options = self.ssh[1:]
        batch = (
            f'put "{source}" "{CLOUD_ROOT}/inbox/.{name}.part"\n'
            f'chmod 640 "{CLOUD_ROOT}/inbox/.{name}.part"\n'
            f'rename "{CLOUD_ROOT}/inbox/.{name}.part" "{CLOUD_ROOT}/inbox/{name}"\n'
        )
        self._run(["sftp", *options, "-b", "-", self.remote], stdin=batch.encode())

    @staticmethod
    def _run(command: list[str], *, stdin: bytes | None = None):
        result = subprocess.run(command, input=stdin, capture_output=True, timeout=180)
        if result.returncode:
            raise RuntimeError(
                "SSH transfer failed: " + result.stderr.decode(errors="replace")[-600:]
            )
        return result


def check_capacity(state: Path, archive: Path) -> None:
    for path, reserve in ((state, 2 * 1024**3), (archive, 20 * 1024**3)):
        path.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(path).free < reserve:
            raise ValueError(f"disk reserve not met: {path}")
    files = list(state.rglob("*"))
    if len(files) > 5_000 or sum(p.stat().st_size for p in files if p.is_file()) > 512 * 1024**2:
        raise ValueError("NAS decision SSD workspace exceeds 512 MiB budget")


def update_history(state: Path, archive: Path, bootstrap: Path, inputs, now: datetime):
    path = state / "history.parquet"
    if path.exists():
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("working history exceeds file budget")
        history = [HourBar.model_validate(row) for row in pl.read_parquet(path).to_dicts()]
    else:
        history = read_hour_bars(bootstrap, now=now, days=365)
    merged = {(bar.symbol, bar.ts): bar for bar in history}
    corrections = []
    for bar in inputs.bars:
        key = (bar.symbol, bar.ts)
        old = merged.get(key)
        if old and old.model_dump() != bar.model_dump():
            corrections.append(
                {
                    "previous": old.model_dump(mode="json"),
                    "replacement": bar.model_dump(mode="json"),
                }
            )
        merged[key] = bar
    history = prepare_history(
        [bar for bar in merged.values() if bar.ts >= now - timedelta(days=365)], now
    )
    if len(history) > 35_040:
        raise ValueError("four-symbol annual history exceeds row budget")
    if corrections:
        atomic_json(
            archive / "corrections" / (inputs.snapshot_id + ".json"),
            {"input_snapshot_id": inputs.snapshot_id, "changes": corrections},
        )
    temporary = state / "history.parquet.part"
    if history:
        pl.DataFrame([bar.model_dump() for bar in history]).write_parquet(temporary)
        os.replace(temporary, path)
        digest = sha256_file(path)
        # Archive the exact selected working snapshot, not only a non-restorable hash.
        dest = archive / "history" / (digest + ".parquet")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copyfile(path, dest)
        if sha256_file(dest) != digest:
            raise ValueError("HDD history readback failed")
        atomic_json(
            archive / "history" / (inputs.snapshot_id + ".json"),
            {
                "parquet_sha256": digest,
                "rows": len(history),
                "normalized_data_sha256": content_hash(
                    [b.model_dump(mode="json") for b in history]
                ),
            },
        )
    return history


def write_archive_ack(archive: Path, signing_key, input_key, now: datetime) -> Path:
    entries = []
    cutoff = now - timedelta(days=14)
    for kind, maximum, loader, key in (
        ("inputs", MAX_INPUT_BYTES, load_input, input_key),
        ("results", MAX_RESULT_BYTES, load_result, signing_key.public_key()),
    ):
        for path in sorted((archive / kind).glob("*.json")):
            # The rolling acknowledgement overlaps the seven-day cloud policy.
            if datetime.fromtimestamp(path.stat().st_mtime, UTC) < cutoff:
                continue
            read_json(path, max_bytes=maximum)
            loader(path, key)
            entries.append({"kind": kind, "name": path.name, "sha256": sha256_file(path)})
    if len(entries) > 10_000:
        raise ValueError("archive acknowledgement exceeds entry budget")
    value = {
        "schema_version": "qlab.decision.archive_ack.v1",
        "generated_at": now.isoformat(),
        "entries": entries,
    }
    value["signature"] = sign_payload(value, signing_key)
    path = archive / "archive-ack.json"
    atomic_json(path, value)
    return path


def run_worker(
    *,
    state: Path,
    archive: Path,
    bootstrap: Path,
    private_root: Path,
    code_revision: str,
    now: datetime | None = None,
    transport: Transport | None = None,
) -> AnalysisResult:

    started = time.perf_counter()
    current = now or datetime.now(UTC)
    check_capacity(state, archive)
    transfer = transport or Transport(private_root)
    transfer.pull(archive)
    current = now or datetime.now(UTC)
    input_key = load_public_key(private_root / "producer.pub")
    signing_key = load_signing_key(private_root / "worker.key")
    inputs = load_input(archive / "current-input.json", input_key)
    if inputs.producer_commit != code_revision:
        raise ValueError("worker and cloud producer revisions differ; deploy matching versions")
    if inputs.generated_at > current + timedelta(seconds=5):
        raise ValueError("cloud input is from the future")
    source_path = archive / "inputs" / (inputs.snapshot_id + ".json")
    if not source_path.exists():
        atomic_json(source_path, inputs)
    if load_input(source_path, input_key) != inputs:
        raise ValueError("input archive readback mismatch")
    history = update_history(state, archive, bootstrap, inputs, current)
    publications = read_json(archive / "publication-receipts.json", max_bytes=2 * 1024**2)
    verify_payload(publications, publications.get("signature", ""), input_key)
    if len(publications["publications"]) > 5_000:
        raise ValueError("publication count exceeds budget")
    with Ledger(archive / "forward.duckdb") as ledger:
        for receipt in sorted(publications["publications"], key=lambda item: item["published_at"]):
            import re

            result_id = receipt["result_id"]
            if not re.fullmatch(r"result-[a-f0-9]{64}", result_id):
                raise ValueError("invalid publication reference")
            saved = load_result(
                archive / "results" / (result_id + ".json"), signing_key.public_key()
            )
            ledger.register(
                saved, published_at=datetime.fromisoformat(receipt["published_at"]), now=current
            )
        ledger.mature(history, now=current)
        forward = ledger.summary(now=current)
    previous_path = archive / "current-result.json"
    previous = (
        load_result(previous_path, signing_key.public_key()) if previous_path.exists() else None
    )
    if (
        previous
        and previous.input_snapshot_id == inputs.snapshot_id
        and previous.forward == forward
    ):
        ack = write_archive_ack(archive, signing_key, input_key, current)
        transfer.push(ack, name="archive-ack.json")
        if previous.result_id not in {item["result_id"] for item in publications["publications"]}:
            old = archive / "results" / (previous.result_id + ".json")
            transfer.push(old, name=old.name)
        atomic_json(
            archive / "worker-status.json",
            {
                "status": "UNCHANGED_INPUT_AND_OBSERVATIONS",
                "at": current.isoformat(),
                "result_id": previous.result_id,
                "runtime_seconds": time.perf_counter() - started,
            },
        )
        return previous
    if previous and previous.input_snapshot_id == inputs.snapshot_id:
        advice = previous.advice
    else:
        advice = [
            build_advice(history, inputs=inputs, symbol=symbol, horizon=horizon, now=current)
            for symbol in SYMBOLS
            for horizon in HORIZONS
        ]
    result = AnalysisResult(
        result_id="result-" + "0" * 64,
        generated_at=current,
        input_snapshot_id=inputs.snapshot_id,
        worker_commit=code_revision,
        advice=advice,
        forward=forward,
        history_rows=len(history),
        runtime_seconds=time.perf_counter() - started,
        peak_rss_mib=peak_rss_mib(),
        warnings=inputs.warnings + (["NO_HISTORY_AVAILABLE"] if not history else []),
        signature="pending",
    )
    result = result.model_copy(update={"result_id": result_identity(result)})
    result = result.model_copy(update={"signature": sign_payload(result, signing_key)})
    path = archive / "results" / (result.result_id + ".json")
    atomic_json(path, result)
    if load_result(path, signing_key.public_key()) != result:
        raise ValueError("result archive readback mismatch")
    atomic_json(previous_path, result)
    ack = write_archive_ack(archive, signing_key, input_key, current)
    transfer.push(ack, name="archive-ack.json")
    transfer.push(path, name=path.name)
    atomic_json(
        archive / "worker-status.json",
        {
            "status": "UPLOADED_AWAITING_CLOUD_ACCEPTANCE",
            "at": current.isoformat(),
            "result_id": result.result_id,
            "runtime_seconds": time.perf_counter() - started,
            "peak_rss_mib": peak_rss_mib(),
        },
    )
    return result
