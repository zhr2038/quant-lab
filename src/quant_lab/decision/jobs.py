from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import typer

from quant_lab.decision.current_inputs import collect_current_inputs
from quant_lab.decision.pipeline import accept_results, publish_input
from quant_lab.decision.retention import prune_acknowledged
from quant_lab.decision.storage import atomic_json, load_input
from quant_lab.export_plane.signatures import load_signing_key

app = typer.Typer(help="Bounded decision reference jobs; no exchange writes.")


def publish_current(lake_root, job_root, *, signing_key, code_revision, now):
    # Current REST reads are bounded to four symbols. They do not wait for the
    # historical lake's heavy lock, nor rewrite the lake during compaction.
    path = job_root / "current-input.json"
    previous = (
        load_input(path, load_signing_key(signing_key).public_key()) if path.exists() else None
    )
    try:
        collected = collect_current_inputs(previous)
        snapshot = publish_input(
            lake_root,
            job_root,
            signing_key=signing_key,
            code_revision=code_revision,
            current_inputs=collected,
        )
        status = {
            "status": "WARNING" if snapshot.warnings else "OK",
            "at": snapshot.generated_at.isoformat(),
            "input_snapshot_id": snapshot.snapshot_id,
            "bars": len(snapshot.bars),
            "warnings": snapshot.warnings,
        }
    except (ValueError, OSError) as exc:
        status = {"status": "FAILED", "at": now.isoformat(), "reason": type(exc).__name__}
        atomic_json(job_root / "input-status.json", status)
        raise
    atomic_json(job_root / "input-status.json", status)
    return status


@app.command("accept-only")
def accept_only(
    lake_root: Path = Path("/var/lib/quant-lab/lake"),
    job_root: Path = Path("/var/lib/quant-lab/decision"),
    private_root: Path = Path("/etc/quant-lab/decision"),
) -> None:
    if not any((job_root / "inbox").glob("result-*.json")):
        return
    status = accept_results(
        job_root,
        worker_public_key=private_root / "worker.pub",
        input_public_key=private_root / "producer.pub",
        publication_root=lake_root / "gold" / "decision_reference",
        publication_signing_key=private_root / "producer.key",
    )
    typer.echo(json.dumps(status))
    if status["rejected"]:
        raise typer.Exit(2)


@app.command("cloud-cycle")
def cloud_cycle(
    code_revision: str = typer.Option(...),
    lake_root: Path = Path("/var/lib/quant-lab/lake"),
    job_root: Path = Path("/var/lib/quant-lab/decision"),
    private_root: Path = Path("/etc/quant-lab/decision"),
) -> None:
    now = datetime.now(UTC)
    public = lake_root / "gold" / "decision_reference"
    worker_key = private_root / "worker.pub"
    # Garbage collection is gated by an authenticated NAS readback, never free-space heuristics.
    retained = prune_acknowledged(job_root, public, worker_public_key=worker_key, now=now)
    accepted = accept_results(
        job_root,
        worker_public_key=worker_key,
        input_public_key=private_root / "producer.pub",
        publication_root=public,
        publication_signing_key=private_root / "producer.key",
        now=now,
    )
    inputs = publish_current(
        lake_root,
        job_root,
        signing_key=private_root / "producer.key",
        code_revision=code_revision,
        now=now,
    )
    status = {
        "input": inputs,
        "acceptance": accepted,
        "pruned": len(retained["removed"]),
        "retention_status": retained["status"],
    }
    typer.echo(json.dumps(status))
    if accepted["rejected"]:
        raise typer.Exit(2)


@app.command("nas-worker")
def nas_worker(
    state: Path = Path("/state"),
    archive: Path = Path("/archive"),
    bootstrap: Path = Path("/bootstrap"),
    private_root: Path = Path("/private"),
    code_revision: str = typer.Option(..., envvar="QUANT_LAB_CODE_REVISION"),
) -> None:
    from quant_lab.decision.worker import run_worker

    os.umask(0o027)
    try:
        result = run_worker(
            state=state,
            archive=archive,
            bootstrap=bootstrap,
            private_root=private_root,
            code_revision=code_revision,
        )
        typer.echo(
            json.dumps(
                {
                    "status": "OK",
                    "result_id": result.result_id,
                    "history_rows": result.history_rows,
                    "runtime_seconds": result.runtime_seconds,
                    "peak_rss_mib": result.peak_rss_mib,
                }
            )
        )
    except Exception as exc:
        atomic_json(
            archive / "worker-status.json",
            {
                "status": "FAILED",
                "at": datetime.now(UTC).isoformat(),
                "reason": f"{type(exc).__name__}: {str(exc)[:500]}",
            },
        )
        raise


if __name__ == "__main__":
    app()
