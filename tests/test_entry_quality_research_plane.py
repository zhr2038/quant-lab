from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from polars.testing import assert_frame_equal
from pydantic import ValidationError

import quant_lab.research.entry_quality as entry_quality_module
import quant_lab.research_plane.importer as importer_module
import quant_lab.research_plane.queue as queue_module
import quant_lab.research_worker.runner as runner_module
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.research.entry_quality import (
    ENTRY_QUALITY_HISTORY_OUTPUT_SPECS,
    ENTRY_QUALITY_HISTORY_REPORT_NAMES,
    ENTRY_QUALITY_SCHEMA_VERSION,
    compute_entry_quality_history,
    publish_entry_quality_history_result,
    recover_entry_quality_history_publication,
)
from quant_lab.research_plane.contracts import (
    DEFAULT_RESEARCH_MAX_RESULT_BYTES,
    RESEARCH_SNAPSHOT_SCHEMA,
    RESEARCH_TASK_SCHEMA,
    ResearchDatasetReference,
    ResearchResultManifest,
    ResearchSnapshotManifest,
    ResearchTask,
    ResearchTaskLease,
    ResearchTaskState,
    ResearchTaskStatus,
    ResearchWorkerReceipt,
)
from quant_lab.research_plane.importer import (
    import_entry_quality_history_result,
    validate_entry_quality_history_result_for_import,
)
from quant_lab.research_plane.result import (
    schema_fingerprint,
    validate_entry_quality_history_result_bundle,
    validate_research_task_snapshot,
)
from quant_lab.research_plane.signatures import (
    canonical_json_bytes,
    model_content_sha256,
    sha256_file,
    sign_model,
)
from quant_lab.research_plane.snapshot import (
    ENTRY_QUALITY_INPUT_DATASETS,
    seal_entry_quality_history_snapshot,
)
from quant_lab.research_plane.snapshot_gc import (
    gc_research_snapshot_payloads,
    release_snapshot_payload,
)
from quant_lab.research_plane.status import (
    ensure_research_queue_layout,
    entry_quality_history_plane_status,
    write_research_status,
)
from quant_lab.research_worker.entry_quality_history import (
    _required_market_symbols,
    _scan_projected_dataset,
)
from quant_lab.research_worker.result_writer import write_entry_quality_history_result_bundle
from quant_lab.research_worker.runner import Config, recover_expired_leases
from quant_lab.transfer.snapshot_sync import sync_snapshot_blobs

COMMIT = "a" * 40
TASK_KEY_ID = "cloud-research-v1"
WORKER_KEY_ID = "nas-research-v1"
BUNDLE_ID = "v5-bundle-sha256:" + "b" * 64
GENERATED_AT = datetime(2026, 7, 18, 1, 2, 3, tzinfo=UTC)
DATASETS = [str(path).replace("\\", "/") for path in ENTRY_QUALITY_INPUT_DATASETS]


def _signed_snapshot(
    key: Ed25519PrivateKey,
    *,
    files: list[ResearchDatasetReference] | None = None,
    commit: str = COMMIT,
) -> ResearchSnapshotManifest:
    references = files or []
    provisional = ResearchSnapshotManifest(
        schema_version=RESEARCH_SNAPSHOT_SCHEMA,
        snapshot_id="entry-quality-history-snapshot-test",
        generated_at=GENERATED_AT,
        quant_lab_commit=commit,
        selected_v5_bundle_id=BUNDLE_ID,
        entry_quality_schema_version=ENTRY_QUALITY_SCHEMA_VERSION,
        datasets=DATASETS,
        files=references,
        total_input_bytes=sum(item.size_bytes for item in references),
        total_input_rows=sum(item.row_count for item in references),
        manifest_sha256="0" * 64,
        signature_key_id=TASK_KEY_ID,
        signature="pending",
    )
    digest = model_content_sha256(provisional, blank_fields=("manifest_sha256",))
    unsigned = provisional.model_copy(update={"manifest_sha256": digest})
    return unsigned.model_copy(update={"signature": sign_model(unsigned, key)})


def _signed_task(
    key: Ed25519PrivateKey,
    snapshot: ResearchSnapshotManifest,
    *,
    task_id: str = "entry-quality-history-task-test",
    requested_at: datetime = GENERATED_AT + timedelta(minutes=1),
    commit: str = COMMIT,
    mode: str = "recent_30d",
    cost_mode: str = "conservative",
) -> ResearchTask:
    provisional = ResearchTask(
        schema_version=RESEARCH_TASK_SCHEMA,
        task_id=task_id,
        snapshot_id=snapshot.snapshot_id,
        start_date=date(2026, 6, 19),
        end_date=date(2026, 7, 18),
        mode=mode,
        cost_mode=cost_mode,
        window_hours=24,
        quant_lab_commit=commit,
        entry_quality_schema_version=ENTRY_QUALITY_SCHEMA_VERSION,
        selected_v5_bundle_id=BUNDLE_ID,
        snapshot_manifest_sha256=snapshot.manifest_sha256,
        requested_at=requested_at,
        lease_seconds=3600,
        max_attempts=3,
        signature_key_id=TASK_KEY_ID,
        signature="pending",
    )
    return provisional.model_copy(update={"signature": sign_model(provisional, key)})


def _empty_artifacts(monkeypatch, *, mode: str = "recent_30d", cost_mode: str = "conservative"):
    monkeypatch.setattr(entry_quality_module, "_git_commit", lambda: COMMIT[:7])
    return compute_entry_quality_history(
        trades=pl.DataFrame(),
        lifecycles=pl.DataFrame(),
        market_bars=pl.DataFrame(),
        candidates=pl.DataFrame(),
        labels=pl.DataFrame(),
        costs=pl.DataFrame(),
        start_date=date(2026, 6, 19),
        end_date=date(2026, 7, 18),
        mode=mode,
        cost_mode=cost_mode,
        generated_at=GENERATED_AT,
        generated_from_bundle_id=BUNDLE_ID,
        quant_lab_git_commit=COMMIT,
    )


def _make_result_bundle(tmp_path: Path, monkeypatch):
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    snapshot = _signed_snapshot(task_key)
    task = _signed_task(task_key, snapshot)
    artifacts = _empty_artifacts(monkeypatch)
    root, manifest, receipt = write_entry_quality_history_result_bundle(
        tmp_path / "worker-results",
        task=task,
        snapshot=snapshot,
        artifacts=artifacts,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=worker_key,
        claimed_at=GENERATED_AT + timedelta(minutes=2),
        input_bytes=0,
        cache_hit_bytes=0,
        downloaded_bytes=0,
        peak_rss_bytes=123456,
        compute_duration_seconds=1.25,
        max_result_bytes=50 * 1024**2,
    )
    return SimpleNamespace(
        root=root,
        task_key=task_key,
        worker_key=worker_key,
        snapshot=snapshot,
        task=task,
        artifacts=artifacts,
        manifest=manifest,
        receipt=receipt,
    )


def _validate_bundle(context, root: Path | None = None, *, max_bytes: int = 50 * 1024**2):
    bundle = root or context.root
    manifest = ResearchResultManifest.model_validate_json((bundle / "manifest.json").read_text())
    receipt = ResearchWorkerReceipt.model_validate_json((bundle / "receipt.json").read_text())
    return validate_entry_quality_history_result_bundle(
        bundle,
        manifest=manifest,
        receipt=receipt,
        task=context.task,
        snapshot=context.snapshot,
        worker_public_key=context.worker_key.public_key(),
        expected_worker_key_id=WORKER_KEY_ID,
        max_result_bytes=max_bytes,
    )


def _resign_bundle(
    root: Path,
    worker_key: Ed25519PrivateKey,
    manifest: ResearchResultManifest,
    receipt: ResearchWorkerReceipt,
) -> tuple[ResearchResultManifest, ResearchWorkerReceipt]:
    unsigned_manifest = manifest.model_copy(update={"signature": "pending"})
    manifest = unsigned_manifest.model_copy(
        update={"signature": sign_model(unsigned_manifest, worker_key)}
    )
    (root / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    unsigned_receipt = receipt.model_copy(
        update={
            "result_manifest_sha256": sha256_file(root / "manifest.json"),
            "signature": "pending",
        }
    )
    receipt = unsigned_receipt.model_copy(
        update={"signature": sign_model(unsigned_receipt, worker_key)}
    )
    (root / "receipt.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return manifest, receipt


def _rewrite_output(
    root: Path,
    manifest: ResearchResultManifest,
    dataset_name: str,
    frame: pl.DataFrame,
) -> ResearchResultManifest:
    outputs = []
    for output in manifest.outputs:
        if output.dataset_name != dataset_name:
            outputs.append(output)
            continue
        path = root / output.relative_path
        frame.write_parquet(path)
        outputs.append(
            output.model_copy(
                update={
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "row_count": frame.height,
                    "schema_fingerprint": schema_fingerprint(frame.schema),
                }
            )
        )
    output_bytes = sum(item.size_bytes for item in outputs) + sum(
        item.size_bytes for item in manifest.reports
    )
    return manifest.model_copy(update={"outputs": outputs, "output_bytes": output_bytes})


def _write_cloud_control(queue: Path, context) -> None:
    running = queue / "running" / context.task.task_id
    running.mkdir(parents=True, exist_ok=True)
    (running / "task.json").write_text(context.task.model_dump_json(indent=2), encoding="utf-8")
    snapshot_root = queue / "snapshots" / context.snapshot.snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=True)
    (snapshot_root / "manifest.json").write_text(
        context.snapshot.model_dump_json(indent=2), encoding="utf-8"
    )
    (snapshot_root / "SEALED").write_text(context.snapshot.manifest_sha256 + "\n")
    write_research_status(
        queue,
        ResearchTaskStatus(
            task_id=context.task.task_id,
            snapshot_id=context.snapshot.snapshot_id,
            start_date=context.task.start_date,
            end_date=context.task.end_date,
            mode=context.task.mode,
            cost_mode=context.task.cost_mode,
            state=ResearchTaskState.VALIDATING_ON_CLOUD,
            worker_id="nas-research-worker-01",
            requested_at=context.task.requested_at,
            claimed_at=context.task.requested_at + timedelta(minutes=1),
            heartbeat_at=context.task.requested_at + timedelta(minutes=2),
            attempt=1,
            max_attempts=context.task.max_attempts,
            import_status="result_uploaded",
        ),
    )


def test_research_contracts_forbid_extra_and_canonical_json_is_stable() -> None:
    key = Ed25519PrivateKey.generate()
    snapshot = _signed_snapshot(key)
    task = _signed_task(key, snapshot)
    with pytest.raises(ValidationError):
        ResearchTask.model_validate({**task.model_dump(mode="json"), "exchange_secret": "no"})
    first = canonical_json_bytes({"b": 2, "a": 1})
    second = canonical_json_bytes({"a": 1, "b": 2})
    assert first == second == b'{"a":1,"b":2}'
    unsafe = task.model_dump(mode="json")
    unsafe["live_order_effect"] = "submit_orders"
    with pytest.raises(ValidationError):
        ResearchTask.model_validate(unsafe)


def test_recent_7d_is_normalized_before_snapshot_and_task_signature(
    tmp_path: Path,
    monkeypatch,
) -> None:
    key = Ed25519PrivateKey.generate()
    snapshot = _signed_snapshot(key)
    observed: dict[str, date] = {}

    def fake_snapshot(*_args, start_date: date, end_date: date, **_kwargs):
        observed["start_date"] = start_date
        observed["end_date"] = end_date
        return snapshot

    monkeypatch.setattr(queue_module, "seal_entry_quality_history_snapshot", fake_snapshot)
    task, _ = queue_module.create_entry_quality_history_task(
        tmp_path / "lake",
        tmp_path / "queue",
        start_date=date(2026, 6, 1),
        end_date=date(2026, 7, 18),
        mode="recent_7d",
        cost_mode="conservative",
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
        selected_v5_bundle_id=BUNDLE_ID,
    )
    assert observed == {
        "start_date": date(2026, 7, 12),
        "end_date": date(2026, 7, 18),
    }
    assert task.start_date == date(2026, 7, 12)
    assert task.end_date == date(2026, 7, 18)


def test_research_plane_web_status_is_read_only_and_truthful(tmp_path: Path) -> None:
    queue = tmp_path / "missing-queue"
    idle = entry_quality_history_plane_status(queue)
    assert idle["state"] == "idle"
    assert idle["nas_offline_behavior"] == "wait_no_local_fallback"
    assert not queue.exists()

    ensure_research_queue_layout(queue)
    status = ResearchTaskStatus(
        task_id="eqh-web-status",
        snapshot_id="eqh-web-snapshot",
        start_date=date(2026, 6, 19),
        end_date=date(2026, 7, 18),
        mode="recent_30d",
        cost_mode="conservative",
        state=ResearchTaskState.COMPUTING,
        worker_id="nas-research-worker-01",
        requested_at=GENERATED_AT,
        claimed_at=GENERATED_AT + timedelta(minutes=1),
        heartbeat_at=GENERATED_AT + timedelta(minutes=2),
        input_bytes=100,
        downloaded_bytes=25,
        cache_hit_bytes=75,
        import_status="waiting_for_nas_result",
    )
    write_research_status(queue, status)
    payload = entry_quality_history_plane_status(queue)
    assert payload["state"] == "computing"
    assert payload["task"]["task_id"] == status.task_id
    assert payload["task"]["live_order_effect"] == "none"


def test_task_and_snapshot_signatures_bind_commit_schema_snapshot_and_key() -> None:
    key = Ed25519PrivateKey.generate()
    snapshot = _signed_snapshot(key)
    task = _signed_task(key, snapshot)
    validate_research_task_snapshot(
        task,
        snapshot,
        task_public_key=key.public_key(),
        expected_key_id=TASK_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    with pytest.raises(ValueError, match="unknown_signature_key"):
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id="unknown-key",
        )
    with pytest.raises(ValueError, match="current_commit_mismatch"):
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id=TASK_KEY_ID,
            expected_quant_lab_commit="c" * 40,
        )
    changed = task.model_copy(update={"window_hours": 48})
    with pytest.raises(ValueError, match="signature verification failed"):
        validate_research_task_snapshot(
            changed,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id=TASK_KEY_ID,
        )


def test_snapshot_selects_only_six_datasets_and_required_time_ranges(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    inside = datetime(2026, 5, 10, 12, tzinfo=UTC)
    rows_by_dataset = {
        "silver/v5_trade_event": {"ts_utc": inside, "symbol": "SOL-USDT"},
        "silver/v5_order_lifecycle": {"decision_ts": inside, "symbol": "SOL-USDT"},
        "silver/v5_candidate_event": {
            "ts_utc": inside,
            "candidate_id": "candidate-1",
            "symbol": "SOL-USDT",
        },
        "gold/v5_candidate_label": {
            "decision_ts": inside,
            "label_ts": inside + timedelta(hours=24),
            "candidate_id": "candidate-1",
        },
        "gold/cost_bucket_daily": {"as_of_date": "2026-05-10", "symbol": "SOL-USDT"},
    }
    for dataset, row in rows_by_dataset.items():
        path = lake / dataset / "inside.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame([row]).write_parquet(path)
    market_root = lake / "silver/market_bar"
    market_root.mkdir(parents=True, exist_ok=True)
    for name, ts in {
        "too-old": datetime(2026, 5, 8, 23, tzinfo=UTC),
        "pre-window": datetime(2026, 5, 9, 1, tzinfo=UTC),
        "forward-window": datetime(2026, 5, 13, 23, tzinfo=UTC),
        "too-new": datetime(2026, 5, 14, 1, tzinfo=UTC),
    }.items():
        pl.DataFrame(
            [{"ts": ts, "symbol": "SOL-USDT", "timeframe": "1H", "close": 100.0}]
        ).write_parquet(market_root / f"{name}.parquet")

    key = Ed25519PrivateKey.generate()
    queue = tmp_path / "queue"
    manifest = seal_entry_quality_history_snapshot(
        lake,
        queue,
        start_date=date(2026, 5, 10),
        end_date=date(2026, 5, 10),
        selected_v5_bundle_id=BUNDLE_ID,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )

    assert set(manifest.datasets) == set(DATASETS)
    market_names = {
        Path(item.relative_path).name
        for item in manifest.files
        if item.dataset_name == "silver/market_bar"
    }
    assert market_names == {"pre-window.parquet", "forward-window.parquet"}
    assert len({item.dataset_name for item in manifest.files}) == 6
    same = seal_entry_quality_history_snapshot(
        lake,
        queue,
        start_date=date(2026, 5, 10),
        end_date=date(2026, 5, 10),
        selected_v5_bundle_id=BUNDLE_ID,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert same.snapshot_id == manifest.snapshot_id

    assert release_snapshot_payload(queue, manifest.snapshot_id, reason="test") is True
    assert not (queue / "snapshots" / manifest.snapshot_id / "files").exists()
    assert (queue / "snapshots" / manifest.snapshot_id / "FILES_RELEASED.json").is_file()
    rehydrated = seal_entry_quality_history_snapshot(
        lake,
        queue,
        start_date=date(2026, 5, 10),
        end_date=date(2026, 5, 10),
        selected_v5_bundle_id=BUNDLE_ID,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert rehydrated.snapshot_id == manifest.snapshot_id
    assert (queue / "snapshots" / manifest.snapshot_id / "files").is_dir()
    assert not (queue / "snapshots" / manifest.snapshot_id / "FILES_RELEASED.json").exists()

    new_candidate = lake / "silver/v5_candidate_event/new.parquet"
    pl.DataFrame(
        [{"ts_utc": inside + timedelta(hours=1), "candidate_id": "candidate-2"}]
    ).write_parquet(new_candidate)
    changed = seal_entry_quality_history_snapshot(
        lake,
        queue,
        start_date=date(2026, 5, 10),
        end_date=date(2026, 5, 10),
        selected_v5_bundle_id=BUNDLE_ID,
        signing_key=key,
        signature_key_id=TASK_KEY_ID,
        quant_lab_commit=COMMIT,
    )
    assert changed.snapshot_id != manifest.snapshot_id
    assert (queue / "snapshots" / manifest.snapshot_id / "SEALED").is_file()


def test_snapshot_blob_sync_cold_then_warm_and_interrupted_partial(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    pl.DataFrame({"ts": [GENERATED_AT], "value": [1]}).write_parquet(source)
    reference = ResearchDatasetReference(
        dataset_name=DATASETS[0],
        source_relative_path="silver/v5_trade_event/source.parquet",
        relative_path="silver/v5_trade_event/source.parquet",
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        row_count=1,
        mtime_ns=source.stat().st_mtime_ns,
        min_ts=GENERATED_AT,
        max_ts=GENERATED_AT,
    )
    manifest = _signed_snapshot(Ed25519PrivateKey.generate(), files=[reference])
    calls: list[str] = []

    def fetch(relative: str, target: Path) -> None:
        calls.append(relative)
        target.write_bytes(source.read_bytes())

    first = sync_snapshot_blobs(
        manifest,
        data_root=tmp_path / "data",
        fetch_blob=fetch,
        min_free_disk_bytes=0,
        max_snapshot_bytes=10 * 1024**2,
    )
    second = sync_snapshot_blobs(
        manifest,
        data_root=tmp_path / "data",
        fetch_blob=fetch,
        min_free_disk_bytes=0,
        max_snapshot_bytes=10 * 1024**2,
    )
    assert first.downloaded_files == 1
    assert second.downloaded_files == 0
    assert second.cache_hits == 1
    assert len(calls) == 1

    other_root = tmp_path / "interrupted"

    def broken_fetch(_relative: str, target: Path) -> None:
        target.write_bytes(b"partial")
        raise ConnectionError("scp interrupted")

    with pytest.raises(ConnectionError, match="interrupted"):
        sync_snapshot_blobs(
            manifest,
            data_root=other_root,
            fetch_blob=broken_fetch,
            min_free_disk_bytes=0,
            max_snapshot_bytes=10 * 1024**2,
        )
    assert not list((other_root / "snapshots").glob(f"{manifest.snapshot_id}*"))


def test_snapshot_gc_protects_active_references_then_releases_completed_payload(
    tmp_path: Path,
) -> None:
    queue = ensure_research_queue_layout(tmp_path / "queue")
    key = Ed25519PrivateKey.generate()
    source = tmp_path / "payload.parquet"
    pl.DataFrame({"ts": [GENERATED_AT], "value": [1]}).write_parquet(source)
    reference = ResearchDatasetReference(
        dataset_name=DATASETS[0],
        source_relative_path="silver/v5_trade_event/payload.parquet",
        relative_path="silver/v5_trade_event/payload.parquet",
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        row_count=1,
        mtime_ns=source.stat().st_mtime_ns,
        min_ts=GENERATED_AT,
        max_ts=GENERATED_AT,
    )
    snapshot = _signed_snapshot(key, files=[reference])
    task = _signed_task(key, snapshot)
    snapshot_root = queue / "snapshots" / snapshot.snapshot_id
    payload = snapshot_root / "files" / reference.relative_path
    payload.parent.mkdir(parents=True)
    shutil.copy2(source, payload)
    (snapshot_root / "manifest.json").write_text(snapshot.model_dump_json(indent=2))
    (snapshot_root / "SEALED").write_text(snapshot.manifest_sha256 + "\n")
    pending = queue / "pending" / task.task_id
    pending.mkdir()
    (pending / "task.json").write_text(task.model_dump_json(indent=2))

    active = gc_research_snapshot_payloads(
        queue,
        retention_days=0,
        max_payload_bytes=0,
        now=GENERATED_AT + timedelta(days=30),
    )
    assert active.released_snapshot_count == 0
    assert payload.is_file()

    os.replace(pending, queue / "completed" / task.task_id)
    released = gc_research_snapshot_payloads(
        queue,
        retention_days=0,
        max_payload_bytes=0,
        now=GENERATED_AT + timedelta(days=30),
    )
    assert released.released_snapshot_ids == (snapshot.snapshot_id,)
    assert not (snapshot_root / "files").exists()
    assert (snapshot_root / "manifest.json").is_file()
    assert (snapshot_root / "SEALED").is_file()
    assert (snapshot_root / "FILES_RELEASED.json").is_file()
    assert "payload_released" in (queue / "audit" / "snapshot_gc.jsonl").read_text()


def test_snapshot_blob_sync_rejects_insufficient_disk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.parquet"
    pl.DataFrame({"ts": [GENERATED_AT], "value": [1]}).write_parquet(source)
    reference = ResearchDatasetReference(
        dataset_name=DATASETS[0],
        source_relative_path="silver/v5_trade_event/source.parquet",
        relative_path="silver/v5_trade_event/source.parquet",
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        row_count=1,
        mtime_ns=source.stat().st_mtime_ns,
        min_ts=GENERATED_AT,
        max_ts=GENERATED_AT,
    )
    manifest = _signed_snapshot(Ed25519PrivateKey.generate(), files=[reference])
    monkeypatch.setattr(
        "quant_lab.transfer.snapshot_sync.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    with pytest.raises(RuntimeError, match="insufficient_nas_disk_space"):
        sync_snapshot_blobs(
            manifest,
            data_root=tmp_path / "disk-full",
            fetch_blob=lambda _relative, _target: None,
            min_free_disk_bytes=1,
            max_snapshot_bytes=10 * 1024**2,
        )


def test_snapshot_blob_sync_rejects_symlink_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.parquet"
    pl.DataFrame({"ts": [GENERATED_AT], "value": [1]}).write_parquet(source)
    reference = ResearchDatasetReference(
        dataset_name=DATASETS[0],
        source_relative_path="silver/v5_trade_event/source.parquet",
        relative_path="silver/v5_trade_event/source.parquet",
        sha256=sha256_file(source),
        size_bytes=source.stat().st_size,
        row_count=1,
        mtime_ns=source.stat().st_mtime_ns,
        min_ts=GENERATED_AT,
        max_ts=GENERATED_AT,
    )
    manifest = _signed_snapshot(Ed25519PrivateKey.generate(), files=[reference])
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: path.name.endswith(".partial") or real_is_symlink(path),
    )

    def fetch(_relative: str, target: Path) -> None:
        target.write_bytes(source.read_bytes())

    with pytest.raises(RuntimeError, match="blob_symlink_forbidden"):
        sync_snapshot_blobs(
            manifest,
            data_root=tmp_path / "symlink",
            fetch_blob=fetch,
            min_free_disk_bytes=0,
            max_snapshot_bytes=10 * 1024**2,
        )


def test_worker_lazy_scan_projects_columns_and_filters_actual_required_symbols(
    tmp_path: Path,
) -> None:
    path = tmp_path / "market.parquet"
    pl.DataFrame(
        [
            {
                "ts": GENERATED_AT,
                "symbol": "DOGE-USDT",
                "timeframe": "1H",
                "close": 1.0,
                "secret": "drop",
            },
            {
                "ts": GENERATED_AT,
                "symbol": "SOL/USDT",
                "timeframe": "1H",
                "close": 100.0,
                "secret": "drop",
            },
            {
                "ts": GENERATED_AT,
                "symbol": "XRP-USDT",
                "timeframe": "1H",
                "close": 2.0,
                "secret": "drop",
            },
            {
                "ts": GENERATED_AT,
                "symbol": "DOGE-USDT",
                "timeframe": "5m",
                "close": 1.0,
                "secret": "drop",
            },
        ]
    ).write_parquet(path)
    candidate = pl.DataFrame({"symbol": ["DOGE/USDT"]})
    required = _required_market_symbols({"candidates": candidate})
    frame = _scan_projected_dataset(
        [path],
        columns=("ts", "symbol", "timeframe", "close"),
        time_columns=("ts",),
        start_dt=GENERATED_AT - timedelta(hours=1),
        end_dt=GENERATED_AT + timedelta(hours=1),
        symbols=required,
        timeframe="1H",
    )
    assert "secret" not in frame.columns
    assert set(frame.get_column("symbol")) == {"DOGE-USDT", "SOL/USDT"}


@pytest.mark.parametrize("mode", ["full", "recent_7d", "recent_30d", "walk_forward"])
@pytest.mark.parametrize("cost_mode", ["conservative", "quant_lab"])
def test_pure_compute_matches_legacy_publish_for_all_modes(
    tmp_path: Path,
    monkeypatch,
    mode: str,
    cost_mode: str,
) -> None:
    monkeypatch.setattr(entry_quality_module, "_git_commit", lambda: COMMIT[:7])
    lake = tmp_path / f"{mode}-{cost_mode}"
    frames = _history_input_frames()
    paths = {
        "trades": "silver/v5_trade_event",
        "lifecycles": "silver/v5_order_lifecycle",
        "market_bars": "silver/market_bar",
        "candidates": "silver/v5_candidate_event",
        "labels": "gold/v5_candidate_label",
        "costs": "gold/cost_bucket_daily",
    }
    for name, relative in paths.items():
        write_parquet_dataset(frames[name], lake / relative)
    artifacts = compute_entry_quality_history(
        **frames,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 10),
        mode=mode,
        cost_mode=cost_mode,
        generated_at=GENERATED_AT,
        generated_from_bundle_id="",
    )
    entry_quality_module._legacy_build_and_publish_entry_quality_history(
        lake,
        start_date=date(2026, 5, 1),
        end_date=date(2026, 5, 10),
        mode=mode,
        cost_mode=cost_mode,
    )
    for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS:
        actual = read_parquet_dataset(lake / spec.relative_path)
        expected = artifacts.frames_by_dataset()[spec.dataset_name]
        assert_frame_equal(
            _normalize_generated_at(actual),
            _normalize_generated_at(expected),
            check_row_order=False,
        )
    for name in ENTRY_QUALITY_HISTORY_REPORT_NAMES:
        legacy = (lake / "reports" / name).read_bytes()
        expected = artifacts.reports[name]
        if name.endswith(".csv"):
            assert_frame_equal(
                _normalize_generated_at(pl.read_csv(io.BytesIO(legacy))),
                _normalize_generated_at(pl.read_csv(io.BytesIO(expected))),
                check_row_order=False,
                check_dtypes=False,
            )
        else:
            assert legacy.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n")


def test_result_bundle_strict_validation_and_common_rejections(tmp_path: Path, monkeypatch) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    validated = _validate_bundle(context)
    assert set(validated.frames) == {
        spec.dataset_name for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS
    }
    assert set(validated.reports) == set(ENTRY_QUALITY_HISTORY_REPORT_NAMES)

    tampered = tmp_path / "tampered"
    shutil.copytree(context.root, tampered)
    output = tampered / context.manifest.outputs[0].relative_path
    output.write_bytes(output.read_bytes() + b"x")
    with pytest.raises(ValueError, match="file_integrity_mismatch"):
        _validate_bundle(context, tampered)

    with pytest.raises(ValueError, match="size_limit"):
        _validate_bundle(context, max_bytes=1)

    wrong_task = context.task.model_copy(update={"quant_lab_commit": "c" * 40})
    with pytest.raises(ValueError, match="quant_lab_commit_mismatch"):
        validate_entry_quality_history_result_bundle(
            context.root,
            manifest=context.manifest,
            receipt=context.receipt,
            task=wrong_task,
            snapshot=context.snapshot,
            worker_public_key=context.worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=50 * 1024**2,
        )

    wrong_worker_manifest = context.manifest.model_copy(update={"worker_commit": "c" * 40})
    wrong_worker_receipt = context.receipt.model_copy(update={"worker_commit": "c" * 40})
    wrong_worker_manifest, wrong_worker_receipt = _resign_bundle(
        context.root,
        context.worker_key,
        wrong_worker_manifest,
        wrong_worker_receipt,
    )
    with pytest.raises(ValueError, match="worker_code_mismatch"):
        validate_entry_quality_history_result_bundle(
            context.root,
            manifest=wrong_worker_manifest,
            receipt=wrong_worker_receipt,
            task=context.task,
            snapshot=context.snapshot,
            worker_public_key=context.worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=50 * 1024**2,
        )

    unknown_key_manifest = wrong_worker_manifest.model_copy(update={"worker_key_id": "unknown"})
    with pytest.raises(ValueError, match="unknown_worker_key"):
        validate_entry_quality_history_result_bundle(
            context.root,
            manifest=unknown_key_manifest,
            receipt=wrong_worker_receipt,
            task=context.task,
            snapshot=context.snapshot,
            worker_public_key=context.worker_key.public_key(),
            expected_worker_key_id=WORKER_KEY_ID,
            max_result_bytes=50 * 1024**2,
        )


def test_result_rejects_anti_leakage_failure_and_live_state(tmp_path: Path, monkeypatch) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    anti_root = tmp_path / "anti-fail"
    shutil.copytree(context.root, anti_root)
    manifest = ResearchResultManifest.model_validate_json((anti_root / "manifest.json").read_text())
    receipt = ResearchWorkerReceipt.model_validate_json((anti_root / "receipt.json").read_text())
    anti_output = next(
        item
        for item in manifest.outputs
        if item.dataset_name == "v5_entry_quality_history_anti_leakage_check"
    )
    anti = pl.read_parquet(anti_root / anti_output.relative_path).with_columns(
        pl.when(pl.col("check_name") == "history_window_respected")
        .then(pl.lit("FAIL"))
        .otherwise(pl.col("status"))
        .alias("status"),
        pl.when(pl.col("check_name") == "history_window_respected")
        .then(pl.lit(1))
        .otherwise(pl.col("violation_count"))
        .alias("violation_count"),
    )
    manifest = _rewrite_output(
        anti_root,
        manifest,
        "v5_entry_quality_history_anti_leakage_check",
        anti,
    )
    _resign_bundle(anti_root, context.worker_key, manifest, receipt)
    with pytest.raises(ValueError, match="anti_leakage_failed"):
        _validate_bundle(context, anti_root)

    live_root = tmp_path / "live-state"
    shutil.copytree(context.root, live_root)
    manifest = ResearchResultManifest.model_validate_json((live_root / "manifest.json").read_text())
    receipt = ResearchWorkerReceipt.model_validate_json((live_root / "receipt.json").read_text())
    metrics_output = next(
        item for item in manifest.outputs if item.dataset_name == "v5_entry_quality_history_metrics"
    )
    metrics = pl.read_parquet(live_root / metrics_output.relative_path).with_columns(
        pl.lit("LIVE_SMALL_READY").alias("metrics_json")
    )
    manifest = _rewrite_output(
        live_root,
        manifest,
        "v5_entry_quality_history_metrics",
        metrics,
    )
    _resign_bundle(live_root, context.worker_key, manifest, receipt)
    with pytest.raises(ValueError, match="live_state_forbidden"):
        _validate_bundle(context, live_root)


def test_result_rejects_missing_output_commit_provenance(tmp_path: Path, monkeypatch) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    missing_root = tmp_path / "missing-output-commit"
    shutil.copytree(context.root, missing_root)
    manifest = ResearchResultManifest.model_validate_json(
        (missing_root / "manifest.json").read_text()
    )
    receipt = ResearchWorkerReceipt.model_validate_json(
        (missing_root / "receipt.json").read_text()
    )
    metrics_output = next(
        item for item in manifest.outputs if item.dataset_name == "v5_entry_quality_history_metrics"
    )
    metrics = pl.read_parquet(missing_root / metrics_output.relative_path).with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("quant_lab_git_commit")
    )
    manifest = _rewrite_output(
        missing_root,
        manifest,
        "v5_entry_quality_history_metrics",
        metrics,
    )
    _resign_bundle(missing_root, context.worker_key, manifest, receipt)

    with pytest.raises(ValueError, match="quant_lab_git_commit"):
        _validate_bundle(context, missing_root)


def test_result_rejects_partial_null_and_stale_row_provenance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    partial_root = tmp_path / "partial-null-commit"
    shutil.copytree(context.root, partial_root)
    manifest = ResearchResultManifest.model_validate_json(
        (partial_root / "manifest.json").read_text()
    )
    receipt = ResearchWorkerReceipt.model_validate_json(
        (partial_root / "receipt.json").read_text()
    )
    metrics_output = next(
        item for item in manifest.outputs if item.dataset_name == "v5_entry_quality_history_metrics"
    )
    metrics = pl.read_parquet(partial_root / metrics_output.relative_path)
    metrics = pl.concat([metrics, metrics], how="vertical").with_row_index("row_index")
    metrics = metrics.with_columns(
        pl.when(pl.col("row_index") == 0)
        .then(pl.lit(None, dtype=pl.Utf8))
        .otherwise(pl.col("quant_lab_git_commit"))
        .alias("quant_lab_git_commit")
    ).drop("row_index")
    manifest = _rewrite_output(
        partial_root,
        manifest,
        "v5_entry_quality_history_metrics",
        metrics,
    )
    _resign_bundle(partial_root, context.worker_key, manifest, receipt)
    with pytest.raises(ValueError, match="scope_null.*quant_lab_git_commit"):
        _validate_bundle(context, partial_root)

    stale_root = tmp_path / "stale-source-version"
    shutil.copytree(context.root, stale_root)
    manifest = ResearchResultManifest.model_validate_json(
        (stale_root / "manifest.json").read_text()
    )
    receipt = ResearchWorkerReceipt.model_validate_json(
        (stale_root / "receipt.json").read_text()
    )
    metrics_output = next(
        item for item in manifest.outputs if item.dataset_name == "v5_entry_quality_history_metrics"
    )
    metrics = pl.read_parquet(stale_root / metrics_output.relative_path).with_columns(
        pl.lit(f"entry_quality:{'b' * 40}").alias("source_version")
    )
    manifest = _rewrite_output(
        stale_root,
        manifest,
        "v5_entry_quality_history_metrics",
        metrics,
    )
    _resign_bundle(stale_root, context.worker_key, manifest, receipt)
    with pytest.raises(ValueError, match="scope_mismatch.*source_version"):
        _validate_bundle(context, stale_root)


def test_result_rejects_schema_row_count_and_unsafe_path(tmp_path: Path, monkeypatch) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    row_root = tmp_path / "row-count"
    shutil.copytree(context.root, row_root)
    manifest = ResearchResultManifest.model_validate_json((row_root / "manifest.json").read_text())
    receipt = ResearchWorkerReceipt.model_validate_json((row_root / "receipt.json").read_text())
    outputs = [
        item.model_copy(update={"row_count": item.row_count + 1}) if index == 0 else item
        for index, item in enumerate(manifest.outputs)
    ]
    manifest = manifest.model_copy(update={"outputs": outputs})
    _resign_bundle(row_root, context.worker_key, manifest, receipt)
    with pytest.raises(ValueError, match="row_count_mismatch"):
        _validate_bundle(context, row_root)

    schema_root = tmp_path / "schema"
    shutil.copytree(context.root, schema_root)
    manifest = ResearchResultManifest.model_validate_json(
        (schema_root / "manifest.json").read_text()
    )
    receipt = ResearchWorkerReceipt.model_validate_json((schema_root / "receipt.json").read_text())
    metrics_output = next(
        item for item in manifest.outputs if item.dataset_name == "v5_entry_quality_history_metrics"
    )
    metrics = pl.read_parquet(schema_root / metrics_output.relative_path).with_columns(
        pl.col("metrics_json").str.len_chars().cast(pl.Int64).alias("metrics_json")
    )
    manifest = _rewrite_output(
        schema_root,
        manifest,
        "v5_entry_quality_history_metrics",
        metrics,
    )
    _resign_bundle(schema_root, context.worker_key, manifest, receipt)
    with pytest.raises(ValueError, match="schema_mismatch"):
        _validate_bundle(context, schema_root)

    payload = context.manifest.outputs[0].model_dump(mode="json")
    payload["relative_path"] = "../escape.parquet"
    with pytest.raises(ValidationError, match="unsafe relative path"):
        type(context.manifest.outputs[0]).model_validate(payload)


def test_publish_is_atomic_across_gold_and_reports(tmp_path: Path, monkeypatch) -> None:
    artifacts = _empty_artifacts(monkeypatch)
    lake = tmp_path / "lake"
    publish_entry_quality_history_result(
        lake,
        artifacts,
        generation_id="generation-old",
        snapshot_id="snapshot-old",
        task_id="task-old",
        reports=artifacts.reports,
    )
    before_pointer = (lake / "gold/entry_quality_history_generation.json").read_bytes()
    before_report = (lake / "reports/entry_quality_historical_metrics.json").read_bytes()
    changed_reports = dict(artifacts.reports)
    changed_reports["entry_quality_historical_metrics.json"] = b'{"changed":true}'
    changed = replace(artifacts, reports=changed_reports)
    real_replace = os.replace
    failed = False

    def fail_once(source, destination):
        nonlocal failed
        if (
            not failed
            and Path(destination) == lake / "reports/entry_quality_historical_metrics.json"
        ):
            failed = True
            raise OSError("injected report switch failure")
        return real_replace(source, destination)

    monkeypatch.setattr(entry_quality_module.os, "replace", fail_once)
    with pytest.raises(OSError, match="injected"):
        publish_entry_quality_history_result(
            lake,
            changed,
            generation_id="generation-new",
            snapshot_id="snapshot-new",
            task_id="task-new",
            reports=changed.reports,
        )
    assert (lake / "gold/entry_quality_history_generation.json").read_bytes() == before_pointer
    assert (lake / "reports/entry_quality_historical_metrics.json").read_bytes() == before_report
    for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS:
        metadata = json.loads((lake / spec.relative_path / "_research_generation.json").read_text())
        assert metadata["generation_id"] == "generation-old"


def test_publish_journal_recovers_after_process_loss_between_directory_swaps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifacts = _empty_artifacts(monkeypatch)
    lake = tmp_path / "lake"
    publish_entry_quality_history_result(
        lake,
        artifacts,
        generation_id="generation-old",
        snapshot_id="snapshot-old",
        task_id="task-old",
        reports=artifacts.reports,
    )
    spec = ENTRY_QUALITY_HISTORY_OUTPUT_SPECS[0]
    target = lake / spec.relative_path
    transaction_id = "deadbeef" * 4
    staging_root = lake / "gold" / ".__eqh_s_deadbeef"
    backup_root = lake / "gold" / ".__eqh_b_deadbeef"
    staging_root.mkdir()
    backup_root.mkdir()
    backup = backup_root / "d00"
    os.replace(target, backup)
    journal = {
        "schema_version": "entry_quality_history_publish_transaction.v1",
        "transaction_id": transaction_id,
        "generation_id": "generation-new",
        "snapshot_id": "snapshot-new",
        "task_id": "task-new",
        "staging_root": "gold/.__eqh_s_deadbeef",
        "backup_root": "gold/.__eqh_b_deadbeef",
        "items": [
            {
                "kind": "directory",
                "target": str(spec.relative_path).replace("\\", "/"),
                "staged": "gold/.__eqh_s_deadbeef/d00",
                "backup": "gold/.__eqh_b_deadbeef/d00",
                "target_existed": True,
            }
        ],
    }
    journal_path = lake / "gold" / ".entry_quality_history_publish_transaction.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")

    assert recover_entry_quality_history_publication(lake) is True
    assert target.is_dir()
    assert not journal_path.exists()
    metadata = json.loads((target / "_research_generation.json").read_text())
    assert metadata["generation_id"] == "generation-old"

    staging_root.mkdir()
    backup_root.mkdir()
    (backup_root / "d00").mkdir()
    committed_journal = journal | {
        "generation_id": "generation-old",
        "snapshot_id": "snapshot-old",
        "task_id": "task-old",
    }
    journal_path.write_text(json.dumps(committed_journal), encoding="utf-8")
    assert recover_entry_quality_history_publication(lake) is True
    assert target.is_dir()
    assert not staging_root.exists()
    assert not backup_root.exists()
    assert not journal_path.exists()


def test_importer_retries_publish_infrastructure_failure_without_rejection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    queue = ensure_research_queue_layout(tmp_path / "queue")
    lake = tmp_path / "lake"
    _write_cloud_control(queue, context)
    shutil.copytree(context.root, queue / "results/inbox" / context.task.task_id)

    with monkeypatch.context() as patcher:
        patcher.setattr(
            importer_module,
            "publish_entry_quality_history_result",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk temporarily full")),
        )
        result = import_entry_quality_history_result(
            lake,
            queue,
            context.task.task_id,
            task_public_key=context.task_key.public_key(),
            worker_public_key=context.worker_key.public_key(),
            expected_task_key_id=TASK_KEY_ID,
            expected_worker_key_id=WORKER_KEY_ID,
            expected_quant_lab_commit=COMMIT,
        )
    assert result.state == "publish_retry_pending"
    assert (queue / "results/inbox" / context.task.task_id).is_dir()
    assert (queue / "running" / context.task.task_id).is_dir()
    assert not (queue / "results/rejected" / context.task.task_id).exists()
    status = json.loads((queue / "status" / f"{context.task.task_id}.json").read_text())
    assert status["import_status"] == "publish_retry_pending"
    assert '"status":"RETRY"' in (
        queue / "validation" / f"{context.task.task_id}.jsonl"
    ).read_text()

    recovered = import_entry_quality_history_result(
        lake,
        queue,
        context.task.task_id,
        task_public_key=context.task_key.public_key(),
        worker_public_key=context.worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert recovered.state == "completed"


def test_importer_publishes_all_tables_is_idempotent_and_recovers_finalize(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    queue = ensure_research_queue_layout(tmp_path / "queue")
    lake = tmp_path / "lake"
    _write_cloud_control(queue, context)
    shutil.copytree(context.root, queue / "results/inbox" / context.task.task_id)

    imported = import_entry_quality_history_result(
        lake,
        queue,
        context.task.task_id,
        task_public_key=context.task_key.public_key(),
        worker_public_key=context.worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert imported.state == "completed"
    assert len(imported.published_rows) == 11
    assert (queue / "results/imported" / context.task.task_id).is_dir()
    assert (queue / "completed" / context.task.task_id).is_dir()
    for spec in ENTRY_QUALITY_HISTORY_OUTPUT_SPECS:
        metadata = json.loads((lake / spec.relative_path / "_research_generation.json").read_text())
        assert metadata["generation_id"] == imported.generation_id
    second = import_entry_quality_history_result(
        lake,
        queue,
        context.task.task_id,
        task_public_key=context.task_key.public_key(),
        worker_public_key=context.worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert second.idempotent is True

    # Simulate a second task whose Gold publish succeeds but the inbox archive move
    # is interrupted. The retry must finalize, not reject or republish.
    retry_context = SimpleNamespace(
        task_key=context.task_key,
        worker_key=context.worker_key,
        snapshot=context.snapshot,
        artifacts=context.artifacts,
    )
    retry_context.task = _signed_task(
        context.task_key,
        context.snapshot,
        task_id="eqh-retry",
    )
    retry_root, retry_manifest, retry_receipt = write_entry_quality_history_result_bundle(
        tmp_path / "rr",
        task=retry_context.task,
        snapshot=retry_context.snapshot,
        artifacts=retry_context.artifacts,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        worker_key_id=WORKER_KEY_ID,
        worker_signing_key=retry_context.worker_key,
        claimed_at=GENERATED_AT + timedelta(minutes=2),
        input_bytes=0,
        cache_hit_bytes=0,
        downloaded_bytes=0,
        peak_rss_bytes=123,
        compute_duration_seconds=1,
        max_result_bytes=50 * 1024**2,
    )
    retry_context.root = retry_root
    retry_context.manifest = retry_manifest
    retry_context.receipt = retry_receipt
    _write_cloud_control(queue, retry_context)
    shutil.copytree(retry_root, queue / "results/inbox" / retry_context.task.task_id)
    real_replace = os.replace
    failed = False

    def fail_archive_once(source, destination):
        nonlocal failed
        if (
            not failed
            and Path(source) == queue / "results/inbox" / retry_context.task.task_id
            and Path(destination) == queue / "results/imported" / retry_context.task.task_id
        ):
            failed = True
            raise OSError("injected archive failure")
        return real_replace(source, destination)

    with monkeypatch.context() as patcher:
        patcher.setattr(importer_module.os, "replace", fail_archive_once)
        with pytest.raises(OSError, match="archive failure"):
            import_entry_quality_history_result(
                lake,
                queue,
                retry_context.task.task_id,
                task_public_key=retry_context.task_key.public_key(),
                worker_public_key=retry_context.worker_key.public_key(),
                expected_task_key_id=TASK_KEY_ID,
                expected_worker_key_id=WORKER_KEY_ID,
                expected_quant_lab_commit=COMMIT,
            )
    status = json.loads((queue / "status" / f"{retry_context.task.task_id}.json").read_text())
    assert status["state"] == "publishing"
    assert status["import_status"] == "finalize_pending"
    recovered = import_entry_quality_history_result(
        lake,
        queue,
        retry_context.task.task_id,
        task_public_key=retry_context.task_key.public_key(),
        worker_public_key=retry_context.worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )
    assert recovered.state == "completed"
    assert (queue / "results/rejected" / retry_context.task.task_id).exists() is False


def test_validate_only_does_not_publish_or_change_queue_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    queue = ensure_research_queue_layout(tmp_path / "queue")
    lake = tmp_path / "lake"
    _write_cloud_control(queue, context)
    shutil.copytree(context.root, queue / "results/inbox" / context.task.task_id)
    status_path = queue / "status" / f"{context.task.task_id}.json"
    status_before = status_path.read_bytes()

    validated = validate_entry_quality_history_result_for_import(
        queue,
        context.task.task_id,
        task_public_key=context.task_key.public_key(),
        worker_public_key=context.worker_key.public_key(),
        expected_task_key_id=TASK_KEY_ID,
        expected_worker_key_id=WORKER_KEY_ID,
        expected_quant_lab_commit=COMMIT,
    )

    assert validated.task_id == context.task.task_id
    assert validated.output_rows == context.receipt.output_rows
    assert validated.anti_leakage_status == "PASS"
    assert status_path.read_bytes() == status_before
    assert (queue / "running" / context.task.task_id).is_dir()
    assert (queue / "results/inbox" / context.task.task_id).is_dir()
    assert not (queue / "results/imported" / context.task.task_id).exists()
    assert not (lake / "gold").exists()


def test_importer_rejects_superseded_task_before_publish(tmp_path: Path, monkeypatch) -> None:
    context = _make_result_bundle(tmp_path, monkeypatch)
    queue = ensure_research_queue_layout(tmp_path / "queue")
    lake = tmp_path / "lake"
    _write_cloud_control(queue, context)
    shutil.copytree(context.root, queue / "results/inbox" / context.task.task_id)
    newer = _signed_task(
        context.task_key,
        context.snapshot,
        task_id="entry-quality-history-task-newer",
        requested_at=context.task.requested_at + timedelta(hours=1),
    )
    newer_dir = queue / "pending" / newer.task_id
    newer_dir.mkdir(parents=True)
    (newer_dir / "task.json").write_text(newer.model_dump_json(indent=2))
    with pytest.raises(ValueError, match="superseded"):
        import_entry_quality_history_result(
            lake,
            queue,
            context.task.task_id,
            task_public_key=context.task_key.public_key(),
            worker_public_key=context.worker_key.public_key(),
            expected_task_key_id=TASK_KEY_ID,
            expected_worker_key_id=WORKER_KEY_ID,
            expected_quant_lab_commit=COMMIT,
        )
    assert (queue / "results/rejected" / context.task.task_id).is_dir()
    assert not (lake / "gold/entry_quality_history_generation.json").exists()


@pytest.mark.parametrize(
    "worker_state",
    [ResearchTaskState.COMPUTING, ResearchTaskState.VALIDATING_ON_CLOUD],
)
def test_expired_worker_lease_requeues_without_racing_live_status(
    tmp_path: Path,
    monkeypatch,
    worker_state: ResearchTaskState,
) -> None:
    config = Config(
        cloud_host="cloud",
        cloud_user="worker",
        cloud_port=22,
        ssh_key_path=tmp_path / "ssh",
        known_hosts_path=tmp_path / "known_hosts",
        cloud_queue_root="/queue",
        data_root=tmp_path / "data",
        task_public_key_path=tmp_path / "task.pub",
        task_key_id=TASK_KEY_ID,
        worker_signing_key_path=tmp_path / "worker.key",
        worker_key_id=WORKER_KEY_ID,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        run_once=True,
        poll_seconds=30,
        heartbeat_seconds=30,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1,
        max_result_bytes=1,
        heavy_job_lock=tmp_path / "heavy.lock",
        batch_fetch_workers=1,
    )
    task_id = "entry-quality-history-lease"
    now = GENERATED_AT + timedelta(hours=2)
    status = ResearchTaskStatus(
        task_id=task_id,
        snapshot_id="entry-quality-history-snapshot-test",
        start_date=date(2026, 6, 19),
        end_date=date(2026, 7, 18),
        mode="recent_30d",
        cost_mode="conservative",
        state=worker_state,
        worker_id=config.worker_id,
        requested_at=GENERATED_AT,
        claimed_at=GENERATED_AT,
        heartbeat_at=GENERATED_AT,
        lease_expires_at=GENERATED_AT + timedelta(minutes=10),
        attempt=1,
        max_attempts=3,
    )

    def fake_ssh(_config, command: str, *, check: bool = True):
        assert "find" in command
        return subprocess.CompletedProcess([], 0, task_id + "\n", "")

    def fake_read(_config, _task_id, work):
        path = work / "status.previous.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(status.model_dump_json(indent=2))
        return status

    uploaded = []
    monkeypatch.setattr("quant_lab.research_worker.runner._ssh", fake_ssh)
    monkeypatch.setattr("quant_lab.research_worker.runner._read_local_or_remote_status", fake_read)
    monkeypatch.setattr("quant_lab.research_worker.runner._remote_exists", lambda *_args: False)
    monkeypatch.setattr(
        "quant_lab.research_worker.runner._read_local_or_remote_lease",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        "quant_lab.research_worker.runner._conditional_remote_transition",
        lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        "quant_lab.research_worker.runner._upload_status",
        lambda _config, value, _work: uploaded.append(value),
    )
    assert recover_expired_leases(config, now=now) == 1
    assert uploaded[-1].state == ResearchTaskState.PENDING
    assert uploaded[-1].last_error == "LEASE_EXPIRED"


def test_status_upload_uses_unique_remote_temporary_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _research_worker_config(tmp_path)
    status = ResearchTaskStatus(
        task_id="entry-quality-history-status-upload",
        snapshot_id="entry-quality-history-snapshot-test",
        start_date=date(2026, 6, 19),
        end_date=date(2026, 7, 18),
        mode="recent_30d",
        cost_mode="conservative",
        state=ResearchTaskState.COMPUTING,
        worker_id=config.worker_id,
        requested_at=GENERATED_AT,
        claimed_at=GENERATED_AT,
        heartbeat_at=GENERATED_AT,
        lease_expires_at=GENERATED_AT + timedelta(minutes=10),
        attempt=1,
        max_attempts=3,
    )
    remote_paths: list[str] = []
    commands: list[str] = []

    def fake_scp(_config, local_path: Path, remote_path: str, **_kwargs) -> None:
        assert local_path.read_text("utf-8")
        assert not runner_module._STATUS_UPLOAD_LOCK._is_owned()
        remote_paths.append(remote_path)

    def fake_ssh(_config, command: str, *, check: bool = True):
        commands.append(command)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(runner_module, "_scp_to", fake_scp)
    monkeypatch.setattr(runner_module, "_ssh", fake_ssh)

    runner_module._upload_status(config, status, tmp_path)
    runner_module._upload_status(config, status, tmp_path)

    assert len(set(remote_paths)) == 2
    assert all(path.endswith(".tmp") for path in remote_paths)
    final_path = f"{config.cloud_queue_root}/status/{status.task_id}.json"
    assert all(final_path in command for command in commands)


def test_worker_handoff_exposes_inbox_before_cloud_status_ownership(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _research_worker_config(tmp_path)
    key = Ed25519PrivateKey.generate()
    snapshot = _signed_snapshot(key)
    task = _signed_task(key, snapshot)
    status = ResearchTaskStatus(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        start_date=task.start_date,
        end_date=task.end_date,
        mode=task.mode,
        cost_mode=task.cost_mode,
        state=ResearchTaskState.UPLOADING,
        requested_at=task.requested_at,
        attempt=1,
        max_attempts=task.max_attempts,
    )
    events: list[str] = []
    monkeypatch.setattr(
        runner_module,
        "_upload_result_partial",
        lambda *_args: events.append("partial_uploaded") or "/queue/results/inbox/.partial",
    )
    monkeypatch.setattr(
        runner_module,
        "_mark_result_partial_ready",
        lambda *_args: events.append("partial_ready"),
    )
    monkeypatch.setattr(
        runner_module,
        "_stop_heartbeat",
        lambda *_args: events.append("heartbeat_stopped"),
    )
    monkeypatch.setattr(
        runner_module,
        "_upload_status",
        lambda *_args: pytest.fail("worker must not publish cloud-owned state"),
    )
    monkeypatch.setattr(
        runner_module,
        "_finalize_result_upload",
        lambda *_args: events.append("inbox_visible"),
    )

    result = runner_module._handoff_result_to_cloud(
        config,
        task,
        status,
        tmp_path,
        tmp_path / "result",
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert result.state == ResearchTaskState.VALIDATING_ON_CLOUD
    assert events == [
        "partial_uploaded",
        "partial_ready",
        "heartbeat_stopped",
        "inbox_visible",
    ]


def test_worker_heartbeat_updates_only_monotonic_lease(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _research_worker_config(tmp_path)
    key = Ed25519PrivateKey.generate()
    snapshot = _signed_snapshot(key)
    task = _signed_task(key, snapshot)
    claimed_at = GENERATED_AT
    lease = ResearchTaskLease(
        task_id=task.task_id,
        snapshot_id=task.snapshot_id,
        worker_id=config.worker_id,
        claimed_at=claimed_at,
        heartbeat_at=claimed_at,
        lease_expires_at=claimed_at + timedelta(seconds=task.lease_seconds),
    )
    uploaded: list[ResearchTaskLease] = []

    class OneHeartbeat:
        calls = 0

        def wait(self, _seconds: int) -> bool:
            self.calls += 1
            return self.calls > 1

    monkeypatch.setattr(
        runner_module,
        "_upload_lease",
        lambda _config, value, _work: uploaded.append(value),
    )
    monkeypatch.setattr(
        runner_module,
        "_upload_status",
        lambda *_args: pytest.fail("heartbeat must not rewrite business status"),
    )

    runner_module._heartbeat_loop(config, task, lease, tmp_path, OneHeartbeat())

    assert len(uploaded) == 1
    assert uploaded[0].sequence == 1
    assert uploaded[0].heartbeat_at > lease.heartbeat_at


def test_worker_recovers_complete_hidden_result_before_lease_requeue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _research_worker_config(tmp_path)
    task_id = "entry-quality-history-handoff-recovery"
    partial = f"{config.cloud_queue_root}/results/inbox/.{task_id}.abc.partial"
    inbox = f"{config.cloud_queue_root}/results/inbox/{task_id}"
    visible = False

    def fake_exists(_config, path: str) -> bool:
        if path == inbox:
            return visible
        return path == f"{partial}/{runner_module._HANDOFF_READY_MARKER}"

    def fake_finalize(_config, _task_id: str, _partial: str) -> None:
        nonlocal visible
        assert _task_id == task_id
        assert _partial == partial
        visible = True

    monkeypatch.setattr(runner_module, "_remote_exists", fake_exists)
    monkeypatch.setattr(runner_module, "_list_result_partials", lambda *_args: [partial])
    monkeypatch.setattr(runner_module, "_finalize_result_upload", fake_finalize)

    assert runner_module._recover_handoff_visibility(config, task_id) is True
    assert visible is True


def test_worker_failure_does_not_regress_cloud_owned_inbox(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _research_worker_config(tmp_path)
    monkeypatch.setattr(runner_module, "_remote_exists", lambda *_args: True)
    monkeypatch.setattr(
        runner_module,
        "_upload_status",
        lambda *_args: pytest.fail("worker must not overwrite cloud-owned status"),
    )
    monkeypatch.setattr(
        runner_module,
        "_ssh",
        lambda *_args, **_kwargs: pytest.fail("worker must not move cloud-owned task"),
    )
    runner_module._handle_failure(config, "entry-quality-history-owned", RuntimeError("late"))

    monkeypatch.setattr(
        runner_module,
        "_remote_exists",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("network uncertain")),
    )
    runner_module._handle_failure(
        config,
        "entry-quality-history-uncertain",
        RuntimeError("late"),
    )


def test_worker_ssh_and_scp_time_out_explicitly(tmp_path: Path, monkeypatch) -> None:
    config = _research_worker_config(tmp_path)
    monkeypatch.setattr(
        runner_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 1))
        ),
    )
    with pytest.raises(RuntimeError, match="ssh_timeout"):
        runner_module._ssh(config, "true")
    with pytest.raises(RuntimeError, match="scp_from_timeout"):
        runner_module._scp_from(config, "/remote", tmp_path / "local")
    with pytest.raises(RuntimeError, match="scp_to_timeout"):
        runner_module._scp_to(config, tmp_path / "local", "/remote")


def test_research_plane_deployment_examples_share_keys_paths_and_memory_limit() -> None:
    root = Path(__file__).resolve().parents[1]
    request_unit = (
        root / "deploy/systemd/quant-lab-entry-quality-history-request.service"
    ).read_text()
    cloud_env = (root / "deploy/systemd/research-plane.env.example").read_text()
    nas_env = (root / "deploy/nas_research_worker/.env.example").read_text()
    tmpfiles = (root / "deploy/tmpfiles.d/quant-lab-research-plane.conf").read_text()
    permission_script = (
        root / "deploy/scripts/upgrade_research_queue_permissions.sh"
    ).read_text()

    assert "/var/lib/quant-lab/lake/bronze/lake_file_index" in request_unit
    assert "/var/lib/quant-lab/lake/ops/lake_file_index" not in request_unit
    read_write_line = next(
        line for line in request_unit.splitlines() if line.startswith("ReadWritePaths=")
    )
    assert read_write_line.split() == [
        "ReadWritePaths=/var/lib/quant-lab/research_queue",
        "/var/lib/quant-lab/lake/bronze/lake_file_index",
    ]
    assert "/var/lib/quant-lab/lake/bronze/lake_file_index" in tmpfiles
    assert (
        "d /var/lib/quant-lab/research_queue 2770 quantlab quant-research -"
        in tmpfiles
    )
    assert "research_queue/lease 2770 quantlab quant-research" in tmpfiles
    for relative in (
        "requests/pending",
        "requests/processing",
        "requests/completed",
        "requests/failed",
        "results/imported",
    ):
        assert f"research_queue/{relative} 2770 quantlab quant-research" in tmpfiles
        assert relative in permission_script
    assert "groupadd --system" in permission_script
    assert "useradd" in permission_script
    assert "chmod 2770" in permission_script
    assert "chmod 0660" in permission_script
    assert (
        'find "${queue_root}" -xdev -path "${queue_root}/snapshots" -prune '
        "-o -type d"
    ) in permission_script
    assert (
        'find "${queue_root}" -xdev -path "${queue_root}/snapshots" -prune '
        "-o -type f"
    ) in permission_script
    assert (
        'find "${queue_root}/snapshots" -xdev -mindepth 1 '
        '\\\n  -exec chown -h "${service_user}:${research_group}" {} +'
    ) in permission_script
    assert "chmod 777" not in permission_script
    assert "QUANT_LAB_RESEARCH_TASK_KEY_ID=cloud-research-v1" in cloud_env
    assert "QUANT_RESEARCH_TASK_KEY_ID=cloud-research-v1" in nas_env
    assert "QUANT_LAB_RESEARCH_WORKER_KEY_ID=nas-research-v1" in cloud_env
    assert "QUANT_RESEARCH_WORKER_KEY_ID=nas-research-v1" in nas_env
    assert f"QUANT_LAB_RESEARCH_MAX_RESULT_BYTES={DEFAULT_RESEARCH_MAX_RESULT_BYTES}" in cloud_env
    assert f"MAX_RESULT_BYTES={DEFAULT_RESEARCH_MAX_RESULT_BYTES}" in nas_env


def test_queue_permission_upgrade_preserves_sealed_snapshot_modes(tmp_path: Path) -> None:
    if os.name != "posix":
        pytest.skip("permission-mode integration test requires a POSIX host")

    import grp
    import pwd

    root = Path(__file__).resolve().parents[1]
    queue_root = tmp_path / "research_queue"
    sealed_dir = queue_root / "snapshots" / "snapshot-1"
    sealed_file = sealed_dir / "manifest.json"
    legacy_dir = queue_root / "pending" / "task-1"
    legacy_file = legacy_dir / "task.json"
    sealed_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    sealed_file.write_text("{}")
    legacy_file.write_text("{}")
    sealed_dir.chmod(0o550)
    sealed_file.chmod(0o440)
    legacy_dir.chmod(0o700)
    legacy_file.chmod(0o600)

    user = pwd.getpwuid(os.getuid()).pw_name
    group = grp.getgrgid(os.getgid()).gr_name
    env = {
        **os.environ,
        "QUANT_LAB_RESEARCH_QUEUE_ROOT": str(queue_root),
        "QUANT_LAB_SERVICE_USER": user,
        "QUANT_LAB_RESEARCH_USER": user,
        "QUANT_LAB_RESEARCH_GROUP": group,
    }
    subprocess.run(
        ["bash", str(root / "deploy/scripts/upgrade_research_queue_permissions.sh")],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )

    assert sealed_dir.stat().st_mode & 0o7777 == 0o550
    assert sealed_file.stat().st_mode & 0o7777 == 0o440
    assert legacy_dir.stat().st_mode & 0o7777 == 0o2770
    assert legacy_file.stat().st_mode & 0o7777 == 0o660


def test_run_once_returns_nonzero_when_claimed_task_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _research_worker_config(tmp_path)
    handled: list[tuple[str, Exception]] = []
    task_id = "entry-quality-history-run-once-failure"
    runner_module.STOP.clear()
    monkeypatch.setattr(runner_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        runner_module.Config,
        "from_env",
        classmethod(lambda _cls: config),
    )
    monkeypatch.setattr(runner_module, "_validate_config", lambda _config: None)
    monkeypatch.setattr(runner_module, "recover_expired_leases", lambda _config: 0)
    monkeypatch.setattr(runner_module, "claim_next_task", lambda _config: task_id)
    monkeypatch.setattr(
        runner_module,
        "process_claimed_task",
        lambda _config, _task_id: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        runner_module,
        "_handle_failure",
        lambda _config, value, exc: handled.append((value, exc)),
    )

    try:
        assert runner_module.main() == 1
    finally:
        runner_module.STOP.clear()

    assert handled and handled[0][0] == task_id
    assert isinstance(handled[0][1], RuntimeError)


def _research_worker_config(tmp_path: Path) -> Config:
    return Config(
        cloud_host="cloud",
        cloud_user="worker",
        cloud_port=22,
        ssh_key_path=tmp_path / "ssh",
        known_hosts_path=tmp_path / "known_hosts",
        cloud_queue_root="/queue",
        data_root=tmp_path / "data",
        task_public_key_path=tmp_path / "task.pub",
        task_key_id=TASK_KEY_ID,
        worker_signing_key_path=tmp_path / "worker.key",
        worker_key_id=WORKER_KEY_ID,
        worker_id="nas-research-worker-01",
        worker_commit=COMMIT,
        run_once=True,
        poll_seconds=30,
        heartbeat_seconds=30,
        min_free_disk_bytes=0,
        max_snapshot_bytes=1,
        max_result_bytes=1,
        heavy_job_lock=tmp_path / "heavy.lock",
        batch_fetch_workers=1,
    )


def _history_input_frames() -> dict[str, pl.DataFrame]:
    market_rows = []
    start = datetime(2026, 4, 30, tzinfo=UTC)
    for symbol in ("SOL-USDT", "BTC-USDT"):
        for hour in range(15 * 24):
            ts = start + timedelta(hours=hour)
            market_rows.append(
                {
                    "symbol": symbol,
                    "timeframe": "1H",
                    "ts": ts,
                    "open": 100.0,
                    "high": 110.0,
                    "low": 90.0,
                    "close": 100.0,
                }
            )
    trades = pl.DataFrame(
        [
            {
                "run_id": "old",
                "trade_id": "trade-old",
                "ts_utc": datetime(2026, 5, 2, 20, tzinfo=UTC),
                "symbol": "SOL-USDT",
                "side": "buy",
                "action": "entry",
                "price": 108.0,
                "realized_net_bps": -10.0,
            },
            {
                "run_id": "recent",
                "trade_id": "trade-recent",
                "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                "symbol": "SOL-USDT",
                "side": "buy",
                "action": "entry",
                "price": 108.0,
                "realized_net_bps": 20.0,
            },
        ]
    )
    candidates = pl.DataFrame(
        [
            {
                "candidate_id": "candidate-recent",
                "run_id": "recent",
                "ts_utc": datetime(2026, 5, 10, 20, tzinfo=UTC),
                "symbol": "SOL-USDT",
                "entry_close": 108.0,
                "regime_state": "normal",
                "risk_level": "normal",
                "f4_volume_expansion": 0.0,
                "f5_rsi_trend_confirm": 0.0,
                "estimated_spread_bps": 2.0,
            }
        ]
    )
    labels = pl.DataFrame(
        [
            {
                "candidate_id": "candidate-recent",
                "decision_ts": datetime(2026, 5, 10, 21, tzinfo=UTC),
                "label_ts": datetime(2026, 5, 11, 20, tzinfo=UTC),
                "horizon_hours": 24,
                "net_bps_after_cost": 15.0,
                "label_status": "complete",
            }
        ]
    )
    return {
        "trades": trades,
        "lifecycles": pl.DataFrame(),
        "market_bars": pl.DataFrame(market_rows),
        "candidates": candidates,
        "labels": labels,
        "costs": pl.DataFrame(
            [
                {
                    "symbol": "SOL-USDT",
                    "as_of_date": "2026-05-10",
                    "roundtrip_all_in_cost_bps": 35.0,
                }
            ]
        ),
    }


def test_candidate_label_identity_accepts_first_hourly_decision_bar() -> None:
    frames = _history_input_frames()

    artifacts = compute_entry_quality_history(
        **frames,
        start_date="2026-05-01",
        end_date="2026-05-10",
        mode="recent_30d",
        cost_mode="conservative",
        generated_at=GENERATED_AT,
        generated_from_bundle_id=BUNDLE_ID,
    )

    check = artifacts.anti_leakage_check.filter(
        pl.col("check_name") == "candidate_label_identity"
    ).to_dicts()[0]
    assert check["status"] == "PASS"
    assert check["violation_count"] == 0


def test_candidate_label_identity_rejects_noncausal_decision_bar() -> None:
    frames = _history_input_frames()
    frames["labels"] = frames["labels"].with_columns(
        pl.lit(datetime(2026, 5, 10, 20, 30, tzinfo=UTC)).alias("decision_ts")
    )

    artifacts = compute_entry_quality_history(
        **frames,
        start_date="2026-05-01",
        end_date="2026-05-10",
        mode="recent_30d",
        cost_mode="conservative",
        generated_at=GENERATED_AT,
        generated_from_bundle_id=BUNDLE_ID,
    )

    check = artifacts.anti_leakage_check.filter(
        pl.col("check_name") == "candidate_label_identity"
    ).to_dicts()[0]
    assert check["status"] == "FAIL"
    assert check["violation_count"] == 1


def test_candidate_label_identity_uses_normalized_symbol_fallback() -> None:
    frames = _history_input_frames()
    frames["candidates"] = frames["candidates"].drop("symbol").with_columns(
        pl.lit("SOL-USDT").alias("normalized_symbol")
    )

    artifacts = compute_entry_quality_history(
        **frames,
        start_date="2026-05-01",
        end_date="2026-05-10",
        mode="recent_30d",
        cost_mode="conservative",
        generated_at=GENERATED_AT,
        generated_from_bundle_id=BUNDLE_ID,
    )

    check = artifacts.anti_leakage_check.filter(
        pl.col("check_name") == "candidate_label_identity"
    ).to_dicts()[0]
    assert check["status"] == "PASS"
    assert check["violation_count"] == 0


def _normalize_generated_at(frame: pl.DataFrame) -> pl.DataFrame:
    if "generated_at_utc" in frame.columns:
        frame = frame.drop("generated_at_utc")
    keys = [
        column
        for column in (
            "start_date",
            "end_date",
            "window_mode",
            "cost_mode",
            "source_event_key",
            "group_key",
            "threshold_bps",
            "horizon_hours",
            "check_name",
        )
        if column in frame.columns
    ]
    return frame.sort(keys) if keys and not frame.is_empty() else frame
