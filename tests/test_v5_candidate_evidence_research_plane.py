from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from polars.testing import assert_frame_equal
from typer.testing import CliRunner

from quant_lab import cli as cli_module
from quant_lab.cli import app
from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.candidate_labels import LABEL_SCHEMA, build_and_publish_candidate_labels
from quant_lab.research.strategy_evidence import (
    SAMPLE_SCHEMA,
    SUMMARY_SCHEMA,
    build_and_publish_strategy_evidence,
)
from quant_lab.research_plane import importer as importer_module
from quant_lab.research_plane import queue as queue_module
from quant_lab.research_plane import v5_candidate_evidence_result as result_module
from quant_lab.research_plane.importer import (
    import_entry_quality_history_result,
    validate_entry_quality_history_result_for_import,
)
from quant_lab.research_plane.queue import (
    V5CandidateEvidenceTaskRequestResult,
    create_v5_candidate_evidence_task,
)
from quant_lab.research_plane.result import validate_research_task_snapshot
from quant_lab.research_plane.signatures import sign_model
from quant_lab.research_plane.snapshot_gc import release_snapshot_payload
from quant_lab.research_plane.status import research_plane_status
from quant_lab.research_plane.v5_candidate_evidence_contracts import (
    V5_CANDIDATE_EVIDENCE_HORIZONS,
    V5CandidateEvidenceSnapshotManifest,
    V5CandidateEvidenceTask,
    V5CandidateEvidenceTaskPayload,
)
from quant_lab.research_plane.v5_candidate_evidence_publish import (
    V5_CANDIDATE_EVIDENCE_DATASETS,
    V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS,
    publish_v5_candidate_evidence_generation,
    verify_v5_candidate_evidence_generation_fast,
)
from quant_lab.research_plane.v5_candidate_evidence_snapshot import (
    cleanup_stale_v5_candidate_evidence_rehydrate_partials,
    materialize_v5_candidate_evidence_snapshot,
    preflight_v5_candidate_evidence_snapshot,
    rehydrate_v5_candidate_evidence_snapshot_payload,
    verify_v5_candidate_evidence_snapshot_manifest,
)
from quant_lab.research_worker import runner as runner_module
from quant_lab.research_worker.v5_candidate_evidence import (
    V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS,
    compute_v5_candidate_evidence_result,
)
from quant_lab.research_worker.v5_candidate_evidence_result_writer import (
    write_v5_candidate_evidence_result_bundle,
)

COMMIT = "a" * 40
CLI_RUNNER = CliRunner()


def test_v5_candidate_evidence_contract_is_strict() -> None:
    payload = V5CandidateEvidenceTaskPayload(as_of_date=date(2026, 7, 10))

    assert payload.mode == "incremental"
    assert payload.lookback_days == 8
    assert payload.horizon_hours == V5_CANDIDATE_EVIDENCE_HORIZONS
    assert payload.include_historical_outcomes is False

    with pytest.raises(ValueError, match="horizons must match"):
        V5CandidateEvidenceTaskPayload(
            as_of_date=date(2026, 7, 10),
            horizon_hours=(4, 8, 12),
        )
    with pytest.raises(ValueError):
        V5CandidateEvidenceTaskPayload.model_validate(
            {
                "as_of_date": "2026-07-10",
                "include_historical_outcomes": True,
            }
        )
    with pytest.raises(ValueError):
        V5CandidateEvidenceTaskPayload.model_validate(
            {"as_of_date": "2026-07-10", "unknown": "rejected"}
        )
    with pytest.raises(ValueError):
        V5CandidateEvidenceTaskPayload.model_validate(
            {"as_of_date": "2026-07-10", "lookback_days": 7}
        )


def test_worker_gate_skips_v5_candidate_evidence_before_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands: list[str] = []

    def fake_ssh(_config: object, command: str, *, check: bool = True) -> object:
        commands.append(command)
        return subprocess.CompletedProcess([], 44, "", "")

    monkeypatch.setattr(runner_module, "_ssh", fake_ssh)
    config = SimpleNamespace(
        cloud_queue_root="/queue",
        worker_id="nas-research-worker-01",
        factor_factory_enabled=False,
        v5_candidate_evidence_enabled=False,
    )
    assert runner_module.claim_next_task(config) is None
    assert "allow_v5_candidate_evidence=0" in commands[0]
    assert '"v5_candidate_evidence"' in commands[0]


def test_v5_candidate_evidence_cli_no_change_is_success_and_local_fallback_is_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    lake.mkdir()
    queue.mkdir()
    key = Ed25519PrivateKey.generate()
    key_path = tmp_path / "task-key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setenv("QUANT_LAB_NAS_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("QUANT_LAB_NAS_V5_CANDIDATE_EVIDENCE_ENABLED", "1")
    monkeypatch.setattr(
        cli_module,
        "create_v5_candidate_evidence_task",
        lambda *_args, **_kwargs: V5CandidateEvidenceTaskRequestResult(
            state="already_current",
            task_created=False,
            snapshot_materialized=False,
            current_generation_id="v5-generation-current",
            reason="v5_candidate_evidence_inputs_unchanged",
            snapshot_id="v5-snapshot-current",
            fingerprint_matches_generation=True,
        ),
    )
    result = CLI_RUNNER.invoke(
        app,
        [
            "request-v5-candidate-evidence",
            "--lake-root",
            str(lake),
            "--queue-root",
            str(queue),
            "--signing-key-path",
            str(key_path),
            "--key-id",
            "test-cloud-key",
            "--quant-lab-commit",
            COMMIT,
            "--date",
            "2026-07-10",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "V5_CANDIDATE_EVIDENCE_ALREADY_CURRENT" in result.output
    assert '"state": "already_current"' in result.output

    monkeypatch.delenv("QUANT_LAB_LOCAL_V5_CANDIDATE_EVIDENCE_ENABLED", raising=False)
    local = CLI_RUNNER.invoke(
        app,
        [
            "build-v5-candidate-labels",
            "--lake-root",
            str(lake),
        ],
    )
    assert local.exit_code != 0
    assert "local fallback disabled" in local.output


def test_v5_candidate_evidence_fingerprint_is_projection_scoped(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)

    original = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    repeated = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )

    assert repeated.snapshot_id == original.snapshot_id
    assert repeated.input_fingerprint.input_fingerprint_digest == (
        original.input_fingerprint.input_fingerprint_digest
    )
    assert original.candidate_symbols == ("BTC-USDT-SWAP",)
    assert original.candidate_run_ids == ("run-1",)
    assert original.selected_timeframes == (("BTC-USDT-SWAP", "1H"),)
    assert original.market_window_start == datetime(2026, 7, 9, 0, 30, tzinfo=UTC)

    _write_rows(
        lake / "silver" / "v5_candidate_event" / "unrelated.parquet",
        [
            {
                "strategy": "v5",
                "candidate_id": "outside-window",
                "run_id": "run-outside",
                "ts_utc": datetime(2026, 6, 1, tzinfo=UTC),
                "symbol": "ETH-USDT-SWAP",
            }
        ],
    )
    _write_rows(
        lake / "silver" / "market_bar" / "unrelated.parquet",
        [
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="15m",
                ts=datetime(2026, 7, 9, 2, tzinfo=UTC),
                close=101.0,
            ),
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 3, tzinfo=UTC),
                close=102.0,
                is_closed=False,
            ),
        ],
    )
    unrelated = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    assert unrelated.snapshot_id == original.snapshot_id
    assert unrelated.input_fingerprint.input_fingerprint_digest == (
        original.input_fingerprint.input_fingerprint_digest
    )

    _write_rows(
        lake / "silver" / "v5_run_summary" / "unrelated.parquet",
        [
            {
                "run_id": "run-unrelated",
                "bundle_ts": datetime(2026, 7, 9, 0, tzinfo=UTC),
                "block_reason": "unrelated",
            }
        ],
    )
    unrelated_run = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    assert unrelated_run.snapshot_id == original.snapshot_id

    relevant_event_path = (
        lake / "silver" / "v5_candidate_event" / "new-relevant-event.parquet"
    )
    _write_rows(
        relevant_event_path,
        [
            {
                "strategy": "v5",
                "candidate_id": "candidate-2",
                "run_id": "run-1",
                "ts_utc": datetime(2026, 7, 9, 0, 45, tzinfo=UTC),
                "symbol": "BTC-USDT-SWAP",
                "strategy_candidate": "v5.alt_impulse_shadow",
                "cost_bps": 8.0,
            }
        ],
    )
    event_change = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    assert event_change.input_fingerprint.candidate_event_digest != (
        original.input_fingerprint.candidate_event_digest
    )
    relevant_event_path.unlink()

    parameter_change = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 11),
        quant_lab_commit=COMMIT,
    )
    assert parameter_change.snapshot_id != original.snapshot_id

    _write_rows(
        lake / "silver" / "market_bar" / "relevant.parquet",
        [
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 3, tzinfo=UTC),
                close=102.0,
            )
        ],
    )
    relevant = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    assert relevant.snapshot_id != original.snapshot_id
    assert relevant.input_fingerprint.market_bar_digest != (
        original.input_fingerprint.market_bar_digest
    )

    commit_change = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit="b" * 40,
    )
    assert commit_change.snapshot_id != relevant.snapshot_id

    _write_rows(
        lake / "silver" / "v5_run_summary" / "runs.parquet",
        [
            {
                "run_id": "run-1",
                "bundle_ts": datetime(2026, 7, 9, 0, tzinfo=UTC),
                "block_reason": "changed",
            }
        ],
    )
    run_change = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    assert run_change.input_fingerprint.run_summary_digest != (
        relevant.input_fingerprint.run_summary_digest
    )


def test_v5_candidate_evidence_snapshot_is_signed_and_immutable(tmp_path: Path) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    private_key = Ed25519PrivateKey.generate()
    preflight = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )

    created = materialize_v5_candidate_evidence_snapshot(
        preflight,
        signing_key=private_key,
        signature_key_id="test-cloud-key",
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    root = queue / "snapshots" / preflight.snapshot_id

    assert created.snapshot_materialized is True
    assert created.snapshot_rehydrated is False
    assert {item.dataset_name for item in created.manifest.files} == {
        "silver/v5_candidate_event",
        "silver/market_bar",
        "silver/v5_run_summary",
    }
    verify_v5_candidate_evidence_snapshot_manifest(
        created.manifest,
        final_root=root,
        public_key=private_key.public_key(),
    )

    reused = materialize_v5_candidate_evidence_snapshot(
        preflight,
        signing_key=private_key,
        signature_key_id="test-cloud-key",
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    assert reused.snapshot_materialized is False

    extra = root / "files" / "silver" / "market_bar" / "unexpected.parquet"
    # The production snapshot is deliberately sealed read-only. Temporarily
    # unlock only this leaf so the test can simulate an out-of-band mutation
    # on POSIX as well as Windows, then restore the sealed permissions.
    extra.parent.chmod(0o750)
    try:
        pl.DataFrame({"value": [1]}).write_parquet(extra)
        extra.chmod(0o440)
    finally:
        extra.parent.chmod(0o550)
    with pytest.raises(ValueError, match="file_set_mismatch"):
        verify_v5_candidate_evidence_snapshot_manifest(
            created.manifest,
            final_root=root,
            public_key=private_key.public_key(),
        )


def test_v5_candidate_evidence_task_signature_and_snapshot_bindings_are_strict(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    key = Ed25519PrivateKey.generate()
    created = create_v5_candidate_evidence_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        signing_key=key,
        signature_key_id="test-cloud-key",
        quant_lab_commit=COMMIT,
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    assert created.task is not None
    task = created.task
    snapshot_root = queue / "snapshots" / created.snapshot_id
    snapshot = V5CandidateEvidenceSnapshotManifest.model_validate_json(
        (snapshot_root / "manifest.json").read_text("utf-8")
    )
    validate_research_task_snapshot(
        task,
        snapshot,
        task_public_key=key.public_key(),
        expected_key_id="test-cloud-key",
        expected_quant_lab_commit=COMMIT,
        snapshot_root=snapshot_root,
    )

    bad_signature = task.model_copy(update={"signature": "invalid"})
    with pytest.raises(ValueError, match="signature"):
        validate_research_task_snapshot(
            bad_signature,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id="test-cloud-key",
        )
    wrong_snapshot_unsigned = task.model_copy(
        update={"snapshot_id": "v5-candidate-evidence-other", "signature": "pending"}
    )
    wrong_snapshot = wrong_snapshot_unsigned.model_copy(
        update={"signature": sign_model(wrong_snapshot_unsigned, key)}
    )
    with pytest.raises(ValueError, match="snapshot_id_mismatch"):
        validate_research_task_snapshot(
            wrong_snapshot,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id="test-cloud-key",
        )
    wrong_fingerprint_unsigned = task.model_copy(
        update={"input_fingerprint_digest": "b" * 64, "signature": "pending"}
    )
    wrong_fingerprint = wrong_fingerprint_unsigned.model_copy(
        update={"signature": sign_model(wrong_fingerprint_unsigned, key)}
    )
    with pytest.raises(ValueError, match="identity_mismatch"):
        validate_research_task_snapshot(
            wrong_fingerprint,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id="test-cloud-key",
        )
    with pytest.raises(ValueError, match="current_commit_mismatch"):
        validate_research_task_snapshot(
            task,
            snapshot,
            task_public_key=key.public_key(),
            expected_key_id="test-cloud-key",
            expected_quant_lab_commit="b" * 40,
        )


def test_v5_candidate_evidence_rehydrate_preserves_manifest_and_signature(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    private_key = Ed25519PrivateKey.generate()
    preflight = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    created = materialize_v5_candidate_evidence_snapshot(
        preflight,
        signing_key=private_key,
        signature_key_id="test-cloud-key",
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    root = queue / "snapshots" / preflight.snapshot_id
    manifest_bytes = (root / "manifest.json").read_bytes()
    signature = created.manifest.signature

    assert release_snapshot_payload(queue, preflight.snapshot_id, reason="test")
    assert not (root / "files").exists()
    assert (root / "FILES_RELEASED.json").is_file()

    changed_source = lake / "silver" / "market_bar" / "rehydrate-change.parquet"
    _write_rows(
        changed_source,
        [
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 3, tzinfo=UTC),
                close=102.0,
            )
        ],
    )
    with pytest.raises(RuntimeError, match="rehydrate_identity_mismatch"):
        rehydrate_v5_candidate_evidence_snapshot_payload(
            lake,
            queue,
            preflight.snapshot_id,
            signing_key=private_key,
            signature_key_id="test-cloud-key",
            max_input_bytes=10 * 1024 * 1024,
            max_input_uncompressed_bytes=10 * 1024 * 1024,
            max_input_rows=10_000,
        )
    changed_source.unlink()

    restored = rehydrate_v5_candidate_evidence_snapshot_payload(
        lake,
        queue,
        preflight.snapshot_id,
        signing_key=private_key,
        signature_key_id="test-cloud-key",
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )

    assert restored.snapshot_materialized is True
    assert restored.snapshot_rehydrated is True
    assert restored.manifest.signature == signature
    assert (root / "manifest.json").read_bytes() == manifest_bytes
    assert (root / "files").is_dir()
    assert not (root / "FILES_RELEASED.json").exists()
    verify_v5_candidate_evidence_snapshot_manifest(
        restored.manifest,
        final_root=root,
        public_key=private_key.public_key(),
    )


def test_v5_candidate_evidence_stale_rehydrate_partial_cleanup(tmp_path: Path) -> None:
    queue = tmp_path / "queue"
    partial = queue / "snapshots" / ".rehydrate.v5-test.abandoned.partial"
    partial.mkdir(parents=True)
    (queue / "audit").mkdir(parents=True)
    (partial / "REHYDRATE.json").write_text(
        json.dumps(
            {
                "schema_version": (
                    "quant_lab_v5_candidate_evidence_snapshot_rehydrate.v1"
                ),
                "snapshot_id": "v5-test",
                "manifest_sha256": "a" * 64,
                "started_at": datetime(2026, 7, 10, tzinfo=UTC).isoformat(),
            }
        ),
        encoding="utf-8",
    )

    removed = cleanup_stale_v5_candidate_evidence_rehydrate_partials(
        queue,
        stale_after_seconds=0,
    )

    assert partial.name in removed
    assert not partial.exists()
    assert "snapshot_rehydrate_partial_cleaned" in (
        queue / "audit" / "v5_candidate_evidence_snapshot.jsonl"
    ).read_text("utf-8")


def test_v5_candidate_evidence_worker_is_symbol_staged_and_decision_free(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    event_path = lake / "silver" / "v5_candidate_event" / "events.parquet"
    pl.read_parquet(event_path).with_columns(
        pl.lit("alt_impulse_shadow").alias("strategy_candidate")
    ).write_parquet(event_path)
    private_key = Ed25519PrivateKey.generate()
    preflight = preflight_v5_candidate_evidence_snapshot(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        quant_lab_commit=COMMIT,
    )
    created = materialize_v5_candidate_evidence_snapshot(
        preflight,
        signing_key=private_key,
        signature_key_id="test-cloud-key",
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    task = V5CandidateEvidenceTask(
        task_id="v5-candidate-evidence-task-1",
        snapshot_id=created.manifest.snapshot_id,
        input_fingerprint_digest=created.manifest.input_fingerprint_digest,
        quant_lab_commit=COMMIT,
        snapshot_manifest_sha256=created.manifest.manifest_sha256,
        as_of_date=date(2026, 7, 10),
        requested_at=datetime(2026, 7, 10, 12, tzinfo=UTC),
        signature_key_id="test-cloud-key",
        signature="test-signature",
    )

    compute = compute_v5_candidate_evidence_result(
        queue / "snapshots" / created.manifest.snapshot_id,
        created.manifest,
        task,
        max_snapshot_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
        min_free_disk_bytes=0,
        work_dir=tmp_path / "worker",
    )

    labels = compute.labels.collect(LABEL_SCHEMA)
    samples = compute.samples.collect(SAMPLE_SCHEMA)
    assert labels.height == 7
    assert samples.height == 7
    assert labels.get_column("horizon_hours").sort().to_list() == list(
        V5_CANDIDATE_EVIDENCE_HORIZONS
    )
    assert labels.get_column("strategy_candidate").unique().to_list() == [
        "alt_impulse_shadow"
    ]
    assert samples.get_column("strategy_candidate").unique().to_list() == [
        "v5.alt_impulse_shadow"
    ]
    assert "decision" not in samples.columns
    assert compute.anti_leakage["status"] == "PASS"
    assert len(compute.anti_leakage["checks"]) == len(
        V5_CANDIDATE_EVIDENCE_ANTI_LEAKAGE_CHECKS
    )
    assert all(item["violation_count"] == 0 for item in compute.anti_leakage["checks"])

    worker_key = Ed25519PrivateKey.generate()
    result_root, manifest, receipt = write_v5_candidate_evidence_result_bundle(
        tmp_path / "results",
        task=task,
        snapshot=created.manifest,
        compute=compute,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        worker_key_id="test-worker-key",
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 10, 12, 1, tzinfo=UTC),
        input_bytes=created.manifest.total_input_bytes,
        cache_hit_bytes=0,
        downloaded_bytes=created.manifest.total_input_bytes,
        peak_rss_bytes=compute.peak_rss_bytes,
        compute_duration_seconds=1.0,
        max_result_bytes=10 * 1024 * 1024,
        max_result_uncompressed_bytes=10 * 1024 * 1024,
        max_partition_bytes=5 * 1024 * 1024,
        max_partition_uncompressed_bytes=5 * 1024 * 1024,
        max_file_count=100,
    )
    assert result_root.is_dir()
    assert receipt.output_rows == 14
    assert {item.dataset_name for item in manifest.outputs} == {
        "v5_candidate_label_delta",
        "strategy_evidence_sample_delta",
    }
    late_unsigned = manifest.model_copy(
        update={
            "previous_generation_id": "v5-candidate-evidence-generation-old",
            "previous_generation_digest": "b" * 64,
            "signature": "pending",
        }
    )
    late_manifest = late_unsigned.model_copy(
        update={"signature": sign_model(late_unsigned, worker_key)}
    )
    with pytest.raises(ValueError, match="task_binding_mismatch"):
        result_module.validate_v5_candidate_evidence_result_bundle(
            result_root,
            manifest=late_manifest,
            receipt=receipt,
            task=task,
            snapshot=created.manifest,
            worker_public_key=worker_key.public_key(),
            expected_worker_key_id="test-worker-key",
        )

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("global key scan ran before capacity gate")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(result_module, "_validate_unique_keys", unexpected)
    try:
        with pytest.raises(ValueError, match="result_size_limit_exceeded"):
            result_module.validate_v5_candidate_evidence_result_bundle(
                result_root,
                manifest=manifest,
                receipt=receipt,
                task=task,
                snapshot=created.manifest,
                worker_public_key=worker_key.public_key(),
                expected_worker_key_id="test-worker-key",
                max_result_bytes=1,
            )
    finally:
        monkeypatch.undo()

    validated = result_module.validate_v5_candidate_evidence_result_bundle(
        result_root,
        manifest=manifest,
        receipt=receipt,
        task=task,
        snapshot=created.manifest,
        worker_public_key=worker_key.public_key(),
        expected_worker_key_id="test-worker-key",
        max_result_bytes=10 * 1024 * 1024,
        max_result_uncompressed_bytes=10 * 1024 * 1024,
        max_partition_bytes=5 * 1024 * 1024,
        max_partition_uncompressed_bytes=5 * 1024 * 1024,
        max_file_count=100,
    )
    _seed_other_strategy_evidence_source(lake, samples)
    published = publish_v5_candidate_evidence_generation(
        lake,
        validated,
        snapshot_root=queue / "snapshots" / created.manifest.snapshot_id,
    )
    assert published["published"] is True
    verified_rows = verify_v5_candidate_evidence_generation_fast(
        lake,
        manifest.generation_id,
        expected_input_fingerprint=manifest.input_fingerprint_digest,
    )
    assert set(verified_rows) == {
        "v5_candidate_label",
        "v5_candidate_quality_daily",
        "v5_candidate_outcome_summary",
        "strategy_evidence_sample",
        "strategy_evidence",
        "strategy_evidence_quality",
    }
    shared_sample = pl.read_parquet(lake / "gold" / "strategy_evidence_sample" / "data.parquet")
    shared_summary = pl.read_parquet(lake / "gold" / "strategy_evidence" / "data.parquet")
    assert shared_sample.filter(pl.col("source") == "research.alpha_factory.test").height == 1
    assert shared_summary.filter(pl.col("source") == "research.alpha_factory.test").height == 1
    assert shared_summary.filter(
        pl.col("source") == "research.strategy_evidence.v0.1"
    ).get_column("decision").is_in(
        [
            "RESEARCH_ONLY",
            "KEEP_SHADOW",
            "REGIME_SHADOW",
            "KILL",
            "PAPER_READY",
            "LIVE_SMALL_READY",
        ]
    ).all()

    def materializer_must_not_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("snapshot materializer ran on no-change fast path")

    request_patch = pytest.MonkeyPatch()
    request_patch.setattr(
        queue_module,
        "materialize_v5_candidate_evidence_snapshot",
        materializer_must_not_run,
    )
    try:
        no_change = create_v5_candidate_evidence_task(
            lake,
            queue,
            as_of_date=date(2026, 7, 10),
            signing_key=private_key,
            signature_key_id="test-cloud-key",
            quant_lab_commit=COMMIT,
            max_input_bytes=10 * 1024 * 1024,
            max_input_uncompressed_bytes=10 * 1024 * 1024,
            max_input_rows=10_000,
        )
    finally:
        request_patch.undo()
    assert no_change.state == "already_current"
    assert no_change.task_created is False
    assert no_change.snapshot_materialized is False

    label_path = lake / "gold" / "v5_candidate_label" / "data.parquet"
    damaged = pl.read_parquet(label_path).with_columns(
        (pl.col("cost_bps") + 1.0).alias("cost_bps")
    )
    damaged.write_parquet(label_path)
    with pytest.raises(RuntimeError, match="dataset_hash_mismatch"):
        verify_v5_candidate_evidence_generation_fast(lake, manifest.generation_id)
    integrity_failure = create_v5_candidate_evidence_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        signing_key=private_key,
        signature_key_id="test-cloud-key",
        quant_lab_commit=COMMIT,
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    assert integrity_failure.state == "generation_integrity_failed"
    assert integrity_failure.task_created is False


def test_v5_candidate_evidence_six_gold_tables_match_legacy_fixed_fixture(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    legacy_lake = tmp_path / "legacy-lake"
    plane_lake = tmp_path / "plane-lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(source)
    shutil.copytree(source, legacy_lake)
    shutil.copytree(source, plane_lake)

    build_and_publish_candidate_labels(
        legacy_lake,
        as_of_date=date(2026, 7, 10),
        mode="incremental",
        lookback_days=8,
    )
    build_and_publish_strategy_evidence(
        legacy_lake,
        as_of_date="2026-07-10",
        mode="incremental",
        lookback_days=8,
        include_historical_outcomes=False,
    )

    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    created = create_v5_candidate_evidence_task(
        plane_lake,
        queue,
        as_of_date=date(2026, 7, 10),
        signing_key=task_key,
        signature_key_id="test-cloud-key",
        quant_lab_commit=COMMIT,
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    assert created.task is not None
    snapshot = V5CandidateEvidenceSnapshotManifest.model_validate_json(
        (
            queue / "snapshots" / created.snapshot_id / "manifest.json"
        ).read_text("utf-8")
    )
    compute = compute_v5_candidate_evidence_result(
        queue / "snapshots" / created.snapshot_id,
        snapshot,
        created.task,
        max_snapshot_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
        min_free_disk_bytes=0,
        work_dir=tmp_path / "worker-parity",
    )
    result_root, result_manifest, receipt = write_v5_candidate_evidence_result_bundle(
        tmp_path / "parity-results",
        task=created.task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        worker_key_id="test-worker-key",
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 10, 12, 1, tzinfo=UTC),
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=0,
        downloaded_bytes=snapshot.total_input_bytes,
        peak_rss_bytes=compute.peak_rss_bytes,
        compute_duration_seconds=1.0,
        max_result_bytes=10 * 1024 * 1024,
        max_result_uncompressed_bytes=10 * 1024 * 1024,
        max_partition_bytes=5 * 1024 * 1024,
        max_partition_uncompressed_bytes=5 * 1024 * 1024,
        max_file_count=100,
    )
    validated = result_module.validate_v5_candidate_evidence_result_bundle(
        result_root,
        manifest=result_manifest,
        receipt=receipt,
        task=created.task,
        snapshot=snapshot,
        worker_public_key=worker_key.public_key(),
        expected_worker_key_id="test-worker-key",
        max_result_bytes=10 * 1024 * 1024,
        max_result_uncompressed_bytes=10 * 1024 * 1024,
        max_partition_bytes=5 * 1024 * 1024,
        max_partition_uncompressed_bytes=5 * 1024 * 1024,
        max_file_count=100,
    )
    publish_v5_candidate_evidence_generation(
        plane_lake,
        validated,
        snapshot_root=queue / "snapshots" / created.snapshot_id,
    )

    for dataset_name, relative_path in V5_CANDIDATE_EVIDENCE_DATASETS.items():
        legacy = _normalized_parity_frame(
            read_parquet_dataset(legacy_lake / relative_path),
            V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS[dataset_name],
        )
        plane = _normalized_parity_frame(
            read_parquet_dataset(plane_lake / relative_path),
            V5_CANDIDATE_EVIDENCE_PRIMARY_KEYS[dataset_name],
        )
        assert_frame_equal(plane, legacy, check_row_order=True, check_column_order=True)


def test_v5_candidate_evidence_importer_validates_publishes_and_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    created = create_v5_candidate_evidence_task(
        lake,
        queue,
        as_of_date=date(2026, 7, 10),
        signing_key=task_key,
        signature_key_id="test-cloud-key",
        quant_lab_commit=COMMIT,
        max_input_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
    )
    assert created.task is not None
    task = created.task
    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    snapshot = created.snapshot_id
    manifest = V5CandidateEvidenceSnapshotManifest.model_validate_json(
        (queue / "snapshots" / snapshot / "manifest.json").read_text("utf-8")
    )
    compute = compute_v5_candidate_evidence_result(
        queue / "snapshots" / snapshot,
        manifest,
        task,
        max_snapshot_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
        min_free_disk_bytes=0,
        work_dir=tmp_path / "worker-import",
    )
    write_v5_candidate_evidence_result_bundle(
        queue / "results" / "inbox",
        task=task,
        snapshot=manifest,
        compute=compute,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        worker_key_id="test-worker-key",
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 10, 12, 1, tzinfo=UTC),
        input_bytes=manifest.total_input_bytes,
        cache_hit_bytes=0,
        downloaded_bytes=manifest.total_input_bytes,
        peak_rss_bytes=compute.peak_rss_bytes,
        compute_duration_seconds=1.0,
        max_result_bytes=10 * 1024 * 1024,
        max_result_uncompressed_bytes=10 * 1024 * 1024,
        max_partition_bytes=5 * 1024 * 1024,
        max_partition_uncompressed_bytes=5 * 1024 * 1024,
        max_file_count=100,
    )
    common = {
        "task_public_key": task_key.public_key(),
        "worker_public_key": worker_key.public_key(),
        "expected_task_key_id": "test-cloud-key",
        "expected_worker_key_id": "test-worker-key",
        "expected_quant_lab_commit": COMMIT,
        "v5_candidate_evidence_max_result_bytes": 10 * 1024 * 1024,
        "v5_candidate_evidence_max_result_uncompressed_bytes": 10 * 1024 * 1024,
        "v5_candidate_evidence_max_partition_bytes": 5 * 1024 * 1024,
        "v5_candidate_evidence_max_partition_uncompressed_bytes": 5 * 1024 * 1024,
        "v5_candidate_evidence_max_file_count": 100,
    }
    validation = validate_entry_quality_history_result_for_import(
        queue,
        task.task_id,
        **common,
    )
    assert validation.output_rows == 14
    assert validation.anti_leakage_status == "PASS"

    real_publish = importer_module.publish_v5_candidate_evidence_generation
    monkeypatch.setattr(
        importer_module,
        "publish_v5_candidate_evidence_generation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("retry-test")),
    )
    retry = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        **common,
    )
    assert retry.state == "publish_retry_pending"
    assert (queue / "results" / "inbox" / task.task_id).is_dir()
    assert (queue / "running" / task.task_id).is_dir()
    monkeypatch.setattr(
        importer_module,
        "publish_v5_candidate_evidence_generation",
        real_publish,
    )
    imported = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        **common,
    )
    assert imported.state == "completed"
    assert imported.idempotent is False
    assert set(imported.published_rows) == {
        "v5_candidate_label",
        "v5_candidate_quality_daily",
        "v5_candidate_outcome_summary",
        "strategy_evidence_sample",
        "strategy_evidence",
        "strategy_evidence_quality",
    }
    assert (queue / "completed" / task.task_id).is_dir()
    assert (queue / "results" / "imported" / task.task_id).is_dir()
    assert not (queue / "snapshots" / snapshot / "files").exists()
    status = research_plane_status(queue)["tasks"]["v5_candidate_evidence"]
    assert status["candidate_evidence_generation_id"] == imported.generation_id
    assert status["last_completed_at"] is not None
    assert status["generation_age_seconds"] >= 0
    assert status["input_fingerprint"] is not None
    assert status["pending_running_state"] is None

    repeated = import_entry_quality_history_result(
        lake,
        queue,
        task.task_id,
        **common,
    )
    assert repeated.idempotent is True
    assert repeated.generation_id == imported.generation_id


def test_v5_candidate_evidence_pending_requests_coalesce_and_supersede(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    key = Ed25519PrivateKey.generate()
    arguments = {
        "as_of_date": date(2026, 7, 10),
        "signing_key": key,
        "signature_key_id": "test-cloud-key",
        "quant_lab_commit": COMMIT,
        "max_input_bytes": 10 * 1024 * 1024,
        "max_input_uncompressed_bytes": 10 * 1024 * 1024,
        "max_input_rows": 10_000,
    }

    first = create_v5_candidate_evidence_task(lake, queue, **arguments)
    repeated = create_v5_candidate_evidence_task(lake, queue, **arguments)

    assert first.state == "task_created"
    assert first.task_created is True
    assert repeated.state == "coalesced"
    assert repeated.task_created is False
    assert repeated.snapshot_materialized is False
    assert repeated.task is not None and first.task is not None
    assert repeated.task.task_id == first.task.task_id

    _write_rows(
        lake / "silver" / "market_bar" / "new-closed-bar.parquet",
        [
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 3, tzinfo=UTC),
                close=102.0,
            )
        ],
    )
    successor = create_v5_candidate_evidence_task(lake, queue, **arguments)

    assert successor.state == "task_created"
    assert successor.task_created is True
    assert successor.task is not None
    assert successor.task.task_id != first.task.task_id
    assert (queue / "cancelled" / first.task.task_id).is_dir()
    assert (queue / "pending" / successor.task.task_id).is_dir()


def test_v5_candidate_evidence_running_gets_one_successor_and_late_result_is_rejected(
    tmp_path: Path,
) -> None:
    lake = tmp_path / "lake"
    queue = tmp_path / "queue"
    _seed_snapshot_sources(lake)
    task_key = Ed25519PrivateKey.generate()
    worker_key = Ed25519PrivateKey.generate()
    arguments = {
        "as_of_date": date(2026, 7, 10),
        "signing_key": task_key,
        "signature_key_id": "test-cloud-key",
        "quant_lab_commit": COMMIT,
        "max_input_bytes": 10 * 1024 * 1024,
        "max_input_uncompressed_bytes": 10 * 1024 * 1024,
        "max_input_rows": 10_000,
    }
    first = create_v5_candidate_evidence_task(lake, queue, **arguments)
    assert first.task is not None
    task = first.task
    os.replace(queue / "pending" / task.task_id, queue / "running" / task.task_id)
    snapshot = V5CandidateEvidenceSnapshotManifest.model_validate_json(
        (
            queue / "snapshots" / task.snapshot_id / "manifest.json"
        ).read_text("utf-8")
    )
    compute = compute_v5_candidate_evidence_result(
        queue / "snapshots" / task.snapshot_id,
        snapshot,
        task,
        max_snapshot_bytes=10 * 1024 * 1024,
        max_input_uncompressed_bytes=10 * 1024 * 1024,
        max_input_rows=10_000,
        min_free_disk_bytes=0,
        work_dir=tmp_path / "worker-late",
    )
    write_v5_candidate_evidence_result_bundle(
        queue / "results" / "inbox",
        task=task,
        snapshot=snapshot,
        compute=compute,
        worker_id="nas-worker-test",
        worker_commit=COMMIT,
        worker_key_id="test-worker-key",
        worker_signing_key=worker_key,
        claimed_at=datetime(2026, 7, 10, 12, 1, tzinfo=UTC),
        input_bytes=snapshot.total_input_bytes,
        cache_hit_bytes=0,
        downloaded_bytes=snapshot.total_input_bytes,
        peak_rss_bytes=compute.peak_rss_bytes,
        compute_duration_seconds=1.0,
        max_result_bytes=10 * 1024 * 1024,
        max_result_uncompressed_bytes=10 * 1024 * 1024,
        max_partition_bytes=5 * 1024 * 1024,
        max_partition_uncompressed_bytes=5 * 1024 * 1024,
        max_file_count=100,
    )

    _write_rows(
        lake / "silver" / "market_bar" / "new-closed-bar.parquet",
        [
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 3, tzinfo=UTC),
                close=102.0,
            )
        ],
    )
    successor = create_v5_candidate_evidence_task(lake, queue, **arguments)
    assert successor.task is not None
    assert (queue / "running" / task.task_id).is_dir()
    assert (queue / "pending" / successor.task.task_id).is_dir()
    assert len(list((queue / "pending").glob("*/task.json"))) == 1

    with pytest.raises(ValueError, match="superseded_by_newer_snapshot"):
        import_entry_quality_history_result(
            lake,
            queue,
            task.task_id,
            task_public_key=task_key.public_key(),
            worker_public_key=worker_key.public_key(),
            expected_task_key_id="test-cloud-key",
            expected_worker_key_id="test-worker-key",
            expected_quant_lab_commit=COMMIT,
            v5_candidate_evidence_max_result_bytes=10 * 1024 * 1024,
            v5_candidate_evidence_max_result_uncompressed_bytes=10 * 1024 * 1024,
            v5_candidate_evidence_max_partition_bytes=5 * 1024 * 1024,
            v5_candidate_evidence_max_partition_uncompressed_bytes=5 * 1024 * 1024,
            v5_candidate_evidence_max_file_count=100,
        )
    assert not (
        lake / "gold" / "v5_candidate_evidence_generation.json"
    ).exists()
    assert (queue / "results" / "rejected" / task.task_id).is_dir()
    assert (queue / "failed" / task.task_id).is_dir()


def _seed_snapshot_sources(lake: Path) -> None:
    _write_rows(
        lake / "silver" / "v5_candidate_event" / "events.parquet",
        [
            {
                "strategy": "v5",
                "candidate_id": "candidate-1",
                "run_id": "run-1",
                "ts_utc": datetime(2026, 7, 9, 0, 30, tzinfo=UTC),
                "symbol": "BTC-USDT-SWAP",
                "strategy_candidate": "v5.alt_impulse_shadow",
                "candidate_state": "ACTIVE",
                "cost_bps": 8.0,
                "cost_source": "global_default",
                "bundle_sha256": "b" * 64,
                "source_path_inside_bundle": "candidate/events.json",
            }
        ],
    )
    _write_rows(
        lake / "silver" / "market_bar" / "bars.parquet",
        [
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 1, tzinfo=UTC),
                close=100.0,
            ),
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="1H",
                ts=datetime(2026, 7, 9, 2, tzinfo=UTC),
                close=101.0,
            ),
            _market_row(
                symbol="BTC-USDT-SWAP",
                timeframe="15m",
                ts=datetime(2026, 7, 9, 1, tzinfo=UTC),
                close=99.0,
            ),
        ],
    )
    _write_rows(
        lake / "silver" / "v5_run_summary" / "runs.parquet",
        [
            {
                "run_id": "run-1",
                "bundle_ts": datetime(2026, 7, 9, 0, tzinfo=UTC),
                "block_reason": "none",
            }
        ],
    )


def _market_row(
    *,
    symbol: str,
    timeframe: str,
    ts: datetime,
    close: float,
    is_closed: bool = True,
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "ts": ts,
        "close": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "is_closed": is_closed,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(path)


def _normalized_parity_frame(
    frame: pl.DataFrame,
    primary_keys: tuple[str, ...],
) -> pl.DataFrame:
    volatile = [
        column
        for column in ("created_at", "generated_at", "updated_at", "audit_ts")
        if column in frame.columns
    ]
    normalized = frame.drop(volatile)
    sort_columns = [column for column in primary_keys if column in normalized.columns]
    return normalized.sort(sort_columns) if sort_columns else normalized


def _seed_other_strategy_evidence_source(lake: Path, samples: pl.DataFrame) -> None:
    sample = samples.head(1).with_columns(
        [
            pl.lit("alpha").alias("strategy"),
            pl.lit("alpha-candidate-1").alias("candidate_id"),
            pl.lit("alpha-run-1").alias("run_id"),
            pl.lit("alpha.strategy").alias("strategy_candidate"),
            pl.lit("alpha.strategy").alias("candidate_name"),
            pl.lit("alpha_factory").alias("source_type"),
            pl.lit("alpha-event-1").alias("source_event_key"),
            pl.lit("research.alpha_factory.test").alias("source"),
        ]
    ).select(list(SAMPLE_SCHEMA)).cast(SAMPLE_SCHEMA, strict=True)
    sample_path = lake / "gold" / "strategy_evidence_sample" / "data.parquet"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample.write_parquet(sample_path)

    row = {column: None for column in SUMMARY_SCHEMA}
    row.update(
        {
            "strategy": "alpha",
            "evidence_version": "alpha-test-v1",
            "as_of_date": "2026-07-10",
            "strategy_candidate": "alpha.strategy",
            "candidate_name": "alpha.strategy",
            "symbol": "BTC-USDT-SWAP",
            "regime_state": "TEST",
            "horizon_hours": 4,
            "sample_count": 1,
            "complete_sample_count": 1,
            "decision": "RESEARCH_ONLY",
            "decision_reasons": "[]",
            "created_at": datetime(2026, 7, 10, 12, tzinfo=UTC),
            "source": "research.alpha_factory.test",
        }
    )
    summary = pl.DataFrame([row], schema=SUMMARY_SCHEMA, orient="row")
    summary_path = lake / "gold" / "strategy_evidence" / "data.parquet"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(summary_path)
