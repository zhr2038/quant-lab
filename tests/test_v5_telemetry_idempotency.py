import json
from pathlib import Path

import polars as pl
import pytest

import quant_lab.strategy_telemetry.ingest as ingest_module
from quant_lab.data.lake import read_parquet_dataset, write_parquet_dataset
from quant_lab.strategy_telemetry.ingest import ingest_v5_inbox
from tests.v5_bundle_fixture import make_tar, make_v5_bundle_fixture


def test_ingest_v5_inbox_skips_already_imported_sha256(tmp_path):
    inbox = tmp_path / "inbox"
    bundle = make_v5_bundle_fixture(inbox / "v5_live_followup_bundle_20260510T140249Z.tar.gz")
    lake = tmp_path / "lake"

    first = ingest_v5_inbox(inbox, lake, tmp_path / "restricted", tmp_path / "redacted")
    second = ingest_v5_inbox(inbox, lake, tmp_path / "restricted", tmp_path / "redacted")

    manifests = read_parquet_dataset(lake / Path("bronze/strategy_telemetry/v5/bundle_manifest"))
    assert bundle.exists()
    assert len(first.processed) == 1
    assert len(second.processed) == 0
    assert second.skipped_files == [str(bundle)]
    assert manifests.height == 1


def test_ingest_v5_inbox_can_process_newest_limited_bundle(tmp_path):
    inbox = tmp_path / "inbox"
    older = make_tar(
        inbox / "v5_live_followup_bundle_20260510T010000Z.tar.gz",
        {
            "summaries/window_summary.json": '{"run_count": 1}',
            "raw/state/kill_switch.json": '{"enabled": false}',
        },
    )
    newer = make_tar(
        inbox / "v5_live_followup_bundle_20260510T030000Z.tar.gz",
        {
            "summaries/window_summary.json": '{"run_count": 2}',
            "raw/state/kill_switch.json": '{"enabled": false}',
        },
    )
    lake = tmp_path / "lake"

    first = ingest_v5_inbox(
        inbox,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        max_bundles=1,
        newest_first=True,
        run_analysis=False,
        refresh_candidate_gold=False,
    )
    second = ingest_v5_inbox(
        inbox,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        max_bundles=1,
        newest_first=True,
        max_skipped_files_reported=0,
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    assert older.exists()
    assert newer.exists()
    assert [Path(result.bundle_path).name for result in first.processed] == [newer.name]
    assert [Path(result.bundle_path).name for result in second.processed] == [older.name]
    assert second.skipped_files == []
    assert any("skipped_files_truncated" in warning for warning in second.warnings)


def test_ingest_v5_inbox_can_scan_only_latest_window(tmp_path):
    inbox = tmp_path / "inbox"
    older = make_tar(
        inbox / "v5_live_followup_bundle_20260510T010000Z.tar.gz",
        {
            "summaries/window_summary.json": '{"run_count": 1}',
            "raw/state/kill_switch.json": '{"enabled": false}',
        },
    )
    newer = make_tar(
        inbox / "v5_live_followup_bundle_20260510T030000Z.tar.gz",
        {
            "summaries/window_summary.json": '{"run_count": 2}',
            "raw/state/kill_switch.json": '{"enabled": false}',
        },
    )
    lake = tmp_path / "lake"
    ingest_v5_inbox(
        inbox,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        max_bundles=1,
        newest_first=True,
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    second = ingest_v5_inbox(
        inbox,
        lake,
        tmp_path / "restricted",
        tmp_path / "redacted",
        max_bundles=1,
        max_scan_bundles=1,
        newest_first=True,
        max_skipped_files_reported=0,
        run_analysis=False,
        refresh_candidate_gold=False,
    )

    assert older.exists()
    assert newer.exists()
    assert second.processed == []
    assert second.skipped_files == []
    assert any("max_scan_bundles_limit_applied:1_of_2" in warning for warning in second.warnings)


def test_stable_upsert_hashes_only_new_rows_when_existing_keys_are_complete(
    tmp_path,
    monkeypatch,
):
    dataset = tmp_path / "lake" / "silver" / "v5_paper_strategy_run"
    first = {
        "strategy": "v5",
        "source_path_inside_bundle": "summaries/paper_strategy_runs.csv",
        "bundle_name": "first.tar.gz",
        "bundle_sha256": "first",
        "raw_payload_json": '{"strategy_id":"paper-1","run_id":"run-1"}',
    }
    existing = ingest_module._with_stable_row_key(first)
    write_parquet_dataset(pl.DataFrame([existing]), dataset)

    real_stable_row_key = ingest_module._stable_row_key
    hashed_bundle_names: list[str] = []

    def tracked_stable_row_key(row):
        hashed_bundle_names.append(str(row.get("bundle_name") or ""))
        return real_stable_row_key(row)

    monkeypatch.setattr(ingest_module, "_stable_row_key", tracked_stable_row_key)
    second = {
        **first,
        "bundle_name": "second.tar.gz",
        "bundle_sha256": "second",
    }

    row_count = ingest_module._upsert_stable_rows(dataset, [second])
    result = read_parquet_dataset(dataset)

    assert row_count == 1
    assert result.height == 1
    assert result["bundle_name"][0] == "second.tar.gz"
    assert hashed_bundle_names == ["second.tar.gz"]


def test_stable_upsert_repairs_legacy_rows_without_stable_keys(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "v5_paper_strategy_run"
    legacy = {
        "strategy": "v5",
        "source_path_inside_bundle": "summaries/paper_strategy_runs.csv",
        "bundle_name": "legacy.tar.gz",
        "bundle_sha256": "legacy",
        "raw_payload_json": '{"strategy_id":"paper-legacy","run_id":"run-1"}',
    }
    write_parquet_dataset(pl.DataFrame([legacy]), dataset)
    current = {
        **legacy,
        "bundle_name": "current.tar.gz",
        "bundle_sha256": "current",
        "raw_payload_json": '{"strategy_id":"paper-current","run_id":"run-2"}',
    }

    row_count = ingest_module._upsert_stable_rows(dataset, [current])
    result = read_parquet_dataset(dataset)

    assert row_count == 2
    assert result.height == 2
    assert result["stable_row_key"].null_count() == 0
    assert result["stable_row_key"].n_unique() == 2


def test_bundle_manifest_is_written_only_after_silver_succeeds(tmp_path, monkeypatch):
    inbox = tmp_path / "inbox"
    bundle = make_v5_bundle_fixture(
        inbox / "v5_live_followup_bundle_20260510T140249Z.tar.gz"
    )
    lake = tmp_path / "lake"

    def fail_silver(*args, **kwargs):
        raise MemoryError("simulated silver failure")

    monkeypatch.setattr(ingest_module, "_write_silver", fail_silver)

    with pytest.raises(MemoryError, match="simulated silver failure"):
        ingest_module.ingest_v5_bundle(
            bundle,
            lake,
            tmp_path / "restricted",
            tmp_path / "redacted",
            run_analysis=False,
            refresh_candidate_gold=False,
        )

    manifest = read_parquet_dataset(
        lake / "bronze" / "strategy_telemetry" / "v5" / "bundle_manifest"
    )
    secret_scan = read_parquet_dataset(
        lake / "bronze" / "strategy_telemetry" / "v5" / "secret_scan"
    )
    assert manifest.is_empty()
    assert secret_scan.height == 1


def test_event_upsert_retry_of_same_bundle_does_not_double_source_count(tmp_path):
    dataset = tmp_path / "lake" / "silver" / "v5_quant_lab_request"
    row = {
        "strategy": "v5",
        "bundle_sha256": "same-bundle",
        "bundle_name": "same.tar.gz",
        "bundle_ts": "2026-05-10T14:02:49Z",
        "source_path_inside_bundle": "reports/quant_lab_requests.jsonl",
        "event_id": "request-1",
        "request_id": "request-1",
        "raw_payload_json": json.dumps({"request_id": "request-1", "ok": True}),
    }

    ingest_module._upsert_event_rows(dataset, [row])
    ingest_module._upsert_event_rows(dataset, [row])
    result = read_parquet_dataset(dataset)

    assert result.height == 1
    assert int(result["source_count"][0]) == 1
