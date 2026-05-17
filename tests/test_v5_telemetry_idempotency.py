from pathlib import Path

from quant_lab.data.lake import read_parquet_dataset
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
