from pathlib import Path

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.strategy_telemetry.ingest import ingest_v5_inbox
from tests.v5_bundle_fixture import make_v5_bundle_fixture


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
