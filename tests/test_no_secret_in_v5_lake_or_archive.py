from quant_lab.strategy_telemetry.ingest import ingest_v5_bundle
from tests.v5_bundle_fixture import SECRET_VALUE, make_v5_bundle_fixture


def test_no_secret_in_lake_or_redacted_archive(tmp_path):
    bundle = make_v5_bundle_fixture(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        secret=True,
    )
    lake = tmp_path / "lake"
    redacted = tmp_path / "redacted"

    result = ingest_v5_bundle(bundle, lake, tmp_path / "restricted", redacted)

    assert result.secret_scan.high_severity_count > 0
    lake_text = _read_all_text(lake)
    redacted_text = _read_all_text(redacted)
    assert SECRET_VALUE not in lake_text
    assert SECRET_VALUE not in redacted_text
    assert "<REDACTED>" in redacted_text


def _read_all_text(root) -> str:
    chunks = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
    return "\n".join(chunks)
