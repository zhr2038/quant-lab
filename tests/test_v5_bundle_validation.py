from quant_lab.strategy_telemetry.bundle import validate_v5_bundle
from quant_lab.strategy_telemetry.models import BundleLimits
from tests.v5_bundle_fixture import make_tar, make_v5_bundle_fixture


def test_bundle_validation_accepts_valid_fixture(tmp_path):
    bundle = make_v5_bundle_fixture(tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz")

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.valid is True
    assert result.rejected is False
    assert result.sha256
    assert "raw/state/kill_switch.json" in result.detected_files
    assert "reports/candidate_snapshot.csv" in result.detected_files


def test_bundle_validation_rejects_path_traversal(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {"../../evil": "bad"},
    )

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.rejected is True
    assert any("path traversal" in reason for reason in result.reasons)


def test_bundle_validation_rejects_symlink(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {"raw/state/kill_switch.json": "{}"},
        symlink="raw/link",
    )

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.rejected is True
    assert any("symlink" in reason for reason in result.reasons)
