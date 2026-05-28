import tarfile
from io import BytesIO

from quant_lab.strategy_telemetry.bundle import validate_v5_bundle
from quant_lab.strategy_telemetry.models import BundleLimits
from quant_lab.strategy_telemetry.sanitize import redact_text, scan_for_secrets


def test_v5_bundle_validation_rejects_path_traversal(tmp_path):
    bundle = tmp_path / "v5_live_followup_bundle_20260528T010000Z.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        data = b"unsafe"
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(data)
        archive.addfile(info, BytesIO(data))

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.valid is False
    assert result.rejected is True
    assert any("path traversal" in reason for reason in result.reasons)


def test_v5_bundle_secret_scan_and_redaction_share_patterns():
    text = "api_key: SHOULD_NOT_LEAK\napi_secret: SHOULD_NOT_LEAK_2\n"

    scan = scan_for_secrets(text)
    redacted = redact_text(text)

    assert scan.high_severity_count > 0
    assert "SHOULD_NOT_LEAK" not in redacted
    assert "<REDACTED>" in redacted
