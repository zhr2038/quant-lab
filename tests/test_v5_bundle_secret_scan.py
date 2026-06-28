import quant_lab.export.daily as daily_export_module
from quant_lab.strategy_telemetry.bundle import safe_extract_v5_bundle
from quant_lab.strategy_telemetry.models import BundleLimits
from quant_lab.strategy_telemetry.sanitize import redact_extracted_bundle, scan_for_secrets
from tests.v5_bundle_fixture import SECRET_VALUE, make_v5_bundle_fixture


def test_secret_scan_redacts_config(tmp_path):
    bundle = make_v5_bundle_fixture(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        secret=True,
    )
    extracted = tmp_path / "extracted"
    redacted = tmp_path / "redacted"
    safe_extract_v5_bundle(bundle, extracted, BundleLimits())

    scan = scan_for_secrets(extracted)
    result = redact_extracted_bundle(extracted, redacted)
    redacted_text = "\n".join(
        path.read_text(encoding="utf-8") for path in redacted.rglob("*") if path.is_file()
    )

    assert scan.high_severity_count > 0
    assert result.redacted_files
    assert SECRET_VALUE not in redacted_text
    assert "<REDACTED>" in redacted_text


def test_secret_scan_allows_already_redacted_values():
    scan = scan_for_secrets(
        "api_key: <REDACTED>\n"
        "api_secret: <REDACTED>\n"
        "passphrase: <REDACTED>\n"
        "api_key: <REDACTED>}\n"
        "api_secret: <REDACTED>}\n"
        "passphrase: <REDACTED>}\n"
        "allow_insecure_http_with_token: <REDACTED>\n"
    )

    assert scan.high_severity_count == 0
    assert scan.medium_severity_count == 0


def test_secret_scan_flags_plaintext_credentials_as_high_severity():
    scan = scan_for_secrets(
        "api_key: SHOULD_NOT_LEAK_1\n"
        "api_secret: SHOULD_NOT_LEAK_2\n"
        "passphrase: SHOULD_NOT_LEAK_3\n"
    )

    assert scan.high_severity_count == 3


def test_redacted_v5_secret_scan_allows_large_clean_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(
        daily_export_module,
        "V5_BUNDLE_SECRET_SCAN_MAX_MEMBER_BYTES",
        32,
    )
    summaries = tmp_path / "redacted" / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "paper_strategy_runs.csv").write_text(
        "strategy_id,ts_utc,status\n" + ("paper,2026-06-28T00:00:00Z,OK\n" * 8),
        encoding="utf-8",
    )

    assert daily_export_module._v5_redacted_files_secret_reasons(tmp_path / "redacted") == []


def test_redacted_v5_secret_scan_streams_large_csv_and_flags_secret(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        daily_export_module,
        "V5_BUNDLE_SECRET_SCAN_MAX_MEMBER_BYTES",
        32,
    )
    summaries = tmp_path / "redacted" / "summaries"
    summaries.mkdir(parents=True)
    (summaries / "paper_strategy_runs.csv").write_text(
        "strategy_id,ts_utc,status\n"
        + ("paper,2026-06-28T00:00:00Z,OK\n" * 8)
        + "api_key: SHOULD_NOT_LEAK\n",
        encoding="utf-8",
    )

    assert daily_export_module._v5_redacted_files_secret_reasons(tmp_path / "redacted") == [
        "summaries/paper_strategy_runs.csv: 1 high, 0 medium"
    ]
