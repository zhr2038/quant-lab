import gzip

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


def test_bundle_validation_accepts_large_quant_lab_jsonl_gzip_paths(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260606T082013Z.tar.gz",
        {
            "raw/large/reports/quant_lab_usage.jsonl.gz": gzip.compress(
                b'{"ts":"2026-06-06T08:20:13Z","mode":"shadow"}\n'
            ),
            "raw/large/reports/quant_lab_requests.jsonl.gz": gzip.compress(
                b'{"ts":"2026-06-06T08:20:13Z","path":"/v1/health"}\n'
            ),
        },
    )

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.valid is True
    assert "raw/large/reports/quant_lab_usage.jsonl.gz" in result.detected_files
    assert "raw/large/reports/quant_lab_requests.jsonl.gz" in result.detected_files
    assert not any("unknown file path" in warning for warning in result.warnings)


def test_bundle_validation_accepts_expanded_universe_summary_paths(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260606T082113Z.tar.gz",
        {
            "summaries/expanded_universe_advisory_reader.csv": (
                "strategy_id,symbol\nWLD_EXPANDED_UNIVERSE_PAPER_V1,WLD-USDT\n"
            ),
            "summaries/expanded_universe_paper_runs.csv": (
                "strategy_id,symbol\nWLD_EXPANDED_UNIVERSE_PAPER_V1,WLD-USDT\n"
            ),
            "summaries/expanded_universe_paper_daily.csv": (
                "strategy_id,symbol\nWLD_EXPANDED_UNIVERSE_PAPER_V1,WLD-USDT\n"
            ),
        },
    )

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.valid is True
    assert "summaries/expanded_universe_advisory_reader.csv" in result.detected_files
    assert "summaries/expanded_universe_paper_daily.csv" in result.detected_files
    assert not any("unknown file path" in warning for warning in result.warnings)


def test_bundle_validation_rejects_symlink(tmp_path):
    bundle = make_tar(
        tmp_path / "v5_live_followup_bundle_20260510T140249Z.tar.gz",
        {"raw/state/kill_switch.json": "{}"},
        symlink="raw/link",
    )

    result = validate_v5_bundle(bundle, BundleLimits())

    assert result.rejected is True
    assert any("symlink" in reason for reason in result.reasons)
