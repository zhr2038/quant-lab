from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.ops.data_quality import run_data_quality


def _risk_row(**updates):
    now = datetime(2026, 5, 28, 2, tzinfo=UTC)
    row = {
        "permission": "ABORT",
        "permission_status": "ACTIVE_ABORT",
        "as_of_ts": now,
        "expires_at": now + timedelta(minutes=90),
        "enforceable": True,
    }
    row.update(updates)
    return row


def test_risk_permission_quality_passes_when_unexpired(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame([_risk_row()]),
        lake / "gold" / "risk_permission",
    )

    result = run_data_quality(
        lake,
        dataset_names=["risk_permission"],
        reference_at=datetime(2026, 5, 28, 2, 30, tzinfo=UTC),
    ).to_dict()

    not_expired = next(
        check for check in result["checks"] if check["rule"] == "risk_permission_not_expired"
    )
    assert not_expired["status"] == "PASS"


def test_risk_permission_quality_fails_when_expired(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                _risk_row(
                    as_of_ts=datetime(2026, 5, 28, 1, tzinfo=UTC),
                    expires_at=datetime(2026, 5, 28, 1, 30, tzinfo=UTC),
                    permission_status="EXPIRED_ABORT",
                    enforceable=False,
                )
            ]
        ),
        lake / "gold" / "risk_permission",
    )

    result = run_data_quality(
        lake,
        dataset_names=["risk_permission"],
        reference_at=datetime(2026, 5, 28, 2, 30, tzinfo=UTC),
    ).to_dict()

    not_expired = next(
        check for check in result["checks"] if check["rule"] == "risk_permission_not_expired"
    )
    assert result["status"] == "FAIL"
    assert not_expired["status"] == "FAIL"
    assert "publish-risk-permission" in not_expired["next_action"]


def test_risk_permission_quality_fails_active_expired_status(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                _risk_row(
                    as_of_ts=datetime(2026, 5, 28, 1, tzinfo=UTC),
                    expires_at=datetime(2026, 5, 28, 1, 30, tzinfo=UTC),
                    permission_status="ACTIVE_ABORT",
                    enforceable=True,
                )
            ]
        ),
        lake / "gold" / "risk_permission",
    )

    result = run_data_quality(
        lake,
        dataset_names=["risk_permission"],
        reference_at=datetime(2026, 5, 28, 2, 30, tzinfo=UTC),
    ).to_dict()

    active_not_expired = next(
        check for check in result["checks"] if check["rule"] == "risk_permission_active_not_expired"
    )
    assert result["status"] == "FAIL"
    assert active_not_expired["status"] == "FAIL"
    assert active_not_expired["observed_value"] == "1"


def test_risk_permission_quality_fails_non_active_enforceable(tmp_path):
    lake = tmp_path / "lake"
    write_parquet_dataset(
        pl.DataFrame(
            [
                _risk_row(
                    expires_at=datetime(2026, 5, 28, 4, tzinfo=UTC),
                    permission_status="STALE_ABORT",
                    enforceable=True,
                )
            ]
        ),
        lake / "gold" / "risk_permission",
    )

    result = run_data_quality(
        lake,
        dataset_names=["risk_permission"],
        reference_at=datetime(2026, 5, 28, 2, 30, tzinfo=UTC),
    ).to_dict()

    consistency = next(
        check
        for check in result["checks"]
        if check["rule"] == "risk_permission_enforceable_consistency"
    )
    assert result["status"] == "FAIL"
    assert consistency["status"] == "FAIL"
    assert consistency["observed_value"] == "1"
