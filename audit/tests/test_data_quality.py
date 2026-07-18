from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.data_quality import check_candle_frame, check_funding_frame  # noqa: E402


def _candle(ts: datetime, close: float = 100.0) -> dict:
    return {
        "ts": int(ts.timestamp() * 1000),
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "quote_volume": 1000.0,
        "confirm": "1",
    }


def test_out_of_order_is_checked_before_sorting() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pl.DataFrame([_candle(start), _candle(start + timedelta(hours=2)), _candle(start + timedelta(hours=1))])
    result = check_candle_frame(frame, "TEST-USDT")
    assert result["out_of_order"] == 1
    assert result["severity"] == "critical"


def test_funding_cadence_is_inferred_not_hardcoded_to_eight_hours() -> None:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rows = [
        {
            "funding_time": int((start + timedelta(hours=4 * i)).timestamp() * 1000),
            "funding_rate": 0.0001,
        }
        for i in range(6 * 30 + 1)
    ]
    result = check_funding_frame(pl.DataFrame(rows), "TEST-USDT", required_history_days=20)
    assert result["inferred_cadence_hours"] == pytest.approx(4.0)
    assert result["cadence_coverage"] == pytest.approx(1.0, abs=0.01)
    assert result["severity"] == "ok"

