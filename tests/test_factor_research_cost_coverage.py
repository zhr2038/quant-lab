from datetime import UTC, date, datetime

import polars as pl

from quant_lab.research_worker.factor_research import (
    MAX_POINT_IN_TIME_COST_AGE_HOURS,
    MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY,
    MIN_RESEARCH_PROXY_ROUNDTRIP_COST_BPS,
    POINT_IN_TIME_COST_EVALUATION_DAYS,
    _attach_point_in_time_cost_validity,
    _join_cost_point_in_time,
    _merge_point_in_time_cost_sources,
    _normalize_cost_source,
    _normalize_spread_proxy_cost_source,
)


def _costs(rows: list[dict[str, object]]) -> pl.DataFrame:
    return _normalize_cost_source(
        pl.DataFrame(rows),
        value_candidates=("total_cost_bps_p75",),
    )


def _samples(*timestamps: datetime) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTC-USDT"] * len(timestamps),
            "decision_ts": list(timestamps),
        },
        schema={
            "symbol": pl.Utf8,
            "decision_ts": pl.Datetime(time_zone="UTC"),
        },
    )


def test_cost_is_unavailable_before_created_at_and_source_classes_stay_separate() -> None:
    costs = _costs(
        [
            {
                "day": date(2026, 1, 1),
                "created_at": "2026-01-01T23:00:00Z",
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 30.0,
                "cost_source": "public_spread_proxy",
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "eligible_for_live_cost_coverage": False,
            },
            {
                "day": date(2026, 1, 2),
                "created_at": "2026-01-02T23:00:00Z",
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 25.0,
                "cost_source": "actual_fills",
                "fallback_level": "NONE",
                "eligible_for_live_cost_coverage": True,
            },
            {
                "day": date(2026, 1, 3),
                "created_at": "2026-01-03T23:00:00Z",
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 20.0,
                "cost_source": "bootstrap_cost_probe",
                "fallback_level": "COST_PROBE_ONLY",
                "eligible_for_live_cost_coverage": False,
            },
        ]
    )
    samples = _samples(
        datetime(2026, 1, 1, 22, tzinfo=UTC),
        datetime(2026, 1, 2, 12, tzinfo=UTC),
        datetime(2026, 1, 3, 12, tzinfo=UTC),
        datetime(2026, 1, 4, 12, tzinfo=UTC),
        datetime(2026, 1, 5, 13, tzinfo=UTC),
    )

    joined = _attach_point_in_time_cost_validity(_join_cost_point_in_time(samples, costs))

    assert joined.get_column("cost_bps").to_list() == [None, 30.0, 25.0, 30.0, 30.0]
    assert joined.get_column("cost_point_in_time_valid").to_list() == [
        False,
        True,
        True,
        True,
        False,
    ]
    assert joined.get_column("cost_public_proxy_valid").to_list() == [
        False,
        True,
        False,
        False,
        False,
    ]
    assert joined.get_column("cost_trusted_point_in_time_valid").to_list() == [
        False,
        False,
        True,
        False,
        False,
    ]
    assert joined.get_column("cost_bootstrap_valid").to_list() == [
        False,
        False,
        False,
        True,
        False,
    ]
    assert joined.get_column("cost_stale").to_list() == [False, False, False, False, True]


def test_daily_cost_without_created_at_is_only_available_next_utc_day() -> None:
    costs = _costs(
        [
            {
                "day": date(2026, 1, 1),
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 30.0,
                "cost_source": "public_spread_proxy",
                "fallback_level": "PUBLIC_SPREAD_PROXY",
                "eligible_for_live_cost_coverage": False,
            }
        ]
    )
    samples = _samples(
        datetime(2026, 1, 1, 23, tzinfo=UTC),
        datetime(2026, 1, 2, 1, tzinfo=UTC),
    )

    joined = _attach_point_in_time_cost_validity(_join_cost_point_in_time(samples, costs))

    assert joined.get_column("cost_bps").to_list() == [None, 30.0]
    assert joined.get_column("cost_point_in_time_valid").to_list() == [False, True]


def test_point_in_time_cost_policy_is_locked_to_bounded_research_window() -> None:
    assert POINT_IN_TIME_COST_EVALUATION_DAYS == 30
    assert MAX_POINT_IN_TIME_COST_AGE_HOURS == 36
    assert MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY == 60
    assert MIN_RESEARCH_PROXY_ROUNDTRIP_COST_BPS == 30.0


def _spread_rows(day: date, *, symbol: str = "BTC-USDT") -> pl.DataFrame:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return pl.DataFrame(
        {
            "symbol": [symbol] * MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY,
            "channel": ["books5"] * MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY,
            "minute_ts": [
                start.replace(minute=index % 60, hour=index // 60)
                for index in range(MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY)
            ],
            "spread_bps": [5.0] * MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY,
        }
    )


def test_spread_rollup_is_only_available_after_closed_day_and_uses_cost_floor() -> None:
    costs = _normalize_spread_proxy_cost_source(_spread_rows(date(2026, 1, 1)))
    samples = _samples(
        datetime(2026, 1, 1, 23, 59, tzinfo=UTC),
        datetime(2026, 1, 2, 1, tzinfo=UTC),
    )

    joined = _attach_point_in_time_cost_validity(_join_cost_point_in_time(samples, costs))

    assert costs.item(0, "cost_observation_count") == 60
    assert joined.get_column("cost_bps").to_list() == [None, 30.0]
    assert joined.get_column("cost_reconstructed_proxy_valid").to_list() == [False, True]
    assert joined.item(1, "eligible_for_live_cost_coverage") is False


def test_sparse_spread_day_does_not_claim_point_in_time_cost_coverage() -> None:
    rows = _spread_rows(date(2026, 1, 1)).head(MIN_POINT_IN_TIME_SPREAD_SAMPLES_PER_DAY - 1)

    assert _normalize_spread_proxy_cost_source(rows).is_empty()


def test_late_materialized_row_cannot_override_causal_spread_proxy() -> None:
    published = _costs(
        [
            {
                "day": date(2026, 1, 1),
                "created_at": "2026-01-04T01:00:00Z",
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 99.0,
                "cost_source": "bootstrap_cost_probe",
                "fallback_level": "COST_PROBE_ONLY",
                "eligible_for_live_cost_coverage": False,
            }
        ]
    )
    reconstructed = _normalize_spread_proxy_cost_source(_spread_rows(date(2026, 1, 3)))
    merged = _merge_point_in_time_cost_sources(published, reconstructed)
    sample = _samples(datetime(2026, 1, 4, 2, tzinfo=UTC))

    joined = _attach_point_in_time_cost_validity(_join_cost_point_in_time(sample, merged))

    assert published.item(0, "cost_materialized_late") is True
    assert joined.item(0, "cost_bps") == 30.0
    assert joined.item(0, "cost_source") == "public_spread_proxy"
    assert joined.item(0, "cost_reconstructed_proxy_valid") is True
    assert joined.item(0, "cost_trusted_point_in_time_valid") is False


def test_timely_trusted_cost_supersedes_same_day_reconstructed_proxy() -> None:
    published = _costs(
        [
            {
                "day": date(2026, 1, 1),
                "created_at": "2026-01-01T23:00:00Z",
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 12.0,
                "cost_source": "actual_fills",
                "fallback_level": "NONE",
                "eligible_for_live_cost_coverage": True,
            }
        ]
    )
    reconstructed = _normalize_spread_proxy_cost_source(_spread_rows(date(2026, 1, 1)))
    merged = _merge_point_in_time_cost_sources(published, reconstructed)
    sample = _samples(datetime(2026, 1, 2, 1, tzinfo=UTC))

    joined = _attach_point_in_time_cost_validity(_join_cost_point_in_time(sample, merged))

    assert joined.item(0, "cost_bps") == 12.0
    assert joined.item(0, "cost_source") == "actual_fills"
    assert joined.item(0, "cost_trusted_point_in_time_valid") is True
    assert joined.item(0, "cost_reconstructed_proxy_valid") is False


def test_fresher_proxy_is_not_deleted_by_older_cost_with_same_available_time() -> None:
    published = _costs(
        [
            {
                "day": date(2026, 1, 1),
                "created_at": "2026-01-03T00:00:00Z",
                "symbol": "BTC-USDT",
                "total_cost_bps_p75": 12.0,
                "cost_source": "actual_fills",
                "fallback_level": "NONE",
                "eligible_for_live_cost_coverage": True,
            }
        ]
    )
    reconstructed = _normalize_spread_proxy_cost_source(
        _spread_rows(date(2026, 1, 2))
    )
    merged = _merge_point_in_time_cost_sources(published, reconstructed)
    sample = _samples(datetime(2026, 1, 3, 20, tzinfo=UTC))

    joined = _attach_point_in_time_cost_validity(
        _join_cost_point_in_time(sample, merged)
    )

    assert joined.item(0, "_cost_bps_source_day") == date(2026, 1, 2)
    assert joined.item(0, "cost_source") == "public_spread_proxy"
    assert joined.item(0, "cost_point_in_time_valid") is True
