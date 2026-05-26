from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from quant_lab.data.lake import write_parquet_dataset
from quant_lab.research.diagnostics_refresh import refresh_research_diagnostics
from quant_lab.web import readers


def test_refresh_research_diagnostics_skips_closed_portfolio_items(tmp_path):
    lake = tmp_path / "lake"
    now = datetime.now(UTC)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-26",
                    "research_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
                    "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
                    "status": "KILL",
                    "action": "CLOSE_RESEARCH",
                    "created_at": now,
                },
                {
                    "as_of_date": "2026-05-26",
                    "research_id": "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW",
                    "strategy_candidate": "v5.btc_strict_probe_exit_policy_review",
                    "status": "PAUSED",
                    "action": "PAUSED_TO_WEEKLY",
                    "created_at": now,
                },
                {
                    "as_of_date": "2026-05-26",
                    "research_id": "BNB_SWING_EXIT_POLICY_REVIEW",
                    "strategy_candidate": "v5.bnb_swing_exit_policy_review",
                    "status": "KILL",
                    "action": "CLOSE_RESEARCH",
                    "created_at": now,
                },
            ]
        ),
        lake / "gold" / "research_portfolio_status",
    )

    result = refresh_research_diagnostics(lake, as_of_date="2026-05-26")

    assert result.ran_count == 0
    assert result.skipped_count == 3
    assert {job.status for job in result.jobs} == {"skipped"}
    assert not (lake / "gold" / "sol_protect_paper_loss_attribution").exists()
    assert not (lake / "gold" / "btc_probe_exit_policy_review").exists()
    assert not (lake / "gold" / "bnb_swing_exit_policy_review").exists()


def test_data_health_hides_stale_closed_research_diagnostics(tmp_path):
    lake = tmp_path / "lake"
    now = datetime.now(UTC)
    old = now - timedelta(days=3)
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "generated_at_utc": old,
                    "as_of_date": "2026-05-23",
                    "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
                }
            ]
        ),
        lake / "gold" / "sol_protect_paper_loss_attribution",
    )
    write_parquet_dataset(
        pl.DataFrame(
            [
                {
                    "as_of_date": "2026-05-26",
                    "research_id": "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
                    "strategy_candidate": "v5.sol_protect_alpha6_low_exception",
                    "status": "KILL",
                    "action": "CLOSE_RESEARCH",
                    "created_at": now,
                }
            ]
        ),
        lake / "gold" / "research_portfolio_status",
    )

    stale_rows = readers.data_health_summary(lake)["stale_datasets"].to_dicts()

    assert not any(row["dataset"] == "sol_protect_paper_loss_attribution" for row in stale_rows)
