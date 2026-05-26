from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from quant_lab.data.lake import read_parquet_dataset
from quant_lab.research.bnb_swing_exit_policy import (
    build_and_publish_bnb_swing_exit_policy_review,
)
from quant_lab.research.btc_probe_exit_policy import (
    build_and_publish_btc_probe_exit_policy_review,
)
from quant_lab.research.portfolio import (
    RESEARCH_PORTFOLIO_STATUS_DATASET,
    research_portfolio_blocks_daily_refresh,
)
from quant_lab.research.sol_protect_paper_loss import (
    build_and_publish_sol_protect_paper_loss_attribution,
)


class DiagnosticRefreshJobResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_name: str
    status: str
    reason: str = ""
    research_keys: list[str] = Field(default_factory=list)
    output: dict[str, Any] = Field(default_factory=dict)


class DiagnosticRefreshResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    as_of_date: str
    generated_at: datetime
    ran_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    jobs: list[DiagnosticRefreshJobResult] = Field(default_factory=list)


@dataclass(frozen=True)
class _DiagnosticJob:
    job_name: str
    research_keys: tuple[str, ...]
    runner: Callable[[Path, date], Any]


DIAGNOSTIC_REFRESH_JOBS: tuple[_DiagnosticJob, ...] = (
    _DiagnosticJob(
        job_name="build-sol-protect-paper-loss-attribution",
        research_keys=(
            "SOL_PROTECT_ALPHA6_LOW_EXCEPTION_PAPER_V1",
            "v5.sol_protect_alpha6_low_exception",
        ),
        runner=lambda root, day: build_and_publish_sol_protect_paper_loss_attribution(
            root,
            as_of_date=day,
        ),
    ),
    _DiagnosticJob(
        job_name="build-btc-probe-exit-policy-review",
        research_keys=(
            "BTC_STRICT_PROBE_EXIT_POLICY_REVIEW",
            "v5.btc_strict_probe_exit_policy_review",
        ),
        runner=lambda root, day: build_and_publish_btc_probe_exit_policy_review(
            root,
            as_of_date=day,
        ),
    ),
    _DiagnosticJob(
        job_name="build-bnb-swing-exit-policy-review",
        research_keys=(
            "BNB_SWING_EXIT_POLICY_REVIEW",
            "v5.bnb_swing_exit_policy_review",
        ),
        runner=lambda root, day: build_and_publish_bnb_swing_exit_policy_review(
            root,
            as_of_date=day,
        ),
    ),
)


def refresh_research_diagnostics(
    lake_root: str | Path,
    *,
    as_of_date: str | date | None = None,
) -> DiagnosticRefreshResult:
    root = Path(lake_root)
    day = _parse_day(as_of_date)
    generated_at = datetime.now(UTC)
    portfolio = read_parquet_dataset(root / RESEARCH_PORTFOLIO_STATUS_DATASET)
    jobs: list[DiagnosticRefreshJobResult] = []
    for job in DIAGNOSTIC_REFRESH_JOBS:
        keys = set(job.research_keys)
        if research_portfolio_blocks_daily_refresh(portfolio, keys):
            jobs.append(
                DiagnosticRefreshJobResult(
                    job_name=job.job_name,
                    status="skipped",
                    reason="research_portfolio_closed_or_paused",
                    research_keys=sorted(keys),
                )
            )
            continue
        result = job.runner(root, day)
        output = (
            result.model_dump(mode="json")
            if hasattr(result, "model_dump")
            else {"result": str(result)}
        )
        jobs.append(
            DiagnosticRefreshJobResult(
                job_name=job.job_name,
                status="ran",
                research_keys=sorted(keys),
                output=output,
            )
        )
    return DiagnosticRefreshResult(
        as_of_date=day.isoformat(),
        generated_at=generated_at,
        ran_count=sum(1 for job in jobs if job.status == "ran"),
        skipped_count=sum(1 for job in jobs if job.status == "skipped"),
        jobs=jobs,
    )


def _parse_day(value: str | date | None) -> date:
    if value is None:
        return datetime.now(UTC).date()
    if isinstance(value, date):
        return value
    if str(value).lower() == "auto":
        return datetime.now(UTC).date()
    return date.fromisoformat(str(value))
