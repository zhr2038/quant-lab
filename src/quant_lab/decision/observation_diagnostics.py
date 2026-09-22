"""Descriptive, scope-bound coverage and outcome diagnostics, never account PnL."""

from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean

from quant_lab.decision.contracts import HORIZONS, SYMBOLS
from quant_lab.decision.contracts_v3 import (
    MatureNonOverlappingGroup,
    RegistrationCoverage,
    RegistrationGap,
    SymbolRegistration,
)

HOUR = timedelta(hours=1)


def registration_coverage(rows: list[dict], *, now: datetime) -> RegistrationCoverage:
    # A running hour has not missed its next-entry deadline. No backfill is inferred.
    due = now.replace(minute=0, second=0, microsecond=0)
    first = min((row["entry_at"] for row in rows), default=None)
    expected = []
    if first is not None:
        expected = [first + i * HOUR for i in range(max(0, int((due - first) / HOUR) + 1))]
    result = []
    for symbol in SYMBOLS:
        slots = {}
        horizons = defaultdict(set)
        for row in rows:
            if row["symbol"] == symbol:
                entry = row["entry_at"]
                slots[entry] = min(slots.get(entry, row["published_at"]), row["published_at"])
                horizons[entry].add(row["horizon"])
        complete = {entry for entry, values in horizons.items() if values == set(HORIZONS)}
        missing = [entry for entry in expected if entry not in complete]
        gaps = []
        for entry in missing:
            if gaps and entry == gaps[-1][1] + HOUR:
                gaps[-1][1] = entry
            else:
                gaps.append([entry, entry])
        delays = [
            (published - (entry - HOUR)).total_seconds() for entry, published in slots.items()
        ]
        registered = len(expected) - len(missing)
        result.append(
            SymbolRegistration(
                symbol=symbol,
                expected_hours=len(expected),
                registered_hours=registered,
                missing_hours=len(missing),
                pending_deadline_hours=sum(entry > due for entry in slots),
                partial_horizon_hours=sum(entry in slots for entry in missing),
                coverage_fraction=registered / len(expected) if expected else None,
                late_publication_hours=sum(delay > 900 for delay in delays),
                maximum_publication_delay_seconds=max(delays, default=None),
                latest_registered_entry_at=max(slots, default=None),
                gaps=[
                    RegistrationGap(
                        first_entry_at=a, last_entry_at=b, hours=int((b - a) / HOUR) + 1
                    )
                    for a, b in gaps[-50:]
                ],
                omitted_gap_ranges=max(0, len(gaps) - 50),
            )
        )
    return RegistrationCoverage(
        first_entry_at=first,
        due_through_entry_at=due,
        by_symbol=result,
        status="NOT_STARTED"
        if first is None
        else ("GAPS" if any(row.missing_hours for row in result) else "COMPLETE"),
    )


def mature_non_overlapping(rows: list[dict], *, now: datetime) -> list[MatureNonOverlappingGroup]:
    # Select using registered timestamps before looking at maturity, action or return.
    # Missing labels cannot shift the sample and accidentally select a better outcome.
    last_exit = {}
    groups = defaultdict(list)
    for row in sorted(
        rows, key=lambda r: (r["entry_at"], r["symbol"], r["horizon"], r["opportunity"])
    ):
        key = row["symbol"], row["horizon"]
        if key in last_exit and row["entry_at"] < last_exit[key]:
            continue
        last_exit[key] = row["exit_at"]
        groups[(*key, row["action"])].append(row)
    result = []
    for (symbol, horizon, action), selected in sorted(groups.items()):
        matured = [r for r in selected if r["label_at"] is not None and r["label_at"] <= now]
        waiting = sum(r not in matured and r["exit_at"] + HOUR > now for r in selected)
        gross = [r["gross_bps"] for r in matured if r["gross_bps"] is not None]
        net = [r["net_bps"] for r in matured if r["net_bps"] is not None]
        result.append(
            MatureNonOverlappingGroup(
                symbol=symbol,
                horizon_hours=horizon,
                action=action,
                selected_observations=len(selected),
                matured_observations=len(matured),
                waiting_observations=waiting,
                missing_label_observations=len(selected) - len(matured) - waiting,
                net_observations=len(net),
                gross_mean_bps=mean(gross) if gross else None,
                net_mean_bps=mean(net) if net else None,
            )
        )
    return result
