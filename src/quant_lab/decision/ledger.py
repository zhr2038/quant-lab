"""Prospective price labels recorded only after actual cloud publication.

This ledger neither simulates a portfolio nor attributes returns to V5.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from quant_lab.decision.contracts import AnalysisResult, ForwardGroup, ForwardSummary, HourBar


class Ledger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(path), config={"memory_limit": "256MB", "threads": "1"})
        self.con.execute("SET TimeZone='UTC'")
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS observations (
              opportunity VARCHAR, horizon INTEGER, experiment VARCHAR, advice_id VARCHAR,
              symbol VARCHAR, action VARCHAR, published_at TIMESTAMPTZ,
              entry_at TIMESTAMPTZ, exit_at TIMESTAMPTZ, cost DOUBLE,
              advice_json VARCHAR, label_at TIMESTAMPTZ, gross_bps DOUBLE, net_bps DOUBLE,
              label_evidence_json VARCHAR,
              PRIMARY KEY (opportunity, horizon, experiment)
            )
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.con.close()

    def register(self, result: AnalysisResult, *, published_at: datetime, now: datetime) -> int:
        if published_at < result.generated_at or published_at > now:
            raise ValueError("publication time is inconsistent with result availability")
        inserted = 0
        for advice in result.advice:
            # A delayed or expired publication can never become a prospective forecast.
            if advice.reference_entry_at is None or published_at >= min(
                advice.reference_entry_at, advice.expires_at
            ):
                continue
            rows = self.con.execute(
                """
                INSERT INTO observations
                  (opportunity,horizon,experiment,advice_id,symbol,action,published_at,
                   entry_at,exit_at,cost,advice_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING RETURNING advice_id
            """,
                [
                    advice.opportunity_id,
                    advice.horizon_hours,
                    advice.experiment_version,
                    advice.advice_id,
                    advice.symbol,
                    advice.action,
                    published_at,
                    advice.reference_entry_at,
                    advice.reference_exit_at,
                    advice.cost.roundtrip_bps,
                    advice.model_dump_json(),
                ],
            ).fetchall()
            inserted += len(rows)
        return inserted

    def mature(self, bars: list[HourBar], *, now: datetime) -> None:
        available = {
            (bar.symbol, bar.ts): bar
            for bar in bars
            if bar.ts + timedelta(hours=1) <= now and bar.ingest_ts <= now
        }
        pending = self.con.execute(
            """
            SELECT opportunity,horizon,experiment,symbol,entry_at,exit_at,cost
            FROM observations WHERE label_at IS NULL AND exit_at + INTERVAL 1 HOUR <= ?
        """,
            [now],
        ).fetchall()
        for opportunity, horizon, experiment, symbol, entry_at, _exit_at, cost in pending:
            window = [
                available.get((symbol, entry_at + timedelta(hours=hour)))
                for hour in range(horizon + 1)
            ]
            if any(bar is None for bar in window):
                continue
            gross = (window[-1].open / window[0].open - 1) * 10_000
            evidence = json.dumps([bar.model_dump(mode="json") for bar in window])
            self.con.execute(
                """
                UPDATE observations SET label_at=?,gross_bps=?,net_bps=?,label_evidence_json=?
                WHERE opportunity=? AND horizon=? AND experiment=? AND label_at IS NULL
            """,
                [
                    now,
                    gross,
                    None if cost is None else gross - cost,
                    evidence,
                    opportunity,
                    horizon,
                    experiment,
                ],
            )

    def summary(self, *, now: datetime) -> ForwardSummary:
        start, opportunities, total, mature, waiting = self.con.execute(
            """
            SELECT min(published_at),count(DISTINCT opportunity),count(*),count(label_at),
              count(*) FILTER (WHERE label_at IS NULL AND exit_at + INTERVAL 1 HOUR > ?)
            FROM observations
        """,
            [now],
        ).fetchone()
        groups = self.con.execute("""
            SELECT horizon,action,count(*),avg(gross_bps),avg(net_bps)
            FROM observations WHERE label_at IS NOT NULL
            GROUP BY horizon,action ORDER BY horizon,action
        """).fetchall()
        return ForwardSummary(
            started_at=start,
            registered_opportunities=opportunities,
            registered_horizon_observations=total,
            matured_observations=mature,
            waiting_observations=waiting,
            missing_label_observations=total - mature - waiting,
            by_group=[
                ForwardGroup(
                    horizon_hours=h, action=a, observations=n, gross_mean_bps=g, net_mean_bps=v
                )
                for h, a, n, g, v in groups
            ],
        )
