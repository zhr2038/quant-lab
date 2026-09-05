"""Prospective price labels recorded only after actual cloud publication.

This ledger neither simulates a portfolio nor attributes returns to V5.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import duckdb

from quant_lab.contracts.models import require_utc
from quant_lab.decision.contracts import AnalysisResult, ForwardGroup, HourBar
from quant_lab.decision.contracts_v2 import STRATEGY_VERSION, ScopedForwardSummary


class Ledger:
    def __init__(self, path: Path):
        self.legacy_replay_preserved = 0
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

        # Additive migration: archived advice_json and primary-key identity are unchanged.
        self.con.execute("BEGIN TRANSACTION")
        try:
            self.con.execute(
                "ALTER TABLE observations ADD COLUMN IF NOT EXISTS strategy_version VARCHAR"
            )
            self.con.execute(
                "ALTER TABLE observations ADD COLUMN IF NOT EXISTS cost_version VARCHAR"
            )
            self.con.execute(
                "UPDATE observations SET strategy_version="
                "coalesce(json_extract_string(advice_json, '$.strategy_version'), ?) "
                "WHERE strategy_version IS NULL",
                [STRATEGY_VERSION],
            )
            self.con.execute(
                "UPDATE observations SET cost_version="
                "coalesce(json_extract_string(advice_json, '$.cost.version'), "
                "'legacy_unversioned') "
                "WHERE cost_version IS NULL"
            )
            self.con.execute("COMMIT")
        except Exception:
            self.con.execute("ROLLBACK")
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.con.close()

    def register(
        self,
        result: AnalysisResult,
        *,
        published_at: datetime,
        now: datetime,
        allow_legacy_v1_replay: bool = False,
    ) -> int:
        # Only the signed archive reader may request first-observation compatibility.
        # New v2 experiments retain strict version admission even with this flag.
        if allow_legacy_v1_replay and result.schema_version != "qlab.decision.result.v1":
            raise ValueError("legacy replay compatibility requires an archived v1 result")
        if published_at < result.generated_at or published_at > now:
            raise ValueError("publication time is inconsistent with result availability")
        for advice in result.advice:
            existing = self.con.execute(
                "SELECT strategy_version,cost_version FROM observations "
                "WHERE opportunity=? AND horizon=? AND experiment=?",
                [advice.opportunity_id, advice.horizon_hours, advice.experiment_version],
            ).fetchone()
            bound = (getattr(advice, "strategy_version", STRATEGY_VERSION), advice.cost.version)
            if existing is not None and existing != bound:
                if allow_legacy_v1_replay and existing[0] == bound[0]:
                    # Retired v1 producers refreshed costs inside an opportunity. The
                    # old ledger sealed its first publication; never overwrite it or
                    # reinterpret the later receipt as a new observation.
                    self.legacy_replay_preserved += 1
                    continue
                raise ValueError("opportunity version changed; register a new experiment")
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
                   entry_at,exit_at,cost,advice_json,strategy_version,cost_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING RETURNING advice_id
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
                    getattr(advice, "strategy_version", STRATEGY_VERSION),
                    advice.cost.version,
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

    def summary(
        self,
        *,
        now: datetime,
        experiment: str,
        strategy_version: str,
        cost_versions: list[str],
        published_from: datetime,
        published_until: datetime,
    ) -> ScopedForwardSummary:
        now, published_from, published_until = map(
            require_utc, (now, published_from, published_until)
        )
        if not experiment or not strategy_version or not cost_versions or not all(cost_versions):
            raise ValueError(
                "forward summary requires explicit experiment, strategy and cost versions"
            )
        if published_from > published_until or published_until > now:
            raise ValueError("invalid summary interval")
        clause = (
            "experiment=? AND strategy_version=? AND cost_version IN ("
            + ",".join("?" for _ in cost_versions)
            + ") AND published_at>=? AND published_at<=?"
        )
        params = [experiment, strategy_version, *cost_versions, published_from, published_until]
        start, opportunities, total, mature, waiting = self.con.execute(
            """
            SELECT min(published_at),count(DISTINCT opportunity),count(*),
              count(*) FILTER (WHERE label_at<=?),
              count(*) FILTER (WHERE (label_at IS NULL OR label_at>?)
                AND exit_at + INTERVAL 1 HOUR > ?)
            FROM observations WHERE """
            + clause,
            [now, now, now, *params],
        ).fetchone()
        groups = self.con.execute(
            """
            SELECT horizon,action,count(*),avg(gross_bps),avg(net_bps)
            FROM observations WHERE label_at<=? AND """
            + clause
            + " GROUP BY horizon,action ORDER BY horizon,action",
            [now, *params],
        ).fetchall()
        windows = self.con.execute(
            "SELECT symbol,opportunity,min(entry_at),max(exit_at) FROM observations WHERE "
            + clause
            + " GROUP BY symbol,opportunity ORDER BY min(entry_at),symbol,opportunity",
            params,
        ).fetchall()
        last_exit, independent = {}, 0
        for symbol, _opportunity, entry, exit_at in windows:
            if symbol not in last_exit or entry >= last_exit[symbol]:
                independent += 1
                last_exit[symbol] = exit_at
        return ScopedForwardSummary(
            experiment=experiment,
            strategy_version=strategy_version,
            cost_versions=sorted(set(cost_versions)),
            published_from=published_from,
            published_until=published_until,
            non_overlapping_opportunities=independent,
            overlapping_price_observations=total - independent,
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
