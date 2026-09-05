from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from quant_lab.costs.model import estimate_cost_from_lake
from quant_lab.data.lake import _all_parquet_files
from quant_lab.decision.contracts import SYMBOLS, CostObservation, HourBar, InputSnapshot
from quant_lab.decision.current_inputs import CurrentInputs
from quant_lab.decision.storage import (
    MAX_INPUT_BYTES,
    MAX_RESULT_BYTES,
    atomic_json,
    input_identity,
    load_input,
    load_result,
    read_json,
    validate_result,
)
from quant_lab.export_plane.signatures import load_public_key, load_signing_key, sign_payload


def read_hour_bars(lake_root: Path, *, now: datetime, days: int) -> list[HourBar]:
    if not 1 <= days <= 365:
        raise ValueError("history window must be between 1 and 365 days")
    files = _all_parquet_files(lake_root / "silver" / "market_bar")
    if len(files) > 5_000:
        raise ValueError("market file count exceeds decision-job budget; compact on NAS")
    if not files:
        return []
    # Projection/predicate pushdown keeps the online producer bounded. A broken
    # input file fails the job instead of being silently removed from its evidence.
    with duckdb.connect(config={"memory_limit": "256MB", "threads": "1"}) as con:
        con.execute("SET TimeZone='UTC'")
        rows = con.execute(
            """
            SELECT replace(symbol, '-', '') AS symbol, cast(ts AS TIMESTAMPTZ) AS ts,
                   open, high, low, close, volume, cast(ingest_ts AS TIMESTAMPTZ) AS ingest_ts
            FROM read_parquet(?, union_by_name=true, hive_partitioning=false)
            WHERE lower(venue)='okx' AND lower(market_type)='spot'
              AND lower(timeframe)='1h' AND is_closed=true
              AND replace(symbol, '-', '') IN ('BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT')
              AND cast(ts AS TIMESTAMPTZ) >= ?
              AND cast(ts AS TIMESTAMPTZ) + INTERVAL 1 HOUR <= ?
              AND cast(ingest_ts AS TIMESTAMPTZ) <= ?
            QUALIFY row_number() OVER (
                PARTITION BY replace(symbol, '-', ''), ts ORDER BY ingest_ts DESC
            )=1
            ORDER BY symbol, ts
            LIMIT ?
            """,
            [
                [str(path) for path in files],
                now - timedelta(days=days),
                now,
                now,
                days * 24 * 4 + 1,
            ],
        ).fetchall()
    if len(rows) > days * 24 * 4:
        raise ValueError("hour-bar row budget exceeded")
    columns = ("symbol", "ts", "open", "high", "low", "close", "volume", "ingest_ts")
    return [HourBar(**dict(zip(columns, row, strict=True))) for row in rows]


def collect_costs(lake_root: Path, *, notional_usdt: float) -> list[CostObservation]:
    costs = []
    for symbol in SYMBOLS:
        estimate = estimate_cost_from_lake(
            lake_root,
            symbol=symbol,
            regime="normal",
            notional_usdt=notional_usdt,
            quantile="p75",
        )
        costs.append(
            CostObservation(
                symbol=symbol,
                roundtrip_bps=estimate.roundtrip_all_in_cost_bps,
                notional_usdt=notional_usdt,
                source=estimate.cost_source or estimate.source,
                quality=estimate.cost_quality,
                version=estimate.cost_model_version,
                # A global default has no observed timestamp. The legacy estimator
                # stamps "now" for API availability; that is not new cost evidence.
                as_of=None if estimate.cost_source == "global_default" else estimate.as_of_ts,
                trusted_for_paper=estimate.cost_trusted_for_paper,
                actual_sample_count=estimate.live_cost_sample_count,
                missing_reasons=estimate.cost_trust_block_reasons[:20],
            )
        )
    return costs


def publish_input(
    lake_root: Path,
    root: Path,
    *,
    signing_key: Path,
    code_revision: str,
    now: datetime | None = None,
    notional_usdt: float = 20,
    current_inputs: CurrentInputs | None = None,
) -> InputSnapshot:
    current = current_inputs.generated_at if current_inputs else now or datetime.now(UTC)
    key = load_signing_key(signing_key)
    bars = current_inputs.bars if current_inputs else read_hour_bars(lake_root, now=current, days=8)
    costs = current_inputs.costs if current_inputs else collect_costs(
        lake_root, notional_usdt=notional_usdt
    )
    warnings = (current_inputs.warnings if current_inputs else []) + [
        f"{symbol}:CURRENT_MARKET_MISSING"
        for symbol in SYMBOLS
        if not any(bar.symbol == symbol for bar in bars)
    ]
    value = InputSnapshot(
        snapshot_id="input-" + "0" * 64,
        generated_at=current,
        producer_commit=code_revision,
        bars=bars,
        costs=costs,
        warnings=warnings,
        signature="pending",
    )
    value = value.model_copy(update={"snapshot_id": input_identity(value)})
    path = root / "inputs" / (value.snapshot_id + ".json")
    if path.exists():
        # Identical market facts never acquire a new generation timestamp by refresh.
        value = load_input(path, key.public_key())
    else:
        value = value.model_copy(update={"signature": sign_payload(value, key)})
        if len(value.model_dump_json().encode()) > MAX_INPUT_BYTES:
            raise ValueError("prepared input exceeds transfer budget")
        atomic_json(path, value)
    atomic_json(root / "current-input.json", value)
    return value


def accept_results(
    root: Path,
    *,
    worker_public_key: Path,
    input_public_key: Path,
    publication_root: Path | None = None,
    publication_signing_key: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    current = now or datetime.now(UTC)
    worker_key, producer_key = load_public_key(worker_public_key), load_public_key(input_public_key)
    files = sorted((root / "inbox").glob("result-*.json"))
    if len(files) > 5_000:
        raise ValueError("inbox exceeds file budget")
    accepted = []
    rejected = []
    current_path = root / "current-result.json"
    public = publication_root or root
    # publication.json is the atomic public commit point. Recover a crash between
    # that replace and writing workflow pointers/receipts without inventing a time.
    if (public / "publication.json").exists():
        envelope = read_json(public / "publication.json", max_bytes=MAX_RESULT_BYTES + 4096)
        committed = validate_result(envelope["result"], worker_key)
        receipt = envelope["publication"]
        if receipt["result_id"] != committed.result_id or not receipt.get("published_at"):
            raise ValueError("publication receipt does not bind its result")
        receipt_path = root / "receipts" / (committed.result_id + ".json")
        if not receipt_path.exists():
            atomic_json(receipt_path, receipt)
            atomic_json(current_path, committed)
        for advice in committed.advice:
            detail = public / "advice" / (advice.advice_id + ".json")
            if not detail.exists():
                atomic_json(
                    detail, {"advice": advice.model_dump(mode="json"), "publication": receipt}
                )
    previous = load_result(current_path, worker_key) if current_path.exists() else None
    for path in files:
        if (root / "receipts" / path.name).exists():
            # An acknowledged duplicate transfer has no independent evidence value.
            duplicate = load_result(path, worker_key)
            if duplicate.result_id + ".json" != path.name:
                raise ValueError("duplicate filename mismatch")
            path.unlink()
            continue
        try:
            result = load_result(path, worker_key)
            if path.name != result.result_id + ".json":
                raise ValueError("result filename does not match identity")
            inputs = load_input(
                root / "inputs" / (result.input_snapshot_id + ".json"), producer_key
            )
            if result.generated_at > current + timedelta(seconds=5):
                raise ValueError("result timestamp is in the future")
            if result.generated_at < inputs.generated_at:
                raise ValueError("result predates input")
            if result.worker_commit != inputs.producer_commit:
                raise ValueError("worker and producer code versions differ")
            for advice in result.advice:
                market = max(
                    (bar for bar in inputs.bars if bar.symbol == advice.symbol),
                    key=lambda bar: bar.ts,
                    default=None,
                )
                if advice.market_asof != (market.ts + timedelta(hours=1) if market else None):
                    raise ValueError("advice market context is not the signed input")
                if advice.reference_entry_at != (
                    advice.market_asof + timedelta(hours=1) if advice.market_asof else None
                ):
                    raise ValueError("advice delay differs from the frozen experiment")
                source_cost = next(
                    (cost for cost in inputs.costs if cost.symbol == advice.symbol), None
                )
                if source_cost is not None and advice.cost != source_cost:
                    raise ValueError("advice cost differs from signed input")
                if (market is None or source_cost is None) and advice.action != "NO_VIEW":
                    raise ValueError("missing signed input requires NO_VIEW")
            atomic_json(root / "results" / path.name, result)
            receipt = {
                "status": "accepted",
                "result_id": result.result_id,
                "input_snapshot_id": result.input_snapshot_id,
                "accepted_at": current.isoformat(),
                "published_at": None,
            }
            previous_input = (
                load_input(root / "inputs" / (previous.input_snapshot_id + ".json"), producer_key)
                if previous is not None
                else None
            )
            advances = previous is None or (
                result.generated_at > previous.generated_at
                and inputs.generated_at >= previous_input.generated_at
            )
            if advances:
                receipt["published_at"] = current.isoformat()
                envelope = {"result": result.model_dump(mode="json"), "publication": receipt}
                atomic_json(public / "publication.json", envelope)
                for advice in result.advice:
                    detail = public / "advice" / (advice.advice_id + ".json")
                    if not detail.exists():
                        atomic_json(
                            detail,
                            {"advice": advice.model_dump(mode="json"), "publication": receipt},
                        )
                atomic_json(current_path, result)
                previous = result
            atomic_json(root / "receipts" / path.name, receipt)
            # The immutable result and receipt now exist; inbox is transfer staging.
            path.unlink()
            accepted.append(result.result_id)
        except (ValueError, OSError) as exc:
            rejected.append({"file": path.name, "reason": str(exc)[:500]})
    status = {
        "checked_at": current.isoformat(),
        "accepted": accepted,
        "rejected": rejected,
        "status": "WARNING" if rejected else "OK",
    }
    atomic_json(root / "accept-status.json", status)
    publications = []
    for path in sorted((root / "receipts").glob("result-*.json")):
        receipt = read_json(path, max_bytes=2_048)
        if receipt.get("published_at"):
            publications.append(receipt)
    if len(publications) > 5_000:
        raise ValueError("publication receipt budget exceeded; archive acknowledged history")
    value = {"publications": publications}
    if publication_signing_key is not None:
        value["signature"] = sign_payload(value, load_signing_key(publication_signing_key))
    atomic_json(root / "publication-receipts.json", value)
    return status


def read_current_input(root: Path) -> dict[str, Any]:
    return read_json(root / "current-input.json", max_bytes=MAX_INPUT_BYTES)
