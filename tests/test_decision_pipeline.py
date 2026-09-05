import json
from datetime import timedelta

import polars as pl
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from quant_lab.decision.contracts import HORIZONS, SYMBOLS
from quant_lab.decision.contracts_v2 import (
    EXPERIMENT_VERSION,
    STRATEGY_VERSION,
    AnalysisResultV2,
    ScopedForwardSummary,
)
from quant_lab.decision.engine import build_advice
from quant_lab.decision.pipeline import accept_results, read_hour_bars
from quant_lab.decision.storage import (
    atomic_json,
    effective_snapshot,
    input_identity,
    load_input,
    load_result,
    read_json,
    result_identity,
)
from quant_lab.export_plane.signatures import sign_payload
from tests.test_decision_engine import inputs, series


def test_default_cost_does_not_acquire_a_fresh_observation_timestamp(tmp_path, monkeypatch):
    from quant_lab.costs.model import _global_default_estimate
    from quant_lab.decision.pipeline import collect_costs

    monkeypatch.setattr(
        "quant_lab.decision.pipeline.estimate_cost_from_lake",
        lambda root, **kw: _global_default_estimate(**kw),
    )
    costs = collect_costs(tmp_path, notional_usdt=20)
    assert all(c.as_of is None and not c.trusted_for_paper for c in costs)


def public_file(directory, name, key):
    path = directory / name
    path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return path


@pytest.fixture
def artifacts(tmp_path):
    producer, worker = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    bars = series(1_600)
    now = bars[-1].ingest_ts + timedelta(minutes=10)
    snapshot = inputs(bars, now)
    snapshot = snapshot.model_copy(update={"snapshot_id": input_identity(snapshot)})
    snapshot = snapshot.model_copy(update={"signature": sign_payload(snapshot, producer)})
    atomic_json(tmp_path / "inputs" / (snapshot.snapshot_id + ".json"), snapshot)
    result = AnalysisResultV2(
        result_id="result-" + "0" * 64,
        generated_at=now,
        input_snapshot_id=snapshot.snapshot_id,
        worker_commit="abcdef1",
        advice=[
            build_advice(bars, inputs=snapshot, symbol=symbol, horizon=horizon, now=now)
            for symbol in SYMBOLS
            for horizon in HORIZONS
        ],
        forward=ScopedForwardSummary(
            experiment=EXPERIMENT_VERSION,
            strategy_version=STRATEGY_VERSION,
            cost_versions=[snapshot.costs[0].version],
            published_from=now - timedelta(days=180),
            published_until=now,
            non_overlapping_opportunities=0,
            overlapping_price_observations=0,
        ),
        history_rows=len(bars),
        runtime_seconds=0.1,
        peak_rss_mib=10,
        signature="pending",
    )
    result = result.model_copy(update={"result_id": result_identity(result)})
    result = result.model_copy(update={"signature": sign_payload(result, worker)})
    path = tmp_path / "inbox" / (result.result_id + ".json")
    atomic_json(path, result)
    return {
        "root": tmp_path,
        "producer": producer,
        "worker": worker,
        "input": snapshot,
        "result": result,
        "result_path": path,
        "now": now,
        "bars": bars,
        "input_public_key": public_file(tmp_path, "producer.pub", producer),
        "worker_public_key": public_file(tmp_path, "worker.pub", worker),
    }


def accept(a):
    return accept_results(
        a["root"],
        worker_public_key=a["worker_public_key"],
        input_public_key=a["input_public_key"],
        now=a["now"],
    )


def test_signed_artifacts_accept_once_and_preserve_provenance(artifacts):
    a = artifacts
    assert (
        load_input(
            a["root"] / "inputs" / (a["input"].snapshot_id + ".json"), a["producer"].public_key()
        )
        == a["input"]
    )
    assert accept(a)["accepted"] == [a["result"].result_id]
    assert accept(a)["accepted"] == []
    current = load_result(a["root"] / "current-result.json", a["worker"].public_key())
    assert current == a["result"]
    assert current.input_snapshot_id == a["input"].snapshot_id


def test_signature_corruption_cannot_publish(artifacts):
    a = artifacts
    value = json.loads(a["result_path"].read_text(encoding="utf-8"))
    value["history_rows"] += 1
    atomic_json(a["result_path"], value)
    status = accept(a)
    assert status["rejected"]
    assert "signature" in status["rejected"][0]["reason"]
    assert not (a["root"] / "current-result.json").exists()


def test_valid_signature_cannot_grant_live_effect(artifacts):
    a = artifacts
    value = a["result"].model_dump(mode="json")
    value["advice"][0]["live_order_effect"] = "allow"
    value["result_id"] = result_identity(value)
    value["signature"] = sign_payload(value, a["worker"])
    a["result_path"].unlink()
    atomic_json(a["root"] / "inbox" / (value["result_id"] + ".json"), value)
    status = accept(a)
    assert status["rejected"]
    assert not (a["root"] / "current-result.json").exists()


def test_expired_view_does_not_change_original_forecast(artifacts):
    result = artifacts["result"]
    original = result.model_dump_json()
    expired = effective_snapshot(result, artifacts["now"] + timedelta(hours=2))
    assert expired["effective_status"] == "EXPIRED"
    assert all(row["effective_action"] == "NO_VIEW" for row in expired["advice"])
    assert result.model_dump_json() == original


def test_oversized_input_fails_before_json_parsing(tmp_path):
    path = tmp_path / "big.json"
    path.write_bytes(b" " * 101)
    with pytest.raises(ValueError, match="byte budget"):
        read_json(path, max_bytes=100)


def test_reader_uses_closed_okx_spot_hour_bars_only(tmp_path):
    bars = series(50)
    rows = []
    for bar in bars:
        row = bar.model_dump()
        row.update(venue="okx", market_type="spot", timeframe="1H", is_closed=True)
        rows.append(row)
    rows.extend([{**rows[0], "venue": "other"}, {**rows[0], "timeframe": "1m"}])
    path = tmp_path / "silver" / "market_bar"
    path.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(path / "data.parquet")
    selected = read_hour_bars(tmp_path, now=bars[-1].ts + timedelta(minutes=30), days=8)
    assert len(selected) == 49
    assert selected[-1].ts == bars[-2].ts


def test_frozen_v1_signature_and_identity_still_load():
    from pathlib import Path

    from quant_lab.export_plane.signatures import load_public_key

    fixture = Path(__file__).parent / "fixtures" / "decision_v1"
    raw = read_json(fixture / "signed-result.json", max_bytes=512 * 1024)
    loaded = load_result(fixture / "signed-result.json", load_public_key(fixture / "worker.pub"))
    assert loaded.schema_version == "qlab.decision.result.v1"
    assert loaded.model_dump(mode="json") == raw
    assert all("eligibility" not in a for a in raw["advice"])
