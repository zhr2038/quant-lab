import subprocess
from datetime import UTC, datetime, timedelta

import pytest
import typer
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.decision.contracts_v2 import parse_result
from quant_lab.decision.contracts_v3 import AnalysisResultV3
from quant_lab.decision.ledger import Ledger
from quant_lab.decision.observation_diagnostics import mature_non_overlapping, registration_coverage
from quant_lab.decision.storage import atomic_json, load_result, read_json, result_identity
from quant_lab.decision.worker import Transport
from quant_lab.export_plane.signatures import sign_payload
from tests import test_decision_pipeline as fixtures
from tests.test_decision_operations import scoped_summary

artifacts = fixtures.artifacts

START = datetime(2026, 9, 5, 20, tzinfo=UTC)


def observation(hour, horizon=4, *, symbol="BTCUSDT", labeled=True, net=10, action="DEFER"):
    entry = START + timedelta(hours=hour)
    exit_at = entry + timedelta(hours=horizon)
    return {
        "symbol": symbol,
        "horizon": horizon,
        "opportunity": f"{symbol}-{hour}",
        "action": action,
        "entry_at": entry,
        "exit_at": exit_at,
        "published_at": entry - timedelta(minutes=50),
        "label_at": exit_at + timedelta(hours=1) if labeled else None,
        "gross_bps": net + 20 if labeled else None,
        "net_bps": net if labeled else None,
    }


def test_registration_counts_internal_and_trailing_gaps_without_backfilling():
    # Complete hours 0 and 2; hour 3 is partial; hour 5 is ahead of its deadline.
    rows = [observation(hour, h) for hour in (0, 2, 5) for h in (4, 24)]
    rows.append(observation(3, 4))
    # First publication delay is counted once per symbol/hour, not once per horizon.
    for row in rows:
        if row["entry_at"] == START + timedelta(hours=2):
            row["published_at"] = row["entry_at"] - timedelta(minutes=30)
    coverage = registration_coverage(rows, now=START + timedelta(hours=4, minutes=30))
    btc = coverage.by_symbol[0]
    assert (btc.expected_hours, btc.registered_hours, btc.missing_hours) == (5, 2, 3)
    assert btc.coverage_fraction == 0.4
    assert btc.partial_horizon_hours == 1 and btc.pending_deadline_hours == 1
    assert btc.late_publication_hours == 1 and btc.maximum_publication_delay_seconds == 1800
    assert [(gap.hours, gap.first_entry_at) for gap in btc.gaps] == [
        (1, START + timedelta(hours=1)),
        (2, START + timedelta(hours=3)),
    ]
    assert all(row.missing_hours == 5 for row in coverage.by_symbol[1:])
    assert coverage.status == "GAPS" and coverage.backfilled_forward_observations == 0
    with pytest.raises(ValueError, match="coverage status"):
        type(coverage).model_validate({**coverage.model_dump(), "status": "COMPLETE"})
    # No elapsed deadline means no artificial missing sample or full coverage claim.
    early = registration_coverage(rows[:2], now=START - timedelta(minutes=1))
    assert early.by_symbol[0].expected_hours == 0
    assert early.by_symbol[0].coverage_fraction is None
    assert registration_coverage([], now=START).status == "NOT_STARTED"


def test_mature_nonoverlap_selection_does_not_replace_missing_or_losing_outcomes():
    rows = [
        observation(0, labeled=False),
        observation(1, net=999),
        observation(4, net=-20),
        observation(8, net=40, action="KEEP_BASELINE"),
        observation(24, 24, net=900),
        observation(0, 24, net=50),
    ]
    now = START + timedelta(hours=30)
    groups = mature_non_overlapping(rows, now=now)
    defer = next(g for g in groups if g.horizon_hours == 4 and g.action == "DEFER")
    assert defer.selected_observations == 2
    assert defer.matured_observations == 1 and defer.missing_label_observations == 1
    assert defer.net_mean_bps == -20  # the overlapping +999 cannot replace missing entry 0
    day = next(g for g in groups if g.horizon_hours == 24)
    assert day.selected_observations == 2 and day.matured_observations == 1
    assert day.waiting_observations == 1 and day.net_mean_bps == 50
    assert sum(g.matured_observations for g in groups) == 3


def test_v3_statistics_preserve_archived_v2_signature_and_identity(artifacts, monkeypatch):
    a = artifacts
    original = a["result"].model_dump(mode="json")
    assert parse_result(original).model_dump(mode="json") == original
    assert load_result(a["result_path"], a["worker"].public_key()) == a["result"]
    with Ledger(a["root"] / "forward.duckdb") as ledger:
        ledger.register(a["result"], published_at=a["now"], now=a["now"])
        forward = scoped_summary(ledger, a, now=a["now"])
    value = {
        **original,
        "schema_version": "qlab.decision.result.v3",
        "forward": forward.model_dump(mode="json"),
    }
    result = AnalysisResultV3.model_validate(value)
    result = result.model_copy(update={"result_id": result_identity(result)})
    result = result.model_copy(update={"signature": sign_payload(result, a["worker"])})
    atomic_json(a["root"] / "v3.json", result)
    assert load_result(a["root"] / "v3.json", a["worker"].public_key()) == result
    assert result.advice == a["result"].advice
    assert result.forward.cross_symbol_independence is False
    assert all(advice.eligibility.live_execution_eligible is False for advice in result.advice)
    assert read_json(a["result_path"], max_bytes=512 * 1024) == original
    # The same signed v3 survives actual cloud acceptance and the read-only API.
    a["result_path"].unlink()
    atomic_json(a["root"] / "inbox" / (result.result_id + ".json"), result)
    from quant_lab.decision.pipeline import accept_results

    accepted = accept_results(
        a["root"],
        worker_public_key=a["worker_public_key"],
        input_public_key=a["input_public_key"],
        publication_root=a["root"] / "lake/gold/decision_reference",
        now=a["now"],
    )
    assert accepted["accepted"] == [result.result_id] and not accepted["rejected"]
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(a["root"] / "lake"))
    monkeypatch.setenv("QUANT_LAB_DECISION_WORKER_PUBLIC_KEY", str(a["worker_public_key"]))
    with TestClient(create_app()) as client:
        response = client.get("/v1/trade-advice/latest")
    assert response.status_code == 200
    view = response.json()
    assert view["reference_contracts"][0]["reference_schema"] == "qlab.decision.result.v3"
    assert view["forward"]["registration"] == result.forward.registration.model_dump(mode="json")


def test_retention_integrity_failure_does_not_stop_current_input_or_result_acceptance(
    tmp_path, monkeypatch
):
    from quant_lab.decision.jobs import cloud_cycle

    calls = []
    monkeypatch.setattr(
        "quant_lab.decision.jobs.prune_acknowledged",
        lambda *a, **kw: (_ for _ in ()).throw(ValueError("bad ack")),
    )
    monkeypatch.setattr(
        "quant_lab.decision.jobs.accept_results",
        lambda *a, **kw: calls.append("accept") or {"rejected": []},
    )
    monkeypatch.setattr(
        "quant_lab.decision.jobs.publish_current",
        lambda *a, **kw: calls.append("input") or {"status": "OK"},
    )
    with pytest.raises(typer.Exit) as error:
        cloud_cycle(
            code_revision="abcdef1", lake_root=tmp_path, job_root=tmp_path, private_root=tmp_path
        )
    assert error.value.exit_code == 2
    assert calls == ["accept", "input"]
    status = read_json(tmp_path / "retention-status.json", max_bytes=4096)
    assert status["status"] == "BLOCKED_RETENTION_ERROR" and status["removed"] is None
    assert status["removal_count_known"] is False


def test_transport_retries_once_for_transient_failure_but_not_authentication(monkeypatch):
    calls, sleeps = [], []
    outcomes = [
        subprocess.CompletedProcess([], 255, b"", b"Connection reset by peer"),
        subprocess.CompletedProcess([], 0, b"ok", b""),
    ]

    def run(*args, **kwargs):
        calls.append(args)
        return outcomes.pop(0)

    monkeypatch.setattr("quant_lab.decision.worker.subprocess.run", run)
    monkeypatch.setattr("quant_lab.decision.worker.time.sleep", sleeps.append)
    assert Transport._run(["ssh", "host"]).stdout == b"ok"
    assert len(calls) == 2 and sleeps == [2]
    outcomes.append(subprocess.CompletedProcess([], 255, b"", b"Permission denied (publickey)"))
    with pytest.raises(RuntimeError, match="Permission denied"):
        Transport._run(["ssh", "host"])
    assert len(calls) == 3 and sleeps == [2]
