import json
import os
import shutil
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from fastapi.testclient import TestClient

from quant_lab.api.main import create_app
from quant_lab.decision.ledger import Ledger
from quant_lab.decision.pipeline import accept_results
from quant_lab.decision.retention import prune_acknowledged
from quant_lab.decision.storage import atomic_json, read_json, result_identity
from quant_lab.decision.worker import run_worker
from quant_lab.export_plane.signatures import sha256_file, sign_payload
from tests import test_decision_pipeline as fixtures
from tests.test_decision_engine import series

accept = fixtures.accept
artifacts = fixtures.artifacts


def test_cloud_publication_proceeds_independently_of_current_input_status(artifacts, monkeypatch):
    from quant_lab.decision.jobs import cloud_cycle
    from quant_lab.export_plane.signatures import verify_payload

    a = artifacts
    (a["root"] / "producer.key").write_bytes(
        a["producer"].private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setattr(
        "quant_lab.decision.jobs.publish_current",
        lambda *args, **kw: {"status": "WARNING"},
    )
    cloud_cycle(
        code_revision="abcdef1",
        lake_root=a["root"] / "lake",
        job_root=a["root"],
        private_root=a["root"],
    )
    publication = a["root"] / "lake/gold/decision_reference/publication.json"
    assert read_json(publication, max_bytes=1024**2)["result"]["result_id"] == a["result"].result_id
    receipts = read_json(a["root"] / "publication-receipts.json", max_bytes=1024**2)
    verify_payload(receipts, receipts["signature"], a["producer"].public_key())


def test_only_actual_timely_publication_registers_and_replays_do_not_add(artifacts):
    a = artifacts
    with Ledger(a["root"] / "forward.duckdb") as ledger:
        assert ledger.register(a["result"], published_at=a["now"], now=a["now"]) == 2
        assert ledger.register(a["result"], published_at=a["now"], now=a["now"]) == 0
        summary = scoped_summary(ledger, a, now=a["now"])
        assert summary.registered_opportunities == 1
        assert summary.registered_horizon_observations == 2
        assert summary.waiting_observations == 2
        assert summary.v5_received_count is None
        assert summary.incremental_account_pnl is None
    with Ledger(a["root"] / "late.duckdb") as ledger:
        later = a["now"] + timedelta(hours=2)
        assert ledger.register(a["result"], published_at=later, now=later) == 0
        with pytest.raises(ValueError, match="publication time"):
            ledger.register(a["result"], published_at=a["now"] - timedelta(seconds=1), now=later)


def test_labels_require_closed_observed_complete_window_and_freeze_original_cost(artifacts):
    a = artifacts
    bars = series(1_700)
    advice = next(v for v in a["result"].advice if v.symbol == "BTCUSDT" and v.horizon_hours == 4)
    matured = advice.reference_exit_at + timedelta(hours=1)
    with Ledger(a["root"] / "forward.duckdb") as ledger:
        ledger.register(a["result"], published_at=a["now"], now=a["now"])
        ledger.mature(bars, now=matured - timedelta(seconds=1))
        assert (
            scoped_summary(ledger, a, now=matured - timedelta(seconds=1)).matured_observations == 0
        )
        missing = [v for v in bars if v.ts != advice.reference_entry_at + timedelta(hours=1)]
        ledger.mature(missing, now=matured)
        assert scoped_summary(ledger, a, now=matured).missing_label_observations == 1
        ledger.mature(bars, now=matured)
        summary = scoped_summary(ledger, a, now=matured)
        assert summary.matured_observations == 1
        assert summary.net_mean_bps is None
        group = summary.by_group[0]
        assert group.horizon_hours == 4
        assert group.net_mean_bps == pytest.approx(group.gross_mean_bps - advice.cost.roundtrip_bps)
        # Later corrected prices do not rewrite a sealed outcome.
        corrected = [v.model_copy(update={"open": v.open * 2}) for v in bars]
        ledger.mature(corrected, now=matured)
        assert scoped_summary(ledger, a, now=matured) == summary


def test_api_reads_only_published_gold_and_fails_closed_on_corruption(artifacts, monkeypatch):
    a = artifacts
    gold = a["root"] / "lake" / "gold" / "decision_reference"
    accept_results(
        a["root"],
        worker_public_key=a["worker_public_key"],
        input_public_key=a["input_public_key"],
        publication_root=gold,
        now=a["now"],
    )
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(a["root"] / "lake"))
    monkeypatch.setenv("QUANT_LAB_DECISION_WORKER_PUBLIC_KEY", str(a["worker_public_key"]))
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "only-test-token")
    with TestClient(create_app()) as client:
        response = client.get("/v1/trade-advice/latest")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["effective_status"] == "EXPIRED"
        assert all(v["effective_action"] == "NO_VIEW" for v in response.json()["advice"])
        advice_id = response.json()["advice"][0]["advice_id"]
        detail = client.get(f"/v1/trade-advice/{advice_id}")
        assert detail.status_code == 200
        assert detail.json()["advice_id"] == advice_id
        assert detail.json()["effective_action"] == "NO_VIEW"
        raw = read_json(gold / "publication.json", max_bytes=1024**2)
        raw["result"]["history_rows"] += 1
        atomic_json(gold / "publication.json", raw)
        response = client.get("/v1/trade-advice/latest")
        assert response.status_code == 503
        assert response.json()["advice"] == []
        assert client.get("/assets/decision/not-a-file").status_code == 404


def test_api_empty_result_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path))
    monkeypatch.delenv("QUANT_LAB_API_TOKEN", raising=False)
    with TestClient(create_app()) as client:
        data = client.get("/v1/trade-advice/latest").json()
        assert data["effective_status"] == "NO_RESULT"
        assert data["advice"] == []


def test_public_workbench_does_not_open_other_methods_or_strategy_apis(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_LAB_LAKE_ROOT", str(tmp_path))
    monkeypatch.setenv("QUANT_LAB_API_TOKEN", "private-strategy-token")
    with TestClient(create_app()) as client:
        assert client.get("/v1/trade-advice/latest").status_code == 200
        assert client.get("/v1/trade-advice/latest?view=web").status_code == 200
        assert client.get("/v1/trade-advice/advice-" + "a" * 64).status_code == 404
        assert client.post("/v1/trade-advice/latest").status_code == 401
        assert client.post("/v1/trade-advice/advice-" + "a" * 64).status_code == 401
        assert client.get("/v1/trade-advice/latest/private").status_code == 401
        assert client.get("/v1/catalog/datasets").status_code == 401
        assert (
            client.get(
                "/v1/catalog/datasets", headers={"Authorization": "Bearer private-strategy-token"}
            ).status_code
            == 200
        )


def test_public_workbench_respects_explicit_ip_restrictions(monkeypatch):
    monkeypatch.setenv("QUANT_LAB_ALLOWED_CLIENT_IPS", "192.0.2.1")
    with TestClient(create_app()) as client:
        assert client.get("/v1/trade-advice/latest").status_code == 403


def test_published_receipt_recovery_preserves_actual_time(artifacts):
    a = artifacts
    accept(a)
    receipt_path = a["root"] / "receipts" / (a["result"].result_id + ".json")
    receipt = read_json(receipt_path, max_bytes=4096)
    receipt_path.unlink()
    (a["root"] / "current-result.json").unlink()
    a["now"] += timedelta(minutes=10)
    accept(a)
    assert read_json(receipt_path, max_bytes=4096) == receipt
    assert (a["root"] / "current-result.json").exists()


def test_newer_result_cannot_bind_old_or_changed_cost_inputs(artifacts):
    a = artifacts
    value = a["result"].model_dump(mode="json")
    value["worker_commit"] = "different-commit"
    value["result_id"] = result_identity(value)
    value["signature"] = sign_payload(value, a["worker"])
    a["result_path"].unlink()
    atomic_json(a["root"] / "inbox" / (value["result_id"] + ".json"), value)
    assert "versions differ" in accept(a)["rejected"][0]["reason"]


def test_retention_requires_signed_exact_nas_readback_and_preserves_current(artifacts):
    a = artifacts
    accept(a)
    current = a["root"] / "results" / (a["result"].result_id + ".json")
    old = a["root"] / "inputs" / ("input-" + "f" * 64 + ".json")
    old.write_text('{"archived": true}', encoding="utf-8")
    now = datetime.now(UTC)
    for path in (old, current):
        stamp = (now - timedelta(days=8)).timestamp()
        os.utime(path, (stamp, stamp))
    ack = {
        "schema_version": "qlab.decision.archive_ack.v1",
        "generated_at": now.isoformat(),
        "entries": [
            {"kind": "inputs", "name": old.name, "sha256": "0" * 64},
            {"kind": "results", "name": current.name, "sha256": sha256_file(current)},
        ],
    }
    ack["signature"] = sign_payload(ack, a["worker"])
    atomic_json(a["root"] / "inbox" / "archive-ack.json", ack)
    with pytest.raises(ValueError, match="does not match"):
        prune_acknowledged(a["root"], a["root"], worker_public_key=a["worker_public_key"], now=now)
    assert old.exists() and current.exists()
    ack["entries"][0]["sha256"] = sha256_file(old)
    ack["signature"] = sign_payload(ack, a["worker"])
    atomic_json(a["root"] / "inbox" / "archive-ack.json", ack)
    result = prune_acknowledged(
        a["root"], a["root"], worker_public_key=a["worker_public_key"], now=now
    )
    assert len(result["removed"]) == 1
    assert not old.exists() and current.exists()


def test_worker_archives_then_uploads_and_unchanged_input_never_renews_advice(
    artifacts, monkeypatch
):
    a = artifacts
    state, archive, private = [a["root"] / name for name in ("state", "nas", "private")]
    private.mkdir()
    shutil.copyfile(a["input_public_key"], private / "producer.pub")
    (private / "worker.key").write_bytes(
        a["worker"].private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    monkeypatch.setattr("quant_lab.decision.worker.read_hour_bars", lambda *args, **kw: a["bars"])
    monkeypatch.setattr("quant_lab.decision.worker.peak_rss_mib", lambda: 20)
    monkeypatch.setattr(
        "quant_lab.decision.worker.check_capacity",
        lambda s, ar: [p.mkdir(exist_ok=True) for p in (s, ar)],
    )
    publications = {"publications": []}
    uploaded = []

    class FakeTransport:
        def pull(self, root):
            atomic_json(root / "current-input.json", a["input"])
            value = dict(publications)
            value["signature"] = sign_payload(value, a["producer"])
            atomic_json(root / "publication-receipts.json", value)

        def push(self, path, *, name):
            assert path.exists()
            uploaded.append(name)
            assert json.loads(path.read_text(encoding="utf-8"))["signature"]

    args = {
        "state": state,
        "archive": archive,
        "bootstrap": a["root"],
        "private_root": private,
        "code_revision": "abcdef1",
        "transport": FakeTransport(),
    }
    first = run_worker(**args, now=a["now"])
    publications["publications"] = [
        {"result_id": first.result_id, "published_at": a["now"].isoformat()}
    ]
    second = run_worker(**args, now=a["now"] + timedelta(minutes=5))
    assert second.advice == first.advice
    assert second.forward.registered_opportunities == 1
    publications["publications"].append(
        {"result_id": second.result_id, "published_at": second.generated_at.isoformat()}
    )
    third = run_worker(**args, now=a["now"] + timedelta(minutes=10))
    assert third == second
    assert uploaded.count(second.result_id + ".json") == 1


def scoped_summary(ledger, a, *, now):
    from quant_lab.decision.contracts_v2 import EXPERIMENT_VERSION, STRATEGY_VERSION

    return ledger.summary(
        now=now,
        experiment=EXPERIMENT_VERSION,
        strategy_version=STRATEGY_VERSION,
        cost_versions=[a["input"].costs[0].version],
        published_from=a["now"] - timedelta(days=180),
        published_until=now,
    )


def test_registered_opportunity_cannot_silently_change_versions(artifacts):
    a = artifacts
    with Ledger(a["root"] / "immutable-scope.duckdb") as ledger:
        assert ledger.register(a["result"], published_at=a["now"], now=a["now"]) == 2
        changed = a["result"].model_copy(
            update={
                "advice": [
                    advice.model_copy(
                        update={"cost": advice.cost.model_copy(update={"version": "changed-cost"})}
                    )
                    for advice in a["result"].advice
                ]
            }
        )
        with pytest.raises(ValueError, match="new experiment"):
            ledger.register(changed, published_at=a["now"], now=a["now"])
        with pytest.raises(ValueError, match="archived v1"):
            ledger.register(
                changed, published_at=a["now"], now=a["now"], allow_legacy_v1_replay=True
            )
        assert scoped_summary(ledger, a, now=a["now"]).registered_horizon_observations == 2


def test_archived_v1_cost_refresh_replay_keeps_first_observation(tmp_path):
    from pathlib import Path

    from quant_lab.decision.storage import load_result
    from quant_lab.export_plane.signatures import load_public_key

    fixture = Path(__file__).parent / "fixtures/decision_v1"
    original = load_result(fixture / "signed-result.json", load_public_key(fixture / "worker.pub"))
    current = original.generated_at
    changed = original.model_copy(
        update={
            "advice": [
                a.model_copy(update={"cost": a.cost.model_copy(update={"version": "refreshed"})})
                for a in original.advice
            ]
        }
    )
    with Ledger(tmp_path / "archive.duckdb") as ledger:
        inserted = ledger.register(original, published_at=current, now=current)
        assert inserted > 0
        query = "SELECT * FROM observations ORDER BY opportunity,horizon"
        before = ledger.con.execute(query).fetchall()
        with pytest.raises(ValueError, match="new experiment"):
            ledger.register(changed, published_at=current, now=current)
        assert ledger.register(
            changed, published_at=current, now=current, allow_legacy_v1_replay=True
        ) == 0
        assert ledger.legacy_replay_preserved == inserted
        assert ledger.con.execute(query).fetchall() == before


def test_forward_summary_never_mixes_experiments_or_cost_versions(artifacts):
    from quant_lab.decision.contracts_v2 import STRATEGY_VERSION

    a = artifacts
    with Ledger(a["root"] / "scopes.duckdb") as ledger:
        assert ledger.register(a["result"], published_at=a["now"], now=a["now"]) == 2
        ledger.con.execute(
            "UPDATE observations SET label_at=?,gross_bps=130,net_bps=100", [a["now"]]
        )
        ledger.con.execute(
            "INSERT INTO observations SELECT opportunity,horizon,'opposite-experiment',advice_id,"
            "symbol,action,published_at,entry_at,exit_at,cost,advice_json,label_at,-970,-1000,"
            "label_evidence_json,strategy_version,cost_version FROM observations"
        )
        positive = scoped_summary(ledger, a, now=a["now"])
        assert (
            positive.registered_opportunities == 1 and positive.registered_horizon_observations == 2
        )
        assert positive.non_overlapping_opportunities == 1 and positive.actual_trades is None
        assert all(group.net_mean_bps == 100 for group in positive.by_group)
        negative = ledger.summary(
            now=a["now"],
            experiment="opposite-experiment",
            strategy_version=STRATEGY_VERSION,
            cost_versions=[a["input"].costs[0].version],
            published_from=a["now"] - timedelta(days=180),
            published_until=a["now"],
        )
        assert all(group.net_mean_bps == -1000 for group in negative.by_group)
        other_cost = ledger.summary(
            now=a["now"],
            experiment=positive.experiment,
            strategy_version=STRATEGY_VERSION,
            cost_versions=["unseen-version"],
            published_from=positive.published_from,
            published_until=a["now"],
        )
        assert other_cost.registered_horizon_observations == 0
