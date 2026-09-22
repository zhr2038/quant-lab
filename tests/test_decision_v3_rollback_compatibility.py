"""Rollback retains v2 producer behavior while preserving already signed v3 history."""

import shutil
from datetime import timedelta

from cryptography.hazmat.primitives import serialization

from quant_lab.decision.contracts import SYMBOLS
from quant_lab.decision.contracts_v2 import parse_result
from quant_lab.decision.contracts_v3 import AnalysisResultV3
from quant_lab.decision.pipeline import accept_results
from quant_lab.decision.storage import atomic_json, load_result, read_json, result_identity
from quant_lab.decision.worker import run_worker
from quant_lab.export_plane.signatures import sign_payload
from tests import test_decision_pipeline as fixtures

artifacts = fixtures.artifacts


def signed_v3(a):
    raw = a["result"].model_dump(mode="json")
    raw["schema_version"] = "qlab.decision.result.v3"
    raw["forward"].update(
        registration={
            "first_entry_at": None,
            "due_through_entry_at": a["now"].replace(minute=0, second=0, microsecond=0),
            "status": "NOT_STARTED",
            "by_symbol": [
                {
                    "symbol": symbol,
                    "expected_hours": 0,
                    "registered_hours": 0,
                    "missing_hours": 0,
                    "partial_horizon_hours": 0,
                    "pending_deadline_hours": 0,
                    "late_publication_hours": 0,
                    "latest_registered_entry_at": None,
                    "gaps": [],
                    "omitted_gap_ranges": 0,
                }
                for symbol in SYMBOLS
            ],
        },
        mature_non_overlapping_by_group=[],
    )
    result = AnalysisResultV3.model_validate(raw)
    result = result.model_copy(update={"result_id": result_identity(result)})
    return result.model_copy(update={"signature": sign_payload(result, a["worker"])})


def test_rollback_reader_accepts_signed_v3_without_changing_v2_identity(artifacts):
    a = artifacts
    previous = a["result"].model_dump(mode="json")
    assert parse_result(previous).model_dump(mode="json") == previous
    result = signed_v3(a)
    a["result_path"].unlink()
    path = a["root"] / "inbox" / (result.result_id + ".json")
    atomic_json(path, result)
    assert load_result(path, a["worker"].public_key()) == result
    status = accept_results(
        a["root"],
        worker_public_key=a["worker_public_key"],
        input_public_key=a["input_public_key"],
        now=a["now"],
    )
    assert status["accepted"] == [result.result_id] and status["rejected"] == []
    assert read_json(a["root"] / "publication.json", max_bytes=1024**2)["result"] == (
        result.model_dump(mode="json")
    )


def test_rollback_worker_replays_v3_current_and_archive_then_emits_v2(artifacts, monkeypatch):
    a = artifacts
    state, archive, private = [a["root"] / name for name in ("state", "nas", "private")]
    private.mkdir()
    archive.mkdir()
    shutil.copyfile(a["input_public_key"], private / "producer.pub")
    (private / "worker.key").write_bytes(
        a["worker"].private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    prior = signed_v3(a)
    prior_path = archive / "results" / (prior.result_id + ".json")
    atomic_json(prior_path, prior)
    atomic_json(archive / "current-result.json", prior)
    before = prior_path.read_bytes()
    monkeypatch.setattr("quant_lab.decision.worker.read_hour_bars", lambda *args, **kw: a["bars"])
    monkeypatch.setattr("quant_lab.decision.worker.peak_rss_mib", lambda: 20)
    monkeypatch.setattr(
        "quant_lab.decision.worker.check_capacity",
        lambda s, ar: [p.mkdir(exist_ok=True) for p in (s, ar)],
    )
    uploaded = []

    class FakeTransport:
        def pull(self, root):
            atomic_json(root / "current-input.json", a["input"])
            value = {
                "publications": [
                    {"result_id": prior.result_id, "published_at": a["now"].isoformat()}
                ]
            }
            value["signature"] = sign_payload(value, a["producer"])
            atomic_json(root / "publication-receipts.json", value)

        def push(self, path, *, name):
            uploaded.append(name)

    result = run_worker(
        state=state,
        archive=archive,
        bootstrap=a["root"],
        private_root=private,
        code_revision="abcdef1",
        now=a["now"] + timedelta(minutes=5),
        transport=FakeTransport(),
    )
    assert result.schema_version == "qlab.decision.result.v2"
    assert "registration" not in result.forward.model_dump()
    assert result.forward.registered_horizon_observations == 2
    assert result.advice == prior.advice
    assert prior_path.read_bytes() == before
    # The old acknowledgement scan successfully verifies and retains the v3 archive.
    ack = read_json(archive / "archive-ack.json", max_bytes=2 * 1024**2)
    assert any(entry["name"] == prior_path.name for entry in ack["entries"])
    assert uploaded == ["archive-ack.json", result.result_id + ".json"]
