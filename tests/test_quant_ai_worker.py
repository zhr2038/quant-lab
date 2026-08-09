from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from deploy.nas_ai_worker import worker
from quant_lab.ai_research.contracts import (
    AIResearchTask,
    DataCollectionProposal,
    EvidenceDocument,
    EvidenceReference,
    Stage1Diagnosis,
    Stage2ProposalSet,
    TaskPreflight,
    compute_task_packet_sha256,
)


class _Output(BaseModel):
    value: int


class _Response:
    is_error = False

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def json(self) -> dict[str, object]:
        return {
            "id": "response-test",
            "status": "completed",
            "output_text": self._output_text,
            "usage": {},
        }


def test_schema_retry_includes_structured_feedback(monkeypatch) -> None:
    requests: list[dict[str, object]] = []
    responses = iter([_Response('{"value":"invalid"}'), _Response('{"value":2}')])

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, _url: str, **kwargs: object) -> _Response:
            requests.append(kwargs["json"])  # type: ignore[arg-type]
            return next(responses)

    monkeypatch.setattr(worker.httpx, "Client", _Client)
    monkeypatch.setattr(worker, "_sleep", lambda _seconds: None)
    config = SimpleNamespace(
        model="gpt-test",
        reasoning_effort="xhigh",
        api_key="secret-never-logged",
        api_base_url="http://local/v1",
        api_timeout_seconds=60,
        api_retries=1,
    )

    result = worker._responses_call(
        config,
        system_prompt="system",
        user_payload={"task_id": "task-1"},
        output_model=_Output,
        schema_name="test_output",
        max_output_tokens=4000,
        stage="stage1",
    )

    assert result["attempts"] == 2
    assert result["validation_events"][0]["event"] == "SCHEMA_VALIDATION_FAILED"
    assert len(requests[1]["input"]) == 3  # type: ignore[arg-type]
    assert "secret-never-logged" not in str(requests)


def test_semantic_retry_includes_bounded_feedback(monkeypatch) -> None:
    requests: list[dict[str, object]] = []
    responses = iter([_Response('{"value":1}'), _Response('{"value":2}')])

    class _Client:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> _Client:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def post(self, _url: str, **kwargs: object) -> _Response:
            requests.append(kwargs["json"])  # type: ignore[arg-type]
            return next(responses)

    def _validate(output: BaseModel) -> None:
        if output.value != 2:  # type: ignore[attr-defined]
            raise worker._StructuredOutputSemanticError(
                [
                    {
                        "path": "value",
                        "type": "test_semantic_error",
                        "message": "value must equal 2",
                    }
                ]
            )

    monkeypatch.setattr(worker.httpx, "Client", _Client)
    monkeypatch.setattr(worker, "_sleep", lambda _seconds: None)
    config = SimpleNamespace(
        model="gpt-test",
        reasoning_effort="xhigh",
        api_key="secret-never-logged",
        api_base_url="http://local/v1",
        api_timeout_seconds=60,
        api_retries=1,
    )

    result = worker._responses_call(
        config,
        system_prompt="system",
        user_payload={"task_id": "task-1"},
        output_model=_Output,
        schema_name="test_output",
        max_output_tokens=4000,
        stage="stage2",
        semantic_validator=_validate,
    )

    assert result["attempts"] == 2
    assert result["validation_events"][0]["event"] == "SEMANTIC_VALIDATION_FAILED"
    retry_input = requests[1]["input"]  # type: ignore[index]
    assert "allowed_evidence_members" in str(retry_input)
    assert "secret-never-logged" not in str(requests)


def test_stage2_semantic_validation_rejects_unrouted_evidence() -> None:
    proposal = DataCollectionProposal(
        proposal_id="proposal-1",
        title="Collect execution evidence",
        observed_data_gap="Cost evidence is incomplete.",
        required_dataset="cost_evidence",
        required_fields=["fee_bps"],
        collection_scope="Read-only evidence collection.",
        collection_method="Read the existing audited dataset.",
        availability_lag_requirement="One day or less.",
        freshness_requirement="As of the current pack.",
        quality_checks=["Verify provenance."],
        acceptance_criteria=["The field is non-null."],
        stopping_conditions=["Stop if provenance is missing."],
        evidence_refs=[
            EvidenceReference(
                section="cost_and_execution",
                source_member="derived/cost_evidence_timeline_audit.json",
                claim="Cost evidence is incomplete.",
            )
        ],
        research_thread_id="thread-1",
        source_finding_ids=["finding-1"],
    )
    output = Stage2ProposalSet(
        task_id="task-1",
        executive_summary="One bounded proposal.",
        data_collection_proposals=[proposal],
    )
    task = SimpleNamespace(
        task_id="task-1",
        allowed_hypothesis_families=["data_quality"],
    )
    diagnosis = SimpleNamespace(
        primary_bottlenecks=[SimpleNamespace(finding_id="finding-1")],
        contradictions=[],
        missing_evidence=[],
    )

    with pytest.raises(worker._StructuredOutputSemanticError) as exc_info:
        worker._validate_stage2_output(
            output,
            task=task,
            diagnosis=diagnosis,
            allowed_sections={"factor_research"},
            evidence_members={"factor_research": {"reports/factor.csv"}},
        )

    assert exc_info.value.errors[0]["type"] == "evidence_section_not_routed"


def test_run_research_materializes_stage2_evidence_member_contract(monkeypatch) -> None:
    provisional = AIResearchTask(
        task_id="task-stage2-routing",
        created_at=datetime(2026, 8, 9, tzinfo=UTC),
        source_pack_name="expert.zip",
        source_pack_sha256="a" * 64,
        packet_sha256="0" * 64,
        sections={
            "factor_research": [
                EvidenceDocument(
                    source_member="reports/factor.csv",
                    source_format="csv",
                    content_sha256="b" * 64,
                    source_size_bytes=10,
                    content={"rows": []},
                )
            ]
        },
        allowed_hypothesis_families=["data_quality"],
    )
    task = provisional.model_copy(
        update={"packet_sha256": compute_task_packet_sha256(provisional)}
    )
    diagnosis = Stage1Diagnosis(
        task_id=task.task_id,
        system_state="READY_FOR_PROPOSALS",
        executive_summary="Route the bounded factor evidence.",
        stage2_allowed=True,
        route_sections=["factor_research"],
    )
    proposals = Stage2ProposalSet(
        task_id=task.task_id,
        executive_summary="No proposal is needed for this fixture.",
        no_action_reasons=["The fixture only validates evidence routing."],
    )
    calls: list[dict[str, object]] = []

    def _fake_responses_call(*_args: object, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        output = diagnosis if len(calls) == 1 else proposals
        return {
            "output_text": output.model_dump_json(),
            "usage": {},
            "attempts": 1,
            "response_id": f"response-{len(calls)}",
            "validation_events": [],
        }

    monkeypatch.setattr(worker, "_responses_call", _fake_responses_call)
    config = SimpleNamespace(
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        worker_id="nas-test",
        max_output_tokens_stage1=1000,
        max_output_tokens_stage2=1000,
    )

    result = worker.run_research(config, task)

    assert result.proposals == proposals
    stage2_payload = calls[1]["user_payload"]
    assert isinstance(stage2_payload, dict)
    assert stage2_payload["allowed_evidence_sections"] == ["factor_research"]
    assert stage2_payload["allowed_evidence_members"] == {
        "factor_research": ["reports/factor.csv"]
    }


def test_stage1_prompt_distinguishes_publication_lag_from_content_mismatch() -> None:
    prompt = worker.stage1_system_prompt()

    assert "proposal_content_snapshot_match=true" in prompt
    assert "不得称为内容哈希冲突" in prompt


def test_worker_result_carries_materialized_effective_preflight(monkeypatch) -> None:
    preflight = TaskPreflight(
        status="WARN",
        checked_at=datetime(2026, 7, 17, tzinfo=UTC),
        available_sections=["factor_research"],
        truncated_document_count=0,
        warnings=["paper_runtime_stale"],
    )
    provisional = AIResearchTask(
        task_id="task-effective-preflight",
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        source_pack_name="expert.zip",
        source_pack_sha256="a" * 64,
        packet_sha256="0" * 64,
        sections={
            "factor_research": [
                EvidenceDocument(
                    source_member="reports/factor.csv",
                    source_format="csv",
                    content_sha256="b" * 64,
                    source_size_bytes=10,
                    content={"rows": []},
                )
            ]
        },
        preflight=preflight,
        allowed_hypothesis_families=["behavioral_underreaction"],
    )
    task = provisional.model_copy(
        update={"packet_sha256": compute_task_packet_sha256(provisional)}
    )
    diagnosis = Stage1Diagnosis(
        task_id=task.task_id,
        system_state="REVIEW_REQUIRED",
        executive_summary="Keep the current evidence in diagnostic review.",
        stage2_allowed=False,
    )
    monkeypatch.setattr(
        worker,
        "_responses_call",
        lambda *_args, **_kwargs: {
            "output_text": diagnosis.model_dump_json(),
            "usage": {},
            "attempts": 1,
            "response_id": "response-stage1",
            "validation_events": [],
        },
    )
    config = SimpleNamespace(
        model="gpt-5.6-sol",
        reasoning_effort="xhigh",
        worker_id="nas-test",
        max_output_tokens_stage1=1000,
        max_output_tokens_stage2=1000,
    )

    result = worker.run_research(config, task)

    assert result.effective_preflight == preflight
    assert result.warnings == ["stage2_skipped:REVIEW_REQUIRED"]


def test_worker_refuses_legacy_prompt_task_instead_of_mixing_contracts() -> None:
    provisional = AIResearchTask(
        prompt_version="quant_lab.ai_research.prompt.v3",
        task_id="task-legacy-prompt",
        created_at=datetime(2026, 7, 17, tzinfo=UTC),
        source_pack_name="expert.zip",
        source_pack_sha256="a" * 64,
        packet_sha256="0" * 64,
        sections={
            "factor_research": [
                EvidenceDocument(
                    source_member="reports/factor.csv",
                    source_format="csv",
                    content_sha256="b" * 64,
                    source_size_bytes=10,
                    content={"rows": []},
                )
            ]
        },
        allowed_factor_templates=["feature"],
    )
    task = provisional.model_copy(
        update={"packet_sha256": compute_task_packet_sha256(provisional)}
    )

    with pytest.raises(ValueError, match="prompt version"):
        worker.run_research(SimpleNamespace(), task)


def test_upload_result_stages_before_atomic_inbox_publish(monkeypatch, tmp_path) -> None:
    ssh_calls: list[list[str]] = []
    scp_calls: list[tuple[Path, str]] = []
    config = SimpleNamespace(remote_queue_root="/var/lib/quant-lab/ai_queue")
    local_result = tmp_path / "result.json"
    local_result.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        worker,
        "_ssh",
        lambda _config, args, **_kwargs: ssh_calls.append(args),
    )
    monkeypatch.setattr(
        worker,
        "_scp_to",
        lambda _config, local, remote: scp_calls.append((local, remote)),
    )

    worker._upload_result(config, "task-atomic", local_result)

    assert scp_calls == [
        (
            local_result,
            "/var/lib/quant-lab/ai_queue/results/inbox/.staging/task-atomic/result.json.tmp",
        )
    ]
    assert "results/inbox/.staging/task-atomic" in ssh_calls[0][2]
    publish_script = ssh_calls[1][2]
    assert 'mv "$running" "$completed"' in publish_script
    assert 'mv "$stage" "$publish"' in publish_script
    assert publish_script.index('mv "$stage" "$publish"') < publish_script.index(
        'mv "$running" "$completed"'
    )


def test_worker_start_recovery_requeues_orphan_and_finishes_visible_handoff(
    monkeypatch,
) -> None:
    ssh_calls: list[list[str]] = []
    config = SimpleNamespace(remote_queue_root="/var/lib/quant-lab/ai_queue")

    def _fake_ssh(
        _config: object,
        args: list[str],
        **_kwargs: object,
    ) -> SimpleNamespace:
        ssh_calls.append(args)
        return SimpleNamespace(
            stdout="requeued:task-orphan\nhandoff:task-visible\n",
            stderr="",
            returncode=0,
        )

    monkeypatch.setattr(worker, "_ssh", _fake_ssh)

    assert worker.recover_interrupted_tasks(config) == (1, 1)

    script = ssh_calls[0][2]
    assert "results/inbox/.staging" in script
    assert 'mv "$stage" "$inbox"' in script
    assert '[ -f "$inbox/result.json" ] || [ -f "$imported/result.json" ]' in script
    assert 'mv "$running" "$completed"' in script
    assert 'mv "$running" "$pending"' in script
