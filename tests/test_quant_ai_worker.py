from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from deploy.nas_ai_worker import worker
from quant_lab.ai_research.contracts import (
    AIResearchTask,
    EvidenceDocument,
    Stage1Diagnosis,
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
        allowed_factor_templates=["feature"],
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
            "/var/lib/quant-lab/ai_queue/results/staging/task-atomic/result.json.tmp",
        )
    ]
    assert "results/staging/task-atomic" in ssh_calls[0][2]
    publish_script = ssh_calls[1][2]
    assert 'mv "$running" "$completed"' in publish_script
    assert 'mv "$stage" "$publish"' in publish_script
    assert publish_script.index('mv "$running" "$completed"') < publish_script.index(
        'mv "$stage" "$publish"'
    )
