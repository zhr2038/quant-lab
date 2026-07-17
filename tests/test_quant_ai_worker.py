from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from deploy.nas_ai_worker import worker


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
