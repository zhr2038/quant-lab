from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from audit.auditlib.checkpoints import is_stage_complete, write_checkpoint  # noqa: E402


def test_checkpoint_resume_requires_matching_output_hash(tmp_path: Path) -> None:
    output = tmp_path / "result.csv"
    output.write_text("value\n1\n", encoding="utf-8")
    write_checkpoint(
        root=tmp_path,
        stage="data",
        started_at="2026-07-18T00:00:00Z",
        git_commit="abc123",
        snapshot_id="snap-test",
        command="audit --stage data",
        outputs=[output],
    )
    assert is_stage_complete(tmp_path, "data")

    output.write_text("value\n2\n", encoding="utf-8")
    assert not is_stage_complete(tmp_path, "data")


def test_checkpoint_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    output.write_text('{"ok":true}\n', encoding="utf-8")
    kwargs = dict(
        root=tmp_path,
        stage="leakage",
        started_at="2026-07-18T00:00:00Z",
        finished_at="2026-07-18T00:01:00Z",
        git_commit="abc123",
        snapshot_id="snap-test",
        command="audit --stage leakage",
        outputs=[output],
    )
    first = write_checkpoint(**kwargs)
    first_bytes = first.read_bytes()
    second = write_checkpoint(**kwargs)
    assert second.read_bytes() == first_bytes
