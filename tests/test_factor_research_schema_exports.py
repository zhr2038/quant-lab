from __future__ import annotations

import json
from pathlib import Path

from quant_lab.research.factor_research.contracts import ResearchHypothesis, ResearchTrial
from quant_lab.research_plane.contracts import FactorResearchResultManifest, FactorResearchTask

ROOT = Path(__file__).resolve().parents[1]


def test_committed_factor_research_schemas_match_runtime_contracts() -> None:
    models = {
        "research_hypothesis.schema.json": ResearchHypothesis,
        "research_trial.schema.json": ResearchTrial,
        "factor_research_task.schema.json": FactorResearchTask,
        "factor_research_result.schema.json": FactorResearchResultManifest,
    }
    for filename, model in models.items():
        committed = json.loads((ROOT / "schemas" / filename).read_text(encoding="utf-8"))
        runtime = model.model_json_schema(mode="validation")
        assert committed["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert committed["$id"].endswith(filename)
        assert committed["type"] == "object"
        assert committed["additionalProperties"] is False
        assert committed["properties"] == runtime["properties"]
        assert committed["required"] == runtime["required"]
        assert committed.get("$defs") == runtime.get("$defs")
