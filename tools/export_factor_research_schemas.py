from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from quant_lab.research.factor_research.contracts import (  # noqa: E402
    ResearchHypothesis,
    ResearchTrial,
)
from quant_lab.research_plane.contracts import (  # noqa: E402
    FactorResearchResultManifest,
    FactorResearchTask,
)

SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_MODELS = {
    "research_hypothesis.schema.json": ResearchHypothesis,
    "research_trial.schema.json": ResearchTrial,
    "factor_research_task.schema.json": FactorResearchTask,
    "factor_research_result.schema.json": FactorResearchResultManifest,
}


def schema_documents() -> dict[str, dict[str, Any]]:
    documents: dict[str, dict[str, Any]] = {}
    for filename, model in SCHEMA_MODELS.items():
        documents[filename] = {
            "$schema": SCHEMA_DRAFT,
            "$id": f"https://schemas.hrhome.top/quant-lab/{filename}",
            **model.model_json_schema(mode="validation"),
        }
    return documents


def write_schemas(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for filename, document in schema_documents().items():
        (output_root / filename).write_text(
            json.dumps(document, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical Factor Research v2 schemas")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "schemas")
    args = parser.parse_args()
    write_schemas(args.output_root.resolve())


if __name__ == "__main__":
    main()
