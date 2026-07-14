"""Read-only AI research plane for quant-lab.

The package intentionally does not import packet/importer modules at import time so
NAS workers can reuse contracts and prompts without installing the full quant-lab
data stack.
"""

from quant_lab.ai_research.contracts import (
    AI_PROMPT_VERSION,
    AI_RESULT_SCHEMA_VERSION,
    AI_STAGE1_SCHEMA_VERSION,
    AI_STAGE2_SCHEMA_VERSION,
    AI_TASK_SCHEMA_VERSION,
    AIResearchResult,
    AIResearchTask,
    Stage1Diagnosis,
    Stage2ProposalSet,
)

__all__ = [
    "AI_PROMPT_VERSION",
    "AI_RESULT_SCHEMA_VERSION",
    "AI_STAGE1_SCHEMA_VERSION",
    "AI_STAGE2_SCHEMA_VERSION",
    "AI_TASK_SCHEMA_VERSION",
    "AIResearchResult",
    "AIResearchTask",
    "Stage1Diagnosis",
    "Stage2ProposalSet",
]
