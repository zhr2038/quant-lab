from __future__ import annotations

from importlib.resources import files


PROMPT_PACKAGE = "quant_lab.ai_research.prompts"


def load_prompt(name: str) -> str:
    resource = files(PROMPT_PACKAGE).joinpath(name)
    return resource.read_text(encoding="utf-8").strip()


def stage1_system_prompt() -> str:
    return load_prompt("stage1_system.md")


def stage2_system_prompt() -> str:
    return load_prompt("stage2_system.md")
