from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm.schemas import AllowedAgentName, PromptSpec

DEFAULT_PROMPT_VERSION = "v1.0.0"


class PromptLoadError(RuntimeError):
    pass


class _SafeFormatDict(defaultdict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def get_prompt_path(agent: AllowedAgentName, version: str = DEFAULT_PROMPT_VERSION) -> Path:
    root = Path(__file__).resolve().parent.parent
    return root / "prompts" / agent / f"{version}.yaml"


@lru_cache(maxsize=64)
def load_prompt(agent: AllowedAgentName, version: str = DEFAULT_PROMPT_VERSION) -> PromptSpec:
    path = get_prompt_path(agent, version)
    if not path.exists():
        raise PromptLoadError(f"Prompt file not found: {path}")

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PromptLoadError(f"Invalid YAML in prompt file {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise PromptLoadError(f"Prompt file must contain a YAML mapping: {path}")

    try:
        return PromptSpec.model_validate(payload)
    except ValidationError as exc:
        raise PromptLoadError(f"Prompt validation failed for {path}: {exc}") from exc


def render_prompt(agent: AllowedAgentName, version: str = DEFAULT_PROMPT_VERSION, **variables: Any) -> str:
    prompt = load_prompt(agent, version)
    if not variables:
        return prompt.system
    return prompt.system.format_map(_SafeFormatDict(str, variables))
