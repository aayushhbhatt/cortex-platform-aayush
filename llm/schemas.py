from typing import Any, Literal

from pydantic import BaseModel, Field

AllowedAgentName = Literal["supervisor", "knowledge", "action", "research"]


class PromptSpec(BaseModel):
    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    agent: AllowedAgentName
    description: str = ""
    system: str = Field(..., min_length=1)
    output_format: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRequest(BaseModel):
    prompt_name: str
    prompt_version: str
    agent: AllowedAgentName
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(512, ge=1, le=8192)
    response_format: Literal["text", "json"] = "json"


class LLMResponse(BaseModel):
    success: bool
    content: str = ""
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    error_type: str | None = None
    message: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict)
