from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

ErrorType = Literal[
    "validation_error",
    "rate_limit",
    "timeout",
    "auth_error",
    "provider_error",
    "context_overflow",
    "budget_exceeded",
    "no_results",
    "all_sources_failed",
    "circuit_open",
    "unknown",
]


class ToolSuccessResponse(BaseModel):
    success: Literal[True] = True
    tool_name: str = Field(..., min_length=1, description="Name of the tool that produced this response.")
    data: Any = Field(..., description="Structured tool-specific payload.")
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata such as source, mode, or timing.",
    )


class ToolErrorResponse(BaseModel):
    success: Literal[False] = False
    tool_name: str = Field(..., min_length=1, description="Name of the tool that failed.")
    error_type: ErrorType = Field(..., description="Structured error category.")
    message: str = Field(..., min_length=1, description="LLM-readable explanation of what went wrong.")
    recoverable: bool = Field(..., description="Whether the agent may retry or attempt an alternate tool.")
    retry_after_seconds: float | None = Field(
        default=None,
        ge=0,
        description="Optional wait time before retrying, mainly for rate-limit or transient provider errors.",
    )
    fallback_suggestion: str | None = Field(
        default=None,
        description="Concrete suggestion for the agent if this tool cannot complete the request.",
    )
    details: dict[str, Any] = Field(default_factory=dict, description="Machine-readable diagnostic details.")


class SupportTicketInput(BaseModel):
    title: str = Field(..., min_length=5, max_length=100, description="Short support ticket title. Example: 'VPN connection failure'.")
    description: str = Field(
        ...,
        min_length=20,
        max_length=2000,
        description="Detailed issue or request description. Include what happened, impact, and requested outcome.",
    )
    category: Literal["IT", "HR", "Facilities", "Finance", "Legal"] = Field(
        ...,
        description="Department responsible for handling the request.",
    )
    priority: Literal["low", "medium", "high", "urgent"] = Field(
        "medium",
        description="Urgency level. Use urgent only for outages, blockers, or time-sensitive escalations.",
    )
    user_id: str = Field("unknown", min_length=1, max_length=100, description="Requester user ID. Example: 'u12345'.")


class ResearchQueryInput(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="The user's external research question. Example: 'current AI governance trends'.",
    )
    intent: Literal["general", "academic", "policy"] = Field(
        "general",
        description="Research intent. Use general for broad web context, academic for scholarly topics, policy for regulations or governance.",
    )
    max_results: int = Field(3, ge=1, le=5, description="Maximum number of results to return. Bounded to control cost.")


class KnowledgeSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Internal company knowledge question. Example: 'How do I apply for parental leave?'.",
    )
    user_tier: Literal["standard", "manager", "exec"] = Field(
        "standard",
        description="Access tier used for RBAC filtering.",
    )
    top_k: int = Field(5, ge=1, le=10, description="Number of retrieved chunks to consider.")
    data_dir: str = Field(
        "data",
        min_length=1,
        max_length=300,
        description="Local data directory containing company documents.",
    )

    registry_db_path: str | None = Field(
        default=None,
        max_length=300,
        description="Optional SQLite ingestion registry path. Example: 'data/cortex_registry.sqlite'.",
    )
    use_pgvector: bool = Field(
        default=False,
        description="Whether to use PGVector-backed semantic search when available.",
    )
    database_url: str | None = Field(
        default=None,
        max_length=500,
        description="Optional Postgres/PGVector database URL. If omitted, DATABASE_URL may be used.",
    )
    expanded_queries_override: list[str] | None = Field(
        default=None,
        max_length=5,
        description="Optional query variants from the query rewriting layer.",
    )

    @field_validator("expanded_queries_override")
    @classmethod
    def _validate_expanded_queries_override(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        return [str(item)[:500] for item in value][:5]


class UserContextInput(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100, description="User ID whose memory context should be loaded.")
    session_id: str | None = Field(default=None, max_length=100, description="Optional session ID for recent conversation memory.")
    message_limit: int = Field(10, ge=1, le=20, description="Maximum number of recent session messages to load.")


class ForgetUserContextInput(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100, description="User ID whose stored entity facts should be forgotten.")
    entity_type: Literal["department", "role", "project", "team", "preference", "location", "work_style", "note"] | None = Field(
        default=None,
        description="Optional entity type to forget. If omitted, all long-term entity facts for the user are deleted.",
    )
    include_session_messages: bool = Field(
        default=False,
        description="Whether to also clear session messages. Defaults to False for safety.",
    )
    session_id: str | None = Field(
        default=None,
        max_length=100,
        description="Session ID to clear if include_session_messages is true.",
    )


class TicketStatusInput(BaseModel):
    ticket_id: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="Support ticket ID. Example: 'TKT-20260524-9EEA16'.",
    )


class EscalateIssueInput(BaseModel):
    issue: str = Field(
        ...,
        min_length=10,
        max_length=1000,
        description="Issue to escalate. Include the problem and business impact.",
    )
    ticket_id: str | None = Field(
        default=None,
        max_length=40,
        description="Existing support ticket ID if available. Example: 'TKT-20260524-9EEA16'.",
    )
    priority: Literal["high", "urgent"] = Field(
        "high",
        description="Escalation priority. Use urgent only for production blockers, outages, or severe business impact.",
    )
    user_id: str = Field(
        "unknown",
        min_length=1,
        max_length=100,
        description="Requester user ID.",
    )


class NotificationInput(BaseModel):
    recipient: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Recipient identifier, email address, Slack channel, or user group.",
    )
    message: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Notification message to send.",
    )
    channel: Literal["slack", "email", "system"] = Field(
        "system",
        description="Notification channel. Offline mock only; no real message is sent.",
    )
    user_id: str = Field(
        "unknown",
        min_length=1,
        max_length=100,
        description="Requester user ID.",
    )
