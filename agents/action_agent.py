from dataclasses import asdict
import json
import os
import re
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from agents.supervisor import CortexState
from llm.client import OpenAIChatClient, build_json_request
from llm.prompt_loader import load_prompt
from memory.context import MemoryContext, finalize_agent_memory, load_memory_context
from reliability.cost_tracker import CostTracker, estimate_tool_cost
from reliability.errors import BudgetExceededError, classify_error
from tools.action_tools import (
    create_support_ticket_impl,
    escalate_issue_impl,
    get_ticket_status_impl,
    send_notification_impl,
)
from tools.tool_contracts import error_response

load_dotenv()

_TICKET_ID_RE = re.compile(r"\bTKT-\d{8}-[A-Z0-9]{6}\b")
_ACTION_TYPES = (
    "create_ticket",
    "ticket_lookup",
    "ticket_status",
    "escalate_issue",
    "send_notification",
    "unsupported",
)
_SAFE_TICKET_ENTITY_TYPES = {"department", "team", "role", "project", "location", "work_style", "preference"}
_MAX_TICKET_DESCRIPTION_LEN = 2000
_MAX_SAFE_CONTEXT_VALUE_LEN = 120

ActionType = Literal[
    "create_ticket",
    "ticket_lookup",
    "ticket_status",
    "escalate_issue",
    "send_notification",
    "software_request",
    "unsupported",
]


class ActionClassification(BaseModel):
    action_type: ActionType = Field(..., description="The action workflow the user is requesting.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)


def _env_float(name: str, default: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(min_value, min(max_value, value))


def get_action_classifier_mode() -> str:
    """Return action classifier mode: deterministic or llm."""
    mode = os.getenv("ACTION_CLASSIFIER_MODE", "deterministic").strip().lower()
    return mode if mode in {"deterministic", "llm"} else "deterministic"


def get_action_classifier_confidence_threshold() -> float:
    """Return action classifier confidence threshold clamped to 0.0-1.0."""
    return _env_float("ACTION_CLASSIFIER_CONFIDENCE_THRESHOLD", default=0.75)


def parse_action_classification(raw_output: str) -> ActionClassification:
    """Parse and validate LLM action classification output."""
    try:
        return ActionClassification.model_validate(json.loads(raw_output))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from action classifier: {exc}") from exc
    except ValidationError as exc:
        raise ValueError(f"Invalid action classification schema: {exc}") from exc


def _classification_payload(
    *,
    ok: bool,
    action_type: str | None = None,
    confidence: float | None = None,
    reason: str | None = None,
    raw_output: str | None = None,
    error_type: str | None = None,
    message: str | None = None,
) -> dict:
    return {
        "ok": ok,
        "action_type": action_type,
        "confidence": confidence,
        "reason": reason,
        "raw_output": raw_output,
        "error_type": error_type,
        "message": message,
    }


def llm_classify_action(query: str, memory_context: MemoryContext) -> dict:
    """Attempt LLM-based action-type classification."""
    user_prompt = (
        f"User query:\n{query}\n\n"
        f"Memory context:\n{memory_context.context_text}\n\n"
        "Allowed action types:\n"
        + "\n".join(f"- {action}" for action in _ACTION_TYPES)
        + "\n\nReturn JSON only with keys: action_type, confidence, reason."
    )

    try:
        prompt = load_prompt("action")
        request = build_json_request(agent="action", prompt=prompt, user_prompt=user_prompt, max_tokens=256)
        response = OpenAIChatClient().complete(request)
    except Exception as exc:
        return _classification_payload(ok=False, error_type="exception", message=str(exc))

    if not response.success:
        return _classification_payload(
            ok=False,
            raw_output=response.content,
            error_type=response.error_type,
            message=response.message,
        )

    try:
        parsed = parse_action_classification(response.content)
    except ValueError as exc:
        return _classification_payload(
            ok=False,
            raw_output=response.content,
            error_type="invalid_output",
            message=str(exc),
        )

    return _classification_payload(
        ok=True,
        action_type=parsed.action_type,
        confidence=parsed.confidence,
        reason=parsed.reason,
        raw_output=response.content,
    )


def _classifier_debug(
    *,
    mode: str,
    method: str,
    deterministic_action_type: str,
    llm_action_type: str | None,
    confidence: float | None,
    threshold: float,
    confidence_gate_passed: bool | None,
    fallback_used: bool,
    fallback_reason: str | None,
    reason: str | None,
    raw_llm_output: str | None,
) -> dict:
    return {
        "mode": mode,
        "method": method,
        "deterministic_action_type": deterministic_action_type,
        "llm_action_type": llm_action_type,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "confidence_gate_passed": confidence_gate_passed,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "reason": reason,
        "raw_llm_output": raw_llm_output,
    }


def classify_action_type(query: str, memory_context: MemoryContext) -> tuple[str, dict]:
    """Classify action type using deterministic mode or optional LLM mode."""
    threshold = get_action_classifier_confidence_threshold()
    deterministic_action_type = _classify_action_request(query)
    mode = get_action_classifier_mode()

    if mode == "deterministic":
        return deterministic_action_type, _classifier_debug(
            mode="deterministic",
            method="deterministic",
            deterministic_action_type=deterministic_action_type,
            llm_action_type=None,
            confidence=None,
            threshold=threshold,
            confidence_gate_passed=None,
            fallback_used=False,
            fallback_reason=None,
            reason=None,
            raw_llm_output=None,
        )

    llm_result = llm_classify_action(query, memory_context)
    if not llm_result["ok"]:
        return deterministic_action_type, _classifier_debug(
            mode="llm",
            method="llm_fallback",
            deterministic_action_type=deterministic_action_type,
            llm_action_type=None,
            confidence=None,
            threshold=threshold,
            confidence_gate_passed=False,
            fallback_used=True,
            fallback_reason=llm_result.get("error_type") or llm_result.get("message"),
            reason=None,
            raw_llm_output=llm_result.get("raw_output"),
        )

    confidence = llm_result.get("confidence")
    llm_action_type = llm_result.get("action_type")
    if confidence is None or confidence < threshold:
        return deterministic_action_type, _classifier_debug(
            mode="llm",
            method="llm_fallback",
            deterministic_action_type=deterministic_action_type,
            llm_action_type=llm_action_type,
            confidence=confidence,
            threshold=threshold,
            confidence_gate_passed=False,
            fallback_used=True,
            fallback_reason="low_confidence",
            reason=llm_result.get("reason"),
            raw_llm_output=llm_result.get("raw_output"),
        )

    return llm_action_type, _classifier_debug(
        mode="llm",
        method="llm",
        deterministic_action_type=deterministic_action_type,
        llm_action_type=llm_action_type,
        confidence=confidence,
        threshold=threshold,
        confidence_gate_passed=True,
        fallback_used=False,
        fallback_reason=None,
        reason=llm_result.get("reason"),
        raw_llm_output=llm_result.get("raw_output"),
    )


def _classify_action_request(query: str) -> str:
    q = " ".join(query.lower().split())

    if "request access to" in q or "request access" in q:
        return "software_request"

    if any(
        phrase in q
        for phrase in (
            "what is the ticket number",
            "what is the support ticket number",
            "what ticket did you create",
            "what was the ticket id",
            "show me the ticket id",
            "previous ticket",
            "last ticket",
            "what is my request number",
        )
    ):
        return "ticket_lookup"

    if any(
        phrase in q
        for phrase in (
            "status of ticket",
            "ticket status",
            "check ticket",
            "what is the status",
            "is my ticket resolved",
        )
    ):
        return "ticket_status"
    if _TICKET_ID_RE.search(query) and any(term in q for term in ("status", "check", "resolved", "open")):
        return "ticket_status"

    if any(
        phrase in q
        for phrase in (
            "escalate this",
            "escalate issue",
            "urgent escalation",
            "this is blocking production",
            "raise priority",
            "make this urgent",
        )
    ):
        return "escalate_issue"

    if any(
        phrase in q
        for phrase in (
            "notify",
            "send notification",
            "send email",
            "send slack",
            "message the team",
            "alert",
        )
    ):
        return "send_notification"

    if any(
        phrase in q
        for phrase in (
            "create a ticket",
            "create ticket",
            "create support ticket",
            "create a support ticket",
            "open a ticket",
            "log a ticket",
            "raise a ticket",
            "submit a ticket",
            "file a ticket",
            "report an issue",
            "create a request",
        )
    ):
        return "create_ticket"

    return "unsupported"


def _extract_ticket_id_from_query(query: str) -> str | None:
    match = _TICKET_ID_RE.search(query)
    return match.group(0) if match else None


def _find_latest_ticket_id(memory_context: MemoryContext | dict | None) -> str | None:
    if not memory_context:
        return None

    messages = memory_context.recent_messages if isinstance(memory_context, MemoryContext) else memory_context.get("recent_messages", [])
    assistant_messages = [message for message in messages if isinstance(message, dict) and message.get("role") == "assistant"]
    other_messages = [message for message in messages if isinstance(message, dict) and message.get("role") != "assistant"]

    for message_group in (assistant_messages, other_messages):
        for message in reversed(message_group):
            match = _TICKET_ID_RE.search(str(message.get("content", "")))
            if match:
                return match.group(0)
    return None


def _extract_safe_ticket_context(memory_context: MemoryContext | dict | None) -> dict[str, str]:
    if not memory_context:
        return {}

    entities = memory_context.entities if isinstance(memory_context, MemoryContext) else memory_context.get("entities", [])
    if not isinstance(entities, list):
        return {}

    safe_context = {}
    for entity in entities:
        if not isinstance(entity, dict):
            continue
        entity_type = str(entity.get("entity_type", "")).strip().lower()
        entity_value = str(entity.get("entity_value", "")).strip()
        if entity_type in _SAFE_TICKET_ENTITY_TYPES and entity_value:
            safe_context[entity_type] = entity_value[:_MAX_SAFE_CONTEXT_VALUE_LEN]

    return {key: safe_context[key] for key in sorted(safe_context)}


def _build_enriched_description(base_description: str, safe_context: dict[str, str]) -> str:
    if not safe_context:
        return base_description

    context_block = "\n\nUser context:\n" + "\n".join(f"- {key}: {value}" for key, value in safe_context.items())
    available = _MAX_TICKET_DESCRIPTION_LEN - len(base_description)
    if available <= 0:
        return base_description[:_MAX_TICKET_DESCRIPTION_LEN]
    if len(context_block) <= available:
        return base_description + context_block
    return base_description + context_block[:available].rstrip()


def _load_memory_context(state: CortexState, user_id: str, session_id: str) -> MemoryContext:
    if state.get("memory_context"):
        return MemoryContext(**state["memory_context"])
    return load_memory_context(user_id=user_id, session_id=session_id)


def _tool_error(tool_name: str, exc: Exception) -> dict:
    response = error_response(tool_name, classify_error(exc))
    return response if isinstance(response, dict) else response.model_dump()


def _finalize(
    state: CortexState,
    memory_context: MemoryContext,
    response: str,
    debug: dict,
    tool_results: list,
    cost_usd: float,
) -> dict:
    finalizer = finalize_agent_memory(
        user_id=state.get("user_id", "unknown"),
        session_id=state.get("session_id") or "session_local",
        query=state["query"],
        response=response,
    )
    debug["memory_finalizer"] = finalizer["memory_debug"]
    return {
        "response": response,
        "agent_used": "action",
        "tool_results": tool_results,
        "cost_usd": cost_usd,
        "debug": debug,
        "memory_messages": finalizer["memory_messages"],
        "memory_context": asdict(memory_context),
        "memory_context_text": memory_context.context_text,
    }


def _handle_budget_exceeded(state: CortexState, memory_context: MemoryContext, debug: dict, tracker: CostTracker, exc: BudgetExceededError) -> dict:
    error = classify_error(exc)
    debug["budget_precheck"] = {"checked": True, "passed": False, "tool_call_skipped": True}
    debug["error"] = {
        "error_type": error.error_type,
        "message": error.message,
        "retryable": error.retryable,
        "details": error.details,
    }
    return _finalize(state, memory_context, f"Request exceeded budget: {error.message}", debug, [], tracker.total_cost())


def action_agent(state: CortexState) -> dict:
    query = state["query"].strip()
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id") or "session_local"
    memory_context = _load_memory_context(state, user_id, session_id)

    action_type, classifier_debug = classify_action_type(query, memory_context)
    debug = {
        "action_type": action_type,
        "side_effect_executed": False,
        "cost": CostTracker().to_dict(),
        "classifier": classifier_debug,
    }

    if action_type == "ticket_lookup":
        ticket_id = _find_latest_ticket_id(memory_context)
        debug["ticket_lookup"] = {"found": bool(ticket_id), "ticket_id": ticket_id}
        response = (
            f"The latest support ticket number I found is {ticket_id}."
            if ticket_id
            else "I could not find a previous support ticket number in this session."
        )
        return _finalize(state, memory_context, response, debug, [], 0.0)

    if action_type == "unsupported":
        response = "I can help create support tickets, check ticket status, escalate issues, send notifications, or create software requests. Please provide the action details."
        return _finalize(state, memory_context, response, debug, [], 0.0)

    tracker = CostTracker()
    tracker.add_cost("tools", "action_workflow", estimate_tool_cost(1), {"preflight": True})
    try:
        tracker.check_budget()
    except BudgetExceededError as exc:
        return _handle_budget_exceeded(state, memory_context, debug, tracker, exc)

    debug["budget_precheck"] = {"checked": True, "passed": True, "tool_call_skipped": False}

    if action_type == "ticket_status":
        query_ticket_id = _extract_ticket_id_from_query(query)
        ticket_id = query_ticket_id or _find_latest_ticket_id(memory_context)
        debug["ticket_status"] = {"ticket_id": ticket_id, "source": "query" if query_ticket_id else "memory"}
        if not ticket_id:
            debug["side_effect_executed"] = False
            response = "I could not find a ticket ID to check. Please provide the ticket number or ask about the most recent ticket."
            return _finalize(state, memory_context, response, debug, [], tracker.total_cost())

        output = get_ticket_status_impl(ticket_id=ticket_id)
        status = output.get("data", {}).get("status", "unknown") if output.get("success") else "unknown"
        response = f"Ticket {ticket_id} is currently {status}."
        debug["mapped_tool"] = "get_ticket_status"
        return _finalize(state, memory_context, response, debug, [output], tracker.total_cost())

    if action_type in {"create_ticket", "software_request"}:
        debug["mapped_tool"] = "create_support_ticket"
        debug["category_strategy"] = "default_IT_no_inference"
        debug["side_effect_executed"] = True

        safe_context = _extract_safe_ticket_context(memory_context)
        description = query if len(query) >= 20 else f"{query} Please include additional details so this request can be fulfilled accurately."
        try:
            output = create_support_ticket_impl(
                title=query[:80] if query else "Cortex Action Request",
                description=_build_enriched_description(description, safe_context),
                category="IT",
                priority="medium",
                user_id=user_id,
            )
        except Exception as exc:
            output = _tool_error("create_support_ticket", exc)

        debug["memory_enrichment"] = {"applied": bool(safe_context), "fields": list(safe_context.keys())}
        if not isinstance(output, dict):
            output = _tool_error("create_support_ticket", RuntimeError("invalid tool response"))

        ticket_id = output.get("data", {}).get("ticket_id", "unknown")
        response = f"Created software request ticket {ticket_id}." if action_type == "software_request" else f"Created support ticket {ticket_id}."
        return _finalize(state, memory_context, response, debug, [output], tracker.total_cost())

    if action_type == "escalate_issue":
        debug["mapped_tool"] = "escalate_issue"
        debug["side_effect_executed"] = True
        try:
            output = escalate_issue_impl(
                issue=query,
                ticket_id=_extract_ticket_id_from_query(query),
                priority="urgent" if "urgent" in query.lower() or "blocking production" in query.lower() else "high",
                user_id=user_id,
            )
        except Exception as exc:
            output = _tool_error("escalate_issue", exc)

        escalation_id = output.get("data", {}).get("escalation_id", "unknown")
        priority = output.get("data", {}).get("priority", "high")
        response = f"Escalated the issue as {escalation_id} with {priority} priority."
        return _finalize(state, memory_context, response, debug, [output], tracker.total_cost())

    if action_type == "send_notification":
        debug["mapped_tool"] = "send_notification"
        debug["side_effect_executed"] = True
        try:
            output = send_notification_impl(recipient="IT team", message=query, channel="system", user_id=user_id)
        except Exception as exc:
            output = _tool_error("send_notification", exc)

        data = output.get("data", {})
        response = (
            f"Notification {data.get('notification_id', 'unknown')} was recorded for "
            f"recipient {data.get('recipient', 'unknown')} via channel {data.get('channel', 'system')}."
        )
        return _finalize(state, memory_context, response, debug, [output], tracker.total_cost())

    response = "I can help create support tickets, check ticket status, escalate issues, send notifications, or create software requests. Please provide the action details."
    return _finalize(state, memory_context, response, debug, [], 0.0)
