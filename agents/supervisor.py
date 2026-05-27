import json
import os
import uuid
import time
from typing import Any, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
from llm.client import OpenAIChatClient, build_json_request
from llm.prompt_loader import load_prompt
from observability.logging import get_log_path, log_query_event
from observability.tracing import build_query_trace_event, new_request_id

load_dotenv()
class CortexState(TypedDict):
    query: str
    user_id: str
    user_tier: str
    intent: Optional[str]
    session_id: Optional[str]
    request_id: Optional[str]

    agent_used: Optional[str]
    tool_results: List[dict]
    used_chunks: List[str]
    context: List[str]
    response: Optional[str]
    cost_usd: float

    memory_messages: List[dict]
    memory_context: dict
    memory_context_text: str

    debug: dict[str, Any]
    citations: List[dict]
    trace: dict


SupervisorIntent = Literal["knowledge", "action", "research", "unsupported"]

AVAILABLE_ROUTES = ["knowledge", "action", "research", "unsupported"]

KNOWLEDGE_KEYWORDS = (
    "policy",
    "leave",
    "benefit",
    "benefits",
    "handbook",
    "document",
    "documents",
    "docs",
    "knowledge",
    "parental",
    "compensation",
    "remote work",
    "data security",
    "acceptable use",
    "company overview",
    "manager handbook",
)
ACTION_KEYWORDS = (
    "ticket",
    "request",
    "escalate",
    "create",
    "workflow",
    "notify",
    "notification",
    "status",
    "support",
    "software access",
)
RESEARCH_KEYWORDS = (
    "research",
    "search",
    "current",
    "latest",
    "recent",
    "trends",
    "news",
    "web",
    "external",
    "internet",
    "market",
    "competitor",
    "benchmark",
    "study",
    "paper",
    "regulation",
    "governance",
    "governance update",
    "industry",
    "compare",
    "comparison",
)


class SupervisorRouterDecision(BaseModel):
    intent: SupervisorIntent = Field(..., description="Selected Cortex route.")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)


def deterministic_route_intent(query: str) -> SupervisorIntent:
    lowered_query = (query or "").lower()
    if any(keyword in lowered_query for keyword in KNOWLEDGE_KEYWORDS):
        return "knowledge"
    if any(keyword in lowered_query for keyword in ACTION_KEYWORDS):
        return "action"
    if any(keyword in lowered_query for keyword in RESEARCH_KEYWORDS):
        return "research"
    return "unsupported"


def get_supervisor_router_mode() -> str:
    mode = os.getenv("SUPERVISOR_ROUTER_MODE", "deterministic").strip().lower()
    return mode if mode in {"deterministic", "llm"} else "deterministic"


def get_supervisor_router_confidence_threshold() -> float:
    try:
        threshold = float(os.getenv("SUPERVISOR_ROUTER_CONFIDENCE_THRESHOLD", "0.70"))
    except (TypeError, ValueError):
        return 0.70
    return max(0.0, min(1.0, threshold))


def parse_supervisor_router_decision(raw_output: str) -> SupervisorRouterDecision:
    try:
        payload = json.loads(raw_output)
        return SupervisorRouterDecision.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Invalid supervisor router decision: {exc}") from exc


def _build_router_user_prompt(query: str, user_tier: str, memory_context_text: str) -> str:
    return (
        f"User tier: {user_tier}\n"
        f"Memory context: {memory_context_text or '(none)'}\n"
        f"Query: {query}\n\n"
        "Return only JSON matching:\n"
        '{"intent": "knowledge|action|research|unsupported", "confidence": 0.0, "reason": "short reason"}'
    )


def _failed_llm_route(error_type: str, message: str | None = None, raw_output: str | None = None) -> dict:
    return {
        "ok": False,
        "intent": None,
        "confidence": None,
        "reason": None,
        "raw_output": raw_output,
        "error_type": error_type,
        "message": message,
    }


def llm_route_intent(query: str, user_tier: str = "standard", memory_context_text: str = "") -> dict:
    try:
        prompt = load_prompt("supervisor")
        request = build_json_request(
            agent="supervisor",
            prompt=prompt,
            user_prompt=_build_router_user_prompt(query, user_tier, memory_context_text),
            max_tokens=256,
        )
        response = OpenAIChatClient().complete(request)
        if not response.success:
            return _failed_llm_route(response.error_type or "llm_error", response.message, response.content or None)

        decision = parse_supervisor_router_decision(response.content)
        return {
            "ok": True,
            "intent": decision.intent,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "raw_output": response.content,
            "error_type": None,
            "message": None,
        }
    except Exception as exc:
        return _failed_llm_route("router_exception", str(exc))


def _router_debug(
    *,
    method: str,
    mode: str,
    threshold: float,
    confidence: float | None = None,
    confidence_gate_passed: bool | None = None,
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    raw_llm_output: str | None = None,
    reason: str | None = None,
) -> dict:
    return {
        "method": method,
        "mode": mode,
        "confidence": confidence,
        "confidence_threshold": threshold,
        "confidence_gate_passed": confidence_gate_passed,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "raw_llm_output": raw_llm_output,
        "reason": reason,
        "available_routes": AVAILABLE_ROUTES,
    }


def route_intent(query: str, user_tier: str = "standard", memory_context_text: str = "") -> dict:
    mode = get_supervisor_router_mode()
    threshold = get_supervisor_router_confidence_threshold()

    if mode != "llm":
        return {
            "intent": deterministic_route_intent(query),
            "router_debug": _router_debug(method="deterministic", mode="deterministic", threshold=threshold),
        }

    llm_result = llm_route_intent(query, user_tier=user_tier, memory_context_text=memory_context_text)
    confidence = llm_result.get("confidence")
    raw_output = llm_result.get("raw_output")
    reason = llm_result.get("reason")

    if not llm_result["ok"]:
        fallback_reason = llm_result.get("error_type") or llm_result.get("message")
        return {
            "intent": deterministic_route_intent(query),
            "router_debug": _router_debug(
                method="llm_fallback",
                mode="llm",
                threshold=threshold,
                confidence_gate_passed=False,
                fallback_used=True,
                fallback_reason=fallback_reason,
                raw_llm_output=raw_output,
                reason=llm_result.get("message"),
            ),
        }

    proposed_intent = llm_result.get("intent")
    if proposed_intent == "unsupported":
        return {
            "intent": "unsupported",
            "router_debug": _router_debug(
                method="llm",
                mode="llm",
                threshold=threshold,
                confidence=confidence,
                confidence_gate_passed=True,
                raw_llm_output=raw_output,
                reason=reason,
            ),
        }

    if confidence is None or confidence < threshold:
        return {
            "intent": deterministic_route_intent(query),
            "router_debug": _router_debug(
                method="llm_fallback",
                mode="llm",
                threshold=threshold,
                confidence=confidence,
                confidence_gate_passed=False,
                fallback_used=True,
                fallback_reason="low_confidence",
                raw_llm_output=raw_output,
                reason=reason,
            ),
        }

    return {
        "intent": proposed_intent,
        "router_debug": _router_debug(
            method="llm",
            mode="llm",
            threshold=threshold,
            confidence=confidence,
            confidence_gate_passed=True,
            raw_llm_output=raw_output,
            reason=reason,
        ),
    }


def router_node(state: CortexState) -> dict:
    decision = route_intent(
        query=state["query"],
        user_tier=state.get("user_tier", "standard"),
        memory_context_text=state.get("memory_context_text", ""),
    )
    return {
        "intent": decision["intent"],
        "request_id": state.get("request_id") or new_request_id(),
        "debug": {
            **state.get("debug", {}),
            "router": decision["router_debug"],
        },
    }


def unsupported_agent(state: CortexState) -> dict:
    response = (
        "I could not determine a valid Cortex task from that query. "
        "Please ask about company knowledge, support actions, or external research."
    )
    return {
        "response": response,
        "agent_used": "unsupported",
        "tool_results": [],
        "cost_usd": 0.0,
        "citations": [],
        "used_chunks": [],
        "context": [],
        "debug": {
            **state.get("debug", {}),
            "unsupported": {"reason": "router_no_supported_intent"},
        },
    }


def route_by_intent(state: CortexState) -> str:
    intent = state.get("intent")
    if intent in {"knowledge", "action", "research", "unsupported"}:
        return intent
    return "unsupported"


def memory_enrichment_node(state: CortexState) -> dict:
    from dataclasses import asdict
    from memory.context import load_memory_context

    user_id = state.get("user_id") or "demo-user"
    session_id = state.get("session_id") or f"session_{uuid.uuid4().hex[:12]}"
    memory_context = load_memory_context(user_id=user_id, session_id=session_id)
    return {
        "session_id": session_id,
        "memory_context": asdict(memory_context),
        "memory_context_text": memory_context.context_text,
    }


def build_supervisor_graph():
    from agents.action_agent import action_agent
    from agents.knowledge_agent import knowledge_agent
    from agents.research_agent import research_agent

    graph = StateGraph(CortexState)
    graph.add_node("router", router_node)
    graph.add_node("memory", memory_enrichment_node)
    graph.add_node("knowledge", knowledge_agent)
    graph.add_node("action", action_agent)
    graph.add_node("research", research_agent)
    graph.add_node("unsupported", unsupported_agent)

    graph.set_entry_point("router")
    graph.add_edge("router", "memory")
    graph.add_conditional_edges(
        "memory",
        route_by_intent,
        {
            "knowledge": "knowledge",
            "action": "action",
            "research": "research",
            "unsupported": "unsupported",
        },
    )

    graph.add_edge("knowledge", END)
    graph.add_edge("action", END)
    graph.add_edge("research", END)
    graph.add_edge("unsupported", END)

    return graph.compile()


def create_initial_state(
    query: str,
    user_id: str = "demo-user",
    user_tier: str = "standard",
    session_id: str | None = None,
) -> CortexState:
    return {
        "query": query,
        "user_id": user_id,
        "user_tier": user_tier,
        "intent": None,
        "context": [],
        "response": None,
        "cost_usd": 0.0,
        "agent_used": None,
        "session_id": session_id or f"session_{uuid.uuid4().hex[:12]}",
        "memory_messages": [],
        "tool_results": [],
        "debug": {},
        "citations": [],
        "used_chunks": [],
        "request_id": new_request_id(),
        "trace": {},
        "memory_context": {},
        "memory_context_text": "",
    }


def invoke_with_trace(
    query: str,
    user_id: str = "demo-user",
    user_tier: str = "standard",
    session_id: str | None = None,
) -> dict:
    state = create_initial_state(
        query=query,
        user_id=user_id,
        user_tier=user_tier,
        session_id=session_id,
    )
    if session_id and not state.get("session_id"):
        state["session_id"] = session_id
    if not state.get("request_id"):
        state["request_id"] = new_request_id()

    start = time.perf_counter()
    try:
        final_state = build_supervisor_graph().invoke(state)
        latency_ms = max((time.perf_counter() - start) * 1000.0, 0.0)
        event = build_query_trace_event(
            initial_state=state,
            final_state=final_state,
            latency_ms=latency_ms,
            exc=None,
        )
        logged_event = log_query_event(event)

        existing_trace = final_state.get("trace") if isinstance(final_state.get("trace"), dict) else {}
        final_state["trace"] = {
            **existing_trace,
            "latency_ms": latency_ms,
            "log_event": logged_event,
            "log_path": str(get_log_path()),
        }
        return final_state
    except Exception as exc:
        latency_ms = max((time.perf_counter() - start) * 1000.0, 0.0)
        event = build_query_trace_event(
            initial_state=state,
            final_state=None,
            latency_ms=latency_ms,
            exc=exc,
        )
        log_query_event(event)
        raise
