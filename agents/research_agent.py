from dataclasses import asdict
import re

from agents.supervisor import CortexState
from memory.context import MemoryContext, finalize_agent_memory, load_memory_context
from reliability.cost_tracker import CostTracker, estimate_tool_cost
from reliability.errors import BudgetExceededError, classify_error
from tools.research_tools import web_search_impl
from tools.tool_contracts import error_response

ACADEMIC_TERMS = {"paper", "research paper", "study", "journal", "arxiv", "citation", "scholarly", "literature", "benchmark"}
POLICY_TERMS = {"policy", "regulation", "compliance", "law", "governance", "standard", "framework", "risk", "security policy"}
AMBIGUOUS_TERMS = {"what about recent trends", "find more about that", "what is happening in this area", "what is happening in this space", "research this more", "any recent updates", "what are current trends", "tell me more about this", "what are recent developments"}
RESEARCH_NOTE_KEYWORDS = {"ai", "governance", "compliance", "rag", "retrieval", "llm", "agent", "security", "cloud", "platform", "policy", "regulation", "automation", "data", "infrastructure"}
WEAK_PROFILE_TYPES = {"department", "role", "location", "team", "work_style"}
WEAK_QUERY_MARKERS = {"in my department": "department", "for my team": "team", "in my location": "location", "for my role": "role"}
SENSITIVE_HINTS = {"health", "medical", "medication", "anxiety", "depression", "therapy", "religion", "political", "ethnicity", "race", "sexuality", "union", "criminal", "pregnancy", "disability"}


def _is_ambiguous_query(query: str) -> bool:
    q = query.lower().strip(" ?.!")
    return any(term in q for term in AMBIGUOUS_TERMS)


def _classify_research_intent(query: str, memory_context=None) -> str:
    q = query.lower()
    if any(t in q for t in ACADEMIC_TERMS):
        return "academic"
    if any(t in q for t in POLICY_TERMS):
        return "policy"
    if _is_ambiguous_query(query) and memory_context and getattr(memory_context, "context_text", ""):
        ctxt = memory_context.context_text.lower()
        if any(t in ctxt for t in ACADEMIC_TERMS):
            return "academic"
        if any(t in ctxt for t in POLICY_TERMS):
            return "policy"
    return "general"


def _memory_context_to_dict(memory_context) -> dict:
    if not memory_context:
        return {}
    if isinstance(memory_context, dict):
        return memory_context
    if hasattr(memory_context, "model_dump"):
        return memory_context.model_dump()
    return {
        "entities": getattr(memory_context, "entities", []),
        "recent_messages": getattr(memory_context, "recent_messages", []),
        "context_text": getattr(memory_context, "context_text", ""),
    }


def _contains_sensitive_hint(text: str) -> bool:
    lowered = (text or "").lower()
    return any(h in lowered for h in SENSITIVE_HINTS)


def _extract_research_topic_from_entities(memory_context, query: str) -> tuple[str | None, str]:
    context = _memory_context_to_dict(memory_context)
    entities = context.get("entities", []) or []
    q = query.lower()

    def _entity_parts(ent):
        if isinstance(ent, dict):
            return ent.get("entity_type"), ent.get("entity_value")
        return getattr(ent, "entity_type", None), getattr(ent, "entity_value", None)

    explicit_weak_type = None
    for marker, weak_type in WEAK_QUERY_MARKERS.items():
        if marker in q:
            explicit_weak_type = weak_type
            break

    projects, notes, prefs, weak_explicit = [], [], [], []
    for ent in entities:
        ent_type, ent_value = _entity_parts(ent)
        if not ent_type or not ent_value:
            continue
        value = str(ent_value).strip()
        if not value or _contains_sensitive_hint(value):
            continue
        lower_value = value.lower()
        if ent_type == "project":
            projects.append(value)
        elif ent_type == "note" and any(k in lower_value for k in RESEARCH_NOTE_KEYWORDS):
            notes.append(value)
        elif ent_type == "preference" and any(k in lower_value for k in RESEARCH_NOTE_KEYWORDS):
            prefs.append(value)
        elif explicit_weak_type and ent_type == explicit_weak_type:
            weak_explicit.append(value)

    for bucket in (projects, notes, prefs, weak_explicit):
        if bucket:
            return bucket[0], "entity"
    return None, "none"


def _extract_research_topic_from_recent_messages(memory_context) -> tuple[str | None, str]:
    context = _memory_context_to_dict(memory_context)
    messages = context.get("recent_messages", []) or []
    patterns = [
        r"i am working on\s+([^.!?]+)",
        r"my project is\s+([^.!?]+)",
        r"research\s+([^.!?]+)",
        r"trends in\s+([^.!?]+)",
        r"tell me about\s+([^.!?]+)",
        r"what is happening with\s+([^.!?]+)",
    ]
    action_markers = {"create", "ticket", "open", "assign", "escalate", "close"}

    def _extract_topic(text: str) -> str | None:
        lowered = text.lower().strip()
        if any(x in lowered for x in action_markers) and not any(k in lowered for k in RESEARCH_NOTE_KEYWORDS):
            return None
        if _contains_sensitive_hint(lowered):
            return None
        for pat in patterns:
            m = re.search(pat, lowered)
            if m:
                raw = m.group(1).strip(" .!?")
                if raw and not _contains_sensitive_hint(raw):
                    return text[text.lower().find(raw):text.lower().find(raw)+len(raw)] if raw in text.lower() else raw
        return None

    assistant_candidates = []
    for msg in reversed(messages):
        role = (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")) or ""
        content = (msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or ""
        if not content:
            continue
        topic = _extract_topic(content)
        if not topic:
            continue
        if role == "user":
            return topic, "recent_message"
        if role == "assistant":
            assistant_candidates.append(topic)
    if assistant_candidates:
        return assistant_candidates[0], "recent_message"
    return None, "none"


def _rewrite_ambiguous_research_query(query: str, topic: str) -> str:
    trimmed = query.strip()
    base = trimmed.rstrip(" ?.!")
    q_lower = trimmed.lower()
    if "current trends" in q_lower or "recent trends" in q_lower or "recent developments" in q_lower:
        rewritten = f"{base} in {topic}?"
    elif "find more about that" in q_lower:
        rewritten = f"Find more about {topic}."
    elif "what is happening in this area" in q_lower or "what is happening in this space" in q_lower:
        rewritten = f"What is happening in {topic}?"
    else:
        rewritten = f"{base} about {topic}?"
    return rewritten[:300]


def _build_memory_aware_research_query(query: str, memory_context=None) -> tuple[str, str, str | None]:
    if not _is_ambiguous_query(query) or not memory_context:
        return query, "none", None
    topic, source = _extract_research_topic_from_entities(memory_context, query)
    if not topic:
        topic, source = _extract_research_topic_from_recent_messages(memory_context)
    if not topic:
        return query, "none", None
    return _rewrite_ambiguous_research_query(query, topic), source, topic


def _summarize_search_results(results: list[dict], max_items: int = 3) -> str:
    top = results[:max_items]
    lines = [f"Found {len(results)} external research results:"]
    for i, r in enumerate(top, 1):
        snippet = (r.get("snippet") or "").strip()
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        lines.append(f"{i}. {r.get('title', 'Untitled')} — {snippet}")
    return "\n".join(lines)


def _safe_tool_summary(tool_result: dict) -> str:
    if tool_result.get("success"):
        data = tool_result.get("data", {})
        results = data.get("results", [])
        if not results:
            return data.get("message", "External research is currently offline/not implemented for this environment.")
        return _summarize_search_results(results)

    error = tool_result.get("error")
    error_type = tool_result.get("error_type") or (getattr(error, "error_type", None) if error else None) or (error.get("error_type") if isinstance(error, dict) else None) or "unknown_error"
    message = tool_result.get("message") or (getattr(error, "message", None) if error else None) or (error.get("message") if isinstance(error, dict) else None) or "No details provided."
    return f"Research tool error ({error_type}): {message}"


def research_agent(state: CortexState) -> dict:
    query = state["query"]
    user_id = state.get("user_id", "unknown")
    session_id = state.get("session_id") or "session_local"
    if state.get("memory_context"):
        memory_context = MemoryContext(**state["memory_context"])
    else:
        memory_context = load_memory_context(user_id=user_id, session_id=session_id)

    effective_query, rewrite_source, rewrite_topic = _build_memory_aware_research_query(query, memory_context)
    intent = _classify_research_intent(effective_query, memory_context)
    tracker = CostTracker()
    tracker.add_cost("tools", "web_search", estimate_tool_cost(1), {"tool_results_count": 1})
    debug = {
        "cost": tracker.to_dict(),
        "research_intent": intent,
        "original_query": query,
        "effective_query": effective_query,
        "memory_rewrite_applied": effective_query != query,
        "memory_rewrite_source": rewrite_source,
        "memory_rewrite_topic": rewrite_topic,
    }

    budget_precheck = {"checked": True, "passed": True, "tool_call_skipped": False}
    try:
        tracker.check_budget()
    except BudgetExceededError as exc:
        budget_precheck.update({"passed": False, "tool_call_skipped": True})
        err = classify_error(exc)
        debug["budget_precheck"] = budget_precheck
        debug["error"] = {"error_type": err.error_type, "message": err.message, "retryable": err.retryable, "details": err.details}
        response = f"Request exceeded budget: {err.message}"
        finalizer = finalize_agent_memory(user_id=user_id, session_id=session_id, query=query, response=response)
        return {"response": response, "agent_used": "research", "tool_results": [], "cost_usd": tracker.total_cost(), "debug": {**debug, "memory_finalizer": finalizer["memory_debug"]}, "memory_messages": finalizer["memory_messages"], "memory_context": asdict(memory_context), "memory_context_text": memory_context.context_text}

    debug["budget_precheck"] = budget_precheck
    try:
        tool_result = web_search_impl(query=effective_query, intent=intent, max_results=3)
    except Exception as exc:
        tool_error = classify_error(exc)
        tool_result = error_response("web_search", tool_error)

    response = _safe_tool_summary(tool_result)
    finalizer = finalize_agent_memory(user_id=user_id, session_id=session_id, query=query, response=response)
    debug["memory_finalizer"] = finalizer["memory_debug"]
    return {"response": response, "agent_used": "research", "tool_results": [tool_result], "cost_usd": tracker.total_cost(), "debug": debug, "memory_messages": finalizer["memory_messages"], "memory_context": asdict(memory_context), "memory_context_text": memory_context.context_text}
