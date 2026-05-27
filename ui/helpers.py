from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


AGENT_DESCRIPTIONS = {
    "knowledge": {
        "label": "Knowledge",
        "icon": "📚",
        "summary": "Answers from internal company documents with RBAC-aware RAG and citations.",
    },
    "action": {
        "label": "Action",
        "icon": "🛠️",
        "summary": "Triggers controlled workflows through validated tools.",
    },
    "research": {
        "label": "Research",
        "icon": "🌐",
        "summary": "Uses external research tools for information outside the company corpus.",
    },
    "unsupported": {
        "label": "Unsupported",
        "icon": "⚠️",
        "summary": "Returns a safe fallback when Cortex cannot confidently handle the request.",
    },
}


def safe_jsonable(value: Any):
    """Convert dataclasses and objects into Streamlit-displayable values."""
    if is_dataclass(value):
        return {key: safe_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): safe_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [safe_jsonable(item) for item in value]
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: safe_jsonable(item) for key, item in vars(value).items()}
    return value


def get_nested(data: dict | None, path: list[str], default=None):
    current = data or {}
    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def truncate_text(value: Any, limit: int = 180) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def fmt_money(value: Any) -> str:
    try:
        return f"${float(value):.6f}"
    except Exception:
        return "$0.000000"


def get_agent_debug(final_state: dict | None) -> dict:
    """Return agent debug regardless of whether implementation stores it flat or under debug.agent."""
    debug = (final_state or {}).get("debug") or {}
    if not isinstance(debug, dict):
        return {}
    nested_agent = debug.get("agent")
    if isinstance(nested_agent, dict):
        return nested_agent
    return debug


def get_router_debug(final_state: dict | None) -> dict:
    debug = (final_state or {}).get("debug") or {}
    if not isinstance(debug, dict):
        return {}
    nested_router = debug.get("router")
    if isinstance(nested_router, dict):
        return nested_router
    router_keys = {
        "method",
        "router_method",
        "llm_enabled",
        "confidence",
        "fallback_used",
        "fallback_reason",
        "confidence_gate",
        "arguments",
    }
    return {key: debug.get(key) for key in router_keys if key in debug}


def summarize_router_debug(final_state: dict | None) -> dict:
    router_debug = get_router_debug(final_state)
    return {
        "router_method": router_debug.get("method") or router_debug.get("router_method") or "deterministic/unknown",
        "llm_enabled": router_debug.get("llm_enabled"),
        "confidence": router_debug.get("confidence"),
        "fallback_used": bool(router_debug.get("fallback_used", False)),
        "fallback_reason": router_debug.get("fallback_reason"),
        "confidence_gate": router_debug.get("confidence_gate"),
        "arguments": router_debug.get("arguments") or {},
    }


def memory_summary(final_state: dict | None) -> dict:
    recent_messages = get_nested(final_state, ["memory_context", "recent_messages"], default=[]) or []
    entities = get_nested(final_state, ["memory_context", "entities"], default=[]) or []
    context_text = (final_state or {}).get("memory_context_text")
    memory_messages = (final_state or {}).get("memory_messages") or []
    return {
        "recent_message_count": len(recent_messages),
        "entity_count": len(entities),
        "context_text_available": bool(context_text),
        "memory_message_count": len(memory_messages),
    }


def get_tool_data(item: dict | None) -> dict:
    item = item or {}
    data = item.get("data")
    return data if isinstance(data, dict) else {}


def get_tool_meta(item: dict | None) -> dict:
    item = item or {}
    meta = item.get("meta")
    return meta if isinstance(meta, dict) else {}


def tool_primary_id(item: dict | None) -> str:
    item = item or {}
    data = get_tool_data(item)
    for key in ("ticket_id", "escalation_id", "notification_id", "request_id", "id"):
        if item.get(key):
            return str(item[key])
        if data.get(key):
            return str(data[key])
    return ""


def tool_results_to_rows(tool_results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for item in tool_results or []:
        data = get_tool_data(item)
        meta = get_tool_meta(item)
        rows.append(
            {
                "status": "success" if item.get("success") else "error",
                "tool": item.get("tool_name") or item.get("tool") or "unknown",
                "primary_id": tool_primary_id(item),
                "summary": truncate_text(
                    item.get("message")
                    or data.get("message")
                    or data.get("title")
                    or item.get("error")
                    or item.get("error_type")
                    or "",
                    140,
                ),
                "mode": meta.get("mode") or item.get("mode") or "",
                "error_type": item.get("error_type") or get_nested(item, ["error", "error_type"], default=""),
                "recoverable": item.get("recoverable", ""),
            }
        )
    return rows


def citations_to_rows(citations) -> list[dict]:
    rows = []
    for idx, citation in enumerate(citations or [], start=1):
        item = safe_jsonable(citation)
        rows.append(
            {
                "rank": idx,
                "title": item.get("title", ""),
                "doc_id": item.get("doc_id", ""),
                "chunk_id": item.get("chunk_id", ""),
                "access": item.get("access_level", ""),
                "source": item.get("source", ""),
            }
        )
    return rows


def used_chunks_to_rows(used_chunks) -> list[dict]:
    rows = []
    for idx, chunk in enumerate(used_chunks or [], start=1):
        item = safe_jsonable(chunk)
        if isinstance(item, dict):
            rows.append(
                {
                    "rank": idx,
                    "chunk_id": item.get("chunk_id") or item.get("id") or truncate_text(item, 90),
                    "title": item.get("title", ""),
                    "access": item.get("access_level", ""),
                    "score": item.get("score", ""),
                }
            )
        else:
            rows.append({"rank": idx, "chunk_id": str(item), "title": "", "access": "", "score": ""})
    return rows


def research_results_to_rows(tool_results: list[dict]) -> list[dict]:
    rows = []
    for tool in tool_results or []:
        data = get_tool_data(tool)
        for idx, result in enumerate(data.get("results") or [], start=1):
            rows.append(
                {
                    "rank": idx,
                    "title": truncate_text(result.get("title", "Untitled"), 90),
                    "snippet": truncate_text(result.get("snippet", ""), 180),
                    "source": result.get("source") or result.get("url") or result.get("link") or "",
                }
            )
    return rows


def build_run_summary(final_state: dict | None) -> dict:
    fs = final_state or {}
    agent = fs.get("agent_used") or "unknown"
    intent = fs.get("intent") or "unknown"
    tools = fs.get("tool_results") or []
    mem = memory_summary(fs)
    return {
        "agent": agent,
        "intent": intent,
        "cost": fmt_money(fs.get("cost_usd", 0.0)),
        "tools": len(tools),
        "citations": len(fs.get("citations") or []),
        "memory": f"{mem['recent_message_count']} msg / {mem['entity_count']} facts",
        "request_id": fs.get("request_id", "N/A"),
    }


def build_timeline_rows(final_state: dict | None) -> list[dict]:
    fs = final_state or {}
    agent = fs.get("agent_used") or "unknown"
    intent = fs.get("intent") or "unknown"
    tools = fs.get("tool_results") or []
    mem = memory_summary(fs)
    agent_debug = get_agent_debug(fs)
    router = summarize_router_debug(fs)

    if agent == "knowledge":
        execution = f"RAG retrieval completed with {len(fs.get('citations') or [])} citations and {len(fs.get('used_chunks') or [])} used chunks."
    elif agent == "action":
        execution = f"Action workflow: {agent_debug.get('action_type') or agent_debug.get('mapped_tool') or 'action tool'}."
    elif agent == "research":
        execution = f"Research query: {agent_debug.get('effective_query') or fs.get('query')}"
    elif agent == "unsupported":
        execution = "No specialist workflow executed."
    else:
        execution = "No execution details available."

    return [
        {"step": "01", "stage": "Request", "detail": truncate_text(fs.get("query", ""), 180)},
        {"step": "02", "stage": "Route", "detail": f"{intent} via {router['router_method']}"},
        {"step": "03", "stage": "Memory", "detail": f"Loaded {mem['recent_message_count']} recent messages and {mem['entity_count']} facts"},
        {"step": "04", "stage": "Agent", "detail": f"{agent.title()} agent selected"},
        {"step": "05", "stage": "Execution", "detail": execution},
        {"step": "06", "stage": "Tools", "detail": f"{len(tools)} tool result(s) returned"},
        {"step": "07", "stage": "Response", "detail": "Response generated" if fs.get("response") else "No response"},
    ]


# Backwards-compatible names used by existing tests/UI.
def retrieval_results_to_rows(results) -> list[dict]:
    rows = []
    for rank, result in enumerate(results or [], start=1):
        rows.append(
            {
                "rank": rank,
                "title": getattr(result, "title", ""),
                "doc_id": getattr(result, "doc_id", ""),
                "chunk_id": getattr(result, "chunk_id", ""),
                "access_level": getattr(result, "access_level", ""),
                "category": getattr(result, "category", ""),
                "score": getattr(result, "score", 0.0),
                "retrieval_method": getattr(result, "retrieval_method", ""),
                "fused_from": ", ".join(getattr(result, "fused_from", []) or []),
                "source": getattr(result, "source", ""),
            }
        )
    return rows


def get_first_result_metadata(results) -> dict:
    if not results:
        return {}
    return dict(getattr(results[0], "metadata", {}) or {})


def build_execution_trace_rows(final_state: dict) -> list[dict]:
    return build_timeline_rows(final_state)


def build_agent_detail_rows(final_state: dict) -> list[dict]:
    fs = final_state or {}
    agent = fs.get("agent_used")
    debug = get_agent_debug(fs)
    first_tool = (fs.get("tool_results") or [{}])[0]
    if agent == "knowledge":
        return [
            {"label": "Original query", "value": fs.get("query")},
            {"label": "Effective query", "value": debug.get("effective_query")},
            {"label": "User tier", "value": fs.get("user_tier")},
            {"label": "Citations", "value": len(fs.get("citations") or [])},
            {"label": "Used chunks", "value": len(fs.get("used_chunks") or [])},
            {"label": "Tool", "value": "search_knowledge_base"},
        ]
    if agent == "action":
        return [
            {"label": "Action type", "value": debug.get("action_type") or fs.get("intent")},
            {"label": "Mapped tool", "value": debug.get("mapped_tool") or first_tool.get("tool_name")},
            {"label": "Side effect", "value": debug.get("side_effect_executed", first_tool.get("success"))},
            {"label": "Budget precheck", "value": debug.get("budget_precheck")},
            {"label": "Primary ID", "value": tool_primary_id(first_tool)},
        ]
    if agent == "research":
        return [
            {"label": "Research intent", "value": debug.get("research_intent") or fs.get("intent")},
            {"label": "Original query", "value": debug.get("original_query") or fs.get("query")},
            {"label": "Effective query", "value": debug.get("effective_query")},
            {"label": "Memory rewrite", "value": debug.get("memory_rewrite_applied")},
            {"label": "Provider mode", "value": get_tool_meta(first_tool).get("mode")},
        ]
    return [{"label": "Status", "value": "No agent details available."}]


def build_agent_capability_rows() -> list[dict]:
    return [
        {"agent": "Supervisor", "capability": "Routes to the right specialist.", "signal": "intent + router debug"},
        {"agent": "Knowledge", "capability": "RAG over company documents.", "signal": "citations + used chunks"},
        {"agent": "Action", "capability": "Runs workflow tools.", "signal": "tool result + budget precheck"},
        {"agent": "Research", "capability": "External research via web_search.", "signal": "provider mode + search results"},
        {"agent": "Memory", "capability": "Loads recent session and user facts.", "signal": "messages + entities"},
    ]


def build_routing_explanation(intent: str | None, agent_used: str | None) -> str:
    route = (agent_used or intent or "").lower()
    if route == "knowledge":
        return "Policy, document, or internal knowledge query routed to the Knowledge Agent."
    if route == "action":
        return "Workflow, ticket, notification, or escalation request routed to the Action Agent."
    if route == "research":
        return "Open-ended or external information request routed to the Research Agent."
    if route == "unsupported":
        return "Cortex did not find a supported specialist workflow for this request."
    return "Routing outcome was not available."


def get_knowledge_tool_data(final_state: dict | None) -> dict:
    for tool in (final_state or {}).get("tool_results", []):
        if tool.get("tool_name") == "search_knowledge_base" and tool.get("success"):
            data = tool.get("data")
            return data if isinstance(data, dict) else {}
    return {}


def get_knowledge_rag_debug(final_state: dict | None) -> dict:
    data = get_knowledge_tool_data(final_state)
    debug = data.get("debug") if isinstance(data.get("debug"), dict) else {}
    return {
        "retrieval_trace": debug.get("retrieval_trace") or {},
        "ranking_debug": debug.get("ranking_debug") or {},
        "access_debug": debug.get("access_debug") or {},
        "query_understanding": debug.get("query_understanding") or {},
        "generation_debug": {key: debug.get(key) for key in ("mode", "result_count", "used_chunk_count", "top_score", "retrieval_methods", "query_type") if key in debug},
        "retrieval_results": data.get("retrieval_results") or [],
    }


def rag_query_processing_rows(final_state: dict | None) -> list[dict]:
    debug = get_agent_debug(final_state)
    rag = get_knowledge_rag_debug(final_state)
    rewrite_debug = debug.get("llm_query_rewrite_debug") or {}
    query_understanding = rag["query_understanding"]
    filters = query_understanding.get("filters", {}) or {}
    trace = rag["retrieval_trace"]
    return [{"field":"Original query","value":debug.get("original_query") or (final_state or {}).get("query")},{"field":"Deterministic memory-aware query","value":debug.get("deterministic_effective_query")},{"field":"Final effective query","value":debug.get("final_effective_query")},{"field":"LLM rewrite mode","value":rewrite_debug.get("mode")},{"field":"LLM rewrite confidence","value":rewrite_debug.get("confidence")},{"field":"Expanded queries used","value":", ".join(trace.get("expanded_queries_used") or [])},{"field":"Query type","value":query_understanding.get("query_type")},{"field":"Category filter","value":filters.get("category")},{"field":"Doc ID hint","value":filters.get("doc_id_hint")},{"field":"Section hint","value":filters.get("section_hint")},{"field":"Access hint","value":filters.get("access_level_hint")}]


def rag_access_rows(final_state: dict | None) -> list[dict]:
    access = get_knowledge_rag_debug(final_state)["access_debug"]
    return [{"field":"User tier","value":access.get("user_tier")},{"field":"Allowed access levels","value":", ".join(access.get("allowed_access_levels") or [])},{"field":"Chunks before filter","value":access.get("total_chunks_before_filter")},{"field":"Chunks after filter","value":access.get("total_chunks_after_filter")},{"field":"Filtered out count","value":access.get("filtered_out_count")}]


def rag_retrieval_results_rows(final_state: dict | None) -> list[dict]:
    results = get_knowledge_rag_debug(final_state)["retrieval_results"]
    rows = []
    for item in results:
        rerank = item.get("rerank_debug") or {}
        rows.append({"rank": item.get("rank"),"doc_id": item.get("doc_id"),"title": item.get("title"),"chunk_id": item.get("chunk_id"),"score": item.get("score"),"fused_from": ", ".join(item.get("fused_from") or []),"bm25_rank": (item.get("ranks") or {}).get("bm25"),"vector_rank": (item.get("ranks") or {}).get("vector"),"boost": rerank.get("boost_score"),"boost_reasons": ", ".join(rerank.get("boost_reasons") or []),"section": item.get("section_title"),"chunk_strategy": item.get("chunk_strategy"),"access": item.get("access_level"),"preview": truncate_text(item.get("content_preview"), 220)})
    return rows

def rag_ranking_debug_rows(final_state: dict | None, key: str) -> list[dict]:
    return list(get_knowledge_rag_debug(final_state)["ranking_debug"].get(key) or [])


def rag_generation_rows(final_state: dict | None) -> list[dict]:
    rag = get_knowledge_rag_debug(final_state)
    generation = rag["generation_debug"]
    context_used = get_knowledge_tool_data(final_state).get("debug", {}).get("context_used", [])
    return [{"field":"Generation mode","value":generation.get("mode")},{"field":"Used chunk count","value":generation.get("used_chunk_count")},{"field":"Citations count","value":len((final_state or {}).get("citations") or [])},{"field":"Context length","value":sum(len(str(x)) for x in context_used)},{"field":"Top score","value":generation.get("top_score")},{"field":"Retrieval methods used","value":", ".join(generation.get("retrieval_methods") or [])}]
