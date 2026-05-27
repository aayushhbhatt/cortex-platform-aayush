from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
import uuid


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:12]}"


def summarize_tool_results(tool_results: list[dict] | None) -> dict[str, Any]:
    results = tool_results or []
    tool_names: list[str] = []
    tool_error_types: list[str] = []

    for item in results:
        if not isinstance(item, dict):
            continue
        tool_name = item.get("tool_name") or item.get("name")
        if tool_name:
            tool_names.append(str(tool_name))

        err = item.get("error_type")
        nested_err = item.get("error", {}).get("error_type") if isinstance(item.get("error"), dict) else None
        if item.get("success") is False:
            if err:
                tool_error_types.append(str(err))
            elif nested_err:
                tool_error_types.append(str(nested_err))

    return {
        "tool_results_count": len(results),
        "tool_names": tool_names,
        "tool_error_types": tool_error_types,
    }


def extract_error_type(final_state: dict | None, exc: Exception | None = None) -> str | None:
    if exc is not None:
        return exc.__class__.__name__.replace("Error", "").lower() + "_error"

    if not isinstance(final_state, dict):
        return None

    debug = final_state.get("debug") if isinstance(final_state.get("debug"), dict) else {}
    debug_error = debug.get("error") if isinstance(debug.get("error"), dict) else {}
    if debug_error.get("error_type"):
        return str(debug_error["error_type"])

    debug_cost = debug.get("cost") if isinstance(debug.get("cost"), dict) else {}
    if debug_cost.get("budget_exceeded"):
        return "budget_exceeded"

    for item in final_state.get("tool_results") or []:
        if not isinstance(item, dict):
            continue
        if item.get("success") is False:
            if item.get("error_type"):
                return str(item["error_type"])
            nested_error = item.get("error")
            if isinstance(nested_error, dict) and nested_error.get("error_type"):
                return str(nested_error["error_type"])

    return None


def build_query_trace_event(*, initial_state: dict, final_state: dict | None, latency_ms: float, exc: Exception | None = None) -> dict:
    summary = summarize_tool_results((final_state or {}).get("tool_results"))
    error_type = extract_error_type(final_state, exc=exc)

    request_id = (final_state or {}).get("request_id") or initial_state.get("request_id")
    user_id = (final_state or {}).get("user_id") or initial_state.get("user_id")
    session_id = (final_state or {}).get("session_id") or initial_state.get("session_id")
    intent = (final_state or {}).get("intent") or initial_state.get("intent")
    agent_used = (final_state or {}).get("agent_used")
    cost_usd = (final_state or {}).get("cost_usd", 0.0)

    event: dict[str, Any] = {
        "event_type": "query_trace",
        "request_id": request_id,
        "user_id": user_id,
        "session_id": session_id,
        "intent": intent,
        "agent_used": agent_used,
        "latency_ms": float(latency_ms),
        "cost_usd": float(cost_usd or 0.0),
        "tool_results_count": int(summary["tool_results_count"]),
        "error_type": error_type,
        "success": error_type is None and exc is None,
        "query": (final_state or {}).get("query") or initial_state.get("query"),
        "response_chars": len((final_state or {}).get("response") or ""),
        "citations_count": len((final_state or {}).get("citations") or []),
        "used_chunks_count": len((final_state or {}).get("used_chunks") or []),
        "memory_messages_count": len((final_state or {}).get("memory_messages") or []),
        "tool_names": summary["tool_names"],
        "tool_error_types": summary["tool_error_types"],
    }

    if final_state is None and exc is not None:
        event["agent_used"] = None
        event["cost_usd"] = 0.0
        event["tool_results_count"] = 0
        event["success"] = False
        if not event["error_type"]:
            event["error_type"] = "unknown_error"

    return event
