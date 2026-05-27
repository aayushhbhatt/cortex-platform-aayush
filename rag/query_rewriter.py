from __future__ import annotations

import json
import os
from pydantic import BaseModel, Field

from llm.client import OpenAIChatClient, build_json_request
from llm.prompt_loader import load_prompt
from dotenv import load_dotenv
load_dotenv()

SENSITIVE_TERMS = {
    "health", "medical", "medication", "anxiety", "depression", "therapy", "religion",
    "politics", "ethnicity", "race", "sexuality", "union", "criminal", "pregnancy", "disability",
}


class KnowledgeQueryRewrite(BaseModel):
    rewritten_query: str = Field(..., min_length=3, max_length=500)
    expanded_queries: list[str] = Field(default_factory=list, max_length=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)


def get_knowledge_query_rewrite_mode() -> str:
    return os.getenv("KNOWLEDGE_QUERY_REWRITE_MODE", "deterministic").strip().lower() or "deterministic"


def get_knowledge_query_rewrite_confidence_threshold() -> float:
    raw = os.getenv("KNOWLEDGE_QUERY_REWRITE_CONFIDENCE_THRESHOLD", "0.75")
    try:
        return float(raw)
    except ValueError:
        return 0.75


def parse_knowledge_query_rewrite(raw_output: str) -> KnowledgeQueryRewrite:
    payload = json.loads(raw_output)
    return KnowledgeQueryRewrite.model_validate(payload)


def _sanitize_memory(memory_context_text: str) -> str:
    lowered = memory_context_text.lower()
    if any(term in lowered for term in SENSITIVE_TERMS):
        return "[memory omitted due to sensitivity]"
    return memory_context_text


def llm_rewrite_knowledge_query(query: str, memory_context_text: str = "", user_tier: str = "standard") -> dict:
    try:
        prompt = load_prompt("knowledge")
        client = OpenAIChatClient()
        user_prompt = (
            "Rewrite this internal company retrieval query. Return JSON only with keys: "
            "rewritten_query, expanded_queries, confidence, reason. Preserve user intent, do not invent facts. "
            f"User tier: {user_tier}. Query: {query}\nMemory context: {_sanitize_memory(memory_context_text)}"
        )
        response = client.complete(build_json_request(agent="knowledge", prompt=prompt, user_prompt=user_prompt, max_tokens=350))
        if not response.success:
            return {"ok": False, "rewrite": None, "raw_output": None, "error_type": response.error_type, "message": response.message}
        rewrite = parse_knowledge_query_rewrite(response.content or "")
        return {"ok": True, "rewrite": rewrite, "raw_output": response.content, "error_type": None, "message": None}
    except Exception as exc:
        return {"ok": False, "rewrite": None, "raw_output": None, "error_type": "provider_error", "message": str(exc)}


def rewrite_query_for_retrieval(query: str, memory_context_text: str = "", user_tier: str = "standard") -> dict:
    mode = get_knowledge_query_rewrite_mode()
    threshold = get_knowledge_query_rewrite_confidence_threshold()
    if mode != "llm":
        return {
            "query": query[:500],
            "expanded_queries": None,
            "debug": {"mode": "deterministic", "method": "deterministic", "confidence": None, "confidence_threshold": threshold, "fallback_used": False, "fallback_reason": None, "reason": None, "raw_llm_output": None},
        }

    result = llm_rewrite_knowledge_query(query=query, memory_context_text=memory_context_text, user_tier=user_tier)
    if not result["ok"] or not result["rewrite"]:
        return {
            "query": query[:500],
            "expanded_queries": None,
            "debug": {"mode": "llm", "method": "llm_fallback", "confidence": None, "confidence_threshold": threshold, "fallback_used": True, "fallback_reason": result.get("error_type") or "rewrite_failed", "reason": None, "raw_llm_output": result.get("raw_output")},
        }

    rewrite: KnowledgeQueryRewrite = result["rewrite"]
    if rewrite.confidence < threshold:
        return {
            "query": query[:500],
            "expanded_queries": None,
            "debug": {"mode": "llm", "method": "llm_fallback", "confidence": rewrite.confidence, "confidence_threshold": threshold, "fallback_used": True, "fallback_reason": "low_confidence", "reason": rewrite.reason, "raw_llm_output": result.get("raw_output")},
        }

    expanded: list[str] = []
    seen: set[str] = set()
    for item in rewrite.expanded_queries[:5]:
        clean = " ".join(item.split())[:500]
        if clean and clean not in seen:
            seen.add(clean)
            expanded.append(clean)

    rewritten = " ".join(rewrite.rewritten_query.split())[:500]
    if not rewritten:
        rewritten = query[:500]
    return {
        "query": rewritten,
        "expanded_queries": expanded or None,
        "debug": {"mode": "llm", "method": "llm", "confidence": rewrite.confidence, "confidence_threshold": threshold, "fallback_used": False, "fallback_reason": None, "reason": rewrite.reason, "raw_llm_output": result.get("raw_output")},
    }
