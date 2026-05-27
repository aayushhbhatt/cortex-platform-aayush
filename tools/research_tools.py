import json
import os
import urllib.parse
import urllib.request
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from pydantic import ValidationError

from tools.error_contracts import ProviderAuthError, ProviderError
from reliability.runtime import RESEARCH_TOOL_POLICY, RuntimeReliabilityPolicy, execute_with_reliability
from tools.error_contracts import error_response, success_response, validation_error_response
from tools.schemas import ResearchQueryInput

DUCKDUCKGO_POLICY = RuntimeReliabilityPolicy(tool_name="web_search.duckduckgo")
SERPAPI_POLICY = RuntimeReliabilityPolicy(tool_name="web_search.serpapi")


def _is_live_web_search_enabled() -> bool:
    value = os.getenv("ENABLE_LIVE_WEB_SEARCH", "false").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    url = "https://duckduckgo.com/?" + urllib.parse.urlencode({"q": query, "format": "json", "no_redirect": 1, "no_html": 1})
    req = urllib.request.Request(url, headers={"User-Agent": "cortex-research-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise ProviderError(f"DuckDuckGo search failed: {exc}") from exc

    results: list[dict] = []
    for item in payload.get("RelatedTopics", []) or []:
        if len(results) >= max_results:
            break
        if isinstance(item, dict) and item.get("FirstURL") and item.get("Text"):
            results.append({"title": item.get("Text", "")[:120], "url": item.get("FirstURL", ""), "snippet": item.get("Text", ""), "provider": "duckduckgo"})
    return results


def _search_serpapi(query: str, max_results: int) -> list[dict]:
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        raise ProviderAuthError("SERPAPI_API_KEY is not configured.")
    url = "https://serpapi.com/search.json?" + urllib.parse.urlencode({"q": query, "api_key": api_key, "num": max_results, "engine": "google"})
    req = urllib.request.Request(url, headers={"User-Agent": "cortex-research-agent/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise ProviderError(f"SerpAPI search failed: {exc}") from exc

    results: list[dict] = []
    for item in payload.get("organic_results", []) or []:
        if len(results) >= max_results:
            break
        results.append({"title": item.get("title", ""), "url": item.get("link", ""), "snippet": item.get("snippet", ""), "provider": "serpapi"})
    return results


def _web_search_offline_validated(inp: ResearchQueryInput) -> dict:
    return success_response(
        tool_name="web_search",
        data={
            "query": inp.query,
            "intent": inp.intent,
            "results": [],
            "message": "External web search is not implemented in offline mode.",
        },
        meta={"mode": "offline", "input_schema": "ResearchQueryInput"},
    ).model_dump()


def _web_search_live_validated(inp: ResearchQueryInput) -> dict:
    attempts: list[dict] = []

    def _attempt(provider: str, op, policy: RuntimeReliabilityPolicy):
        out = execute_with_reliability(
            tool_name=f"web_search.{provider}",
            operation=op,
            policy=policy,
            fallback_suggestion="Try internal knowledge search or retry later.",
        )
        if out.get("success"):
            return out.get("data", {}).get("results", []), out
        attempts.append({"provider": provider, "success": False, "error_type": out.get("error_type", "unknown"), "message": out.get("message", "provider failed")})
        return [], out

    ddg_results, ddg_out = _attempt("duckduckgo", lambda: {"success": True, "data": {"results": _search_duckduckgo(inp.query, inp.max_results)}}, DUCKDUCKGO_POLICY)
    if ddg_results:
        attempts.append({"provider": "duckduckgo", "success": True, "result_count": len(ddg_results)})
        return success_response(tool_name="web_search", data={"query": inp.query, "intent": inp.intent, "results": ddg_results, "message": f"Found {len(ddg_results)} external research results."}, meta={"mode": "live", "input_schema": "ResearchQueryInput", "provider_used": "duckduckgo", "provider_attempts": attempts}).model_dump()

    serp_results, serp_out = _attempt("serpapi", lambda: {"success": True, "data": {"results": _search_serpapi(inp.query, inp.max_results)}}, SERPAPI_POLICY)
    if serp_results:
        attempts.append({"provider": "serpapi", "success": True, "result_count": len(serp_results)})
        return success_response(tool_name="web_search", data={"query": inp.query, "intent": inp.intent, "results": serp_results, "message": f"Found {len(serp_results)} external research results."}, meta={"mode": "live", "input_schema": "ResearchQueryInput", "provider_used": "serpapi", "provider_attempts": attempts}).model_dump()

    details = {"provider_attempts": attempts}
    if ddg_out.get("success") is False and ddg_out.get("error_type") == "circuit_open":
        details["duckduckgo_circuit_open"] = True
    if serp_out.get("success") is False and serp_out.get("error_type") == "circuit_open":
        details["serpapi_circuit_open"] = True

    return error_response(tool_name="web_search", error_type="all_sources_failed", message="All live web providers failed to return results.", recoverable=False, fallback_suggestion="Try internal knowledge search or retry later.", details=details).model_dump()


def web_search_impl(query: str, intent: str = "general", max_results: int = 3) -> dict:
    tool_name = "web_search"
    try:
        inp = ResearchQueryInput.model_validate({"query": query, "intent": intent, "max_results": max_results})
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    if not _is_live_web_search_enabled():
        return execute_with_reliability(
            tool_name=tool_name,
            operation=lambda: _web_search_offline_validated(inp),
            policy=RESEARCH_TOOL_POLICY,
            fallback_suggestion="Retry later or continue with internal knowledge search.",
        )

    return _web_search_live_validated(inp)


@tool
def web_search(query: str, intent: str = "general", max_results: int = 3) -> dict:
    """Perform bounded external-topic research with offline-first behavior.

    When to use:
    - User asks for external context not present in internal knowledge.
    - User asks for broad market, academic, or policy developments.

    When NOT to use:
    - User asks for internal process/policy answers already in knowledge tools.
    - User asks for operational actions (for example ticket creation).

    Inputs:
    - query: search query text.
    - intent: one of general, academic, policy.
    - max_results: bounded provider result count.

    Returns:
    - ToolSuccessResponse with deterministic offline results by default.
    - ToolErrorResponse on validation/provider failures.

    Notes:
    - Offline deterministic mode is the default unless ENABLE_LIVE_WEB_SEARCH=true.
    - In live mode, DuckDuckGo → SerpAPI fallback is handled in the tool layer.
    - Research Agent remains provider-agnostic.

    Example:
    - web_search(query="latest AI governance policy updates", intent="policy", max_results=3)
    """
    return web_search_impl(query, intent, max_results)
