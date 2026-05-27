import os
from agents.supervisor import CortexState
from dataclasses import asdict
from dotenv import load_dotenv
from memory.context import MemoryContext, build_memory_aware_query, finalize_agent_memory, load_memory_context
from memory.session_memory import create_session_memory
from reliability.cost_tracker import CostTracker, estimate_generation_cost, estimate_memory_cost, estimate_retrieval_cost
from reliability.errors import BudgetExceededError, classify_error
from tools.knowledge_tools import search_knowledge_base_impl
from rag.query_rewriter import rewrite_query_for_retrieval

load_dotenv()

def knowledge_agent(state: CortexState) -> dict:
    query = state["query"]
    user_id = state.get("user_id", "demo-user")
    session_id = state.get("session_id") or "session_local"
    user_tier = state.get("user_tier", "standard")
    session_memory = create_session_memory(prefer_redis=True)
    if state.get("memory_context"):
        memory_context = MemoryContext(**state["memory_context"])
    else:
        memory_context = load_memory_context(user_id=user_id, session_id=session_id, session_memory=session_memory)

    original_query = query
    deterministic_effective_query = build_memory_aware_query(query, memory_context)
    rewrite_result = rewrite_query_for_retrieval(
        query=deterministic_effective_query,
        memory_context_text=memory_context.context_text,
        user_tier=user_tier,
    )
    final_effective_query = rewrite_result.get("query", deterministic_effective_query)
    expanded_queries_override = rewrite_result.get("expanded_queries")
    data_dir = os.getenv("RAG_DATA_DIR", "data")
    registry_db_path = os.getenv("RAG_REGISTRY_DB_PATH", "data/cortex_registry.sqlite")
    use_pgvector = True
    database_url = state.get("database_url") or os.getenv("DATABASE_URL")

    tool_result = search_knowledge_base_impl(
        query=final_effective_query,
        user_tier=user_tier,
        top_k=5,
        data_dir=data_dir,
        registry_db_path=registry_db_path,
        use_pgvector=use_pgvector,
        database_url=database_url,
        expanded_queries_override=expanded_queries_override,
    )

    if tool_result.get("success"):
        tool_data = tool_result.get("data", {})
        response = tool_data.get("answer", "")
        citations = tool_data.get("citations", [])
        used_chunks = tool_data.get("used_chunks", [])
        debug = dict(tool_data.get("debug") or {})
        retrieval_results = tool_data.get("retrieval_results", [])
    else:
        response = tool_result.get("message") or "I could not load the company knowledge base."
        citations = []
        used_chunks = []
        debug = {"tool_error": tool_result}
        retrieval_results = []

    finalizer = finalize_agent_memory(user_id=user_id, session_id=session_id, query=query, response=response, session_memory=session_memory)
    messages = finalizer["memory_messages"]

    tracker = CostTracker()
    context_proxy = "\n".join(used_chunks)
    tracker.add_cost("retrieval", "retrieve", estimate_retrieval_cost(len(used_chunks)), {"result_count": len(used_chunks)})
    tracker.add_cost(
        "generation",
        "generate_grounded_answer",
        estimate_generation_cost(len(context_proxy), len(response)),
        {"context_chars": len(context_proxy), "output_chars": len(response)},
    )
    tracker.add_cost("memory", "session_messages", estimate_memory_cost(len(messages)), {"message_count": len(messages)})

    debug["original_query"] = original_query
    debug["deterministic_effective_query"] = deterministic_effective_query
    debug["final_effective_query"] = final_effective_query
    debug["memory_rewrite_applied"] = deterministic_effective_query != original_query
    debug["llm_query_rewrite_debug"] = rewrite_result.get("debug")
    debug["expanded_queries_used"] = expanded_queries_override
    debug["rag_config"] = {
        "registry_db_path": registry_db_path,
        "use_pgvector": use_pgvector,
        "database_url_configured": bool(database_url),
    }
    debug["retrieval_config"] = {
        "registry_db_path": registry_db_path,
        "use_pgvector": use_pgvector,
        "database_url_configured": bool(database_url),
        "expanded_queries_passed": expanded_queries_override or [],
    }
    debug["retrieval_results_count"] = len(retrieval_results)
    debug["memory_context"] = memory_context.debug
    debug["memory_finalizer"] = finalizer["memory_debug"]
    debug["memory_context_text_available"] = bool(memory_context.context_text)
    debug["cost"] = tracker.to_dict()
    try:
        tracker.check_budget()
    except BudgetExceededError as exc:
        err = classify_error(exc)
        debug["error"] = {
            "error_type": err.error_type,
            "message": err.message,
            "retryable": err.retryable,
            "details": err.details,
        }
        response = "I could not complete this request because the estimated cost exceeded the allowed budget."

    return {
        "context": [],
        "response": response,
        "agent_used": "knowledge",
        "citations": citations,
        "used_chunks": used_chunks,
        "debug": debug,
        "cost_usd": tracker.total_cost(),
        "memory_messages": messages,
        "memory_context": asdict(memory_context),
        "memory_context_text": memory_context.context_text,
        "tool_results": [tool_result],
        "retrieval_results": retrieval_results,
    }
