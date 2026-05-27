from dataclasses import asdict
from langchain_core.tools import tool
from pydantic import ValidationError

from rag.pipeline import generate_grounded_answer, retrieve
from reliability.runtime import KNOWLEDGE_TOOL_POLICY, execute_with_reliability
from tools.error_contracts import error_response, success_response, validation_error_response
from tools.schemas import KnowledgeSearchInput


def _sanitize_expanded_queries(expanded_queries_override: list[str] | None) -> list[str] | None:
    if not expanded_queries_override:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in expanded_queries_override:
        text = str(item or "").strip()[:500]
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)
        if len(cleaned) >= 5:
            break
    return cleaned or None


def _retrieval_result_to_row(result, rank: int) -> dict:
    metadata = getattr(result, "metadata", {}) or {}
    rerank_debug = metadata.get("rerank_debug", {})
    return {
        "rank": rank,
        "chunk_id": getattr(result, "chunk_id", None),
        "doc_id": getattr(result, "doc_id", None),
        "title": getattr(result, "title", None),
        "category": getattr(result, "category", None),
        "access_level": getattr(result, "access_level", None),
        "score": getattr(result, "score", None),
        "retrieval_method": getattr(result, "retrieval_method", None),
        "fused_from": getattr(result, "fused_from", None),
        "ranks": getattr(result, "ranks", None),
        "source": getattr(result, "source", None),
        "chunk_index": getattr(result, "chunk_index", None),
        "content_preview": str(getattr(result, "content", ""))[:500],
        "section_title": metadata.get("section_title") or metadata.get("section", ""),
        "chunk_strategy": metadata.get("chunk_strategy") or ("section" if metadata.get("section_title") else ""),
        "embedding_provider": metadata.get("embedding_provider", ""),
        "rerank_debug": rerank_debug,
    }


def _search_knowledge_base_validated(inp: KnowledgeSearchInput) -> dict:
    results = retrieve(
        inp.query,
        top_k=inp.top_k,
        data_dir=inp.data_dir,
        user_tier=inp.user_tier,
        registry_db_path=inp.registry_db_path,
        use_pgvector=inp.use_pgvector,
        database_url=inp.database_url,
        expanded_queries_override=inp.expanded_queries_override,
    )
    answer_result = generate_grounded_answer(inp.query, results)

    if not results:
        return error_response(
            tool_name="search_knowledge_base",
            error_type="no_results",
            message="No matching company knowledge was found for this query.",
            recoverable=True,
            fallback_suggestion="Try a broader query, different wording, or web_search for external context.",
            details={"query": inp.query, "user_tier": inp.user_tier},
        ).model_dump()

    first_metadata = results[0].metadata if results else {}
    retrieval_trace = first_metadata.get("retrieval_trace", {})
    ranking_debug = first_metadata.get("ranking_debug", {})
    access_debug = first_metadata.get("access_debug", {})
    query_understanding = first_metadata.get("query_understanding", {})

    return success_response(
        tool_name="search_knowledge_base",
        data={
            "answer": answer_result.answer,
            "citations": [asdict(c) for c in answer_result.citations],
            "used_chunks": answer_result.used_chunks,
            "debug": {
                **answer_result.debug,
                "retrieval_trace": retrieval_trace,
                "ranking_debug": ranking_debug,
                "access_debug": access_debug,
                "query_understanding": query_understanding,
            },
            "result_count": len(results),
            "retrieval_results": [_retrieval_result_to_row(result, rank) for rank, result in enumerate(results, start=1)],
        },
        meta={"mode": "local_rag", "input_schema": "KnowledgeSearchInput"},
    ).model_dump()


def search_knowledge_base_impl(
    query: str,
    user_tier: str = "standard",
    top_k: int = 5,
    data_dir: str = "data",
    registry_db_path: str | None = None,
    use_pgvector: bool = False,
    database_url: str | None = None,
    expanded_queries_override: list[str] | None = None,
) -> dict:
    tool_name = "search_knowledge_base"
    try:
        inp = KnowledgeSearchInput.model_validate(
            {
                "query": query,
                "user_tier": user_tier,
                "top_k": top_k,
                "data_dir": data_dir,
                "registry_db_path": registry_db_path,
                "use_pgvector": use_pgvector,
                "database_url": database_url,
                "expanded_queries_override": _sanitize_expanded_queries(expanded_queries_override),
            }
        )
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    return execute_with_reliability(
        tool_name=tool_name,
        operation=lambda: _search_knowledge_base_validated(inp),
        policy=KNOWLEDGE_TOOL_POLICY,
        fallback_suggestion="Try a simpler query or verify the local data directory is available.",
    )


@tool
def search_knowledge_base(query: str, user_tier: str = "standard", top_k: int = 5, data_dir: str = "data") -> dict:
    """Search RBAC-aware internal company knowledge and return grounded answers with citations."""
    return search_knowledge_base_impl(query, user_tier, top_k, data_dir)
