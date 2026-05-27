from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.hybrid_search import bm25_search, embed_chunks, reciprocal_rank_fusion, vector_search
from rag.access_control import explain_access_filtering, filter_by_tier
from rag.ingest import chunk_documents, load_documents
from rag.record_manager import IngestionRegistry
from rag.query_understanding import understand_query
from rag.reranker import rerank_results
from rag.vector_store import PGVectorStore
from rag.generation import generate_answer
from rag.embeddings import LocalHashEmbeddingProvider, OpenAIEmbeddingProvider


@dataclass
class RetrievalResult:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    chunk_index: int
    category: str
    access_level: str
    source: str
    content_hash: str
    score: float
    retrieval_method: str
    fused_from: list[str] = field(default_factory=list)
    ranks: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def load_retrieval_chunks(
    data_dir: str | Path = "data",
    registry_db_path: str | Path | None = None,
) -> tuple[list[dict], dict]:
    registry_path = Path(registry_db_path) if registry_db_path is not None else None
    if registry_path is not None and registry_path.exists():
        registry = IngestionRegistry(registry_path)
        registry.initialize()
        chunks = registry.get_active_chunks()
        return chunks, {
            "chunk_source": "registry",
            "registry_db_path": str(registry_path),
            "data_dir": str(data_dir),
            "ingestion_mutated": False,
        }

    documents = load_documents(data_dir)
    chunks = chunk_documents(documents, chunk_strategy=os.getenv("RAG_CHUNK_STRATEGY", "section"))
    return chunks, {
        "chunk_source": "direct_data_fallback",
        "registry_db_path": str(registry_path) if registry_path is not None else None,
        "data_dir": str(data_dir),
        "ingestion_mutated": False,
    }


def retrieve(
    query: str,
    top_k: int = 5,
    data_dir: str | Path = "data",
    user_tier: str = "standard",
    registry_db_path: str | Path | None = None,
    use_pgvector: bool = True,
    database_url: str | None = None,
    expanded_queries_override: list[str] | None = None,
    rewrite_debug: dict | None = None,
) -> list[RetrievalResult]:
    understanding = understand_query(query)
    if not understanding.normalized_query:
        return []

    chunks, chunk_debug = load_retrieval_chunks(data_dir=data_dir, registry_db_path=registry_db_path)

    if not chunks:
        return []

    allowed_chunks = filter_by_tier(chunks, user_tier)
    if not allowed_chunks:
        return []

    access_debug = explain_access_filtering(len(chunks), len(allowed_chunks), user_tier)
    if expanded_queries_override:
        expanded_queries = [understanding.normalized_query] + list(expanded_queries_override)
        expanded_queries = list(dict.fromkeys([q for q in expanded_queries if q]))
    else:
        expanded_queries = understanding.expanded_queries
    allowed_levels = sorted({chunk.get("access_level", "general") for chunk in allowed_chunks})

    use_pgvector = True if use_pgvector is None else bool(use_pgvector)
    provider_note = None
    try:
        provider = OpenAIEmbeddingProvider()
        in_memory_provider = provider
    except Exception:
        provider = LocalHashEmbeddingProvider()
        in_memory_provider = provider
        provider_note = "OpenAI embedding provider unavailable; using local_hash fallback."

    try:
        embedded_chunks = embed_chunks(allowed_chunks, provider=in_memory_provider)
    except Exception:
        in_memory_provider = LocalHashEmbeddingProvider()
        embedded_chunks = embed_chunks(allowed_chunks, provider=in_memory_provider)
        provider_note = "OpenAI in-memory embedding failed; using local_hash fallback."
    vector_store = PGVectorStore(database_url=database_url) if use_pgvector else None

    result_lists: list[list[dict]] = []
    bm25_rows: list[dict] = []
    vector_rows: list[dict] = []
    bm25_total = 0
    vector_total = 0
    pgvector_used = False
    pgvector_fallback_used = False
    pgvector_fallback_reason = ""
    for variant in expanded_queries:
        bm25_results = bm25_search(variant, allowed_chunks, top_k=top_k)
        bm25_total += len(bm25_results)
        for rank, row in enumerate(bm25_results, start=1):
            if len(bm25_rows) < 20:
                bm25_rows.append(
                    {"query": variant, "rank": rank, "chunk_id": row.get("chunk_id"), "doc_id": row.get("doc_id"), "title": row.get("title"), "score": row.get("score")}
                )
        result_lists.append(bm25_results)

        vector_results: list[dict]
        if use_pgvector and vector_store is not None and vector_store.is_configured():
            try:
                vector_results = vector_store.search(variant, top_k=top_k, access_levels=allowed_levels)
                pgvector_used = True
            except Exception as exc:
                pgvector_fallback_used = True
                pgvector_fallback_reason = str(exc)
                vector_results = vector_search(variant, embedded_chunks, top_k=top_k)
        else:
            vector_results = vector_search(variant, embedded_chunks, top_k=top_k)
        for rank, row in enumerate(vector_results, start=1):
            if len(vector_rows) < 20:
                vector_rows.append(
                    {"query": variant, "rank": rank, "chunk_id": row.get("chunk_id"), "doc_id": row.get("doc_id"), "title": row.get("title"), "score": row.get("score"), "provider": "pgvector" if pgvector_used else in_memory_provider.name}
                )
        vector_total += len(vector_results)
        result_lists.append(vector_results)

    fused = reciprocal_rank_fusion(result_lists, k=60, top_k=max(top_k, 10))
    reranked = rerank_results(query=query, results=fused, understanding=understanding, top_k=top_k)
    retrieval_trace = {
        "query": query,
        "normalized_query": understanding.normalized_query,
        "expanded_queries_used": expanded_queries,
        "top_k": top_k,
        "rrf_k": 60,
        "candidate_pool_size": len(fused),
        "bm25_result_count": bm25_total,
        "vector_result_count": vector_total,
        "rrf_result_count": len(fused),
        "final_result_count": len(reranked),
        "use_pgvector": use_pgvector,
        "vector_provider": "pgvector" if pgvector_used else "in_memory",
        "embedding_provider": in_memory_provider.name if not pgvector_used else provider.name,
        "chunk_source": chunk_debug.get("chunk_source"),
        "registry_db_path": chunk_debug.get("registry_db_path"),
    }
    if provider_note:
        retrieval_trace["embedding_provider_note"] = provider_note
    if pgvector_fallback_used:
        retrieval_trace["pgvector_fallback_used"] = True
        retrieval_trace["pgvector_fallback_reason"] = pgvector_fallback_reason

    rrf_rows = [
        {"rank": rank, "chunk_id": row.get("chunk_id"), "doc_id": row.get("doc_id"), "title": row.get("title"), "score": row.get("score"), "fused_from": row.get("fused_from"), "ranks": row.get("ranks")}
        for rank, row in enumerate(fused[:20], start=1)
    ]
    final_rows = [
        {"rank": rank, "chunk_id": row.get("chunk_id"), "doc_id": row.get("doc_id"), "title": row.get("title"), "score": row.get("score"), "rerank_debug": (row.get("metadata") or {}).get("rerank_debug", {})}
        for rank, row in enumerate(reranked, start=1)
    ]
    ranking_debug = {"bm25_results": bm25_rows, "vector_results": vector_rows, "rrf_fused_results": rrf_rows, "final_selected_chunks": final_rows}

    output: list[RetrievalResult] = []
    for item in reranked:
        item_metadata = dict(item.get("metadata", {}))
        item_metadata["access_debug"] = access_debug
        item_metadata["retrieval_debug"] = chunk_debug
        item_metadata["query_understanding"] = {
            "query_type": understanding.query_type,
            "filters": understanding.filters,
            "expanded_queries": understanding.expanded_queries,
        }
        item_metadata["retrieval_trace"] = retrieval_trace
        item_metadata["ranking_debug"] = ranking_debug
        item_metadata["rerank_debug"] = item_metadata.get("rerank_debug", {})
        item_metadata["chunk_strategy"] = item_metadata.get("chunk_strategy") or item.get("chunk_strategy")
        item_metadata["section_title"] = item_metadata.get("section_title") or item.get("section_title")
        item_metadata["embedding_provider"] = item_metadata.get("embedding_provider") or ("pgvector" if pgvector_used else in_memory_provider.name)
        if rewrite_debug is not None:
            item_metadata["rewrite_debug"] = rewrite_debug
        payload = dict(item)
        payload["metadata"] = item_metadata
        output.append(RetrievalResult(**payload))
    return output


def build_context(results: list[RetrievalResult]) -> list[str]:
    return [f"[{result.chunk_id}] {result.title}: {result.content}" for result in results]


def answer_from_context(query: str, results: list[RetrievalResult]) -> str:
    answer_result = generate_answer(query, results)
    return answer_result.answer


def generate_grounded_answer(query: str, results: list[RetrievalResult]):
    return generate_answer(query, results)


if __name__ == "__main__":
    default_registry = Path("data/cortex_registry.sqlite")
    registry_arg = default_registry if default_registry.exists() else None
    results = retrieve("What is the parental leave policy?", top_k=5, registry_db_path=registry_arg)
    print(f"Retrieved {len(results)} chunks for query")
    for result in results:
        print(f"- {result.chunk_id} | {result.title} | {result.score:.4f}")
