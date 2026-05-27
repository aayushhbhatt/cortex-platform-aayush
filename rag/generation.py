from __future__ import annotations

from dataclasses import dataclass, field
import re


@dataclass
class Citation:
    chunk_id: str
    doc_id: str
    title: str
    source: str
    access_level: str
    chunk_index: int


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    context_used: list[str]
    used_chunks: list[str]
    debug: dict = field(default_factory=dict)


def _format_context_item(result, content: str | None = None) -> str:
    body = result.content if content is None else content
    return f"[{result.chunk_id}] {result.title}\nsource: {result.source}\ncontent: {body}"


def build_bounded_context(results, max_chars: int = 4000) -> list[str]:
    if max_chars <= 0:
        return []

    context_items: list[str] = []
    used_chars = 0

    for result in results:
        full_item = _format_context_item(result)
        if used_chars + len(full_item) <= max_chars:
            context_items.append(full_item)
            used_chars += len(full_item)
            continue

        if context_items:
            break

        prefix = f"[{result.chunk_id}] {result.title}\nsource: {result.source}\ncontent: "
        remaining = max_chars - len(prefix)
        if remaining <= 0:
            break
        truncated_content = result.content[:remaining].rstrip()
        context_items.append(_format_context_item(result, truncated_content))
        break

    return context_items


def build_citations(results) -> list[Citation]:
    citations: list[Citation] = []
    seen_chunk_ids: set[str] = set()
    for result in results:
        if result.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(result.chunk_id)
        citations.append(
            Citation(
                chunk_id=result.chunk_id,
                doc_id=result.doc_id,
                title=result.title,
                source=result.source,
                access_level=result.access_level,
                chunk_index=result.chunk_index,
            )
        )
    return citations


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentence = parts[0] if parts and parts[0] else text.strip()
    return sentence[:220].strip()


def extractive_answer(query: str, results) -> str:
    del query
    if not results:
        return "I could not find relevant company knowledge for this query."

    top_results = list(results[:3])
    titles = ", ".join(result.title for result in top_results)
    snippets = " ".join(f"{result.title}: {_first_sentence(result.content)}" for result in top_results)
    return f"Based on the retrieved company documents in company knowledge ({titles}), here is what I found: {snippets}"


def generate_answer(query: str, results, max_context_chars: int = 4000) -> AnswerResult:
    if not results:
        return AnswerResult(
            answer="I could not find relevant company knowledge for this query.",
            citations=[],
            context_used=[],
            used_chunks=[],
            debug={"mode": "extractive_fallback", "result_count": 0},
        )

    bounded_context = build_bounded_context(results, max_chars=max_context_chars)
    used_results = list(results[: len(bounded_context)])
    citations = build_citations(used_results)
    answer = extractive_answer(query, used_results)

    retrieval_methods = []
    for item in used_results:
        if item.retrieval_method not in retrieval_methods:
            retrieval_methods.append(item.retrieval_method)

    debug = {
        "mode": "extractive_fallback",
        "result_count": len(results),
        "used_chunk_count": len(used_results),
        "top_score": results[0].score if results else None,
        "retrieval_methods": retrieval_methods,
    }

    query_understanding = (used_results[0].metadata or {}).get("query_understanding", {}) if used_results else {}
    if "query_type" in query_understanding:
        debug["query_type"] = query_understanding["query_type"]

    return AnswerResult(
        answer=answer,
        citations=citations,
        context_used=bounded_context,
        used_chunks=[result.chunk_id for result in used_results],
        debug=debug,
    )
