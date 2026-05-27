from __future__ import annotations

from rag.query_understanding import QueryUnderstanding, normalize_query


def _token_set(text: str) -> set[str]:
    normalized = normalize_query(text or "")
    return {token for token in normalized.split() if token}


def _title_overlap_score(query: str, title: str) -> tuple[float, str | None]:
    query_tokens = _token_set(query)
    title_tokens = _token_set(title)
    if not query_tokens or not title_tokens:
        return 0.0, None
    overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
    score = min(0.10, overlap * 0.10)
    return score, "title_overlap" if score > 0 else None


def _contains_exact_phrase(query: str, title: str, content: str) -> bool:
    normalized_query = normalize_query(query)
    if not normalized_query:
        return False
    return normalized_query in normalize_query(title) or normalized_query in normalize_query(content)


def _section_match(section_hint: str | None, metadata: dict, title: str, content: str) -> bool:
    if not section_hint:
        return False
    hint = normalize_query(section_hint)
    section_title = normalize_query(str(metadata.get("section_title", "")))
    return hint in section_title or hint in normalize_query(title) or hint in normalize_query(content)


def rerank_results(
    query: str,
    results: list[dict],
    understanding: QueryUnderstanding,
    top_k: int = 5,
) -> list[dict]:
    reranked: list[dict] = []
    filters = understanding.filters or {}

    for item in results:
        result = dict(item)
        metadata = dict(result.get("metadata", {}) or {})
        base_score = float(result.get("score", 0.0) or 0.0)
        boost = 0.0
        reasons: list[str] = []

        doc_id = str(result.get("doc_id", ""))
        title = str(result.get("title", ""))
        content = str(result.get("content", ""))
        category = str(result.get("category", ""))
        access_level = str(result.get("access_level", ""))
        chunk_strategy = str(metadata.get("chunk_strategy", ""))
        section_title = str(metadata.get("section_title", ""))

        if filters.get("doc_id_hint") and filters["doc_id_hint"] == doc_id:
            boost += 0.30
            reasons.append("doc_id_hint_match")
        if filters.get("category") and filters["category"] == category:
            boost += 0.12
            reasons.append("category_match")
        if filters.get("access_level_hint") and filters["access_level_hint"] == access_level:
            boost += 0.04
            reasons.append("access_level_hint_match")
        if _section_match(filters.get("section_hint"), metadata, title, content):
            boost += 0.10
            reasons.append("section_hint_match")

        title_boost, title_reason = _title_overlap_score(query, title)
        if title_boost > 0:
            boost += title_boost
            if title_reason:
                reasons.append(title_reason)

        if _contains_exact_phrase(query, title, content):
            boost += 0.08
            reasons.append("exact_phrase_overlap")

        query_terms = _token_set(query)
        section_terms = _token_set(section_title)
        if chunk_strategy == "section" and filters.get("section_hint"):
            if normalize_query(filters["section_hint"]) in normalize_query(section_title):
                boost += 0.08
                reasons.append("section_title_match")
        if query_terms and section_terms and (query_terms & section_terms):
            boost += 0.04
            reasons.append("section_title_overlap")

        final_score = base_score + boost
        rerank_debug = {
            "base_rrf_score": base_score,
            "boost_score": boost,
            "final_score": final_score,
            "boost_reasons": reasons,
        }
        metadata["rerank_debug"] = rerank_debug
        result["metadata"] = metadata
        result["score"] = final_score
        reranked.append(result)

    reranked.sort(key=lambda row: row.get("score", 0.0), reverse=True)
    return reranked[:top_k]
