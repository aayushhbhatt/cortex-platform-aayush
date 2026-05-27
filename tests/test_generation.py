import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.generation import build_bounded_context, build_citations, generate_answer
from rag.pipeline import RetrievalResult, answer_from_context


def _result(chunk_id: str, content: str, metadata: dict | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id="doc-1",
        title=f"Title {chunk_id}",
        content=content,
        chunk_index=0,
        category="hr_policy",
        access_level="public",
        source="data/doc.txt",
        content_hash="hash",
        score=0.9,
        retrieval_method="rrf",
        metadata=metadata or {},
    )


def test_generate_answer_no_results() -> None:
    result = generate_answer("query", [])
    assert "could not find relevant company knowledge" in result.answer
    assert result.citations == []
    assert result.context_used == []
    assert result.used_chunks == []


def test_build_bounded_context_limits_size() -> None:
    results = [_result("a", "x" * 500), _result("b", "y" * 500)]
    context = build_bounded_context(results, max_chars=200)
    assert context
    assert sum(len(item) for item in context) <= 220


def test_build_citations_deduplicates_chunks() -> None:
    results = [_result("dup", "one"), _result("dup", "two"), _result("other", "three")]
    citations = build_citations(results)
    assert [c.chunk_id for c in citations] == ["dup", "other"]


def test_generate_answer_returns_citations_and_used_chunks() -> None:
    results = [_result("a", "First sentence. Extra details."), _result("b", "Second sentence.")]
    answer = generate_answer("leave", results)
    assert answer.answer
    assert answer.citations
    assert answer.used_chunks == ["a", "b"]
    assert answer.debug["mode"] == "extractive_fallback"


def test_generate_answer_includes_query_type_debug() -> None:
    results = [_result("a", "Policy text.", metadata={"query_understanding": {"query_type": "hr_policy"}})]
    answer = generate_answer("query", results)
    assert answer.debug["query_type"] == "hr_policy"


def test_answer_from_context_backwards_compatible() -> None:
    results = [_result("a", "Policy text.")]
    answer = answer_from_context("query", results)
    assert isinstance(answer, str)
