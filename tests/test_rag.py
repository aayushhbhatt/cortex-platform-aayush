import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.hybrid_search import bm25_search, embed_chunks, reciprocal_rank_fusion, vector_search
from rag.pipeline import retrieve
from rag.query_understanding import QueryUnderstanding
from rag.reranker import rerank_results


def test_rag_retrieve_returns_ranked_results_with_metadata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local_hash")
    data_dir = tmp_path / "data"; data_dir.mkdir()
    (data_dir / "parental_leave_policy.txt").write_text("Access Level: public\nParental leave policy and benefits.", encoding="utf-8")
    results = retrieve("What is the parental leave policy?", top_k=3, data_dir=data_dir)
    assert isinstance(results, list) and results
    top = results[0]
    assert top.chunk_id and top.doc_id and top.title
    assert isinstance(top.score, float)
    assert top.retrieval_method
    assert isinstance(top.metadata, dict)
    assert any(k in top.metadata for k in ["query_understanding", "retrieval_debug", "retrieval_trace"])


def test_rag_hybrid_search_combines_bm25_vector_rrf(monkeypatch) -> None:
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local_hash")
    chunks = [
        {"chunk_id":"a","doc_id":"d1","title":"Parental Leave","content":"parental leave policy","chunk_index":0,"category":"hr_policy","access_level":"public","source":"s","content_hash":"h","metadata":{}},
        {"chunk_id":"b","doc_id":"d2","title":"Network","content":"vpn outage response","chunk_index":0,"category":"technical_policy","access_level":"general","source":"s","content_hash":"h2","metadata":{}},
    ]
    bm25 = bm25_search("parental leave", chunks, top_k=2)
    vector = vector_search("parental leave", embed_chunks(chunks), top_k=2)
    fused = reciprocal_rank_fusion([bm25, vector], top_k=2)
    assert vector and fused
    assert fused[0]["retrieval_method"] == "rrf"
    assert "fused_from" in fused[0]


def test_rag_reranker_boosts_query_understanding_hint() -> None:
    results = [
        {"chunk_id":"x1","doc_id":"other","title":"Other","content":"policy","chunk_index":0,"category":"general","access_level":"public","source":"s","content_hash":"h","metadata":{},"score":0.9,"retrieval_method":"rrf"},
        {"chunk_id":"x2","doc_id":"parental_leave_policy","title":"Parental","content":"leave","chunk_index":1,"category":"hr_policy","access_level":"public","source":"s","content_hash":"h2","metadata":{},"score":0.8,"retrieval_method":"rrf"},
    ]
    understanding = QueryUnderstanding(original_query="parental leave", normalized_query="parental leave", expanded_queries=["parental leave"], query_type="hr_policy", filters={"doc_id_hint": "parental_leave_policy", "category": "hr_policy"})
    reranked = rerank_results("parental leave", results, understanding, top_k=2)
    assert reranked[0]["doc_id"] == "parental_leave_policy"
    assert "rerank_debug" in reranked[0]["metadata"]
