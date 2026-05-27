from __future__ import annotations
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.evaluate import EvaluationCase, evaluate_retrieval, ndcg_at_k, precision_at_k, reciprocal_rank


def _r(doc_id: str, chunk_id: str) -> SimpleNamespace:
    return SimpleNamespace(doc_id=doc_id, chunk_id=chunk_id)


def test_metric_functions_precision_mrr_ndcg() -> None:
    case = EvaluationCase("q", "standard", ["doc1"], ["doc1"], "")
    results = [_r("doc1", "doc1::chunk_0"), _r("doc2", "doc2::chunk_0")]
    assert precision_at_k(results, case, k=2) >= 0
    assert reciprocal_rank(results, case) > 0
    assert ndcg_at_k(results, case, k=2) >= 0


def test_evaluate_retrieval_returns_summary_shape(tmp_path: Path, monkeypatch) -> None:
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text('[{"query":"q","user_tier":"standard","relevant_doc_ids":["doc1"],"relevant_chunk_id_prefixes":["doc1"],"notes":"n"}]', encoding="utf-8")
    monkeypatch.setattr("evaluation.evaluate.retrieve", lambda *args, **kwargs: [_r("doc1", "doc1::chunk_0")])
    out = evaluate_retrieval(dataset_path=dataset_path, top_k=3)
    assert set(["case_count", "precision_at_k", "mrr", "ndcg_at_k", "cases"]).issubset(out.keys())


def test_evaluation_config_accepts_pgvector_flags(tmp_path: Path, monkeypatch) -> None:
    seen = {}
    dataset_path = tmp_path / "golden.json"
    dataset_path.write_text('[{"query":"q","user_tier":"standard","relevant_doc_ids":["doc1"],"relevant_chunk_id_prefixes":["doc1"],"notes":"n"}]', encoding="utf-8")
    def fake_retrieve(*args, **kwargs):
        seen.update(kwargs); return [_r("doc1", "doc1::chunk_0")]
    monkeypatch.setattr("evaluation.evaluate.retrieve", fake_retrieve)
    out = evaluate_retrieval(dataset_path=dataset_path, top_k=3, use_pgvector=True, database_url="dummy")
    assert seen.get("use_pgvector") is True
    assert out.get("retrieval_config") is not None
