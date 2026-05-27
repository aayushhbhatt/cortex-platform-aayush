

import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import retrieve
from rag.vector_store import PGVectorStore


def test_pgvector_store_initializes_without_database_url_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    store = PGVectorStore(database_url=None)
    assert store.is_configured() is False
    with pytest.raises(ValueError):
        store.initialize()


def test_vector_store_upsert_and_search_sql_contract_with_mock() -> None:
    store = PGVectorStore(database_url="postgresql://example", dimensions=8)
    sql, params = store._build_search_sql_and_params("[0.1,0.2]", top_k=3, access_levels=["public"])
    assert "access_level = ANY(%s)" in sql
    assert params[1] == ["public"]


def test_vector_store_search_handles_unavailable_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local_hash")
    data_dir = tmp_path / "data"; data_dir.mkdir()
    (data_dir / "faq.txt").write_text("Access Level: public\nParental leave policy includes paid leave.", encoding="utf-8")
    results = retrieve("parental leave", data_dir=data_dir, use_pgvector=True, database_url=None)
    assert isinstance(results, list)
