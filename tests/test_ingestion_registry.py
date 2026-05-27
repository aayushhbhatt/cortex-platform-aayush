import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))



from rag.ingest import ingest_directory
from rag.pipeline import RetrievalResult, retrieve
from rag.record_manager import IngestionRegistry


def test_registry_initializes_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "registry.sqlite"
    registry = IngestionRegistry(db_path)
    registry.initialize()

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    assert {"documents", "chunks", "ingestion_runs"}.issubset(tables)


def test_ingest_directory_creates_document_and_chunks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("Access Level: public\nParental leave policy details.", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"

    summary = ingest_directory(data_dir=data_dir, db_path=db_path)
    registry = IngestionRegistry(db_path)
    chunks = registry.get_active_chunks()

    assert summary["documents_created"] == 1
    assert summary["chunks_active"] > 0
    assert chunks


def test_ingest_directory_unchanged_document_skips_new_version(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("Access Level: public\nSame content always.", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"

    first = ingest_directory(data_dir=data_dir, db_path=db_path)
    second = ingest_directory(data_dir=data_dir, db_path=db_path)
    registry = IngestionRegistry(db_path)
    latest = registry.get_latest_document("doc")
    active_chunks = registry.get_active_chunks()

    assert first["documents_created"] == 1
    assert second["documents_unchanged"] == 1
    assert latest is not None and latest["version"] == 1
    assert len(active_chunks) == first["chunks_active"]


def test_ingest_directory_changed_document_creates_new_version(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    file_path = data_dir / "doc.txt"
    file_path.write_text("Access Level: public\noriginal content", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"

    ingest_directory(data_dir=data_dir, db_path=db_path)
    file_path.write_text("Access Level: public\nupdated content with more words", encoding="utf-8")
    second = ingest_directory(data_dir=data_dir, db_path=db_path)

    registry = IngestionRegistry(db_path)
    latest = registry.get_latest_document("doc")
    active = registry.get_active_document("doc")
    active_chunks = registry.get_active_chunks()

    assert second["documents_updated"] == 1
    assert latest is not None and latest["version"] == 2
    assert active is not None and active["version"] == 2
    assert all("::v2::" in chunk["chunk_id"] for chunk in active_chunks)


def test_registry_active_chunks_match_retrieval_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("Access Level: public\nParental leave policy details.", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    ingest_directory(data_dir=data_dir, db_path=db_path)

    registry = IngestionRegistry(db_path)
    chunk = registry.get_active_chunks()[0]
    required = {
        "chunk_id",
        "doc_id",
        "title",
        "content",
        "chunk_index",
        "category",
        "access_level",
        "source",
        "content_hash",
        "metadata",
    }
    assert required.issubset(chunk.keys())


def test_retrieve_can_use_registry_active_chunks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("Access Level: public\nParental leave policy includes paid leave.", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"
    ingest_directory(data_dir=data_dir, db_path=db_path)

    results = retrieve("parental leave", registry_db_path=db_path, user_tier="standard")

    assert results
    assert isinstance(results[0], RetrievalResult)


def test_retrieve_registry_respects_rbac(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "public.txt").write_text("Access Level: public\nGeneral FAQ and onboarding.", encoding="utf-8")
    (data_dir / "restricted.txt").write_text(
        "CLASSIFICATION: RESTRICTED\nAURORA-IPO-TIMELINE AURORA-IPO-TIMELINE", encoding="utf-8"
    )
    db_path = tmp_path / "registry.sqlite"
    ingest_directory(data_dir=data_dir, db_path=db_path)

    standard_results = retrieve("AURORA-IPO-TIMELINE", registry_db_path=db_path, user_tier="standard")
    exec_results = retrieve("AURORA-IPO-TIMELINE", registry_db_path=db_path, user_tier="exec")

    assert all(item.access_level != "executive" for item in standard_results)
    assert any(item.access_level == "executive" for item in exec_results)


def test_ingestion_explicitly_persists_active_chunks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "doc.txt").write_text("Access Level: public\nParental leave policy details.", encoding="utf-8")
    db_path = tmp_path / "registry.sqlite"

    ingest_directory(data_dir=data_dir, db_path=db_path)

    assert db_path.exists()
    registry = IngestionRegistry(db_path)
    chunks = registry.get_active_chunks()
    assert chunks
