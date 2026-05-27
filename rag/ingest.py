from __future__ import annotations

import hashlib
import re
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.record_manager import IngestionRegistry
from rag.vector_store import PGVectorStore
from rag.embeddings import get_default_embedding_provider


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_access_level(text: str) -> str:
    lowered = text.lower()
    if any(
        marker in lowered
        for marker in (
            "executive level access only",
            "strictly confidential",
            "classification: restricted",
        )
    ):
        return "executive"
    if "access level: manager" in lowered:
        return "manager"
    if "access level: public" in lowered:
        return "public"
    if "access level: general" in lowered:
        return "general"
    return "general"


def infer_category(filename: str, text: str) -> str:
    haystack = f"{filename} {text}".lower()

    executive_terms = (
        r"\bexecutive strategy\b",
        r"\bboard\b",
        r"\bipo\b",
        r"\bc-suite\b",
        r"\barr\b",
        r"\bfy2025 executive\b",
        r"\brestricted\b",
        r"\bstrictly confidential\b",
    )
    technical_terms = (
        r"\bdata security\b",
        r"\bacceptable use\b",
        r"\bsoftware request\b",
        r"\binformation security\b",
        r"\bit policy\b",
        r"\bvpn\b",
        r"\bencryption\b",
        r"\bincident reporting\b",
        r"\bapproved software\b",
        r"\bservice portal\b",
    )
    hr_terms = (
        r"\bparental leave\b",
        r"\bleave policy\b",
        r"\bcompensation\b",
        r"\bbenefits\b",
        r"\bperformance review\b",
        r"\bcode of conduct\b",
        r"\bremote work\b",
        r"\bhybrid work\b",
        r"\bmanager handbook\b",
        r"\bpeople operations\b",
        r"\bhr policy\b",
        r"\bemployee handbook\b",
    )

    if any(re.search(pattern, haystack) for pattern in executive_terms):
        return "executive_strategy"
    if any(re.search(pattern, haystack) for pattern in technical_terms):
        return "technical_policy"
    if any(re.search(pattern, haystack) for pattern in hr_terms):
        return "hr_policy"
    return "general"


def infer_title(path: str | Path, text: str) -> str:
    file_path = Path(path)
    skip_prefixes = (
        "access level:",
        "document number:",
        "policy number:",
        "effective date:",
        "last reviewed:",
        "document owner:",
        "classification:",
        "prepared by:",
        "distribution:",
    )

    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line.strip())
        if not cleaned:
            continue
        upper = cleaned.upper()
        lower = cleaned.lower()
        if upper == "MERIDIAN TECHNOLOGIES":
            continue
        if lower.startswith(skip_prefixes):
            continue
        decorative_ratio = sum(1 for ch in cleaned if not ch.isalnum() and not ch.isspace()) / len(cleaned)
        if decorative_ratio >= 0.7:
            continue
        return cleaned
    return file_path.stem.replace("_", " ").replace("-", " ").title()


def normalize_txt_document(path: str | Path) -> dict:
    file_path = Path(path)
    content = file_path.read_text(encoding="utf-8")

    return {
        "doc_id": file_path.stem,
        "title": infer_title(file_path, content),
        "content": content,
        "category": infer_category(file_path.name, content),
        "access_level": parse_access_level(content),
        "source": str(file_path),
        "content_hash": compute_content_hash(content),
        "metadata": {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "size_chars": len(content),
        },
    }


def load_documents(data_dir: str | Path = "data") -> list[dict]:
    root = Path(data_dir)
    if not root.exists():
        raise ValueError(f"Data directory not found: {root}")

    txt_files = sorted(root.rglob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in data directory: {root}")

    docs = [normalize_txt_document(path) for path in txt_files]
    return sorted(docs, key=lambda d: d["doc_id"])


def detect_section_boundaries(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    common = {"eligibility","how to apply","benefits continuation","responsibilities","security requirements","approved software","request process","scope","purpose","policy","procedure","exceptions","contacts"}

    def is_heading(line: str) -> bool:
        stripped = line.strip()
        if not stripped or len(stripped) > 90:
            return False
        lower = stripped.lower().rstrip(":")
        if lower in common:
            return True
        if re.match(r"^\d+\.\s+[A-Za-z]", stripped):
            return True
        if stripped.endswith(":") and 2 <= len(stripped) <= 80:
            return True
        letters = [c for c in stripped if c.isalpha()]
        return bool(letters) and stripped == stripped.upper()

    heads=[i for i,l in enumerate(lines) if is_heading(l)]
    if not heads:
        return []
    sections=[]
    for idx,start in enumerate(heads):
        end=heads[idx+1] if idx+1<len(heads) else len(lines)
        title=lines[start].strip().rstrip(':')
        body="\n".join(lines[start+1:end]).strip()
        sections.append((title,body))
    return [(t,b) for t,b in sections if b]


def chunk_document_by_sections(document: dict, max_section_words: int = 450, overlap_words: int = 60) -> list[dict]:
    sections = detect_section_boundaries(document.get("content", ""))
    if not sections:
        return chunk_document(document)
    chunks=[]
    stride=max_section_words-overlap_words
    for s_idx,(title,body) in enumerate(sections):
        words=body.split()
        if not words:
            continue
        for p_idx,start in enumerate(range(0,len(words),stride)):
            part=words[start:start+max_section_words]
            if not part:
                continue
            content=" ".join(part)
            meta=dict(document.get("metadata",{}))
            meta.update({"chunk_strategy":"section","section_title":title,"section_index":s_idx,"section_part_index":p_idx,"filename":meta.get("filename")})
            chunks.append({"chunk_id":f"{document['doc_id']}::section_{s_idx:03d}::part_{p_idx:03d}","doc_id":document["doc_id"],"title":document["title"],"content":content,"chunk_index":len(chunks),"category":document["category"],"access_level":document["access_level"],"source":document["source"],"content_hash":compute_content_hash(content),"metadata":meta})
            if start+max_section_words>=len(words):
                break
    return chunks


def chunk_document(document: dict, chunk_size: int = 700, chunk_overlap: int = 120) -> list[dict]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    words = document["content"].split()
    if not words:
        return []

    stride = chunk_size - chunk_overlap
    chunks: list[dict] = []
    for index, start in enumerate(range(0, len(words), stride)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            continue
        chunk_content = " ".join(chunk_words)
        chunks.append(
            {
                "chunk_id": f"{document['doc_id']}::chunk_{index:03d}",
                "doc_id": document["doc_id"],
                "title": document["title"],
                "content": chunk_content,
                "chunk_index": index,
                "category": document["category"],
                "access_level": document["access_level"],
                "source": document["source"],
                "content_hash": compute_content_hash(chunk_content),
                "metadata": {**dict(document.get("metadata", {})), "chunk_strategy": "word_window"},
            }
        )
        if start + chunk_size >= len(words):
            break

    return chunks


def chunk_documents(documents: list[dict], chunk_size: int = 700, chunk_overlap: int = 120, chunk_strategy: str = "word_window") -> list[dict]:
    chunks: list[dict] = []
    for doc in documents:
        if chunk_strategy == "section":
            chunks.extend(chunk_document_by_sections(doc, max_section_words=min(chunk_size, 450), overlap_words=min(chunk_overlap, 60)))
        else:
            chunks.extend(chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
    return sorted(chunks, key=lambda c: c["chunk_id"])


def ingest_directory(
    data_dir: str | Path = "data",
    db_path: str | Path = "data/cortex_registry.sqlite",
    chunk_size: int = 700,
    chunk_overlap: int = 120,
    sync_vector_store: bool = False,
    database_url: str | None = None,
    chunk_strategy: str = "word_window",
) -> dict:
    documents = load_documents(data_dir)
    registry = IngestionRegistry(db_path)
    registry.initialize()

    created = 0
    updated = 0
    unchanged = 0

    for document in documents:
        chunks = chunk_document_by_sections(document, max_section_words=min(chunk_size,450), overlap_words=min(chunk_overlap,60)) if chunk_strategy == "section" else chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        result = registry.upsert_document_version(document, chunks)
        if result["status"] == "created":
            created += 1
        elif result["status"] == "updated":
            updated += 1
        else:
            unchanged += 1

    active_chunks = registry.get_active_chunks()
    registry.record_ingestion_run(
        data_dir=str(data_dir),
        documents_seen=len(documents),
        documents_changed=created + updated,
        chunks_active=len(active_chunks),
    )

    summary = {
        "documents_seen": len(documents),
        "documents_created": created,
        "documents_updated": updated,
        "documents_unchanged": unchanged,
        "chunks_active": len(active_chunks),
        "db_path": str(Path(db_path)),
        "chunk_strategy": chunk_strategy,
    }

    if sync_vector_store:
        provider = get_default_embedding_provider()
        vector_store = PGVectorStore(database_url=database_url, embedding_provider=provider)
        upserted = vector_store.upsert_chunks(active_chunks)
        deactivated = vector_store.deactivate_missing_active_chunks({chunk["chunk_id"] for chunk in active_chunks})
        summary["vector_chunks_upserted"] = upserted
        summary["vector_chunks_deactivated"] = deactivated
        summary["embedding_provider"] = provider.name
        summary["embedding_dimensions"] = provider.dimensions

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest TXT docs into SQLite registry (and optionally PGVector).")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--registry-db-path", "--db-path", dest="registry_db_path", default="data/cortex_registry.sqlite")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--sync-vector-store", action="store_true")
    parser.add_argument("--chunk-strategy", choices=["word_window", "section"], default="word_window")
    parser.add_argument("--database-url", default="postgresql://postgres:cortex123@localhost:5432/cortex")
    args = parser.parse_args()

    summary = ingest_directory(
        data_dir=args.data_dir,
        db_path=args.registry_db_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        sync_vector_store=args.sync_vector_store,
        database_url=args.database_url,
        chunk_strategy=args.chunk_strategy,
    )
    print(f"Documents seen: {summary['documents_seen']}")
    print(f"Created: {summary['documents_created']}")
    print(f"Updated: {summary['documents_updated']}")
    print(f"Unchanged: {summary['documents_unchanged']}")
    print(f"Active chunks: {summary['chunks_active']}")
    print(f"Registry: {summary['db_path']}")
    print(f"Vector sync: {'enabled' if args.sync_vector_store else 'disabled'}")
