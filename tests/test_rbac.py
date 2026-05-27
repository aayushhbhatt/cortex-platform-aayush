import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.access_control import allowed_access_levels, filter_by_tier
from rag.pipeline import retrieve


def test_allowed_access_levels_differ_by_tier() -> None:
    standard = allowed_access_levels("standard")
    manager = allowed_access_levels("manager")
    exec_levels = allowed_access_levels("exec")
    assert standard != manager != exec_levels
    assert "executive" in exec_levels
    assert "executive" not in standard


def test_filter_by_tier_blocks_restricted_chunks() -> None:
    chunks = [{"chunk_id": x, "access_level": x} for x in ["public", "general", "manager", "executive"]]
    assert [c["chunk_id"] for c in filter_by_tier(chunks, "standard")] == ["public", "general"]
    assert [c["chunk_id"] for c in filter_by_tier(chunks, "exec")] == ["public", "general", "manager", "executive"]


def test_retrieve_respects_rbac(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"; data_dir.mkdir()
    (data_dir / "std.txt").write_text("Access Level: public\nVPN support basics", encoding="utf-8")
    (data_dir / "exec.txt").write_text("CLASSIFICATION: RESTRICTED\nAURORA-IPO-TIMELINE", encoding="utf-8")
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "local_hash")
    assert all(r.access_level != "executive" for r in retrieve("AURORA-IPO-TIMELINE", user_tier="standard", data_dir=data_dir))
    assert any(r.access_level == "executive" for r in retrieve("AURORA-IPO-TIMELINE", user_tier="exec", data_dir=data_dir))
