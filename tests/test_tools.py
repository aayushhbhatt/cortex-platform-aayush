import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.action_tools import create_support_ticket_impl
from tools.knowledge_tools import search_knowledge_base_impl
from tools.research_tools import web_search_impl


def test_knowledge_tool_success_contract(monkeypatch):
    monkeypatch.setattr("tools.knowledge_tools.retrieve", lambda *args, **kwargs: [])
    out = search_knowledge_base_impl(query="parental leave", user_tier="standard", top_k=3)
    assert "success" in out and "tool_name" in out
    if out["success"]:
        assert set(["data", "meta"]).issubset(out.keys())
    else:
        assert out["error_type"] in {"no_results", "validation_error", "unknown"}


def test_action_tool_validation_error_contract():
    out = create_support_ticket_impl("bad", "short", "invalid")
    assert out["success"] is False
    assert out["error_type"] == "validation_error"
    assert out["recoverable"] is False
    assert out.get("details") is not None


def test_research_tool_offline_or_fallback_contract():
    out = web_search_impl("latest SOC2 best practices", intent="general", max_results=3)
    assert "success" in out and "tool_name" in out
    if out["success"]:
        assert "data" in out and "meta" in out
    else:
        assert "error_type" in out and "message" in out
