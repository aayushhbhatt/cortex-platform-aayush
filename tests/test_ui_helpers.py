import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import RetrievalResult
from ui.helpers import build_execution_trace_rows, get_first_result_metadata, tool_results_to_rows


def test_build_run_summary_handles_final_state():
    final_state = {
        "agent_used": "knowledge", "intent": "knowledge", "cost_usd": 0.01,
        "tool_results": [{"tool_name": "search_knowledge_base", "success": True}],
        "memory_context_text": "Recent conversation: ...",
        "trace": [{"event": "query_received"}],
    }
    rows = build_execution_trace_rows(final_state)
    assert isinstance(rows, list)
    assert rows


def test_rag_ui_helpers_extract_retrieval_debug():
    rr = RetrievalResult("c","d","t","body",0,"general","public","src","h",0.5,"rrf", metadata={"retrieval_trace": [], "query_understanding": {}})
    meta = get_first_result_metadata([rr])
    assert "retrieval_trace" in meta or "query_understanding" in meta


def test_tool_results_to_rows_handles_success_and_error():
    rows = tool_results_to_rows([
        {"tool_name": "x", "success": True, "data": {"ok": True}},
        {"tool_name": "y", "success": False, "error_type": "validation_error", "message": "bad"},
    ])
    assert len(rows) == 2
