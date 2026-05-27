import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import supervisor as supervisor_module
from agents.supervisor import invoke_with_trace
from observability.logging import log_query_event
from observability.tracing import build_query_trace_event


def test_log_query_event_writes_jsonl(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setenv("CORTEX_QUERY_LOG_PATH", str(log_path))

    event = log_query_event({"event_type": "query_trace", "request_id": "req_test"})

    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["request_id"] == "req_test"
    assert "timestamp_utc" in payload
    assert event["request_id"] == "req_test"


def test_build_query_trace_event_success() -> None:
    initial_state = {"request_id": "req_1", "user_id": "u1", "session_id": "s1", "query": "q"}
    final_state = {"intent": "knowledge", "agent_used": "knowledge", "cost_usd": 0.12, "tool_results": [{"tool_name": "kb", "success": True}]}

    event = build_query_trace_event(initial_state=initial_state, final_state=final_state, latency_ms=12.3)
    assert event["request_id"] == "req_1"
    assert event["user_id"] == "u1"
    assert event["session_id"] == "s1"
    assert event["intent"] == "knowledge"
    assert event["agent_used"] == "knowledge"
    assert event["latency_ms"] == 12.3
    assert event["cost_usd"] == 0.12
    assert event["tool_results_count"] == 1
    assert event["error_type"] is None
    assert event["success"] is True


def test_build_query_trace_event_tool_error() -> None:
    initial_state = {"request_id": "req_2", "user_id": "u1", "session_id": "s1", "query": "q"}
    final_state = {"tool_results": [{"success": False, "tool_name": "web_search", "error_type": "timeout"}]}

    event = build_query_trace_event(initial_state=initial_state, final_state=final_state, latency_ms=1.0)
    assert event["error_type"] == "timeout"
    assert event["success"] is False


def test_build_query_trace_event_exception() -> None:
    event = build_query_trace_event(
        initial_state={"request_id": "req_3", "user_id": "u", "session_id": "s"},
        final_state=None,
        latency_ms=9.5,
        exc=RuntimeError("boom"),
    )
    assert event["error_type"] in {"runtime_error", "unknown_error"}
    assert event["success"] is False
    assert event["latency_ms"] == 9.5


def test_invoke_with_trace_logs_success(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setenv("CORTEX_QUERY_LOG_PATH", str(log_path))

    class _Graph:
        def invoke(self, state):
            return {
                **state,
                "intent": "knowledge",
                "agent_used": "knowledge",
                "response": "ok",
                "cost_usd": 0.001,
                "tool_results": [{"tool_name": "kb", "success": True}],
            }

    monkeypatch.setattr(supervisor_module, "build_supervisor_graph", lambda: _Graph())

    final_state = invoke_with_trace("What is the parental leave policy?", user_id="u1", user_tier="standard", session_id="s1")

    assert "latency_ms" in final_state["trace"]
    payload = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert payload["request_id"]
    assert payload["user_id"] == "u1"
    assert payload["session_id"] == "s1"
    assert payload["intent"] == "knowledge"
    assert payload["agent_used"] == "knowledge"
    assert payload["latency_ms"] >= 0
    assert payload["cost_usd"] == 0.001
    assert payload["tool_results_count"] == 1


def test_invoke_with_trace_logs_failure(monkeypatch, tmp_path: Path) -> None:
    log_path = tmp_path / "queries.jsonl"
    monkeypatch.setenv("CORTEX_QUERY_LOG_PATH", str(log_path))

    class _Graph:
        def invoke(self, state):
            raise RuntimeError("boom")

    monkeypatch.setattr(supervisor_module, "build_supervisor_graph", lambda: _Graph())

    with pytest.raises(RuntimeError):
        invoke_with_trace("q", user_id="u1", user_tier="standard", session_id="s1")

    payload = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert payload["success"] is False
    assert payload["error_type"]
