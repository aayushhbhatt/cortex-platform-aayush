import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm.client import OpenAIChatClient, reset_llm_circuit_breaker_for_tests
from llm.schemas import LLMRequest, LLMResponse


def _req() -> LLMRequest:
    return LLMRequest(prompt_name="supervisor_router", prompt_version="v1.0.0", agent="supervisor", system_prompt="sys", user_prompt="user", model="gpt-4o-mini")


def test_llm_client_disabled_and_missing_key_return_structured_errors(monkeypatch) -> None:
    reset_llm_circuit_breaker_for_tests()
    monkeypatch.setenv("ENABLE_CORTEX_LLM", "false")
    r1 = OpenAIChatClient(api_key=None).complete(_req())
    assert r1.success is False and r1.error_type == "llm_disabled"
    reset_llm_circuit_breaker_for_tests()
    reset_llm_circuit_breaker_for_tests()
    monkeypatch.setenv("ENABLE_CORTEX_LLM", "true")
    r2 = OpenAIChatClient(api_key=None).complete(_req())
    assert r2.success is False and r2.error_type in {"auth_error", "provider_error", "circuit_open"}


def test_llm_client_retries_retryable_failure_then_success(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_CORTEX_LLM", "true")
    monkeypatch.setattr("llm.client.time.sleep", lambda *_: None)
    calls = {"n": 0}
    def fake_once(self, request):
        calls["n"] += 1
        if calls["n"] == 1:
            return LLMResponse(success=False, error_type="provider_error", error_message="temporary")
        return LLMResponse(success=True, content="ok", model=request.model)
    monkeypatch.setattr(OpenAIChatClient, "_complete_once", fake_once)
    out = OpenAIChatClient(api_key="test").complete(_req())
    assert out.success is True and calls["n"] == 2 and "reliability" in (out.raw or {})


def test_llm_client_circuit_opens_after_failures(monkeypatch) -> None:
    monkeypatch.setenv("ENABLE_CORTEX_LLM", "true")
    monkeypatch.setattr("llm.client.time.sleep", lambda *_: None)
    monkeypatch.setattr(OpenAIChatClient, "_complete_once", lambda self, req: LLMResponse(success=False, error_type="provider_error", error_message="down"))
    client = OpenAIChatClient(api_key="test")
    for _ in range(6):
        out = client.complete(_req())
    assert out.error_type in {"provider_error", "circuit_open"}
    reset_llm_circuit_breaker_for_tests()
