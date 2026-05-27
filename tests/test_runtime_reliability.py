import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliability.circuit_breaker import CircuitBreaker
from reliability.rate_limiter import RateLimiter
from reliability.runtime import RuntimeReliabilityPolicy, execute_with_reliability
from tools.error_contracts import success_response


def test_execute_with_reliability_success():
    out = execute_with_reliability(tool_name="x", operation=lambda: success_response(tool_name="x", data={"ok": True}).model_dump())
    assert out["success"] is True and out["data"]["ok"] is True


def test_execute_with_reliability_rate_limit_returns_structured_error():
    limiter = RateLimiter(capacity=1, refill_rate=0.01); limiter.acquire()
    out = execute_with_reliability(
        tool_name="x",
        operation=lambda: {"success": True},
        policy=RuntimeReliabilityPolicy(tool_name="x", rate_limiter=limiter),
    )
    assert out["success"] is False and out["error_type"] == "rate_limit" and out["recoverable"] is True


def test_execute_with_reliability_retries_then_circuit_or_success(monkeypatch):
    monkeypatch.setattr("reliability.retry.time.sleep", lambda *_: None)
    calls = {"n": 0}
    def op():
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("transient")
        return success_response(tool_name="x", data={"ok": True}).model_dump()
    out = execute_with_reliability(tool_name="x", operation=op, policy=RuntimeReliabilityPolicy(tool_name="x", circuit_breaker=CircuitBreaker(3, 60)))
    assert out["success"] is True and calls["n"] == 2
