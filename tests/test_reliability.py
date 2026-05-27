import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reliability.circuit_breaker import CircuitBreaker
from reliability.errors import CircuitOpenError, RateLimitExceededError
from reliability.rate_limiter import RateLimiter
from reliability.retry import with_retry


def test_token_bucket_rate_limiter_allows_then_limits():
    limiter = RateLimiter(capacity=1, refill_rate=1)
    assert limiter.acquire(tokens=1) is True
    with pytest.raises(RateLimitExceededError):
        limiter.acquire(tokens=1, wait=False)


def test_retry_with_exponential_backoff_eventually_succeeds(monkeypatch):
    monkeypatch.setattr("reliability.retry.time.sleep", lambda *_: None)
    attempts = {"n": 0}

    @with_retry(max_attempts=3, base_delay=0.01)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RateLimitExceededError("later")
        return "ok"

    assert flaky() == "ok"
    assert attempts["n"] == 3


def test_circuit_breaker_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout_seconds=30)
    cb.record_failure(); cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.call(lambda: "ok")
