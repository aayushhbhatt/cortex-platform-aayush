import threading
import time

from reliability.errors import RateLimitExceededError

RateLimitError = RateLimitExceededError


class RateLimiter:
    def __init__(self, capacity: float = 10.0, refill_rate: float = 1.0):
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")
        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self._last_refill
        if elapsed <= 0:
            return
        self._tokens = min(self.capacity, self._tokens + (elapsed * self.refill_rate))
        self._last_refill = now

    def acquire(self, tokens: float = 1.0, wait: bool = False, timeout: float = 5.0) -> bool:
        if tokens <= 0:
            raise ValueError("tokens must be > 0")

        deadline = time.time() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                needed = tokens - self._tokens

            if not wait:
                raise RateLimitExceededError("Rate limit exceeded. No tokens available.")

            if time.time() >= deadline:
                raise RateLimitExceededError("Rate limit wait timeout exceeded.")

            sleep_seconds = min(max(needed / self.refill_rate, 0.001), 0.1)
            time.sleep(sleep_seconds)
