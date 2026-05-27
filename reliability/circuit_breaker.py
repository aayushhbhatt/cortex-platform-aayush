from dataclasses import dataclass
from enum import Enum
import time

from reliability.errors import CircuitOpenError


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout_seconds: float = 30.0

    def __post_init__(self):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = False

    def allow_request(self) -> bool:
        now = time.time()
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            if self._opened_at is None:
                self._opened_at = now
            if now - self._opened_at >= self.recovery_timeout_seconds:
                self.state = CircuitState.HALF_OPEN
                if not self._half_open_in_flight:
                    self._half_open_in_flight = True
                    return True
            return False
        if self.state == CircuitState.HALF_OPEN:
            if not self._half_open_in_flight:
                self._half_open_in_flight = True
                return True
            return False
        return False

    def record_success(self) -> None:
        self.failure_count = 0
        self.state = CircuitState.CLOSED
        self._opened_at = None
        self._half_open_in_flight = False

    def record_failure(self) -> None:
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._opened_at = time.time()
            self._half_open_in_flight = False
            self.failure_count = self.failure_threshold
            return

        self.failure_count += 1
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            self._opened_at = time.time()
            self._half_open_in_flight = False

    def call(self, func, *args, **kwargs):
        if not self.allow_request():
            raise CircuitOpenError("Circuit is open; request rejected.")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result
