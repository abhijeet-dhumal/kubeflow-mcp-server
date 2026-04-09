# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resilience patterns: circuit breaker, retry, rate limiting."""

import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for Kubernetes API calls."""

    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3

    state: CircuitState = field(default=CircuitState.CLOSED)
    failure_count: int = field(default=0)
    last_failure_time: float = field(default=0.0)
    half_open_calls: int = field(default=0)

    def can_execute(self) -> bool:
        """Check if call is allowed."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                return True
            return False

        return self.half_open_calls < self.half_open_max_calls

    def record_success(self) -> None:
        """Record successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
        else:
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("Circuit breaker: HALF_OPEN -> OPEN")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker: CLOSED -> OPEN (failures={self.failure_count})"
            )


_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    """Return the named CircuitBreaker, creating it on first access.

    Using per-name breakers prevents a failure in one tool (e.g.
    ``get_runtime_packages``) from tripping the circuit for unrelated
    tools (e.g. ``fine_tune``).
    """
    if name not in _breakers:
        _breakers[name] = CircuitBreaker()
    return _breakers[name]


def reset_breakers() -> None:
    """Clear all circuit breakers (for testing)."""
    _breakers.clear()


def with_circuit_breaker(
    breaker: CircuitBreaker | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to wrap function with a circuit breaker.

    If *breaker* is ``None`` a dedicated breaker keyed by the function name
    is created automatically, so each decorated function has its own
    independent failure budget.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        cb = breaker if breaker is not None else get_breaker(func.__name__)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            if not cb.can_execute():
                raise RuntimeError(f"Circuit breaker open for {func.__name__}")

            try:
                result = func(*args, **kwargs)
                cb.record_success()
                return result
            except Exception:
                cb.record_failure()
                raise

        return wrapper

    return decorator


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: float = 0.1,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retry with exponential backoff and jitter."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break

                    delay = min(base_delay * (2**attempt), max_delay)
                    delay += random.uniform(-jitter * delay, jitter * delay)
                    logger.debug(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} in {delay:.2f}s"
                    )
                    time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper

    return decorator


async def retry_with_backoff_async(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 1.0,
    **kwargs: Any,
) -> Any:
    """Async retry with exponential backoff."""
    last_exception: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == max_retries:
                break
            delay = base_delay * (2**attempt)
            await asyncio.sleep(delay)

    raise last_exception  # type: ignore


@dataclass
class RateLimiter:
    """Token bucket rate limiter."""

    rate: float = 10.0
    capacity: float = 10.0
    _tokens: float = field(init=False)
    _last_update: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = self.capacity
        self._last_update = time.time()

    def acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens. Returns True if successful."""
        now = time.time()
        elapsed = now - self._last_update
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_update = now

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False


@dataclass
class SessionManager:
    """Detect and recover from stale Kubernetes sessions."""

    max_age: float = 300.0
    _timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record_activity(self) -> None:
        """Record session activity."""
        self._timestamps.append(time.time())

    def is_stale(self) -> bool:
        """Check if session appears stale."""
        if not self._timestamps:
            return False
        return time.time() - self._timestamps[-1] > self.max_age
