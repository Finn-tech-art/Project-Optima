from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from functools import wraps
from typing import ParamSpec, TypeVar

from backend.core.constants import (
    DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    DEFAULT_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS,
    DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS,
)
from backend.core.exceptions import CircuitBreakerOpenError, ProjectOptimaError


P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


def _is_failure_retryable(error: Exception) -> bool:
    if isinstance(error, ProjectOptimaError):
        return error.retryable
    return True


@dataclass(slots=True)
class CircuitBreaker:
    failure_threshold: int = DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD
    recovery_timeout_seconds: float = DEFAULT_CIRCUIT_BREAKER_RECOVERY_TIMEOUT_SECONDS
    half_open_max_calls: int = DEFAULT_CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS
    failure_predicate: Callable[[Exception], bool] = _is_failure_retryable
    name: str = "default"

    _state: CircuitState = field(init=False, default=CircuitState.CLOSED)
    _failure_count: int = field(init=False, default=0)
    _opened_at: float | None = field(init=False, default=None)
    _half_open_calls: int = field(init=False, default=0)
    _lock: threading.Lock = field(init=False, repr=False, default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if self.failure_threshold <= 0:
            raise ValueError("failure_threshold must be greater than 0")
        if self.recovery_timeout_seconds < 0:
            raise ValueError("recovery_timeout_seconds must be greater than or equal to 0")
        if self.half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be greater than 0")

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._advance_state_if_ready(time.monotonic())
            return self._state

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    def snapshot(self) -> dict[str, int | float | str | None]:
        with self._lock:
            now = time.monotonic()
            self._advance_state_if_ready(now)
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "opened_at": self._opened_at,
                "half_open_calls": self._half_open_calls,
                "recovery_timeout_seconds": self.recovery_timeout_seconds,
                "failure_threshold": self.failure_threshold,
                "half_open_max_calls": self.half_open_max_calls,
            }

    def _advance_state_if_ready(self, now: float) -> None:
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return

        if now - self._opened_at >= self.recovery_timeout_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0

    def _before_call(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._advance_state_if_ready(now)

            if self._state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker '{self.name}' is open.",
                    context=self.snapshot(),
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is half-open and probe capacity is exhausted.",
                        context=self.snapshot(),
                    )
                self._half_open_calls += 1

    def _on_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._half_open_calls = 0
            self._opened_at = None
            self._state = CircuitState.CLOSED

    def _on_failure(self, error: Exception) -> None:
        if not self.failure_predicate(error):
            return

        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._trip_open_locked()
                return

            self._failure_count += 1
            if self._failure_count >= self.failure_threshold:
                self._trip_open_locked()

    def _trip_open_locked(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._half_open_calls = 0

    def call(self, func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        self._before_call()
        try:
            result = func(*args, **kwargs)
        except Exception as error:
            self._on_failure(error)
            raise
        self._on_success()
        return result

    async def call_async(
        self,
        func: Callable[P, Awaitable[T]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T:
        self._before_call()
        try:
            result = await func(*args, **kwargs)
        except Exception as error:
            self._on_failure(error)
            raise
        self._on_success()
        return result


def with_circuit_breaker(
    circuit_breaker: CircuitBreaker,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                return await circuit_breaker.call_async(func, *args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return circuit_breaker.call(func, *args, **kwargs)

        return sync_wrapper

    return decorator
