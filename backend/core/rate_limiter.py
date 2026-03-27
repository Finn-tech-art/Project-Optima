from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field

from backend.core.constants import (
    DEFAULT_RATE_LIMIT_CAPACITY,
    DEFAULT_RATE_LIMIT_REFILL_RATE,
    DEFAULT_RATE_LIMIT_TIMEOUT_SECONDS,
)
from backend.core.exceptions import RateLimitExceededError


@dataclass(slots=True)
class RateLimiter:
    """
    Token-bucket rate limiter for sync and async backend workloads.
    """

    capacity: int = DEFAULT_RATE_LIMIT_CAPACITY
    refill_rate: float = DEFAULT_RATE_LIMIT_REFILL_RATE
    timeout_seconds: float = DEFAULT_RATE_LIMIT_TIMEOUT_SECONDS
    _tokens: float = field(init=False)
    _last_refill_time: float = field(init=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be greater than 0")
        if self.refill_rate <= 0:
            raise ValueError("refill_rate must be greater than 0")
        if self.timeout_seconds < 0:
            raise ValueError("timeout_seconds must be greater than or equal to 0")

        self._tokens = float(self.capacity)
        self._last_refill_time = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self, now: float) -> None:
        elapsed = max(now - self._last_refill_time, 0.0)
        if elapsed == 0:
            return

        replenished = elapsed * self.refill_rate
        self._tokens = min(float(self.capacity), self._tokens + replenished)
        self._last_refill_time = now

    def _try_consume_unlocked(self, tokens: int, now: float) -> bool:
        self._refill(now)

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def available_tokens(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._refill(now)
            return self._tokens

    def try_acquire(self, tokens: int = 1) -> bool:
        if tokens <= 0:
            raise ValueError("tokens must be greater than 0")

        with self._lock:
            now = time.monotonic()
            return self._try_consume_unlocked(tokens, now)

    def acquire(self, tokens: int = 1, timeout_seconds: float | None = None) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be greater than 0")

        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout < 0:
            raise ValueError("timeout_seconds must be greater than or equal to 0")

        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                now = time.monotonic()
                if self._try_consume_unlocked(tokens, now):
                    return

                remaining = deadline - now
                if remaining <= 0:
                    break

                missing_tokens = max(tokens - self._tokens, 0.0)
                wait_time = min(missing_tokens / self.refill_rate, remaining)

            if wait_time > 0:
                time.sleep(wait_time)
            else:
                time.sleep(0.001)

        raise RateLimitExceededError(
            "Timed out while waiting for rate limiter tokens.",
            context={
                "requested_tokens": tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "timeout_seconds": timeout,
            },
        )

    async def acquire_async(
        self,
        tokens: int = 1,
        timeout_seconds: float | None = None,
    ) -> None:
        if tokens <= 0:
            raise ValueError("tokens must be greater than 0")

        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout < 0:
            raise ValueError("timeout_seconds must be greater than or equal to 0")

        deadline = time.monotonic() + timeout

        while True:
            with self._lock:
                now = time.monotonic()
                if self._try_consume_unlocked(tokens, now):
                    return

                remaining = deadline - now
                if remaining <= 0:
                    break

                missing_tokens = max(tokens - self._tokens, 0.0)
                wait_time = min(missing_tokens / self.refill_rate, remaining)

            if wait_time > 0:
                await asyncio.sleep(wait_time)
            else:
                await asyncio.sleep(0.001)

        raise RateLimitExceededError(
            "Timed out while waiting for async rate limiter tokens.",
            context={
                "requested_tokens": tokens,
                "capacity": self.capacity,
                "refill_rate": self.refill_rate,
                "timeout_seconds": timeout,
            },
        )
