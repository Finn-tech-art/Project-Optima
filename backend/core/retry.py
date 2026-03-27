from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from backend.core.constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF_MULTIPLIER,
    DEFAULT_RETRY_BASE_DELAY_SECONDS,
    DEFAULT_RETRY_JITTER_SECONDS,
    DEFAULT_RETRY_MAX_DELAY_SECONDS,
)
from backend.core.exceptions import ProjectOptimaError, RetryExhaustedError


P = ParamSpec("P")
T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Retry:
    """Configuration for bounded exponential retry."""

    attempts: int = DEFAULT_RETRY_ATTEMPTS
    base_delay_seconds: float = DEFAULT_RETRY_BASE_DELAY_SECONDS
    max_delay_seconds: float = DEFAULT_RETRY_MAX_DELAY_SECONDS
    backoff_multiplier: float = DEFAULT_RETRY_BACKOFF_MULTIPLIER
    jitter_seconds: float = DEFAULT_RETRY_JITTER_SECONDS

    def compute_delay(self, attempt_number: int) -> float:
        """
        Compute an exponential backoff delay with bounded jitter.

        `attempt_number` is 1-based for retry attempts after the initial call.
        """
        exponential_delay = self.base_delay_seconds * (
            self.backoff_multiplier ** max(attempt_number - 1, 0)
        )
        capped_delay = min(exponential_delay, self.max_delay_seconds)
        jitter = random.uniform(0.0, self.jitter_seconds)
        return capped_delay + jitter


def is_retryable_exception(error: Exception) -> bool:
    if isinstance(error, ProjectOptimaError):
        return error.retryable
    return False


def _resolve_should_retry(
    should_retry: Callable[[Exception], bool] | None,
) -> Callable[[Exception], bool]:
    return should_retry or is_retryable_exception


def _build_retry_exhausted_error(
    error: Exception,
    *,
    attempts: int,
    context: dict[str, Any] | None = None,
) -> RetryExhaustedError:
    return RetryExhaustedError(
        "Retry budget exhausted.",
        attempts=attempts,
        last_error=error,
        context=context,
    )


def execute_with_retry(
    func: Callable[P, T],
    *args: P.args,
    retry: Retry | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
    **kwargs: P.kwargs,
) -> T:
    policy = retry or Retry()
    retry_predicate = _resolve_should_retry(should_retry)

    last_error: Exception | None = None

    for attempt_index in range(policy.attempts):
        try:
            return func(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            last_error = error
            is_last_attempt = attempt_index >= policy.attempts - 1

            if not retry_predicate(error):
                raise

            if is_last_attempt:
                raise _build_retry_exhausted_error(
                    error,
                    attempts=policy.attempts,
                    context={"mode": "sync", "function": getattr(func, "__name__", "unknown")},
                ) from error

            delay = policy.compute_delay(attempt_index + 1)
            time.sleep(delay)

    if last_error is not None:
        raise _build_retry_exhausted_error(
            last_error,
            attempts=policy.attempts,
            context={"mode": "sync", "function": getattr(func, "__name__", "unknown")},
        ) from last_error

    raise RetryExhaustedError(
        "Retry loop exited unexpectedly.",
        attempts=policy.attempts,
        context={"mode": "sync", "function": getattr(func, "__name__", "unknown")},
    )


async def execute_with_retry_async(
    func: Callable[P, Awaitable[T]],
    *args: P.args,
    retry: Retry | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
    **kwargs: P.kwargs,
) -> T:
    policy = retry or Retry()
    retry_predicate = _resolve_should_retry(should_retry)

    last_error: Exception | None = None

    for attempt_index in range(policy.attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as error:  # noqa: BLE001
            last_error = error
            is_last_attempt = attempt_index >= policy.attempts - 1

            if not retry_predicate(error):
                raise

            if is_last_attempt:
                raise _build_retry_exhausted_error(
                    error,
                    attempts=policy.attempts,
                    context={"mode": "async", "function": getattr(func, "__name__", "unknown")},
                ) from error

            delay = policy.compute_delay(attempt_index + 1)
            await asyncio.sleep(delay)

    if last_error is not None:
        raise _build_retry_exhausted_error(
            last_error,
            attempts=policy.attempts,
            context={"mode": "async", "function": getattr(func, "__name__", "unknown")},
        ) from last_error

    raise RetryExhaustedError(
        "Async retry loop exited unexpectedly.",
        attempts=policy.attempts,
        context={"mode": "async", "function": getattr(func, "__name__", "unknown")},
    )


def with_retry(
    retry: Retry | None = None,
    *,
    should_retry: Callable[[Exception], bool] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorate a sync or async function with retry behavior.
    """
    policy = retry or Retry()
    retry_predicate = _resolve_should_retry(should_retry)

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
                return await execute_with_retry_async(
                    func,
                    *args,
                    retry=policy,
                    should_retry=retry_predicate,
                    **kwargs,
                )

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return execute_with_retry(
                func,
                *args,
                retry=policy,
                should_retry=retry_predicate,
                **kwargs,
            )

        return sync_wrapper

    return decorator
