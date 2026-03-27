from __future__ import annotations

from typing import Any


class ProjectOptimaError(Exception):
    """Base exception for all Project-Optima failures."""

    default_code = "project_optima_error"
    default_retryable = False

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        retryable: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.default_code
        self.retryable = (
            self.default_retryable if retryable is None else retryable
        )
        self.context = context or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "code": self.code,
            "retryable": self.retryable,
            "context": self.context,
        }

    def __str__(self) -> str:
        return self.message


class ConfigurationError(ProjectOptimaError):
    default_code = "configuration_error"


class MissingEnvironmentVariableError(ConfigurationError):
    default_code = "missing_environment_variable"


class ValidationError(ProjectOptimaError):
    default_code = "validation_error"


class InfrastructureError(ProjectOptimaError):
    default_code = "infrastructure_error"
    default_retryable = True


class RetryExhaustedError(InfrastructureError):
    default_code = "retry_exhausted"

    def __init__(
        self,
        message: str,
        *,
        attempts: int,
        last_error: Exception | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        merged_context["attempts"] = attempts
        if last_error is not None:
            merged_context["last_error_type"] = last_error.__class__.__name__
            merged_context["last_error_message"] = str(last_error)

        super().__init__(
            message,
            code=self.default_code,
            retryable=False,
            context=merged_context,
        )
        self.attempts = attempts
        self.last_error = last_error


class RateLimitExceededError(InfrastructureError):
    default_code = "rate_limit_exceeded"


class CircuitBreakerOpenError(InfrastructureError):
    default_code = "circuit_breaker_open"


class PersistenceError(ProjectOptimaError):
    default_code = "persistence_error"
    default_retryable = True


class CheckpointPersistenceError(PersistenceError):
    default_code = "checkpoint_persistence_error"


class StateCorruptionError(PersistenceError):
    default_code = "state_corruption_error"
    default_retryable = False


class InferenceError(ProjectOptimaError):
    default_code = "inference_error"
    default_retryable = True


class GroqInferenceError(InferenceError):
    default_code = "groq_inference_error"


class TimeoutError(ProjectOptimaError):
    default_code = "timeout_error"
    default_retryable = True


class ExecutionError(ProjectOptimaError):
    default_code = "execution_error"


class SubprocessExecutionError(ExecutionError):
    default_code = "subprocess_execution_error"

    def __init__(
        self,
        message: str,
        *,
        command: list[str] | None = None,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        if command is not None:
            merged_context["command"] = command
        if exit_code is not None:
            merged_context["exit_code"] = exit_code
        if stdout:
            merged_context["stdout"] = stdout
        if stderr:
            merged_context["stderr"] = stderr

        super().__init__(
            message,
            code=self.default_code,
            retryable=retryable,
            context=merged_context,
        )
        self.command = command or []
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


class KrakenCLIError(SubprocessExecutionError):
    default_code = "kraken_cli_error"


class KrakenCLIUnavailableError(KrakenCLIError):
    default_code = "kraken_cli_unavailable"
    default_retryable = False

    def __init__(
        self,
        message: str = "Kraken CLI is unavailable or not installed.",
        *,
        command: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            command=command,
            retryable=False,
            context=context,
        )


class KrakenCLIResponseError(KrakenCLIError):
    default_code = "kraken_cli_response_error"

    def __init__(
        self,
        message: str,
        *,
        payload: Any = None,
        command: list[str] | None = None,
        exit_code: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        if payload is not None:
            merged_context["payload"] = payload

        super().__init__(
            message,
            command=command,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            retryable=retryable,
            context=merged_context,
        )


class MarketDataError(ProjectOptimaError):
    default_code = "market_data_error"
    default_retryable = True


class StrategyError(ProjectOptimaError):
    default_code = "strategy_error"


class NodeExecutionError(ProjectOptimaError):
    default_code = "node_execution_error"

    def __init__(
        self,
        message: str,
        *,
        node_name: str,
        retryable: bool = False,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context = dict(context or {})
        merged_context["node_name"] = node_name
        super().__init__(
            message,
            code=self.default_code,
            retryable=retryable,
            context=merged_context,
        )
        self.node_name = node_name


class TrustLayerError(ProjectOptimaError):
    default_code = "trust_layer_error"


class ERC8004RegistryError(TrustLayerError):
    default_code = "erc8004_registry_error"


class AttestationError(TrustLayerError):
    default_code = "attestation_error"


class AuthorizationError(TrustLayerError):
    default_code = "authorization_error"
    default_retryable = False
