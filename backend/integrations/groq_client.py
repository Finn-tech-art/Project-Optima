from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from groq import Groq
from groq import APIError as GroqAPIError
from groq import RateLimitError as GroqRateLimitError

from backend.core.constants import (
    DEFAULT_GROQ_MAX_TOKENS,
    DEFAULT_GROQ_TEMPERATURE,
)
from backend.core.exceptions import GroqInferenceError
from security.secrets import AppSecrets


# This block defines the normalized request shape for Groq inference.
# It takes: model selection, prompt content, and optional generation controls.
# It gives: one typed object that callers can pass into the Groq client wrapper.
@dataclass(slots=True, frozen=True)
class GroqInferenceRequest:
    model: str
    system_prompt: str
    user_prompt: str
    temperature: float = DEFAULT_GROQ_TEMPERATURE
    max_tokens: int = DEFAULT_GROQ_MAX_TOKENS
    response_format: dict[str, Any] | None = None


# This block defines the normalized response shape returned by the wrapper.
# It takes: raw Groq response data.
# It gives: a backend-friendly object for later nodes and policy layers.
@dataclass(slots=True, frozen=True)
class GroqInferenceResponse:
    model: str
    content: str
    raw: Any

    def json(self) -> dict[str, Any]:
        """
        This block attempts to parse the model output as JSON.
        It takes: the text content returned by the model.
        It gives: a parsed JSON object for structured-output workflows.
        """
        try:
            parsed = json.loads(self.content)
        except json.JSONDecodeError as error:
            raise GroqInferenceError(
                "Groq response content is not valid JSON.",
                context={"content": self.content},
            ) from error

        if not isinstance(parsed, dict):
            raise GroqInferenceError(
                "Groq response JSON must be an object.",
                context={"parsed_type": type(parsed).__name__},
            )

        return parsed


# This block is the main Groq inference adapter.
# It takes: loaded app secrets and optional default model configuration.
# It gives: a reusable client for structured and plain-text LPU-backed inference.
class GroqClient:
    def __init__(
        self,
        *,
        secrets: AppSecrets,
        default_model: str,
        fallback_model: str | None = None,
    ) -> None:
        secrets.require_groq()

        if not default_model.strip():
            raise GroqInferenceError("A default Groq model must be configured.")

        self.default_model = default_model.strip()
        self.fallback_model = fallback_model.strip() if fallback_model else None
        self._client = Groq(api_key=secrets.groq_api_key)

    def infer(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = DEFAULT_GROQ_TEMPERATURE,
        max_tokens: int = DEFAULT_GROQ_MAX_TOKENS,
        response_format: dict[str, Any] | None = None,
    ) -> GroqInferenceResponse:
        """
        This block sends one synchronous inference request to Groq.
        It takes: prompt content, optional model override, and generation settings.
        It gives: a normalized response object with text content and raw payload.
        """
        selected_model = (model or self.default_model).strip()

        try:
            response = self._client.chat.completions.create(
                model=selected_model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except GroqRateLimitError as error:
            if self.fallback_model and selected_model != self.fallback_model:
                return self.infer(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

            raise GroqInferenceError(
                "Groq rate limit exceeded.",
                retryable=True,
                context={"model": selected_model},
            ) from error
        except GroqAPIError as error:
            raise GroqInferenceError(
                "Groq API request failed.",
                retryable=True,
                context={"model": selected_model, "error": str(error)},
            ) from error
        except Exception as error:  # noqa: BLE001
            raise GroqInferenceError(
                "Unexpected Groq inference failure.",
                retryable=True,
                context={"model": selected_model, "error": str(error)},
            ) from error

        choice = response.choices[0] if response.choices else None
        content = ""
        if choice and choice.message and choice.message.content:
            content = choice.message.content

        return GroqInferenceResponse(
            model=selected_model,
            content=content,
            raw=response,
        )

    def infer_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = DEFAULT_GROQ_TEMPERATURE,
        max_tokens: int = DEFAULT_GROQ_MAX_TOKENS,
    ) -> dict[str, Any]:
        """
        This block requests structured JSON output from Groq.
        It takes: prompt content and optional generation settings.
        It gives: a parsed JSON object for downstream policy/agent logic.
        """
        response = self.infer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.json()
