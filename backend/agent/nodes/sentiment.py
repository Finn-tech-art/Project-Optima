from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.prompts.xml_wrapper import build_xml_wrapped_prompt


# This block defines the callable shape expected for sentiment inference.
# It takes: a system prompt and user prompt.
# It gives: a dict containing at least the raw model output and optional model metadata.
SentimentInferenceProvider = Callable[[str, str], dict[str, Any]]


# This block configures how the sentiment node behaves.
# It takes: an inference provider callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class SentimentNodeConfig:
    inference_provider: SentimentInferenceProvider


# This block builds the sentiment graph node.
# It takes: a config object containing the inference provider.
# It gives: a LangGraph-compatible node function that enriches the state with raw inference output.
def build_sentiment_node(
    config: SentimentNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def sentiment_node(state: RRAAgentState) -> dict[str, Any]:
        # This block reads the current market and regime context from state.
        # It takes: the current graph state.
        # It gives: the context needed to build a structured inference prompt.
        market_snapshot = state.market_snapshot
        regime_value = (
            state.metadata.get("regime", {}).get("value", "trend")
            if isinstance(state.metadata.get("regime"), dict)
            else "trend"
        )

        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            state=state,
            regime_value=regime_value,
            market_snapshot=market_snapshot.model_dump() if market_snapshot else {},
        )

        # This block wraps the prompts in the XML response contract expected by the parser layer.
        # It takes: the base system prompt and user prompt.
        # It gives: a deterministic prompt package for structured model output.
        wrapped_prompt = build_xml_wrapped_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # This block invokes the configured inference provider.
        # It takes: the wrapped system prompt and wrapped user prompt.
        # It gives: raw model output plus optional model metadata.
        inference_result = config.inference_provider(
            wrapped_prompt.system_prompt,
            wrapped_prompt.user_prompt,
        )
        if not isinstance(inference_result, dict):
            raise ValueError("inference_provider must return a dictionary")

        raw_output = str(inference_result.get("raw_output", "")).strip()
        if not raw_output:
            raise ValueError("inference_provider must return a non-empty 'raw_output' value")

        model_name = inference_result.get("model_name")
        if model_name is not None:
            model_name = str(model_name).strip() or None

        # This block updates the inference sub-state with raw model artifacts.
        # It takes: the current inference state plus raw output and model metadata.
        # It gives: a serializable inference-state update for downstream validation nodes.
        inference_state = state.inference.model_dump()
        inference_state["raw_output"] = raw_output
        inference_state["model_name"] = model_name

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and inference metadata.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(
            f"sentiment inference completed using model={model_name or 'unknown'}"
        )

        # This block returns the partial state update expected by LangGraph.
        # It takes: the updated inference state and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "inference": inference_state,
            "messages": messages,
        }

    return sentiment_node


# This block builds the system prompt for the sentiment node.
# It takes: no runtime input.
# It gives: the fixed system-level instructions for structured signal generation.
def _build_system_prompt() -> str:
    return (
        "You are the sentiment and signal-generation node for the Reflexive Reputation Arb "
        "system. Produce one structured trade candidate based on the provided market context. "
        "Do not include prose outside the required XML response envelope."
    )


# This block builds the user prompt for the sentiment node.
# It takes: current graph state, normalized regime, and current market snapshot data.
# It gives: a context-rich prompt that the inference layer can turn into a structured signal.
def _build_user_prompt(
    *,
    state: RRAAgentState,
    regime_value: str,
    market_snapshot: dict[str, Any],
) -> str:
    return (
        f"Symbol: {state.symbol}\n"
        f"Trace ID: {state.trace_id}\n"
        f"Thread ID: {state.thread_id}\n"
        f"User Objective: {state.user_objective or 'Generate a trade candidate.'}\n"
        f"Regime: {regime_value}\n"
        f"Market Snapshot: {market_snapshot}\n"
        "Generate a single candidate trade intent with summary and risk notes."
    )
