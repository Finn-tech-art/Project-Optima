from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.data.models import RecordSource, TradeIntentRecord
from backend.validation.normalizers import normalize_llm_signal_payload
from backend.validation.parsers import parse_structured_output
from backend.validation.schemas import LLMSignalOutput


# This block defines the callable shape expected for optional intent enrichers.
# It takes: the validated LLM signal output and current graph state.
# It gives: optional additional raw payload fields to attach to the TradeIntentRecord.
IntentPayloadEnricher = Callable[[LLMSignalOutput, RRAAgentState], dict[str, Any]]


# This block configures how the intent node behaves.
# It takes: an optional payload enricher callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class IntentNodeConfig:
    payload_enricher: IntentPayloadEnricher | None = None


# This block builds the intent graph node.
# It takes: a config object containing optional enrichment hooks.
# It gives: a LangGraph-compatible node function that validates inference output into a trade intent.
def build_intent_node(
    config: IntentNodeConfig | None = None,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    resolved_config = config or IntentNodeConfig()

    def intent_node(state: RRAAgentState) -> dict[str, Any]:
        # This block reads the raw model output produced by the sentiment node.
        # It takes: the current graph state.
        # It gives: the raw inference text that must be parsed and validated.
        raw_output = state.inference.raw_output
        if not raw_output or not raw_output.strip():
            raise ValueError("Intent node requires state.inference.raw_output to be present.")

        # This block parses the raw output into structured content.
        # It takes: JSON, fenced JSON, or XML-wrapped inference output.
        # It gives: a ParsedStructuredOutput object with decoded payload data.
        parsed_output = parse_structured_output(raw_output)

        # This block normalizes minor model-output inconsistencies.
        # It takes: the parsed payload dictionary.
        # It gives: a canonical payload shape aligned with strict schemas.
        normalized_payload = normalize_llm_signal_payload(parsed_output.payload)

        # This block validates the normalized payload against the strict signal schema.
        # It takes: the normalized payload dictionary.
        # It gives: a typed LLMSignalOutput object trusted by the rest of the graph.
        signal_output = LLMSignalOutput.model_validate(normalized_payload)

        # This block optionally enriches the raw payload attached to the trade record.
        # It takes: the validated signal output and current graph state.
        # It gives: additional structured metadata for storage and audit trails.
        enriched_payload: dict[str, Any] = {}
        if resolved_config.payload_enricher is not None:
            enriched_payload = resolved_config.payload_enricher(signal_output, state)
            if not isinstance(enriched_payload, dict):
                raise ValueError("payload_enricher must return a dictionary")

        # This block builds the durable trade-intent record from the validated signal.
        # It takes: the validated trade intent plus current graph identifiers.
        # It gives: a storage-safe TradeIntentRecord for guardrails and later persistence.
        trade_intent = signal_output.trade_intent
        trade_record = TradeIntentRecord(
            trace_id=state.trace_id,
            symbol=trade_intent.symbol,
            side=trade_intent.side,
            order_type=trade_intent.order_type,
            notional_usd=trade_intent.notional_usd,
            position_size_bps=trade_intent.position_size_bps,
            signal_score=trade_intent.signal_score,
            model_confidence=trade_intent.model_confidence,
            slippage_bps=trade_intent.slippage_bps,
            regime=trade_intent.regime,
            thesis=trade_intent.thesis,
            time_in_force=trade_intent.time_in_force,
            source=RecordSource.LLM,
            raw_payload={
                "parsed_payload": parsed_output.payload,
                "normalized_payload": normalized_payload,
                "memory": state.metadata.get("memory", {}),
                **enriched_payload,
            },
        )

        # This block updates the inference sub-state with validated artifacts.
        # It takes: the parsed output, normalized payload, and validated signal object.
        # It gives: a serializable inference-state update for downstream nodes.
        inference_state = state.inference.model_dump()
        inference_state["parsed_payload"] = parsed_output.payload
        inference_state["normalized_payload"] = normalized_payload
        inference_state["signal_output"] = signal_output.model_dump()

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and validated trade candidate.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(
            "intent assembled "
            f"symbol={trade_record.symbol} "
            f"side={trade_record.side.value} "
            f"notional_usd={trade_record.notional_usd}"
        )

        # This block returns the partial state update expected by LangGraph.
        # It takes: the trade record, validated inference artifacts, and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "trade_intent_record": trade_record,
            "inference": inference_state,
            "messages": messages,
        }

    return intent_node
