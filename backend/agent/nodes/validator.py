from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState


# This block defines the callable shape expected for optional proof builders.
# It takes: the current graph state after guardrail approval.
# It gives: a dictionary containing proof, signature, or attestation packaging data.
ValidatorPayloadProvider = Callable[[RRAAgentState], dict[str, Any]]


# This block configures how the validator node behaves.
# It takes: an optional payload provider callback for proof/signature generation.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class ValidatorNodeConfig:
    payload_provider: ValidatorPayloadProvider | None = None


# This block builds the validator graph node.
# It takes: a config object containing optional proof-generation hooks.
# It gives: a LangGraph-compatible node function that prepares execution-safe validation artifacts.
def build_validator_node(
    config: ValidatorNodeConfig | None = None,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    resolved_config = config or ValidatorNodeConfig()

    def validator_node(state: RRAAgentState) -> dict[str, Any]:
        # This block ensures the graph has a policy-approved trade before validation continues.
        # It takes: the current graph state.
        # It gives: a fail-fast error if policy approval or the trade candidate is missing.
        if state.trade_intent_record is None:
            raise ValueError("Validator node requires state.trade_intent_record to be present.")

        if not state.policy.allowed:
            raise ValueError("Validator node requires an allowed policy state before execution.")

        if state.trust_snapshot is None:
            raise ValueError("Validator node requires state.trust_snapshot to be present.")

        # This block collects any optional proof or signature payloads from the configured provider.
        # It takes: the current graph state.
        # It gives: optional validation artifacts for future ERC-8004 or EIP-712 workflows.
        proof_payload: dict[str, Any] = {}
        if resolved_config.payload_provider is not None:
            proof_payload = resolved_config.payload_provider(state)
            if not isinstance(proof_payload, dict):
                raise ValueError("payload_provider must return a dictionary")

        # This block builds the canonical execution request envelope.
        # It takes: the approved trade, trust snapshot, and optional validation artifacts.
        # It gives: a deterministic payload that the execution node can consume directly.
        execution_request = {
            "trace_id": state.trace_id,
            "thread_id": state.thread_id,
            "symbol": state.trade_intent_record.symbol,
            "side": state.trade_intent_record.side.value,
            "order_type": state.trade_intent_record.order_type.value,
            "notional_usd": state.trade_intent_record.notional_usd,
            "position_size_bps": state.trade_intent_record.position_size_bps,
            "slippage_bps": state.trade_intent_record.slippage_bps,
            "time_in_force": state.trade_intent_record.time_in_force,
            "trust": {
                "agent_address": state.trust_snapshot.agent_address,
                "erc8004_registered": state.trust_snapshot.erc8004_registered,
                "valid_attestation": state.trust_snapshot.valid_attestation,
                "trust_score": state.trust_snapshot.trust_score,
                "registry_reachable": state.trust_snapshot.registry_reachable,
            },
            "policy": {
                "action": state.policy.action,
                "allowed": state.policy.allowed,
                "violations": list(state.policy.violations),
            },
            "proof_payload": proof_payload,
        }

        # This block updates metadata with validator output for downstream execution and audit use.
        # It takes: the existing metadata and the canonical execution request envelope.
        # It gives: a metadata dict containing the validator-stage artifacts.
        metadata = dict(state.metadata)
        metadata["validator"] = {
            "validated": True,
            "execution_request": execution_request,
            "proof_payload": proof_payload,
        }

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and approved trade metadata.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(
            "validator completed "
            f"symbol={state.trade_intent_record.symbol} "
            f"side={state.trade_intent_record.side.value}"
        )

        # This block returns the partial state update expected by LangGraph.
        # It takes: the updated metadata and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "metadata": metadata,
            "messages": messages,
        }

    return validator_node
