from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from backend.agent.state import RRAAgentState
from backend.data.models import PolicyAction as StoredPolicyAction
from backend.data.models import PolicyDecisionRecord, PolicyViolationRecord, RecordSource
from security.policy import GuardrailsPolicy


# This block configures how the guardrail node behaves.
# It takes: a loaded GuardrailsPolicy instance.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class GuardrailNodeConfig:
    policy: GuardrailsPolicy


# This block builds the guardrail graph node.
# It takes: a config object containing the loaded policy engine.
# It gives: a LangGraph-compatible node function that applies deterministic policy checks.
def build_guardrail_node(
    config: GuardrailNodeConfig,
) -> Callable[[RRAAgentState], dict[str, object]]:
    def guardrail_node(state: RRAAgentState) -> dict[str, object]:
        # This block ensures a validated trade candidate exists before policy evaluation.
        # It takes: the current graph state.
        # It gives: a fail-fast error if the intent node has not produced a trade record.
        if state.trade_intent_record is None:
            raise ValueError("Guardrail node requires state.trade_intent_record to be present.")

        # This block builds the runtime context expected by the policy engine.
        # It takes: the current graph state.
        # It gives: a policy-compatible context dictionary from trust, market, and metadata.
        runtime_context = state.to_runtime_context()

        # This block evaluates the trade candidate against the loaded policy.
        # It takes: the trade record converted to policy input plus runtime context.
        # It gives: a deterministic allow/reduce/block decision with violations.
        decision = config.policy.evaluate(
            trade=state.trade_intent_record.to_policy_trade(),
            context=runtime_context,
        )

        # This block converts policy violations into durable record models.
        # It takes: the policy-engine violation objects.
        # It gives: storage-safe PolicyViolationRecord entries for auditability.
        violation_records = [
            PolicyViolationRecord(
                code=violation.code,
                message=violation.message,
                path=violation.path,
                details=violation.details,
                reducible=violation.reducible,
            )
            for violation in decision.violations
        ]

        # This block builds the durable policy decision record.
        # It takes: the current trade record and the evaluated policy decision.
        # It gives: a storage-safe PolicyDecisionRecord for persistence and replay.
        decision_record = PolicyDecisionRecord(
            trade_intent_id=state.trade_intent_record.id,
            action=StoredPolicyAction(decision.action.value),
            policy_name=decision.policy_name,
            policy_version=decision.policy_version,
            violations=violation_records,
            source=RecordSource.POLICY,
        )

        # This block updates the trade intent when policy evaluation reduces the candidate.
        # It takes: the normalized trade returned by the policy engine.
        # It gives: an updated TradeIntentRecord aligned with the deterministic policy result.
        updated_trade_record = state.trade_intent_record.model_copy(
            update={
                "notional_usd": float(
                    decision.normalized_trade.get(
                        "notional_usd",
                        state.trade_intent_record.notional_usd,
                    )
                ),
                "position_size_bps": int(
                    decision.normalized_trade.get(
                        "position_size_bps",
                        state.trade_intent_record.position_size_bps,
                    )
                ),
                "slippage_bps": float(
                    decision.normalized_trade.get(
                        "slippage_bps",
                        state.trade_intent_record.slippage_bps,
                    )
                ),
            }
        )

        # This block builds the serializable policy sub-state stored on the graph.
        # It takes: the evaluated decision and durable decision record.
        # It gives: a graph-friendly policy state for downstream routing.
        policy_state = {
            "action": decision.action.value,
            "allowed": decision.allowed,
            "violations": [
                {
                    "code": violation.code,
                    "message": violation.message,
                    "path": violation.path,
                    "details": violation.details,
                    "reducible": violation.reducible,
                }
                for violation in decision.violations
            ],
            "record": decision_record,
        }


        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and policy result.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(
            "guardrail evaluated "
            f"action={decision.action.value} "
            f"violations={len(decision.violations)}"
        )

        # This block returns the partial state update expected by LangGraph.
        # It takes: the updated trade record, policy state, and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "trade_intent_record": updated_trade_record,
            "policy": policy_state,
            "messages": messages,
        }

    return guardrail_node
