from __future__ import annotations

from typing import Any

from backend.agent.state import RRAAgentState


# This block builds the finalize graph node.
# It takes: no external configuration because finalization should remain deterministic and simple.
# It gives: a LangGraph-compatible node function that marks the run as complete.
def build_finalize_node() -> callable:
    def finalize_node(state: RRAAgentState) -> dict[str, Any]:
        # This block derives the terminal status from policy and execution state.
        # It takes: the current graph state.
        # It gives: one compact final-status label for audit and replay.
        final_status = _derive_final_status(state)

        # This block builds a final metadata snapshot for observability.
        # It takes: the current graph state and derived terminal status.
        # It gives: a compact summary of the completed run.
        metadata = dict(state.metadata)
        metadata["final"] = {
            "status": final_status,
            "trace_id": state.trace_id,
            "thread_id": state.thread_id,
            "symbol": state.symbol,
            "policy_action": state.policy.action,
            "policy_allowed": state.policy.allowed,
            "execution_attempted": state.execution.attempted,
            "execution_status": state.execution.status,
            "message_count": len(state.messages) + 1,
        }

        # This block appends the terminal message for the run.
        # It takes: the existing message history and final-status label.
        # It gives: a final timeline entry for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(f"finalize completed status={final_status}")

        # This block returns the partial state update expected by LangGraph.
        # It takes: the final metadata and message timeline.
        # It gives: a deterministic terminal update to the graph state.
        return {
            "metadata": metadata,
            "messages": messages,
        }

    return finalize_node


# This block determines the final graph outcome.
# It takes: the current graph state after all earlier nodes have run.
# It gives: a single status string representing the terminal outcome.
def _derive_final_status(state: RRAAgentState) -> str:
    if state.execution.attempted:
        return state.execution.status or "execution_attempted"

    if state.policy.action == "block":
        return "blocked"

    if state.policy.action == "reduce":
        return "reduced"

    if state.policy.allowed:
        return "validated"

    return "completed"
