from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.data.models import ExecutionRecord, ExecutionStatus


# This block defines the callable shape expected for execution providers.
# It takes: the validated execution request payload and current graph state.
# It gives: a dictionary containing execution status and optional execution records.
ExecutionProvider = Callable[[dict[str, Any], RRAAgentState], dict[str, Any]]


# This block configures how the execution node behaves.
# It takes: an execution provider callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class ExecutionNodeConfig:
    execution_provider: ExecutionProvider


# This block builds the execution graph node.
# It takes: a config object containing the execution provider.
# It gives: a LangGraph-compatible node function that submits approved trades to execution.
def build_execution_node(
    config: ExecutionNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def execution_node(state: RRAAgentState) -> dict[str, Any]:
        # This block ensures the validator node already prepared an execution request.
        # It takes: the current graph state.
        # It gives: a fail-fast error if execution is attempted without validated inputs.
        validator_metadata = state.metadata.get("validator", {})
        if not isinstance(validator_metadata, dict):
            raise ValueError("Execution node requires validator metadata to be a dictionary.")

        execution_request = validator_metadata.get("execution_request")
        if not isinstance(execution_request, dict):
            raise ValueError(
                "Execution node requires metadata['validator']['execution_request'] to be present."
            )

        # This block invokes the configured execution provider.
        # It takes: the canonical execution request and current graph state.
        # It gives: raw execution output from the hardened exchange adapter.
        execution_result = config.execution_provider(execution_request, state)
        if not isinstance(execution_result, dict):
            raise ValueError("execution_provider must return a dictionary")

        # This block normalizes the execution status returned by the provider.
        # It takes: the raw provider result.
        # It gives: a strict execution status for graph state and routing.
        status = _coerce_execution_status(execution_result.get("status", "failed"))

        # This block normalizes any execution records returned by the provider.
        # It takes: the raw `records` list from the provider.
        # It gives: a typed list of ExecutionRecord models.
        raw_records = execution_result.get("records", [])
        if not isinstance(raw_records, list):
            raise ValueError("execution_provider result field 'records' must be a list")

        execution_records = [
            record if isinstance(record, ExecutionRecord) else ExecutionRecord.model_validate(record)
            for record in raw_records
        ]

        # This block updates the execution sub-state stored on the graph.
        # It takes: the normalized execution status, records, and last-error value.
        # It gives: a graph-friendly execution state for terminal handling and replay.
        execution_state = {
            "attempted": True,
            "status": status.value,
            "last_error": (
                str(execution_result.get("last_error")).strip()
                if execution_result.get("last_error") is not None
                else None
            ),
            "records": execution_records,
        }

        # This block stores the raw execution response in metadata for observability.
        # It takes: the existing metadata and provider output.
        # It gives: a metadata dict containing execution-stage artifacts.
        metadata = dict(state.metadata)
        metadata["execution"] = {
            "orders_in_last_minute": int(
                execution_result.get(
                    "orders_in_last_minute",
                    state.metadata.get("execution", {}).get("orders_in_last_minute", 0)
                    if isinstance(state.metadata.get("execution"), dict)
                    else 0,
                )
            ),
            "failed_orders_count": int(
                execution_result.get(
                    "failed_orders_count",
                    state.metadata.get("execution", {}).get("failed_orders_count", 0)
                    if isinstance(state.metadata.get("execution"), dict)
                    else 0,
                )
            ),
            "pre_trade_risk_check_passed": bool(
                execution_result.get(
                    "pre_trade_risk_check_passed",
                    state.metadata.get("execution", {}).get("pre_trade_risk_check_passed", True)
                    if isinstance(state.metadata.get("execution"), dict)
                    else True,
                )
            ),
            "raw_result": execution_result,
        }

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and normalized execution status.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(f"execution completed status={status.value}")

        # This block returns the partial state update expected by LangGraph.
        # It takes: the updated execution state, metadata, and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "execution": execution_state,
            "metadata": metadata,
            "messages": messages,
        }

    return execution_node


# This block coerces raw provider status values into the strict ExecutionStatus enum.
# It takes: a raw status value from the execution provider.
# It gives: a canonical ExecutionStatus value or raises if unsupported.
def _coerce_execution_status(value: Any) -> ExecutionStatus:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "partial_fill": "partially_filled",
        "partial": "partially_filled",
    }
    normalized = aliases.get(normalized, normalized)

    try:
        return ExecutionStatus(normalized)
    except ValueError as error:
        raise ValueError(f"Unsupported execution status: {value}") from error
