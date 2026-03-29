from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.data.models import (
    ExecutionRecord,
    MarketSnapshotRecord,
    PolicyDecisionRecord,
    TradeIntentRecord,
    TrustSnapshotRecord,
)
from backend.validation.schemas import LLMSignalOutput


# This block defines the shared Pydantic behavior for agent state models.
# It takes: state updates produced by graph nodes.
# It gives: strict, checkpoint-friendly state validation.
class AgentStateModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


# This block stores the latest inference artifact produced by the strategy layer.
# It takes: raw model text, parsed structured output, and validated signal output.
# It gives: one normalized inference snapshot for downstream nodes.
class InferenceState(AgentStateModel):
    raw_output: str | None = None
    parsed_payload: dict[str, Any] | None = None
    normalized_payload: dict[str, Any] | None = None
    signal_output: LLMSignalOutput | None = None
    model_name: str | None = None


# This block stores the latest policy-evaluation result.
# It takes: the decision action, violation details, and persisted policy record if available.
# It gives: a normalized policy snapshot for downstream execution gating.
class PolicyState(AgentStateModel):
    action: str | None = None
    allowed: bool = False
    violations: list[dict[str, Any]] = Field(default_factory=list)
    record: PolicyDecisionRecord | None = None


# This block stores execution-layer status for the current trade flow.
# It takes: execution attempts, order status, and optional persisted execution records.
# It gives: a compact execution state for graph routing and observability.
class ExecutionState(AgentStateModel):
    attempted: bool = False
    status: str | None = None
    last_error: str | None = None
    records: list[ExecutionRecord] = Field(default_factory=list)


# This block defines the main LangGraph state contract for the RRA workflow.
# It takes: trace identifiers, current market and trust context, inference outputs, and policy/execution artifacts.
# It gives: one stable state object that every graph node can read and update safely.
class RRAAgentState(AgentStateModel):
    trace_id: str
    thread_id: str
    symbol: str
    user_objective: str | None = None

    market_snapshot: MarketSnapshotRecord | None = None
    trust_snapshot: TrustSnapshotRecord | None = None
    trade_intent_record: TradeIntentRecord | None = None

    inference: InferenceState = Field(default_factory=InferenceState)
    policy: PolicyState = Field(default_factory=PolicyState)
    execution: ExecutionState = Field(default_factory=ExecutionState)

    messages: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # This block normalizes the tracked symbol for consistency across nodes and persistence layers.
    # It takes: the symbol value from the caller.
    # It gives: an uppercase symbol stored consistently in graph state.
    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()

    # This block appends a human-readable message to the state timeline.
    # It takes: a new message string from a graph node.
    # It gives: an updated message history for debugging and checkpoint replay.
    def add_message(self, message: str) -> None:
        normalized = message.strip()
        if normalized:
            self.messages.append(normalized)

    # This block returns the runtime context shape expected by the validation and policy layer.
    # It takes: the current state object.
    # It gives: a policy-compatible context dictionary built from current snapshots and metadata.
    def to_runtime_context(self) -> dict[str, Any]:
        return {
            "trust": self.trust_snapshot.to_policy_context() if self.trust_snapshot else {},
            "positions": self._dict_metadata_section("positions"),
            "market": self.market_snapshot.to_policy_context() if self.market_snapshot else {},
            "execution": self._dict_metadata_section("execution"),
            "system": self._dict_metadata_section("system"),
            "flags": self._dict_metadata_section("flags"),
        }

    # This block safely reads a metadata section as a dictionary.
    # It takes: the metadata key requested by downstream logic.
    # It gives: either the stored dictionary or an empty dictionary.
    def _dict_metadata_section(self, key: str) -> dict[str, Any]:
        value = self.metadata.get(key, {})
        if isinstance(value, dict):
            return value
        return {}
