from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.validation.schemas import MarketRegime, OrderType, TradeSide


# This block defines the persistence/source labels used across stored records.
# It takes: string source names from agent, operator, policy, or system layers.
# It gives: a constrained enum for record provenance.
class RecordSource(StrEnum):
    LLM = "llm"
    OPERATOR = "operator"
    POLICY = "policy"
    SYSTEM = "system"
    EXECUTION = "execution"


# This block defines the policy actions that may be persisted in decision history.
# It takes: allow/reduce/block decisions from the policy engine.
# It gives: a constrained enum for storage-safe policy action values.
class PolicyAction(StrEnum):
    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"


# This block defines execution lifecycle states for trade attempts.
# It takes: the outcome of execution-layer processing.
# It gives: a normalized enum for durable execution history.
class ExecutionStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# This block defines the shared Pydantic behavior for all data-layer models.
# It takes: inbound model payloads from validation, policy, and execution layers.
# It gives: strict, serialization-friendly storage models.
class DataModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )


# This block defines common audit fields shared by persisted records.
# It takes: no explicit input beyond object creation time.
# It gives: stable IDs and UTC timestamps for storage and tracing.
class AuditModel(DataModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # This block ensures timestamps are always stored in UTC.
    # It takes: a datetime value from the caller or default factory.
    # It gives: a timezone-aware UTC datetime for storage consistency.
    @field_validator("created_at")
    @classmethod
    def ensure_utc_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


# This block stores a validated trade intent as a durable record.
# It takes: trade intent data from validated LLM or operator flows.
# It gives: a storage-safe trade record for PostgreSQL and execution history.
class TradeIntentRecord(AuditModel):
    trace_id: str
    symbol: str
    side: TradeSide
    order_type: OrderType
    notional_usd: float = Field(gt=0)
    position_size_bps: int = Field(gt=0, le=10_000)
    signal_score: float = Field(ge=0.0, le=1.0)
    model_confidence: float = Field(ge=0.0, le=1.0)
    slippage_bps: float = Field(ge=0.0)
    regime: MarketRegime
    thesis: str | None = None
    time_in_force: str = Field(default="gtc")
    source: RecordSource = RecordSource.LLM
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    # This block normalizes the stored trade symbol for indexing and joins.
    # It takes: the symbol field from the caller.
    # It gives: an uppercase storage-safe symbol.
    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()

    # This block converts the record back into the policy trade shape.
    # It takes: the stored trade record.
    # It gives: a plain dictionary compatible with security.policy.
    def to_policy_trade(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "notional_usd": self.notional_usd,
            "position_size_bps": self.position_size_bps,
            "signal_score": self.signal_score,
            "model_confidence": self.model_confidence,
            "slippage_bps": self.slippage_bps,
            "regime": self.regime.value,
            "thesis": self.thesis,
            "time_in_force": self.time_in_force,
        }


# This block stores one policy violation as part of a policy decision.
# It takes: code/message/path/details from the security policy engine.
# It gives: a stable violation record for audit and debugging.
class PolicyViolationRecord(DataModel):
    code: str
    message: str
    path: str
    details: dict[str, Any] = Field(default_factory=dict)
    reducible: bool = False


# This block stores the result of evaluating a trade against policy guardrails.
# It takes: the policy decision, linked trade record, and violation list.
# It gives: a durable audit record for safety and compliance review.
class PolicyDecisionRecord(AuditModel):
    trade_intent_id: str
    action: PolicyAction
    policy_name: str
    policy_version: str
    violations: list[PolicyViolationRecord] = Field(default_factory=list)
    source: RecordSource = RecordSource.POLICY

    # This block returns whether the policy engine allowed the trade.
    # It takes: the stored policy decision record.
    # It gives: a quick boolean for downstream callers.
    @property
    def allowed(self) -> bool:
        return self.action == PolicyAction.ALLOW


# This block stores a trust snapshot built from ERC-8004 and TEE attestation layers.
# It takes: trust and attestation data observed at a specific point in time.
# It gives: a durable trust record for policy evaluation, debugging, and replay.
class TrustSnapshotRecord(AuditModel):
    agent_address: str
    trust_score: float | None = Field(default=None, ge=0.0, le=1.0)
    erc8004_registered: bool
    valid_attestation: bool
    registry_reachable: bool
    attestation_age_seconds: int = Field(ge=0)
    attested: bool = False
    tee_measurement: str | None = None
    tee_enclave_id: str | None = None
    raw_registry: dict[str, Any] = Field(default_factory=dict)
    raw_attestation: dict[str, Any] = Field(default_factory=dict)

    # This block normalizes the stored agent address for lookups and joins.
    # It takes: the caller-provided agent address.
    # It gives: a lowercase canonical storage representation.
    @field_validator("agent_address")
    @classmethod
    def normalize_agent_address(cls, value: str) -> str:
        return value.lower()

    # This block converts the record into the trust context shape expected by policy.
    # It takes: the stored trust snapshot record.
    # It gives: a plain dictionary ready for policy evaluation.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "trust_score": self.trust_score or 0.0,
            "erc8004_registered": self.erc8004_registered,
            "valid_attestation": self.valid_attestation,
            "registry_reachable": self.registry_reachable,
            "attestation_age_seconds": self.attestation_age_seconds,
            "attested": self.attested,
            "tee_measurement": self.tee_measurement,
            "tee_enclave_id": self.tee_enclave_id,
        }


# This block stores a market snapshot used by validation and policy evaluation.
# It takes: live market-state data captured for a symbol at a point in time.
# It gives: a durable market record for replay, analytics, and agent context.
class MarketSnapshotRecord(AuditModel):
    symbol: str
    bid: float | None = Field(default=None, ge=0.0)
    ask: float | None = Field(default=None, ge=0.0)
    mid: float | None = Field(default=None, ge=0.0)
    spread_bps: float = Field(ge=0.0)
    realized_volatility: float = Field(ge=0.0)
    orderbook_stale_seconds: float = Field(ge=0.0)
    market_data_available: bool = True
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    # This block normalizes the stored symbol for indexing and joins.
    # It takes: the symbol field from the caller.
    # It gives: an uppercase storage-safe symbol.
    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()

    # This block converts the record into the market context shape expected by policy.
    # It takes: the stored market snapshot record.
    # It gives: a plain dictionary ready for policy evaluation.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "spread_bps": self.spread_bps,
            "realized_volatility": self.realized_volatility,
            "orderbook_stale_seconds": self.orderbook_stale_seconds,
            "market_data_available": self.market_data_available,
        }


# This block stores the outcome of an execution attempt or order lifecycle update.
# It takes: linked trade identifiers, exchange references, and execution status details.
# It gives: a durable execution history record for reconciliation and monitoring.
class ExecutionRecord(AuditModel):
    trade_intent_id: str
    policy_decision_id: str | None = None
    symbol: str
    side: TradeSide
    order_type: OrderType
    status: ExecutionStatus
    exchange_order_id: str | None = None
    requested_notional_usd: float = Field(gt=0)
    executed_notional_usd: float | None = Field(default=None, ge=0.0)
    average_fill_price: float | None = Field(default=None, ge=0.0)
    failure_reason: str | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    source: RecordSource = RecordSource.EXECUTION

    # This block normalizes the stored symbol for indexing and joins.
    # It takes: the symbol field from the caller.
    # It gives: an uppercase storage-safe symbol.
    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


# This block stores metadata for LangGraph checkpoint-related persistence.
# It takes: thread identifiers, checkpoint keys, and optional graph state metadata.
# It gives: a typed record that postgres/checkpoint helpers can share.
class CheckpointRecord(AuditModel):
    thread_id: str
    checkpoint_ns: str
    checkpoint_id: str
    state_version: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: RecordSource = RecordSource.SYSTEM


# This block stores semantic memory documents intended for vector storage.
# It takes: retrieval text, metadata, and optional embedding vector values.
# It gives: a shared model that Qdrant and higher-level memory logic can use consistently.
class MemoryDocument(AuditModel):
    collection: str
    text: str = Field(min_length=1)
    symbol: str | None = None
    trace_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    vector: list[float] | None = None
    source: RecordSource = RecordSource.SYSTEM

    # This block normalizes the optional symbol for retrieval metadata consistency.
    # It takes: an optional symbol string.
    # It gives: an uppercase symbol or None.
    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()

    # This block converts the model into a Qdrant payload structure.
    # It takes: the stored memory document.
    # It gives: a metadata payload suitable for vector upsert operations.
    def to_qdrant_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "collection": self.collection,
            "text": self.text,
            "symbol": self.symbol,
            "trace_id": self.trace_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "source": self.source.value,
            "created_at": self.created_at.isoformat(),
        }
