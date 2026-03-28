from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# This block defines the valid trade directions accepted by the system.
# It takes: raw string inputs from LLM output or operator input.
# It gives: a normalized enum value that downstream logic can trust.
class TradeSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


# This block defines the supported execution order types.
# It takes: raw order type strings from validated inputs.
# It gives: a constrained enum for later execution and policy layers.
class OrderType(StrEnum):
    LIMIT = "limit"
    MARKET = "market"


# This block defines the supported market regimes expected from the strategy layer.
# It takes: regime labels from model output or runtime classification.
# It gives: a normalized enum used by policy and orchestration logic.
class MarketRegime(StrEnum):
    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"


# This block defines the shared Pydantic model behavior for all validation schemas.
# It takes: arbitrary inbound payloads.
# It gives: strict schema validation with trimmed strings and no unexpected fields.
class ValidationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        populate_by_name=True,
    )


# This block validates the trade intent that may come from an LLM or operator action.
# It takes: symbol, side, order configuration, sizing, and strategy-quality metadata.
# It gives: a normalized trade object safe to pass into policy evaluation.
class TradeIntent(ValidationModel):
    symbol: str = Field(min_length=3, max_length=32)
    side: TradeSide
    order_type: OrderType = OrderType.LIMIT
    notional_usd: float = Field(gt=0)
    position_size_bps: int = Field(gt=0, le=10_000)
    signal_score: float = Field(ge=0.0, le=1.0)
    model_confidence: float = Field(ge=0.0, le=1.0)
    slippage_bps: float = Field(ge=0.0, le=10_000)
    regime: MarketRegime
    thesis: str | None = Field(default=None, max_length=2_000)
    time_in_force: str = Field(default="gtc", min_length=2, max_length=16)

    # This block normalizes the trading symbol into a consistent uppercase format.
    # It takes: a raw symbol string like "btcusd" or " btc-usd ".
    # It gives: an uppercase symbol suitable for policy and execution layers.
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/")
        if not normalized or any(char not in allowed for char in normalized):
            raise ValueError("symbol contains unsupported characters")
        return normalized

    # This block normalizes time-in-force values into lowercase.
    # It takes: a raw time-in-force string.
    # It gives: a predictable value for later execution logic.
    @field_validator("time_in_force")
    @classmethod
    def validate_time_in_force(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"gtc", "ioc", "fok"}:
            raise ValueError("time_in_force must be one of: gtc, ioc, fok")
        return normalized

    # This block enforces a small safety rule for market orders.
    # It takes: the validated fields on the trade intent.
    # It gives: a clearer schema-level rejection for obviously unsafe market orders.
    @model_validator(mode="after")
    def validate_market_order_slippage(self) -> TradeIntent:
        if self.order_type == OrderType.MARKET and self.slippage_bps <= 0:
            raise ValueError("market orders must provide a positive slippage_bps value")
        return self

    # This block converts the typed model into the exact trade dict expected by policy.py.
    # It takes: the validated TradeIntent object.
    # It gives: a plain dictionary for downstream policy evaluation.
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


# This block validates the full LLM output payload for a strategy decision.
# It takes: the model-generated signal plus explanation and review metadata.
# It gives: one strict envelope that the agent layer can trust before policy evaluation.
class LLMSignalOutput(ValidationModel):
    trade_intent: TradeIntent
    summary: str = Field(min_length=1, max_length=2_000)
    risks: list[str] = Field(default_factory=list, max_length=10)
    requires_human_review: bool = False

    # This block ensures the risk list is clean and bounded.
    # It takes: raw risk strings from model output.
    # It gives: a compact list of normalized, non-empty risk notes.
    @field_validator("risks")
    @classmethod
    def validate_risks(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if len(normalized) > 10:
            raise ValueError("risks cannot contain more than 10 entries")
        return normalized


# This block validates the trust context flowing in from ERC-8004 and attestation layers.
# It takes: on-chain registration state, trust score, and enclave-attestation metadata.
# It gives: a normalized trust payload for the security policy engine.
class TrustContext(ValidationModel):
    trust_score: float = Field(ge=0.0, le=1.0)
    erc8004_registered: bool
    valid_attestation: bool
    registry_reachable: bool
    attestation_age_seconds: int = Field(ge=0)
    attested: bool = False
    tee_measurement: str | None = None
    tee_enclave_id: str | None = None

    # This block converts the trust model into the exact shape expected by policy.py.
    # It takes: the validated TrustContext object.
    # It gives: a plain dictionary ready for policy evaluation.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "trust_score": self.trust_score,
            "erc8004_registered": self.erc8004_registered,
            "valid_attestation": self.valid_attestation,
            "registry_reachable": self.registry_reachable,
            "attestation_age_seconds": self.attestation_age_seconds,
            "attested": self.attested,
            "tee_measurement": self.tee_measurement,
            "tee_enclave_id": self.tee_enclave_id,
        }


# This block validates current portfolio and position state.
# It takes: current exposure and open-position counts from the account/risk layer.
# It gives: a clean position snapshot for policy enforcement.
class PositionsContext(ValidationModel):
    total_open_exposure_bps: float = Field(ge=0.0, le=10_000.0)
    asset_open_exposure_bps: float = Field(ge=0.0, le=10_000.0)
    open_positions_count: int = Field(ge=0)

    # This block converts the positions model into the shape expected by policy.py.
    # It takes: the validated PositionsContext object.
    # It gives: a plain dictionary for policy evaluation.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "total_open_exposure_bps": self.total_open_exposure_bps,
            "asset_open_exposure_bps": self.asset_open_exposure_bps,
            "open_positions_count": self.open_positions_count,
        }


# This block validates live market-state inputs used by the policy layer.
# It takes: spread, volatility, staleness, and market-data availability.
# It gives: a normalized market context safe for execution gating.
class MarketContext(ValidationModel):
    spread_bps: float = Field(ge=0.0)
    realized_volatility: float = Field(ge=0.0)
    orderbook_stale_seconds: float = Field(ge=0.0)
    market_data_available: bool = True

    # This block converts the market model into the exact policy input shape.
    # It takes: the validated MarketContext object.
    # It gives: a plain dictionary for the policy engine.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "spread_bps": self.spread_bps,
            "realized_volatility": self.realized_volatility,
            "orderbook_stale_seconds": self.orderbook_stale_seconds,
            "market_data_available": self.market_data_available,
        }


# This block validates runtime execution-pressure inputs.
# It takes: recent order counts, failure counts, and pre-trade risk status.
# It gives: a normalized execution context for policy checks.
class ExecutionContext(ValidationModel):
    orders_in_last_minute: int = Field(ge=0)
    failed_orders_count: int = Field(ge=0)
    pre_trade_risk_check_passed: bool

    # This block converts the execution model into the exact policy input shape.
    # It takes: the validated ExecutionContext object.
    # It gives: a plain dictionary for the policy engine.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "orders_in_last_minute": self.orders_in_last_minute,
            "failed_orders_count": self.failed_orders_count,
            "pre_trade_risk_check_passed": self.pre_trade_risk_check_passed,
        }


# This block validates infrastructure health inputs.
# It takes: subsystem health from Kraken CLI, checkpointing, inference, and clock sync.
# It gives: a normalized system-health context for policy checks.
class SystemContext(ValidationModel):
    kraken_cli_available: bool
    checkpoint_store_available: bool
    inference_backend_degraded: bool = False
    clock_skew_ms: int = Field(ge=0)

    # This block converts the system model into the exact policy input shape.
    # It takes: the validated SystemContext object.
    # It gives: a plain dictionary for the policy engine.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "kraken_cli_available": self.kraken_cli_available,
            "checkpoint_store_available": self.checkpoint_store_available,
            "inference_backend_degraded": self.inference_backend_degraded,
            "clock_skew_ms": self.clock_skew_ms,
        }


# This block validates explicit kill-switch and operator-state flags.
# It takes: human override flags and safety state from operations/risk controls.
# It gives: a normalized flag context for deny-rule evaluation.
class FlagsContext(ValidationModel):
    manual_kill_switch: bool = False
    daily_loss_limit_breached: bool = False
    wallet_balance_usd: float = Field(ge=0.0)
    untrusted_operator_context: bool = False

    # This block converts the flags model into the exact policy input shape.
    # It takes: the validated FlagsContext object.
    # It gives: a plain dictionary for the policy engine.
    def to_policy_context(self) -> dict[str, Any]:
        return {
            "manual_kill_switch": self.manual_kill_switch,
            "daily_loss_limit_breached": self.daily_loss_limit_breached,
            "wallet_balance_usd": self.wallet_balance_usd,
            "untrusted_operator_context": self.untrusted_operator_context,
        }


# This block validates the full policy-evaluation payload before it reaches security/policy.py.
# It takes: a trade intent plus all typed runtime context models.
# It gives: one top-level object that can be converted into the exact policy input contract.
class PolicyEvaluationInput(ValidationModel):
    trade: TradeIntent
    trust: TrustContext
    positions: PositionsContext
    market: MarketContext
    execution: ExecutionContext
    system: SystemContext
    flags: FlagsContext

    # This block converts the typed model into the exact argument structure expected by policy.py.
    # It takes: the validated PolicyEvaluationInput object.
    # It gives: a dict with `trade` and `context` ready for policy.evaluate(...).
    def to_policy_input(self) -> dict[str, Any]:
        return {
            "trade": self.trade.to_policy_trade(),
            "context": {
                "trust": self.trust.to_policy_context(),
                "positions": self.positions.to_policy_context(),
                "market": self.market.to_policy_context(),
                "execution": self.execution.to_policy_context(),
                "system": self.system.to_policy_context(),
                "flags": self.flags.to_policy_context(),
            },
        }


# This block validates operator-supplied manual trade requests.
# It takes: direct human inputs when an operator wants to create or override a trade intent.
# It gives: a safe, typed request that can later be converted into a TradeIntent.
class OperatorTradeRequest(ValidationModel):
    operator_id: str = Field(min_length=1, max_length=128)
    symbol: str = Field(min_length=3, max_length=32)
    side: TradeSide
    order_type: OrderType = OrderType.LIMIT
    requested_notional_usd: float = Field(gt=0)
    requested_position_size_bps: int = Field(gt=0, le=10_000)
    max_slippage_bps: float = Field(ge=0.0, le=10_000)
    regime: MarketRegime
    reason: str = Field(min_length=1, max_length=2_000)
    dry_run: bool = True

    # This block normalizes the operator symbol into uppercase.
    # It takes: a raw symbol string from manual input.
    # It gives: a clean symbol that matches the trade-intent schema.
    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_/")
        if not normalized or any(char not in allowed for char in normalized):
            raise ValueError("symbol contains unsupported characters")
        return normalized

    # This block converts operator input into a standard TradeIntent object.
    # It takes: default signal metadata for manually initiated requests.
    # It gives: a TradeIntent that the rest of the pipeline can process normally.
    def to_trade_intent(
        self,
        *,
        signal_score: float = 1.0,
        model_confidence: float = 1.0,
    ) -> TradeIntent:
        return TradeIntent(
            symbol=self.symbol,
            side=self.side,
            order_type=self.order_type,
            notional_usd=self.requested_notional_usd,
            position_size_bps=self.requested_position_size_bps,
            signal_score=signal_score,
            model_confidence=model_confidence,
            slippage_bps=self.max_slippage_bps,
            regime=self.regime,
            thesis=self.reason,
        )
