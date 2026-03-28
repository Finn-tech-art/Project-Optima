from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from backend.core.exceptions import ConfigurationError, ValidationError


class PolicyAction(StrEnum):
    """The final action the policy engine can return."""

    ALLOW = "allow"
    REDUCE = "reduce"
    BLOCK = "block"


@dataclass(slots=True, frozen=True)
class PolicyViolation:
    """
    This block represents one policy failure.
    It takes: a rule code, a human-readable message, the config path, and optional details.
    It gives: a structured violation that logs, UIs, and execution gates can consume directly.
    """

    code: str
    message: str
    path: str
    details: dict[str, Any] = field(default_factory=dict)
    reducible: bool = False


@dataclass(slots=True, frozen=True)
class PolicyDecision:
    """
    This block represents the outcome of policy evaluation.
    It takes: the final action, the list of violations, and the normalized trade candidate.
    It gives: one machine-readable decision object for the rest of the stack.
    """

    action: PolicyAction
    violations: list[PolicyViolation]
    normalized_trade: dict[str, Any]
    policy_name: str
    policy_version: str

    @property
    def allowed(self) -> bool:
        return self.action == PolicyAction.ALLOW

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "allowed": self.allowed,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "violations": [
                {
                    "code": violation.code,
                    "message": violation.message,
                    "path": violation.path,
                    "details": violation.details,
                    "reducible": violation.reducible,
                }
                for violation in self.violations
            ],
            "normalized_trade": self.normalized_trade,
        }


@dataclass(slots=True)
class GuardrailsPolicy:
    """
    This block is the main policy engine.
    It takes: loaded YAML policy data and an optional source path.
    It gives: validated access to guardrails and evaluation methods for runtime trade checks.
    """

    raw_policy: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> GuardrailsPolicy:
        """
        This block loads policy from disk.
        It takes: a filesystem path to guardrails.yaml.
        It gives: a ready-to-use GuardrailsPolicy instance.
        """
        source_path = Path(path)
        if not source_path.exists():
            raise ConfigurationError(
                f"Guardrails policy file does not exist: {source_path}",
                context={"path": str(source_path)},
            )

        try:
            raw_data = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as error:
            raise ConfigurationError(
                "Failed to parse guardrails YAML.",
                context={"path": str(source_path), "error": str(error)},
            ) from error

        if not isinstance(raw_data, dict):
            raise ConfigurationError(
                "Guardrails policy root must be a mapping.",
                context={"path": str(source_path)},
            )

        policy = cls(raw_policy=raw_data, source_path=source_path)
        policy.validate()
        return policy

    def validate(self) -> None:
        """
        This block validates the minimum shape of the policy document.
        It takes: the raw YAML mapping.
        It gives: confidence that later evaluation code can safely read required sections.
        """
        self._require_mapping("meta")
        self._require_mapping("enforcement")
        self._require_mapping("trust_requirements")
        self._require_mapping("position_limits")
        self._require_mapping("trade_limits")
        self._require_mapping("signal_requirements")
        self._require_mapping("market_conditions")
        self._require_mapping("execution_controls")
        self._require_mapping("system_safety")
        self._require_mapping("deny_rules")

        meta = self.section("meta")
        if "name" not in meta or "version" not in meta:
            raise ValidationError(
                "Policy meta section must include name and version.",
                context={"section": "meta"},
            )

    def section(self, name: str) -> dict[str, Any]:
        """
        This block fetches one policy section.
        It takes: a section name like 'trade_limits' or 'trust_requirements'.
        It gives: the section mapping, defaulting to an empty dict if missing.
        """
        value = self.raw_policy.get(name, {})
        if not isinstance(value, dict):
            raise ValidationError(
                f"Policy section '{name}' must be a mapping.",
                context={"section": name},
            )
        return value

    def policy_name(self) -> str:
        """Return the policy document name for logs and decisions."""
        return str(self.section("meta").get("name", "unknown-policy"))

    def policy_version(self) -> str:
        """Return the policy version for logs and decisions."""
        return str(self.section("meta").get("version", "0.0.0"))

    def _require_mapping(self, section_name: str) -> None:
        """
        This block enforces that required YAML sections are mappings.
        It takes: a section name.
        It gives: an early validation failure if the policy document is malformed.
        """
        value = self.raw_policy.get(section_name)
        if not isinstance(value, dict):
            raise ValidationError(
                f"Policy section '{section_name}' must be a mapping.",
                context={"section": section_name},
            )

    def _merged_trade_limits(self, symbol: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        This block merges global defaults with per-asset overrides.
        It takes: the trade symbol.
        It gives: effective position limits and trade limits for that asset.
        """
        position_limits = copy.deepcopy(self.section("position_limits"))
        trade_limits = copy.deepcopy(self.section("trade_limits"))

        asset_overrides = self.raw_policy.get("asset_overrides", {})
        if not isinstance(asset_overrides, dict):
            raise ValidationError(
                "Policy section 'asset_overrides' must be a mapping.",
                context={"section": "asset_overrides"},
            )

        override = asset_overrides.get(symbol.upper(), {})
        if override:
            if not isinstance(override, dict):
                raise ValidationError(
                    f"Asset override for '{symbol}' must be a mapping.",
                    context={"symbol": symbol},
                )

            for key in ("max_position_size_bps",):
                if key in override:
                    position_limits[key] = override[key]

            for key in ("max_notional_usd", "max_slippage_bps"):
                if key in override:
                    trade_limits[key] = override[key]

        return position_limits, trade_limits

    def evaluate(
        self,
        trade: dict[str, Any],
        context: dict[str, Any],
    ) -> PolicyDecision:
        """
        This block evaluates a proposed trade.
        It takes: the trade candidate plus runtime context from trust, market, and account layers.
        It gives: a final allow/reduce/block decision and a normalized trade payload.
        """
        if not isinstance(trade, dict):
            raise ValidationError("Trade input must be a mapping.")
        if not isinstance(context, dict):
            raise ValidationError("Context input must be a mapping.")

        violations: list[PolicyViolation] = []
        normalized_trade = copy.deepcopy(trade)

        symbol = str(trade.get("symbol", "")).upper().strip()
        if not symbol:
            violations.append(
                PolicyViolation(
                    code="missing_symbol",
                    message="Trade symbol is required.",
                    path="trade.symbol",
                )
            )

        position_limits, trade_limits = self._merged_trade_limits(symbol or "UNKNOWN")

        self._evaluate_trust(context, violations)
        self._evaluate_position_limits(normalized_trade, context, position_limits, violations)
        self._evaluate_trade_limits(normalized_trade, trade_limits, violations)
        self._evaluate_signal_requirements(normalized_trade, violations)
        self._evaluate_market_conditions(context, violations)
        self._evaluate_execution_controls(context, violations)
        self._evaluate_system_safety(context, violations)
        self._evaluate_deny_rules(context, violations)

        action = self._resolve_action(violations)
        return PolicyDecision(
            action=action,
            violations=violations,
            normalized_trade=normalized_trade,
            policy_name=self.policy_name(),
            policy_version=self.policy_version(),
        )

    def _evaluate_trust(
        self,
        context: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block evaluates identity and trust prerequisites.
        It takes: trust and attestation runtime context.
        It gives: violations if the agent identity does not meet minimum trust requirements.
        """
        trust_policy = self.section("trust_requirements")
        trust_context = context.get("trust", {})
        if not isinstance(trust_context, dict):
            raise ValidationError("Context field 'trust' must be a mapping.")

        trust_score = float(trust_context.get("trust_score", 0.0))
        min_trust_score = float(trust_policy.get("min_trust_score", 0.0))
        if trust_score < min_trust_score:
            violations.append(
                PolicyViolation(
                    code="trust_score_below_floor",
                    message="Trust score is below the required minimum.",
                    path="trust_requirements.min_trust_score",
                    details={"actual": trust_score, "required": min_trust_score},
                )
            )

        if trust_policy.get("require_erc8004_registration", False):
            if not bool(trust_context.get("erc8004_registered", False)):
                violations.append(
                    PolicyViolation(
                        code="erc8004_registration_required",
                        message="ERC-8004 registration is required.",
                        path="trust_requirements.require_erc8004_registration",
                    )
                )

        if trust_policy.get("require_valid_attestation", False):
            if not bool(trust_context.get("valid_attestation", False)):
                violations.append(
                    PolicyViolation(
                        code="valid_attestation_required",
                        message="A valid TEE attestation is required.",
                        path="trust_requirements.require_valid_attestation",
                    )
                )

        max_age = int(trust_policy.get("attestation_max_age_seconds", 0))
        actual_age = int(trust_context.get("attestation_age_seconds", 0))
        if max_age > 0 and actual_age > max_age:
            violations.append(
                PolicyViolation(
                    code="attestation_too_old",
                    message="TEE attestation age exceeds the allowed maximum.",
                    path="trust_requirements.attestation_max_age_seconds",
                    details={"actual": actual_age, "allowed": max_age},
                )
            )

        if trust_policy.get("deny_if_registry_unreachable", False):
            if not bool(trust_context.get("registry_reachable", True)):
                violations.append(
                    PolicyViolation(
                        code="registry_unreachable",
                        message="ERC-8004 registry is unreachable.",
                        path="trust_requirements.deny_if_registry_unreachable",
                    )
                )

    def _evaluate_position_limits(
        self,
        trade: dict[str, Any],
        context: dict[str, Any],
        position_limits: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block checks portfolio and position-size constraints.
        It takes: the candidate trade, current position state, and policy limits.
        It gives: either a clamped trade size or blocking violations for exposure breaches.
        """
        positions_context = context.get("positions", {})
        if not isinstance(positions_context, dict):
            raise ValidationError("Context field 'positions' must be a mapping.")

        requested_bps = float(
            trade.get(
                "position_size_bps",
                position_limits.get("default_position_size_bps", 0),
            )
        )
        trade["position_size_bps"] = requested_bps

        max_position_bps = float(position_limits.get("max_position_size_bps", requested_bps))
        if requested_bps > max_position_bps:
            trade["position_size_bps"] = max_position_bps
            violations.append(
                PolicyViolation(
                    code="position_size_above_limit",
                    message="Requested position size exceeds the asset limit.",
                    path="position_limits.max_position_size_bps",
                    details={"requested": requested_bps, "allowed": max_position_bps},
                    reducible=True,
                )
            )

        total_open_exposure = float(positions_context.get("total_open_exposure_bps", 0.0))
        proposed_total = total_open_exposure + float(trade["position_size_bps"])
        max_total_exposure = float(position_limits.get("max_total_open_exposure_bps", proposed_total))
        if proposed_total > max_total_exposure:
            violations.append(
                PolicyViolation(
                    code="total_open_exposure_exceeded",
                    message="Total open exposure would exceed the allowed maximum.",
                    path="position_limits.max_total_open_exposure_bps",
                    details={"projected": proposed_total, "allowed": max_total_exposure},
                )
            )

        asset_exposure = float(positions_context.get("asset_open_exposure_bps", 0.0))
        projected_asset_exposure = asset_exposure + float(trade["position_size_bps"])
        max_asset_exposure = float(position_limits.get("max_single_asset_exposure_bps", projected_asset_exposure))
        if projected_asset_exposure > max_asset_exposure:
            violations.append(
                PolicyViolation(
                    code="single_asset_exposure_exceeded",
                    message="Single-asset exposure would exceed the allowed maximum.",
                    path="position_limits.max_single_asset_exposure_bps",
                    details={"projected": projected_asset_exposure, "allowed": max_asset_exposure},
                )
            )

        open_positions = int(positions_context.get("open_positions_count", 0))
        max_positions = int(position_limits.get("max_concurrent_positions", open_positions))
        if open_positions >= max_positions:
            violations.append(
                PolicyViolation(
                    code="max_concurrent_positions_reached",
                    message="Maximum concurrent positions already reached.",
                    path="position_limits.max_concurrent_positions",
                    details={"current": open_positions, "allowed": max_positions},
                )
            )

    def _evaluate_trade_limits(
        self,
        trade: dict[str, Any],
        trade_limits: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block checks order-level constraints.
        It takes: the normalized trade candidate and effective trade limits.
        It gives: a clamped notional where possible, or hard violations for unsafe orders.
        """
        notional_usd = float(trade.get("notional_usd", 0.0))
        max_notional = float(trade_limits.get("max_notional_usd", notional_usd))
        min_notional = float(trade_limits.get("min_notional_usd", 0.0))
        max_slippage = float(trade_limits.get("max_slippage_bps", 0.0))
        actual_slippage = float(trade.get("slippage_bps", 0.0))
        order_type = str(trade.get("order_type", "limit")).lower()

        if notional_usd < min_notional:
            violations.append(
                PolicyViolation(
                    code="notional_below_minimum",
                    message="Trade notional is below the minimum allowed.",
                    path="trade_limits.min_notional_usd",
                    details={"actual": notional_usd, "required": min_notional},
                )
            )

        if notional_usd > max_notional:
            trade["notional_usd"] = max_notional
            violations.append(
                PolicyViolation(
                    code="notional_above_limit",
                    message="Trade notional exceeds the asset limit.",
                    path="trade_limits.max_notional_usd",
                    details={"requested": notional_usd, "allowed": max_notional},
                    reducible=True,
                )
            )

        if actual_slippage > max_slippage:
            violations.append(
                PolicyViolation(
                    code="slippage_above_limit",
                    message="Expected slippage exceeds the allowed maximum.",
                    path="trade_limits.max_slippage_bps",
                    details={"actual": actual_slippage, "allowed": max_slippage},
                )
            )

        market_order_notional_limit = float(
            trade_limits.get("deny_market_orders_above_notional_usd", 0.0)
        )
        if order_type == "market" and notional_usd > market_order_notional_limit:
            violations.append(
                PolicyViolation(
                    code="market_order_notional_exceeded",
                    message="Market order notional exceeds the allowed threshold.",
                    path="trade_limits.deny_market_orders_above_notional_usd",
                    details={"actual": notional_usd, "allowed": market_order_notional_limit},
                )
            )

    def _evaluate_signal_requirements(
        self,
        trade: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block checks signal quality from the strategy layer.
        It takes: signal score, model confidence, and regime from the candidate trade.
        It gives: violations when the model output quality is too weak for execution.
        """
        signal_policy = self.section("signal_requirements")
        signal_score = float(trade.get("signal_score", 0.0))
        model_confidence = float(trade.get("model_confidence", 0.0))
        regime = str(trade.get("regime", "")).strip()

        min_signal = float(signal_policy.get("min_signal_score", 0.0))
        min_confidence = float(signal_policy.get("min_model_confidence", 0.0))
        allowed_regimes = signal_policy.get("allowed_regimes", [])

        if signal_score < min_signal:
            violations.append(
                PolicyViolation(
                    code="signal_score_below_floor",
                    message="Signal score is below the required minimum.",
                    path="signal_requirements.min_signal_score",
                    details={"actual": signal_score, "required": min_signal},
                )
            )

        if model_confidence < min_confidence:
            violations.append(
                PolicyViolation(
                    code="model_confidence_below_floor",
                    message="Model confidence is below the required minimum.",
                    path="signal_requirements.min_model_confidence",
                    details={"actual": model_confidence, "required": min_confidence},
                )
            )

        if allowed_regimes and regime not in allowed_regimes:
            violations.append(
                PolicyViolation(
                    code="regime_not_allowed",
                    message="Current regime is not allowed by policy.",
                    path="signal_requirements.allowed_regimes",
                    details={"actual": regime, "allowed": allowed_regimes},
                )
            )

    def _evaluate_market_conditions(
        self,
        context: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block checks live market safety conditions.
        It takes: spread, volatility, and market-data freshness from runtime context.
        It gives: violations when market quality is too poor for safe execution.
        """
        market_policy = self.section("market_conditions")
        market_context = context.get("market", {})
        if not isinstance(market_context, dict):
            raise ValidationError("Context field 'market' must be a mapping.")

        spread_bps = float(market_context.get("spread_bps", 0.0))
        max_spread_bps = float(market_policy.get("max_spread_bps", spread_bps))
        if spread_bps > max_spread_bps:
            violations.append(
                PolicyViolation(
                    code="spread_above_limit",
                    message="Market spread exceeds the allowed maximum.",
                    path="market_conditions.max_spread_bps",
                    details={"actual": spread_bps, "allowed": max_spread_bps},
                )
            )

        realized_volatility = float(market_context.get("realized_volatility", 0.0))
        max_volatility = float(market_policy.get("max_realized_volatility", realized_volatility))
        if realized_volatility > max_volatility:
            violations.append(
                PolicyViolation(
                    code="volatility_above_limit",
                    message="Market volatility exceeds the allowed maximum.",
                    path="market_conditions.max_realized_volatility",
                    details={"actual": realized_volatility, "allowed": max_volatility},
                )
            )

        stale_seconds = float(market_context.get("orderbook_stale_seconds", 0.0))
        max_stale_seconds = float(market_policy.get("deny_if_orderbook_stale_seconds_gt", stale_seconds))
        if stale_seconds > max_stale_seconds:
            violations.append(
                PolicyViolation(
                    code="orderbook_stale",
                    message="Order book data is too stale for execution.",
                    path="market_conditions.deny_if_orderbook_stale_seconds_gt",
                    details={"actual": stale_seconds, "allowed": max_stale_seconds},
                )
            )

        if market_policy.get("deny_if_market_data_unavailable", False):
            if not bool(market_context.get("market_data_available", True)):
                violations.append(
                    PolicyViolation(
                        code="market_data_unavailable",
                        message="Market data is unavailable.",
                        path="market_conditions.deny_if_market_data_unavailable",
                    )
                )

    def _evaluate_execution_controls(
        self,
        context: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block checks recent execution pressure and operational pacing.
        It takes: recent order counts, failed order counts, and risk-check status.
        It gives: violations when the system is pushing too hard or skipping pre-trade controls.
        """
        execution_policy = self.section("execution_controls")
        execution_context = context.get("execution", {})
        if not isinstance(execution_context, dict):
            raise ValidationError("Context field 'execution' must be a mapping.")

        recent_orders = int(execution_context.get("orders_in_last_minute", 0))
        max_orders = int(execution_policy.get("max_orders_per_minute", recent_orders))
        if recent_orders >= max_orders:
            violations.append(
                PolicyViolation(
                    code="orders_per_minute_exceeded",
                    message="Recent order rate exceeds the allowed maximum.",
                    path="execution_controls.max_orders_per_minute",
                    details={"actual": recent_orders, "allowed": max_orders},
                )
            )

        failed_orders = int(execution_context.get("failed_orders_count", 0))
        max_failed_orders = int(execution_policy.get("max_failed_orders_before_pause", failed_orders))
        if failed_orders >= max_failed_orders:
            violations.append(
                PolicyViolation(
                    code="failed_order_pause_threshold_reached",
                    message="Failed order threshold reached; execution should pause.",
                    path="execution_controls.max_failed_orders_before_pause",
                    details={"actual": failed_orders, "allowed": max_failed_orders},
                )
            )

        if execution_policy.get("require_pre_trade_risk_check", False):
            if not bool(execution_context.get("pre_trade_risk_check_passed", False)):
                violations.append(
                    PolicyViolation(
                        code="pre_trade_risk_check_required",
                        message="Pre-trade risk check is required and has not passed.",
                        path="execution_controls.require_pre_trade_risk_check",
                    )
                )

    def _evaluate_system_safety(
        self,
        context: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block checks infrastructure health dependencies.
        It takes: subsystem health from Kraken, persistence, inference, and system clock context.
        It gives: violations when degraded infrastructure should stop execution.
        """
        safety_policy = self.section("system_safety")
        system_context = context.get("system", {})
        if not isinstance(system_context, dict):
            raise ValidationError("Context field 'system' must be a mapping.")

        if safety_policy.get("deny_if_kraken_cli_unavailable", False):
            if not bool(system_context.get("kraken_cli_available", True)):
                violations.append(
                    PolicyViolation(
                        code="kraken_cli_unavailable",
                        message="Kraken CLI is unavailable.",
                        path="system_safety.deny_if_kraken_cli_unavailable",
                    )
                )

        if safety_policy.get("deny_if_checkpoint_store_unavailable", False):
            if not bool(system_context.get("checkpoint_store_available", True)):
                violations.append(
                    PolicyViolation(
                        code="checkpoint_store_unavailable",
                        message="Checkpoint store is unavailable.",
                        path="system_safety.deny_if_checkpoint_store_unavailable",
                    )
                )

        if safety_policy.get("deny_if_inference_backend_degraded", False):
            if bool(system_context.get("inference_backend_degraded", False)):
                violations.append(
                    PolicyViolation(
                        code="inference_backend_degraded",
                        message="Inference backend is degraded.",
                        path="system_safety.deny_if_inference_backend_degraded",
                    )
                )

        clock_skew_ms = int(system_context.get("clock_skew_ms", 0))
        max_clock_skew_ms = int(safety_policy.get("deny_if_clock_skew_ms_gt", clock_skew_ms))
        if clock_skew_ms > max_clock_skew_ms:
            violations.append(
                PolicyViolation(
                    code="clock_skew_above_limit",
                    message="Clock skew exceeds the allowed maximum.",
                    path="system_safety.deny_if_clock_skew_ms_gt",
                    details={"actual": clock_skew_ms, "allowed": max_clock_skew_ms},
                )
            )

    def _evaluate_deny_rules(
        self,
        context: dict[str, Any],
        violations: list[PolicyViolation],
    ) -> None:
        """
        This block evaluates explicit kill-switch style rules.
        It takes: boolean flags and balances from runtime context.
        It gives: hard-stop violations for conditions that should always deny execution.
        """
        deny_rules = self.section("deny_rules")
        flags = context.get("flags", {})
        if not isinstance(flags, dict):
            raise ValidationError("Context field 'flags' must be a mapping.")

        if deny_rules.get("deny_if_manual_kill_switch", False):
            if bool(flags.get("manual_kill_switch", False)):
                violations.append(
                    PolicyViolation(
                        code="manual_kill_switch_active",
                        message="Manual kill switch is active.",
                        path="deny_rules.deny_if_manual_kill_switch",
                    )
                )

        if deny_rules.get("deny_if_daily_loss_limit_breached", False):
            if bool(flags.get("daily_loss_limit_breached", False)):
                violations.append(
                    PolicyViolation(
                        code="daily_loss_limit_breached",
                        message="Daily loss limit has been breached.",
                        path="deny_rules.deny_if_daily_loss_limit_breached",
                    )
                )

        min_wallet_balance = float(deny_rules.get("deny_if_wallet_balance_below_usd", 0.0))
        wallet_balance = float(flags.get("wallet_balance_usd", min_wallet_balance))
        if wallet_balance < min_wallet_balance:
            violations.append(
                PolicyViolation(
                    code="wallet_balance_below_minimum",
                    message="Wallet balance is below the minimum required threshold.",
                    path="deny_rules.deny_if_wallet_balance_below_usd",
                    details={"actual": wallet_balance, "required": min_wallet_balance},
                )
            )

        if deny_rules.get("deny_if_untrusted_operator_context", False):
            if bool(flags.get("untrusted_operator_context", False)):
                violations.append(
                    PolicyViolation(
                        code="untrusted_operator_context",
                        message="Operator context is marked untrusted.",
                        path="deny_rules.deny_if_untrusted_operator_context",
                    )
                )

    def _resolve_action(self, violations: list[PolicyViolation]) -> PolicyAction:
        """
        This block converts raw violations into a final action.
        It takes: the list of violations plus enforcement config from guardrails.
        It gives: one deterministic allow/reduce/block result.
        """
        if not violations:
            return PolicyAction.ALLOW

        enforcement = self.section("enforcement")
        if enforcement.get("mode") == "hard-fail":
            return PolicyAction.BLOCK

        if enforcement.get("on_policy_violation") == "block":
            return PolicyAction.BLOCK

        if all(violation.reducible for violation in violations):
            return PolicyAction.REDUCE

        return PolicyAction.BLOCK


def load_policy(path: str | Path) -> GuardrailsPolicy:
    """
    This block is a small convenience loader.
    It takes: a path to guardrails.yaml.
    It gives: a validated GuardrailsPolicy instance for callers that want one-line loading.
    """
    return GuardrailsPolicy.from_file(path)
