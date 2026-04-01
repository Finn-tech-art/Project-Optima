from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.data.models import MarketSnapshotRecord


# This block defines the callable shape expected for market snapshot providers.
# It takes: the current graph state.
# It gives: the latest market snapshot record for the tracked symbol.
MarketSnapshotProvider = Callable[[RRAAgentState], MarketSnapshotRecord]


# This block configures how the market-context node behaves.
# It takes: a snapshot provider and optional default execution metadata.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class MarketContextNodeConfig:
    snapshot_provider: MarketSnapshotProvider


# This block builds the market-context graph node.
# It takes: a config object containing the market snapshot provider.
# It gives: a LangGraph-compatible node function that enriches the state with market context.
def build_market_context_node(
    config: MarketContextNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def market_context_node(state: RRAAgentState) -> dict[str, Any]:
        # This block loads the latest market snapshot for the current symbol.
        # It takes: the current graph state and the configured snapshot provider.
        # It gives: a typed MarketSnapshotRecord for downstream policy and execution logic.
        snapshot = config.snapshot_provider(state)

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and the newly loaded market snapshot.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(
            f"market_context loaded for {snapshot.symbol} with spread_bps={snapshot.spread_bps}"
        )

        runtime_context = _read_runtime_context_overrides(snapshot)

        # This block prepares default execution metadata expected by later validation and policy code.
        # It takes: any existing metadata already present on the state.
        # It gives: a metadata dict with the execution, system, flags, and positions keys initialized.
        metadata = dict(state.metadata)
        metadata["execution"] = _merge_metadata_section(
            existing=metadata.get("execution"),
            defaults={
                "orders_in_last_minute": 0,
                "failed_orders_count": 0,
                "pre_trade_risk_check_passed": False,
            },
            overrides=runtime_context.get("execution"),
        )
        metadata["system"] = _merge_metadata_section(
            existing=metadata.get("system"),
            defaults={
                "kraken_cli_available": False,
                "checkpoint_store_available": True,
                "inference_backend_degraded": False,
                "clock_skew_ms": 0,
            },
            overrides=runtime_context.get("system"),
        )
        metadata["flags"] = _merge_metadata_section(
            existing=metadata.get("flags"),
            defaults={
                "manual_kill_switch": False,
                "daily_loss_limit_breached": False,
                "wallet_balance_usd": 0.0,
                "untrusted_operator_context": False,
            },
            overrides=runtime_context.get("flags"),
        )
        metadata["positions"] = _merge_metadata_section(
            existing=metadata.get("positions"),
            defaults={
                "total_open_exposure_bps": 0.0,
                "asset_open_exposure_bps": 0.0,
                "open_positions_count": 0,
            },
            overrides=runtime_context.get("positions"),
        )

        # This block returns the partial state update expected by LangGraph.
        # It takes: the loaded market snapshot and normalized metadata.
        # It gives: a deterministic update to the graph state.
        return {
            "market_snapshot": snapshot,
            "messages": messages,
            "metadata": metadata,
        }

    return market_context_node


# This block reads any runtime-context overrides embedded in the market snapshot payload.
# It takes: a market snapshot record returned by the configured provider.
# It gives: a dict of optional metadata-section overrides for policy/runtime context.
def _read_runtime_context_overrides(snapshot: MarketSnapshotRecord) -> dict[str, Any]:
    raw_payload = snapshot.raw_payload
    if not isinstance(raw_payload, dict):
        return {}

    runtime_context = raw_payload.get("runtime_context", {})
    if isinstance(runtime_context, dict):
        return runtime_context

    return {}


# This block merges default metadata, existing state metadata, and provider overrides.
# It takes: an existing metadata section, section defaults, and optional override fields.
# It gives: one normalized dictionary suitable for graph-state updates.
def _merge_metadata_section(
    *,
    existing: Any,
    defaults: dict[str, Any],
    overrides: Any,
) -> dict[str, Any]:
    merged = dict(defaults)

    if isinstance(existing, dict):
        merged.update(existing)

    if isinstance(overrides, dict):
        merged.update(overrides)

    return merged
