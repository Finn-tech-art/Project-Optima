from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


from backend.agent.state import RRAAgentState


# This block defines the callable shape expected for portfolio context providers.
# It takes: the current graph state.
# It gives: a dictionary containing position and account state for policy evaluation.
PortfolioContextProvider = Callable[[RRAAgentState], dict[str, Any]]


# This block configures how the portfolio node behaves.
# It takes: a portfolio context provider callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class PortfolioNodeConfig:
    portfolio_provider: PortfolioContextProvider


# This block builds the portfolio graph node.
# It takes: a config object containing the portfolio context provider.
# It gives: a LangGraph-compatible node function that enriches the state with account context.
def build_portfolio_node(
    config: PortfolioNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def portfolio_node(state: RRAAgentState) -> dict[str, Any]:
        # This block loads the latest portfolio and account context.
        # It takes: the current graph state and configured provider callback.
        # It gives: raw portfolio facts that will be normalized for policy evaluation.
        portfolio_context = config.portfolio_provider(state)
        if not isinstance(portfolio_context, dict):
            raise ValueError("portfolio_provider must return a dictionary")

        # This block reads or initializes the graph metadata map.
        # It takes: the existing state metadata.
        # It gives: a mutable metadata copy that can be safely updated.
        metadata = dict(state.metadata)

        # This block normalizes position-related fields used by the policy engine.
        # It takes: the raw provider output.
        # It gives: the exact 'positions' shape expected by security.policy.
        metadata["positions"] = {
            "total_open_exposure_bps": float(
                portfolio_context.get("total_open_exposure_bps", 0.0)
            ),
            "asset_open_exposure_bps": float(
                portfolio_context.get("asset_open_exposure_bps", 0.0)
            ),
            "open_positions_count": int(
                portfolio_context.get("open_positions_count", 0)
            ),
        }

        # This block merges balance and operator-level safety flags.
        # It takes: the raw provider output plus any pre-existing flags.
        # It gives: the exact 'flags' shape expected by security.policy.
        existing_flags = metadata.get("flags", {})
        if not isinstance(existing_flags, dict):
            existing_flags = {}

        metadata["flags"] = {
            "manual_kill_switch": bool(
                portfolio_context.get(
                    "manual_kill_switch",
                    existing_flags.get("manual_kill_switch", False),
                )
            ),
            "daily_loss_limit_breached": bool(
                portfolio_context.get(
                    "daily_loss_limit_breached",
                    existing_flags.get("daily_loss_limit_breached", False),
                )
            ),
            "wallet_balance_usd": float(
                portfolio_context.get(
                    "wallet_balance_usd",
                    existing_flags.get("wallet_balance_usd", 0.0),
                )
            ),
            "untrusted_operator_context": bool(
                portfolio_context.get(
                    "untrusted_operator_context",
                    existing_flags.get("untrusted_operator_context", False),
                )
            ),
        }

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and the normalized portfolio context.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(
            "portfolio context loaded "
            f"open_positions={metadata['positions']['open_positions_count']} "
            f"wallet_balance_usd={metadata['flags']['wallet_balance_usd']}"
        )

        # This block returns the partial state update expected by LangGraph.
        # It takes: the normalized metadata and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "metadata": metadata,
            "messages": messages,
        }

    return portfolio_node
