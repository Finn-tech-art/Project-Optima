from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.validation.schemas import MarketRegime


# This block defines the callable shape expected for regime providers.
# It takes: the current graph state.
# It gives: either a MarketRegime, a valid regime string, or a dict containing regime metadata.
RegimeProvider = Callable[[RRAAgentState], MarketRegime | str | dict[str, Any]]


# This block configures how the regime node behaves.
# It takes: a regime provider callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class RegimeNodeConfig:
    regime_provider: RegimeProvider


# This block builds the regime graph node.
# It takes: a config object containing the regime provider.
# It gives: a LangGraph-compatible node function that enriches the state with regime context.
def build_regime_node(
    config: RegimeNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def regime_node(state: RRAAgentState) -> dict[str, Any]:
        # This block loads the current regime signal from the configured provider.
        # It takes: the current graph state and provider callback.
        # It gives: a raw regime payload to be normalized into graph metadata.
        raw_regime = config.regime_provider(state)

        # This block normalizes the provider output into a strict MarketRegime plus metadata.
        # It takes: the raw regime payload from the provider.
        # It gives: a canonical regime value and optional supporting metadata.
        regime, regime_metadata = _normalize_regime_payload(raw_regime)

        # This block updates graph metadata with the current regime context.
        # It takes: the existing state metadata and normalized regime details.
        # It gives: an updated metadata dict that downstream nodes can rely on.
        metadata = dict(state.metadata)
        metadata["regime"] = {
            "value": regime.value,
            "details": regime_metadata,
        }

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and normalized regime value.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(f"regime classified as {regime.value}")

        # This block returns the partial state update expected by LangGraph.
        # It takes: the normalized regime metadata and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "metadata": metadata,
            "messages": messages,
        }

    return regime_node


# This block normalizes provider output into a strict MarketRegime and metadata bundle.
# It takes: a MarketRegime, regime string, or a dict containing regime details.
# It gives: one canonical MarketRegime and an optional supporting metadata dictionary.
def _normalize_regime_payload(
    raw_regime: MarketRegime | str | dict[str, Any],
) -> tuple[MarketRegime, dict[str, Any]]:
    if isinstance(raw_regime, MarketRegime):
        return raw_regime, {}

    if isinstance(raw_regime, str):
        return _coerce_regime(raw_regime), {}

    if isinstance(raw_regime, dict):
        if "value" not in raw_regime:
            raise ValueError("Regime provider dict output must include a 'value' field.")

        regime = _coerce_regime(raw_regime["value"])
        metadata = {
            key: value
            for key, value in raw_regime.items()
            if key != "value"
        }
        return regime, metadata

    raise ValueError("Unsupported regime provider output type.")


# This block coerces flexible regime strings into the strict MarketRegime enum.
# It takes: a raw regime string from the provider.
# It gives: a canonical MarketRegime value or raises if unsupported.
def _coerce_regime(value: Any) -> MarketRegime:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "meanreversion": "mean_reversion",
        "mean_reverting": "mean_reversion",
        "trending": "trend",
        "trend_following": "trend",
        "break_out": "breakout",
    }
    normalized = aliases.get(normalized, normalized)

    try:
        return MarketRegime(normalized)
    except ValueError as error:
        raise ValueError(f"Unsupported market regime: {value}") from error
