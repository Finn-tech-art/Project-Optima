from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.data.models import MemoryDocument


# This block defines the callable shape expected for memory providers.
# It takes: the current graph state.
# It gives: a list of memory documents or memory-like dictionaries relevant to the current symbol/context.
MemoryProvider = Callable[[RRAAgentState], list[MemoryDocument | dict[str, Any]]]


# This block configures how the memory node behaves.
# It takes: a memory provider callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class MemoryNodeConfig:
    memory_provider: MemoryProvider


# This block builds the memory graph node.
# It takes: a config object containing the memory provider.
# It gives: a LangGraph-compatible node function that enriches the state with retrieved memory.
def build_memory_node(
    config: MemoryNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def memory_node(state: RRAAgentState) -> dict[str, Any]:
        # This block retrieves historical memory relevant to the current graph state.
        # It takes: the current state and configured memory provider callback.
        # It gives: a list of memory documents or raw memory payloads.
        raw_memories = config.memory_provider(state)
        if not isinstance(raw_memories, list):
            raise ValueError("memory_provider must return a list")

        # This block normalizes memory items into a consistent serializable representation.
        # It takes: the raw memory items returned by the provider.
        # It gives: a list of normalized memory dictionaries for later prompt and intent use.
        normalized_memories = [
            _normalize_memory_item(item)
            for item in raw_memories
        ]

        # This block updates graph metadata with the retrieved memory context.
        # It takes: the existing state metadata and normalized memory list.
        # It gives: a metadata dict that downstream nodes can consume deterministically.
        metadata = dict(state.metadata)
        metadata["memory"] = {
            "count": len(normalized_memories),
            "items": normalized_memories,
        }

        # This block appends a human-readable trace message for observability.
        # It takes: the existing message history and memory count.
        # It gives: an updated message timeline for checkpoints and debugging.
        messages = list(state.messages)
        messages.append(f"memory retrieved count={len(normalized_memories)}")

        # This block returns the partial state update expected by LangGraph.
        # It takes: the normalized memory metadata and message timeline.
        # It gives: a deterministic update to the graph state.
        return {
            "metadata": metadata,
            "messages": messages,
        }

    return memory_node


# This block normalizes a memory item into a serializable dictionary.
# It takes: either a MemoryDocument object or a memory-like dictionary.
# It gives: a normalized dictionary suitable for prompts, logs, and checkpoints.
def _normalize_memory_item(item: MemoryDocument | dict[str, Any]) -> dict[str, Any]:
    if isinstance(item, MemoryDocument):
        return {
            "id": item.id,
            "collection": item.collection,
            "text": item.text,
            "symbol": item.symbol,
            "trace_id": item.trace_id,
            "tags": list(item.tags),
            "metadata": dict(item.metadata),
            "source": item.source.value,
            "created_at": item.created_at.isoformat(),
        }

    if isinstance(item, dict):
        return dict(item)

    raise ValueError("Unsupported memory item type returned by memory_provider")
