from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from backend.agent.state import RRAAgentState
from backend.data.models import TrustSnapshotRecord
from security.erc8004_registry import ERC8004RegistryClient
from security.tee_attestation import TEEAttestationResult


# This block defines the callable shape expected for trust snapshot providers.
# It takes: the current graph state.
# It gives: the latest trust snapshot record for the active agent or operator context.
TrustSnapshotProvider = Callable[[RRAAgentState], TrustSnapshotRecord]

# This block defines the callable shape expected for resolving the active agent address.
# It takes: the current graph state.
# It gives: the wallet address whose trust state should be checked.
AgentAddressResolver = Callable[[RRAAgentState], str]

# This block defines the callable shape expected for optional attestation providers.
# It takes: the current graph state.
# It gives: a validated TEE attestation result or None when attestation is unavailable.
AttestationProvider = Callable[[RRAAgentState], TEEAttestationResult | None]


# This block configures the generic trust-context node for mock/testing flows.
# It takes: a trust snapshot provider callback.
# It gives: one reusable node configuration object.
@dataclass(slots=True, frozen=True)
class TrustContextNodeConfig:
    snapshot_provider: TrustSnapshotProvider


# This block configures the live trust-context node for real ERC-8004 and TEE integration.
# It takes: a registry client, address resolver, and optional attestation provider.
# It gives: one reusable config object for production-grade trust loading.
@dataclass(slots=True, frozen=True)
class LiveTrustContextNodeConfig:
    registry_client: ERC8004RegistryClient
    agent_address_resolver: AgentAddressResolver
    attestation_provider: AttestationProvider | None = None


# This block builds the generic trust-context graph node.
# It takes: a config object containing the trust snapshot provider.
# It gives: a LangGraph-compatible node function that enriches the state with trust context.
def build_trust_context_node(
    config: TrustContextNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def trust_context_node(state: RRAAgentState) -> dict[str, Any]:
        snapshot = config.snapshot_provider(state)
        return _build_trust_state_update(state=state, snapshot=snapshot)

    return trust_context_node


# This block builds the live trust-context graph node using the real registry client.
# It takes: a config object containing live trust-layer dependencies.
# It gives: a LangGraph-compatible node function that loads ERC-8004 and optional TEE context.
def build_live_trust_context_node(
    config: LiveTrustContextNodeConfig,
) -> Callable[[RRAAgentState], dict[str, Any]]:
    def trust_context_node(state: RRAAgentState) -> dict[str, Any]:
        # This block resolves the active agent address from graph state.
        # It takes: the current state plus the configured address resolver.
        # It gives: the wallet address to query against the ERC-8004 registry.
        agent_address = config.agent_address_resolver(state)

        # This block loads the on-chain trust context from the live registry.
        # It takes: the resolved agent address and configured registry client.
        # It gives: normalized registration, trust score, and reachability data.
        registry_context = config.registry_client.build_trust_context(agent_address)

        # This block loads the optional TEE attestation result when configured.
        # It takes: the current graph state and attestation provider callback.
        # It gives: attestation metadata that can be merged into the trust snapshot.
        attestation_result = (
            config.attestation_provider(state)
            if config.attestation_provider is not None
            else None
        )

        # This block merges registry and attestation data into the durable trust record.
        # It takes: normalized registry output plus optional attestation evidence.
        # It gives: one TrustSnapshotRecord for policy, validation, and checkpoint replay.
        snapshot = TrustSnapshotRecord(
            agent_address=registry_context.agent_address,
            trust_score=registry_context.trust_score,
            erc8004_registered=registry_context.erc8004_registered,
            valid_attestation=(
                attestation_result.valid_attestation
                if attestation_result is not None
                else registry_context.valid_attestation
            ),
            registry_reachable=registry_context.registry_reachable,
            attestation_age_seconds=(
                attestation_result.age_seconds or 0
                if attestation_result is not None
                else 0
            ),
            attested=(
                attestation_result.attested
                if attestation_result is not None
                else registry_context.valid_attestation
            ),
            tee_measurement=(
                attestation_result.measurement
                if attestation_result is not None
                else None
            ),
            tee_enclave_id=(
                attestation_result.enclave_id
                if attestation_result is not None
                else None
            ),
            raw_registry=registry_context.raw,
            raw_attestation=(
                attestation_result.raw
                if attestation_result is not None
                else {}
            ),
        )

        return _build_trust_state_update(state=state, snapshot=snapshot)

    return trust_context_node


# This block converts a trust snapshot into a LangGraph state update.
# It takes: the current state and a normalized trust snapshot record.
# It gives: the partial state update expected by LangGraph.
def _build_trust_state_update(
    *,
    state: RRAAgentState,
    snapshot: TrustSnapshotRecord,
) -> dict[str, Any]:
    # This block appends a human-readable trace message for observability.
    # It takes: the existing message history and the loaded trust snapshot.
    # It gives: an updated message timeline for checkpoints and debugging.
    messages = list(state.messages)
    messages.append(
        "trust_context loaded "
        f"registered={snapshot.erc8004_registered} "
        f"attested={snapshot.valid_attestation} "
        f"trust_score={snapshot.trust_score}"
    )

    # This block merges trust-derived metadata into the existing state metadata.
    # It takes: any existing metadata plus trust snapshot details.
    # It gives: a metadata dict that preserves audit context for later nodes.
    metadata = dict(state.metadata)
    metadata["trust"] = {
        "agent_address": snapshot.agent_address,
        "trust_score": snapshot.trust_score,
        "erc8004_registered": snapshot.erc8004_registered,
        "valid_attestation": snapshot.valid_attestation,
        "registry_reachable": snapshot.registry_reachable,
        "attestation_age_seconds": snapshot.attestation_age_seconds,
        "attested": snapshot.attested,
        "tee_measurement": snapshot.tee_measurement,
        "tee_enclave_id": snapshot.tee_enclave_id,
    }

    # This block initializes system defaults if earlier nodes have not already done so.
    # It takes: the existing metadata map.
    # It gives: a stable system metadata shape expected by later validation and policy code.
    metadata.setdefault(
        "system",
        {
            "kraken_cli_available": False,
            "checkpoint_store_available": True,
            "inference_backend_degraded": False,
            "clock_skew_ms": 0,
        },
    )

    # This block returns the partial state update expected by LangGraph.
    # It takes: the loaded trust snapshot and merged metadata.
    # It gives: a deterministic update to the graph state.
    return {
        "trust_snapshot": snapshot,
        "messages": messages,
        "metadata": metadata,
    }
