from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.agent.graph import RRAGraphConfig, RRAGraphNodeConfigs, compile_rra_graph
from backend.agent.nodes.execution import ExecutionNodeConfig, ExecutionProvider
from backend.agent.nodes.guardrail import GuardrailNodeConfig
from backend.agent.nodes.intent import IntentNodeConfig, IntentPayloadEnricher
from backend.agent.nodes.market_context import (
    MarketContextNodeConfig,
    MarketSnapshotProvider,
)
from backend.agent.nodes.memory import MemoryNodeConfig, MemoryProvider
from backend.agent.nodes.portfolio import PortfolioContextProvider, PortfolioNodeConfig
from backend.agent.nodes.regime import RegimeNodeConfig, RegimeProvider
from backend.agent.nodes.sentiment import SentimentInferenceProvider, SentimentNodeConfig
from backend.agent.nodes.trust_context import (
    AgentAddressResolver,
    AttestationProvider,
    LiveTrustContextNodeConfig,
    TrustContextNodeConfig,
    TrustSnapshotProvider,
)
from backend.agent.nodes.validator import ValidatorNodeConfig, ValidatorPayloadProvider
from backend.config.settings import AppSettings, get_settings
from backend.core.exceptions import ConfigurationError
from backend.data.models import MarketSnapshotRecord
from backend.integrations.groq_client import GroqClient
from security.erc8004_registry import ERC8004RegistryClient, RegistryMethodMap
from security.policy import GuardrailsPolicy, load_policy
from security.secrets import AppSecrets, load_secrets


# This block stores the externally supplied providers needed to assemble the graph runtime.
# It takes: callbacks for market, portfolio, regime, memory, execution, and trust resolution.
# It gives: one normalized bundle of runtime dependencies for graph assembly.
@dataclass(slots=True, frozen=True)
class RRARuntimeProviders:
    market_snapshot_provider: MarketSnapshotProvider
    portfolio_provider: PortfolioContextProvider
    regime_provider: RegimeProvider
    memory_provider: MemoryProvider
    execution_provider: ExecutionProvider
    agent_address_resolver: AgentAddressResolver | None = None
    trust_snapshot_provider: TrustSnapshotProvider | None = None
    attestation_provider: AttestationProvider | None = None
    validator_payload_provider: ValidatorPayloadProvider | None = None
    intent_payload_enricher: IntentPayloadEnricher | None = None
    registry_methods: RegistryMethodMap | None = None


# This block stores the assembled runtime services and compiled graph.
# It takes: loaded settings, secrets, policy, optional live registry client, and the compiled graph.
# It gives: one object that higher-level entrypoints can use to run the RRA workflow.
@dataclass(slots=True)
class RRARuntime:
    settings: AppSettings
    secrets: AppSecrets
    policy: GuardrailsPolicy
    graph: Any
    groq_client: GroqClient
    registry_client: ERC8004RegistryClient | None = None


# This block creates the Project-Optima runtime and compiles the LangGraph workflow.
# It takes: externally supplied providers plus optional settings and policy paths.
# It gives: one assembled runtime object with services and compiled graph ready to invoke.
def create_rra_runtime(
    *,
    providers: RRARuntimeProviders,
    settings: AppSettings | None = None,
    env_file: str | Path | None = Path(".env"),
    policy_path: str | Path = Path("security/guardrails.yaml"),
) -> RRARuntime:
    resolved_settings = settings or get_settings()

    # This block loads secrets for the subsystems we can initialize at runtime startup.
    # It takes: the configured env file plus the required subsystem flags.
    # It gives: a validated AppSecrets object for Groq and optional live registry access.
    secrets = load_secrets(
        env_file=env_file,
        require_groq=True,
        require_langgraph=False,
        require_erc8004=providers.registry_methods is not None,
    )

    # This block loads the deterministic policy engine.
    # It takes: the configured guardrails YAML path.
    # It gives: a validated GuardrailsPolicy instance for the guardrail node.
    policy = load_policy(policy_path)

    # This block initializes the Groq client used by the sentiment node.
    # It takes: loaded secrets plus the configured default and fallback model names.
    # It gives: a reusable inference client for structured signal generation.
    groq_client = GroqClient(
        secrets=secrets,
        default_model=resolved_settings.groq_model,
        fallback_model=resolved_settings.groq_fast_model,
    )

    # This block builds the graph node configuration bundle.
    # It takes: the supplied runtime providers plus initialized services.
    # It gives: one RRAGraphNodeConfigs object ready for graph compilation.
    node_configs = RRAGraphNodeConfigs(
        market_context=MarketContextNodeConfig(
            snapshot_provider=providers.market_snapshot_provider
        ),
        portfolio=PortfolioNodeConfig(
            portfolio_provider=providers.portfolio_provider
        ),
        regime=RegimeNodeConfig(
            regime_provider=providers.regime_provider
        ),
        sentiment=SentimentNodeConfig(
            inference_provider=_build_sentiment_inference_provider(
                groq_client=groq_client,
                settings=resolved_settings,
            )
        ),
        memory=MemoryNodeConfig(
            memory_provider=providers.memory_provider
        ),
        guardrail=GuardrailNodeConfig(
            policy=policy
        ),
        execution=ExecutionNodeConfig(
            execution_provider=providers.execution_provider
        ),
        intent=IntentNodeConfig(
            payload_enricher=providers.intent_payload_enricher
        ),
        validator=ValidatorNodeConfig(
            payload_provider=providers.validator_payload_provider
        ),
    )

    registry_client: ERC8004RegistryClient | None = None

    # This block selects either the live registry-backed trust node or a generic trust provider.
    # It takes: the supplied provider bundle and optional registry method map.
    # It gives: exactly one trust-context configuration path for the graph.
    if providers.registry_methods is not None:
        if providers.agent_address_resolver is None:
            raise ConfigurationError(
                "Live ERC-8004 trust wiring requires an agent_address_resolver."
            )

        registry_client = _build_registry_client(
            settings=resolved_settings,
            methods=providers.registry_methods,
        )
        node_configs.live_trust_context = LiveTrustContextNodeConfig(
            registry_client=registry_client,
            agent_address_resolver=providers.agent_address_resolver,
            attestation_provider=providers.attestation_provider,
        )
    elif providers.trust_snapshot_provider is not None:
        node_configs.trust_context = TrustContextNodeConfig(
            snapshot_provider=providers.trust_snapshot_provider
        )
    else:
        raise ConfigurationError(
            "Runtime requires either trust_snapshot_provider or registry_methods for live trust."
        )

    # This block compiles the graph with the configured checkpoint path.
    # It takes: the node configs and LangGraph checkpoint settings.
    # It gives: a compiled graph object ready for invoke or stream execution.
    graph = compile_rra_graph(
        node_configs=node_configs,
        config=RRAGraphConfig(
            checkpoint_db_path=str(resolved_settings.checkpoint_db_path),
        ),
    )

    return RRARuntime(
        settings=resolved_settings,
        secrets=secrets,
        policy=policy,
        graph=graph,
        groq_client=groq_client,
        registry_client=registry_client,
    )


# This block builds the live ERC-8004 registry client from runtime settings.
# It takes: loaded settings and the resolved registry method map.
# It gives: a validated ERC8004RegistryClient ready for live trust lookups.
def _build_registry_client(
    *,
    settings: AppSettings,
    methods: RegistryMethodMap,
) -> ERC8004RegistryClient:
    erc8004_config = settings.erc8004_config()
    rpc_url = erc8004_config.get("rpc_url")
    registry_address = erc8004_config.get("registry_address")
    abi_path = erc8004_config.get("abi_path")
    chain_id = erc8004_config.get("chain_id")

    if not isinstance(rpc_url, str) or not rpc_url.strip():
        raise ConfigurationError("ERC8004_RPC_URL must be configured for live trust.")

    if not isinstance(registry_address, str) or not registry_address.strip():
        raise ConfigurationError(
            "ERC8004_REGISTRY_ADDRESS must be configured for live trust."
        )

    if not isinstance(abi_path, str) or not abi_path.strip():
        raise ConfigurationError("ERC8004_ABI_PATH must be configured for live trust.")

    return ERC8004RegistryClient.from_abi_file(
        rpc_url=rpc_url,
        contract_address=registry_address,
        abi_path=abi_path,
        methods=methods,
        chain_id=int(chain_id),
    )


# This block builds the sentiment inference provider used by the sentiment node.
# It takes: the initialized Groq client and runtime settings.
# It gives: a node-compatible inference callback that returns raw output and model metadata.
def _build_sentiment_inference_provider(
    *,
    groq_client: GroqClient,
    settings: AppSettings,
) -> SentimentInferenceProvider:
    def inference_provider(system_prompt: str, user_prompt: str) -> dict[str, Any]:
        # This block sends the sentiment prompt to Groq using the configured model.
        # It takes: the wrapped system and user prompts from the sentiment node.
        # It gives: raw model output plus the resolved model name for graph state.
        response = groq_client.infer(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=settings.groq_model,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
        )
        return {
            "raw_output": response.content,
            "model_name": response.model,
            "raw_response": response.raw,
        }

    return inference_provider


# This block builds a simple development market snapshot provider.
# It takes: bid, ask, spread, and volatility values for a symbol.
# It gives: a reusable provider callback that returns one synthetic MarketSnapshotRecord.
def build_static_market_snapshot_provider(
    *,
    symbol: str,
    bid: float,
    ask: float,
    spread_bps: float,
    realized_volatility: float,
    orderbook_stale_seconds: float = 0.0,
    market_data_available: bool = True,
) -> MarketSnapshotProvider:
    mid = (bid + ask) / 2 if bid and ask else None

    def provider(state) -> MarketSnapshotRecord:
        # This block returns a deterministic market snapshot for development wiring.
        # It takes: the current graph state.
        # It gives: one static MarketSnapshotRecord for the configured symbol.
        return MarketSnapshotRecord(
            symbol=symbol or state.symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_bps=spread_bps,
            realized_volatility=realized_volatility,
            orderbook_stale_seconds=orderbook_stale_seconds,
            market_data_available=market_data_available,
            raw_payload={},
        )

    return provider
