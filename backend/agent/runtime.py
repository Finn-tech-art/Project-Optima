from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

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
from backend.data.models import (
    ExecutionRecord,
    ExecutionStatus,
    MarketSnapshotRecord,
    RecordSource,
)
from backend.integrations.groq_client import GroqClient
from backend.integrations.kraken_cli import (
    KrakenCLIClient,
    KrakenCLIConfig,
    KrakenCLIExecutionMode,
)
from backend.integrations.kraken_rest import KrakenRESTClient, KrakenRESTConfig
from security.erc8004_registry import ERC8004RegistryClient, RegistryMethodMap
from security.policy import GuardrailsPolicy, load_policy
from security.secrets import AppSecrets, load_secrets


# This block defines the default method map for the current ERC-8004 Identity Registry ABI.
# It takes: no runtime input.
# It gives: the ABI-aligned method names for live registration and metadata lookups.
DEFAULT_IDENTITY_REGISTRY_METHODS = RegistryMethodMap(
    registration_check="balanceOf",
    wallet_lookup="getAgentWallet",
    metadata_lookup="getMetadata",
)


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
    sentiment_inference_provider: SentimentInferenceProvider | None = None
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
    groq_client: GroqClient | None = None
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
    resolved_registry_methods = _resolve_registry_methods(providers)

    # This block loads secrets for the subsystems we can initialize at runtime startup.
    # It takes: the configured env file plus the required subsystem flags.
    # It gives: a validated AppSecrets object for Groq and optional live registry access.
    secrets = load_secrets(
        env_file=env_file,
        require_groq=providers.sentiment_inference_provider is None,
        require_langgraph=False,
        require_erc8004=resolved_registry_methods is not None,
    )

    # This block loads the deterministic policy engine.
    # It takes: the configured guardrails YAML path.
    # It gives: a validated GuardrailsPolicy instance for the guardrail node.
    policy = load_policy(policy_path)

    # This block initializes the Groq client used by the sentiment node.
    # It takes: loaded secrets plus the configured default and fallback model names.
    # It gives: a reusable inference client for structured signal generation.
    groq_client: GroqClient | None = None
    sentiment_inference_provider = providers.sentiment_inference_provider
    if sentiment_inference_provider is None:
        groq_client = GroqClient(
            secrets=secrets,
            default_model=resolved_settings.groq_model,
            fallback_model=resolved_settings.groq_fast_model,
        )
        sentiment_inference_provider = _build_sentiment_inference_provider(
            groq_client=groq_client,
            settings=resolved_settings,
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
            inference_provider=sentiment_inference_provider
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
    if resolved_registry_methods is not None:
        if providers.agent_address_resolver is None:
            raise ConfigurationError(
                "Live ERC-8004 trust wiring requires an agent_address_resolver."
            )

        registry_client = _build_registry_client(
            settings=resolved_settings,
            methods=resolved_registry_methods,
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


# This block resolves which registry method map the runtime should use.
# It takes: the externally supplied runtime providers.
# It gives: an explicit method map for live identity-registry mode, or None for provider-backed trust.
def _resolve_registry_methods(
    providers: RRARuntimeProviders,
) -> RegistryMethodMap | None:
    if providers.registry_methods is not None:
        return providers.registry_methods

    if providers.agent_address_resolver is not None:
        return DEFAULT_IDENTITY_REGISTRY_METHODS

    return None


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


# This block builds a live Kraken REST client from runtime settings and secrets.
# It takes: optional settings and env-file inputs.
# It gives: an authenticated Kraken REST client for market, balance, and execution flows.
def build_kraken_rest_client(
    *,
    settings: AppSettings | None = None,
    secrets: AppSecrets | None = None,
    env_file: str | Path | None = Path(".env"),
) -> KrakenRESTClient:
    resolved_settings = settings or get_settings()
    resolved_secrets = secrets or load_secrets(
        env_file=env_file,
        require_kraken_api=True,
    )
    resolved_secrets.require_kraken_api()

    return KrakenRESTClient(
        KrakenRESTConfig(
            api_key=resolved_secrets.kraken_api_key or "",
            api_secret=resolved_secrets.kraken_api_secret or "",
            base_url=resolved_settings.kraken_rest_url,
            timeout_seconds=resolved_settings.kraken_timeout_seconds,
            validate_only=resolved_settings.kraken_validate_only,
        )
    )


# This block builds a live market snapshot provider backed by Kraken ticker data.
# It takes: an authenticated Kraken REST client.
# It gives: a graph-compatible provider that loads bid/ask and market-health context.
def build_kraken_market_snapshot_provider(
    *,
    client: KrakenRESTClient,
) -> MarketSnapshotProvider:
    def provider(state) -> MarketSnapshotRecord:
        ticker = client.get_ticker(state.symbol)
        mid = (ticker.bid + ticker.ask) / 2
        spread_bps = ((ticker.ask - ticker.bid) / mid) * 10_000 if mid > 0 else 0.0

        return MarketSnapshotRecord(
            symbol=state.symbol,
            bid=ticker.bid,
            ask=ticker.ask,
            mid=mid,
            spread_bps=spread_bps,
            realized_volatility=0.0,
            orderbook_stale_seconds=0.0,
            market_data_available=True,
            raw_payload={
                "pair": ticker.pair,
                "altname": ticker.altname,
                "ticker": ticker.raw,
                "runtime_context": {
                    "system": {
                        "kraken_cli_available": client.ping(),
                        "checkpoint_store_available": True,
                        "inference_backend_degraded": False,
                        "clock_skew_ms": 0,
                    },
                    "execution": {
                        "orders_in_last_minute": 0,
                        "failed_orders_count": 0,
                        "pre_trade_risk_check_passed": True,
                    },
                },
            },
        )

    return provider


# This block builds a live portfolio provider backed by Kraken balances and open orders.
# It takes: an authenticated Kraken REST client.
# It gives: a graph-compatible provider that approximates exposure and wallet state.
def build_kraken_portfolio_provider(
    *,
    client: KrakenRESTClient,
) -> PortfolioContextProvider:
    def provider(state) -> dict[str, Any]:
        balances = client.get_balances()
        open_orders = client.get_open_orders()
        ticker = client.get_ticker(state.symbol)
        mid = (ticker.bid + ticker.ask) / 2

        usd_balance = _extract_usd_balance(balances)
        base_amount = _extract_symbol_base_amount(balances, state.symbol)
        asset_value_usd = base_amount * mid
        total_equity_usd = usd_balance + asset_value_usd
        total_open_exposure_bps = (
            (asset_value_usd / total_equity_usd) * 10_000 if total_equity_usd > 0 else 0.0
        )

        non_cash_positions = sum(
            1
            for asset, amount in balances.items()
            if _is_non_cash_asset(asset) and _safe_float(amount) > 0
        )

        return {
            "total_open_exposure_bps": total_open_exposure_bps,
            "asset_open_exposure_bps": total_open_exposure_bps,
            "open_positions_count": non_cash_positions or len(open_orders),
            "wallet_balance_usd": usd_balance,
            "manual_kill_switch": False,
            "daily_loss_limit_breached": False,
            "untrusted_operator_context": False,
        }

    return provider


# This block builds a live execution provider backed by Kraken AddOrder.
# It takes: an authenticated Kraken REST client and a live/dry-run toggle.
# It gives: a graph-compatible provider that submits validated orders to Kraken.
def build_kraken_execution_provider(
    *,
    client: KrakenRESTClient,
    validate_only: bool | None = None,
) -> ExecutionProvider:
    def provider(execution_request: dict[str, Any], state) -> dict[str, Any]:
        if state.trade_intent_record is None:
            raise ConfigurationError(
                "Kraken execution provider requires state.trade_intent_record."
            )

        ticker = client.get_ticker(state.trade_intent_record.symbol)
        reference_price = ticker.last or ((ticker.bid + ticker.ask) / 2)
        if reference_price <= 0:
            raise ConfigurationError("Kraken ticker reference price must be positive.")

        requested_notional = float(state.trade_intent_record.notional_usd)
        volume = requested_notional / reference_price
        order_result = client.add_order(
            symbol=state.trade_intent_record.symbol,
            side=state.trade_intent_record.side.value,
            order_type=state.trade_intent_record.order_type.value,
            volume=volume,
            price=(reference_price if state.trade_intent_record.order_type.value == "limit" else None),
            time_in_force=state.trade_intent_record.time_in_force,
            validate_only=validate_only,
        )

        txids = order_result.get("txid", [])
        if not isinstance(txids, list):
            txids = []

        validation_mode = (
            client.config.validate_only if validate_only is None else validate_only
        )
        status = (
            ExecutionStatus.PENDING.value
            if validation_mode
            else ExecutionStatus.SUBMITTED.value
        )

        execution_record = ExecutionRecord(
            trade_intent_id=state.trade_intent_record.id,
            policy_decision_id=(
                state.policy.record.id if state.policy.record is not None else None
            ),
            symbol=state.trade_intent_record.symbol,
            side=state.trade_intent_record.side,
            order_type=state.trade_intent_record.order_type,
            status=ExecutionStatus(status),
            exchange_order_id=(txids[0] if txids else f"kraken-{uuid4().hex[:12]}"),
            requested_notional_usd=requested_notional,
            executed_notional_usd=(None if validation_mode else requested_notional),
            average_fill_price=(None if validation_mode else reference_price),
            raw_payload={
                "validation_mode": validation_mode,
                "request": execution_request,
                "result": order_result,
                "reference_price": reference_price,
                "volume": volume,
            },
            source=RecordSource.EXECUTION,
        )

        return {
            "status": status,
            "records": [execution_record],
            "orders_in_last_minute": 1,
            "failed_orders_count": 0,
            "pre_trade_risk_check_passed": True,
            "raw_exchange_response": order_result,
        }

    return provider


# This block builds the WSL-backed Kraken CLI client from runtime settings.
# It takes: optional settings already loaded by the backend.
# It gives: a subprocess-backed Kraken CLI client ready for market and execution flows.
def build_kraken_cli_client(
    *,
    settings: AppSettings | None = None,
) -> KrakenCLIClient:
    resolved_settings = settings or get_settings()
    cli_launcher = resolved_settings.kraken_cli_path
    if os.name == "nt" and cli_launcher.strip().lower() == "kraken":
        cli_launcher = "wsl.exe"

    # This block pads the CLI timeout for WSL startup and Kraken CLI cold-start latency.
    # It takes: the configured timeout from settings.
    # It gives: a more forgiving timeout floor that avoids false failures on Windows + WSL.
    timeout_seconds = max(resolved_settings.kraken_timeout_seconds, 60.0)

    return KrakenCLIClient(
        KrakenCLIConfig(
            cli_path=cli_launcher,
            wsl_distro=resolved_settings.kraken_wsl_distro,
            wsl_cli_path=resolved_settings.kraken_wsl_cli_path,
            timeout_seconds=timeout_seconds,
        )
    )


# This block builds a live market snapshot provider backed by Kraken CLI ticker data.
# It takes: a subprocess-backed Kraken CLI client.
# It gives: a graph-compatible provider that enriches state with live market data.
def build_kraken_cli_market_snapshot_provider(
    *,
    client: KrakenCLIClient,
) -> MarketSnapshotProvider:
    def provider(state) -> MarketSnapshotRecord:
        status = client.status()
        ticker_payload = client.ticker(state.symbol)
        pair_id, pair_data = _extract_cli_ticker_payload(ticker_payload)
        bid = _read_cli_price_field(pair_data, "b")
        ask = _read_cli_price_field(pair_data, "a")
        last = _read_cli_optional_price_field(pair_data, "c")
        mid = (bid + ask) / 2
        spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 else 0.0

        return MarketSnapshotRecord(
            symbol=state.symbol,
            bid=bid,
            ask=ask,
            mid=mid,
            spread_bps=spread_bps,
            realized_volatility=0.0,
            orderbook_stale_seconds=0.0,
            market_data_available=True,
            raw_payload={
                "pair": pair_id,
                "ticker": pair_data,
                "status": status,
                "runtime_context": {
                    "execution": {
                        "orders_in_last_minute": 0,
                        "failed_orders_count": 0,
                        "pre_trade_risk_check_passed": True,
                    },
                    "system": {
                        "kraken_cli_available": status.get("status") == "online",
                        "checkpoint_store_available": True,
                        "inference_backend_degraded": False,
                        "clock_skew_ms": 0,
                    },
                },
            },
        )

    return provider


# This block builds a portfolio provider backed by Kraken CLI balances and open orders.
# It takes: a subprocess-backed Kraken CLI client plus a live/paper mode flag.
# It gives: a graph-compatible provider that reports wallet and exposure context.
def build_kraken_cli_portfolio_provider(
    *,
    client: KrakenCLIClient,
    paper: bool = False,
) -> PortfolioContextProvider:
    def provider(state) -> dict[str, Any]:
        ticker_payload = client.ticker(state.symbol)
        _, pair_data = _extract_cli_ticker_payload(ticker_payload)
        bid = _read_cli_price_field(pair_data, "b")
        ask = _read_cli_price_field(pair_data, "a")
        mid = (bid + ask) / 2

        if paper:
            balance_payload = client.paper_balance()
            status_payload = client.paper_status()
            balances = balance_payload.get("balances", {})
            if not isinstance(balances, dict):
                balances = {}

            usd_bucket = balances.get("USD", {})
            if not isinstance(usd_bucket, dict):
                usd_bucket = {}

            wallet_available_usd = _safe_float(usd_bucket.get("available"))
            usd_total = _safe_float(usd_bucket.get("total"))
            base_symbol = _extract_base_symbol(state.symbol)
            base_balance = _extract_cli_paper_asset_balance(balance_payload, base_symbol)
            asset_value_usd = base_balance * mid
            wallet_balance_usd = usd_total + asset_value_usd
            total_equity = wallet_balance_usd

            open_positions_count = int(status_payload.get("open_orders", 0))
            if base_balance > 0:
                open_positions_count += 1

            total_open_exposure_bps = 0.0
            asset_open_exposure_bps = 0.0
            if total_equity > 0:
                total_open_exposure_bps = (asset_value_usd / total_equity) * 10_000
                asset_open_exposure_bps = total_open_exposure_bps
        else:
            balance_payload = client.balance()
            open_orders_payload = client.open_orders()
            wallet_balance_usd = _extract_cli_wallet_balance(balance_payload)
            wallet_available_usd = wallet_balance_usd
            open_positions_count = _count_cli_open_orders(open_orders_payload)
            total_open_exposure_bps = 0.0
            asset_open_exposure_bps = 0.0

            base_symbol = _extract_base_symbol(state.symbol)
            base_balance = _extract_cli_asset_balance(balance_payload, base_symbol)
            asset_value_usd = base_balance * mid
            total_equity = wallet_balance_usd + asset_value_usd
            if total_equity > 0:
                total_open_exposure_bps = (asset_value_usd / total_equity) * 10_000
                asset_open_exposure_bps = total_open_exposure_bps

        return {
            "total_open_exposure_bps": total_open_exposure_bps,
            "asset_open_exposure_bps": asset_open_exposure_bps,
            "open_positions_count": open_positions_count,
            "wallet_balance_usd": wallet_balance_usd,
            "wallet_available_usd": wallet_available_usd,
            "base_asset_units": base_balance if "base_balance" in locals() else 0.0,
            "asset_value_usd": asset_value_usd if "asset_value_usd" in locals() else 0.0,
            "manual_kill_switch": False,
            "daily_loss_limit_breached": False,
            "untrusted_operator_context": False,
        }

    return provider


# This block builds an execution provider backed by Kraken CLI.
# It takes: a subprocess-backed Kraken CLI client and the desired execution mode.
# It gives: a graph-compatible provider for paper, validate-only, or live execution.
def build_kraken_cli_execution_provider(
    *,
    client: KrakenCLIClient,
    mode: KrakenCLIExecutionMode = KrakenCLIExecutionMode.PAPER,
) -> ExecutionProvider:
    def provider(execution_request: dict[str, Any], state) -> dict[str, Any]:
        if state.trade_intent_record is None:
            raise ConfigurationError(
                "Kraken CLI execution provider requires state.trade_intent_record."
            )

        ticker_payload = client.ticker(state.trade_intent_record.symbol)
        _, pair_data = _extract_cli_ticker_payload(ticker_payload)
        bid = _read_cli_price_field(pair_data, "b")
        ask = _read_cli_price_field(pair_data, "a")
        reference_price = _read_cli_optional_price_field(pair_data, "c") or ((bid + ask) / 2)
        if reference_price <= 0:
            raise ConfigurationError("Kraken CLI reference price must be positive.")

        requested_notional = float(state.trade_intent_record.notional_usd)
        submitted_notional = requested_notional

        # This block constrains paper-trade size to the available local paper balances.
        # It takes: the requested notional plus the paper wallet's available USD or base-asset balance.
        # It gives: a safe submitted notional that fits the simulated account, or a clear error.
        if mode == KrakenCLIExecutionMode.PAPER:
            balance_payload = client.paper_balance()
            fee_buffer_multiplier = 0.997

            if state.trade_intent_record.side.value == "buy":
                available_usd = _extract_cli_paper_available_usd(balance_payload)
                max_affordable_notional = max(0.0, available_usd * fee_buffer_multiplier)
                submitted_notional = min(requested_notional, max_affordable_notional)

                if submitted_notional < 10.0:
                    raise ConfigurationError(
                        "Kraken paper wallet has too little free USD for a meaningful buy. "
                        "Reduce exposure or reset the paper account.",
                        context={
                            "requested_notional_usd": requested_notional,
                            "available_wallet_balance_usd": available_usd,
                            "max_affordable_notional_usd": max_affordable_notional,
                        },
                    )
            else:
                base_symbol = _extract_base_symbol(state.trade_intent_record.symbol)
                available_base_units = _extract_cli_paper_asset_balance(
                    balance_payload,
                    base_symbol,
                )
                max_affordable_notional = max(
                    0.0,
                    available_base_units * reference_price * fee_buffer_multiplier,
                )
                submitted_notional = min(requested_notional, max_affordable_notional)

                if submitted_notional < 10.0:
                    raise ConfigurationError(
                        "Kraken paper wallet has too little base-asset inventory for a meaningful sell. "
                        "Open or grow the position before attempting a reduction.",
                        context={
                            "requested_notional_usd": requested_notional,
                            "available_base_units": available_base_units,
                            "max_affordable_notional_usd": max_affordable_notional,
                        },
                    )

        volume = submitted_notional / reference_price

        order_payload = client.place_order(
            pair=state.trade_intent_record.symbol,
            side=state.trade_intent_record.side.value,
            order_type=state.trade_intent_record.order_type.value,
            volume=volume,
            price=(
                reference_price
                if state.trade_intent_record.order_type.value == "limit"
                else None
            ),
            time_in_force=state.trade_intent_record.time_in_force,
            mode=mode,
        )

        execution_status = _resolve_cli_execution_status(mode=mode, payload=order_payload)
        exchange_order_id = _extract_cli_order_id(order_payload, mode=mode)

        execution_record = ExecutionRecord(
            trade_intent_id=state.trade_intent_record.id,
            policy_decision_id=(
                state.policy.record.id if state.policy.record is not None else None
            ),
            symbol=state.trade_intent_record.symbol,
            side=state.trade_intent_record.side,
            order_type=state.trade_intent_record.order_type,
            status=ExecutionStatus(execution_status),
            exchange_order_id=exchange_order_id,
            requested_notional_usd=submitted_notional,
            executed_notional_usd=(
                submitted_notional
                if mode in {KrakenCLIExecutionMode.PAPER, KrakenCLIExecutionMode.LIVE}
                else None
            ),
            average_fill_price=(
                reference_price
                if mode in {KrakenCLIExecutionMode.PAPER, KrakenCLIExecutionMode.LIVE}
                else None
            ),
            raw_payload={
                "mode": mode.value,
                "request": execution_request,
                "response": order_payload,
                "reference_price": reference_price,
                "volume": volume,
                "requested_notional_usd": requested_notional,
                "submitted_notional_usd": submitted_notional,
            },
            source=RecordSource.EXECUTION,
        )

        return {
            "status": execution_status,
            "records": [execution_record],
            "orders_in_last_minute": 1,
            "failed_orders_count": 0,
            "pre_trade_risk_check_passed": True,
            "raw_exchange_response": order_payload,
        }

    return provider


# This block extracts a USD-equivalent cash balance from Kraken balance data.
# It takes: the raw balance mapping returned by Kraken.
# It gives: the first matching USD-like balance as a float.
def _extract_usd_balance(balances: dict[str, str]) -> float:
    for asset in ("ZUSD", "USD", "USDT", "USDC"):
        if asset in balances:
            return _safe_float(balances[asset])
    return 0.0


# This block extracts the current symbol's base-asset amount from Kraken balances.
# It takes: the raw balance mapping and a repo-native trading symbol.
# It gives: the available base-asset quantity as a float.
def _extract_symbol_base_amount(balances: dict[str, str], symbol: str) -> float:
    normalized_symbol = symbol.upper()
    base_asset = normalized_symbol.replace("USD", "").replace("USDT", "").replace("USDC", "")
    asset_aliases = {
        "BTC": ("XXBT", "XBT", "BTC"),
        "ETH": ("XETH", "ETH"),
        "SOL": ("SOL",),
    }
    for candidate in asset_aliases.get(base_asset, (base_asset, f"X{base_asset}")):
        if candidate in balances:
            return _safe_float(balances[candidate])
    return 0.0


# This block identifies balance keys that represent active non-cash holdings.
# It takes: one Kraken asset code.
# It gives: a boolean indicating whether the asset should count as a position.
def _is_non_cash_asset(asset: str) -> bool:
    return asset not in {"ZUSD", "USD", "USDT", "USDC", "KFEE"}


# This block safely coerces Kraken numeric strings into floats.
# It takes: a raw numeric value from Kraken.
# It gives: a float value, or 0.0 if the field is absent or malformed.
def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# This block extracts the first pair payload from Kraken CLI ticker JSON.
# It takes: the raw JSON payload returned by `kraken ticker -o json`.
# It gives: the pair id and the inner ticker data mapping.
def _extract_cli_ticker_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not isinstance(payload, dict) or not payload:
        raise ConfigurationError("Kraken CLI ticker payload was empty or invalid.")

    pair_id, pair_data = next(iter(payload.items()))
    if not isinstance(pair_data, dict):
        raise ConfigurationError("Kraken CLI ticker pair payload must be a mapping.")
    return str(pair_id), pair_data


# This block reads a required ticker price field from Kraken CLI JSON.
# It takes: the inner ticker mapping and a field key like 'a' or 'b'.
# It gives: the first numeric price value as a float.
def _read_cli_price_field(payload: dict[str, Any], field_name: str) -> float:
    value = payload.get(field_name)
    if isinstance(value, list) and value:
        return float(value[0])
    raise ConfigurationError(f"Kraken CLI ticker field '{field_name}' is missing.")


# This block reads an optional ticker price field from Kraken CLI JSON.
# It takes: the inner ticker mapping and a field key like 'c'.
# It gives: the first numeric price value as a float, or None when absent.
def _read_cli_optional_price_field(payload: dict[str, Any], field_name: str) -> float | None:
    value = payload.get(field_name)
    if isinstance(value, list) and value:
        return float(value[0])
    return None


# This block extracts the wallet's USD-equivalent balance from Kraken CLI JSON.
# It takes: the raw JSON returned by `kraken balance -o json`.
# It gives: a float USD balance for policy deny-rule evaluation.
def _extract_cli_wallet_balance(payload: dict[str, Any]) -> float:
    for asset in ("ZUSD", "USD", "USDT", "USDC"):
        value = payload.get(asset)
        if value is not None:
            return _safe_float(value)
    return 0.0


# This block extracts a base-asset balance from Kraken CLI JSON.
# It takes: the raw balance payload and a normalized base symbol like BTC or ETH.
# It gives: the available base-asset quantity as a float.
def _extract_cli_asset_balance(payload: dict[str, Any], base_symbol: str) -> float:
    asset_aliases = {
        "BTC": ("XXBT", "XBT", "BTC"),
        "ETH": ("XETH", "ETH"),
        "SOL": ("SOL",),
    }
    for candidate in asset_aliases.get(base_symbol, (base_symbol, f"X{base_symbol}")):
        if candidate in payload:
            return _safe_float(payload[candidate])
    return 0.0


# This block extracts a base-asset balance from Kraken paper-balance JSON.
# It takes: the raw JSON returned by `kraken paper balance -o json` plus a base symbol.
# It gives: the available base-asset quantity held in the paper account.
def _extract_cli_paper_asset_balance(payload: dict[str, Any], base_symbol: str) -> float:
    balances = payload.get("balances", {})
    if not isinstance(balances, dict):
        return 0.0

    asset_aliases = {
        "BTC": ("XXBT", "XBT", "BTC"),
        "ETH": ("XETH", "ETH"),
        "SOL": ("SOL",),
    }
    for candidate in asset_aliases.get(base_symbol, (base_symbol, f"X{base_symbol}")):
        asset_bucket = balances.get(candidate)
        if isinstance(asset_bucket, dict):
            return _safe_float(asset_bucket.get("available"))

    return 0.0


# This block extracts the available USD balance from Kraken paper-balance JSON.
# It takes: the raw JSON returned by `kraken paper balance -o json`.
# It gives: the currently available USD amount for new paper orders.
def _extract_cli_paper_available_usd(payload: dict[str, Any]) -> float:
    balances = payload.get("balances", {})
    if not isinstance(balances, dict):
        return 0.0

    usd_bucket = balances.get("USD", {})
    if not isinstance(usd_bucket, dict):
        return 0.0

    return _safe_float(usd_bucket.get("available"))


# This block counts open live orders from Kraken CLI JSON.
# It takes: the raw JSON returned by `kraken open-orders -o json`.
# It gives: the current number of open live orders.
def _count_cli_open_orders(payload: dict[str, Any]) -> int:
    open_orders = payload.get("open")
    if isinstance(open_orders, dict):
        return len(open_orders)
    return 0


# This block extracts a repo-native base symbol from a quote pair like BTCUSD.
# It takes: the current graph symbol.
# It gives: the base asset symbol for balance matching.
def _extract_base_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    for quote in ("USDT", "USDC", "USD"):
        if normalized.endswith(quote):
            return normalized[: -len(quote)]
    return normalized


# This block resolves the execution status from Kraken CLI order output.
# It takes: the chosen execution mode and raw JSON order payload.
# It gives: a storage-safe execution status string for the graph state.
def _resolve_cli_execution_status(
    *,
    mode: KrakenCLIExecutionMode,
    payload: dict[str, Any],
) -> str:
    if mode == KrakenCLIExecutionMode.VALIDATE:
        return ExecutionStatus.PENDING.value

    if mode == KrakenCLIExecutionMode.PAPER:
        action = str(payload.get("action", "")).lower()
        if "filled" in action:
            return ExecutionStatus.FILLED.value
        return ExecutionStatus.SUBMITTED.value

    return ExecutionStatus.SUBMITTED.value


# This block extracts the best available order identifier from Kraken CLI order output.
# It takes: the raw JSON order payload and chosen execution mode.
# It gives: a stable exchange-order identifier for execution history records.
def _extract_cli_order_id(
    payload: dict[str, Any],
    *,
    mode: KrakenCLIExecutionMode,
) -> str | None:
    if mode == KrakenCLIExecutionMode.PAPER:
        value = payload.get("order_id")
        return str(value) if value is not None else None

    result = payload.get("result")
    if isinstance(result, dict):
        txid = result.get("txid")
        if isinstance(txid, list) and txid:
            return str(txid[0])
    return None
