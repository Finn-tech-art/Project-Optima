from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from backend.agent.nodes.execution import ExecutionNodeConfig, build_execution_node
from backend.agent.nodes.finalize import build_finalize_node
from backend.agent.nodes.guardrail import GuardrailNodeConfig, build_guardrail_node
from backend.agent.nodes.intent import IntentNodeConfig, build_intent_node
from backend.agent.nodes.market_context import (
    MarketContextNodeConfig,
    build_market_context_node,
)
from backend.agent.nodes.memory import MemoryNodeConfig, build_memory_node
from backend.agent.nodes.portfolio import PortfolioNodeConfig, build_portfolio_node
from backend.agent.nodes.regime import RegimeNodeConfig, build_regime_node
from backend.agent.nodes.sentiment import SentimentNodeConfig, build_sentiment_node
from backend.agent.nodes.trust_context import (
    LiveTrustContextNodeConfig,
    TrustContextNodeConfig,
    build_live_trust_context_node,
    build_trust_context_node,
)
from backend.agent.nodes.validator import ValidatorNodeConfig, build_validator_node
from backend.agent.state import RRAAgentState
from backend.data.checkpoints import create_sqlite_checkpointer


# This block defines the canonical node names used by the RRA orchestration graph.
# It takes: no runtime input.
# It gives: one stable naming contract for graph wiring, tracing, and checkpoint inspection.
MARKET_CONTEXT_NODE = "market_context"
TRUST_CONTEXT_NODE = "trust_context"
PORTFOLIO_NODE = "portfolio"
REGIME_NODE = "regime"
SENTIMENT_NODE = "sentiment"
MEMORY_NODE = "memory"
INTENT_NODE = "intent"
GUARDRAIL_NODE = "guardrail"
VALIDATOR_NODE = "validator"
EXECUTION_NODE = "execution"
FINALIZE_NODE = "finalize"


# This block stores the node-builder configs used when constructing the graph.
# It takes: per-node config objects for context loading, inference, policy, validation, and execution.
# It gives: one graph-ready configuration bundle.
@dataclass(slots=True)
class RRAGraphNodeConfigs:
    market_context: MarketContextNodeConfig
    portfolio: PortfolioNodeConfig
    regime: RegimeNodeConfig
    sentiment: SentimentNodeConfig
    memory: MemoryNodeConfig
    guardrail: GuardrailNodeConfig
    execution: ExecutionNodeConfig
    trust_context: TrustContextNodeConfig | None = None
    live_trust_context: LiveTrustContextNodeConfig | None = None
    intent: IntentNodeConfig | None = None
    validator: ValidatorNodeConfig | None = None


# This block defines the build and compile configuration for the LangGraph workflow.
# It takes: optional checkpoint settings and graph-level naming metadata.
# It gives: one normalized config object for graph compilation.
@dataclass(slots=True, frozen=True)
class RRAGraphConfig:
    name: str = "rra-trading-graph"
    checkpoint_db_path: str | None = None
    interrupt_before: tuple[str, ...] = ()
    interrupt_after: tuple[str, ...] = ()


# This block decides where the graph should route after guardrails run.
# It takes: the current state after deterministic policy and security checks.
# It gives: the next node name, either validator or finalize.
def route_after_guardrail(state: RRAAgentState) -> str:
    action = (state.policy.action or "").lower()

    if action == "allow" and state.policy.allowed:
        return VALIDATOR_NODE

    return FINALIZE_NODE


# This block decides where the graph should route after validation and proof packaging.
# It takes: the current state after validator execution.
# It gives: the next node name, either execution or finalize.
def route_after_validator(state: RRAAgentState) -> str:
    action = (state.policy.action or "").lower()

    if action == "allow" and state.policy.allowed:
        return EXECUTION_NODE

    return FINALIZE_NODE


# This block decides where the graph should route after execution.
# It takes: the current execution state.
# It gives: the next node name, which currently always finalizes the run.
def route_after_execution(state: RRAAgentState) -> str:
    return FINALIZE_NODE


# This block builds the trust-context node from either live or provider-backed config.
# It takes: the graph node config bundle.
# It gives: the correct trust-context node implementation for the current runtime.
def _resolve_trust_context_node(configs: RRAGraphNodeConfigs):
    if configs.live_trust_context is not None:
        return build_live_trust_context_node(configs.live_trust_context)

    if configs.trust_context is not None:
        return build_trust_context_node(configs.trust_context)

    raise ValueError(
        "RRAGraphNodeConfigs requires either trust_context or live_trust_context."
    )


# This block creates the uncompiled StateGraph definition for the RRA workflow.
# It takes: the concrete node-builder config bundle.
# It gives: a fully wired StateGraph that can later be compiled with persistence.
def build_rra_state_graph(configs: RRAGraphNodeConfigs) -> StateGraph:
    graph = StateGraph(RRAAgentState)

    # This block builds each graph node from its config object.
    # It takes: the provided node configs.
    # It gives: concrete LangGraph node callables for the full workflow.
    market_context_node = build_market_context_node(configs.market_context)
    trust_context_node = _resolve_trust_context_node(configs)
    portfolio_node = build_portfolio_node(configs.portfolio)
    regime_node = build_regime_node(configs.regime)
    sentiment_node = build_sentiment_node(configs.sentiment)
    memory_node = build_memory_node(configs.memory)
    intent_node = build_intent_node(configs.intent)
    guardrail_node = build_guardrail_node(configs.guardrail)
    validator_node = build_validator_node(configs.validator)
    execution_node = build_execution_node(configs.execution)
    finalize_node = build_finalize_node()

    # This block registers each workflow stage as a named graph node.
    # It takes: the concrete node callables.
    # It gives: a graph with all core RRA stages attached.
    graph.add_node(MARKET_CONTEXT_NODE, market_context_node)
    graph.add_node(TRUST_CONTEXT_NODE, trust_context_node)
    graph.add_node(PORTFOLIO_NODE, portfolio_node)
    graph.add_node(REGIME_NODE, regime_node)
    graph.add_node(SENTIMENT_NODE, sentiment_node)
    graph.add_node(MEMORY_NODE, memory_node)
    graph.add_node(INTENT_NODE, intent_node)
    graph.add_node(GUARDRAIL_NODE, guardrail_node)
    graph.add_node(VALIDATOR_NODE, validator_node)
    graph.add_node(EXECUTION_NODE, execution_node)
    graph.add_node(FINALIZE_NODE, finalize_node)

    # This block defines the main RRA orchestration path before branching.
    # It takes: the registered node names.
    # It gives: a deterministic pre-execution sequence from context loading to guardrails.
    graph.add_edge(START, MARKET_CONTEXT_NODE)
    graph.add_edge(MARKET_CONTEXT_NODE, TRUST_CONTEXT_NODE)
    graph.add_edge(TRUST_CONTEXT_NODE, PORTFOLIO_NODE)
    graph.add_edge(PORTFOLIO_NODE, REGIME_NODE)
    graph.add_edge(REGIME_NODE, SENTIMENT_NODE)
    graph.add_edge(SENTIMENT_NODE, MEMORY_NODE)
    graph.add_edge(MEMORY_NODE, INTENT_NODE)
    graph.add_edge(INTENT_NODE, GUARDRAIL_NODE)

    # This block adds guardrail-based routing.
    # It takes: the state after deterministic security and risk checks.
    # It gives: either proof validation or immediate finalization.
    graph.add_conditional_edges(
        GUARDRAIL_NODE,
        route_after_guardrail,
        {
            VALIDATOR_NODE: VALIDATOR_NODE,
            FINALIZE_NODE: FINALIZE_NODE,
        },
    )

    # This block adds validator-based routing.
    # It takes: the state after proof and signing validation.
    # It gives: either hardened execution or immediate finalization.
    graph.add_conditional_edges(
        VALIDATOR_NODE,
        route_after_validator,
        {
            EXECUTION_NODE: EXECUTION_NODE,
            FINALIZE_NODE: FINALIZE_NODE,
        },
    )

    # This block routes execution into the finalizer.
    # It takes: the state after exchange submission.
    # It gives: a clean terminal path for the current graph scaffold.
    graph.add_conditional_edges(
        EXECUTION_NODE,
        route_after_execution,
        {
            FINALIZE_NODE: FINALIZE_NODE,
        },
    )

    graph.add_edge(FINALIZE_NODE, END)
    return graph


# This block compiles the RRA graph with optional checkpoint persistence.
# It takes: the graph node configs plus optional compile configuration.
# It gives: a compiled LangGraph object ready for invoke or stream execution.
def compile_rra_graph(
    *,
    node_configs: RRAGraphNodeConfigs,
    config: RRAGraphConfig | None = None,
) -> CompiledStateGraph:
    resolved_config = config or RRAGraphConfig()
    graph = build_rra_state_graph(node_configs)

    checkpointer = None
    if resolved_config.checkpoint_db_path:
        checkpointer = create_sqlite_checkpointer(resolved_config.checkpoint_db_path)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=list(resolved_config.interrupt_before),
        interrupt_after=list(resolved_config.interrupt_after),
        name=resolved_config.name,
    )
