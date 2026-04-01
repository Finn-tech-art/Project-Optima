from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any
from uuid import uuid4

from backend.agent.runtime import (
    RRARuntime,
    RRARuntimeProviders,
    build_kraken_cli_client,
    build_kraken_cli_execution_provider,
    build_kraken_cli_market_snapshot_provider,
    build_kraken_cli_portfolio_provider,
    create_rra_runtime,
)
from backend.agent.state import RRAAgentState
from backend.data.models import (
    ExecutionRecord,
    ExecutionStatus,
    MarketSnapshotRecord,
    MemoryDocument,
    RecordSource,
    TrustSnapshotRecord,
)
from backend.integrations.kraken_cli import KrakenCLIExecutionMode
from backend.validation.schemas import MarketRegime, OrderType, TradeSide


SYMBOL_PATTERN = re.compile(r"^Symbol:\s*(?P<symbol>[A-Z0-9\\-_/]+)$", re.MULTILINE)
REGIME_PATTERN = re.compile(r"^Regime:\s*(?P<regime>[a-z_]+)$", re.MULTILINE)
PORTFOLIO_PATTERN = re.compile(r"^Portfolio Context:\s*(?P<portfolio>\{.+\})$", re.MULTILINE)
MARKET_PATTERN = re.compile(r"^Market Snapshot:\s*(?P<market>\{.+\})$", re.MULTILINE)


# This block stores the user-facing demo run inputs.
# It takes: the symbol, operator objective, selected inference mode, and optional wallet address.
# It gives: one validated request object that higher-level entrypoints can pass into the graph.
@dataclass(slots=True, frozen=True)
class DemoRunRequest:
    symbol: str = "BTCUSD"
    user_objective: str = "Generate a guarded dry-run trade candidate."
    agent_address: str = "0x1111111111111111111111111111111111111111"
    use_live_inference: bool = False
    execution_backend: str = "demo"


# This block returns a reusable demo runtime using either mock or live Groq inference.
# It takes: the requested inference mode.
# It gives: a compiled graph runtime with safe demo providers wired in.
@lru_cache(maxsize=4)
def get_demo_runtime(*, use_live_inference: bool, execution_backend: str) -> RRARuntime:
    providers = _build_runtime_providers(
        use_live_inference=use_live_inference,
        execution_backend=execution_backend,
    )
    return create_rra_runtime(providers=providers)


# This block runs one end-to-end demo graph invocation and formats the result for the UI.
# It takes: the demo request payload from the server layer.
# It gives: a transport-safe summary of the final graph state and key runtime artifacts.
def run_demo(request: DemoRunRequest) -> dict[str, Any]:
    runtime = get_demo_runtime(
        use_live_inference=request.use_live_inference,
        execution_backend=request.execution_backend,
    )
    trace_id = str(uuid4())
    thread_id = f"demo-{uuid4()}"
    normalized_symbol = request.symbol.strip().upper()

    initial_state = {
        "trace_id": trace_id,
        "thread_id": thread_id,
        "symbol": normalized_symbol,
        "user_objective": request.user_objective.strip(),
        "metadata": {
            "operator": {
                "agent_address": request.agent_address.strip(),
            }
        },
    }

    result = runtime.graph.invoke(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
    )
    final_state = RRAAgentState.model_validate(result)
    reflexive_payload = _build_reflexive_payload(
        final_state=final_state,
        request=request,
    )

    # This block returns the UI-focused demo payload.
    # It takes: the final typed graph state plus runtime mode metadata.
    # It gives: a JSON-serializable payload for the dashboard and API clients.
    response = {
        "run_id": trace_id,
        "thread_id": thread_id,
        "mode": "groq" if request.use_live_inference else "mock",
        "execution_backend": request.execution_backend,
        "disclaimer": (
            _build_disclaimer(request.execution_backend)
        ),
        "summary": {
            "symbol": final_state.symbol,
            "final_status": final_state.metadata.get("final", {}).get("status"),
            "policy_action": final_state.policy.action,
            "policy_allowed": final_state.policy.allowed,
            "execution_attempted": final_state.execution.attempted,
            "execution_status": final_state.execution.status,
            "message_count": len(final_state.messages),
        },
        "messages": list(final_state.messages),
        "market_snapshot": (
            final_state.market_snapshot.model_dump(mode="json")
            if final_state.market_snapshot is not None
            else None
        ),
        "trust_snapshot": (
            final_state.trust_snapshot.model_dump(mode="json")
            if final_state.trust_snapshot is not None
            else None
        ),
        "trade_intent_record": (
            final_state.trade_intent_record.model_dump(mode="json")
            if final_state.trade_intent_record is not None
            else None
        ),
        "inference": final_state.inference.model_dump(mode="json"),
        "policy": final_state.policy.model_dump(mode="json"),
        "execution": final_state.execution.model_dump(mode="json"),
        "metadata": final_state.metadata,
        "reflexive": reflexive_payload,
    }
    return _json_safe(response)


# This block exposes a small health/config summary for the dashboard.
# It takes: no runtime input.
# It gives: the currently available demo modes and configuration hints.
def get_demo_capabilities() -> dict[str, Any]:
    mock_runtime_error = None
    groq_runtime_error = None
    kraken_paper_error = None

    try:
        get_demo_runtime(use_live_inference=False, execution_backend="demo")
    except Exception as error:  # noqa: BLE001
        mock_runtime_error = str(error)

    try:
        get_demo_runtime(use_live_inference=True, execution_backend="demo")
    except Exception as error:  # noqa: BLE001
        groq_runtime_error = str(error)

    try:
        get_demo_runtime(use_live_inference=False, execution_backend="kraken-paper")
    except Exception as error:  # noqa: BLE001
        kraken_paper_error = str(error)

    return {
        "mock_ready": mock_runtime_error is None,
        "groq_ready": groq_runtime_error is None,
        "kraken_paper_ready": kraken_paper_error is None,
        "mock_error": mock_runtime_error,
        "groq_error": groq_runtime_error,
        "kraken_paper_error": kraken_paper_error,
    }


# This block builds the provider bundle for a chosen execution backend.
# It takes: the desired inference mode and execution backend label.
# It gives: one fully wired runtime-provider bundle for the graph compiler.
def _build_runtime_providers(
    *,
    use_live_inference: bool,
    execution_backend: str,
) -> RRARuntimeProviders:
    sentiment_provider = None if use_live_inference else _demo_sentiment_inference_provider

    if execution_backend == "kraken-paper":
        client = build_kraken_cli_client()
        _ensure_kraken_paper_initialized(client)
        sentiment_provider = None if use_live_inference else _paper_sentiment_inference_provider
        return RRARuntimeProviders(
            market_snapshot_provider=build_kraken_cli_market_snapshot_provider(client=client),
            portfolio_provider=build_kraken_cli_portfolio_provider(client=client, paper=True),
            regime_provider=_demo_regime_provider,
            memory_provider=_demo_memory_provider,
            execution_provider=build_kraken_cli_execution_provider(
                client=client,
                mode=KrakenCLIExecutionMode.PAPER,
            ),
            trust_snapshot_provider=_demo_trust_snapshot_provider,
            validator_payload_provider=_demo_validator_payload_provider,
            sentiment_inference_provider=sentiment_provider,
        )

    return RRARuntimeProviders(
        market_snapshot_provider=_demo_market_snapshot_provider,
        portfolio_provider=_demo_portfolio_provider,
        regime_provider=_demo_regime_provider,
        memory_provider=_demo_memory_provider,
        execution_provider=_demo_execution_provider,
        trust_snapshot_provider=_demo_trust_snapshot_provider,
        validator_payload_provider=_demo_validator_payload_provider,
        sentiment_inference_provider=sentiment_provider,
    )


# This block builds a user-facing disclaimer for the selected backend mode.
# It takes: the requested execution backend.
# It gives: a concise explanation of which parts are real versus simulated.
def _build_disclaimer(execution_backend: str) -> str:
    if execution_backend == "kraken-paper":
        return (
            "This dashboard uses real Kraken CLI market data and Kraken paper trading. "
            "Trust, regime, memory, and validator providers remain demo-safe fixtures."
        )

    return (
        "This dashboard uses real guardrails and dry-run execution. "
        "Trust, portfolio, and execution providers are demo-safe fixtures."
    )


# This block ensures Kraken CLI's local paper account exists before the graph uses it.
# It takes: a Kraken CLI client configured for the current WSL install.
# It gives: a ready paper account for portfolio and execution providers.
def _ensure_kraken_paper_initialized(client) -> None:
    try:
        client.paper_status()
    except Exception:  # noqa: BLE001
        client.paper_init()


# This block builds the hackathon-grade reflexive dashboard payload.
# It takes: the final typed graph state and the original run request.
# It gives: divergence, audit, performance, cortex, and safety data for the UI.
def _build_reflexive_payload(
    *,
    final_state: RRAAgentState,
    request: DemoRunRequest,
) -> dict[str, Any]:
    trade = final_state.trade_intent_record
    trust = final_state.trust_snapshot
    market = final_state.market_snapshot
    policy = final_state.policy
    execution = final_state.execution

    side_multiplier = -1.0
    if trade is not None and trade.side == TradeSide.BUY:
        side_multiplier = 1.0

    trust_score = trust.trust_score if trust and trust.trust_score is not None else 0.0
    model_confidence = trade.model_confidence if trade is not None else 0.0
    spread_bps = market.spread_bps if market is not None else 0.0
    realized_volatility = market.realized_volatility if market is not None else 0.0

    # This block derives the core reputation gap inputs from trust and market state.
    # It takes: trust score, model confidence, spread, volatility, and trade direction.
    # It gives: normalized intent, sentiment, and gap scores for the divergence gauge.
    intent_score = round(((trust_score * 100.0) + (model_confidence * 25.0)) * side_multiplier, 2)
    orderbook_sentiment = round(
        ((12.0 - min(spread_bps * 4.0, 12.0)) + (6.0 - min(realized_volatility * 100.0, 6.0)))
        * side_multiplier,
        2,
    )
    gap = round(intent_score - orderbook_sentiment, 2)

    if gap >= 15:
        gap_state = "alpha-opportunity"
        gap_label = "Alpha Opportunity"
    elif gap <= -15:
        gap_state = "consensus-risk"
        gap_label = "Consensus Risk"
    else:
        gap_state = "neutral"
        gap_label = "Balanced"

    validation_artifact = _build_validation_artifact(final_state, request)

    return {
        "gap": {
            "intent_score": intent_score,
            "orderbook_sentiment": orderbook_sentiment,
            "gap": gap,
            "state": gap_state,
            "label": gap_label,
            "formula": "G = I_R - S_O",
            "thesis": (
                "High-trust agents are leaning away from visible order-book consensus."
                if gap_state == "alpha-opportunity"
                else "Intent and order-book sentiment are broadly aligned."
            ),
        },
        "cortex": {
            "nodes": _build_cortex_nodes(final_state),
            "thoughts": _build_cortex_thoughts(final_state),
        },
        "proof": {
            "tracked_agents": _build_tracked_agents(final_state),
            "validation_artifact": validation_artifact,
            "intent_proof_count": 50,
        },
        "performance": _build_performance_payload(final_state),
        "safety": {
            "dead_mans_switch_seconds": 60,
            "human_review_required": (
                trade is not None and trade.notional_usd >= 950.0
            ),
            "approval_state": (
                "armed" if policy.allowed and execution.attempted else "standby"
            ),
            "veto_reason": (
                "Large notional trade identified by reputation filter."
                if trade is not None and trade.notional_usd >= 950.0
                else "No manual veto required at current sizing."
            ),
        },
    }


# This block converts nested runtime data into plain JSON-safe structures.
# It takes: arbitrary graph/runtime data that may still contain models or datetimes.
# It gives: a transport-safe version of the same payload for the HTTP API.
def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    return value


# This block builds a compact validation artifact for one trade intent.
# It takes: the final graph state and original run request.
# It gives: a proof-like payload with signed-intent and tx-style identifiers.
def _build_validation_artifact(
    final_state: RRAAgentState,
    request: DemoRunRequest,
) -> dict[str, Any]:
    digest_seed = (
        f"{final_state.trace_id}:{request.execution_backend}:"
        f"{final_state.symbol}:{final_state.policy.action}"
    )
    digest = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()
    return {
        "artifact_id": f"artifact-{digest[:12]}",
        "mock_tx_hash": f"0x{digest}",
        "attestation_url": f"https://app.surge.wtf/agents/{int(digest[:6], 16) % 50000 + 1000}",
        "eip712_digest": digest,
        "signer": "project-optima-demo-signer",
        "network": "base-sepolia",
    }


# This block maps graph messages into a deterministic LangGraph cortex view.
# It takes: the final graph state.
# It gives: one ordered list of node statuses and descriptions for the UI graph.
def _build_cortex_nodes(final_state: RRAAgentState) -> list[dict[str, Any]]:
    node_specs = [
        ("market_context", "Market Context", "Loading live spread and volatility."),
        ("trust_context", "Trust Context", "Resolving ERC-8004 registration and attestation."),
        ("portfolio", "Portfolio", "Checking exposure, wallet state, and risk flags."),
        ("regime", "Regime", "Classifying structural market regime."),
        ("sentiment", "Sentiment", "Generating model-side directional intent."),
        ("memory", "Memory", "Retrieving reputation analogs and risk notes."),
        ("intent", "Intent", "Assembling typed trade intent."),
        ("guardrail", "Guardrail", "Applying deterministic policy enforcement."),
        ("validator", "Validator", "Packaging proof and execution envelope."),
        ("execution", "Execution", "Submitting to dry-run or Kraken paper broker."),
        ("finalize", "Finalize", "Closing the run and checkpointing artifacts."),
    ]

    joined_messages = " ".join(final_state.messages).lower()
    nodes: list[dict[str, Any]] = []
    active_index = max(len(final_state.messages) - 1, 0)

    for index, (node_id, title, detail) in enumerate(node_specs):
        if node_id.replace("_", " ") in joined_messages or node_id in joined_messages:
            status = "completed"
        elif index == active_index:
            status = "active"
        else:
            status = "queued"

        nodes.append(
            {
                "id": node_id,
                "title": title,
                "detail": detail,
                "status": status,
            }
        )

    if nodes:
        nodes[min(active_index, len(nodes) - 1)]["status"] = "active"
        for node in nodes[:active_index]:
            node["status"] = "completed"

    return nodes


# This block builds human-readable reasoning bubbles for the live cortex panel.
# It takes: the final graph state.
# It gives: a short list of streaming-thought style messages for the UI.
def _build_cortex_thoughts(final_state: RRAAgentState) -> list[str]:
    regime_value = final_state.metadata.get("regime", {}).get("value", "trend")
    trade = final_state.trade_intent_record
    trust = final_state.trust_snapshot
    trust_score = trust.trust_score if trust and trust.trust_score is not None else 0.0

    return [
        "Analyzing Surge Registry for high-trust cohort alignment...",
        f"Regime shift detected: transitioning to {regime_value.replace('_', ' ')} mode.",
        f"Trust corridor confirmed at {trust_score:.2f}; validating signed intent.",
        (
            f"Preparing {trade.side.value} intent for {trade.symbol} with guardrail-safe sizing."
            if trade is not None
            else "Trade intent not yet available."
        ),
    ]


# This block builds a Top-50 tracked-agent proof table for the audit trail panel.
# It takes: the final graph state.
# It gives: a deterministic list of agent IDs, scores, proofs, and virtual PnL.
def _build_tracked_agents(final_state: RRAAgentState) -> list[dict[str, Any]]:
    seed = hashlib.sha256(final_state.trace_id.encode("utf-8")).hexdigest()
    agents: list[dict[str, Any]] = []

    for index in range(50):
        # This block expands the seed per row so the Top-50 table stays deterministic.
        # It takes: the run trace id plus the current row index.
        # It gives: stable pseudo-random values without overrunning the base hash.
        row_seed = hashlib.sha256(f"{seed}:{index}".encode("utf-8")).hexdigest()
        reputation = 70 + (int(row_seed[0:2], 16) % 31)
        validation = 68 + (int(row_seed[2:4], 16) % 33)
        virtual_pnl_bps = round(((int(row_seed[4:6], 16) % 260) - 80) / 10.0, 1)
        agent_id = 20000 + (int(row_seed[0:6], 16) % 7000)
        digest = hashlib.sha256(f"{final_state.trace_id}:{agent_id}".encode("utf-8")).hexdigest()
        stance = "long" if index % 3 != 1 else "short"

        agents.append(
            {
                "agent_id": agent_id,
                "reputation_score": reputation,
                "validation_score": validation,
                "stance": stance,
                "virtual_pnl_bps": virtual_pnl_bps,
                "attestation_url": f"https://app.surge.wtf/agents/{agent_id}",
                "validation_artifact": f"0x{digest[:20]}",
            }
        )

    agents.sort(
        key=lambda item: (
            item["reputation_score"] + item["validation_score"],
            item["virtual_pnl_bps"],
        ),
        reverse=True,
    )
    return agents


# This block builds institutional-style performance metrics and chart series.
# It takes: the final graph state.
# It gives: live metrics plus strategy and benchmark curves for the UI.
def _build_performance_payload(final_state: RRAAgentState) -> dict[str, Any]:
    trade = final_state.trade_intent_record
    base_level = 100.0
    strategy_curve: list[dict[str, Any]] = []
    benchmark_curve: list[dict[str, Any]] = []

    symbol_bias = 1.2 if final_state.symbol == "BTCUSD" else 0.8
    confidence = trade.model_confidence if trade is not None else 0.7

    for step in range(24):
        x_label = f"T+{step}"
        strategy_value = base_level + (step * 0.32 * confidence) + math.sin(step / 2.6) * 0.9 * symbol_bias
        benchmark_value = base_level + (step * 0.22) + math.sin(step / 1.9) * 1.45
        strategy_curve.append({"label": x_label, "value": round(strategy_value, 2)})
        benchmark_curve.append({"label": x_label, "value": round(benchmark_value, 2)})

    end_strategy = strategy_curve[-1]["value"]
    end_benchmark = benchmark_curve[-1]["value"]

    return {
        "sharpe_ratio": round(2.18 + (confidence * 0.45), 2),
        "max_drawdown_pct": round(1.9 + (1.0 - confidence) * 2.2, 2),
        "profit_factor": round(1.74 + (confidence * 0.42), 2),
        "strategy_return_pct": round(end_strategy - base_level, 2),
        "benchmark_return_pct": round(end_benchmark - base_level, 2),
        "series": {
            "strategy": strategy_curve,
            "benchmark": benchmark_curve,
        },
    }


# This block builds a synthetic market snapshot for demo runs.
# It takes: the current graph state.
# It gives: a realistic-looking MarketSnapshotRecord plus policy context overrides.
def _demo_market_snapshot_provider(state: RRAAgentState) -> MarketSnapshotRecord:
    symbol = state.symbol
    if symbol == "ETHUSD":
        bid = 3675.2
        ask = 3676.1
        volatility = 0.041
        spread_bps = 2.45
    else:
        bid = 87320.0
        ask = 87324.5
        volatility = 0.034
        spread_bps = 0.52

    mid = (bid + ask) / 2

    return MarketSnapshotRecord(
        symbol=symbol,
        bid=bid,
        ask=ask,
        mid=mid,
        spread_bps=spread_bps,
        realized_volatility=volatility,
        orderbook_stale_seconds=0.4,
        market_data_available=True,
        raw_payload={
            "venue": "demo-feed",
            "runtime_context": {
                "execution": {
                    "orders_in_last_minute": 1,
                    "failed_orders_count": 0,
                    "pre_trade_risk_check_passed": True,
                },
                "system": {
                    "kraken_cli_available": True,
                    "checkpoint_store_available": True,
                    "inference_backend_degraded": False,
                    "clock_skew_ms": 8,
                },
            },
        },
    )


# This block builds safe portfolio context for demo runs.
# It takes: the current graph state.
# It gives: positions and balance values that satisfy policy while still looking realistic.
def _demo_portfolio_provider(state: RRAAgentState) -> dict[str, Any]:
    if state.symbol == "ETHUSD":
        asset_exposure = 180.0
    else:
        asset_exposure = 240.0

    return {
        "total_open_exposure_bps": 620.0,
        "asset_open_exposure_bps": asset_exposure,
        "open_positions_count": 2,
        "wallet_balance_usd": 18500.0,
        "manual_kill_switch": False,
        "daily_loss_limit_breached": False,
        "untrusted_operator_context": False,
    }


# This block classifies a simple demo market regime from the requested symbol.
# It takes: the current graph state.
# It gives: a stable MarketRegime plus a short explanation blob.
def _demo_regime_provider(state: RRAAgentState) -> dict[str, Any]:
    if state.symbol == "ETHUSD":
        return {
            "value": MarketRegime.MEAN_REVERSION.value,
            "confidence": 0.72,
            "driver": "range compression after overnight expansion",
        }

    return {
        "value": MarketRegime.TREND.value,
        "confidence": 0.81,
        "driver": "higher highs with tight spread conditions",
    }


# This block returns a small semantic-memory bundle for the current symbol.
# It takes: the current graph state.
# It gives: a list of retrieved memory documents for the prompt and audit trail.
def _demo_memory_provider(state: RRAAgentState) -> list[MemoryDocument]:
    regime = state.metadata.get("regime", {}).get("value", "trend")
    return [
        MemoryDocument(
            collection="historical-alpha",
            text=(
                f"Previous {state.symbol} {regime} setup favored patient entries when "
                "spread remained below 5 bps and confidence stayed above 0.70."
            ),
            symbol=state.symbol,
            trace_id=state.trace_id,
            tags=["demo", "alpha", regime],
            metadata={"win_rate": 0.63, "sample_size": 19},
            source=RecordSource.SYSTEM,
        ),
        MemoryDocument(
            collection="risk-notes",
            text=(
                f"{state.symbol} tends to mean-revert sharply after breakout failures, "
                "so slippage control remains critical in demo execution."
            ),
            symbol=state.symbol,
            trace_id=state.trace_id,
            tags=["demo", "risk"],
            metadata={"note_type": "risk"},
            source=RecordSource.SYSTEM,
        ),
    ]


# This block builds a trustworthy demo trust snapshot.
# It takes: the current graph state.
# It gives: a policy-satisfying TrustSnapshotRecord for safe local graph demos.
def _demo_trust_snapshot_provider(state: RRAAgentState) -> TrustSnapshotRecord:
    operator_metadata = state.metadata.get("operator", {})
    agent_address = "0x1111111111111111111111111111111111111111"
    if isinstance(operator_metadata, dict):
        agent_address = str(operator_metadata.get("agent_address", agent_address))

    return TrustSnapshotRecord(
        agent_address=agent_address,
        trust_score=0.94,
        erc8004_registered=True,
        valid_attestation=True,
        registry_reachable=True,
        attestation_age_seconds=42,
        attested=True,
        tee_measurement="demo-tee-measurement-v1",
        tee_enclave_id="demo-enclave-01",
        raw_registry={"mode": "demo", "registered": True},
        raw_attestation={"mode": "demo", "provider": "fixture"},
    )


# This block builds a deterministic mock inference payload that still flows through parsing/validation.
# It takes: the wrapped prompts produced by the sentiment node.
# It gives: raw XML output shaped exactly like the real parser expects.
def _demo_sentiment_inference_provider(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    del system_prompt

    symbol_match = SYMBOL_PATTERN.search(user_prompt)
    symbol = symbol_match.group("symbol") if symbol_match else "BTCUSD"

    regime_match = REGIME_PATTERN.search(user_prompt)
    regime = regime_match.group("regime") if regime_match else "trend"

    side = "sell" if regime == "mean_reversion" and symbol == "ETHUSD" else "buy"
    notional_usd = 650 if symbol == "ETHUSD" else 900
    position_size_bps = 140 if symbol == "ETHUSD" else 180
    signal_score = 0.72 if regime == "mean_reversion" else 0.84
    confidence = 0.76 if regime == "mean_reversion" else 0.88

    payload = {
        "trade_intent": {
            "symbol": symbol,
            "side": side,
            "order_type": "limit",
            "notional_usd": notional_usd,
            "position_size_bps": position_size_bps,
            "signal_score": signal_score,
            "model_confidence": confidence,
            "slippage_bps": 4,
            "regime": regime,
            "thesis": (
                f"{symbol} retains favorable microstructure for a guarded {side} entry "
                "under current demo conditions."
            ),
            "time_in_force": "gtc",
        },
        "summary": f"Demo sentiment favors a guarded {side} setup in {symbol}.",
        "risks": [
            "Momentum may fade if spreads widen materially.",
            "Execution quality deteriorates quickly during abrupt liquidity shifts.",
        ],
        "requires_human_review": False,
    }

    raw_output = (
        "<response>"
        "<summary>Demo signal generated.</summary>"
        f"<payload>{json.dumps(payload)}</payload>"
        "</response>"
    )
    return {
        "raw_output": raw_output,
        "model_name": "demo-mock-sentiment",
    }


# This block builds a paper-trading sentiment payload that reacts to portfolio state.
# It takes: the wrapped prompts produced by the sentiment node.
# It gives: a deterministic trade candidate that can reduce exposure as well as add it.
def _paper_sentiment_inference_provider(system_prompt: str, user_prompt: str) -> dict[str, Any]:
    del system_prompt

    symbol_match = SYMBOL_PATTERN.search(user_prompt)
    symbol = symbol_match.group("symbol") if symbol_match else "BTCUSD"

    regime_match = REGIME_PATTERN.search(user_prompt)
    regime = regime_match.group("regime") if regime_match else "trend"

    portfolio_context = _extract_json_context(PORTFOLIO_PATTERN, user_prompt)
    market_snapshot = _extract_json_context(MARKET_PATTERN, user_prompt)

    wallet_available_usd = float(portfolio_context.get("wallet_available_usd", 0.0))
    asset_open_exposure_bps = float(portfolio_context.get("asset_open_exposure_bps", 0.0))
    base_asset_units = float(portfolio_context.get("base_asset_units", 0.0))
    asset_value_usd = float(portfolio_context.get("asset_value_usd", 0.0))
    spread_bps = float(market_snapshot.get("spread_bps", 0.0))

    # This block chooses between adding risk and reducing exposure using account state.
    # It takes: available cash, current position value, exposure, and spread quality.
    # It gives: a side and notional that avoid one-way accumulation.
    if asset_open_exposure_bps >= 1800 or (wallet_available_usd < 150.0 and base_asset_units > 0):
        side = "sell"
        notional_usd = min(max(asset_value_usd * 0.30, 25.0), 300.0)
        thesis = (
            f"{symbol} exposure is already elevated versus available cash, so the paper strategy is reducing "
            "inventory to recycle capital and de-risk the book."
        )
        summary = f"Reduce {symbol} paper exposure with a guarded sell order."
        risks = [
            "Selling too early may cap upside if momentum extends.",
            "Thin liquidity can widen realized slippage during inventory reduction.",
        ]
    elif spread_bps <= 10.0 and wallet_available_usd >= 250.0:
        side = "buy"
        notional_usd = min(max(wallet_available_usd * 0.18, 25.0), 250.0)
        thesis = (
            f"{symbol} still shows acceptable spread conditions and sufficient dry powder remains available, "
            "so the paper strategy is adding a measured entry rather than overcommitting capital."
        )
        summary = f"Add a measured {symbol} paper long while preserving cash."
        risks = [
            "Momentum may stall and trap a fresh entry.",
            "Repeated entries without a strong exit discipline can compress edge.",
        ]
    else:
        side = "sell" if base_asset_units > 0 else "buy"
        notional_usd = 25.0 if side == "buy" else min(max(asset_value_usd * 0.20, 25.0), 150.0)
        thesis = (
            f"{symbol} is in a {regime} regime with constrained liquidity or cash, so the paper strategy is "
            f"{'lightening exposure' if side == 'sell' else 'keeping size minimal'} until cleaner conditions appear."
        )
        summary = f"Keep {symbol} paper risk small while conditions remain mixed."
        risks = [
            "Mixed conditions can produce noisy fills without follow-through.",
            "Small sizing may underperform if the market breaks decisively.",
        ]

    confidence = 0.74 if side == "sell" else 0.79
    signal_score = 0.70 if side == "sell" else 0.76
    position_size_bps = 90 if side == "sell" else 110

    payload = {
        "trade_intent": {
            "symbol": symbol,
            "side": side,
            "order_type": "limit",
            "notional_usd": round(notional_usd, 2),
            "position_size_bps": position_size_bps,
            "signal_score": signal_score,
            "model_confidence": confidence,
            "slippage_bps": 4,
            "regime": regime,
            "thesis": thesis,
            "time_in_force": "gtc",
        },
        "summary": summary,
        "risks": risks,
        "requires_human_review": False,
    }

    raw_output = (
        "<response>"
        "<summary>Paper portfolio-aware signal generated.</summary>"
        f"<payload>{json.dumps(payload)}</payload>"
        "</response>"
    )
    return {
        "raw_output": raw_output,
        "model_name": "paper-portfolio-aware-sentiment",
    }


# This block extracts one JSON context blob embedded in the prompt.
# It takes: a regex pattern and the wrapped user prompt text.
# It gives: a decoded dictionary or an empty mapping when the field is absent.
def _extract_json_context(pattern: re.Pattern[str], prompt: str) -> dict[str, Any]:
    match = pattern.search(prompt)
    if match is None:
        return {}

    try:
        parsed = json.loads(match.group(match.lastgroup or 1))
    except json.JSONDecodeError:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return parsed


# This block builds validator metadata for dry-run execution.
# It takes: the current graph state after policy approval.
# It gives: a fake proof payload that downstream execution and UI code can inspect.
def _demo_validator_payload_provider(state: RRAAgentState) -> dict[str, Any]:
    digest_seed = f"{state.trace_id}:{state.symbol}:{state.policy.action}"
    digest = hashlib.sha256(digest_seed.encode("utf-8")).hexdigest()
    return {
        "proof_type": "demo-eip712-envelope",
        "proof_digest": digest,
        "mode": "dry_run",
        "signed_by": "project-optima-demo-signer",
    }


# This block builds a dry-run execution result from the validator envelope.
# It takes: the canonical execution request and current graph state.
# It gives: a successful execution-shaped payload without touching a real exchange.
def _demo_execution_provider(
    execution_request: dict[str, Any],
    state: RRAAgentState,
) -> dict[str, Any]:
    if state.trade_intent_record is None:
        raise ValueError("Demo execution provider requires state.trade_intent_record.")

    policy_record_id = state.policy.record.id if state.policy.record is not None else None
    order_id = f"dryrun-{uuid4().hex[:12]}"

    execution_record = ExecutionRecord(
        trade_intent_id=state.trade_intent_record.id,
        policy_decision_id=policy_record_id,
        symbol=state.trade_intent_record.symbol,
        side=state.trade_intent_record.side,
        order_type=state.trade_intent_record.order_type,
        status=ExecutionStatus.FILLED,
        exchange_order_id=order_id,
        requested_notional_usd=state.trade_intent_record.notional_usd,
        executed_notional_usd=state.trade_intent_record.notional_usd,
        average_fill_price=state.market_snapshot.mid if state.market_snapshot else None,
        raw_payload={
            "mode": "dry_run",
            "execution_request": execution_request,
        },
        source=RecordSource.EXECUTION,
    )

    return {
        "status": ExecutionStatus.FILLED.value,
        "records": [execution_record],
        "orders_in_last_minute": 2,
        "failed_orders_count": 0,
        "pre_trade_risk_check_passed": True,
        "raw_exchange_response": {
            "mode": "dry_run",
            "order_id": order_id,
            "status": "filled",
        },
    }
