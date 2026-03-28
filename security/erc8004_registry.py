from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from web3 import Web3
from web3.contract import Contract

from backend.core.constants import (
    DEFAULT_ERC8004_CHAIN_ID,
    DEFAULT_REGISTRY_CACHE_TTL_SECONDS,
)
from backend.core.exceptions import ConfigurationError, ERC8004RegistryError


@dataclass(slots=True, frozen=True)
class RegistryMethodMap:
    """
    This block defines which contract function names the client should call.
    It takes: the exact function names from the deployed ERC-8004 registry ABI.
    It gives: a stable alias layer so the rest of the app does not hard-code ABI names.
    """

    registration_check: str | None = None
    trust_score_lookup: str | None = None
    attestation_lookup: str | None = None
    agent_record_lookup: str | None = None


@dataclass(slots=True, frozen=True)
class RegistryConfig:
    """
    This block defines how the registry client connects to chain state.
    It takes: RPC URL, contract address, ABI, optional chain id, and method mappings.
    It gives: one validated config object that can build a registry client safely.
    """

    rpc_url: str
    contract_address: str
    abi: list[dict[str, Any]]
    chain_id: int = DEFAULT_ERC8004_CHAIN_ID
    request_timeout_seconds: float = 5.0
    cache_ttl_seconds: float = DEFAULT_REGISTRY_CACHE_TTL_SECONDS
    methods: RegistryMethodMap = field(default_factory=RegistryMethodMap)


@dataclass(slots=True, frozen=True)
class RegistryTrustContext:
    """
    This block is the normalized trust result returned to policy/evaluation layers.
    It takes: raw registry-derived values for one agent.
    It gives: a machine-friendly trust snapshot that downstream code can consume directly.
    """

    agent_address: str
    erc8004_registered: bool
    trust_score: float | None
    valid_attestation: bool
    registry_reachable: bool
    raw: dict[str, Any]

    def to_policy_context(self) -> dict[str, Any]:
        """
        This block converts registry output into the trust shape expected by policy code.
        It takes: the normalized trust snapshot.
        It gives: a policy-compatible mapping for `context["trust"]`.
        """
        return {
            "trust_score": self.trust_score or 0.0,
            "erc8004_registered": self.erc8004_registered,
            "valid_attestation": self.valid_attestation,
            "registry_reachable": self.registry_reachable,
        }


@dataclass(slots=True)
class _CacheEntry:
    """
    This block stores one cached registry lookup.
    It takes: a value and its expiration timestamp.
    It gives: a lightweight in-memory cache record.
    """

    value: Any
    expires_at: float


class ERC8004RegistryClient:
    """
    This block is the main on-chain registry adapter.
    It takes: a validated RegistryConfig.
    It gives: reachability checks, generic contract calls, and normalized trust lookups.
    """

    def __init__(self, config: RegistryConfig) -> None:
        self.config = config
        self._validate_config(config)

        # This block creates the Web3 transport and contract binding.
        # It takes: the RPC URL, request timeout, contract address, and ABI.
        # It gives: a ready contract object for typed function calls.
        self.web3 = Web3(
            Web3.HTTPProvider(
                config.rpc_url,
                request_kwargs={"timeout": config.request_timeout_seconds},
            )
        )
        self.contract = self._build_contract()

        # This block provides a tiny TTL cache for repeated registry reads.
        # It takes: method+argument lookup keys.
        # It gives: lower RPC overhead and reduced latency on hot paths.
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()

    @classmethod
    def from_abi_file(
        cls,
        *,
        rpc_url: str,
        contract_address: str,
        abi_path: str | Path,
        methods: RegistryMethodMap,
        chain_id: int = DEFAULT_ERC8004_CHAIN_ID,
        request_timeout_seconds: float = 5.0,
        cache_ttl_seconds: float = DEFAULT_REGISTRY_CACHE_TTL_SECONDS,
    ) -> ERC8004RegistryClient:
        """
        This block builds a client from a JSON ABI file on disk.
        It takes: connection details plus a path to the registry ABI JSON.
        It gives: a ready registry client without forcing callers to open the ABI manually.
        """
        path = Path(abi_path)
        if not path.exists():
            raise ConfigurationError(
                "ERC-8004 ABI file does not exist.",
                context={"path": str(path)},
            )

        try:
            abi = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ConfigurationError(
                "ERC-8004 ABI file is not valid JSON.",
                context={"path": str(path), "error": str(error)},
            ) from error

        if not isinstance(abi, list):
            raise ConfigurationError(
                "ERC-8004 ABI root must be a JSON array.",
                context={"path": str(path)},
            )

        config = RegistryConfig(
            rpc_url=rpc_url,
            contract_address=contract_address,
            abi=abi,
            chain_id=chain_id,
            request_timeout_seconds=request_timeout_seconds,
            cache_ttl_seconds=cache_ttl_seconds,
            methods=methods,
        )
        return cls(config)

    def _validate_config(self, config: RegistryConfig) -> None:
        """
        This block validates the minimum registry config shape.
        It takes: the user-provided RegistryConfig.
        It gives: early failure if critical connection details are missing or malformed.
        """
        if not config.rpc_url.strip():
            raise ConfigurationError("ERC-8004 registry RPC URL is required.")
        if not Web3.is_address(config.contract_address):
            raise ConfigurationError(
                "ERC-8004 registry contract address is invalid.",
                context={"contract_address": config.contract_address},
            )
        if not isinstance(config.abi, list) or not config.abi:
            raise ConfigurationError("ERC-8004 registry ABI must be a non-empty list.")
        if config.cache_ttl_seconds < 0:
            raise ConfigurationError("Registry cache TTL must be greater than or equal to 0.")

    def _build_contract(self) -> Contract:
        """
        This block creates the bound contract instance.
        It takes: validated address and ABI config.
        It gives: a Web3 contract object ready for read-only function calls.
        """
        checksum_address = Web3.to_checksum_address(self.config.contract_address)
        return self.web3.eth.contract(address=checksum_address, abi=self.config.abi)

    def _normalize_agent_address(self, agent_address: str) -> str:
        """
        This block normalizes an agent wallet address.
        It takes: a raw agent address string from callers.
        It gives: a checksummed address for safe contract calls and cache keys.
        """
        if not Web3.is_address(agent_address):
            raise ERC8004RegistryError(
                "Agent address is invalid.",
                context={"agent_address": agent_address},
            )
        return Web3.to_checksum_address(agent_address)

    def _cache_key(self, method_name: str, args: tuple[Any, ...]) -> str:
        """
        This block generates a deterministic cache key for a lookup.
        It takes: the contract method name and the call arguments.
        It gives: a repeatable string key for the in-memory TTL cache.
        """
        return f"{method_name}:{repr(args)}"

    def _get_cached(self, key: str) -> Any | None:
        """
        This block fetches a live cached value if one exists.
        It takes: a cache key.
        It gives: the cached value or None if the entry is missing/expired.
        """
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            if time.monotonic() >= entry.expires_at:
                del self._cache[key]
                return None
            return entry.value

    def _set_cached(self, key: str, value: Any) -> None:
        """
        This block stores a value in the TTL cache.
        It takes: a cache key and the new value.
        It gives: a short-lived cached copy of the lookup result.
        """
        if self.config.cache_ttl_seconds == 0:
            return

        with self._cache_lock:
            self._cache[key] = _CacheEntry(
                value=value,
                expires_at=time.monotonic() + self.config.cache_ttl_seconds,
            )

    def _resolve_method_name(self, alias: str) -> str:
        """
        This block converts a high-level alias into a concrete ABI function name.
        It takes: one of the configured aliases like `registration_check`.
        It gives: the exact contract function name required for the call.
        """
        method_name = getattr(self.config.methods, alias)
        if not method_name:
            raise ConfigurationError(
                f"Registry method alias '{alias}' is not configured.",
                context={"alias": alias},
            )
        return method_name

    def _call_contract(self, method_name: str, *args: Any) -> Any:
        """
        This block performs a cached read-only contract call.
        It takes: the target method name and its arguments.
        It gives: the decoded on-chain result or raises a structured registry error.
        """
        cache_key = self._cache_key(method_name, args)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            contract_function = getattr(self.contract.functions, method_name)
        except AttributeError as error:
            raise ERC8004RegistryError(
                "Configured ERC-8004 method does not exist in the ABI.",
                context={"method_name": method_name},
            ) from error

        try:
            result = contract_function(*args).call()
        except Exception as error:  # noqa: BLE001
            raise ERC8004RegistryError(
                "ERC-8004 contract call failed.",
                context={"method_name": method_name, "args": list(args)},
            ) from error

        self._set_cached(cache_key, result)
        return result

    def ping(self) -> bool:
        """
        This block checks whether the registry transport is currently reachable.
        It takes: the configured Web3 provider and optional expected chain id.
        It gives: a simple boolean health signal for policy and readiness checks.
        """
        try:
            if not self.web3.is_connected():
                return False
            chain_id = int(self.web3.eth.chain_id)
            return chain_id == self.config.chain_id
        except Exception:  # noqa: BLE001
            return False

    def is_registered(self, agent_address: str) -> bool:
        """
        This block checks whether an agent is registered in ERC-8004.
        It takes: the agent wallet address.
        It gives: True if the registry says the agent is present, else False.
        """
        normalized_address = self._normalize_agent_address(agent_address)
        method_name = self._resolve_method_name("registration_check")
        result = self._call_contract(method_name, normalized_address)

        if isinstance(result, bool):
            return result
        if isinstance(result, int):
            return result > 0
        if isinstance(result, (bytes, str)):
            return bool(result)
        raise ERC8004RegistryError(
            "Unsupported registration response type from registry.",
            context={"method_name": method_name, "response_type": type(result).__name__},
        )

    def get_trust_score(self, agent_address: str) -> float | None:
        """
        This block fetches the registry trust score for an agent.
        It takes: the agent wallet address.
        It gives: a normalized float trust score, or None if the method is not configured.
        """
        normalized_address = self._normalize_agent_address(agent_address)
        method_name = self._resolve_method_name("trust_score_lookup")
        result = self._call_contract(method_name, normalized_address)

        if result is None:
            return None
        if isinstance(result, (int, float)):
            return float(result)
        raise ERC8004RegistryError(
            "Unsupported trust score response type from registry.",
            context={"method_name": method_name, "response_type": type(result).__name__},
        )

    def get_attestation(self, agent_address: str) -> Any:
        """
        This block fetches the attestation payload or marker for an agent.
        It takes: the agent wallet address.
        It gives: the raw contract response so higher layers can interpret attestation semantics.
        """
        normalized_address = self._normalize_agent_address(agent_address)
        method_name = self._resolve_method_name("attestation_lookup")
        return self._call_contract(method_name, normalized_address)

    def get_agent_record(self, agent_address: str) -> Any:
        """
        This block fetches the raw registry record for an agent.
        It takes: the agent wallet address.
        It gives: the raw ABI-decoded agent record for advanced consumers.
        """
        normalized_address = self._normalize_agent_address(agent_address)
        method_name = self._resolve_method_name("agent_record_lookup")
        return self._call_contract(method_name, normalized_address)

    def build_trust_context(self, agent_address: str) -> RegistryTrustContext:
        """
        This block builds a policy-friendly trust snapshot for one agent.
        It takes: the agent wallet address.
        It gives: registration, trust score, attestation presence, and reachability in one object.
        """
        normalized_address = self._normalize_agent_address(agent_address)
        reachable = self.ping()

        if not reachable:
            return RegistryTrustContext(
                agent_address=normalized_address,
                erc8004_registered=False,
                trust_score=None,
                valid_attestation=False,
                registry_reachable=False,
                raw={},
            )

        registered = self.is_registered(normalized_address)

        trust_score: float | None = None
        if self.config.methods.trust_score_lookup:
            trust_score = self.get_trust_score(normalized_address)

        attestation_value: Any = None
        valid_attestation = False
        if self.config.methods.attestation_lookup:
            attestation_value = self.get_attestation(normalized_address)
            valid_attestation = bool(attestation_value)

        raw: dict[str, Any] = {
            "registered": registered,
            "trust_score": trust_score,
            "attestation": attestation_value,
        }

        if self.config.methods.agent_record_lookup:
            raw["agent_record"] = self.get_agent_record(normalized_address)

        return RegistryTrustContext(
            agent_address=normalized_address,
            erc8004_registered=registered,
            trust_score=trust_score,
            valid_attestation=valid_attestation,
            registry_reachable=True,
            raw=raw,
        )
