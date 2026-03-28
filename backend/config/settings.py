from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from backend.core.constants import (
    APP_NAME,
    APP_NAMESPACE,
    APP_VERSION,
    DEFAULT_ERC8004_CHAIN_ID,
    DEFAULT_GROQ_MAX_TOKENS,
    DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_GROQ_TEMPERATURE,
    DEFAULT_KRAKEN_CLI_BINARY,
    DEFAULT_KRAKEN_CLI_OUTPUT_FORMAT,
    DEFAULT_KRAKEN_CLI_TIMEOUT_SECONDS,
    DEFAULT_LOG_LEVEL,
    DEFAULT_LOGGER_NAME,
    ENV_ERC8004_REGISTRY_ADDRESS,
    ENV_ERC8004_RPC_URL,
    ENV_GROQ_API_KEY,
    ENV_KRAKEN_CLI_PATH,
    ENV_KRAKEN_PROFILE,
    ENV_LANGGRAPH_CHECKPOINT_DB,
    ENV_LOG_LEVEL,
)


# This block defines the main runtime settings model.
# It takes: values from the process environment and optional `.env` file.
# It gives: one typed configuration object for the backend runtime.
class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # This block defines application identity settings.
    # It takes: optional overrides from env vars if you later add them.
    # It gives: stable metadata that other modules can use in logs and telemetry.
    app_name: str = APP_NAME
    app_namespace: str = APP_NAMESPACE
    app_version: str = APP_VERSION
    environment: str = "development"

    # This block defines logging settings.
    # It takes: env-driven log level configuration.
    # It gives: consistent logger defaults for the whole backend.
    log_level: str = Field(default=DEFAULT_LOG_LEVEL, alias=ENV_LOG_LEVEL)
    logger_name: str = DEFAULT_LOGGER_NAME

    # This block defines Groq inference settings.
    # It takes: API key plus model and generation defaults from env or safe defaults.
    # It gives: one normalized Groq config surface for inference clients.
    groq_api_key: str | None = Field(default=None, alias=ENV_GROQ_API_KEY)
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    groq_fast_model: str | None = Field(
        default="groq/compound-mini",
        alias="GROQ_FAST_MODEL",
    )
    groq_temperature: float = Field(
        default=DEFAULT_GROQ_TEMPERATURE,
        alias="GROQ_TEMPERATURE",
    )
    groq_max_tokens: int = Field(
        default=DEFAULT_GROQ_MAX_TOKENS,
        alias="GROQ_MAX_TOKENS",
    )
    groq_timeout_seconds: float = Field(
        default=DEFAULT_GROQ_REQUEST_TIMEOUT_SECONDS,
        alias="GROQ_TIMEOUT_SECONDS",
    )

    # This block defines Kraken CLI execution settings.
    # It takes: CLI path, optional profile, and execution defaults.
    # It gives: a normalized execution config for the hardened order layer.
    kraken_cli_path: str = Field(
        default=DEFAULT_KRAKEN_CLI_BINARY,
        alias=ENV_KRAKEN_CLI_PATH,
    )
    kraken_profile: str | None = Field(default=None, alias=ENV_KRAKEN_PROFILE)
    kraken_output_format: str = Field(
        default=DEFAULT_KRAKEN_CLI_OUTPUT_FORMAT,
        alias="KRAKEN_OUTPUT_FORMAT",
    )
    kraken_timeout_seconds: float = Field(
        default=DEFAULT_KRAKEN_CLI_TIMEOUT_SECONDS,
        alias="KRAKEN_TIMEOUT_SECONDS",
    )

    # This block defines LangGraph persistence settings.
    # It takes: checkpoint database location from env or a repo-local default.
    # It gives: a durable state configuration for graph execution.
    langgraph_checkpoint_db: str = Field(
        default="backend/data/langgraph_checkpoints.db",
        alias=ENV_LANGGRAPH_CHECKPOINT_DB,
    )

    # This block defines ERC-8004 trust-layer settings.
    # It takes: RPC and contract address inputs from env plus a default chain id.
    # It gives: normalized config for the on-chain registry adapter.
    erc8004_rpc_url: str | None = Field(default=None, alias=ENV_ERC8004_RPC_URL)
    erc8004_registry_address: str | None = Field(
        default=None,
        alias=ENV_ERC8004_REGISTRY_ADDRESS,
    )
    erc8004_chain_id: int = Field(
        default=DEFAULT_ERC8004_CHAIN_ID,
        alias="ERC8004_CHAIN_ID",
    )
    erc8004_abi_path: str = Field(
        default="security/abi/erc8004_registry.json",
        alias="ERC8004_ABI_PATH",
    )

    # This block defines TEE attestation settings.
    # It takes: script path, provider metadata, and validation constraints from env/defaults.
    # It gives: one normalized config surface for attestation services.
    tee_script_path: str = Field(
        default="security/tee_attestation.sh",
        alias="TEE_SCRIPT_PATH",
    )
    tee_provider: str = Field(default="phala", alias="TEE_PROVIDER")
    tee_timeout_seconds: float = Field(default=15.0, alias="TEE_TIMEOUT_SECONDS")
    tee_max_age_seconds: int = Field(default=300, alias="TEE_MAX_AGE_SECONDS")

    # This block defines optional Phala endpoint-mode settings.
    # It takes: provider-specific env vars when remote attestation is used.
    # It gives: a clean config bridge into the security layer.
    phala_cloud_attestation_endpoint: str | None = Field(
        default=None,
        alias="PHALA_CLOUD_ATTESTATION_ENDPOINT",
    )
    phala_cloud_api_token: str | None = Field(
        default=None,
        alias="PHALA_CLOUD_API_TOKEN",
    )
    phala_cloud_project_id: str | None = Field(
        default=None,
        alias="PHALA_CLOUD_PROJECT_ID",
    )
    phala_cloud_cluster_id: str | None = Field(
        default=None,
        alias="PHALA_CLOUD_CLUSTER_ID",
    )
    phala_cloud_app_id: str | None = Field(
        default=None,
        alias="PHALA_CLOUD_APP_ID",
    )
    phala_cloud_enroll_command: str | None = Field(
        default=None,
        alias="PHALA_CLOUD_ENROLL_COMMAND",
    )

    # This block normalizes the checkpoint database path into a Path object.
    # It takes: the configured checkpoint DB string.
    # It gives: a pathlib object that persistence code can use directly.
    @property
    def checkpoint_db_path(self) -> Path:
        return Path(self.langgraph_checkpoint_db)

    # This block normalizes the TEE script path into a Path object.
    # It takes: the configured TEE shell script location.
    # It gives: a pathlib object for attestation bootstrap code.
    @property
    def tee_script(self) -> Path:
        return Path(self.tee_script_path)

    # This block returns the Groq-specific config needed by the inference adapter.
    # It takes: the loaded settings object.
    # It gives: a small dict suitable for client bootstrap.
    def groq_config(self) -> dict[str, Any]:
        return {
            "api_key": self.groq_api_key,
            "model": self.groq_model,
            "fast_model": self.groq_fast_model,
            "temperature": self.groq_temperature,
            "max_tokens": self.groq_max_tokens,
            "timeout_seconds": self.groq_timeout_seconds,
        }

    # This block returns the Kraken execution config used by subprocess callers.
    # It takes: the loaded settings object.
    # It gives: a small dict for the hardened Kraken execution layer.
    def kraken_config(self) -> dict[str, Any]:
        return {
            "cli_path": self.kraken_cli_path,
            "profile": self.kraken_profile,
            "output_format": self.kraken_output_format,
            "timeout_seconds": self.kraken_timeout_seconds,
        }

    # This block returns the LangGraph persistence config.
    # It takes: the loaded settings object.
    # It gives: a small dict for graph/checkpoint initialization.
    def langgraph_config(self) -> dict[str, Any]:
        return {
            "checkpoint_db": self.langgraph_checkpoint_db,
            "checkpoint_db_path": self.checkpoint_db_path,
        }

    # This block returns the ERC-8004 registry config used by the trust layer.
    # It takes: the loaded settings object.
    # It gives: the minimum contract/bootstrap settings for the registry client.
    def erc8004_config(self) -> dict[str, Any]:
        return {
            "rpc_url": self.erc8004_rpc_url,
            "registry_address": self.erc8004_registry_address,
            "chain_id": self.erc8004_chain_id,
            "abi_path": self.erc8004_abi_path,
        }

    # This block returns the TEE attestation config used by the security layer.
    # It takes: the loaded settings object.
    # It gives: the script, timeout, and provider metadata for attestation workflows.
    def tee_config(self) -> dict[str, Any]:
        return {
            "script_path": self.tee_script,
            "provider": self.tee_provider,
            "timeout_seconds": self.tee_timeout_seconds,
            "max_age_seconds": self.tee_max_age_seconds,
            "phala_cloud_attestation_endpoint": self.phala_cloud_attestation_endpoint,
            "phala_cloud_api_token": self.phala_cloud_api_token,
            "phala_cloud_project_id": self.phala_cloud_project_id,
            "phala_cloud_cluster_id": self.phala_cloud_cluster_id,
            "phala_cloud_app_id": self.phala_cloud_app_id,
            "phala_cloud_enroll_command": self.phala_cloud_enroll_command,
        }

    # This block returns a safe, partially redacted view of settings for logs/debugging.
    # It takes: the loaded settings object.
    # It gives: a dict that hides sensitive values but preserves enough context to debug config.
    def redacted_dict(self) -> dict[str, Any]:
        return {
            "app_name": self.app_name,
            "app_namespace": self.app_namespace,
            "app_version": self.app_version,
            "environment": self.environment,
            "log_level": self.log_level,
            "logger_name": self.logger_name,
            "groq_api_key": self._mask_secret(self.groq_api_key),
            "groq_model": self.groq_model,
            "groq_fast_model": self.groq_fast_model,
            "groq_temperature": self.groq_temperature,
            "groq_max_tokens": self.groq_max_tokens,
            "groq_timeout_seconds": self.groq_timeout_seconds,
            "kraken_cli_path": self.kraken_cli_path,
            "kraken_profile": self.kraken_profile,
            "kraken_output_format": self.kraken_output_format,
            "kraken_timeout_seconds": self.kraken_timeout_seconds,
            "langgraph_checkpoint_db": self.langgraph_checkpoint_db,
            "erc8004_rpc_url": self.erc8004_rpc_url,
            "erc8004_registry_address": self.erc8004_registry_address,
            "erc8004_chain_id": self.erc8004_chain_id,
            "erc8004_abi_path": self.erc8004_abi_path,
            "tee_script_path": self.tee_script_path,
            "tee_provider": self.tee_provider,
            "tee_timeout_seconds": self.tee_timeout_seconds,
            "tee_max_age_seconds": self.tee_max_age_seconds,
            "phala_cloud_attestation_endpoint": self.phala_cloud_attestation_endpoint,
            "phala_cloud_api_token": self._mask_secret(self.phala_cloud_api_token),
            "phala_cloud_project_id": self.phala_cloud_project_id,
            "phala_cloud_cluster_id": self.phala_cloud_cluster_id,
            "phala_cloud_app_id": self.phala_cloud_app_id,
            "phala_cloud_enroll_command": self.phala_cloud_enroll_command,
        }

    # This block masks secrets for safe logging.
    # It takes: a secret string or None.
    # It gives: a short redacted representation suitable for diagnostics.
    def _mask_secret(self, value: str | None) -> str | None:
        if value is None:
            return None
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"


# This block provides a cached singleton settings loader.
# It takes: no explicit arguments; it reads from env and `.env` once per process.
# It gives: one shared AppSettings object for the runtime.
@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
