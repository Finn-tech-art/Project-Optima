from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from backend.core.constants import (
    ENV_ERC8004_REGISTRY_ADDRESS,
    ENV_ERC8004_RPC_URL,
    ENV_GROQ_API_KEY,
    ENV_KRAKEN_CLI_PATH,
    ENV_KRAKEN_PROFILE,
    ENV_LANGGRAPH_CHECKPOINT_DB,
)
from backend.core.exceptions import ConfigurationError, MissingEnvironmentVariableError


# This block defines environment-variable names that are specific to the TEE/Phala flow.
# It takes: no runtime input beyond the process environment.
# It gives: one canonical place to reference security provider env keys.
ENV_PHALA_CLOUD_ATTESTATION_ENDPOINT = "PHALA_CLOUD_ATTESTATION_ENDPOINT"
ENV_PHALA_CLOUD_API_TOKEN = "PHALA_CLOUD_API_TOKEN"
ENV_PHALA_CLOUD_PROJECT_ID = "PHALA_CLOUD_PROJECT_ID"
ENV_PHALA_CLOUD_CLUSTER_ID = "PHALA_CLOUD_CLUSTER_ID"
ENV_PHALA_CLOUD_APP_ID = "PHALA_CLOUD_APP_ID"
ENV_PHALA_CLOUD_ENROLL_COMMAND = "PHALA_CLOUD_ENROLL_COMMAND"


# This block lists which secret-like fields should always be redacted in logs/debug views.
# It takes: field names from the AppSecrets dataclass.
# It gives: a consistent rule for safe serialization.
REDACTED_FIELDS = frozenset(
    {
        "groq_api_key",
        "phala_cloud_api_token",
    }
)


@dataclass(slots=True, frozen=True)
class SecretsConfig:
    """
    This block defines how secrets should be loaded.
    It takes: an optional .env path and whether loading from .env is enabled.
    It gives: one normalized config object for the secrets manager.
    """

    env_file: Path | None = Path(".env")
    load_dotenv_file: bool = True
    dotenv_override: bool = False


@dataclass(slots=True, frozen=True)
class AppSecrets:
    """
    This block is the normalized secret/config bundle for Project-Optima.
    It takes: environment-derived values for inference, execution, persistence, registry, and TEE layers.
    It gives: one typed object that the rest of the backend can depend on safely.
    """

    groq_api_key: str | None = None
    kraken_cli_path: str | None = None
    kraken_profile: str | None = None
    langgraph_checkpoint_db: str | None = None
    erc8004_rpc_url: str | None = None
    erc8004_registry_address: str | None = None
    phala_cloud_attestation_endpoint: str | None = None
    phala_cloud_api_token: str | None = None
    phala_cloud_project_id: str | None = None
    phala_cloud_cluster_id: str | None = None
    phala_cloud_app_id: str | None = None
    phala_cloud_enroll_command: str | None = None

    def redacted_dict(self) -> dict[str, Any]:
        """
        This block returns a safe debug representation of the secret bundle.
        It takes: the current AppSecrets object.
        It gives: a dict with sensitive values masked for logging or diagnostics.
        """
        data = {
            "groq_api_key": self.groq_api_key,
            "kraken_cli_path": self.kraken_cli_path,
            "kraken_profile": self.kraken_profile,
            "langgraph_checkpoint_db": self.langgraph_checkpoint_db,
            "erc8004_rpc_url": self.erc8004_rpc_url,
            "erc8004_registry_address": self.erc8004_registry_address,
            "phala_cloud_attestation_endpoint": self.phala_cloud_attestation_endpoint,
            "phala_cloud_api_token": self.phala_cloud_api_token,
            "phala_cloud_project_id": self.phala_cloud_project_id,
            "phala_cloud_cluster_id": self.phala_cloud_cluster_id,
            "phala_cloud_app_id": self.phala_cloud_app_id,
            "phala_cloud_enroll_command": self.phala_cloud_enroll_command,
        }

        return {
            key: self._redact_value(key, value)
            for key, value in data.items()
        }

    def require_groq(self) -> AppSecrets:
        """
        This block validates the secrets needed for Groq inference.
        It takes: the current AppSecrets object.
        It gives: the same object back if validation passes, otherwise it raises.
        """
        _require_value(ENV_GROQ_API_KEY, self.groq_api_key)
        return self

    def require_kraken(self) -> AppSecrets:
        """
        This block validates the settings needed for Kraken CLI execution.
        It takes: the current AppSecrets object.
        It gives: the same object back if validation passes, otherwise it raises.
        """
        _require_value(ENV_KRAKEN_CLI_PATH, self.kraken_cli_path)
        return self

    def require_langgraph(self) -> AppSecrets:
        """
        This block validates the settings needed for LangGraph checkpoint persistence.
        It takes: the current AppSecrets object.
        It gives: the same object back if validation passes, otherwise it raises.
        """
        _require_value(ENV_LANGGRAPH_CHECKPOINT_DB, self.langgraph_checkpoint_db)
        return self

    def require_erc8004(self) -> AppSecrets:
        """
        This block validates the settings needed for ERC-8004 registry access.
        It takes: the current AppSecrets object.
        It gives: the same object back if validation passes, otherwise it raises.
        """
        _require_value(ENV_ERC8004_RPC_URL, self.erc8004_rpc_url)
        _require_value(ENV_ERC8004_REGISTRY_ADDRESS, self.erc8004_registry_address)
        return self

    def require_tee(self) -> AppSecrets:
        """
        This block validates the settings needed for TEE attestation.
        It takes: the current AppSecrets object.
        It gives: the same object back if either command mode or endpoint mode is fully configured.
        """
        has_command_mode = bool(self.phala_cloud_enroll_command)
        has_endpoint_mode = bool(self.phala_cloud_attestation_endpoint)

        if not has_command_mode and not has_endpoint_mode:
            raise MissingEnvironmentVariableError(
                "TEE attestation requires PHALA_CLOUD_ENROLL_COMMAND or "
                "PHALA_CLOUD_ATTESTATION_ENDPOINT.",
                context={
                    "required_any_of": [
                        ENV_PHALA_CLOUD_ENROLL_COMMAND,
                        ENV_PHALA_CLOUD_ATTESTATION_ENDPOINT,
                    ]
                },
            )

        if has_endpoint_mode:
            _require_value(
                ENV_PHALA_CLOUD_API_TOKEN,
                self.phala_cloud_api_token,
            )
            _require_value(
                ENV_PHALA_CLOUD_PROJECT_ID,
                self.phala_cloud_project_id,
            )

        return self

    def kraken_env(self) -> dict[str, str]:
        """
        This block builds the child-process environment for Kraken CLI calls.
        It takes: the current AppSecrets object.
        It gives: only the Kraken-relevant environment variables for subprocess execution.
        """
        env: dict[str, str] = {}
        if self.kraken_cli_path:
            env[ENV_KRAKEN_CLI_PATH] = self.kraken_cli_path
        if self.kraken_profile:
            env[ENV_KRAKEN_PROFILE] = self.kraken_profile
        return env

    def tee_env(self) -> dict[str, str]:
        """
        This block builds the child-process environment for TEE enrollment/attestation.
        It takes: the current AppSecrets object.
        It gives: only the provider-specific environment variables needed by tee_attestation.sh.
        """
        env: dict[str, str] = {}

        mapping = {
            ENV_PHALA_CLOUD_ATTESTATION_ENDPOINT: self.phala_cloud_attestation_endpoint,
            ENV_PHALA_CLOUD_API_TOKEN: self.phala_cloud_api_token,
            ENV_PHALA_CLOUD_PROJECT_ID: self.phala_cloud_project_id,
            ENV_PHALA_CLOUD_CLUSTER_ID: self.phala_cloud_cluster_id,
            ENV_PHALA_CLOUD_APP_ID: self.phala_cloud_app_id,
            ENV_PHALA_CLOUD_ENROLL_COMMAND: self.phala_cloud_enroll_command,
        }

        for key, value in mapping.items():
            if value:
                env[key] = value

        return env

    def registry_config_inputs(self) -> dict[str, str]:
        """
        This block builds the minimum input map needed by the ERC-8004 registry client.
        It takes: the current AppSecrets object.
        It gives: a simple config payload for registry bootstrap code.
        """
        self.require_erc8004()
        return {
            "rpc_url": self.erc8004_rpc_url or "",
            "contract_address": self.erc8004_registry_address or "",
        }

    @staticmethod
    def _redact_value(field_name: str, value: str | None) -> str | None:
        """
        This block redacts sensitive values while leaving non-sensitive values readable.
        It takes: a field name and its string value.
        It gives: either the original value or a masked version.
        """
        if value is None:
            return None

        if field_name not in REDACTED_FIELDS:
            return value

        if len(value) <= 8:
            return "***"

        return f"{value[:4]}...{value[-4:]}"


class SecretsManager:
    """
    This block is the runtime loader for Project-Optima secrets.
    It takes: a SecretsConfig that controls .env loading behavior.
    It gives: a validated AppSecrets object loaded from the current process environment.
    """

    def __init__(self, config: SecretsConfig | None = None) -> None:
        self.config = config or SecretsConfig()

    def load(self) -> AppSecrets:
        """
        This block loads all known security and runtime settings from the environment.
        It takes: the configured .env loading rules plus the current process env.
        It gives: one normalized AppSecrets object.
        """
        self._maybe_load_dotenv()

        return AppSecrets(
            groq_api_key=_read_env(ENV_GROQ_API_KEY),
            kraken_cli_path=_read_env(ENV_KRAKEN_CLI_PATH),
            kraken_profile=_read_env(ENV_KRAKEN_PROFILE),
            langgraph_checkpoint_db=_read_env(ENV_LANGGRAPH_CHECKPOINT_DB),
            erc8004_rpc_url=_read_env(ENV_ERC8004_RPC_URL),
            erc8004_registry_address=_read_env(ENV_ERC8004_REGISTRY_ADDRESS),
            phala_cloud_attestation_endpoint=_read_env(ENV_PHALA_CLOUD_ATTESTATION_ENDPOINT),
            phala_cloud_api_token=_read_env(ENV_PHALA_CLOUD_API_TOKEN),
            phala_cloud_project_id=_read_env(ENV_PHALA_CLOUD_PROJECT_ID),
            phala_cloud_cluster_id=_read_env(ENV_PHALA_CLOUD_CLUSTER_ID),
            phala_cloud_app_id=_read_env(ENV_PHALA_CLOUD_APP_ID),
            phala_cloud_enroll_command=_read_env(ENV_PHALA_CLOUD_ENROLL_COMMAND),
        )

    def load_and_validate(
        self,
        *,
        require_groq: bool = False,
        require_kraken: bool = False,
        require_langgraph: bool = False,
        require_erc8004: bool = False,
        require_tee: bool = False,
    ) -> AppSecrets:
        """
        This block loads secrets and validates only the subsystems the caller needs.
        It takes: boolean flags for required subsystems.
        It gives: one AppSecrets object that has already passed the requested checks.
        """
        secrets = self.load()

        if require_groq:
            secrets.require_groq()
        if require_kraken:
            secrets.require_kraken()
        if require_langgraph:
            secrets.require_langgraph()
        if require_erc8004:
            secrets.require_erc8004()
        if require_tee:
            secrets.require_tee()

        return secrets

    def _maybe_load_dotenv(self) -> None:
        """
        This block optionally loads variables from a .env file.
        It takes: the SecretsConfig object.
        It gives: a hydrated process environment before secret extraction begins.
        """
        if not self.config.load_dotenv_file:
            return

        env_file = self.config.env_file
        if env_file is None:
            return

        load_dotenv(dotenv_path=env_file, override=self.config.dotenv_override)


def _read_env(name: str) -> str | None:
    """
    This block reads and normalizes one environment variable.
    It takes: the variable name.
    It gives: a stripped string value or None if the value is missing/blank.
    """
    value = os.getenv(name)
    if value is None:
        return None

    normalized = value.strip()
    return normalized or None


def _require_value(name: str, value: str | None) -> None:
    """
    This block enforces that a required secret/config value is present.
    It takes: the environment variable name and its loaded value.
    It gives: silent success if present, otherwise a structured missing-variable error.
    """
    if value is None:
        raise MissingEnvironmentVariableError(
            f"Required environment variable is missing: {name}",
            context={"env_var": name},
        )


def load_secrets(
    *,
    env_file: str | Path | None = Path(".env"),
    require_groq: bool = False,
    require_kraken: bool = False,
    require_langgraph: bool = False,
    require_erc8004: bool = False,
    require_tee: bool = False,
) -> AppSecrets:
    """
    This block is the one-line convenience loader for callers.
    It takes: an optional env file path plus subsystem validation flags.
    It gives: a fully loaded AppSecrets object ready for use.
    """
    config = SecretsConfig(env_file=Path(env_file) if env_file is not None else None)
    manager = SecretsManager(config=config)
    return manager.load_and_validate(
        require_groq=require_groq,
        require_kraken=require_kraken,
        require_langgraph=require_langgraph,
        require_erc8004=require_erc8004,
        require_tee=require_tee,
    )
