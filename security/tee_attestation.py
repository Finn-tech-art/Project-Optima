from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.core.constants import (
    DEFAULT_ATTESTATION_TTL_SECONDS,
    DEFAULT_SUBPROCESS_TIMEOUT_SECONDS,
)
from backend.core.exceptions import (
    AttestationError,
    ConfigurationError,
    SubprocessExecutionError,
)


@dataclass(slots=True, frozen=True)
class TEEAttestationConfig:
    """
    This block defines how attestation validation should behave.
    It takes: script location, timeout settings, freshness limits, and optional trusted measurements.
    It gives: one normalized config object for the attestation service.
    """

    script_path: Path
    runner: str = "bash"
    timeout_seconds: float = DEFAULT_SUBPROCESS_TIMEOUT_SECONDS
    max_age_seconds: int = DEFAULT_ATTESTATION_TTL_SECONDS
    trusted_measurements: frozenset[str] = field(default_factory=frozenset)
    require_measurement_match: bool = False


@dataclass(slots=True, frozen=True)
class TEEAttestationResult:
    """
    This block represents the normalized output of attestation validation.
    It takes: parsed attestation fields plus validation outcomes.
    It gives: a policy-friendly result object with both normalized and raw data.
    """

    attested: bool
    valid_attestation: bool
    enclave_id: str | None
    measurement: str | None
    issued_at: datetime | None
    age_seconds: int | None
    quote_digest: str | None
    signer: str | None
    raw: dict[str, Any]

    def to_policy_context(self) -> dict[str, Any]:
        """
        This block converts the result into the trust fields expected by policy.
        It takes: the normalized attestation result.
        It gives: a compact mapping suitable for `context["trust"]`.
        """
        return {
            "valid_attestation": self.valid_attestation,
            "attestation_age_seconds": self.age_seconds or 0,
            "attested": self.attested,
            "tee_measurement": self.measurement,
            "tee_enclave_id": self.enclave_id,
        }


class TEEAttestationService:
    """
    This block is the main attestation adapter.
    It takes: a TEEAttestationConfig.
    It gives: file-based validation, raw payload validation, and shell-script enrollment execution.
    """

    def __init__(self, config: TEEAttestationConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """
        This block validates the attestation service configuration.
        It takes: the config provided by the caller.
        It gives: early failure if required paths or numeric limits are invalid.
        """
        if not self.config.script_path:
            raise ConfigurationError("TEE attestation script path is required.")

        if self.config.timeout_seconds <= 0:
            raise ConfigurationError("TEE attestation timeout must be greater than 0.")

        if self.config.max_age_seconds < 0:
            raise ConfigurationError("TEE attestation max age must be greater than or equal to 0.")

    def load_attestation_file(self, path: str | Path) -> TEEAttestationResult:
        """
        This block loads attestation evidence from disk.
        It takes: a JSON file path containing attestation evidence.
        It gives: a validated TEEAttestationResult.
        """
        source_path = Path(path)
        if not source_path.exists():
            raise ConfigurationError(
                "TEE attestation file does not exist.",
                context={"path": str(source_path)},
            )

        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise AttestationError(
                "TEE attestation file is not valid JSON.",
                context={"path": str(source_path), "error": str(error)},
            ) from error

        if not isinstance(payload, dict):
            raise AttestationError(
                "TEE attestation payload must be a JSON object.",
                context={"path": str(source_path)},
            )

        return self.validate_payload(payload)

    def validate_payload(self, payload: dict[str, Any]) -> TEEAttestationResult:
        """
        This block validates raw attestation evidence.
        It takes: a decoded attestation JSON object.
        It gives: a normalized result object with freshness and measurement checks applied.
        """
        if not isinstance(payload, dict):
            raise AttestationError("TEE attestation payload must be a mapping.")

        # This block reads the core identity fields from the payload.
        # It takes: common attestation keys emitted by the attestation provider or wrapper script.
        # It gives: normalized values for later freshness and integrity checks.
        enclave_id = self._read_optional_str(payload, "enclave_id")
        measurement = self._read_optional_str(payload, "measurement")
        quote_digest = self._read_optional_str(payload, "quote_digest")
        signer = self._read_optional_str(payload, "signer")
        attested = bool(payload.get("attested", True))

        # This block parses the issuance timestamp if present.
        # It takes: `issued_at` as an ISO-8601 string.
        # It gives: a timezone-aware datetime and computed age in seconds.
        issued_at = self._parse_timestamp(payload.get("issued_at"))
        age_seconds = self._compute_age_seconds(issued_at)

        # This block evaluates freshness requirements.
        # It takes: the computed evidence age and configured max age.
        # It gives: a boolean freshness result and a structured failure if too old.
        if age_seconds is not None and age_seconds > self.config.max_age_seconds:
            raise AttestationError(
                "TEE attestation evidence is too old.",
                context={
                    "age_seconds": age_seconds,
                    "max_age_seconds": self.config.max_age_seconds,
                    "enclave_id": enclave_id,
                },
            )

        # This block optionally enforces trusted measurement matching.
        # It takes: the reported enclave measurement and the configured allowlist.
        # It gives: an integrity guarantee that only approved enclave builds are accepted.
        if self.config.require_measurement_match:
            if not measurement:
                raise AttestationError(
                    "TEE attestation is missing a measurement value.",
                    context={"enclave_id": enclave_id},
                )

            if (
                self.config.trusted_measurements
                and measurement not in self.config.trusted_measurements
            ):
                raise AttestationError(
                    "TEE attestation measurement is not trusted.",
                    context={
                        "measurement": measurement,
                        "trusted_measurements": sorted(self.config.trusted_measurements),
                    },
                )

        # This block determines the final validation verdict.
        # It takes: attested flag plus the successful completion of validation checks.
        # It gives: a normalized result object that policy code can consume directly.
        return TEEAttestationResult(
            attested=attested,
            valid_attestation=attested,
            enclave_id=enclave_id,
            measurement=measurement,
            issued_at=issued_at,
            age_seconds=age_seconds,
            quote_digest=quote_digest,
            signer=signer,
            raw=payload,
        )

    def run_enrollment(
        self,
        *,
        agent_id: str | None = None,
        output_path: str | Path | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        This block executes the shell-based TEE enrollment script.
        It takes: optional agent id, optional output path, extra CLI args, and env overrides.
        It gives: the decoded JSON payload emitted by the shell script.
        """
        command = [self.config.runner, str(self.config.script_path)]

        if agent_id:
            command.extend(["--agent-id", agent_id])

        if output_path is not None:
            command.extend(["--output", str(output_path)])

        if extra_args:
            command.extend(extra_args)

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as error:
            raise SubprocessExecutionError(
                "TEE attestation script timed out.",
                command=command,
                retryable=False,
                context={"timeout_seconds": self.config.timeout_seconds},
            ) from error
        except OSError as error:
            raise SubprocessExecutionError(
                "Failed to execute TEE attestation script.",
                command=command,
                retryable=False,
                context={"error": str(error)},
            ) from error

        if completed.returncode != 0:
            raise SubprocessExecutionError(
                "TEE attestation script exited with a non-zero status.",
                command=command,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                retryable=False,
            )

        stdout = completed.stdout.strip()
        if not stdout:
            return {}

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise AttestationError(
                "TEE attestation script did not return valid JSON.",
                context={"stdout": stdout, "stderr": completed.stderr},
            ) from error

        if not isinstance(payload, dict):
            raise AttestationError(
                "TEE attestation script returned an invalid payload type.",
                context={"payload_type": type(payload).__name__},
            )

        return payload

    def enroll_and_validate(
        self,
        *,
        agent_id: str | None = None,
        output_path: str | Path | None = None,
        extra_args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> TEEAttestationResult:
        """
        This block runs the shell script and immediately validates its output.
        It takes: the same enrollment parameters accepted by the script runner.
        It gives: one fully validated attestation result.
        """
        payload = self.run_enrollment(
            agent_id=agent_id,
            output_path=output_path,
            extra_args=extra_args,
            env=env,
        )
        return self.validate_payload(payload)

    def _read_optional_str(self, payload: dict[str, Any], key: str) -> str | None:
        """
        This block safely reads an optional string field from the payload.
        It takes: the payload and a target key.
        It gives: a trimmed string value or None.
        """
        value = payload.get(key)
        if value is None:
            return None
        if not isinstance(value, str):
            raise AttestationError(
                "TEE attestation field has an invalid type.",
                context={"field": key, "expected": "str", "actual": type(value).__name__},
            )
        normalized = value.strip()
        return normalized or None

    def _parse_timestamp(self, value: Any) -> datetime | None:
        """
        This block parses the attestation issuance timestamp.
        It takes: a raw `issued_at` value from the payload.
        It gives: a timezone-aware UTC datetime or None if the field is absent.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            raise AttestationError(
                "TEE attestation `issued_at` must be a string.",
                context={"actual": type(value).__name__},
            )

        normalized = value.strip()
        if not normalized:
            return None

        try:
            parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
        except ValueError as error:
            raise AttestationError(
                "TEE attestation `issued_at` is not a valid ISO timestamp.",
                context={"issued_at": value},
            ) from error

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)

        return parsed.astimezone(UTC)

    def _compute_age_seconds(self, issued_at: datetime | None) -> int | None:
        """
        This block computes the attestation age.
        It takes: the attestation issuance time.
        It gives: the evidence age in seconds relative to current UTC time.
        """
        if issued_at is None:
            return None

        now = datetime.now(UTC)
        delta = now - issued_at
        return max(int(delta.total_seconds()), 0)
