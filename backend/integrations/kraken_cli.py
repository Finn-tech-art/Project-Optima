from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from backend.core.exceptions import KrakenCLIResponseError, KrakenCLIUnavailableError


# This block defines the execution modes supported by the Kraken CLI bridge.
# It takes: a desired safety/execution mode from the runtime.
# It gives: a constrained mode enum for live, validate-only, and paper trading.
class KrakenCLIExecutionMode(StrEnum):
    PAPER = "paper"
    VALIDATE = "validate"
    LIVE = "live"


# This block defines the runtime configuration for invoking Kraken CLI through WSL.
# It takes: the Windows-side command path, WSL distro, Linux-side CLI path, and timeout.
# It gives: one normalized config object for subprocess execution.
@dataclass(slots=True, frozen=True)
class KrakenCLIConfig:
    cli_path: str
    wsl_distro: str
    wsl_cli_path: str
    timeout_seconds: float = 10.0


# This block is the subprocess-backed Kraken CLI integration.
# It takes: a validated CLI config.
# It gives: JSON helpers for status, ticker, balances, open orders, and order submission.
@dataclass(slots=True)
class KrakenCLIClient:
    config: KrakenCLIConfig

    # This block checks exchange reachability through the Kraken CLI.
    # It takes: no runtime input beyond the configured CLI command.
    # It gives: the parsed JSON status payload from Kraken CLI.
    def status(self) -> dict[str, Any]:
        return self._run("status")

    # This block fetches ticker data for one or more pairs.
    # It takes: one trading pair string like BTCUSD.
    # It gives: the raw JSON ticker mapping returned by Kraken CLI.
    def ticker(self, pair: str) -> dict[str, Any]:
        return self._run("ticker", pair)

    # This block fetches live cash balances from Kraken CLI.
    # It takes: no runtime input beyond configured auth within the CLI.
    # It gives: the raw JSON balance payload.
    def balance(self) -> dict[str, Any]:
        return self._run("balance")

    # This block fetches open live orders from Kraken CLI.
    # It takes: no runtime input beyond configured auth within the CLI.
    # It gives: the raw JSON open-orders payload.
    def open_orders(self) -> dict[str, Any]:
        return self._run("open-orders")

    # This block initializes the local Kraken paper account.
    # It takes: the starting paper balance and currency.
    # It gives: the raw JSON init response from Kraken CLI.
    def paper_init(
        self,
        *,
        balance: float = 10_000.0,
        currency: str = "USD",
    ) -> dict[str, Any]:
        return self._run(
            "paper",
            "init",
            "--balance",
            str(balance),
            "--currency",
            currency,
        )

    # This block fetches paper-trading balances from Kraken CLI.
    # It takes: no runtime input beyond the local paper account state.
    # It gives: the raw JSON paper-balance payload.
    def paper_balance(self) -> dict[str, Any]:
        return self._run("paper", "balance")

    # This block fetches paper-trading status from Kraken CLI.
    # It takes: no runtime input beyond the local paper account state.
    # It gives: the raw JSON paper-status payload.
    def paper_status(self) -> dict[str, Any]:
        return self._run("paper", "status")

    # This block places or validates an order using the chosen CLI execution mode.
    # It takes: pair, side, order type, volume, optional price and tif, and execution mode.
    # It gives: the raw JSON order response from Kraken CLI.
    def place_order(
        self,
        *,
        pair: str,
        side: str,
        order_type: str,
        volume: float,
        price: float | None = None,
        time_in_force: str | None = None,
        mode: KrakenCLIExecutionMode = KrakenCLIExecutionMode.VALIDATE,
    ) -> dict[str, Any]:
        normalized_side = side.strip().lower()
        normalized_order_type = order_type.strip().lower()

        if normalized_side not in {"buy", "sell"}:
            raise ValueError(f"Unsupported order side: {side}")

        if mode == KrakenCLIExecutionMode.PAPER:
            args = [
                "paper",
                normalized_side,
                pair,
                str(volume),
                "--type",
                normalized_order_type,
            ]
        else:
            args = [
                "order",
                normalized_side,
                pair,
                str(volume),
                "--type",
                normalized_order_type,
            ]

        if price is not None and normalized_order_type != "market":
            args.extend(["--price", str(price)])

        # This block applies time-in-force only on live/validate order commands.
        # It takes: the optional caller-supplied time-in-force and selected execution mode.
        # It gives: CLI arguments that match the official Kraken CLI surface for each mode.
        if time_in_force and mode != KrakenCLIExecutionMode.PAPER:
            args.extend(["--timeinforce", time_in_force.upper()])

        if mode == KrakenCLIExecutionMode.VALIDATE:
            args.append("--validate")

        return self._run(*args)

    # This block runs the Kraken CLI and parses its JSON stdout.
    # It takes: the command arguments to pass through WSL into the Linux-side binary.
    # It gives: the decoded JSON payload or raises a structured CLI error.
    def _run(self, *args: str) -> dict[str, Any]:
        command = [
            self.config.cli_path,
            "-d",
            self.config.wsl_distro,
            "--exec",
            self.config.wsl_cli_path,
            *args,
            "-o",
            "json",
        ]

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as error:
            raise KrakenCLIUnavailableError(
                "Kraken CLI launcher executable was not found.",
                command=command,
                context={"cli_path": self.config.cli_path},
            ) from error
        except subprocess.TimeoutExpired as error:
            raise KrakenCLIResponseError(
                "Kraken CLI command timed out.",
                command=command,
                retryable=True,
                context={"timeout_seconds": self.config.timeout_seconds},
            ) from error
        except OSError as error:
            raise KrakenCLIUnavailableError(
                "Failed to execute Kraken CLI.",
                command=command,
                context={"error": str(error)},
            ) from error

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            raise KrakenCLIResponseError(
                "Kraken CLI command returned a non-zero exit code.",
                command=command,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            )

        if not stdout:
            return {}

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise KrakenCLIResponseError(
                "Kraken CLI did not return valid JSON.",
                command=command,
                stdout=stdout,
                stderr=stderr,
            ) from error

        if not isinstance(payload, dict):
            raise KrakenCLIResponseError(
                "Kraken CLI returned a JSON payload of the wrong type.",
                command=command,
                stdout=stdout,
                stderr=stderr,
                context={"payload_type": type(payload).__name__},
            )

        if payload.get("error") is not None:
            raise KrakenCLIResponseError(
                "Kraken CLI reported an error.",
                command=command,
                stdout=stdout,
                stderr=stderr,
                payload=payload,
            )

        return payload
