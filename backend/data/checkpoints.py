from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver

from backend.core.constants import (
    DEFAULT_CHECKPOINT_NAMESPACE,
    DEFAULT_STATE_VERSION,
)
from backend.core.exceptions import ConfigurationError, PersistenceError
from backend.data.models import CheckpointRecord
from backend.data.postgres import PostgresClient


# This block defines the runtime configuration for LangGraph checkpoint persistence.
# It takes: the SQLite checkpoint DB path, namespace, and default state version.
# It gives: one normalized config object for checkpoint bootstrap and metadata recording.
@dataclass(slots=True, frozen=True)
class CheckpointConfig:
    db_path: Path
    checkpoint_ns: str = DEFAULT_CHECKPOINT_NAMESPACE
    state_version: int = DEFAULT_STATE_VERSION


# This block is the main checkpoint persistence helper for Project-Optima.
# It takes: a checkpoint config and an optional Postgres client for metadata mirroring.
# It gives: a LangGraph SQLite saver plus helper methods for checkpoint audit records.
@dataclass(slots=True)
class LangGraphCheckpointStore:
    config: CheckpointConfig
    postgres_client: PostgresClient | None = None
    _saver: SqliteSaver | None = field(default=None, init=False, repr=False)

    # This block validates the checkpoint configuration before use.
    # It takes: the configured DB path, namespace, and state version.
    # It gives: early failure if the persistence settings are invalid.
    def validate(self) -> None:
        if not str(self.config.db_path).strip():
            raise ConfigurationError("Checkpoint database path is required.")

        if not self.config.checkpoint_ns.strip():
            raise ConfigurationError("Checkpoint namespace is required.")

        if self.config.state_version <= 0:
            raise ConfigurationError("Checkpoint state version must be greater than 0.")

    # This block ensures the SQLite checkpoint directory exists.
    # It takes: the configured DB path.
    # It gives: a ready parent directory so the LangGraph saver can create the file safely.
    def ensure_directory(self) -> Path:
        self.validate()

        db_path = self.config.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return db_path

    # This block creates or returns the LangGraph SQLite saver.
    # It takes: the checkpoint DB path from config.
    # It gives: a reusable SqliteSaver for LangGraph graph compilation.
    def get_saver(self) -> SqliteSaver:
        if self._saver is not None:
            return self._saver

        db_path = self.ensure_directory()

        try:
            self._saver = SqliteSaver.from_conn_string(str(db_path))
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to initialize LangGraph SQLite checkpoint saver.",
                context={"db_path": str(db_path), "error": str(error)},
            ) from error

        return self._saver

    # This block builds a typed checkpoint metadata record.
    # It takes: the thread id, checkpoint id, and optional metadata from the graph layer.
    # It gives: a CheckpointRecord ready for PostgreSQL mirroring or later inspection.
    def build_checkpoint_record(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        metadata: dict[str, Any] | None = None,
        checkpoint_ns: str | None = None,
    ) -> CheckpointRecord:
        if not thread_id.strip():
            raise ConfigurationError("thread_id is required for checkpoint recording.")

        if not checkpoint_id.strip():
            raise ConfigurationError("checkpoint_id is required for checkpoint recording.")

        return CheckpointRecord(
            thread_id=thread_id,
            checkpoint_ns=checkpoint_ns or self.config.checkpoint_ns,
            checkpoint_id=checkpoint_id,
            state_version=self.config.state_version,
            metadata=metadata or {},
        )

    # This block mirrors checkpoint metadata into PostgreSQL if configured.
    # It takes: a typed CheckpointRecord created from graph runtime state.
    # It gives: the persisted checkpoint record id, or a structured failure if mirroring breaks.
    def record_checkpoint(self, record: CheckpointRecord) -> str:
        if self.postgres_client is None:
            raise ConfigurationError(
                "PostgreSQL client is required to record checkpoint metadata."
            )

        try:
            return self.postgres_client.upsert_checkpoint(record)
        except Exception as error:  # noqa: BLE001
            raise PersistenceError(
                "Failed to persist checkpoint metadata to PostgreSQL.",
                context={
                    "thread_id": record.thread_id,
                    "checkpoint_id": record.checkpoint_id,
                    "checkpoint_ns": record.checkpoint_ns,
                    "error": str(error),
                },
            ) from error

    # This block combines checkpoint-record creation and PostgreSQL mirroring.
    # It takes: thread id, checkpoint id, and optional graph metadata.
    # It gives: the typed CheckpointRecord after it has been mirrored when possible.
    def create_and_record_checkpoint(
        self,
        *,
        thread_id: str,
        checkpoint_id: str,
        metadata: dict[str, Any] | None = None,
        checkpoint_ns: str | None = None,
    ) -> CheckpointRecord:
        record = self.build_checkpoint_record(
            thread_id=thread_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
            checkpoint_ns=checkpoint_ns,
        )

        if self.postgres_client is not None:
            self.record_checkpoint(record)

        return record


# This block is a convenience factory for bootstrapping the LangGraph checkpoint store.
# It takes: a SQLite path and an optional Postgres client.
# It gives: a validated LangGraphCheckpointStore ready for graph setup.
def create_checkpoint_store(
    *,
    db_path: str | Path,
    postgres_client: PostgresClient | None = None,
    checkpoint_ns: str = DEFAULT_CHECKPOINT_NAMESPACE,
    state_version: int = DEFAULT_STATE_VERSION,
) -> LangGraphCheckpointStore:
    config = CheckpointConfig(
        db_path=Path(db_path),
        checkpoint_ns=checkpoint_ns,
        state_version=state_version,
    )
    store = LangGraphCheckpointStore(
        config=config,
        postgres_client=postgres_client,
    )
    store.validate()
    return store


# This block is a small convenience helper for callers that only need the SQLite saver.
# It takes: the SQLite checkpoint DB path.
# It gives: a ready SqliteSaver suitable for LangGraph graph compilation.
def create_sqlite_checkpointer(db_path: str | Path) -> SqliteSaver:
    store = create_checkpoint_store(db_path=db_path)
    return store.get_saver()
