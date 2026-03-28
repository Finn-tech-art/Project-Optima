from __future__ import annotations

from dataclasses import dataclass

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Json

from backend.core.exceptions import ConfigurationError, PersistenceError
from backend.data.models import (
    CheckpointRecord,
    ExecutionRecord,
    MarketSnapshotRecord,
    PolicyAction,
    PolicyDecisionRecord,
    PolicyViolationRecord,
    TradeIntentRecord,
    TrustSnapshotRecord,
)


# This block defines the PostgreSQL connection configuration.
# It takes: the DSN, target schema, and optional connect timeout.
# It gives: one normalized config object for the Postgres client.
@dataclass(slots=True, frozen=True)
class PostgresConfig:
    dsn: str
    schema: str = "public"
    connect_timeout_seconds: int = 5


# This block is the main PostgreSQL adapter for structured system records.
# It takes: a validated PostgresConfig object.
# It gives: connection management, schema bootstrap, and typed persistence methods.
class PostgresClient:
    def __init__(self, config: PostgresConfig) -> None:
        self.config = config
        self._validate_config()
        self._connection: psycopg.Connection | None = None

    # This block validates the minimum client configuration.
    # It takes: the configured DSN, schema, and timeout values.
    # It gives: early failure if required connection settings are invalid.
    def _validate_config(self) -> None:
        if not self.config.dsn.strip():
            raise ConfigurationError("PostgreSQL DSN is required.")

        if not self.config.schema.strip():
            raise ConfigurationError("PostgreSQL schema name is required.")

        if self.config.connect_timeout_seconds <= 0:
            raise ConfigurationError("PostgreSQL connect timeout must be greater than 0.")

    # This block opens the PostgreSQL connection lazily.
    # It takes: the stored PostgresConfig.
    # It gives: a live psycopg connection ready for reads and writes.
    def connect(self) -> psycopg.Connection:
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg.connect(
                    self.config.dsn,
                    connect_timeout=self.config.connect_timeout_seconds,
                    row_factory=dict_row,
                    autocommit=False,
                )
            except psycopg.Error as error:
                raise PersistenceError(
                    "Failed to connect to PostgreSQL.",
                    context={"schema": self.config.schema, "error": str(error)},
                ) from error

        return self._connection

    # This block cleanly closes the PostgreSQL connection.
    # It takes: the current live connection, if any.
    # It gives: a released database connection and a reset client state.
    def close(self) -> None:
        if self._connection is not None and not self._connection.closed:
            self._connection.close()
        self._connection = None

    # This block allows the client to be used as a context manager.
    # It takes: standard context-manager entry.
    # It gives: the connected PostgresClient instance.
    def __enter__(self) -> PostgresClient:
        self.connect()
        return self

    # This block ensures the connection is closed on context-manager exit.
    # It takes: standard context-manager exit arguments.
    # It gives: a clean shutdown of the client connection.
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # This block bootstraps the database schema and required tables.
    # It takes: the configured schema name.
    # It gives: a ready PostgreSQL schema for all current Project-Optima records.
    def initialize_schema(self) -> None:
        connection = self.connect()
        schema_identifier = sql.Identifier(self.config.schema)

        statements = [
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema_identifier),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.trade_intents (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    trace_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    notional_usd DOUBLE PRECISION NOT NULL,
                    position_size_bps INTEGER NOT NULL,
                    signal_score DOUBLE PRECISION NOT NULL,
                    model_confidence DOUBLE PRECISION NOT NULL,
                    slippage_bps DOUBLE PRECISION NOT NULL,
                    regime TEXT NOT NULL,
                    thesis TEXT NULL,
                    time_in_force TEXT NOT NULL,
                    source TEXT NOT NULL,
                    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            ).format(schema_identifier),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.policy_decisions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    trade_intent_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    policy_name TEXT NOT NULL,
                    policy_version TEXT NOT NULL,
                    violations JSONB NOT NULL DEFAULT '[]'::jsonb,
                    source TEXT NOT NULL,
                    FOREIGN KEY (trade_intent_id) REFERENCES {}.trade_intents(id)
                )
                """
            ).format(schema_identifier, schema_identifier),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.trust_snapshots (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    agent_address TEXT NOT NULL,
                    trust_score DOUBLE PRECISION NULL,
                    erc8004_registered BOOLEAN NOT NULL,
                    valid_attestation BOOLEAN NOT NULL,
                    registry_reachable BOOLEAN NOT NULL,
                    attestation_age_seconds INTEGER NOT NULL,
                    attested BOOLEAN NOT NULL,
                    tee_measurement TEXT NULL,
                    tee_enclave_id TEXT NULL,
                    raw_registry JSONB NOT NULL DEFAULT '{}'::jsonb,
                    raw_attestation JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            ).format(schema_identifier),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.market_snapshots (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    bid DOUBLE PRECISION NULL,
                    ask DOUBLE PRECISION NULL,
                    mid DOUBLE PRECISION NULL,
                    spread_bps DOUBLE PRECISION NOT NULL,
                    realized_volatility DOUBLE PRECISION NOT NULL,
                    orderbook_stale_seconds DOUBLE PRECISION NOT NULL,
                    market_data_available BOOLEAN NOT NULL,
                    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            ).format(schema_identifier),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.execution_records (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    trade_intent_id TEXT NOT NULL,
                    policy_decision_id TEXT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    exchange_order_id TEXT NULL,
                    requested_notional_usd DOUBLE PRECISION NOT NULL,
                    executed_notional_usd DOUBLE PRECISION NULL,
                    average_fill_price DOUBLE PRECISION NULL,
                    failure_reason TEXT NULL,
                    raw_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source TEXT NOT NULL,
                    FOREIGN KEY (trade_intent_id) REFERENCES {}.trade_intents(id),
                    FOREIGN KEY (policy_decision_id) REFERENCES {}.policy_decisions(id)
                )
                """
            ).format(schema_identifier, schema_identifier, schema_identifier),
            sql.SQL(
                """
                CREATE TABLE IF NOT EXISTS {}.checkpoints (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL,
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source TEXT NOT NULL
                )
                """
            ).format(schema_identifier),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS trade_intents_trace_id_idx ON {}.trade_intents(trace_id)"
            ).format(schema_identifier),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS trade_intents_symbol_idx ON {}.trade_intents(symbol)"
            ).format(schema_identifier),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS trust_snapshots_agent_address_idx ON {}.trust_snapshots(agent_address, created_at DESC)"
            ).format(schema_identifier),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS market_snapshots_symbol_idx ON {}.market_snapshots(symbol, created_at DESC)"
            ).format(schema_identifier),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS execution_records_trade_intent_id_idx ON {}.execution_records(trade_intent_id)"
            ).format(schema_identifier),
            sql.SQL(
                "CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx ON {}.checkpoints(thread_id, checkpoint_ns, checkpoint_id)"
            ).format(schema_identifier),
        ]

        try:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()
        except psycopg.Error as error:
            connection.rollback()
            raise PersistenceError(
                "Failed to initialize PostgreSQL schema.",
                context={"schema": self.config.schema, "error": str(error)},
            ) from error

    # This block inserts a validated trade intent record.
    # It takes: a typed TradeIntentRecord from the validation/agent layer.
    # It gives: the persisted record id for downstream references.
    def insert_trade_intent(self, record: TradeIntentRecord) -> str:
        query = sql.SQL(
            """
            INSERT INTO {}.trade_intents (
                id,
                created_at,
                trace_id,
                symbol,
                side,
                order_type,
                notional_usd,
                position_size_bps,
                signal_score,
                model_confidence,
                slippage_bps,
                regime,
                thesis,
                time_in_force,
                source,
                raw_payload
            ) VALUES (
                %(id)s,
                %(created_at)s,
                %(trace_id)s,
                %(symbol)s,
                %(side)s,
                %(order_type)s,
                %(notional_usd)s,
                %(position_size_bps)s,
                %(signal_score)s,
                %(model_confidence)s,
                %(slippage_bps)s,
                %(regime)s,
                %(thesis)s,
                %(time_in_force)s,
                %(source)s,
                %(raw_payload)s
            )
            """
        ).format(sql.Identifier(self.config.schema))

        payload = {
            "id": record.id,
            "created_at": record.created_at,
            "trace_id": record.trace_id,
            "symbol": record.symbol,
            "side": record.side.value,
            "order_type": record.order_type.value,
            "notional_usd": record.notional_usd,
            "position_size_bps": record.position_size_bps,
            "signal_score": record.signal_score,
            "model_confidence": record.model_confidence,
            "slippage_bps": record.slippage_bps,
            "regime": record.regime.value,
            "thesis": record.thesis,
            "time_in_force": record.time_in_force,
            "source": record.source.value,
            "raw_payload": Json(record.raw_payload),
        }
        self._execute_write(query, payload, operation="insert_trade_intent")
        return record.id

    # This block inserts a policy-decision audit record.
    # It takes: a typed PolicyDecisionRecord linked to a trade intent.
    # It gives: the persisted decision id for execution/audit references.
    def insert_policy_decision(self, record: PolicyDecisionRecord) -> str:
        query = sql.SQL(
            """
            INSERT INTO {}.policy_decisions (
                id,
                created_at,
                trade_intent_id,
                action,
                policy_name,
                policy_version,
                violations,
                source
            ) VALUES (
                %(id)s,
                %(created_at)s,
                %(trade_intent_id)s,
                %(action)s,
                %(policy_name)s,
                %(policy_version)s,
                %(violations)s,
                %(source)s
            )
            """
        ).format(sql.Identifier(self.config.schema))

        payload = {
            "id": record.id,
            "created_at": record.created_at,
            "trade_intent_id": record.trade_intent_id,
            "action": record.action.value,
            "policy_name": record.policy_name,
            "policy_version": record.policy_version,
            "violations": Json([violation.model_dump() for violation in record.violations]),
            "source": record.source.value,
        }
        self._execute_write(query, payload, operation="insert_policy_decision")
        return record.id

    # This block inserts a trust snapshot record.
    # It takes: a typed TrustSnapshotRecord from the trust/security layer.
    # It gives: the persisted trust snapshot id.
    def insert_trust_snapshot(self, record: TrustSnapshotRecord) -> str:
        query = sql.SQL(
            """
            INSERT INTO {}.trust_snapshots (
                id,
                created_at,
                agent_address,
                trust_score,
                erc8004_registered,
                valid_attestation,
                registry_reachable,
                attestation_age_seconds,
                attested,
                tee_measurement,
                tee_enclave_id,
                raw_registry,
                raw_attestation
            ) VALUES (
                %(id)s,
                %(created_at)s,
                %(agent_address)s,
                %(trust_score)s,
                %(erc8004_registered)s,
                %(valid_attestation)s,
                %(registry_reachable)s,
                %(attestation_age_seconds)s,
                %(attested)s,
                %(tee_measurement)s,
                %(tee_enclave_id)s,
                %(raw_registry)s,
                %(raw_attestation)s
            )
            """
        ).format(sql.Identifier(self.config.schema))

        payload = {
            "id": record.id,
            "created_at": record.created_at,
            "agent_address": record.agent_address,
            "trust_score": record.trust_score,
            "erc8004_registered": record.erc8004_registered,
            "valid_attestation": record.valid_attestation,
            "registry_reachable": record.registry_reachable,
            "attestation_age_seconds": record.attestation_age_seconds,
            "attested": record.attested,
            "tee_measurement": record.tee_measurement,
            "tee_enclave_id": record.tee_enclave_id,
            "raw_registry": Json(record.raw_registry),
            "raw_attestation": Json(record.raw_attestation),
        }
        self._execute_write(query, payload, operation="insert_trust_snapshot")
        return record.id

    # This block inserts a market snapshot record.
    # It takes: a typed MarketSnapshotRecord from the market-data layer.
    # It gives: the persisted market snapshot id.
    def insert_market_snapshot(self, record: MarketSnapshotRecord) -> str:
        query = sql.SQL(
            """
            INSERT INTO {}.market_snapshots (
                id,
                created_at,
                symbol,
                bid,
                ask,
                mid,
                spread_bps,
                realized_volatility,
                orderbook_stale_seconds,
                market_data_available,
                raw_payload
            ) VALUES (
                %(id)s,
                %(created_at)s,
                %(symbol)s,
                %(bid)s,
                %(ask)s,
                %(mid)s,
                %(spread_bps)s,
                %(realized_volatility)s,
                %(orderbook_stale_seconds)s,
                %(market_data_available)s,
                %(raw_payload)s
            )
            """
        ).format(sql.Identifier(self.config.schema))

        payload = {
            "id": record.id,
            "created_at": record.created_at,
            "symbol": record.symbol,
            "bid": record.bid,
            "ask": record.ask,
            "mid": record.mid,
            "spread_bps": record.spread_bps,
            "realized_volatility": record.realized_volatility,
            "orderbook_stale_seconds": record.orderbook_stale_seconds,
            "market_data_available": record.market_data_available,
            "raw_payload": Json(record.raw_payload),
        }
        self._execute_write(query, payload, operation="insert_market_snapshot")
        return record.id

    # This block inserts an execution-history record.
    # It takes: a typed ExecutionRecord from the execution layer.
    # It gives: the persisted execution record id.
    def insert_execution_record(self, record: ExecutionRecord) -> str:
        query = sql.SQL(
            """
            INSERT INTO {}.execution_records (
                id,
                created_at,
                trade_intent_id,
                policy_decision_id,
                symbol,
                side,
                order_type,
                status,
                exchange_order_id,
                requested_notional_usd,
                executed_notional_usd,
                average_fill_price,
                failure_reason,
                raw_payload,
                source
            ) VALUES (
                %(id)s,
                %(created_at)s,
                %(trade_intent_id)s,
                %(policy_decision_id)s,
                %(symbol)s,
                %(side)s,
                %(order_type)s,
                %(status)s,
                %(exchange_order_id)s,
                %(requested_notional_usd)s,
                %(executed_notional_usd)s,
                %(average_fill_price)s,
                %(failure_reason)s,
                %(raw_payload)s,
                %(source)s
            )
            """
        ).format(sql.Identifier(self.config.schema))

        payload = {
            "id": record.id,
            "created_at": record.created_at,
            "trade_intent_id": record.trade_intent_id,
            "policy_decision_id": record.policy_decision_id,
            "symbol": record.symbol,
            "side": record.side.value,
            "order_type": record.order_type.value,
            "status": record.status.value,
            "exchange_order_id": record.exchange_order_id,
            "requested_notional_usd": record.requested_notional_usd,
            "executed_notional_usd": record.executed_notional_usd,
            "average_fill_price": record.average_fill_price,
            "failure_reason": record.failure_reason,
            "raw_payload": Json(record.raw_payload),
            "source": record.source.value,
        }
        self._execute_write(query, payload, operation="insert_execution_record")
        return record.id

    # This block upserts checkpoint metadata for LangGraph-related persistence.
    # It takes: a typed CheckpointRecord from the orchestration layer.
    # It gives: the persisted checkpoint record id.
    def upsert_checkpoint(self, record: CheckpointRecord) -> str:
        query = sql.SQL(
            """
            INSERT INTO {}.checkpoints (
                id,
                created_at,
                thread_id,
                checkpoint_ns,
                checkpoint_id,
                state_version,
                metadata,
                source
            ) VALUES (
                %(id)s,
                %(created_at)s,
                %(thread_id)s,
                %(checkpoint_ns)s,
                %(checkpoint_id)s,
                %(state_version)s,
                %(metadata)s,
                %(source)s
            )
            ON CONFLICT (id) DO UPDATE SET
                created_at = EXCLUDED.created_at,
                thread_id = EXCLUDED.thread_id,
                checkpoint_ns = EXCLUDED.checkpoint_ns,
                checkpoint_id = EXCLUDED.checkpoint_id,
                state_version = EXCLUDED.state_version,
                metadata = EXCLUDED.metadata,
                source = EXCLUDED.source
            """
        ).format(sql.Identifier(self.config.schema))

        payload = {
            "id": record.id,
            "created_at": record.created_at,
            "thread_id": record.thread_id,
            "checkpoint_ns": record.checkpoint_ns,
            "checkpoint_id": record.checkpoint_id,
            "state_version": record.state_version,
            "metadata": Json(record.metadata),
            "source": record.source.value,
        }
        self._execute_write(query, payload, operation="upsert_checkpoint")
        return record.id

    # This block fetches the latest trust snapshot for a given agent address.
    # It takes: the agent wallet address.
    # It gives: the newest TrustSnapshotRecord or None if no snapshot exists.
    def get_latest_trust_snapshot(self, agent_address: str) -> TrustSnapshotRecord | None:
        query = sql.SQL(
            """
            SELECT *
            FROM {}.trust_snapshots
            WHERE agent_address = %(agent_address)s
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).format(sql.Identifier(self.config.schema))

        row = self._fetch_one(
            query,
            {"agent_address": agent_address.lower()},
            operation="get_latest_trust_snapshot",
        )
        if row is None:
            return None

        return TrustSnapshotRecord.model_validate(row)

    # This block fetches the latest market snapshot for a given symbol.
    # It takes: the trading symbol.
    # It gives: the newest MarketSnapshotRecord or None if no snapshot exists.
    def get_latest_market_snapshot(self, symbol: str) -> MarketSnapshotRecord | None:
        query = sql.SQL(
            """
            SELECT *
            FROM {}.market_snapshots
            WHERE symbol = %(symbol)s
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).format(sql.Identifier(self.config.schema))

        row = self._fetch_one(
            query,
            {"symbol": symbol.upper()},
            operation="get_latest_market_snapshot",
        )
        if row is None:
            return None

        return MarketSnapshotRecord.model_validate(row)

    # This block fetches recent policy decisions for audit or operator review.
    # It takes: an optional result limit.
    # It gives: a list of typed PolicyDecisionRecord objects ordered by newest first.
    def list_recent_policy_decisions(self, limit: int = 50) -> list[PolicyDecisionRecord]:
        if limit <= 0:
            raise ConfigurationError("Policy decision query limit must be greater than 0.")

        query = sql.SQL(
            """
            SELECT *
            FROM {}.policy_decisions
            ORDER BY created_at DESC
            LIMIT %(limit)s
            """
        ).format(sql.Identifier(self.config.schema))

        rows = self._fetch_all(
            query,
            {"limit": limit},
            operation="list_recent_policy_decisions",
        )

        result: list[PolicyDecisionRecord] = []
        for row in rows:
            violations = [
                PolicyViolationRecord.model_validate(item)
                for item in row.get("violations", [])
            ]
            row["violations"] = violations
            row["action"] = PolicyAction(row["action"])
            result.append(PolicyDecisionRecord.model_validate(row))

        return result

    # This block executes a write query with transaction handling.
    # It takes: a SQL statement, bound parameters, and an operation label.
    # It gives: a committed database write or a structured persistence error.
    def _execute_write(
        self,
        query: sql.Composed,
        params: dict[str, object],
        *,
        operation: str,
    ) -> None:
        connection = self.connect()

        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
            connection.commit()
        except psycopg.Error as error:
            connection.rollback()
            raise PersistenceError(
                "PostgreSQL write operation failed.",
                context={"operation": operation, "error": str(error)},
            ) from error

    # This block executes a single-row fetch query.
    # It takes: a SQL statement, bound parameters, and an operation label.
    # It gives: one row as a dictionary or None if nothing matched.
    def _fetch_one(
        self,
        query: sql.Composed,
        params: dict[str, object],
        *,
        operation: str,
    ) -> dict[str, object] | None:
        connection = self.connect()

        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                row = cursor.fetchone()
            connection.commit()
            return row
        except psycopg.Error as error:
            connection.rollback()
            raise PersistenceError(
                "PostgreSQL read operation failed.",
                context={"operation": operation, "error": str(error)},
            ) from error

    # This block executes a multi-row fetch query.
    # It takes: a SQL statement, bound parameters, and an operation label.
    # It gives: a list of rows as dictionaries.
    def _fetch_all(
        self,
        query: sql.Composed,
        params: dict[str, object],
        *,
        operation: str,
    ) -> list[dict[str, object]]:
        connection = self.connect()

        try:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                rows = list(cursor.fetchall())
            connection.commit()
            return rows
        except psycopg.Error as error:
            connection.rollback()
            raise PersistenceError(
                "PostgreSQL read operation failed.",
                context={"operation": operation, "error": str(error)},
            ) from error
