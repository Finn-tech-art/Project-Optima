from __future__ import annotations

import logging
import os
import sys
import traceback
from datetime import UTC, datetime
from typing import Any

import orjson


LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


class JsonFormatter(logging.Formatter):
    """Emit compact JSON log records for machine-first observability."""

    RESERVED_FIELDS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "message",
        "asctime",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.RESERVED_FIELDS and not key.startswith("_")
        }
        if extras:
            payload["context"] = extras

        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            payload["exception"] = {
                "type": exc_type.__name__ if exc_type else "Exception",
                "message": str(exc_value) if exc_value else "",
                "traceback": traceback.format_exception(exc_type, exc_value, exc_tb),
            }

        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return orjson.dumps(payload).decode("utf-8")


def _resolve_log_level(default: str = "INFO") -> int:
    configured = os.getenv("RRA_LOG_LEVEL", default).upper()
    return LOG_LEVELS.get(configured, logging.INFO)


def configure_logger(
    name: str = "rra",
    *,
    level: int | None = None,
    stream: Any = None,
) -> logging.Logger:
    """
    Configure and return a JSON-structured logger.

    This is idempotent for a given logger name.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level or _resolve_log_level())
    logger.propagate = False

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


def get_logger(name: str = "rra") -> logging.Logger:
    return configure_logger(name=name)


def _log(
    level: int,
    message: str,
    *,
    logger_name: str = "rra",
    exc_info: Any = None,
    **context: Any,
) -> None:
    logger = get_logger(logger_name)
    logger.log(level, message, extra=context, exc_info=exc_info)


def log_debug(message: str, *, logger_name: str = "rra", **context: Any) -> None:
    _log(logging.DEBUG, message, logger_name=logger_name, **context)


def log_info(message: str, *, logger_name: str = "rra", **context: Any) -> None:
    _log(logging.INFO, message, logger_name=logger_name, **context)


def log_warning(message: str, *, logger_name: str = "rra", **context: Any) -> None:
    _log(logging.WARNING, message, logger_name=logger_name, **context)


def log_error(
    message: str,
    *,
    logger_name: str = "rra",
    exc_info: Any = None,
    **context: Any,
) -> None:
    _log(logging.ERROR, message, logger_name=logger_name, exc_info=exc_info, **context)


def log_critical(
    message: str,
    *,
    logger_name: str = "rra",
    exc_info: Any = None,
    **context: Any,
) -> None:
    _log(
        logging.CRITICAL,
        message,
        logger_name=logger_name,
        exc_info=exc_info,
        **context,
    )
