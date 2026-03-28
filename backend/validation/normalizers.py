from __future__ import annotations

from copy import deepcopy
from typing import Any

from backend.core.exceptions import ValidationError


# This block defines top-level alias keys that may appear in model output.
# It takes: raw top-level dictionaries from parsed model responses.
# It gives: a consistent canonical shape for later validation.
TOP_LEVEL_ALIASES: dict[str, str] = {
    "trade": "trade_intent",
    "tradeIntent": "trade_intent",
    "requiresHumanReview": "requires_human_review",
}


# This block defines trade-intent alias keys that may appear in model output.
# It takes: raw trade-intent dictionaries from model responses.
# It gives: canonical field names that match the Pydantic schema layer.
TRADE_INTENT_ALIASES: dict[str, str] = {
    "pair": "symbol",
    "ticker": "symbol",
    "instrument": "symbol",
    "type": "order_type",
    "orderType": "order_type",
    "notional": "notional_usd",
    "notionalUsd": "notional_usd",
    "positionSizeBps": "position_size_bps",
    "signalScore": "signal_score",
    "modelConfidence": "model_confidence",
    "slippage": "slippage_bps",
    "slippageBps": "slippage_bps",
    "timeInForce": "time_in_force",
}


# This block normalizes a raw top-level LLM payload.
# It takes: an untrusted parsed object from the parser layer.
# It gives: a cleaned dictionary that is ready for strict schema validation.
def normalize_llm_signal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("LLM payload must be a dictionary before normalization.")

    normalized = deepcopy(payload)

    # This block rewrites known top-level alias keys into canonical names.
    # It takes: the raw payload dictionary.
    # It gives: a normalized top-level schema shape.
    normalized = _apply_aliases(normalized, TOP_LEVEL_ALIASES)

    # This block ensures there is a trade_intent object to normalize.
    # It takes: the top-level payload dictionary.
    # It gives: an explicit failure if the required trade payload is missing or malformed.
    trade_payload = normalized.get("trade_intent")
    if not isinstance(trade_payload, dict):
        raise ValidationError(
            "LLM payload must include a 'trade_intent' object.",
            context={"payload_keys": sorted(normalized.keys())},
        )

    # This block normalizes the nested trade-intent object.
    # It takes: the raw trade payload from the model.
    # It gives: a canonical trade-intent dictionary aligned to the schema layer.
    normalized["trade_intent"] = normalize_trade_intent_payload(trade_payload)

    # This block normalizes the optional summary field.
    # It takes: the summary value if present.
    # It gives: a trimmed string summary when available.
    if "summary" in normalized and normalized["summary"] is not None:
        normalized["summary"] = str(normalized["summary"]).strip()

    # This block normalizes optional risks into a clean string list.
    # It takes: free-form list-ish model output.
    # It gives: a deterministic list of non-empty string risk items.
    if "risks" in normalized:
        normalized["risks"] = _normalize_string_list(normalized["risks"])

    # This block normalizes the human-review flag.
    # It takes: a possibly string/number/boolean input.
    # It gives: a strict boolean suitable for schema validation.
    if "requires_human_review" in normalized:
        normalized["requires_human_review"] = _coerce_bool(
            normalized["requires_human_review"],
            field_name="requires_human_review",
        )

    return normalized


# This block normalizes a trade-intent payload before schema validation.
# It takes: a raw trade-intent dictionary from an LLM or operator-facing parser.
# It gives: a cleaned dictionary with canonical keys and normalized value formats.
def normalize_trade_intent_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValidationError("Trade intent payload must be a dictionary before normalization.")

    normalized = deepcopy(payload)
    normalized = _apply_aliases(normalized, TRADE_INTENT_ALIASES)

    # This block normalizes string-like enum and identifier fields.
    # It takes: symbol, side, order_type, regime, thesis, and time_in_force values.
    # It gives: consistent lowercase/uppercase/trimmed strings expected by the schema layer.
    if "symbol" in normalized:
        normalized["symbol"] = str(normalized["symbol"]).strip().upper()

    if "side" in normalized:
        normalized["side"] = str(normalized["side"]).strip().lower()

    if "order_type" in normalized:
        normalized["order_type"] = str(normalized["order_type"]).strip().lower()

    if "regime" in normalized:
        normalized["regime"] = _normalize_regime(normalized["regime"])

    if "time_in_force" in normalized:
        normalized["time_in_force"] = str(normalized["time_in_force"]).strip().lower()

    if "thesis" in normalized and normalized["thesis"] is not None:
        normalized["thesis"] = str(normalized["thesis"]).strip()

    # This block normalizes numeric policy-sensitive fields.
    # It takes: values that may arrive as strings, ints, or floats.
    # It gives: canonical numeric types before strict schema validation.
    float_fields = {
        "notional_usd",
        "signal_score",
        "model_confidence",
        "slippage_bps",
    }
    int_fields = {
        "position_size_bps",
    }

    for field_name in float_fields:
        if field_name in normalized:
            normalized[field_name] = _coerce_float(normalized[field_name], field_name=field_name)

    for field_name in int_fields:
        if field_name in normalized:
            normalized[field_name] = _coerce_int(normalized[field_name], field_name=field_name)

    return normalized


# This block rewrites alias keys to canonical keys without inventing new values.
# It takes: a dictionary and a mapping of alias->canonical names.
# It gives: a new dictionary with aliases collapsed into canonical fields.
def _apply_aliases(payload: dict[str, Any], aliases: dict[str, str]) -> dict[str, Any]:
    normalized = dict(payload)

    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)
        elif alias in normalized and canonical in normalized:
            normalized.pop(alias)

    return normalized


# This block normalizes regime values and common variations.
# It takes: a raw regime string from model output.
# It gives: one canonical regime label expected by the schema layer.
def _normalize_regime(value: Any) -> str:
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")

    aliases = {
        "meanreversion": "mean_reversion",
        "mean_reverting": "mean_reversion",
        "trend_following": "trend",
        "trending": "trend",
        "break_out": "breakout",
    }

    return aliases.get(normalized, normalized)


# This block coerces arbitrary values into floats when safe to do so.
# It takes: a raw field value and the field name for error reporting.
# It gives: a float value or raises a structured validation error.
def _coerce_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"Field '{field_name}' could not be coerced to float.",
            context={"field": field_name, "value": value},
        ) from error


# This block coerces arbitrary values into integers when safe to do so.
# It takes: a raw field value and the field name for error reporting.
# It gives: an int value or raises a structured validation error.
def _coerce_int(value: Any, *, field_name: str) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError("boolean is not a valid integer input")
        return int(float(value))
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"Field '{field_name}' could not be coerced to int.",
            context={"field": field_name, "value": value},
        ) from error


# This block coerces flexible input into a strict boolean.
# It takes: strings, numbers, or booleans that represent true/false.
# It gives: a real boolean value or raises a structured validation error.
def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False

    raise ValidationError(
        f"Field '{field_name}' could not be coerced to bool.",
        context={"field": field_name, "value": value},
    )


# This block normalizes a list-like value into a list of strings.
# It takes: either a list of values or a single scalar value.
# It gives: a clean list of non-empty trimmed strings.
def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    normalized = str(value).strip()
    if not normalized:
        return []

    return [normalized]
