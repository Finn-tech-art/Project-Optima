from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

from backend.core.exceptions import ValidationError
from backend.validation.schemas import LLMSignalOutput


JSON_CODE_BLOCK_PATTERN = re.compile(
    r"```(?:json)?\s*(?P<body>\{.*?\})\s*```",
    re.DOTALL,
)

XML_RESPONSE_TAG = "response"
XML_PAYLOAD_TAG = "payload"


# This block stores the result of extracting structured content from raw model output.
# It takes: the raw source text, extracted format metadata, and the decoded payload.
# It gives: one normalized parsing result that later validation code can consume.
@dataclass(slots=True, frozen=True)
class ParsedStructuredOutput:
    raw_text: str
    format: str
    payload: dict[str, Any]

    # This block validates the parsed payload as an LLM signal output.
    # It takes: the decoded payload produced by the parser.
    # It gives: a strict LLMSignalOutput object for downstream policy and agent logic.
    def to_llm_signal_output(self) -> LLMSignalOutput:
        return LLMSignalOutput.model_validate(self.payload)


# This block tries to parse raw model output into a structured payload.
# It takes: untrusted text returned by the model.
# It gives: a normalized ParsedStructuredOutput for JSON or XML-wrapped responses.
def parse_structured_output(raw_text: str) -> ParsedStructuredOutput:
    if not raw_text or not raw_text.strip():
        raise ValidationError("Model output is empty and cannot be parsed.")

    normalized_text = raw_text.strip()

    # This block tries direct JSON first.
    # It takes: the full raw text.
    # It gives: a parsed payload immediately if the model returned plain JSON.
    direct_json = _try_parse_json_object(normalized_text)
    if direct_json is not None:
        return ParsedStructuredOutput(
            raw_text=raw_text,
            format="json",
            payload=direct_json,
        )

    # This block tries fenced JSON next.
    # It takes: markdown-style code blocks containing JSON.
    # It gives: a parsed payload if the model wrapped the JSON in triple backticks.
    fenced_json = _extract_fenced_json(normalized_text)
    if fenced_json is not None:
        return ParsedStructuredOutput(
            raw_text=raw_text,
            format="json_code_block",
            payload=fenced_json,
        )

    # This block tries the XML response envelope last.
    # It takes: XML-wrapped content such as <response><payload>...</payload></response>.
    # It gives: a parsed payload if the model returned the expected XML wrapper.
    xml_payload = _extract_xml_payload(normalized_text)
    if xml_payload is not None:
        return ParsedStructuredOutput(
            raw_text=raw_text,
            format="xml",
            payload=xml_payload,
        )

    raise ValidationError(
        "Unable to parse model output as JSON, fenced JSON, or XML-wrapped payload.",
        context={"raw_text": raw_text},
    )


# This block parses and validates a raw model output string in one call.
# It takes: untrusted text returned by the model.
# It gives: a strict LLMSignalOutput object after parsing and schema validation.
def parse_llm_signal_output(raw_text: str) -> LLMSignalOutput:
    parsed = parse_structured_output(raw_text)
    return parsed.to_llm_signal_output()


# This block attempts to parse a text blob as a top-level JSON object.
# It takes: a candidate JSON string.
# It gives: a decoded dictionary if valid, otherwise None.
def _try_parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        raise ValidationError(
            "Parsed JSON output must be an object.",
            context={"parsed_type": type(parsed).__name__},
        )

    return parsed


# This block extracts JSON from a fenced markdown code block.
# It takes: raw model output text that may contain ```json ... ``` formatting.
# It gives: a decoded dictionary if a valid JSON object is found, otherwise None.
def _extract_fenced_json(text: str) -> dict[str, Any] | None:
    match = JSON_CODE_BLOCK_PATTERN.search(text)
    if not match:
        return None

    body = match.group("body").strip()
    return _try_parse_json_object(body)


# This block extracts the payload from an XML response envelope.
# It takes: raw model output text containing XML.
# It gives: a decoded dictionary parsed from the <payload> contents, otherwise None.
def _extract_xml_payload(text: str) -> dict[str, Any] | None:
    if not text.startswith("<"):
        return None

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None

    if root.tag != XML_RESPONSE_TAG:
        raise ValidationError(
            "XML output root tag must be <response>.",
            context={"root_tag": root.tag},
        )

    payload_node = root.find(XML_PAYLOAD_TAG)
    if payload_node is None:
        raise ValidationError("XML output is missing a <payload> element.")

    # This block reads the payload body from either nested text or a CDATA-style text node.
    # It takes: the XML payload node.
    # It gives: a raw string that should contain a JSON object.
    payload_text = "".join(payload_node.itertext()).strip()
    if not payload_text:
        raise ValidationError("XML <payload> element is empty.")

    parsed_payload = _try_parse_json_object(payload_text)
    if parsed_payload is None:
        raise ValidationError(
            "XML payload does not contain a valid JSON object.",
            context={"payload_text": payload_text},
        )

    return parsed_payload
