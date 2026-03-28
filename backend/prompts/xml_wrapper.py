from __future__ import annotations

from dataclasses import dataclass


RESPONSE_ROOT_TAG = "response"
PAYLOAD_TAG = "payload"
SUMMARY_TAG = "summary"
RULES_TAG = "rules"


# This block defines the XML prompt package returned to the caller.
# It takes: the rendered system prompt, user prompt, and response template.
# It gives: one reusable object that inference code can send to the model.
@dataclass(slots=True, frozen=True)
class XMLWrappedPrompt:
    system_prompt: str
    user_prompt: str
    response_template: str

    # This block returns the prompt package as a plain dictionary.
    # It takes: the XMLWrappedPrompt object.
    # It gives: a serialization-friendly representation for logging or transport.
    def to_dict(self) -> dict[str, str]:
        return {
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "response_template": self.response_template,
        }


# This block builds the XML response template that the model must follow.
# It takes: no runtime input.
# It gives: a strict output contract with summary text and JSON payload sections.
def build_xml_response_template() -> str:
    return f"""\
<{RESPONSE_ROOT_TAG}>
  <{SUMMARY_TAG}>One short summary of the trade decision.</{SUMMARY_TAG}>
  <{PAYLOAD_TAG}>
    {{
      "trade_intent": {{
        "symbol": "BTCUSD",
        "side": "buy",
        "order_type": "limit",
        "notional_usd": 1000,
        "position_size_bps": 100,
        "signal_score": 0.75,
        "model_confidence": 0.80,
        "slippage_bps": 5,
        "regime": "trend",
        "thesis": "Short rationale here.",
        "time_in_force": "gtc"
      }},
      "summary": "One short summary of the decision.",
      "risks": [
        "First key risk",
        "Second key risk"
      ],
      "requires_human_review": false
    }}
  </{PAYLOAD_TAG}>
</{RESPONSE_ROOT_TAG}>"""


# This block builds the instruction section that tells the model how to respond.
# It takes: no runtime input.
# It gives: deterministic formatting instructions aligned with the parser layer.
def build_xml_output_rules() -> str:
    return f"""\
<{RULES_TAG}>
  <rule>Return valid XML only.</rule>
  <rule>The root element must be <{RESPONSE_ROOT_TAG}>.</rule>
  <rule>Include exactly one <{SUMMARY_TAG}> element.</rule>
  <rule>Include exactly one <{PAYLOAD_TAG}> element.</rule>
  <rule>The <{PAYLOAD_TAG}> element must contain a valid JSON object.</rule>
  <rule>Do not wrap the response in markdown code fences.</rule>
  <rule>Do not include explanations before or after the XML.</rule>
  <rule>Use lowercase enum values for side, order_type, regime, and time_in_force.</rule>
</{RULES_TAG}>"""


# This block wraps a base system prompt with XML-output instructions.
# It takes: the original system prompt content.
# It gives: a stronger system prompt that enforces the XML response contract.
def wrap_system_prompt(base_system_prompt: str) -> str:
    normalized_prompt = base_system_prompt.strip()
    rules = build_xml_output_rules()
    template = build_xml_response_template()

    return f"""{normalized_prompt}

You must return your final answer using the exact XML response contract below.

{rules}

Response template:
{template}
"""


# This block wraps a user prompt with a direct formatting reminder.
# It takes: the original user prompt content.
# It gives: a user prompt that reinforces the XML-only response requirement.
def wrap_user_prompt(base_user_prompt: str) -> str:
    normalized_prompt = base_user_prompt.strip()

    return f"""{normalized_prompt}

Return XML only.
The <payload> element must contain a valid JSON object.
Do not use markdown code fences.
"""


# This block creates a full XML-wrapped prompt package for inference calls.
# It takes: a base system prompt and a base user prompt.
# It gives: one XMLWrappedPrompt object ready for the Groq client wrapper.
def build_xml_wrapped_prompt(
    *,
    system_prompt: str,
    user_prompt: str,
) -> XMLWrappedPrompt:
    return XMLWrappedPrompt(
        system_prompt=wrap_system_prompt(system_prompt),
        user_prompt=wrap_user_prompt(user_prompt),
        response_template=build_xml_response_template(),
    )
