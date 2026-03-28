from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.validation.normalizers import normalize_llm_signal_payload
from backend.validation.parsers import ParsedStructuredOutput, parse_structured_output
from backend.validation.schemas import LLMSignalOutput, PolicyEvaluationInput
from security.policy import GuardrailsPolicy, PolicyDecision


# This block stores the result of a full model-output guardrail pass.
# It takes: parsed output, normalized payload, validated schema objects, and policy decision.
# It gives: one structured object that downstream agent nodes can trust and inspect.
@dataclass(slots=True, frozen=True)
class GuardrailEvaluationResult:
    parsed_output: ParsedStructuredOutput
    normalized_payload: dict[str, Any]
    llm_signal_output: LLMSignalOutput
    policy_input: PolicyEvaluationInput
    policy_decision: PolicyDecision

    # This block returns a machine-readable summary of the guardrail evaluation.
    # It takes: the full guardrail evaluation result object.
    # It gives: a nested dictionary suitable for logging, tracing, or debugging.
    def to_dict(self) -> dict[str, Any]:
        return {
            "parsed_output": {
                "format": self.parsed_output.format,
                "payload": self.parsed_output.payload,
            },
            "normalized_payload": self.normalized_payload,
            "llm_signal_output": self.llm_signal_output.model_dump(),
            "policy_input": self.policy_input.model_dump(),
            "policy_decision": self.policy_decision.to_dict(),
        }


# This block is the main validation and policy-evaluation orchestrator.
# It takes: a loaded guardrails policy instance.
# It gives: a reusable object that can evaluate raw model output safely.
class OutputGuardrails:
    def __init__(self, policy: GuardrailsPolicy) -> None:
        self.policy = policy

    @classmethod
    def from_policy_file(cls, path: str | Path) -> OutputGuardrails:
        """
        This block builds the orchestrator from a guardrails YAML file.
        It takes: the path to the policy document.
        It gives: a ready OutputGuardrails instance.
        """
        return cls(policy=GuardrailsPolicy.from_file(path))

    def evaluate_llm_output(
        self,
        *,
        raw_output: str,
        runtime_context: dict[str, Any],
    ) -> GuardrailEvaluationResult:
        """
        This block performs the full guardrail flow on raw LLM output.
        It takes: untrusted model text plus runtime context from trust, market, and system layers.
        It gives: parsed, normalized, validated, and policy-evaluated output in one object.
        """
        # This block parses the raw model text into structured content.
        # It takes: raw JSON, fenced JSON, or XML-wrapped output.
        # It gives: a ParsedStructuredOutput object with decoded payload data.
        parsed_output = parse_structured_output(raw_output)

        # This block normalizes minor output inconsistencies before schema validation.
        # It takes: the parsed payload dictionary.
        # It gives: a canonical payload shape aligned with strict schemas.
        normalized_payload = normalize_llm_signal_payload(parsed_output.payload)

        # This block validates the normalized payload as a strict LLM signal contract.
        # It takes: the normalized payload dictionary.
        # It gives: a typed LLMSignalOutput object trusted by the rest of the system.
        llm_signal_output = LLMSignalOutput.model_validate(normalized_payload)

        # This block combines the validated trade intent with runtime context.
        # It takes: the validated trade output and the current runtime context map.
        # It gives: a strict PolicyEvaluationInput object for the security layer.
        policy_input = PolicyEvaluationInput.model_validate(
            {
                "trade": llm_signal_output.trade_intent.model_dump(),
                "trust": runtime_context.get("trust", {}),
                "positions": runtime_context.get("positions", {}),
                "market": runtime_context.get("market", {}),
                "execution": runtime_context.get("execution", {}),
                "system": runtime_context.get("system", {}),
                "flags": runtime_context.get("flags", {}),
            }
        )

        # This block converts the validated input into the policy engine contract.
        # It takes: the typed PolicyEvaluationInput object.
        # It gives: the exact trade/context dict structure expected by security.policy.
        policy_payload = policy_input.to_policy_input()

        # This block evaluates the validated trade against the loaded security policy.
        # It takes: the policy-compatible trade and runtime context.
        # It gives: a final allow/reduce/block decision with structured violations.
        policy_decision = self.policy.evaluate(
            trade=policy_payload["trade"],
            context=policy_payload["context"],
        )

        return GuardrailEvaluationResult(
            parsed_output=parsed_output,
            normalized_payload=normalized_payload,
            llm_signal_output=llm_signal_output,
            policy_input=policy_input,
            policy_decision=policy_decision,
        )

    def evaluate_validated_signal(
        self,
        *,
        llm_signal_output: LLMSignalOutput,
        runtime_context: dict[str, Any],
    ) -> GuardrailEvaluationResult:
        """
        This block evaluates an already validated signal object.
        It takes: a strict LLMSignalOutput and runtime context.
        It gives: the same full guardrail result structure without re-parsing raw text.
        """
        normalized_payload = llm_signal_output.model_dump()

        policy_input = PolicyEvaluationInput.model_validate(
            {
                "trade": llm_signal_output.trade_intent.model_dump(),
                "trust": runtime_context.get("trust", {}),
                "positions": runtime_context.get("positions", {}),
                "market": runtime_context.get("market", {}),
                "execution": runtime_context.get("execution", {}),
                "system": runtime_context.get("system", {}),
                "flags": runtime_context.get("flags", {}),
            }
        )

        policy_payload = policy_input.to_policy_input()
        policy_decision = self.policy.evaluate(
            trade=policy_payload["trade"],
            context=policy_payload["context"],
        )

        return GuardrailEvaluationResult(
            parsed_output=ParsedStructuredOutput(
                raw_text="",
                format="validated_object",
                payload=normalized_payload,
            ),
            normalized_payload=normalized_payload,
            llm_signal_output=llm_signal_output,
            policy_input=policy_input,
            policy_decision=policy_decision,
        )
