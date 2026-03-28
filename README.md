# RRA Trading Swarm

Project-Optima is the bootstrap implementation of the Reflexive Reputation Arb (RRA) system.

The current scaffold is organized around a strict safety-first pipeline:

1. Prompt the model with a constrained XML response contract
2. Parse and normalize the returned structured output
3. Validate the candidate trade and runtime context with Pydantic
4. Evaluate the candidate against policy-as-code guardrails
5. Only then allow downstream agent and execution layers to proceed

## Current Architecture

The repository currently includes:

- Python 3.12 backend managed with `uv`
- Structured JSON logging
- Shared exception taxonomy
- Retry, rate limiting, and circuit breaker primitives
- Security layer for:
  - ERC-8004 registry integration
  - TEE attestation validation
  - secrets and environment loading
  - policy-as-code enforcement
- Validation layer for:
  - LLM output schemas
  - operator input schemas
  - JSON and XML-wrapped parsing
  - normalization and guardrail orchestration
- Groq integration wrapper for inference

## Repository Layout

```text
Project-Optima/
|-- backend/
|   |-- agent/
|   |   `-- nodes/
|   |-- config/
|   |   `-- settings.py
|   |-- core/
|   |   |-- circuit_breaker.py
|   |   |-- constants.py
|   |   |-- exceptions.py
|   |   |-- logger.py
|   |   |-- rate_limiter.py
|   |   `-- retry.py
|   |-- integrations/
|   |   `-- groq_client.py
|   |-- prompts/
|   |   `-- xml_wrapper.py
|   |-- validation/
|   |   |-- guardrails.py
|   |   |-- normalizers.py
|   |   |-- parsers.py
|   |   `-- schemas.py
|   `-- main.py
|-- frontend/
|   |-- app/
|   `-- components/
|-- scripts/
|-- security/
|   |-- erc8004_registry.py
|   |-- guardrails.yaml
|   |-- policy.py
|   |-- secrets.py
|   |-- tee_attestation.py
|   `-- tee_attestation.sh
|-- .env
|-- .env.example
|-- .python-version
|-- pyproject.toml
`-- uv.lock
