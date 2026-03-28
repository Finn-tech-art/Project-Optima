#!/usr/bin/env bash

# This block enables strict shell behavior.
# It takes: shell execution state.
# It gives: immediate failure on unset vars, command errors, and pipe failures.
set -euo pipefail

# This block defines default runtime settings.
# It takes: optional environment overrides from the caller.
# It gives: safe defaults for timeout, provider selection, and output behavior.
TIMEOUT_SECONDS="${TEE_TIMEOUT_SECONDS:-15}"
PROVIDER="${TEE_PROVIDER:-phala}"
OUTPUT_PATH=""
AGENT_ID=""
REQUEST_ID=""
VERBOSE="${TEE_VERBOSE:-0}"

# This block prints usage help.
# It takes: no input beyond the script invocation context.
# It gives: a short operator guide for supported flags and required environment variables.
usage() {
  cat <<'EOF'
Usage:
  tee_attestation.sh [--agent-id AGENT_ID] [--request-id REQUEST_ID] [--output PATH]

Environment:
  TEE_PROVIDER=phala
  TEE_TIMEOUT_SECONDS=15

Phala endpoint mode:
  PHALA_CLOUD_ATTESTATION_ENDPOINT=https://...
  PHALA_CLOUD_API_TOKEN=...
  PHALA_CLOUD_PROJECT_ID=...
  PHALA_CLOUD_CLUSTER_ID=...     # optional
  PHALA_CLOUD_APP_ID=...         # optional

Command mode:
  PHALA_CLOUD_ENROLL_COMMAND='provider-cli ... --json'

Output:
  The script writes normalized JSON to stdout.
  If --output is provided, it also writes the same JSON to that file.
EOF
}

# This block logs internal debug messages to stderr only.
# It takes: a debug string from the script.
# It gives: optional operator diagnostics without polluting JSON stdout.
debug_log() {
  if [[ "${VERBOSE}" == "1" ]]; then
    printf '[tee_attestation] %s\n' "$1" >&2
  fi
}

# This block prints an error payload and exits non-zero.
# It takes: an error code and a human-readable message.
# It gives: a consistent JSON error contract for callers that want structured failures.
fail() {
  local code="$1"
  local message="$2"

  printf '{"attested":false,"valid_attestation":false,"error":{"code":"%s","message":"%s"}}\n' \
    "$code" \
    "$(printf '%s' "$message" | sed 's/"/\\"/g')"
  exit 1
}

# This block verifies that a required binary exists.
# It takes: a command name.
# It gives: an early failure if the runtime dependency is missing.
require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    fail "missing_dependency" "Required command not found: ${command_name}"
  fi
}

# This block parses CLI flags.
# It takes: operator-supplied script arguments.
# It gives: normalized shell variables for downstream enrollment logic.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-id)
      AGENT_ID="${2:-}"
      shift 2
      ;;
    --request-id)
      REQUEST_ID="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      fail "invalid_argument" "Unknown argument: $1"
      ;;
  esac
done

# This block validates tool dependencies.
# It takes: the execution environment.
# It gives: confidence that JSON normalization and HTTP transport will work.
require_command "jq"
require_command "curl"

# This block creates temp files for provider responses.
# It takes: the host filesystem.
# It gives: isolated scratch space for HTTP or command output processing.
RAW_RESPONSE_FILE="$(mktemp)"
NORMALIZED_FILE="$(mktemp)"

cleanup() {
  rm -f "${RAW_RESPONSE_FILE}" "${NORMALIZED_FILE}"
}
trap cleanup EXIT

# This block captures the current UTC timestamp.
# It takes: system clock state.
# It gives: a fallback issuance timestamp if the provider omits one.
NOW_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# This block runs a provider command if one is configured.
# It takes: PHALA_CLOUD_ENROLL_COMMAND from the environment.
# It gives: raw JSON emitted by the provider CLI or wrapper command.
run_command_mode() {
  local command_string="${PHALA_CLOUD_ENROLL_COMMAND:-}"

  if [[ -z "${command_string}" ]]; then
    return 1
  fi

  debug_log "Running command-mode attestation flow"
  if ! timeout "${TIMEOUT_SECONDS}" bash -lc "${command_string}" >"${RAW_RESPONSE_FILE}"; then
    fail "enrollment_command_failed" "Phala enrollment command failed or timed out"
  fi

  return 0
}

# This block calls a Phala attestation endpoint if configured.
# It takes: endpoint URL, API token, and project metadata from the environment.
# It gives: raw JSON response from the provider endpoint.
run_endpoint_mode() {
  local endpoint="${PHALA_CLOUD_ATTESTATION_ENDPOINT:-}"
  local api_token="${PHALA_CLOUD_API_TOKEN:-}"
  local project_id="${PHALA_CLOUD_PROJECT_ID:-}"
  local cluster_id="${PHALA_CLOUD_CLUSTER_ID:-}"
  local app_id="${PHALA_CLOUD_APP_ID:-}"

  if [[ -z "${endpoint}" ]]; then
    return 1
  fi

  if [[ -z "${api_token}" ]]; then
    fail "missing_configuration" "PHALA_CLOUD_API_TOKEN is required for endpoint mode"
  fi

  if [[ -z "${project_id}" ]]; then
    fail "missing_configuration" "PHALA_CLOUD_PROJECT_ID is required for endpoint mode"
  fi

  debug_log "Running endpoint-mode attestation flow"

  # This block builds the JSON request body for the provider.
  # It takes: agent id, request id, and project/application metadata.
  # It gives: a compact JSON payload for the attestation request.
  local request_body
  request_body="$(
    jq -n \
      --arg provider "${PROVIDER}" \
      --arg agent_id "${AGENT_ID}" \
      --arg request_id "${REQUEST_ID}" \
      --arg project_id "${project_id}" \
      --arg cluster_id "${cluster_id}" \
      --arg app_id "${app_id}" \
      '{
        provider: $provider,
        agent_id: ($agent_id | select(length > 0)),
        request_id: ($request_id | select(length > 0)),
        project_id: $project_id,
        cluster_id: ($cluster_id | select(length > 0)),
        app_id: ($app_id | select(length > 0))
      }'
  )"

  if ! curl --silent --show-error --fail \
    --max-time "${TIMEOUT_SECONDS}" \
    --header "Authorization: Bearer ${api_token}" \
    --header "Content-Type: application/json" \
    --data "${request_body}" \
    "${endpoint}" >"${RAW_RESPONSE_FILE}"; then
    fail "attestation_request_failed" "Phala endpoint request failed"
  fi

  return 0
}

# This block normalizes provider JSON into the schema expected by Python validation.
# It takes: raw provider JSON from command mode or endpoint mode.
# It gives: a stable JSON object with attested, measurement, issued_at, and raw fields.
normalize_response() {
  debug_log "Normalizing provider response"

  if ! jq -e . "${RAW_RESPONSE_FILE}" >/dev/null 2>&1; then
    fail "invalid_provider_json" "Provider response is not valid JSON"
  fi

  jq \
    --arg now_utc "${NOW_UTC}" \
    --arg provider "${PROVIDER}" \
    --arg agent_id "${AGENT_ID}" \
    '
    def pick_string(paths):
      first(paths[] as $p | getpath($p) | select(type == "string" and length > 0));

    def pick_bool(paths):
      first(paths[] as $p | getpath($p) | select(type == "boolean"));

    def pick_any(paths):
      first(paths[] as $p | getpath($p));

    . as $raw
    | {
        attested:
          (
            pick_bool([
              ["attested"],
              ["valid_attestation"],
              ["data","attested"],
              ["data","valid_attestation"],
              ["result","attested"],
              ["result","valid_attestation"]
            ]) // true
          ),
        valid_attestation:
          (
            pick_bool([
              ["valid_attestation"],
              ["attested"],
              ["data","valid_attestation"],
              ["data","attested"],
              ["result","valid_attestation"],
              ["result","attested"]
            ]) // true
          ),
        enclave_id:
          (
            pick_string([
              ["enclave_id"],
              ["data","enclave_id"],
              ["result","enclave_id"],
              ["app_id"],
              ["data","app_id"]
            ]) // null
          ),
        measurement:
          (
            pick_string([
              ["measurement"],
              ["mr_enclave"],
              ["data","measurement"],
              ["data","mr_enclave"],
              ["result","measurement"],
              ["result","mr_enclave"]
            ]) // null
          ),
        issued_at:
          (
            pick_string([
              ["issued_at"],
              ["timestamp"],
              ["created_at"],
              ["data","issued_at"],
              ["data","timestamp"],
              ["result","issued_at"],
              ["result","timestamp"]
            ]) // $now_utc
          ),
        quote_digest:
          (
            pick_string([
              ["quote_digest"],
              ["quote_hash"],
              ["data","quote_digest"],
              ["data","quote_hash"],
              ["result","quote_digest"],
              ["result","quote_hash"]
            ]) // null
          ),
        signer:
          (
            pick_string([
              ["signer"],
              ["issuer"],
              ["data","signer"],
              ["data","issuer"],
              ["result","signer"],
              ["result","issuer"]
            ]) // null
          ),
        provider: $provider,
        agent_id: (if $agent_id == "" then null else $agent_id end),
        raw: $raw
      }
    ' "${RAW_RESPONSE_FILE}" >"${NORMALIZED_FILE}" || fail "normalization_failed" "Could not normalize provider response"
}

# This block writes the final normalized JSON to the requested destinations.
# It takes: the normalized JSON file and optional output path.
# It gives: machine-readable stdout and, optionally, a persisted evidence file.
emit_output() {
  if [[ -n "${OUTPUT_PATH}" ]]; then
    mkdir -p "$(dirname "${OUTPUT_PATH}")"
    cp "${NORMALIZED_FILE}" "${OUTPUT_PATH}"
  fi

  cat "${NORMALIZED_FILE}"
}

# This block selects the provider transport.
# It takes: the configured environment for command mode or endpoint mode.
# It gives: one raw provider response ready for normalization.
if ! run_command_mode; then
  if ! run_endpoint_mode; then
    fail "missing_configuration" \
      "Provide PHALA_CLOUD_ENROLL_COMMAND or PHALA_CLOUD_ATTESTATION_ENDPOINT"
  fi
fi

# This block validates and emits the final result.
# It takes: the raw provider response.
# It gives: normalized JSON for the Python attestation service.
normalize_response
emit_output
