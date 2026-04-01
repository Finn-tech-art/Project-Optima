const STORAGE_KEY = "project_optima_latest_run";
const THEME_KEY = "project_optima_theme";

const runForm = document.getElementById("run-form");
const runButton = document.getElementById("run-button");
const themeToggle = document.getElementById("theme-toggle");
const mockStatus = document.getElementById("mock-status");
const groqStatus = document.getElementById("groq-status");
const krakenPaperStatus = document.getElementById("kraken-paper-status");
const summaryCaption = document.getElementById("summary-caption");
const disclaimer = document.getElementById("disclaimer");
const summaryGrid = document.getElementById("summary-grid");
const gapLabel = document.getElementById("gap-label");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeNeedle = document.getElementById("gauge-needle");
const gapMetrics = document.getElementById("gap-metrics");
const gapThesis = document.getElementById("gap-thesis");
const cortexGraph = document.getElementById("cortex-graph");
const thoughtStream = document.getElementById("thought-stream");
const performanceGrid = document.getElementById("performance-grid");
const equityCurve = document.getElementById("equity-curve");
const proofArtifact = document.getElementById("proof-artifact");
const proofTableBody = document.getElementById("proof-table-body");
const deadmanCountdown = document.getElementById("deadman-countdown");
const hitlStatus = document.getElementById("hitl-status");
const hitlReason = document.getElementById("hitl-reason");
const pauseGraphButton = document.getElementById("pause-graph");
const approveTradeButton = document.getElementById("approve-trade");
const rejectTradeButton = document.getElementById("reject-trade");
const timeline = document.getElementById("timeline");
const walletBalanceDisplay = document.getElementById("wallet-balance-display");
const backendDisplay = document.getElementById("backend-display");
const trustScoreDisplay = document.getElementById("trust-score-display");
const accountStateDisplay = document.getElementById("account-state-display");
const tradeIntent = document.getElementById("trade-intent");
const policyState = document.getElementById("policy-state");
const trustState = document.getElementById("trust-state");
const executionState = document.getElementById("execution-state");

let deadmanTimerId = null;
let lastSafetyPayload = null;
let hitlOverride = null;

// This block applies the saved or requested visual theme to the document.
// It takes: a theme string such as 'light' or 'dark'.
// It gives: synchronized DOM theme state and toggle button copy.
function applyTheme(theme) {
  const resolvedTheme = theme === "dark" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", resolvedTheme);
  window.localStorage.setItem(THEME_KEY, resolvedTheme);
  if (themeToggle) {
    themeToggle.textContent = resolvedTheme === "dark" ? "Light Mode" : "Dark Mode";
  }
}

// This block restores the persisted theme preference on page load.
// It takes: the browser's localStorage state.
// It gives: a stable theme across refreshes and route changes.
function initializeTheme() {
  const storedTheme = window.localStorage.getItem(THEME_KEY) || "light";
  applyTheme(storedTheme);
}

// This block fetches backend readiness information for the sidebar status cards.
// It takes: no operator input beyond the current browser session.
// It gives: a compact readiness snapshot for mock, Groq, and Kraken paper modes.
async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    const capabilities = payload.capabilities || {};
    mockStatus.textContent = capabilities.mock_ready ? "Ready" : "Unavailable";
    groqStatus.textContent = capabilities.groq_ready ? "Ready" : "Unavailable";
    krakenPaperStatus.textContent = capabilities.kraken_paper_ready ? "Ready" : "Unavailable";
  } catch (error) {
    mockStatus.textContent = "Unavailable";
    groqStatus.textContent = "Unavailable";
    krakenPaperStatus.textContent = "Unavailable";
    console.error(error);
  }
}

// This block toggles the primary run button while the workflow is executing.
// It takes: a boolean loading flag.
// It gives: a disabled or enabled button with matching copy.
function setLoading(isLoading) {
  runButton.disabled = isLoading;
  runButton.textContent = isLoading ? "Running..." : "Run Workflow";
}

// This block pretty-prints JSON-like values for the detail panels.
// It takes: any serializable value.
// It gives: a readable multiline string for the operator.
function formatJson(value) {
  if (value === null || value === undefined) {
    return "No data yet.";
  }
  return JSON.stringify(value, null, 2);
}

// This block builds a human-readable fallback when the backend returns structured errors.
// It takes: the structured error object from the API.
// It gives: a compact multiline diagnostic string.
function buildErrorDetails(errorPayload) {
  if (!errorPayload || typeof errorPayload !== "object") {
    return "No structured error details were returned.";
  }

  const lines = [];
  if (errorPayload.type) {
    lines.push(`type: ${errorPayload.type}`);
  }
  if (errorPayload.code) {
    lines.push(`code: ${errorPayload.code}`);
  }
  if (typeof errorPayload.retryable === "boolean") {
    lines.push(`retryable: ${errorPayload.retryable}`);
  }
  if (errorPayload.message) {
    lines.push(`message: ${errorPayload.message}`);
  }
  if (errorPayload.context && Object.keys(errorPayload.context).length > 0) {
    lines.push("");
    lines.push("context:");
    lines.push(JSON.stringify(errorPayload.context, null, 2));
  }

  return lines.join("\n");
}

// This block safely escapes HTML in runtime text.
// It takes: a string-like input.
// It gives: text safe for HTML interpolation.
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// This block wires button groups to hidden inputs so the UI can use buttons instead of dropdowns.
// It takes: the current DOM button groups.
// It gives: synchronized button and hidden-input state for form submission.
function initializeButtonGroups() {
  document.querySelectorAll(".button-group").forEach((group) => {
    const targetId = group.dataset.target;
    const hiddenInput = document.getElementById(targetId);
    if (!hiddenInput) {
      return;
    }

    group.querySelectorAll(".mode-button").forEach((button) => {
      button.addEventListener("click", () => {
        hiddenInput.value = button.dataset.value || "";
        group.querySelectorAll(".mode-button").forEach((candidate) => {
          candidate.classList.toggle("active", candidate === button);
        });
      });
    });
  });
}

// This block renders the compact account strip in a MetaTrader-style snapshot.
// It takes: the latest workflow result from the backend.
// It gives: wallet, backend, trust, and state values near the top of the dashboard.
function renderAccountStrip(result) {
  const walletBalance = Number(result?.metadata?.flags?.wallet_balance_usd || 0);
  const trustScore = Number(result?.trust_snapshot?.trust_score || 0);
  const backend = String(result?.execution_backend || "demo");
  const accountState = String(result?.summary?.final_status || "waiting");

  walletBalanceDisplay.textContent = `$${walletBalance.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
  backendDisplay.textContent = backend;
  trustScoreDisplay.textContent = trustScore.toFixed(2);
  accountStateDisplay.textContent = accountState;
}

// This block renders the compact run summary metrics.
// It takes: the summary object returned by the backend.
// It gives: a tight summary grid for fast scanning.
function renderSummary(summary) {
  summaryGrid.classList.remove("empty");
  summaryGrid.innerHTML = `
    <div class="metric">
      <span class="metric-label">Final Status</span>
      <strong class="metric-value">${summary.final_status ?? "unknown"}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Policy</span>
      <strong class="metric-value">${summary.policy_action ?? "unknown"} / ${summary.policy_allowed ? "allowed" : "blocked"}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Execution</span>
      <strong class="metric-value">${summary.execution_status ?? (summary.execution_attempted ? "attempted" : "skipped")}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Messages</span>
      <strong class="metric-value">${summary.message_count ?? 0}</strong>
    </div>
  `;
}

// This block renders the graph-emitted timeline.
// It takes: the final message list from the graph.
// It gives: one ordered operator timeline.
function renderTimeline(messages) {
  if (!messages || messages.length === 0) {
    timeline.innerHTML = `<li class="timeline-empty">No graph messages were returned.</li>`;
    return;
  }

  timeline.innerHTML = messages
    .map((message) => `<li>${escapeHtml(message)}</li>`)
    .join("");
}

// This block renders the reputation-gap gauge and supporting metrics.
// It takes: the backend-derived reflexive gap payload.
// It gives: a compact divergence view highlighting the current alpha state.
function renderGap(gap) {
  if (!gap) {
    gapLabel.textContent = "Waiting";
    gaugeFill.style.width = "50%";
    gaugeFill.className = "gauge-fill";
    gaugeNeedle.style.left = "50%";
    gapMetrics.innerHTML = "";
    gapThesis.textContent = "Run the workflow to compute the reflexive reputation gap.";
    return;
  }

  const normalized = Math.max(-40, Math.min(40, Number(gap.gap || 0)));
  const percent = ((normalized + 40) / 80) * 100;

  gapLabel.textContent = gap.label || "Waiting";
  gaugeFill.style.width = `${percent}%`;
  gaugeFill.className = `gauge-fill ${gap.state || "neutral"}`;
  gaugeNeedle.style.left = `${percent}%`;
  gapMetrics.innerHTML = `
    <div class="metric">
      <span class="metric-label">Intent Score</span>
      <strong class="metric-value">${Number(gap.intent_score || 0).toFixed(2)}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Order Book Sentiment</span>
      <strong class="metric-value">${Number(gap.orderbook_sentiment || 0).toFixed(2)}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">${escapeHtml(gap.formula || "Gap")}</span>
      <strong class="metric-value">${Number(gap.gap || 0).toFixed(2)}</strong>
    </div>
  `;
  gapThesis.textContent = gap.thesis || "No divergence thesis was returned.";
}

// This block renders the LangGraph cortex view and reasoning bubbles.
// It takes: node status data and streaming-thought copy from the backend.
// It gives: a visual execution map plus human-readable reasoning cues.
function renderCortex(cortex) {
  const nodes = cortex?.nodes || [];
  const thoughts = cortex?.thoughts || [];

  if (nodes.length === 0) {
    cortexGraph.innerHTML = `<div class="cortex-node queued"><strong>Waiting</strong><span>No node activity yet.</span></div>`;
  } else {
    cortexGraph.innerHTML = nodes
      .map((node) => `
        <div class="cortex-node ${escapeHtml(node.status || "queued")}">
          <strong>${escapeHtml(node.title || node.id || "Node")}</strong>
          <span>${escapeHtml(node.detail || "")}</span>
        </div>
      `)
      .join("");
  }

  if (thoughts.length === 0) {
    thoughtStream.innerHTML = `<div class="thought-bubble muted">Waiting for a graph run to emit live reasoning cues.</div>`;
    return;
  }

  thoughtStream.innerHTML = thoughts
    .map((thought, index) => `<div class="thought-bubble ${index === thoughts.length - 1 ? "active" : ""}">${escapeHtml(thought)}</div>`)
    .join("");
}

// This block renders institutional metrics and the equity-vs-benchmark chart.
// It takes: the performance payload from the backend.
// It gives: live metrics plus an SVG curve comparison.
function renderPerformance(performance) {
  if (!performance) {
    performanceGrid.innerHTML = "";
    equityCurve.innerHTML = "";
    return;
  }

  const edge = Number(performance.strategy_return_pct || 0) - Number(performance.benchmark_return_pct || 0);
  performanceGrid.innerHTML = `
    <div class="metric">
      <span class="metric-label">Sharpe Ratio</span>
      <strong class="metric-value">${Number(performance.sharpe_ratio || 0).toFixed(2)}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Max Drawdown</span>
      <strong class="metric-value">${Number(performance.max_drawdown_pct || 0).toFixed(2)}%</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Profit Factor</span>
      <strong class="metric-value">${Number(performance.profit_factor || 0).toFixed(2)}</strong>
    </div>
    <div class="metric">
      <span class="metric-label">Edge vs Benchmark</span>
      <strong class="metric-value">${edge.toFixed(2)}%</strong>
    </div>
  `;

  const strategySeries = performance.series?.strategy || [];
  const benchmarkSeries = performance.series?.benchmark || [];
  equityCurve.innerHTML = buildCurveSvg(strategySeries, benchmarkSeries);
}

// This block turns two equity series into a lightweight SVG chart.
// It takes: strategy and benchmark time series.
// It gives: SVG markup that compares both curves on the same y-scale.
function buildCurveSvg(strategySeries, benchmarkSeries) {
  const allValues = [...strategySeries, ...benchmarkSeries].map((point) => Number(point.value || 0));
  if (allValues.length === 0) {
    return "";
  }

  const width = 640;
  const height = 240;
  const padding = 20;
  const min = Math.min(...allValues);
  const max = Math.max(...allValues);
  const range = Math.max(max - min, 1);

  const buildPath = (series) => series
    .map((point, index) => {
      const x = padding + (index / Math.max(series.length - 1, 1)) * (width - padding * 2);
      const y = height - padding - ((Number(point.value || 0) - min) / range) * (height - padding * 2);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");

  const gridLines = [0.2, 0.5, 0.8]
    .map((ratio) => {
      const y = padding + ratio * (height - padding * 2);
      return `<line x1="${padding}" y1="${y}" x2="${width - padding}" y2="${y}" class="curve-grid" />`;
    })
    .join("");

  return `
    ${gridLines}
    <path d="${buildPath(benchmarkSeries)}" class="curve-line benchmark" />
    <path d="${buildPath(strategySeries)}" class="curve-line strategy" />
  `;
}

// This block renders the proof artifact and Top-50 tracked-agent table.
// It takes: the backend proof payload with validation artifacts and tracked agents.
// It gives: a visible signed-intent audit trail for judges and operators.
function renderProof(proof) {
  if (!proof) {
    proofArtifact.textContent = "No validation artifact available yet.";
    proofTableBody.innerHTML = `<tr><td colspan="7" class="table-empty">Run the workflow to inspect the tracked-agent cohort.</td></tr>`;
    return;
  }

  const artifact = proof.validation_artifact || {};
  proofArtifact.innerHTML = `
    <div class="artifact-header">
      <strong>${escapeHtml(artifact.artifact_id || "artifact-pending")}</strong>
      <span>${escapeHtml(artifact.network || "network-pending")}</span>
    </div>
    <div class="artifact-copy">
      <span>Digest: ${escapeHtml(shortenHex(artifact.eip712_digest))}</span>
      <span>Mock Tx: ${escapeHtml(shortenHex(artifact.mock_tx_hash))}</span>
      <a href="${escapeHtml(artifact.attestation_url || "#")}" target="_blank" rel="noreferrer">View attestation</a>
    </div>
  `;

  const agents = proof.tracked_agents || [];
  proofTableBody.innerHTML = agents
    .map((agent) => `
      <tr>
        <td>${escapeHtml(agent.agent_id)}</td>
        <td>${escapeHtml(agent.reputation_score)}</td>
        <td>${escapeHtml(agent.validation_score)}</td>
        <td><span class="stance-pill ${escapeHtml(agent.stance)}">${escapeHtml(agent.stance)}</span></td>
        <td>${Number(agent.virtual_pnl_bps || 0).toFixed(1)} bps</td>
        <td><a href="${escapeHtml(agent.attestation_url || "#")}" target="_blank" rel="noreferrer">Signed Intent</a></td>
        <td>${escapeHtml(shortenHex(agent.validation_artifact))}</td>
      </tr>
    `)
    .join("");
}

// This block renders the safety countdown and local human-in-the-loop state.
// It takes: the backend safety payload plus any browser-side operator override.
// It gives: a visible dead-man switch and manual intervention state.
function renderSafety(safety) {
  lastSafetyPayload = safety || null;
  if (deadmanTimerId) {
    window.clearInterval(deadmanTimerId);
    deadmanTimerId = null;
  }

  if (!safety) {
    deadmanCountdown.textContent = "60s";
    hitlStatus.textContent = "Standby";
    hitlReason.textContent = "No trade review state available yet.";
    return;
  }

  let secondsRemaining = Number(safety.dead_mans_switch_seconds || 60);
  deadmanCountdown.textContent = `${secondsRemaining}s`;
  deadmanTimerId = window.setInterval(() => {
    secondsRemaining = Math.max(secondsRemaining - 1, 0);
    deadmanCountdown.textContent = `${secondsRemaining}s`;
  }, 1000);

  const baseStatus = safety.human_review_required ? "Awaiting Review" : "Armed";
  const resolvedStatus = hitlOverride || baseStatus;
  hitlStatus.textContent = resolvedStatus;
  hitlReason.textContent = safety.veto_reason || "No manual veto required at current sizing.";
}

// This block applies manual operator overrides to the HITL panel.
// It takes: a local action string from one of the safety buttons.
// It gives: an updated operator status without changing backend state.
function applyHitlOverride(action) {
  if (!lastSafetyPayload) {
    return;
  }

  if (action === "pause") {
    hitlOverride = "Paused by Operator";
    hitlReason.textContent = "Graph paused locally for manual intervention.";
  } else if (action === "approve") {
    hitlOverride = "Approved";
    hitlReason.textContent = "Operator approved the current high-trust trade.";
  } else if (action === "reject") {
    hitlOverride = "Vetoed";
    hitlReason.textContent = "Operator vetoed the current trade candidate before execution.";
  }

  hitlStatus.textContent = hitlOverride;
}

// This block shortens long hex strings for compact UI display.
// It takes: a long digest or transaction hash.
// It gives: a short readable token.
function shortenHex(value) {
  const text = String(value || "");
  if (text.length <= 16) {
    return text;
  }
  return `${text.slice(0, 10)}...${text.slice(-8)}`;
}

// This block resets advanced panels to a neutral waiting state.
// It takes: no runtime input.
// It gives: a clean dashboard surface after errors or before a run.
function resetAdvancedPanels() {
  renderGap(null);
  renderCortex(null);
  renderPerformance(null);
  renderProof(null);
  lastSafetyPayload = null;
  hitlOverride = null;
  renderSafety(null);
  renderAccountStrip(null);
}

// This block persists the latest workflow response for the separate raw-data page.
// It takes: the API result from a completed run.
// It gives: a cached browser snapshot available at /raw.
function storeLatestRun(result) {
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(result));
}

// This block restores the last successful run so the dashboard doesn't boot empty every refresh.
// It takes: the browser's cached run payload from localStorage.
// It gives: a warm-start account strip and the last known operator view.
function restoreLatestRun() {
  const stored = window.localStorage.getItem(STORAGE_KEY);
  if (!stored) {
    renderAccountStrip(null);
    return;
  }

  try {
    const result = JSON.parse(stored);
    renderAccountStrip(result);
  } catch (error) {
    renderAccountStrip(null);
  }
}

// This block runs the workflow request and renders the compact operator surface.
// It takes: the form submission event plus the current button-group state.
// It gives: a fully rendered dashboard for success or failure.
async function runDemo(event) {
  event.preventDefault();
  setLoading(true);
  hitlOverride = null;

  const formData = new FormData(runForm);
  const payload = {
    execution_backend: String(formData.get("execution_backend") || "demo").trim(),
    symbol: String(formData.get("symbol") || "BTCUSD").trim(),
    user_objective: String(formData.get("user_objective") || "").trim(),
    agent_address: String(formData.get("agent_address") || "").trim(),
    use_live_inference: String(formData.get("use_live_inference") || "false") === "true",
  };

  try {
    const response = await fetch("/api/run-demo", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const result = await response.json();

    if (!response.ok) {
      const error = new Error(result?.error?.message || "Demo run failed.");
      error.payload = result?.error || null;
      throw error;
    }

    storeLatestRun(result);
    summaryCaption.textContent = `Run ${result.run_id} using ${result.mode} inference on ${result.execution_backend} backend.`;
    disclaimer.textContent = result.disclaimer || "";
    renderAccountStrip(result);
    renderSummary(result.summary || {});
    renderGap(result.reflexive?.gap || null);
    renderCortex(result.reflexive?.cortex || null);
    renderPerformance(result.reflexive?.performance || null);
    renderProof(result.reflexive?.proof || null);
    renderSafety(result.reflexive?.safety || null);
    renderTimeline(result.messages || []);
    tradeIntent.textContent = formatJson(result.trade_intent_record);
    policyState.textContent = formatJson(result.policy);
    trustState.textContent = formatJson(result.trust_snapshot);
    executionState.textContent = formatJson(result.execution);
  } catch (error) {
    summaryCaption.textContent = "The run failed before reaching a final state.";
    disclaimer.textContent = error.message;
    renderSummary({
      final_status: "error",
      policy_action: "n/a",
      policy_allowed: false,
      execution_status: "not_run",
      execution_attempted: false,
      message_count: 0,
    });
    resetAdvancedPanels();
    renderTimeline([]);
    tradeIntent.textContent = "No trade intent available.";
    policyState.textContent = "No policy state available.";
    trustState.textContent = "No trust snapshot available.";
    executionState.textContent = buildErrorDetails(error.payload) || error.message;
  } finally {
    setLoading(false);
  }
}

pauseGraphButton.addEventListener("click", () => applyHitlOverride("pause"));
approveTradeButton.addEventListener("click", () => applyHitlOverride("approve"));
rejectTradeButton.addEventListener("click", () => applyHitlOverride("reject"));
themeToggle.addEventListener("click", () => {
  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  applyTheme(currentTheme === "dark" ? "light" : "dark");
});
runForm.addEventListener("submit", runDemo);
initializeButtonGroups();
initializeTheme();
resetAdvancedPanels();
restoreLatestRun();
fetchHealth();
