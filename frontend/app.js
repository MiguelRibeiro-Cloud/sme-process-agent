(() => {
  "use strict";

  const elements = {
    chatForm: document.querySelector("#chat-form"),
    messageInput: document.querySelector("#message-input"),
    sendButton: document.querySelector("#send-button"),
    sendButtonLabel: document.querySelector("#send-button-label"),
    actionHint: document.querySelector("#action-hint"),
    sessionLabel: document.querySelector("#session-label"),
    conversation: document.querySelector("#conversation"),
    conversationPanel: document.querySelector(".conversation-panel"),
    conversationIntro: document.querySelector("#conversation-intro"),
    demoGuide: document.querySelector("#demo-guide"),
    activityFeed: document.querySelector("#activity-feed"),
    activityEmpty: document.querySelector("#activity-empty"),
    connectionState: document.querySelector("#connection-state"),
    eventCount: document.querySelector("#event-count"),
    assistantMessageTemplate: document.querySelector("#assistant-message-template"),
    userMessageTemplate: document.querySelector("#user-message-template"),
    traceTurnTemplate: document.querySelector("#trace-turn-template"),
    activityTemplate: document.querySelector("#activity-template"),
    processStatus: document.querySelector("#process-status"),
    processEmpty: document.querySelector("#process-empty"),
    processModel: document.querySelector("#process-model"),
    processName: document.querySelector("#process-name"),
    analyzeProcess: document.querySelector("#analyze-process"),
    resetDiscovery: document.querySelector("#reset-discovery"),
    analysisStatus: document.querySelector("#analysis-status"),
    analysisEmpty: document.querySelector("#analysis-empty"),
    analysisContent: document.querySelector("#analysis-content"),
    analysisStale: document.querySelector("#analysis-stale"),
    analysisSummary: document.querySelector("#analysis-summary"),
    proposalObjective: document.querySelector("#proposal-objective"),
    recommendationList: document.querySelector("#recommendation-list"),
    toBeList: document.querySelector("#to-be-list"),
    verificationSummary: document.querySelector("#verification-summary"),
    verificationList: document.querySelector("#verification-list"),
    runtimeModel: document.querySelector("#runtime-model"),
    mcpStatus: document.querySelector("#mcp-status"),
    mcpStatusLabel: document.querySelector("#mcp-status-label"),
    mcpServerName: document.querySelector("#mcp-server-name"),
    mcpTransport: document.querySelector("#mcp-transport"),
    mcpToolList: document.querySelector("#mcp-tool-list"),
    ragStatus: document.querySelector("#rag-status"),
    ragStatusLabel: document.querySelector("#rag-status-label"),
    ragModel: document.querySelector("#rag-model"),
    ragIndexDetail: document.querySelector("#rag-index-detail"),
    knowledgeCatalog: document.querySelector("#knowledge-catalog"),
    evaluationStatus: document.querySelector("#evaluation-status"),
    evaluationEmpty: document.querySelector("#evaluation-empty"),
    evaluationContent: document.querySelector("#evaluation-content"),
    evaluationMeta: document.querySelector("#evaluation-meta"),
    evaluationDisclaimer: document.querySelector("#evaluation-disclaimer"),
    evaluationMetrics: document.querySelector("#evaluation-metrics"),
    evaluationScenarios: document.querySelector("#evaluation-scenarios"),
  };

  const DEMO_PROMPT = "My sales team takes too long to build quotes.";

  const state = {
    source: null,
    eventCount: 0,
    interactionComplete: false,
    turnCount: 0,
    currentTurn: null,
    lifecycleRows: new Map(),
    processHasKnowledge: false,
  };

  const eventHandlers = new Map([
    ["llm_call_started", renderLlmCallStarted],
    ["llm_call_completed", renderLlmCallCompleted],
    ["tool_call_requested", renderToolCallRequested],
    ["mcp_tool_call_started", renderMcpToolCallStarted],
    ["mcp_tool_call_completed", renderMcpToolCallCompleted],
    ["rag_retrieval_started", renderRagRetrievalStarted],
    ["rag_query_embedded", renderRagQueryEmbedded],
    ["rag_retrieval_completed", renderRagRetrievalCompleted],
    ["rag_evidence_supplied", renderRagEvidenceSupplied],
    ["agent_error", renderAgentError],
    ["assistant_message", renderAssistantMessage],
    ["process_state_updated", renderProcessStateUpdated],
    ["interaction_completed", renderInteractionCompleted],
    ["orchestration_started", renderOrchestrationStarted],
    ["analysis_started", renderOrchestrationStageStarted],
    ["analysis_completed", renderOrchestrationStageCompleted],
    ["automation_design_started", renderOrchestrationStageStarted],
    ["automation_design_completed", renderOrchestrationStageCompleted],
    ["verification_started", renderOrchestrationStageStarted],
    ["verification_completed", renderOrchestrationStageCompleted],
    ["analysis_failed", renderOrchestrationStageFailed],
    ["automation_design_failed", renderOrchestrationStageFailed],
    ["verification_failed", renderOrchestrationStageFailed],
    ["orchestration_failed", renderOrchestrationFailed],
    ["orchestration_completed", renderOrchestrationCompleted],
  ]);

  elements.chatForm.addEventListener("submit", submitMessage);
  elements.resetDiscovery.addEventListener("click", resetDiscovery);
  elements.analyzeProcess?.addEventListener("click", startOrchestration);
  elements.conversation.addEventListener("click", (event) => {
    if (!event.target.closest("#demo-prompt")) return;
    fillComposer(DEMO_PROMPT);
  });
  document.querySelectorAll(".prompt-suggestion").forEach((suggestion) => {
    suggestion.addEventListener("click", () => fillComposer(suggestion.dataset.prompt || ""));
  });
  elements.messageInput.addEventListener("input", () => {
    resizeComposer();
    updateSendAvailability();
  });
  elements.messageInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      elements.chatForm.requestSubmit();
    }
  });

  updateSendAvailability();
  loadSessionViews();
  loadMcpStatus();
  loadRagStatus();
  loadKnowledgeCatalog();
  loadRuntimeStatus();
  loadEvaluationReport();

  function submitMessage(event) {
    event.preventDefault();
    const message = elements.messageInput.value.trim();
    if (!message || state.source) return;
    appendUserMessage(message);
    elements.messageInput.value = "";
    resizeComposer();
    startInteraction(message);
  }

  function fillComposer(prompt) {
    elements.messageInput.value = prompt;
    resizeComposer();
    updateSendAvailability();
    elements.messageInput.focus();
  }

  async function startInteraction(message) {
    beginTurn();
    state.interactionComplete = false;
    setRunningState();
    const url = new URL("/chat-stream", window.location.origin);
    url.searchParams.set("message", message);
    const controller = new AbortController();
    state.source = { close: () => controller.abort() };
    try {
      const response = await fetch(url, {
        headers: { Accept: "text/event-stream" },
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(await requestFailureMessage(response, "Discovery could not be started"));
      }
      if (!response.body) throw new Error("The discovery stream was unavailable.");
      setConnectionState("connected", "Streaming");
      await consumeEventStream(response.body);
      if (!state.interactionComplete) {
        throw new Error("The discovery stream ended before the interaction completed.");
      }
    } catch (error) {
      if (error.name === "AbortError" && state.interactionComplete) return;
      const messageText = error.message || "Discovery could not be started";
      renderAssistantMessage({ type: "assistant_message", content: messageText, error: true });
      finishInteraction(messageText, true);
    }
  }

  async function startOrchestration() {
    if (state.source || !state.processHasKnowledge) return;
    beginTurn("Process orchestration");
    state.interactionComplete = false;
    setRunningState("Analyzing validated process snapshot");
    const controller = new AbortController();
    state.source = { close: () => controller.abort() };
    try {
      const response = await fetch("/analyze-process", {
        method: "POST",
        headers: { Accept: "text/event-stream" },
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(await requestFailureMessage(response, analysisFailureMessage(response.status)));
      }
      if (!response.body) throw new Error("The analysis stream was unavailable.");
      setConnectionState("connected", "Orchestrating");
      await consumeEventStream(response.body);
      if (!state.interactionComplete) {
        throw new Error("The analysis stream ended before orchestration completed.");
      }
    } catch (error) {
      if (error.name === "AbortError" && state.interactionComplete) return;
      renderClientError(error.message || "Process analysis failed");
      finishOrchestrationTrace("Analysis could not be completed", true);
    }
  }

  async function consumeEventStream(body) {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const chunks = buffer.split(/\r?\n\r?\n/);
      buffer = chunks.pop() || "";
      chunks.forEach(dispatchEventChunk);
      if (done) break;
    }
    if (buffer.trim()) dispatchEventChunk(buffer);
  }

  function dispatchEventChunk(chunk) {
    const data = chunk
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (data) receiveEvent({ data });
  }

  function beginTurn(label) {
    state.turnCount += 1;
    state.lifecycleRows = new Map();
    elements.activityEmpty?.remove();
    const fragment = elements.traceTurnTemplate.content.cloneNode(true);
    const turn = fragment.querySelector(".trace-turn");
    turn.dataset.turn = String(state.turnCount);
    turn.querySelector("h3").textContent = label || `Turn ${state.turnCount}`;
    elements.activityFeed.append(fragment);
    state.currentTurn = {
      element: elements.activityFeed.lastElementChild,
      eventCount: 0,
    };
    scrollToEnd(elements.activityFeed);
  }

  function receiveEvent(message) {
    let event;
    try {
      event = JSON.parse(message.data);
    } catch (error) {
      renderClientError("An event could not be read", { raw: message.data });
      finishInteraction("Stream contained an invalid event", true);
      return;
    }
    if (!event || typeof event.type !== "string") {
      renderClientError("An event arrived without a type", event);
      return;
    }
    incrementEventCount();
    const handler = eventHandlers.get(event.type) || renderUnknownEvent;
    handler(event);
  }

  function renderLlmCallStarted(event) {
    appendActivity({
      tone: "running",
      sentence: llmSentence(event, "running"),
      fields: llmFields(event, "running"),
      raw: event,
      lifecycleKey: `llm:${event.operation || "uncorrelated"}`,
    });
  }

  function renderLlmCallCompleted(event) {
    const successful = (event.status || "success") === "success";
    completeLifecycle(`llm:${event.operation || "uncorrelated"}`, {
      tone: successful ? "completed" : "error",
      sentence: llmSentence(event, successful ? "completed" : "error"),
      fields: llmFields(event, event.status || "success"),
    }, event);
  }

  function renderToolCallRequested(event) {
    const model = formatModelName(event.model);
    appendActivity({
      tone: "neutral",
      sentence: `→ ${model || "The model"} requested ${event.tool || "an unnamed tool"}.`,
      fields: [
        ["Technology", event.technology],
        ["Model", event.model],
        ["Tool", event.tool],
        ["Arguments", event.arguments],
        ["Status", "requested"],
      ],
      raw: event,
    });
  }

  function renderMcpToolCallStarted(event) {
    appendActivity({
      tone: "running",
      sentence: `● Calling ${event.tool || "an unnamed tool"} through MCP on ${event.server || "the configured server"}…`,
      fields: mcpFields(event, "running"),
      raw: event,
      lifecycleKey: `mcp:${event.tool || "uncorrelated"}`,
    });
  }

  function renderMcpToolCallCompleted(event) {
    const successful = event.status === "success";
    const sentence = successful
      ? `✓ ${event.server || "The MCP server"} returned a result from ${event.tool || "the tool"} through MCP.`
      : `✕ MCP call to ${event.tool || "the tool"} failed.`;
    completeLifecycle(`mcp:${event.tool || "uncorrelated"}`, {
      tone: successful ? "completed" : "error",
      sentence,
      fields: mcpFields(event, event.status || "completed"),
    }, event);
  }

  function renderRagRetrievalStarted(event) {
    appendActivity({
      tone: "running",
      sentence: `● Searching Northstar company knowledge for “${event.query || "the requested topic"}”…`,
      fields: ragFields(event, "running"),
      raw: event,
      lifecycleKey: "rag:retrieval",
    });
  }

  function renderRagQueryEmbedded(event) {
    appendActivity({
      tone: "completed",
      sentence: `● Embedded the company-knowledge query using ${event.embedding_model || "the configured embedding model"}.`,
      fields: ragFields(event, "embedded"),
      raw: event,
    });
  }

  function renderRagRetrievalCompleted(event) {
    const results = Array.isArray(event.results) ? event.results : [];
    const successful = event.status === "success";
    completeLifecycle("rag:retrieval", {
      tone: successful ? "completed" : "error",
      sentence: successful
        ? `✓ Searched ${event.index_chunk_count || 0} indexed Northstar document chunks and retrieved ${results.length} relevant ${results.length === 1 ? "result" : "results"}.`
        : "✕ Company-knowledge retrieval failed.",
      fields: ragFields(event, event.status || "completed"),
    }, event);
  }

  function renderRagEvidenceSupplied(event) {
    const documents = Array.isArray(event.documents) ? event.documents : [];
    appendActivity({
      tone: "completed",
      sentence: `→ Supplied evidence${documents.length ? ` from ${documents.join(", ")}` : ""} to ${formatModelName(event.model) || "the model"}.`,
      fields: [
        ["Technology", event.technology],
        ["Model", event.model],
        ["Documents", documents],
        ["Chunk IDs", event.chunk_ids],
        ["Status", "supplied"],
      ],
      raw: event,
    });
  }

  function renderAgentError(event) {
    appendActivity({
      tone: "error",
      sentence: `✕ ${agentErrorMessage(event)}.`,
      fields: [["Stage", event.stage], ["Code", event.code], ["Status", "error"]],
      raw: { type: event.type, stage: event.stage, code: event.code },
    });
  }

  function renderAssistantMessage(event) {
    const shouldFollow = isNearEnd(elements.conversation);
    const fragment = elements.assistantMessageTemplate.content.cloneNode(true);
    const message = fragment.querySelector(".chat-message");
    message.classList.toggle("message-error", Boolean(event.error));
    renderSafeMarkdown(fragment.querySelector(".message-body, .message-content > p"), event.content || "");
    elements.conversationIntro?.remove();
    elements.conversation.append(fragment);
    if (shouldFollow) scrollMessageIntoView(elements.conversation.lastElementChild);
    appendActivity({
      tone: event.error ? "error" : "completed",
      sentence: event.error ? "✕ Assistant error response delivered." : "✓ Assistant response delivered.",
      fields: [["Destination", "Conversation"], ["Event type", event.type], ["Status", event.error ? "error" : "delivered"]],
      raw: event,
    });
  }

  function renderProcessStateUpdated(event) {
    renderProcessState(event.state);
    loadAnalysis();
    appendActivity({
      tone: "completed",
      sentence: `✓ Updated process model: ${summarizePatch(event.patch)}.`,
      fields: [
        ["Technology", event.technology],
        ["Validation owner", event.validation_owner],
        ["Ownership", event.state_owner],
        ["Validation", event.validation_status],
        ["Applied patch", event.patch],
        ["Resulting state", event.state],
        ["Status", "applied"],
      ],
      raw: event,
    });
  }

  function renderOrchestrationStarted(event) {
    appendActivity({
      tone: "running",
      sentence: `● Started deterministic orchestration over a validated ProcessState snapshot.`,
      fields: orchestrationFields(event, "running"),
      raw: event,
      lifecycleKey: "orchestration",
    });
  }

  function renderOrchestrationStageStarted(event) {
    const stage = event.stage || "analysis";
    const sentences = {
      analysis: `● Sent validated ProcessState to ${formatModelName(event.model) || "the configured model"} for AS-IS analysis.`,
      automation_design: `● Sent ProcessState and analysis to ${formatModelName(event.model) || "the configured model"} for automation design.`,
      verification: `● Sent the proposal to ${formatModelName(event.model) || "the configured model"} for evidence verification.`,
    };
    appendActivity({
      tone: "running",
      sentence: sentences[stage] || `● Started ${humanize(stage)}.`,
      fields: orchestrationFields(event, "running"),
      raw: event,
      lifecycleKey: `orchestration:${stage}`,
    });
  }

  function renderOrchestrationStageCompleted(event) {
    const stage = event.stage;
    let sentence = `✓ ${humanize(stage)} completed.`;
    if (stage === "analysis") {
      sentence = `✓ Process Analyst identified ${event.finding_count || 0} supported or qualified findings.`;
    } else if (stage === "automation_design") {
      sentence = `✓ Automation Designer proposed ${event.recommendation_count || 0} changes and retained ${event.retained_human_decision_count || 0} human decisions.`;
    } else if (stage === "verification") {
      const counts = event.status_counts || {};
      sentence = `✓ Evidence Verifier: ${counts.supported || 0} supported, ${counts.partially_supported || 0} partial, ${counts.unsupported || 0} unsupported, ${counts.blocked_by_unknown || 0} blocked.`;
    }
    completeLifecycle(`orchestration:${stage}`, {
      tone: "completed",
      sentence,
      fields: orchestrationFields(event, event.status || "success"),
    }, event);
  }

  function renderOrchestrationStageFailed(event) {
    const stage = event.stage || String(event.type).replace("_failed", "");
    completeLifecycle(`orchestration:${stage}`, {
      tone: "error",
      sentence: `✕ ${humanize(stage)} failed validation; later stages were not run.`,
      fields: orchestrationFields(event, "error"),
    }, event);
  }

  function renderOrchestrationFailed(event) {
    completeLifecycle("orchestration", {
      tone: "error",
      sentence: `✕ ${event.message || "Orchestration stopped before completion."}`,
      fields: orchestrationFields(event, "error"),
    }, event);
    state.interactionComplete = true;
    finishOrchestrationTrace("Analysis stopped safely", true);
  }

  function renderOrchestrationCompleted(event) {
    renderAnalysisResult(event.result);
    completeLifecycle("orchestration", {
      tone: "completed",
      sentence: "✓ Orchestration completed and the verified proposal was stored separately from ProcessState.",
      fields: orchestrationFields(event, "success"),
    }, event);
    state.interactionComplete = true;
    finishOrchestrationTrace("Verified proposal ready", false);
  }

  function finishOrchestrationTrace(hint, isError) {
    if (state.currentTurn) {
      const label = state.currentTurn.element.querySelector(".trace-turn-status");
      label.textContent = isError ? "Error" : "Complete";
      label.className = `trace-turn-status ${isError ? "error" : "complete"}`;
    }
    finishInteraction(hint, isError);
  }

  function renderInteractionCompleted(event) {
    const isError = event.status === "error";
    const isPartial = event.status === "partial";
    const status = isPartial ? "Partial" : (isError ? "Error" : "Complete");
    if (state.currentTurn) {
      const label = state.currentTurn.element.querySelector(".trace-turn-status");
      label.textContent = status;
      label.className = `trace-turn-status ${isError ? "error" : (isPartial ? "partial" : "complete")}`;
    }
    state.interactionComplete = true;
    finishInteraction(
      isPartial ? "Response received; extraction needs attention" : (isError ? "Request ended with an error" : "Response and process model complete"),
      isError
    );
  }

  function appendUserMessage(content) {
    const fragment = elements.userMessageTemplate.content.cloneNode(true);
    fragment.querySelector(".message-body, .message-content > p").textContent = content;
    elements.conversationIntro?.remove();
    elements.conversationPanel?.classList.add("has-conversation");
    if (elements.demoGuide) elements.demoGuide.open = false;
    elements.conversation.append(fragment);
    scrollToEnd(elements.conversation, false);
  }

  function renderUnknownEvent(event) {
    appendActivity({
      tone: "neutral",
      sentence: `Received backend event: ${humanize(event.type)}.`,
      fields: Object.entries(event).filter(([key]) => key !== "type").slice(0, 6).map(([key, value]) => [humanize(key), value]),
      raw: event,
    });
  }

  function appendActivity({ tone, sentence, fields, raw, lifecycleKey }) {
    if (!state.currentTurn) return null;
    const fragment = elements.activityTemplate.content.cloneNode(true);
    const item = fragment.querySelector(".activity-item");
    updateActivity(item, { tone, sentence, fields }, raw);
    const eventList = state.currentTurn.element.querySelector(".trace-turn-events");
    eventList.append(fragment);
    const appended = eventList.lastElementChild;
    if (lifecycleKey) state.lifecycleRows.set(lifecycleKey, { item: appended, startedEvent: raw });
    scrollToEnd(elements.activityFeed, true);
    return appended;
  }

  function completeLifecycle(key, view, completedEvent) {
    const lifecycle = state.lifecycleRows.get(key);
    if (!lifecycle) {
      appendActivity({ ...view, raw: completedEvent });
      return;
    }
    updateActivity(lifecycle.item, view, { started: lifecycle.startedEvent, completed: completedEvent });
    state.lifecycleRows.delete(key);
  }

  function updateActivity(item, { tone, sentence, fields }, raw) {
    item.classList.remove("running", "completed", "error", "neutral");
    item.classList.add(tone || "neutral");
    item.querySelector(".trace-sentence").textContent = sentence;
    const fieldList = item.querySelector(".event-fields");
    fieldList.replaceChildren();
    (fields || []).filter(([, value]) => value !== undefined && value !== null && value !== "").forEach(([label, value]) => {
      const wrapper = document.createElement("div");
      const term = document.createElement("dt");
      const description = document.createElement("dd");
      term.textContent = label;
      description.textContent = displayValue(value);
      wrapper.append(term, description);
      fieldList.append(wrapper);
    });
    item.querySelector("pre").textContent = JSON.stringify(raw, null, 2);
  }

  function renderClientError(title, raw = {}) {
    if (!state.currentTurn) {
      showConnectionError(title, "The browser could not complete this action. Please try again.");
      return;
    }
    appendActivity({ tone: "error", sentence: `✕ ${title}.`, fields: [["Origin", "Browser event handling"]], raw });
  }

  function handleStreamError() {
    if (state.interactionComplete) return;
    showConnectionError("Connection interrupted", "The backend event stream closed before the interaction completed. You can try again.");
    finishInteraction("Connection interrupted — try again", true);
  }

  function showConnectionError(titleText, copyText) {
    const existingNotice = elements.activityFeed.querySelector(".stream-notice");
    if (existingNotice) return;
    elements.activityEmpty?.remove();
    const notice = document.createElement("div");
    notice.className = "stream-notice";
    notice.setAttribute("role", "alert");
    const title = document.createElement("strong");
    const copy = document.createElement("span");
    title.textContent = titleText;
    copy.textContent = copyText;
    notice.append(title, copy);
    elements.activityFeed.append(notice);
  }

  function finishInteraction(hint, isError) {
    state.source?.close();
    state.source = null;
    elements.messageInput.disabled = false;
    elements.conversation.setAttribute("aria-busy", "false");
    elements.activityFeed.setAttribute("aria-busy", "false");
    elements.sendButtonLabel.textContent = "Send";
    elements.actionHint.textContent = hint;
    elements.sessionLabel.textContent = isError ? "Needs attention" : "Turn complete";
    elements.resetDiscovery.disabled = false;
    if (elements.analyzeProcess) elements.analyzeProcess.disabled = !state.processHasKnowledge;
    setConnectionState(isError ? "error" : "complete", isError ? "Error" : "Complete");
    updateSendAvailability();
    elements.messageInput.focus();
  }

  function setRunningState(hint = "Receiving backend events") {
    elements.messageInput.disabled = true;
    elements.conversation.setAttribute("aria-busy", "true");
    elements.activityFeed.setAttribute("aria-busy", "true");
    elements.sendButton.disabled = true;
    elements.sendButtonLabel.textContent = "Running";
    elements.actionHint.textContent = hint;
    elements.sessionLabel.textContent = `Turn ${state.turnCount} in progress`;
    elements.resetDiscovery.disabled = true;
    if (elements.analyzeProcess) elements.analyzeProcess.disabled = true;
    setConnectionState("connected", "Connecting");
  }

  async function loadSessionViews() {
    // Resolve the anonymous cookie before issuing another session-aware request.
    await loadProcessState();
    await loadAnalysis();
  }

  async function loadProcessState() {
    try {
      const response = await fetch("/process-state", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("State request failed");
      renderProcessState(await response.json());
    } catch (error) {
      elements.processStatus.textContent = "State unavailable";
    }
  }

  async function loadAnalysis() {
    if (!elements.analysisStatus) return;
    try {
      const response = await fetch("/analysis", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Analysis request failed");
      const payload = await response.json();
      renderAnalysisResult(payload.result || null, Boolean(payload.stale));
    } catch (error) {
      elements.analysisStatus.textContent = "Unavailable";
    }
  }

  async function loadRuntimeStatus() {
    try {
      const response = await fetch("/runtime-status", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Runtime status request failed");
      const runtime = await response.json();
      elements.runtimeModel.textContent = formatModelName(runtime.model) || "Unavailable";
    } catch (error) {
      elements.runtimeModel.textContent = "Unavailable";
    }
  }

  async function loadEvaluationReport() {
    if (!elements.evaluationStatus) return;
    try {
      const response = await fetch("/evaluation-report", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Evaluation report request failed");
      const payload = await response.json();
      renderEvaluationReport(payload.report || null);
    } catch (error) {
      elements.evaluationStatus.textContent = "Report unavailable";
      elements.evaluationEmpty.hidden = false;
      elements.evaluationContent.hidden = true;
    }
  }

  function renderEvaluationReport(report) {
    if (!report) {
      elements.evaluationStatus.textContent = "No saved run";
      elements.evaluationEmpty.hidden = false;
      elements.evaluationContent.hidden = true;
      return;
    }
    elements.evaluationEmpty.hidden = true;
    elements.evaluationContent.hidden = false;
    elements.evaluationStatus.textContent = `${report.scenario_runs_passed}/${report.scenario_runs_total} scenario runs passed`;
    const created = new Date(report.created_at);
    const fields = [
      ["Last evaluation", Number.isNaN(created.valueOf()) ? report.created_at : created.toLocaleString()],
      ["Model", formatModelName(report.model)],
      ["Eval model", formatModelName(report.eval_model)],
      ["Scenarios", String(report.selected_scenario_count)],
      ["Runs", String(report.scenario_execution_count)],
      ["Repeat policy", `${report.runs_per_scenario} run${report.runs_per_scenario === 1 ? "" : "s"} per scenario`],
    ];
    elements.evaluationMeta.replaceChildren(...fields.map(([label, value]) => {
      const item = document.createElement("div");
      const term = document.createElement("span");
      const detail = document.createElement("strong");
      term.textContent = label;
      detail.textContent = value || "Unavailable";
      item.append(term, detail);
      return item;
    }));
    const deterministicChecks = report.results.flatMap((result) => result.criteria).filter((criterion) => criterion.evaluator === "deterministic" && !criterion.skipped).length;
    const semanticChecks = report.results.flatMap((result) => result.criteria).filter((criterion) => criterion.evaluator === "semantic" && !criterion.skipped).length;
    elements.evaluationDisclaimer.textContent = `Deterministic checks: ${deterministicChecks} · Semantic judge checks: ${semanticChecks}. ${report.runs_per_scenario === 1 ? "One run per scenario is a regression signal, not a statistically robust estimate." : "Repeated runs expose model variability."}`;
    elements.evaluationMetrics.replaceChildren(...report.metrics.map((metric) => {
      const row = document.createElement("div");
      const label = document.createElement("span");
      const score = document.createElement("strong");
      label.textContent = metric.metric;
      score.textContent = `${metric.passed}/${metric.total} (${Math.round(metric.pass_rate * 100)}%)`;
      row.append(label, score);
      return row;
    }));
    const grouped = new Map();
    report.results.forEach((result) => {
      const group = grouped.get(result.scenario_id) || [];
      group.push(result);
      grouped.set(result.scenario_id, group);
    });
    elements.evaluationScenarios.replaceChildren(...Array.from(grouped.values()).map((runs) => {
      const card = document.createElement("article");
      const header = document.createElement("header");
      const title = document.createElement("strong");
      const score = document.createElement("span");
      const passed = runs.filter((run) => run.passed).length;
      title.textContent = runs[0].scenario_name;
      score.textContent = `Passed ${passed}/${runs.length} runs`;
      header.append(title, score);
      card.append(header);
      const failures = runs.flatMap((run) => run.criteria.filter((criterion) => !criterion.passed && !criterion.skipped).map((criterion) => ({ run: run.run_number, ...criterion })));
      if (!failures.length) {
        const ok = document.createElement("p");
        ok.textContent = "All evaluated criteria passed.";
        card.append(ok);
      } else {
        const list = document.createElement("ul");
        failures.forEach((failure) => {
          const item = document.createElement("li");
          item.textContent = `Run ${failure.run} · ${failure.evaluator === "semantic" ? "Semantic judge" : "Deterministic"} · ${failure.description}: ${failure.explanation}`;
          list.append(item);
        });
        card.append(list);
      }
      return card;
    }));
  }

  async function loadMcpStatus() {
    try {
      const response = await fetch("/mcp-status", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("MCP status request failed");
      const mcp = await response.json();
      const toolCount = Number(mcp.tools_discovered || 0);
      elements.mcpStatus.classList.toggle("disconnected", !mcp.connected);
      elements.mcpStatusLabel.textContent = mcp.connected
        ? `MCP: Connected · ${toolCount} ${toolCount === 1 ? "tool" : "tools"}`
        : "MCP: Unavailable";
      elements.mcpServerName.textContent = mcp.server || "Northstar Business Systems";
      elements.mcpTransport.textContent = `Transport: ${mcp.transport || "stdio"}`;
      elements.mcpToolList.replaceChildren();
      (mcp.tools || []).forEach((tool) => {
        const item = document.createElement("li");
        const name = document.createElement("strong");
        const description = document.createElement("span");
        const capability = typeof tool === "string" ? { name: tool, description: "" } : tool;
        name.textContent = capability.name || "Unnamed capability";
        description.textContent = capability.description || "Discovered business-system capability.";
        item.append(name, description);
        elements.mcpToolList.append(item);
      });
      if (!mcp.connected && mcp.error) {
        const item = document.createElement("li");
        item.textContent = "Server connection unavailable";
        elements.mcpToolList.append(item);
      }
    } catch (error) {
      elements.mcpStatus.classList.add("disconnected");
      elements.mcpStatusLabel.textContent = "MCP: Status unavailable";
    }
  }

  async function loadRagStatus() {
    if (!elements.ragStatus) return;
    try {
      const response = await fetch("/rag-status", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("RAG status request failed");
      const rag = await response.json();
      elements.ragStatus.classList.toggle("disconnected", !rag.available);
      elements.ragStatusLabel.textContent = rag.available
        ? "Knowledge: Ready"
        : (rag.stale ? "Knowledge: Stale" : "Knowledge: Unavailable");
      elements.ragModel.textContent = `Embedding model: ${rag.embedding_model || "Unavailable"}`;
      elements.ragIndexDetail.textContent = rag.available
        ? `${rag.documents || 0} documents · ${rag.chunks || 0} chunks`
        : (rag.stale ? "Index requires rebuild" : "Index unavailable");
    } catch (error) {
      elements.ragStatus.classList.add("disconnected");
      elements.ragStatusLabel.textContent = "Knowledge: Status unavailable";
      elements.ragModel.textContent = "Embedding model: unavailable";
      elements.ragIndexDetail.textContent = "Index status unavailable";
    }
  }

  async function loadKnowledgeCatalog() {
    if (!elements.knowledgeCatalog) return;
    try {
      const response = await fetch("/knowledge-catalog", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Knowledge catalogue request failed");
      const payload = await response.json();
      const documents = Array.isArray(payload.documents) ? payload.documents : [];
      elements.knowledgeCatalog.replaceChildren(...documents.map((documentMetadata) => {
        const item = document.createElement("li");
        const heading = document.createElement("div");
        const title = document.createElement("strong");
        const count = document.createElement("span");
        const filename = document.createElement("code");
        const topics = document.createElement("p");
        title.textContent = documentMetadata.title || "Untitled document";
        const chunkCount = Number(documentMetadata.chunk_count || 0);
        count.textContent = `${chunkCount} ${chunkCount === 1 ? "chunk" : "chunks"}`;
        filename.textContent = documentMetadata.filename || "";
        topics.textContent = `Topics: ${documentMetadata.topics || "indexed Northstar company knowledge"}`;
        heading.append(title, count);
        item.append(heading, filename, topics);
        return item;
      }));
      if (!documents.length) {
        const item = document.createElement("li");
        item.textContent = "Document catalogue unavailable";
        elements.knowledgeCatalog.append(item);
      }
    } catch (error) {
      const item = document.createElement("li");
      item.textContent = "Document catalogue unavailable";
      elements.knowledgeCatalog.replaceChildren(item);
    }
  }

  async function resetDiscovery() {
    if (state.source) return;
    const confirmed = window.confirm(
      "Reset this discovery? This clears the conversation, process model, recent backend context, and execution trace."
    );
    if (!confirmed) return;
    elements.resetDiscovery.disabled = true;
    try {
      const response = await fetch("/reset-discovery", { method: "POST", headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error("Reset request failed");
      const result = await response.json();
      renderProcessState(result.state);
      renderAnalysisResult(null);
      elements.conversation.replaceChildren(createConversationIntro());
      elements.conversationPanel?.classList.remove("has-conversation");
      elements.activityFeed.replaceChildren(createActivityEmpty());
      state.eventCount = 0;
      state.turnCount = 0;
      state.currentTurn = null;
      state.lifecycleRows = new Map();
      elements.conversation.setAttribute("aria-busy", "false");
      elements.activityFeed.setAttribute("aria-busy", "false");
      elements.eventCount.textContent = "0 events";
      elements.sessionLabel.textContent = "Ready";
      elements.actionHint.textContent = "Discovery reset";
      setConnectionState("idle", "Idle");
    } catch (error) {
      renderClientError("Discovery could not be reset");
    } finally {
      elements.resetDiscovery.disabled = false;
    }
  }

  function renderProcessState(processState = {}) {
    const legacySteps = Array.isArray(processState.steps)
      ? processState.steps.map((description, index) => ({ step_id: `legacy-step-${index + 1}`, description }))
      : [];
    const legacyDecisions = Array.isArray(processState.decisions)
      ? processState.decisions.map((question, index) => ({ decision_id: `legacy-decision-${index + 1}`, question, branches: [] }))
      : [];
    const flow = processState.flow || { steps: legacySteps, decisions: legacyDecisions };
    const hasFlow = (flow.steps || []).length > 0 || (flow.decisions || []).length > 0;
    const hasKnowledge = hasFlow || ["pain_points", "evidence", "conflicts"].some(
      (field) => Array.isArray(processState[field]) && processState[field].length > 0
    );
    state.processHasKnowledge = hasKnowledge;
    if (elements.analyzeProcess) {
      elements.analyzeProcess.disabled = Boolean(state.source) || !hasKnowledge;
      elements.analyzeProcess.title = hasKnowledge
        ? "Run analysis, automation design, and evidence verification"
        : "Discover process knowledge before running analysis";
    }
    elements.processEmpty.hidden = hasKnowledge;
    elements.processModel.hidden = !hasKnowledge;
    elements.processStatus.textContent = hasKnowledge ? "Discovery in progress" : "Awaiting discovery";
    if (!hasKnowledge) return;
    elements.processName.textContent = processState.process_name || "Not named yet";
    document.querySelectorAll(".knowledge-card").forEach((card) => {
      const field = card.dataset.field;
      const list = card.querySelector(".knowledge-list");
      if (field === "flow") {
        renderProcessFlow(list, flow);
        return;
      }
      const values = Array.isArray(processState[field]) ? processState[field] : [];
      list.replaceChildren();
      if (values.length === 0) {
        const empty = document.createElement("p");
        empty.className = "knowledge-empty";
        empty.textContent = "None captured";
        list.append(empty);
        return;
      }
      values.forEach((value, index) => {
        const item = document.createElement("div");
        item.className = "knowledge-item";
        if (field === "evidence" && value && typeof value === "object") {
          const claim = document.createElement("span");
          const source = document.createElement("small");
          claim.textContent = value.claim || "Evidence";
          source.className = "evidence-source";
          source.textContent = evidenceSourceLabel(value);
          item.append(claim, source);
          if (value.source_type === "document" && (value.source || value.chunk_id)) {
            const details = document.createElement("details");
            details.className = "provenance-details";
            const summary = document.createElement("summary");
            summary.textContent = "Source details";
            const metadata = document.createElement("span");
            metadata.textContent = [value.source && `File: ${value.source}`, value.chunk_id && `Chunk: ${value.chunk_id}`].filter(Boolean).join(" · ");
            details.append(summary, metadata);
            item.append(details);
          }
        } else if (field === "conflicts" && value && typeof value === "object") {
          item.classList.add("conflict-item");
          const topic = document.createElement("strong");
          topic.textContent = value.topic || "Evidence conflict";
          const sides = orderConflictSides(value);
          const reported = createConflictSide("Reported practice", sides.reported.claim, sides.reported.source);
          const documented = createConflictSide("Documented policy", sides.documented.claim, sides.documented.source);
          const status = document.createElement("span");
          status.className = "conflict-status";
          status.textContent = `Status: ${humanize(value.status || "unresolved")}`;
          item.append(topic, reported, documented, status);
        } else {
          const text = document.createElement("span");
          text.textContent = String(value);
          item.append(text);
        }
        list.append(item);
      });
    });
  }

  function renderProcessFlow(container, flow = {}) {
    container.replaceChildren();
    const steps = Array.isArray(flow.steps) ? flow.steps : [];
    const decisions = Array.isArray(flow.decisions) ? flow.decisions : [];
    const stepById = new Map(steps.map((step) => [step.step_id, step]));
    const branchStepIds = new Set(
      decisions.flatMap((decision) => (decision.branches || []).flatMap((branch) => branch.next_step_ids || []))
    );
    const rootSteps = steps.filter((step) => !branchStepIds.has(step.step_id));

    if (rootSteps.length === 0 && decisions.length === 0) {
      const empty = document.createElement("p");
      empty.className = "knowledge-empty";
      empty.textContent = "No flow captured";
      container.append(empty);
      return;
    }

    rootSteps.forEach((step, index) => container.append(createFlowStep(step, index + 1)));
    decisions.forEach((decision) => {
      const block = document.createElement("section");
      block.className = "flow-decision";
      const heading = document.createElement("strong");
      heading.textContent = `Decision: ${decision.question}`;
      block.append(heading);

      if (!(decision.branches || []).length) {
        const unresolved = document.createElement("p");
        unresolved.className = "flow-route-unknown";
        unresolved.textContent = "Routes not established yet";
        block.append(unresolved);
      }
      (decision.branches || []).forEach((branch) => {
        const route = document.createElement("div");
        route.className = "flow-branch";
        const condition = document.createElement("span");
        condition.className = "flow-condition";
        condition.textContent = branch.condition;
        route.append(condition);
        (branch.next_step_ids || []).forEach((stepId, index) => {
          const step = stepById.get(stepId);
          if (step) route.append(createFlowStep(step, index + 1, true));
        });
        block.append(route);
      });
      container.append(block);
    });
  }

  function createFlowStep(step, number, nested = false) {
    const item = document.createElement("div");
    item.className = `flow-step${nested ? " nested" : ""}`;
    const marker = document.createElement("span");
    marker.className = "flow-connector";
    marker.textContent = nested ? "→" : String(number);
    const content = document.createElement("div");
    const description = document.createElement("span");
    description.textContent = step.description || "Unnamed step";
    content.append(description);
    const context = [step.actor, step.system].filter(Boolean);
    if (context.length) {
      const metadata = document.createElement("small");
      metadata.textContent = context.join(" · ");
      content.append(metadata);
    }
    item.append(marker, content);
    return item;
  }

  function renderAnalysisResult(result, staleOverride = false) {
    if (!elements.analysisContent) return;
    const available = Boolean(result && result.analysis && result.proposal && result.verification);
    elements.analysisEmpty.hidden = available;
    elements.analysisContent.hidden = !available;
    elements.analysisStatus.textContent = available
      ? ((staleOverride || result.stale) ? "Potentially stale" : "Verified proposal")
      : "Not analyzed";
    if (!available) {
      elements.analysisSummary.textContent = "";
      elements.proposalObjective.textContent = "";
      elements.recommendationList.replaceChildren();
      elements.toBeList.replaceChildren();
      elements.verificationSummary.replaceChildren();
      elements.verificationList.replaceChildren();
      return;
    }

    const isStale = Boolean(staleOverride || result.stale);
    elements.analysisStale.hidden = !isStale;
    elements.analysisSummary.textContent = result.analysis.summary || "No summary returned.";
    ["bottlenecks", "manual_handoffs", "duplicate_work", "policy_practice_gaps", "unresolved_questions"].forEach((field) => {
      const container = document.querySelector(`[data-analysis-list="${field}"]`);
      renderResultList(container, result.analysis[field]);
    });

    const proposal = result.proposal;
    const findings = new Map((result.analysis.findings || []).map((item) => [item.finding_id, item]));
    const verifications = new Map((result.verification.verifications || []).map((item) => [item.recommendation_id, item]));
    elements.proposalObjective.textContent = proposal.objective || "";
    elements.recommendationList.replaceChildren();
    (proposal.recommendations || []).forEach((recommendation) => {
      const check = verifications.get(recommendation.recommendation_id) || { status: "blocked_by_unknown" };
      const card = document.createElement("article");
      card.className = "recommendation-card";
      if (check.status === "unsupported") card.classList.add("rejected");
      if (["partially_supported", "blocked_by_unknown"].includes(check.status)) card.classList.add("unresolved");

      const heading = document.createElement("div");
      heading.className = "recommendation-title";
      const title = document.createElement("h4");
      title.textContent = recommendation.title || recommendation.recommendation_id;
      const badge = document.createElement("span");
      badge.className = `status-badge ${check.status}`;
      badge.textContent = humanize(check.status);
      heading.append(title, badge);
      const description = document.createElement("p");
      description.textContent = recommendation.description || "";
      const addressed = (recommendation.addresses_finding_ids || [])
        .map((identifier) => findings.get(identifier)?.description || identifier);
      const essentials = document.createElement("div");
      essentials.className = "recommendation-essentials";
      [["Problem addressed", addressed], ["Human control", recommendation.human_control || "No explicit control point supplied"]]
        .forEach(([label, value]) => essentials.append(createRecommendationDetail(label, value)));
      const disclosure = document.createElement("details");
      disclosure.className = "recommendation-disclosure";
      const disclosureSummary = document.createElement("summary");
      disclosureSummary.textContent = "Implementation details";
      const details = document.createElement("div");
      details.className = "recommendation-details";
      [["What changes", recommendation.proposed_automation], ["Dependencies", recommendation.dependencies], ["Risks", recommendation.risks], ["Evidence IDs", recommendation.supporting_evidence_ids]]
        .forEach(([label, value]) => details.append(createRecommendationDetail(label, value)));
      disclosure.append(disclosureSummary, details);
      card.append(heading, description, essentials, disclosure);
      elements.recommendationList.append(card);
    });
    if (!(proposal.recommendations || []).length) elements.recommendationList.append(createCompactEmpty("No automation recommendations were proposed."));
    if (proposal.retained_human_decisions?.length) elements.recommendationList.append(createDesignContext("Retained human decisions", proposal.retained_human_decisions));
    if (proposal.assumptions?.length) elements.recommendationList.append(createDesignContext("Assumptions", proposal.assumptions));

    elements.toBeList.replaceChildren();
    [...(proposal.to_be_steps || [])].sort((left, right) => left.order - right.order).forEach((step) => {
      const row = document.createElement("article");
      row.className = `to-be-step ${step.automation_level}`;
      const order = document.createElement("span");
      order.className = "to-be-order";
      order.textContent = String(step.order);
      const copy = document.createElement("div");
      copy.className = "to-be-copy";
      const description = document.createElement("strong");
      description.textContent = step.description;
      const owner = document.createElement("span");
      const ownerLabel = document.createElement("b");
      ownerLabel.textContent = step.human_owner ? "Human owner: " : "Human owner: none";
      owner.append(ownerLabel);
      if (step.human_owner) owner.append(document.createTextNode(String(step.human_owner)));
      copy.append(description, owner);
      const level = document.createElement("span");
      level.className = "automation-label";
      level.textContent = humanize(step.automation_level);
      row.append(order, copy, level);
      elements.toBeList.append(row);
    });
    if (!(proposal.to_be_steps || []).length) elements.toBeList.append(createCompactEmpty("No TO-BE steps were proposed."));

    renderVerification(result.verification, proposal);
  }

  function renderResultList(container, values) {
    if (!container) return;
    container.replaceChildren();
    container.closest("article")?.classList.toggle("is-empty", !Array.isArray(values) || values.length === 0);
    if (!Array.isArray(values) || values.length === 0) {
      const empty = document.createElement("span");
      empty.className = "knowledge-empty";
      empty.textContent = "None identified";
      container.append(empty);
      return;
    }
    values.forEach((value) => {
      const item = document.createElement("span");
      item.className = "result-item";
      item.textContent = String(value);
      container.append(item);
    });
  }

  function createRecommendationDetail(labelText, value) {
    const wrapper = document.createElement("div");
    wrapper.className = "recommendation-detail";
    const label = document.createElement("strong");
    label.textContent = labelText;
    const copy = document.createElement("span");
    copy.textContent = Array.isArray(value) ? (value.join(" · ") || "None identified") : String(value || "None identified");
    wrapper.append(label, copy);
    return wrapper;
  }

  function createDesignContext(label, values) {
    const card = document.createElement("article");
    card.className = "recommendation-card";
    const title = document.createElement("h4");
    title.textContent = label;
    const copy = document.createElement("p");
    copy.textContent = Array.isArray(values) && values.length ? values.join(" · ") : "None identified";
    card.append(title, copy);
    return card;
  }

  function createCompactEmpty(copyText) {
    const empty = document.createElement("p");
    empty.className = "compact-empty";
    empty.textContent = copyText;
    return empty;
  }

  function renderVerification(verification, proposal) {
    const titles = new Map((proposal.recommendations || []).map((item) => [item.recommendation_id, item.title]));
    const statuses = ["supported", "partially_supported", "unsupported", "blocked_by_unknown"];
    elements.verificationSummary.replaceChildren();
    statuses.forEach((status) => {
      const count = (verification.verifications || []).filter((item) => item.status === status).length;
      const chip = document.createElement("span");
      chip.className = `verification-count ${status}`;
      chip.textContent = `${count} ${humanize(status)}`;
      elements.verificationSummary.append(chip);
    });
    elements.verificationList.replaceChildren();
    (verification.verifications || []).forEach((item) => {
      const row = document.createElement("article");
      row.className = `verification-item ${item.status}`;
      const header = document.createElement("header");
      const title = document.createElement("strong");
      title.textContent = titles.get(item.recommendation_id) || item.recommendation_id;
      const badge = document.createElement("span");
      badge.className = `status-badge ${item.status}`;
      badge.textContent = humanize(item.status);
      header.append(title, badge);
      const explanation = document.createElement("p");
      explanation.textContent = item.explanation || "";
      row.append(header, explanation);
      if (Array.isArray(item.missing_information) && item.missing_information.length) {
        const missing = document.createElement("p");
        missing.className = "verification-missing";
        missing.textContent = `Missing: ${item.missing_information.join(" · ")}`;
        row.append(missing);
      }
      elements.verificationList.append(row);
    });
    if (!(verification.verifications || []).length) elements.verificationList.append(createCompactEmpty("No recommendations required verification."));
  }

  function llmSentence(event, phase) {
    const model = formatModelName(event.model) || "The configured model";
    const operation = event.operation;
    if (phase === "running") {
      if (operation === "state_extraction") return `● Asking ${model} through the OpenAI Responses API to extract structured process knowledge…`;
      if (operation === "grounded_response") return `● Asking ${model} through the OpenAI Responses API to ground a response in retrieved evidence…`;
      return `● Sending the process context to ${model} through the OpenAI Responses API for a decision…`;
    }
    if (phase === "error") {
      if (operation === "state_extraction") return `✕ ${model} could not complete structured process extraction.`;
      if (operation === "grounded_response") return `✕ ${model} could not complete the grounded response.`;
      return `✕ ${model} could not complete the decision step.`;
    }
    if (operation === "state_extraction") return `✓ ${model} completed structured process extraction.`;
    if (operation === "grounded_response") return `✓ ${model} completed the response grounded in retrieved evidence.`;
    return `✓ ${model} completed the decision step.`;
  }

  function llmFields(event, status) {
    return [
      ["Technology", event.technology || "OpenAI Responses API"],
      ["Model", event.model],
      ["Stage", event.operation],
      ["Purpose", event.purpose],
      ["Status", status],
    ];
  }

  function orchestrationFields(event, status) {
    return [
      ["Model", event.model],
      ["Stage", event.stage],
      ["Structured output", event.structured_output],
      ["Referenced evidence IDs", event.referenced_evidence_ids],
      ["Findings", event.finding_count],
      ["Recommendations", event.recommendation_count],
      ["Retained human decisions", event.retained_human_decision_count],
      ["Verification statuses", event.status_counts],
      ["Snapshot counts", event.snapshot_counts],
      ["ProcessState fingerprint", event.process_state_fingerprint],
      ["Duration (ms)", event.duration_ms],
      ["Status", status],
    ];
  }

  function mcpFields(event, status) {
    return [
      ["Technology", event.technology || "Model Context Protocol"],
      ["Transport", event.transport],
      ["Server", event.server],
      ["Tool", event.tool],
      ["Arguments", event.arguments],
      ["Outcome", event.result_summary],
      ["Result metadata", event.result_metadata],
      ["Status", status],
    ];
  }

  function ragFields(event, status) {
    return [
      ["Technology", event.technology],
      ["Embedding model", event.embedding_model],
      ["Query", event.query],
      ["Index chunk count", event.index_chunk_count],
      ["Top K", event.top_k],
      ["Results (similarity scores)", event.results],
      ["Status", status],
    ];
  }

  function summarizePatch(patch = {}) {
    const changes = [];
    if (patch.process_name) changes.push(`named the process “${patch.process_name}”`);
    if (patch.process_name_to_replace?.new_value) changes.push(`refined the process name to “${patch.process_name_to_replace.new_value}”`);
    addPatchSummary(changes, patch.actors_to_add, "actor", "actors", true);
    addPatchSummary(changes, patch.systems_to_add, "system", "systems", true);
    addPatchSummary(changes, patch.steps_to_add, "step", "steps");
    addPatchSummary(changes, patch.decisions_to_add, "decision", "decisions");
    addPatchSummary(changes, patch.actors_to_replace, "actor refinement", "actor refinements");
    addPatchSummary(changes, patch.systems_to_replace, "system refinement", "system refinements");
    addPatchSummary(changes, patch.steps_to_replace, "step refinement", "step refinements");
    addPatchSummary(changes, patch.decisions_to_replace, "decision refinement", "decision refinements");
    addPatchSummary(changes, patch.pain_points_to_add, "pain point", "pain points");
    addPatchSummary(changes, patch.unknowns_to_add, "unresolved question", "unresolved questions");
    addPatchSummary(changes, patch.evidence_to_add, "evidence item", "evidence items");
    addPatchSummary(changes, patch.conflicts_to_add, "evidence conflict", "evidence conflicts");
    const resolved = Array.isArray(patch.unknowns_to_remove) ? patch.unknowns_to_remove.length : 0;
    if (resolved) changes.push(`resolved ${resolved} open ${resolved === 1 ? "question" : "questions"}`);
    if (changes.length === 0) return "validated the patch with no material field changes";
    if (changes.length <= 2) return changes.join(" and ");
    return `${changes.slice(0, 2).join(", ")}, and ${changes.length - 2} more ${changes.length - 2 === 1 ? "change" : "changes"}`;
  }

  function addPatchSummary(changes, values, singular, plural, nameSingle = false) {
    if (!Array.isArray(values) || values.length === 0) return;
    if (nameSingle && values.length === 1 && typeof values[0] === "string") {
      changes.push(`added ${values[0]} as ${articleFor(singular)} ${singular}`);
      return;
    }
    changes.push(`added ${values.length} ${values.length === 1 ? singular : plural}`);
  }

  function articleFor(noun) {
    return /^[aeiou]/i.test(noun) ? "an" : "a";
  }

  function createConversationIntro() {
    const intro = document.createElement("div");
    intro.className = "conversation-intro";
    intro.id = "conversation-intro";
    const orb = document.createElement("div");
    orb.className = "agent-orb";
    orb.setAttribute("aria-hidden", "true");
    orb.append(document.createElement("span"));
    const copy = document.createElement("div");
    const title = document.createElement("h3");
    const paragraph = document.createElement("p");
    const demo = document.createElement("button");
    title.textContent = "Interview the Northstar team";
    paragraph.textContent = "Describe a messy process as it really works. The agent will ask questions and the execution trace will show what happens underneath.";
    demo.className = "demo-prompt";
    demo.id = "demo-prompt";
    demo.type = "button";
    demo.textContent = `Try the demo: “${DEMO_PROMPT}”`;
    copy.append(title, paragraph, demo);
    intro.append(orb, copy);
    elements.conversationIntro = intro;
    return intro;
  }

  function createActivityEmpty() {
    const empty = document.createElement("div");
    empty.className = "activity-empty";
    empty.id = "activity-empty";
    const title = document.createElement("h3");
    const copy = document.createElement("p");
    title.textContent = "Waiting for the first backend event";
    copy.textContent = "The execution trace will update as events arrive.";
    empty.append(title, copy);
    elements.activityEmpty = empty;
    return empty;
  }

  function updateSendAvailability() {
    elements.sendButton.disabled = Boolean(state.source) || !elements.messageInput.value.trim();
  }

  function resizeComposer() {
    elements.messageInput.style.height = "auto";
    elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 120)}px`;
  }

  function setConnectionState(className, label) {
    elements.connectionState.className = `connection-state ${className}`;
    elements.connectionState.querySelector("span").textContent = label;
  }

  function incrementEventCount() {
    state.eventCount += 1;
    elements.eventCount.textContent = `${state.eventCount} ${state.eventCount === 1 ? "event" : "events"}`;
    if (state.currentTurn) {
      state.currentTurn.eventCount += 1;
      state.currentTurn.element.dataset.eventCount = String(state.currentTurn.eventCount);
    }
  }

  function scrollToEnd(element, onlyWhenFollowing = false) {
    if (onlyWhenFollowing && !isNearEnd(element)) return;
    element.scrollTo({ top: element.scrollHeight, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }

  function scrollMessageIntoView(message) {
    if (!message) return;
    const top = Math.max(0, message.offsetTop - 18);
    elements.conversation.scrollTo({ top, behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }

  function isNearEnd(element) {
    return element.scrollHeight - element.scrollTop - element.clientHeight < 72;
  }

  function prefersReducedMotion() {
    return Boolean(window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
  }

  function formatModelName(model) {
    if (!model) return "";
    const [family, version, ...variant] = String(model).split("-");
    if (family.toLowerCase() === "gpt" && version) {
      const variantName = variant.map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join(" ");
      return `GPT-${version}${variantName ? ` ${variantName}` : ""}`;
    }
    return String(model);
  }

  function humanize(value) {
    return String(value).replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
  }

  function displayValue(value) {
    return typeof value === "object" ? JSON.stringify(value, null, 2) : String(value);
  }

  function analysisFailureMessage(status) {
    if (status === 409) return "Discover more process knowledge before running analysis";
    if (status === 429) return "Public demo usage limit reached. Please try again later";
    if (status >= 500) return "Process analysis is temporarily unavailable";
    return "Process analysis could not be started";
  }

  async function requestFailureMessage(response, fallback) {
    try {
      const payload = await response.json();
      return typeof payload.detail === "string" && payload.detail.trim()
        ? payload.detail.trim()
        : fallback;
    } catch (error) {
      return fallback;
    }
  }

  function agentErrorMessage(event) {
    const messages = {
      mcp_discovery_unavailable: "The connected business tools are unavailable",
      mcp_tool_call_failure: "The business-system lookup could not be completed",
      rag_index_unavailable: "The company-knowledge index is unavailable",
      rag_index_stale: "The company-knowledge index needs to be rebuilt",
      llm_api_failure: "The model service could not complete this step",
      state_extraction_invalid: "Structured process extraction needs another turn",
    };
    return messages[event.code] || `The agent could not complete ${humanize(event.stage || "this step").toLowerCase()}`;
  }

  function evidenceSourceLabel(value) {
    if (value.source_type === "document") return value.document_title || "Company document";
    if (value.source_type === "user") return "User interview";
    if (value.source_type === "mcp") return "MCP/system result";
    return humanize(value.source_type || "Evidence");
  }

  function orderConflictSides(value) {
    const first = { claim: value.first_claim || "", source: value.first_source || "First source" };
    const second = { claim: value.second_claim || "", source: value.second_source || "Second source" };
    const firstDocumented = /policy|process|guideline|document/i.test(first.source);
    const secondReported = /user|interview|reported/i.test(second.source);
    if (firstDocumented || secondReported) return { reported: second, documented: first };
    return { reported: first, documented: second };
  }

  function createConflictSide(labelText, claimText, sourceText) {
    const side = document.createElement("div");
    side.className = "conflict-side";
    const label = document.createElement("span");
    const claim = document.createElement("p");
    const source = document.createElement("small");
    label.textContent = `${labelText}:`;
    claim.textContent = claimText;
    source.textContent = sourceText;
    side.append(label, claim, source);
    return side;
  }

  function renderSafeMarkdown(container, source) {
    if (!container) return;
    container.replaceChildren();
    const lines = String(source).replace(/\r\n?/g, "\n").split("\n");
    let index = 0;
    while (index < lines.length) {
      if (!lines[index].trim()) {
        index += 1;
        continue;
      }
      const listMatch = lines[index].match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/);
      if (listMatch) {
        const ordered = Boolean(listMatch[2]);
        const list = document.createElement(ordered ? "ol" : "ul");
        while (index < lines.length) {
          const itemMatch = lines[index].match(/^\s*(?:([-+*])|(\d+)[.)])\s+(.+)$/);
          if (!itemMatch || Boolean(itemMatch[2]) !== ordered) break;
          const item = document.createElement("li");
          appendInlineMarkdown(item, itemMatch[3]);
          list.append(item);
          index += 1;
        }
        container.append(list);
        continue;
      }
      const paragraphLines = [];
      while (index < lines.length && lines[index].trim() && !/^\s*(?:[-+*]|\d+[.)])\s+/.test(lines[index])) {
        paragraphLines.push(lines[index]);
        index += 1;
      }
      const paragraph = document.createElement("p");
      paragraphLines.forEach((line, lineIndex) => {
        if (lineIndex) paragraph.append(document.createElement("br"));
        appendInlineMarkdown(paragraph, line);
      });
      container.append(paragraph);
    }
  }

  function appendInlineMarkdown(container, source) {
    const tokenPattern = /(`[^`\n]+`|\*\*[^*\n]+?\*\*|\*[^*\n]+?\*)/g;
    let cursor = 0;
    for (const match of String(source).matchAll(tokenPattern)) {
      if (match.index > cursor) container.append(document.createTextNode(source.slice(cursor, match.index)));
      const token = match[0];
      const node = document.createElement(token.startsWith("`") ? "code" : token.startsWith("**") ? "strong" : "em");
      const edge = token.startsWith("**") ? 2 : 1;
      node.textContent = token.slice(edge, -edge);
      container.append(node);
      cursor = match.index + token.length;
    }
    if (cursor < source.length) container.append(document.createTextNode(source.slice(cursor)));
  }
})();
