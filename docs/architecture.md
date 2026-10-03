# Architecture

This document describes the runtime lifecycle of the SME Process Discovery Agent. Northstar Industrial Services and all associated data are fictional.

## Deployment boundary

```text
Internet
  -> Railway HTTPS
  -> one persistent container
  -> one Uvicorn process with exactly one worker
       +-- FastAPI and static frontend
       +-- bounded in-memory SessionStore
       +-- one shared MCP client and stdio child process
       `-- one shared prebuilt RAG index
```

The one-worker, one-replica boundary is deliberate: anonymous session state, usage counters, concurrency guards, and the process-lifetime fuse live in process memory. A second worker or replica would create an independent state island and inconsistent enforcement. Horizontal scaling therefore requires shared persistence such as Redis or database storage before the process count changes.

Railway terminates public HTTPS and injects `PORT`; the container binds Uvicorn to `0.0.0.0:$PORT`. The deployment healthcheck calls the dependency-free `/health` route. `SESSION_COOKIE_SECURE=true` is required in the Railway environment.

## Request lifecycle

The browser loads static HTML, CSS, and JavaScript from FastAPI. Status endpoints expose presentation-safe metadata. Discovery and analysis operations stream server-sent events so the interface reflects actual backend work rather than a simulated activity sequence.

Each browser receives a high-entropy opaque identifier in an `HttpOnly`, `SameSite=Lax` cookie. The cookie is only a bearer identifier; process facts, conversation text, model settings, credentials, and other user data never enter it. `Secure` is controlled explicitly by `SESSION_COOKIE_SECURE` so local HTTP development works and HTTPS deployments can require secure transport.

```text
Browser
  |
  | opaque HttpOnly session cookie
  v
SessionStore (bounded, process memory)
  +-- anonymous session A
  |     +-- ProcessState
  |     +-- recent conversation context
  |     +-- latest orchestration result
  |     `-- rate/budget counters and execution guards
  `-- anonymous session B
        +-- ProcessState
        +-- recent conversation context
        `-- latest orchestration result

Shared application infrastructure
  +-- configured model client and workflow engines
  +-- one MCP client/session and discovered tool metadata
  +-- RAG service and local index
  +-- runtime and knowledge metadata
  `-- static evaluation report
```

`SessionStore` expires idle sessions lazily, refreshes last access on session-aware use, and caps retained sessions. At capacity it removes expired sessions first and then the least-recently-used idle session. Active streaming requests are not expired; an active session is only an eviction candidate in the exceptional case where every retained session is active and the hard capacity must still be maintained. A restart loses every anonymous session.

The trust boundary is server-side: possession of the random cookie associates a browser with one context, while arbitrary malformed, unknown, or expired values create a fresh isolated context without revealing whether any other identifier exists. This is anonymous isolation, not login-based identity or durable account storage.

The current design assumes one application process. Multiple workers or horizontal replicas would need a shared session and rate-limit backend such as Redis or a database; process memory is not distributed.

## Discovery loop

For each user message, the application:

1. Resolves the visitor's `SessionContext`, then takes a deep copy of that session's `ProcessState` and bounded recent conversation.
2. Reads the MCP tools discovered during application startup.
3. Adds the application-owned company-knowledge search schema.
4. Calls the configured model with the conversation, state context, and capability schemas.
5. Validates and executes at most one selected capability.
6. Calls the model again with the capability result when grounding is required.
7. Runs typed structured extraction over the completed interaction.
8. Validates and atomically applies the proposed `ProcessStatePatch`.
9. Emits state and completion events over SSE.

Failures produce safe terminal events and leave validated state unchanged where mutation did not complete.

## MCP translation boundary

FastAPI starts one reusable stdio MCP client session. Startup performs `tools/list` once and caches the synthetic server's definitions. The application converts each MCP input schema into a Responses API function-tool schema; the model sees capability metadata, not an MCP transport. Calls through the shared SDK session are protected by one narrow asynchronous execution lock because transport-level concurrent request safety is not assumed. Model work, RAG, and unrelated request handling remain concurrent.

When the model requests a tool, Python checks that the tool was discovered, parses the JSON arguments, validates them against the MCP input schema, and invokes the tool through the client session. Tool failures are mapped to bounded public events. The subprocess receives only an OS/runtime environment allowlist.

The MCP server reads deterministic JSON records under `backend/synthetic_company/`. It has no OpenAI dependency and receives no OpenAI credential.

## RAG retrieval

RAG is application-owned rather than part of the MCP server:

1. `python -m backend.rag.index` chunks the synthetic Markdown documents.
2. The configured embedding model embeds those chunks.
3. The generated local index stores document fingerprints, metadata, chunk text, and vectors under ignored `.rag/`.
4. An explicit offline export validates that local index and writes the reviewed release artifact to `deployment/northstar-index.json`; the container never creates embeddings during build or startup.
5. Before retrieval, runtime code validates index version, source fingerprints, and embedding-model compatibility.
6. Only then is the user query embedded and compared with indexed vectors using cosine similarity.

The model receives retrieved text with document title, filename, and chunk ID. Retrieval output is evidence, not an instruction channel. A missing or stale index is explicit and does not trigger an automatic paid rebuild.

## ProcessState and reconciliation

`ProcessState` is the application's durable knowledge boundary. It contains actors, systems, an explicit process flow, pain points, evidence, conflicts, and unknowns. The flow contains typed steps with stable IDs and typed decisions whose conditions point to ordered branch-specific step IDs.

The model proposes a `ProcessStatePatch`; it cannot replace the complete state. Python validates exact replacement targets, stable-ID refinements, branch references, evidence fields, conflict shape, and unknown resolution before constructing a new state. If validation fails, the previous state survives unchanged.

Evidence provenance is explicit:

- `user`: reported real-world practice;
- `document`: retrieved policy, procedure, or guidance;
- `mcp`: structured fact returned by an operational capability.

Document evidence does not prove employee behavior, and user evidence does not become policy. When supported claims conflict, both remain present and an unresolved `EvidenceConflict` records the disagreement.

## Explicit orchestration

Analysis is a fixed application-controlled sequence:

```text
frozen ProcessState snapshot
  -> Process Analyst: typed findings
  -> Automation Designer: typed recommendations
  -> Evidence Verifier: typed verdict for every recommendation
```

The specialists do not call MCP or RAG. Every stage receives a deep copy, and Python validates finding, recommendation, step, and evidence references before continuing. A deterministic provenance guard prevents a recommendation from receiving a supported verdict when its wording relabels the cited source. The completed result and its source-state fingerprint are stored only in the requesting `SessionContext`; later discovery in that same session marks it stale without affecting any other session.

## Public-demo request controls

Discovery and orchestration have independent per-session sliding-window limits, lifetime caps, and non-blocking execution guards. One session can run at most one discovery turn and one orchestration run of each kind at a time; duplicate work is rejected rather than queued. Reset clears only process-derived state and deliberately retains usage counters, so reset is not a limit bypass.

A coarser, bounded throttle uses the direct peer address when available. It does not consume forwarded headers and is not treated as identity. An optional process-lifetime operation fuse can stop further paid request workflows after a configured number of accepted operations. These controls bound a small demo's exposure; they are not enterprise DDoS protection or monetary accounting.

Session-aware routes are `/chat-stream`, `/process-state`, `/reset-discovery`, `/analyze-process`, and `/analysis`. Shared metadata routes are `/runtime-status`, `/mcp-status`, `/rag-status`, `/knowledge-catalog`, and `/evaluation-report`; they neither read visitor state nor create sessions.

## Observability

Discovery emits model-call, tool-request, MCP, RAG, state-update, error, and completion events. Orchestration emits stage start, stage completion, structured output, failure, and final-result events. The frontend groups those events by turn and renders their supplied fields; it does not invent hidden reasoning or expose chain-of-thought.

## Evaluation architecture

`tests/` uses mocks and deterministic local boundaries to validate software behavior without live model calls. The MCP integration tests start the local synthetic stdio server only.

`evals/` defines human-authored scenarios and records backend events plus typed artifacts. Objective criteria use deterministic evaluators. Criteria that require semantic comparison use a constrained structured model judge. Reports label evaluator type, retain per-run outcomes, and are written to ignored `.evals/` only after an explicit command.

The application exposes the report selected by `EVALUATION_REPORT_PATH` read-only. Local development defaults to `.evals/latest.json`; a reviewed release report may later be packaged as `deployment/evaluation-report.json`. Public visitors cannot initiate evaluations, and startup never generates a report.
