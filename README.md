# SME Process Discovery Agent

An evidence-aware AI system that interviews users about business processes, consults operational systems and internal documentation, builds an explicit AS-IS process model, and produces a verified automation proposal.

The repository is a portfolio project. Northstar Industrial Services ("Northstar") and all CRM, pricing, policy, and procedure data are fictional and synthetic; no customer or employer data is included.

## Why this exists

Business processes are usually distributed across people's knowledge, undocumented workarounds, business systems, written procedures, and sometimes conflicting policy and practice. Automating the first description offered by a user can encode the wrong process. This system investigates and preserves uncertainty before it recommends automation.

It is not merely a chatbot: the application owns typed process state, executes model-selected capabilities through controlled boundaries, records evidence provenance, and verifies recommendations against the discovered record.

## Architecture

```text
Browser
  |
  | HTTP + server-sent events
  | opaque HttpOnly session cookie
  v
FastAPI application
  |
  +-- bounded SessionStore
  |      `-- per visitor: ProcessState + recent context + analysis
  |
  +-- Discovery loop -------- configured GPT model
  |      |
  |      +-- MCP client
  |      |     `-- Northstar Business Systems MCP server
  |      |            +-- synthetic CRM
  |      |            `-- synthetic pricing catalogue
  |      |
  |      +-- application-owned RAG
  |      |     +-- embedding model
  |      |     `-- local vector index
  |      |            `-- synthetic company documents
  |      |
  |      `-- validated ProcessState patches
  |
  `-- Explicit orchestration
         ProcessState snapshot
           -> Process Analyst
           -> Automation Designer
           -> Evidence Verifier
```

The model does not speak MCP directly. The application discovers MCP tool definitions, translates them into model tool schemas, lets the model select a capability, validates the requested call, and executes it through the MCP client. RAG remains a separate application-owned capability.

Visitor process discovery is isolated in anonymous, cookie-associated `SessionContext` objects. Model configuration, MCP, RAG, runtime metadata, the knowledge catalogue, and saved evaluation reports remain shared application infrastructure.

See [docs/architecture.md](docs/architecture.md) for the request lifecycle and trust boundaries.

## Architectural principles

### Application-owned state

The model does not own durable process knowledge. Structured extraction proposes a typed `ProcessStatePatch`; Python validates and applies additions, exact replacements, refinements, and unknown resolution. Evidence history is append-only, and `ProcessState.flow` is the canonical branch-aware process representation.

### Evidence-aware discovery

Claims retain their source: user-reported practice, documented policy or procedure, or an MCP system fact. Unsupported inferences stay qualified or become unknowns. Materially incompatible claims are both preserved and linked by an unresolved conflict.

### MCP and RAG solve different problems

- MCP exposes operational systems and executable capabilities, here a synthetic CRM and pricing catalogue.
- RAG performs semantic search over unstructured company knowledge, here synthetic policies, procedures, and guidance.

Keeping the boundaries separate makes the source and meaning of evidence explicit.

### Explicit orchestration

The application controls `ProcessState -> Analyst -> Designer -> Verifier`. Each specialist receives a deep-copied snapshot and typed prior-stage output. Specialists cannot spawn one another, reorder the sequence, or start an autonomous multi-agent conversation. Python validates cross-references and applies deterministic provenance guards.

### Observable by design

The frontend activity trace is built from real backend events streamed over SSE. It does not display invented model "thinking." Completed typed artifacts and capability events are inspectable without exposing prompts, credentials, or embedding vectors.

### Behavioral evaluation

Deterministic tests validate software mechanics and safety boundaries. Scenario-based evaluations exercise nondeterministic AI behavior and report whether the observed result meets stable criteria. A single eval run is a regression signal, not statistical proof or a correctness guarantee.

## Why provenance matters

```text
Employee:
"We only ask our Sales Manager for approval above 15%."

Company policy retrieved through RAG:
the documented approval threshold differs from the reported practice.

Result:
both claims retain their provenance and ProcessState records an unresolved conflict.
```

The system does not silently rewrite practice to match policy or treat a document as proof of what employees actually do.

## What to look for in the demo

- Conversational discovery backed by typed structured state.
- Branch-aware AS-IS process modeling and explicit unknowns.
- Scope-bound discovery that redirects non-process requests without retrieving data or mutating state.
- MCP tool discovery, schema translation, validation, and execution.
- Semantic RAG retrieval with document and chunk provenance.
- Preserved policy/practice conflicts.
- Application-controlled Analyst -> Designer -> Verifier orchestration.
- Verifier challenges and deterministic provenance guards.
- Live execution telemetry and read-only offline eval results.

## Project structure

```text
backend/
  app.py                 FastAPI routes, SSE endpoints, and static UI mount
  agent.py               discovery loop and structured extraction
  sessions.py            anonymous session store, limits, and execution guards
  state.py               typed ProcessState and reconciliation
  mcp_client.py          reusable MCP client boundary
  mcp_server/            synthetic business-system MCP server
  rag/                   chunking, embeddings, index, and retrieval
  orchestration/         Analyst -> Designer -> Verifier workflow
  synthetic_company/     fictional CRM, pricing, and company documents
frontend/                build-free HTML, CSS, and JavaScript UI
evals/                   scenarios, evaluators, runner, and report models
deployment/              reviewed release artifacts packaged into the image
tests/                   deterministic unit, integration, and browser checks
docs/                    architecture and lifecycle documentation
```

## Local setup (Windows PowerShell)

Python 3.11 or newer is required. From a source checkout:

```powershell
git clone <repository-url>
cd sme-process-agent

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .

Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The remaining variables have safe defaults:

| Variable | Requirement | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required for model calls | Discovery, orchestration, evals, and explicit index builds |
| `OPENAI_MODEL` | Optional | Product model; defaults to `gpt-6-luna` |
| `OPENAI_EMBEDDING_MODEL` | Optional | RAG embedding model; defaults to `text-embedding-3-small` |
| `OPENAI_EVAL_MODEL` | Optional, eval only | Semantic judge model; blank uses `OPENAI_MODEL` |
| `RAG_INDEX_PATH` | Optional | Runtime index path; defaults locally to `.rag/northstar-index.json` |
| `EVALUATION_REPORT_PATH` | Optional | Published report path; defaults locally to `.evals/latest.json` |
| `SESSION_COOKIE_NAME` | Optional | Opaque anonymous-session cookie name |
| `SESSION_COOKIE_SECURE` | Optional | `true` for HTTPS deployments; defaults to `false` for local HTTP |
| `SESSION_IDLE_TTL_SECONDS` | Optional | Idle expiry; defaults to 7200 seconds |
| `SESSION_MAX_COUNT` | Optional | In-process session cap; defaults to 100 |
| `PUBLIC_DEMO_CHAT_LIMIT` / `PUBLIC_DEMO_CHAT_WINDOW_SECONDS` | Optional | Discovery rate limit; defaults to 20 per 600 seconds per session |
| `PUBLIC_DEMO_ANALYSIS_LIMIT` / `PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS` | Optional | Analysis rate limit; defaults to 3 per 600 seconds per session |
| `PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION` | Optional | Session lifetime discovery cap; defaults to 40 |
| `PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION` | Optional | Session lifetime analysis cap; defaults to 5 |
| `PUBLIC_DEMO_MAX_MODEL_OPERATIONS` | Optional | Process-lifetime emergency fuse for accepted paid workflows; blank disables it |

Build the local RAG index from the synthetic Markdown documents:

```powershell
python -m backend.rag.index
```

This explicit command makes a paid embeddings API request. Application startup never builds the index or spends embedding tokens silently. If the index is missing or stale, RAG reports that state and retrieval remains unavailable until the command is run again.

Start the application:

```powershell
uvicorn backend.app:app --reload
```

Open <http://127.0.0.1:8000/>.

## Tests

Run the deterministic suite without live OpenAI calls:

```powershell
python -m unittest discover -s tests -v
```

The suite covers typed state reconciliation, branching, MCP integration, RAG compatibility and retrieval mechanics, safe error behavior, orchestration guards, evaluation infrastructure, and the frontend execution trace. The browser smoke test skips only when Chrome or Chromium is unavailable.

No formatter, linter, or static type checker is currently configured. Python byte-compilation is the lightweight static syntax check used for this milestone.

## AI behavior evaluations

Preview the selected scenarios without making model, embedding, MCP, or semantic-judge calls:

```powershell
python -m evals.runner --dry-run
```

Run one scenario or the full suite:

```powershell
python -m evals.runner --scenario crm-routing
python -m evals.runner
```

Live runs consume OpenAI API calls. `--deterministic-only` skips semantic-judge calls but still runs the product model. Development reports are generated under ignored `.evals/`; the frontend can only display the configured saved report and cannot trigger an eval run. The release image includes a deliberately reviewed `deployment/evaluation-report.json` snapshot from one release-candidate run, including any failures. It is a regression signal, not a statistical reliability guarantee.

## RAG index publication strategy

Synthetic source documents are committed. Normal developer output remains ignored at `.rag/northstar-index.json`; deployment uses the intentionally committed, publication-safe `deployment/northstar-index.json`. Build the local index explicitly, then validate and export it without another API call:

```powershell
python -m backend.rag.index
python -m backend.rag.index --export-release
```

The export command checks index version, embedding model, and source-document fingerprints before writing the release artifact. The release JSON contains only embeddings and metadata derived from the fictional documents. At runtime, the same compatibility checks run before any query embedding is requested. A missing or stale release index is reported as unavailable and is never rebuilt during application startup.

Local evaluation output similarly remains ignored under `.evals/`. The reviewed release-candidate snapshot is copied byte-for-byte to `deployment/evaluation-report.json` and selected with `EVALUATION_REPORT_PATH`; no report is generated or copied as part of deployment startup.

## Railway deployment

Railway runs this repository as one persistent Docker service. The root `Dockerfile` uses Python 3.11 slim, starts Uvicorn on Railway's injected `PORT`, and explicitly fixes the worker count at one. One worker and one replica are intentional because anonymous sessions, request limits, and the emergency fuse are process-local. Horizontal scaling or multiple workers require moving session state and related counters to shared persistence such as Redis or database storage.

1. Create a Railway service from the GitHub repository; Railway detects the root `Dockerfile`.
2. Add the required secret `OPENAI_API_KEY`. Set model overrides only if desired.
3. Set `SESSION_COOKIE_SECURE=true` for Railway HTTPS. Keep the cookie `HttpOnly`, `SameSite=Lax`, and `Path=/` behavior unchanged.
4. Choose a deliberate finite `PUBLIC_DEMO_MAX_MODEL_OPERATIONS` value and review the existing per-session rate and lifetime limits. Do not treat the fuse as monetary accounting.
5. Keep exactly one replica. Do not override the image command or add workers.
6. The image defaults to `RAG_INDEX_PATH=/app/deployment/northstar-index.json` and `EVALUATION_REPORT_PATH=/app/deployment/evaluation-report.json`; these may also be set explicitly in Railway.
7. `railway.toml` configures `/health`, a 30-second healthcheck timeout, and restart-on-failure. Keep serverless/sleep behavior off initially.
8. After deployment, verify `/health`, `/`, `/runtime-status`, `/mcp-status`, `/rag-status`, `/knowledge-catalog`, and `/evaluation-report`, then generate a Railway domain. A custom domain can be added later.

The coarse IP limiter deliberately uses only the direct peer address and does not trust arbitrary forwarded headers. Session limits remain the primary application protection; verify the peer address observed behind Railway after deployment before considering narrowly scoped trusted-proxy configuration.

This keeps the repository small and reproducible and makes the only setup-time API cost explicit.

## MCP inspection

The synthetic business-system server can be inspected independently with MCP Inspector:

```powershell
mcp dev backend/mcp_server/server.py
```

It advertises `crm_get_customer` and `pricing_get_product`. Inspector use is optional and is not part of the normal startup path.

## Public API surface

The browser uses the following demo endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Cheap process liveness; no sessions or external calls |
| `GET /chat-stream` | SSE discovery conversation |
| `GET /process-state` | Current session's validated in-memory state |
| `POST /reset-discovery` | Clear only the current session's discovery and analysis |
| `POST /analyze-process` | SSE Analyst -> Designer -> Verifier run |
| `GET /analysis` | Current session's latest derived orchestration result |
| `GET /runtime-status` | Safe model identifier only |
| `GET /mcp-status` | Safe server and discovered-tool metadata |
| `GET /rag-status` | Safe index compatibility metadata |
| `GET /knowledge-catalog` | Titles and topics, never vectors or document text |
| `GET /evaluation-report` | Latest local report, when present; never starts evals |

The session-aware routes are `/chat-stream`, `/process-state`, `/reset-discovery`, `/analyze-process`, and `/analysis`. The health, status, catalogue, runtime, and evaluation-report routes expose shared read-only metadata and do not create anonymous sessions.

## Security and trust boundaries

- `.env`, local virtual environments, local `.rag` indexes, and `.evals` reports are excluded from the container context. The reviewed deployment index is committed separately by design.
- All Northstar data is explicitly fictional and synthetic.
- Model- and user-controlled strings are rendered with DOM text nodes; raw HTML is not trusted.
- Browser-facing failures use bounded messages rather than tracebacks, request objects, prompts, filesystem paths, or credentials.
- MCP tool names and arguments are validated before execution.
- The MCP subprocess receives an allowlist of OS/runtime variables, not application credentials or model configuration.
- Retrieved documents are treated as evidence, not executable instructions, and their provenance remains attached.
- Cookies contain only a random `secrets.token_urlsafe(32)` identifier and use `HttpOnly`, `SameSite=Lax`, `Path=/`, plus configurable `Secure` behavior.
- Unknown, forged, and expired identifiers receive a fresh empty context without revealing another session's existence.
- Idle expiry, LRU capacity eviction, short-window throttles, per-session lifetime caps, concurrent-operation guards, and an optional global fuse bound public-demo memory and paid-request exposure.

These are small-demo controls, not enterprise DDoS protection, account authentication, billing, or exact token/cost accounting. A real production deployment may still require access control, edge rate limiting, operational logging, and deployment-specific security headers.

## Limitations

- The company, records, policies, and procedures are synthetic.
- Anonymous discovery sessions remain in one server process's memory, expire after inactivity, and are lost on restart. There is no account/login persistence.
- One-process deployment is assumed. Multiple workers or horizontal replicas require shared session/rate-limit storage such as Redis or a database.
- The local JSON vector index is intentionally lightweight, not a production vector database.
- The process-flow model represents ordered steps and explicit branches but is not BPMN and does not execute workflows or model arbitrary parallel gateways.
- Recommendations require human review; verification reduces unsupported claims but cannot guarantee correctness.
- Behavioral evals are regression signals and remain sensitive to model variability.
- Authentication, distributed sessions, CAPTCHA, and infrastructure-level traffic protection are not implemented in this milestone.

## License

No software license has been specified. Viewing the public repository does not by itself grant reuse rights.
