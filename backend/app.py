import json
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from threading import Lock
from typing import Annotated, Any, Literal

from fastapi import FastAPI, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from backend.agent import DiscoverySession
from backend.config import (
    EVALUATION_REPORT_PATH,
    OPENAI_EMBEDDING_MODEL,
    OPENAI_MODEL,
    PROJECT_ROOT,
    PUBLIC_DEMO_ANALYSIS_LIMIT,
    PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS,
    PUBLIC_DEMO_CHAT_LIMIT,
    PUBLIC_DEMO_CHAT_WINDOW_SECONDS,
    PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION,
    PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION,
    PUBLIC_DEMO_MAX_MODEL_OPERATIONS,
    RAG_INDEX_PATH,
    SESSION_COOKIE_NAME,
    SESSION_COOKIE_SECURE,
    SESSION_IDLE_TTL_SECONDS,
    SESSION_MAX_COUNT,
)
from backend.mcp_client import MCPClientError, NorthstarMCPClient
from backend.orchestration.service import OrchestrationService
from backend.orchestration.shared import has_process_knowledge
from backend.rag.embeddings import OpenAIEmbeddingProvider
from backend.rag.service import RAGService
from backend.sessions import (
    GlobalOperationBudget,
    IpRateLimiter,
    OperationAdmission,
    SessionContext,
    SessionResolution,
    SessionStore,
)
from evals.report import load_latest_report


FRONTEND_DIR = PROJECT_ROOT / "frontend"

mcp_client = NorthstarMCPClient()
rag_service = RAGService(
    OpenAIEmbeddingProvider(OPENAI_EMBEDDING_MODEL), index_path=RAG_INDEX_PATH
)
discovery_engine = DiscoverySession(
    mcp_client,
    model=OPENAI_MODEL,
    rag_service=rag_service,
    create_default_context=False,
)
orchestration_service = OrchestrationService(
    model=OPENAI_MODEL,
    retain_default_result=False,
)
session_store = SessionStore(
    idle_ttl_seconds=SESSION_IDLE_TTL_SECONDS,
    max_sessions=SESSION_MAX_COUNT,
)
operation_admission = OperationAdmission(
    chat_limit=PUBLIC_DEMO_CHAT_LIMIT,
    chat_window_seconds=PUBLIC_DEMO_CHAT_WINDOW_SECONDS,
    analysis_limit=PUBLIC_DEMO_ANALYSIS_LIMIT,
    analysis_window_seconds=PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS,
    max_chat_turns=PUBLIC_DEMO_MAX_CHAT_TURNS_PER_SESSION,
    max_analysis_runs=PUBLIC_DEMO_MAX_ANALYSIS_RUNS_PER_SESSION,
)
# Direct peer address only: forwarded headers are deployment-specific and deliberately ignored.
ip_rate_limiter = IpRateLimiter(
    chat_limit=PUBLIC_DEMO_CHAT_LIMIT * 5,
    chat_window_seconds=PUBLIC_DEMO_CHAT_WINDOW_SECONDS,
    analysis_limit=PUBLIC_DEMO_ANALYSIS_LIMIT * 5,
    analysis_window_seconds=PUBLIC_DEMO_ANALYSIS_WINDOW_SECONDS,
    max_keys=SESSION_MAX_COUNT * 2,
)
global_operation_budget = GlobalOperationBudget(PUBLIC_DEMO_MAX_MODEL_OPERATIONS)
client: OpenAI | None = None
client_lock = Lock()
MAX_MESSAGE_LENGTH = 4_000


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    try:
        mcp_client.start()
    except MCPClientError:
        # Keep health/status endpoints available so the UI can report the real failure.
        pass
    yield
    mcp_client.stop()


app = FastAPI(
    title="SME Process Discovery Agent",
    description=(
        "Evidence-aware process discovery over synthetic Northstar Industrial "
        "Services data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def get_health():
    """Return process liveness without touching sessions or external dependencies."""
    return {"status": "ok"}


@app.get("/process-state")
def get_process_state(request: Request, response: Response):
    session = resolve_session(request)
    attach_session_cookie(response, session)
    return session.context.state_snapshot()


@app.get("/mcp-status")
def get_mcp_status():
    """Report the capabilities cached by the real startup tools/list call."""
    return mcp_client.status()


@app.get("/runtime-status")
def get_runtime_status():
    """Expose only safe runtime metadata required by the browser UI."""
    return {"model": OPENAI_MODEL}


@app.get("/rag-status")
def get_rag_status():
    """Expose safe compatibility metadata without building or embedding anything."""
    return rag_service.status()


@app.get("/knowledge-catalog")
def get_knowledge_catalog():
    """Expose only safe metadata about the indexed synthetic document corpus."""
    try:
        documents = rag_service.knowledge_catalog()
    except Exception:
        # Keep index paths, parsing details, and other internals out of the browser.
        return {"available": False, "documents": []}
    return {"available": True, "documents": documents}


@app.get("/evaluation-report")
def get_evaluation_report():
    """Return the latest developer-generated eval report without running evals."""
    report = load_latest_report(PROJECT_ROOT, report_path=EVALUATION_REPORT_PATH)
    if report is None:
        return {"status": "not_available", "report": None}
    return {"status": "available", "report": report}


@app.post("/reset-discovery")
def reset_discovery(request: Request, response: Response):
    session = resolve_session(request)
    attach_session_cookie(response, session)
    if not session.context.discovery_guard.acquire(blocking=False):
        return public_error_response(
            409,
            "Wait for the active discovery or analysis request to finish before resetting.",
            session,
        )
    if not session.context.analysis_guard.acquire(blocking=False):
        session.context.discovery_guard.release()
        return public_error_response(
            409,
            "Wait for the active discovery or analysis request to finish before resetting.",
            session,
        )
    try:
        state = session.context.reset()
    finally:
        session.context.analysis_guard.release()
        session.context.discovery_guard.release()
    return {"status": "reset", "state": state, "analysis_cleared": True}


@app.get("/analysis")
def get_analysis(request: Request, response: Response):
    session = resolve_session(request)
    attach_session_cookie(response, session)
    snapshot = session.context.state_snapshot()
    result = orchestration_service.latest(snapshot, session.context)
    if result is None:
        return {"status": "not_available", "result": None, "stale": False}
    return {"status": "complete", "result": result, "stale": result.stale}


@app.post("/analyze-process")
def analyze_process_endpoint(request: Request):
    session = resolve_session(request)
    snapshot = session.context.state_snapshot()
    if not has_process_knowledge(snapshot):
        return public_error_response(
            409, "Discover process knowledge before running analysis.", session
        )
    rejection = admit_paid_operation(request, session.context, "analysis")
    if rejection is not None:
        return public_error_response(rejection[0], rejection[1], session)
    stream = StreamingResponse(
        orchestration_event_stream(session.context, snapshot, session.session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    attach_session_cookie(stream, session)
    return stream


@app.get("/chat-stream")
def chat_stream(
    request: Request,
    message: Annotated[str, Query(min_length=1, max_length=MAX_MESSAGE_LENGTH)],
):
    session = resolve_session(request)
    normalized_message = message.strip()
    if not normalized_message:
        stream: Iterator[str] = error_events(
            code="empty_message",
            stage="request_validation",
            message="The user message was empty.",
            assistant_message="Please enter a message before sending.",
        )
    else:
        rejection = admit_paid_operation(request, session.context, "chat")
        if rejection is not None:
            return public_error_response(rejection[0], rejection[1], session)
        stream = agent_event_stream(
            session.context, normalized_message, session.session_id
        )
    response = StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
    attach_session_cookie(response, session)
    return response


def agent_event_stream(
    session: SessionContext,
    message: str,
    session_id: str | None = None,
) -> Iterator[str]:
    try:
        for event in discovery_engine.event_stream(
            message, get_openai_client(), session
        ):
            yield format_event(event)
    finally:
        if session_id is not None:
            session_store.touch(session_id, session)
        operation_admission.release(session, "chat")


def orchestration_event_stream(
    session: SessionContext,
    snapshot,
    session_id: str | None = None,
) -> Iterator[str]:
    try:
        for event in orchestration_service.event_stream(
            snapshot, get_openai_client(), session
        ):
            yield format_event(event)
    finally:
        if session_id is not None:
            session_store.touch(session_id, session)
        operation_admission.release(session, "analysis")


def resolve_session(request: Request) -> SessionResolution:
    return session_store.get_or_create(request.cookies.get(SESSION_COOKIE_NAME))


def attach_session_cookie(response: Response, session: SessionResolution) -> None:
    if not session.created:
        return
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session.session_id,
        httponly=True,
        secure=SESSION_COOKIE_SECURE,
        samesite="lax",
        path="/",
    )


def public_error_response(
    status_code: int,
    detail: str,
    session: SessionResolution,
) -> JSONResponse:
    response = JSONResponse(status_code=status_code, content={"detail": detail})
    attach_session_cookie(response, session)
    return response


def admit_paid_operation(
    request: Request,
    session: SessionContext,
    operation: Literal["chat", "analysis"],
) -> tuple[int, str] | None:
    operation_name = operation
    client_address = request.client.host if request.client else None
    if not ip_rate_limiter.allow(client_address, operation_name):
        return 429, "Public demo request rate limit reached. Please try again later."
    rejection = operation_admission.acquire(session, operation_name)
    if rejection == "already_running":
        if operation_name == "analysis":
            return 409, "An analysis is already running for this session."
        return 409, "A discovery request is already running for this session."
    if rejection == "rate_limited":
        return 429, "Public demo request rate limit reached. Please try again later."
    if rejection == "session_budget_exhausted":
        return 429, "Public demo usage limit reached for this session."
    if not global_operation_budget.consume():
        operation_admission.release(session, operation_name)
        return 429, "The public demo's temporary usage limit has been reached."
    return None


def get_openai_client() -> OpenAI:
    global client
    if client is None:
        with client_lock:
            if client is None:
                client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


def error_events(
    *, code: str, stage: str, message: str, assistant_message: str
) -> Iterator[str]:
    yield format_event(
        {"type": "agent_error", "code": code, "stage": stage, "message": message}
    )
    yield format_event(
        {"type": "assistant_message", "content": assistant_message, "error": True}
    )
    yield format_event({"type": "interaction_completed", "status": "error"})


def format_event(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


# Keep this last so API routes win before the root static mount.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
