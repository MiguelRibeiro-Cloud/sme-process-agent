import asyncio
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from mcp import types

from backend import app
from backend.agent import DiscoverySession
from backend.mcp_client import NorthstarMCPClient
from backend.orchestration.models import (
    AutomationProposal,
    OrchestrationResult,
    ProcessAnalysis,
    VerifiedProposal,
)
from backend.orchestration.shared import process_state_fingerprint
from backend.state import ProcessState, ProcessStatePatch
from backend.sessions import (
    GlobalOperationBudget,
    IpRateLimiter,
    OperationAdmission,
    SessionContext,
    SessionStore,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def id_factory(*characters):
    values = iter(character * 43 for character in characters)
    return lambda: next(values)


def result_for(state, label):
    return OrchestrationResult(
        process_state_fingerprint=process_state_fingerprint(state),
        model="test-model",
        created_at=datetime.now(UTC),
        analysis=ProcessAnalysis(summary=f"{label} analysis"),
        proposal=AutomationProposal(objective=f"{label} objective"),
        verification=VerifiedProposal(),
    )


class SessionStoreTests(unittest.TestCase):
    def test_access_refreshes_idle_ttl_and_expiry_replaces_with_empty_session(self):
        clock = FakeClock()
        store = SessionStore(
            idle_ttl_seconds=10,
            max_sessions=3,
            clock=clock,
            id_factory=id_factory("a", "b"),
        )
        first = store.get_or_create(None)
        first.context.process_state = ProcessState(actors=["Marta"])
        clock.advance(6)
        self.assertIs(store.get(first.session_id), first.context)
        clock.advance(6)
        self.assertEqual(store.cleanup_expired(), 0)
        clock.advance(4)
        replacement = store.get_or_create(first.session_id)
        self.assertTrue(replacement.created)
        self.assertNotEqual(replacement.session_id, first.session_id)
        self.assertEqual(replacement.context.state_snapshot(), ProcessState())

    def test_capacity_evicts_least_recently_used_idle_session(self):
        clock = FakeClock()
        store = SessionStore(
            idle_ttl_seconds=100,
            max_sessions=2,
            clock=clock,
            id_factory=id_factory("a", "b", "c"),
        )
        first = store.get_or_create(None)
        clock.advance(1)
        second = store.get_or_create(None)
        clock.advance(1)
        store.get(first.session_id)
        clock.advance(1)
        third = store.get_or_create(None)
        self.assertEqual(len(store), 2)
        self.assertIsNotNone(store.get(first.session_id))
        self.assertIsNone(store.get(second.session_id))
        self.assertIsNotNone(store.get(third.session_id))

    def test_cleanup_does_not_remove_active_request(self):
        clock = FakeClock()
        store = SessionStore(
            idle_ttl_seconds=5,
            max_sessions=2,
            clock=clock,
            id_factory=id_factory("a"),
        )
        session = store.get_or_create(None)
        session.context.begin_request()
        clock.advance(20)
        self.assertEqual(store.cleanup_expired(), 0)
        self.assertIs(store.get(session.session_id), session.context)
        session.context.end_request()

    def test_forged_or_nonexistent_identifier_never_resolves_another_session(self):
        store = SessionStore(id_factory=id_factory("a", "b"))
        first = store.get_or_create(None)
        first.context.process_state = ProcessState(systems=["QuoteX"])
        forged = store.get_or_create("z" * 43)
        self.assertNotEqual(forged.session_id, first.session_id)
        self.assertEqual(forged.context.state_snapshot(), ProcessState())


class CookieAndRouteIsolationTests(unittest.TestCase):
    def test_health_is_cheap_stateless_and_dependency_free(self):
        store = Mock(spec=SessionStore)
        rag = Mock()
        mcp = Mock()
        client = TestClient(app.app)
        with (
            patch.object(app, "session_store", store),
            patch.object(app, "rag_service", rag),
            patch.object(app, "mcp_client", mcp),
            patch.object(
                app,
                "get_openai_client",
                side_effect=AssertionError("health must not create an OpenAI client"),
            ),
        ):
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertNotIn("set-cookie", response.headers)
        store.get_or_create.assert_not_called()
        self.assertEqual(rag.mock_calls, [])
        self.assertEqual(mcp.mock_calls, [])

    def test_cookie_is_opaque_httponly_lax_and_secure_follows_configuration(self):
        store = SessionStore(id_factory=id_factory("a"))
        client = TestClient(app.app)
        with patch.object(app, "session_store", store):
            response = client.get("/process-state")
        cookie = response.headers["set-cookie"]
        value = cookie.split("=", 1)[1].split(";", 1)[0]
        self.assertEqual(len(value), 43)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=lax", cookie)
        self.assertIn("Path=/", cookie)
        self.assertNotIn("Secure", cookie)
        self.assertNotIn("Marta", cookie)
        self.assertNotIn("QuoteX", cookie)

        secure_store = SessionStore(id_factory=id_factory("b"))
        secure_client = TestClient(app.app)
        with (
            patch.object(app, "session_store", secure_store),
            patch.object(app, "SESSION_COOKIE_SECURE", True),
        ):
            secure_response = secure_client.get("/process-state")
        self.assertIn("Secure", secure_response.headers["set-cookie"])

    def test_process_state_analysis_and_reset_are_hard_isolated(self):
        store = SessionStore(id_factory=id_factory("a", "b"))
        first_client = TestClient(app.app)
        second_client = TestClient(app.app)
        with patch.object(app, "session_store", store):
            first_client.get("/process-state")
            second_client.get("/process-state")
            first = store.get(first_client.cookies.get(app.SESSION_COOKIE_NAME))
            second = store.get(second_client.cookies.get(app.SESSION_COOKIE_NAME))
            first.process_state = ProcessState(actors=["Marta"], systems=["QuoteX"])
            second.process_state = ProcessState(actors=["Eric"])
            first.orchestration_result = result_for(first.process_state, "Marta")
            second.orchestration_result = result_for(second.process_state, "Eric")

            self.assertEqual(first_client.get("/process-state").json()["actors"], ["Marta"])
            self.assertEqual(second_client.get("/process-state").json()["actors"], ["Eric"])
            self.assertEqual(
                first_client.get("/analysis").json()["result"]["analysis"]["summary"],
                "Marta analysis",
            )
            self.assertEqual(
                second_client.get("/analysis").json()["result"]["analysis"]["summary"],
                "Eric analysis",
            )

            first.process_state = ProcessState(actors=["Marta", "Sales"])
            self.assertTrue(first_client.get("/analysis").json()["stale"])
            self.assertFalse(second_client.get("/analysis").json()["stale"])
            self.assertEqual(first_client.post("/reset-discovery").status_code, 200)
            self.assertEqual(first_client.get("/process-state").json(), ProcessState().model_dump(mode="json"))
            self.assertEqual(second_client.get("/process-state").json()["actors"], ["Eric"])
            self.assertEqual(second_client.get("/analysis").json()["status"], "complete")

    def test_expired_cookie_is_replaced_and_returns_empty_state(self):
        clock = FakeClock()
        store = SessionStore(
            idle_ttl_seconds=10,
            max_sessions=2,
            clock=clock,
            id_factory=id_factory("a", "b"),
        )
        client = TestClient(app.app)
        with patch.object(app, "session_store", store):
            client.get("/process-state")
            original_id = client.cookies.get(app.SESSION_COOKIE_NAME)
            store.get(original_id).process_state = ProcessState(actors=["Marta"])
            clock.advance(11)
            response = client.get("/process-state")
        self.assertEqual(response.json(), ProcessState().model_dump(mode="json"))
        self.assertNotEqual(client.cookies.get(app.SESSION_COOKIE_NAME), original_id)

    def test_shared_metadata_routes_do_not_create_sessions(self):
        store = Mock(spec=SessionStore)
        client = TestClient(app.app)
        with patch.object(app, "session_store", store):
            for path in (
                "/health",
                "/runtime-status",
                "/mcp-status",
                "/rag-status",
                "/knowledge-catalog",
                "/evaluation-report",
            ):
                self.assertEqual(client.get(path).status_code, 200)
        store.get_or_create.assert_not_called()


class ConversationIsolationTests(unittest.TestCase):
    def test_recent_context_and_extraction_state_belong_only_to_supplied_context(self):
        mcp = Mock()
        mcp.discover_tools.return_value = []
        engine = DiscoverySession(mcp, create_default_context=False)
        first = SessionContext()
        second = SessionContext()
        first.recent_messages.append_turn("Secret from A", "A response")

        client = Mock()
        client.responses.create.return_value = SimpleNamespace(
            id="response-b", output_text="What happens next?", output=[]
        )
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=ProcessStatePatch(systems_to_add=["SystemB"])
        )
        list(engine.event_stream("Message from B", client, second))

        discovery_input = client.responses.create.call_args.kwargs["input"]
        self.assertEqual(discovery_input, [{"role": "user", "content": "Message from B"}])
        self.assertNotIn("Secret from A", str(discovery_input))
        self.assertEqual(first.state_snapshot(), ProcessState())
        self.assertEqual(second.state_snapshot().systems, ["SystemB"])


class RateBudgetAndGuardTests(unittest.TestCase):
    def build_admission(self, clock, **overrides):
        values = dict(
            chat_limit=2,
            chat_window_seconds=10,
            analysis_limit=1,
            analysis_window_seconds=10,
            max_chat_turns=4,
            max_analysis_runs=2,
            clock=clock,
        )
        values.update(overrides)
        return OperationAdmission(**values)

    def test_rate_limit_window_and_per_session_separation(self):
        clock = FakeClock()
        admission = self.build_admission(clock)
        first, second = SessionContext(), SessionContext()
        for _ in range(2):
            self.assertIsNone(admission.acquire(first, "chat"))
            admission.release(first, "chat")
        self.assertEqual(admission.acquire(first, "chat"), "rate_limited")
        self.assertIsNone(admission.acquire(second, "chat"))
        admission.release(second, "chat")
        clock.advance(11)
        self.assertIsNone(admission.acquire(first, "chat"))
        admission.release(first, "chat")

    def test_lifetime_budget_and_duplicate_guards(self):
        clock = FakeClock()
        admission = self.build_admission(clock, max_chat_turns=2)
        context = SessionContext()
        self.assertIsNone(admission.acquire(context, "chat"))
        self.assertEqual(admission.acquire(context, "chat"), "already_running")
        admission.release(context, "chat")
        self.assertIsNone(admission.acquire(context, "chat"))
        admission.release(context, "chat")
        clock.advance(11)
        self.assertEqual(admission.acquire(context, "chat"), "session_budget_exhausted")

    def test_guards_release_after_success_and_exception(self):
        context = SessionContext()
        self.assertIsNone(app.operation_admission.acquire(context, "chat"))
        with (
            patch.object(app.discovery_engine, "event_stream", return_value=iter([{"type": "ok"}])),
            patch.object(app, "get_openai_client", return_value=Mock()),
        ):
            list(app.agent_event_stream(context, "hello"))
        self.assertFalse(context.discovery_guard.locked())

        self.assertIsNone(app.operation_admission.acquire(context, "analysis"))
        with (
            patch.object(app.orchestration_service, "event_stream", side_effect=RuntimeError("boom")),
            patch.object(app, "get_openai_client", return_value=Mock()),
        ):
            with self.assertRaises(RuntimeError):
                list(app.orchestration_event_stream(context, ProcessState()))
        self.assertFalse(context.analysis_guard.locked())

    def test_ip_throttle_and_optional_global_fuse_are_deterministic(self):
        clock = FakeClock()
        limiter = IpRateLimiter(
            chat_limit=2,
            chat_window_seconds=10,
            analysis_limit=1,
            analysis_window_seconds=10,
            max_keys=4,
            clock=clock,
        )
        self.assertTrue(limiter.allow("127.0.0.1", "chat"))
        self.assertTrue(limiter.allow("127.0.0.1", "chat"))
        self.assertFalse(limiter.allow("127.0.0.1", "chat"))
        clock.advance(11)
        self.assertTrue(limiter.allow("127.0.0.1", "chat"))
        fuse = GlobalOperationBudget(2)
        self.assertTrue(fuse.consume())
        self.assertTrue(fuse.consume())
        self.assertFalse(fuse.consume())

    def test_paid_route_returns_429_then_recovers_without_blocking_other_session(self):
        clock = FakeClock()
        store = SessionStore(id_factory=id_factory("a", "b"))
        admission = self.build_admission(clock, chat_limit=1, max_chat_turns=10)
        ip_limiter = IpRateLimiter(
            chat_limit=100,
            chat_window_seconds=10,
            analysis_limit=100,
            analysis_window_seconds=10,
            max_keys=4,
            clock=clock,
        )
        first = TestClient(app.app)
        second = TestClient(app.app)
        events = iter([{"type": "interaction_completed", "status": "success"}])
        with (
            patch.object(app, "session_store", store),
            patch.object(app, "operation_admission", admission),
            patch.object(app, "ip_rate_limiter", ip_limiter),
            patch.object(app, "global_operation_budget", GlobalOperationBudget(None)),
            patch.object(app.discovery_engine, "event_stream", side_effect=lambda *args: iter([{"type": "interaction_completed", "status": "success"}])),
            patch.object(app, "get_openai_client", return_value=Mock()),
        ):
            self.assertEqual(first.get("/chat-stream", params={"message": "one"}).status_code, 200)
            limited = first.get("/chat-stream", params={"message": "two"})
            self.assertEqual(limited.status_code, 429)
            self.assertIn("rate limit", limited.json()["detail"].lower())
            self.assertEqual(second.get("/chat-stream", params={"message": "other"}).status_code, 200)
            clock.advance(11)
            self.assertEqual(first.get("/chat-stream", params={"message": "three"}).status_code, 200)

    def test_paid_routes_return_bounded_duplicate_errors(self):
        clock = FakeClock()
        store = SessionStore(id_factory=id_factory("a"))
        admission = self.build_admission(clock, chat_limit=10, analysis_limit=10)
        ip_limiter = IpRateLimiter(
            chat_limit=100,
            chat_window_seconds=10,
            analysis_limit=100,
            analysis_window_seconds=10,
            max_keys=4,
            clock=clock,
        )
        client = TestClient(app.app)
        with (
            patch.object(app, "session_store", store),
            patch.object(app, "operation_admission", admission),
            patch.object(app, "ip_rate_limiter", ip_limiter),
        ):
            client.get("/process-state")
            context = store.get(client.cookies.get(app.SESSION_COOKIE_NAME))
            context.process_state = ProcessState(pain_points=["Slow quotes"])
            self.assertIsNone(admission.acquire(context, "chat"))
            duplicate_chat = client.get("/chat-stream", params={"message": "duplicate"})
            self.assertEqual(duplicate_chat.status_code, 409)
            self.assertEqual(
                duplicate_chat.json()["detail"],
                "A discovery request is already running for this session.",
            )
            admission.release(context, "chat")

            self.assertIsNone(admission.acquire(context, "analysis"))
            duplicate_analysis = client.post("/analyze-process")
            self.assertEqual(duplicate_analysis.status_code, 409)
            self.assertEqual(
                duplicate_analysis.json()["detail"],
                "An analysis is already running for this session.",
            )
            admission.release(context, "analysis")


class SharedMcpConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_shared_sdk_session_serializes_only_tool_call_execution(self):
        active = 0
        maximum_active = 0

        class FakeSdkSession:
            async def call_tool(self, name, arguments):
                nonlocal active, maximum_active
                active += 1
                maximum_active = max(maximum_active, active)
                await asyncio.sleep(0)
                active -= 1
                return types.CallToolResult(
                    content=[], structuredContent={"result": {"tool": name}}
                )

        client = NorthstarMCPClient()
        client._session = FakeSdkSession()
        client._tool_call_lock = asyncio.Lock()
        results = await asyncio.gather(
            client._call_tool("first", {}),
            client._call_tool("second", {}),
        )
        self.assertEqual(maximum_active, 1)
        self.assertEqual(results, [{"tool": "first"}, {"tool": "second"}])


if __name__ == "__main__":
    unittest.main()
