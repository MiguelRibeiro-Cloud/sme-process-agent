import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import agent, app
from backend.agent import DiscoverySession
from backend.config import DEFAULT_OPENAI_MODEL, resolve_openai_model
from backend.mcp_client import MCPStartupError, MCPToolCallError
from backend.sessions import SessionStore
from backend.state import ProcessStatePatch


MCP_TOOLS = [
    {
        "name": "crm_get_customer",
        "description": "Look up a customer in the company's CRM by customer name.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_name": {"type": "string"}},
            "required": ["customer_name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "pricing_get_product",
        "description": "Look up product details and standard price by SKU.",
        "inputSchema": {
            "type": "object",
            "properties": {"sku": {"type": "string"}},
            "required": ["sku"],
            "additionalProperties": False,
        },
    },
]


def model_response(*, response_id="response-1", text="", output=None):
    return SimpleNamespace(id=response_id, output_text=text, output=output or [])


def parsed_response(patch_value=None):
    return SimpleNamespace(output_parsed=patch_value or ProcessStatePatch())


def function_call(*, name="crm_get_customer", arguments=None):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments or json.dumps({"customer_name": "Acme Ltd"}),
        call_id="call-123",
    )


def pricing_function_call(sku="NX-440"):
    return function_call(name="pricing_get_product", arguments=json.dumps({"sku": sku}))


def fake_openai(*, create_responses, patches=None):
    client = Mock()
    client.responses.create.side_effect = list(create_responses)
    client.responses.parse.side_effect = [
        parsed_response(value) for value in (patches or [ProcessStatePatch()])
    ]
    return client


def fake_mcp(result=None):
    client = Mock()
    client.discover_tools.return_value = MCP_TOOLS
    client.call_tool.return_value = result
    return client


class AgentEventStreamTests(unittest.TestCase):
    def run_turn(self, message, openai_client, mcp_client=None, session=None):
        mcp_client = mcp_client or fake_mcp()
        session = session or DiscoverySession(mcp_client)
        return list(session.event_stream(message, openai_client)), session, mcp_client

    def test_no_tool_path_runs_discovery_and_structured_extraction(self):
        openai_client = fake_openai(
            create_responses=[model_response(text="Which approval step takes the most time?")]
        )
        events, _, mcp_client = self.run_turn("Quote preparation is slow", openai_client)
        self.assertEqual(
            [event["type"] for event in events],
            ["llm_call_started", "llm_call_completed", "assistant_message",
             "llm_call_started", "llm_call_completed", "interaction_completed"],
        )
        mcp_client.call_tool.assert_not_called()
        self.assertIs(openai_client.responses.parse.call_args.kwargs["text_format"], ProcessStatePatch)

    def test_configured_model_is_used_for_discovery_and_extraction(self):
        openai_client = fake_openai(
            create_responses=[model_response(text="What happens next?")]
        )
        session = DiscoverySession(fake_mcp(), model="test-configured-model")

        events = list(session.event_stream("Quotes are slow", openai_client))

        self.assertEqual(
            openai_client.responses.create.call_args.kwargs["model"],
            "test-configured-model",
        )
        self.assertEqual(
            openai_client.responses.parse.call_args.kwargs["model"],
            "test-configured-model",
        )
        llm_events = [event for event in events if event["type"].startswith("llm_call_")]
        self.assertTrue(llm_events)
        self.assertTrue(all(event["model"] == "test-configured-model" for event in llm_events))

    def test_configured_model_is_used_for_grounded_response(self):
        openai_client = fake_openai(create_responses=[
            model_response(output=[function_call()]),
            model_response(response_id="response-2", text="Acme Ltd is owned by Sarah Chen."),
        ])
        session = DiscoverySession(
            fake_mcp({"account_owner": "Sarah Chen"}),
            model="test-grounded-model",
        )

        list(session.event_stream("Who owns Acme Ltd?", openai_client))

        models = [call.kwargs["model"] for call in openai_client.responses.create.call_args_list]
        self.assertEqual(models, ["test-grounded-model", "test-grounded-model"])
        self.assertEqual(openai_client.responses.parse.call_args.kwargs["model"], "test-grounded-model")

    def test_mcp_discovery_failure_is_safe_and_skips_the_llm(self):
        mcp_client = fake_mcp()
        mcp_client.discover_tools.side_effect = MCPStartupError("private startup detail")
        openai_client = fake_openai(create_responses=[])
        session = DiscoverySession(mcp_client)
        before = session.state_snapshot()

        events, _, _ = self.run_turn("Who owns Acme Ltd?", openai_client, mcp_client, session)

        self.assertEqual(session.state_snapshot(), before)
        self.assertEqual(next(e for e in events if e["type"] == "agent_error")["code"], "mcp_discovery_unavailable")
        self.assertNotIn("private startup detail", json.dumps(events))
        openai_client.responses.create.assert_not_called()

    def test_discovered_mcp_schemas_are_supplied_to_luna(self):
        openai_client = fake_openai(create_responses=[model_response(text="Tell me more")])
        self.run_turn("The process is slow", openai_client)
        tools = openai_client.responses.create.call_args.kwargs["tools"]
        self.assertEqual(
            [tool["name"] for tool in tools],
            ["crm_get_customer", "pricing_get_product", "search_company_knowledge"],
        )
        self.assertEqual(tools[0]["description"], MCP_TOOLS[0]["description"])
        self.assertEqual(tools[0]["parameters"], MCP_TOOLS[0]["inputSchema"])

    def test_crm_execution_crosses_mcp_client_boundary(self):
        result = {"customer_id": "C-1042", "name": "Acme Ltd", "account_owner": "Sarah Chen"}
        mcp_client = fake_mcp(result)
        openai_client = fake_openai(create_responses=[
            model_response(output=[function_call()]),
            model_response(response_id="response-2", text="Acme Ltd is owned by Sarah Chen."),
        ])
        events, _, _ = self.run_turn("Who owns Acme Ltd?", openai_client, mcp_client)
        mcp_client.call_tool.assert_called_once_with("crm_get_customer", {"customer_name": "Acme Ltd"})
        started = next(event for event in events if event["type"] == "mcp_tool_call_started")
        self.assertEqual(started["server"], "Northstar Business Systems")
        self.assertEqual(started["transport"], "stdio")
        extraction = json.loads(openai_client.responses.parse.call_args.kwargs["input"])
        self.assertEqual(extraction["actual_tool_calls"][0]["result"]["account_owner"], "Sarah Chen")

    def test_pricing_execution_crosses_mcp_client_boundary(self):
        result = {"sku": "NX-440", "name": "Industrial Sensor", "standard_price": 1250.0, "currency": "EUR"}
        mcp_client = fake_mcp(result)
        openai_client = fake_openai(create_responses=[
            model_response(output=[pricing_function_call()]),
            model_response(response_id="response-2", text="The price is EUR 1,250.00."),
        ])
        events, _, _ = self.run_turn("What is the standard price of NX-440?", openai_client, mcp_client)
        mcp_client.call_tool.assert_called_once_with("pricing_get_product", {"sku": "NX-440"})
        completed = next(event for event in events if event["type"] == "mcp_tool_call_completed")
        self.assertEqual(completed["result_metadata"]["record_identifier"], "NX-440")

    def test_missing_result_remains_grounded(self):
        mcp_client = fake_mcp(None)
        openai_client = fake_openai(create_responses=[
            model_response(output=[pricing_function_call("UNKNOWN-1")]),
            model_response(response_id="response-2", text="I couldn't find UNKNOWN-1."),
        ])
        events, _, _ = self.run_turn("What does UNKNOWN-1 cost?", openai_client, mcp_client)
        grounded_input = openai_client.responses.create.call_args_list[1].kwargs["input"]
        self.assertEqual(grounded_input[0]["output"], "null")
        completion = next(event for event in events if event["type"] == "mcp_tool_call_completed")
        self.assertFalse(completion["result_metadata"]["record_found"])
        self.assertNotIn("agent_error", [event["type"] for event in events])

    def test_mcp_failure_is_safe_and_preserves_state(self):
        mcp_client = fake_mcp()
        session = DiscoverySession(mcp_client)
        session._state = agent.ProcessState(actors=["Sales"])
        before = session.state_snapshot()
        mcp_client.call_tool.side_effect = MCPToolCallError("private transport detail")
        openai_client = fake_openai(create_responses=[model_response(output=[function_call()])])
        events, _, _ = self.run_turn("Who owns Acme Ltd?", openai_client, mcp_client, session)
        self.assertEqual(session.state_snapshot(), before)
        self.assertEqual(next(e for e in events if e["type"] == "agent_error")["code"], "mcp_tool_call_failure")
        self.assertNotIn("private transport detail", json.dumps(events))
        openai_client.responses.parse.assert_not_called()

    def test_valid_change_updates_state(self):
        patch_value = ProcessStatePatch(
            process_name="Customer quote preparation",
            actors_to_add=["Sales"],
            pain_points_to_add=["Quote preparation takes too long"],
        )
        openai_client = fake_openai(
            create_responses=[model_response(text="Where does the delay occur?")],
            patches=[patch_value],
        )
        events, _, _ = self.run_turn("Sales takes too long to build quotes", openai_client)
        update = next(event for event in events if event["type"] == "process_state_updated")
        self.assertEqual(update["state"]["process_name"], "Customer quote preparation")
        self.assertEqual(update["patch"]["actors_to_add"], ["Sales"])

    def test_context_and_process_state_ground_follow_up_fragment(self):
        mcp_client = fake_mcp()
        session = DiscoverySession(mcp_client)
        openai_client = fake_openai(
            create_responses=[model_response(text="Where do approvals happen?"),
                              model_response(response_id="response-2", text="What requires Marta's approval?")],
            patches=[ProcessStatePatch(process_name="Customer quote preparation", actors_to_add=["Sales"]),
                     ProcessStatePatch(actors_to_add=["Marta"])],
        )
        self.run_turn("Our sales team prepares customer quotes", openai_client, mcp_client, session)
        self.run_turn("Asking Marta for approvals", openai_client, mcp_client, session)
        second_call = openai_client.responses.create.call_args_list[1].kwargs
        self.assertEqual(second_call["input"][-1]["content"], "Asking Marta for approvals")
        self.assertIn("Customer quote preparation", second_call["instructions"])

    def test_malformed_tool_arguments_are_not_executed(self):
        mcp_client = fake_mcp()
        openai_client = fake_openai(create_responses=[model_response(output=[function_call(arguments="not-json")])])
        events, _, _ = self.run_turn("Find a customer", openai_client, mcp_client)
        self.assertEqual(next(e for e in events if e["type"] == "agent_error")["code"], "malformed_tool_arguments")
        mcp_client.call_tool.assert_not_called()

    def test_unknown_tool_is_not_executed(self):
        mcp_client = fake_mcp()
        openai_client = fake_openai(create_responses=[model_response(output=[function_call(name="unknown")])])
        events, _, _ = self.run_turn("Use another tool", openai_client, mcp_client)
        self.assertEqual(next(e for e in events if e["type"] == "agent_error")["code"], "mcp_tool_unavailable")
        mcp_client.call_tool.assert_not_called()

    def test_malformed_extraction_preserves_existing_state(self):
        mcp_client = fake_mcp()
        session = DiscoverySession(mcp_client)
        first = fake_openai(create_responses=[model_response(text="What happens next?")],
                            patches=[ProcessStatePatch(actors_to_add=["Sales"])])
        self.run_turn("Sales starts", first, mcp_client, session)
        before = session.state_snapshot()
        bad = Mock()
        bad.responses.create.return_value = model_response(text="Thanks")
        bad.responses.parse.return_value = SimpleNamespace(output_parsed={"forbidden": True})
        events, _, _ = self.run_turn("Marta helps", bad, mcp_client, session)
        self.assertEqual(session.state_snapshot(), before)
        self.assertEqual(next(e for e in events if e["type"] == "agent_error")["code"], "state_extraction_invalid")

    def test_openai_failure_emits_safe_terminal_error(self):
        openai_client = Mock()
        openai_client.responses.create.side_effect = RuntimeError("private API detail")
        events, _, _ = self.run_turn("Hello", openai_client)
        self.assertEqual(events[-1], {"type": "interaction_completed", "status": "error"})
        completion = next(event for event in events if event["type"] == "llm_call_completed")
        self.assertEqual(completion["status"], "error")
        self.assertEqual(completion["operation"], "discovery")
        self.assertEqual(next(e for e in events if e["type"] == "agent_error")["code"], "llm_api_failure")
        self.assertNotIn("private API detail", json.dumps(events))


class ConversationWindowAndResetTests(unittest.TestCase):
    def test_recent_window_keeps_only_six_messages(self):
        session = DiscoverySession(fake_mcp(), recent_message_limit=6)
        for number in range(4):
            session.recent_conversation.append_turn(f"user-{number}", f"assistant-{number}")
        messages = session.recent_conversation.snapshot()
        self.assertEqual(len(messages), 6)
        self.assertEqual(messages[0].content, "user-1")
        self.assertEqual(messages[-1].content, "assistant-3")

    def test_reset_clears_state_and_recent_conversation(self):
        session = DiscoverySession(fake_mcp())
        session._state = agent.ProcessState(actors=["Sales"])
        session.recent_conversation.append_turn("one", "two")
        reset_state = session.reset()
        self.assertEqual(reset_state, agent.ProcessState())
        self.assertEqual(len(session.recent_conversation), 0)

    def test_reset_endpoint_resets_only_resolved_session(self):
        session_id = "a" * 43
        store = SessionStore(id_factory=lambda: session_id)
        context = store.get_or_create(None).context
        context.recent_messages.append_turn("one", "two")
        client = TestClient(app.app)
        client.cookies.set(app.SESSION_COOKIE_NAME, session_id)
        with patch.object(app, "session_store", store):
            response = client.post("/reset-discovery")
        self.assertEqual(response.json()["status"], "reset")
        self.assertEqual(len(context.recent_messages), 0)


class RuntimeConfigurationTests(unittest.TestCase):
    def test_model_default_is_sensible_when_environment_value_is_absent_or_blank(self):
        self.assertEqual(resolve_openai_model({}), DEFAULT_OPENAI_MODEL)
        self.assertEqual(resolve_openai_model({"OPENAI_MODEL": "   "}), DEFAULT_OPENAI_MODEL)

    def test_model_value_is_trimmed(self):
        self.assertEqual(
            resolve_openai_model({"OPENAI_MODEL": "  configured-model  "}),
            "configured-model",
        )

    def test_runtime_status_exposes_only_the_model(self):
        with patch.object(app, "OPENAI_MODEL", "safe-test-model"):
            result = app.get_runtime_status()
        self.assertEqual(result, {"model": "safe-test-model"})
        self.assertNotIn("key", json.dumps(result).lower())


class StructuralInvariantTests(unittest.TestCase):
    def test_agent_does_not_import_business_tool_implementations(self):
        source = Path(agent.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        self.assertNotIn("backend.mcp_server.business_tools", imported_modules)
        self.assertNotIn("backend.mcp_server.server", imported_modules)


if __name__ == "__main__":
    unittest.main()
