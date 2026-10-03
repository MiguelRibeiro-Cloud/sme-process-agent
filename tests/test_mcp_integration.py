import sys
import unittest

from backend.agent import mcp_tools_to_openai
from backend.mcp_client import (
    PROJECT_ROOT,
    NorthstarMCPClient,
    mcp_server_parameters,
    mcp_subprocess_environment,
)


class RealMCPBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = NorthstarMCPClient()
        cls.client.start()

    @classmethod
    def tearDownClass(cls):
        cls.client.stop()

    def test_tools_list_discovers_only_business_capabilities(self):
        tools = self.client.discover_tools()
        self.assertEqual(
            [tool["name"] for tool in tools],
            ["crm_get_customer", "pricing_get_product"],
        )
        self.assertNotIn("add", [tool["name"] for tool in tools])
        for tool in tools:
            self.assertEqual(tool["inputSchema"]["type"], "object")
            self.assertTrue(tool["description"])
            parameter = next(iter(tool["inputSchema"]["properties"].values()))
            self.assertTrue(parameter["description"])

    def test_schema_conversion_preserves_mcp_metadata(self):
        mcp_tools = self.client.discover_tools()
        openai_tools = mcp_tools_to_openai(mcp_tools)
        self.assertEqual(openai_tools[0]["name"], mcp_tools[0]["name"])
        self.assertEqual(openai_tools[0]["description"], mcp_tools[0]["description"])
        self.assertEqual(openai_tools[0]["parameters"], mcp_tools[0]["inputSchema"])

    def test_status_exposes_descriptions_from_discovered_tool_metadata(self):
        discovered = self.client.discover_tools()
        status = self.client.status()

        self.assertEqual(
            status["tools"],
            [
                {"name": tool["name"], "description": tool["description"]}
                for tool in discovered
            ],
        )
        self.assertNotIn("inputSchema", status["tools"][0])

    def test_real_crm_and_pricing_calls_use_stdio_session(self):
        customer = self.client.call_tool("crm_get_customer", {"customer_name": "Acme Ltd"})
        product = self.client.call_tool("pricing_get_product", {"sku": "NX-440"})
        self.assertEqual(customer["account_owner"], "Sarah Chen")
        self.assertEqual(product["standard_price"], 1250.0)

    def test_real_missing_result_is_none(self):
        result = self.client.call_tool("pricing_get_product", {"sku": "UNKNOWN-1"})
        self.assertIsNone(result)


class MCPSubprocessEnvironmentTests(unittest.TestCase):
    def test_server_invocation_uses_current_interpreter_module_and_absolute_cwd(self):
        parameters = mcp_server_parameters({"PATH": "runtime-path"})

        self.assertEqual(parameters.command, sys.executable)
        self.assertEqual(parameters.args, ["-m", "backend.mcp_server.server"])
        self.assertEqual(parameters.cwd, PROJECT_ROOT)
        self.assertTrue(parameters.cwd.is_absolute())
        self.assertEqual(parameters.env, {"PATH": "runtime-path"})

    def test_subprocess_receives_runtime_allowlist_without_application_secrets(self):
        environment = {
            "PATH": "runtime-path",
            "SystemRoot": "windows-root",
            "TEMP": "temporary-directory",
            "OPENAI_API_KEY": "not-for-the-subprocess",
            "DATABASE_URL": "not-for-the-subprocess",
            "UNRELATED_SETTING": "not-for-the-subprocess",
        }

        result = mcp_subprocess_environment(environment)

        self.assertEqual(
            result,
            {
                "PATH": "runtime-path",
                "SystemRoot": "windows-root",
                "TEMP": "temporary-directory",
            },
        )


if __name__ == "__main__":
    unittest.main()
