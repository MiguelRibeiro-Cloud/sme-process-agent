"""Application-level boundary around the MCP Python SDK's stdio client."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
from collections.abc import Mapping
from concurrent.futures import Future
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


SERVER_NAME = "Northstar Business Systems"
TRANSPORT = "stdio"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MCP_ENVIRONMENT_KEYS = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "NO_COLOR",
        "PATH",
        "PATHEXT",
        "PYTHONIOENCODING",
        "PYTHONUNBUFFERED",
        "PYTHONUTF8",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)


def mcp_subprocess_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return only OS/runtime variables needed by the local synthetic MCP server."""
    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if key.upper() in MCP_ENVIRONMENT_KEYS
    }


def mcp_server_parameters(
    environment: Mapping[str, str] | None = None,
) -> StdioServerParameters:
    """Build a shell-independent invocation for the bundled MCP server."""
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "backend.mcp_server.server"],
        cwd=PROJECT_ROOT,
        env=mcp_subprocess_environment(environment),
    )


class MCPClientError(RuntimeError):
    """Base class for safe MCP boundary failures."""


class MCPStartupError(MCPClientError):
    """The local server process or MCP session could not be initialized."""


class MCPToolUnavailableError(MCPClientError):
    """The requested tool was not present in the discovered tool set."""


class MCPToolCallError(MCPClientError):
    """The MCP server reported a tool execution error."""


class MCPMalformedResultError(MCPClientError):
    """The MCP result could not be converted into an application value."""


class NorthstarMCPClient:
    """Own one reusable MCP stdio subprocess/session on a background event loop."""

    def __init__(self, *, startup_timeout: float = 10.0, call_timeout: float = 15.0):
        self.startup_timeout = startup_timeout
        self.call_timeout = call_timeout
        self._state_lock = Lock()
        self._ready = Event()
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_signal: asyncio.Event | None = None
        self._session: ClientSession | None = None
        self._tool_call_lock: asyncio.Lock | None = None
        self._tools: list[dict[str, Any]] = []
        self._tool_names: set[str] = set()
        self._startup_error: str | None = None
        self._discovered_at: str | None = None

    def start(self) -> None:
        """Start the server, initialize MCP, and perform one real tools/list call."""
        with self._state_lock:
            if self._thread and self._thread.is_alive() and self._session is not None:
                return
            self._ready.clear()
            self._startup_error = None
            self._tools = []
            self._tool_names = set()
            self._discovered_at = None
            self._thread = Thread(
                target=self._thread_main,
                name="northstar-mcp-client",
                daemon=True,
            )
            self._thread.start()

        if not self._ready.wait(self.startup_timeout):
            self.stop()
            raise MCPStartupError("MCP initialization timed out.")
        if self._startup_error:
            raise MCPStartupError("MCP initialization failed.")

    def stop(self) -> None:
        """Close the MCP session and terminate its stdio subprocess cleanly."""
        with self._state_lock:
            loop = self._loop
            stop_signal = self._stop_signal
            thread = self._thread
        if loop and stop_signal and loop.is_running():
            loop.call_soon_threadsafe(stop_signal.set)
        if thread and thread.is_alive():
            thread.join(timeout=self.startup_timeout)
        with self._state_lock:
            self._thread = None
            self._loop = None
            self._stop_signal = None
            self._session = None
            self._tool_call_lock = None

    def discover_tools(self) -> list[dict[str, Any]]:
        """Return the capabilities cached from the initialization tools/list call."""
        if not self._ready.is_set() or self._startup_error or self._session is None:
            raise MCPStartupError("MCP tools are unavailable because initialization failed.")
        return copy.deepcopy(self._tools)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Invoke a discovered tool across MCP and return its structured value."""
        if name not in self._tool_names:
            raise MCPToolUnavailableError(f"MCP tool '{name}' is not available.")
        future = self._submit(self._call_tool(name, arguments))
        try:
            return future.result(timeout=self.call_timeout)
        except TimeoutError as exc:
            future.cancel()
            raise MCPToolCallError("MCP tool call timed out.") from exc
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPToolCallError("MCP tool call failed.") from exc

    def status(self) -> dict[str, Any]:
        connected = self._ready.is_set() and not self._startup_error and self._session is not None
        return {
            "connected": connected,
            "server": SERVER_NAME,
            "transport": TRANSPORT,
            "tools_discovered": len(self._tools),
            # These safe definitions come directly from the MCP tools/list response.
            # Input schemas stay private here because the status UI only needs names
            # and descriptions to explain the discovered capabilities.
            "tools": [
                {"name": tool["name"], "description": tool["description"]}
                for tool in self._tools
            ],
            "discovered_at": self._discovered_at,
            "error": None if connected else self._startup_error,
        }

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._session_main())
        except BaseException:
            self._startup_error = "The local MCP server session stopped unexpectedly."
            self._ready.set()

    async def _session_main(self) -> None:
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._stop_signal = asyncio.Event()
        self._tool_call_lock = asyncio.Lock()
        # The current interpreter plus module invocation works on Windows and Linux.
        # The child receives only the runtime allowlist, never application secrets.
        parameters = mcp_server_parameters()
        try:
            with open(os.devnull, "w", encoding="utf-8") as error_log:
                async with stdio_client(parameters, errlog=error_log) as streams:
                    async with ClientSession(*streams) as session:
                        self._session = session
                        await session.initialize()
                        tools = await self._list_all_tools(session)
                        self._tools = tools
                        self._tool_names = {tool["name"] for tool in tools}
                        self._discovered_at = datetime.now(UTC).isoformat()
                        self._ready.set()
                        await self._stop_signal.wait()
        except BaseException:
            if not self._ready.is_set():
                self._startup_error = "The local MCP server could not be initialized."
                self._ready.set()
            else:
                self._startup_error = "The local MCP server session was interrupted."
        finally:
            self._session = None
            self._tool_call_lock = None

    async def _list_all_tools(self, session: ClientSession) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await session.list_tools(params=params)
            tools.extend(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "inputSchema": copy.deepcopy(tool.input_schema),
                }
                for tool in result.tools
            )
            cursor = result.next_cursor
            if not cursor:
                return tools

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        session = self._session
        tool_call_lock = self._tool_call_lock
        if session is None or tool_call_lock is None:
            raise MCPStartupError("MCP session is not connected.")
        # The SDK session and stdio transport are shared application infrastructure.
        # Serialize only request execution across that connection; model and RAG work
        # for unrelated visitors remains concurrent.
        async with tool_call_lock:
            result = await session.call_tool(name, arguments)
        if not isinstance(result, types.CallToolResult):
            raise MCPMalformedResultError("MCP returned an unsupported result type.")
        if result.is_error:
            raise MCPToolCallError("The MCP server reported a tool execution error.")
        if result.structured_content is not None:
            structured = result.structured_content
            # MCP SDK 2.x wraps non-model function returns as {"result": value}
            # to satisfy the generated output schema.  Keep that SDK detail inside
            # this boundary and expose the application value to callers.
            if isinstance(structured, dict) and set(structured) == {"result"}:
                return structured["result"]
            return structured
        if len(result.content) == 1 and isinstance(result.content[0], types.TextContent):
            try:
                return json.loads(result.content[0].text)
            except json.JSONDecodeError as exc:
                raise MCPMalformedResultError("MCP returned malformed structured data.") from exc
        raise MCPMalformedResultError("MCP returned no usable structured result.")

    def _submit(self, coroutine: Any) -> Future[Any]:
        loop = self._loop
        if loop is None or not loop.is_running():
            coroutine.close()
            raise MCPStartupError("MCP event loop is not running.")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)
