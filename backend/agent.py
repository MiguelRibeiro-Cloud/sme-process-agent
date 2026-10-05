import copy
import json
from collections.abc import Iterator
from typing import Any, Protocol

from openai import OpenAI
from pydantic import ValidationError

from backend.config import OPENAI_MODEL
from backend.mcp_client import (
    MCPClientError,
    MCPMalformedResultError,
    MCPToolCallError,
    MCPToolUnavailableError,
    SERVER_NAME,
    TRANSPORT,
)
from backend.state import (
    ProcessState,
    ProcessStatePatch,
    apply_process_state_patch,
    effective_patch,
)
from backend.rag.index import IndexStaleError, IndexUnavailableError
from backend.rag.models import RAGStatus, SearchResult
from backend.sessions import (
    RECENT_MESSAGE_LIMIT,
    ConversationMessage,
    RecentConversation,
    SessionContext,
)


RAG_TOOL_NAME = "search_company_knowledge"
RAG_TOOL = {
    "type": "function",
    "name": RAG_TOOL_NAME,
    "description": (
        "Search Northstar internal documentation for documented policies, procedures, "
        "process guidance, thresholds, rules, and other company knowledge that may "
        "confirm, clarify, or contradict information discovered during the interview."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A concise semantic search query for the company documents.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of relevant chunks to retrieve (1 to 5).",
                "minimum": 1,
                "maximum": 5,
                "default": 3,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

DISCOVERY_INSTRUCTIONS = """You are an SME process-discovery agent, not a general-purpose assistant.
Assist with understanding, documenting, investigating, or improving how work and business
processes are performed, and with investigating the connected company environment through the
available business systems and company knowledge. Your job is to understand the user's current
business process actually works. Clarify actors, systems, steps, decision points, pain points,
evidence, and important unknowns. Interpret short or fragmentary answers in light of the recent
conversation and the current validated ProcessState. Ask one focused, natural follow-up question
when that will advance discovery.

If a request is clearly unrelated to both business-process discovery and the connected company
environment, do not answer its underlying question and do not select MCP or company-knowledge
search. Briefly explain that you help understand and improve business processes, then invite the
user to describe a workflow, manual task, approval, handoff, bottleneck, system interaction,
exception, or automation opportunity.
Apply this boundary by the user's intent, not topic keywords: healthcare, finance, travel, and
other domain topics are in scope when the user is discussing a business workflow. Do not switch to
unrelated tasks such as drafting an email merely because a fragment could be read that way.

Use the available business tools only when the data they provide is genuinely useful. Choose tools
based on their capability descriptions and the user's meaning; a tool is not required on every
turn. Company-knowledge search is an application-owned retrieval capability, distinct from MCP
business-system tools. Use it when documented policy or process guidance would help; choose a
semantic query rather than copying brittle keywords. Never invent business-system or process facts.
ProcessState is application-owned context: do not claim to edit it yourself."""

RAG_GROUNDING_INSTRUCTIONS = """Use the retrieved company evidence for documented-policy claims.
If it does not answer the question, say so. Do not invent missing policy. Keep documented policy
distinct from user-described real-world practice, and state material differences without treating
practice as automatically corrected or documents as proof of employee behavior. For a policy or
rule question, state the relevant governing documented rule clearly, then explain its implication
for the user's specific case when the evidence supports one. Apply this pattern concisely to
approval limits, eligibility thresholds, required steps, mandatory controls, and exceptions."""

EXTRACTION_INSTRUCTIONS = """Extract supported business-process knowledge from the completed
interaction and return a typed ProcessStatePatch. If the current user request is clearly unrelated
to business-process discovery and the assistant redirected the user to this product's scope, return
an empty patch. Do not extract facts, evidence, unknowns, or a process name from the unrelated
request or the scope redirect. The patch can add facts, exactly replace a generic fact with a
supported specific fact, refine an existing step or decision by its stable ID, and resolve or narrow
an unknown. Be conservative. Do not invent actors, systems, steps, decisions, branches, pain
points, or evidence from common business practice. Do not repeat existing knowledge unnecessarily.
Never use replacement merely because two strings look similar: replacement requires the current
turn and conversation context to establish that they are the same entity or activity.
For example, when the current state says "System (identity unspecified)" and the user identifies
that same system as QuoteX, use systems_to_replace rather than systems_to_add. Preserve stable step
and decision IDs when refining them.

Evidence source_type must be user, mcp, or document. A user statement is user-reported practice; an
MCP result is a system fact; retrieved text is documented policy or procedure. Never relabel one as
another.
Only add MCP/document evidence when the corresponding real result is included. Document evidence
must retain its document title, source filename, and chunk ID. Treat documents as documented policy
or procedure, not proof of actual employee behavior. Propose a conflict only when two supported
claims about the same operational rule or behavior are materially incompatible and cannot both be
true without qualification. This includes different thresholds, approval owners, process order,
eligibility rules, or required controls. When user-reported practice and retrieved document evidence
materially disagree, preserve both evidence items and propose an unresolved EvidenceConflict; never
rewrite one to match the other. If an apparent mismatch could still be compatible because identity,
scope, terminology, or another key fact is unknown, add an unknown instead of a conflict. When one
dimension is definitely incompatible but another remains uncertain, record the supported conflict
and keep the separate uncertainty open.

Examples:
- Reported practice says approval begins above 12%, while documented policy requires it above 8%:
  preserve both claims and propose an unresolved threshold conflict.
- Reported practice says Dana approves discounts, while policy requires an Operations Manager:
  if Dana's role is unknown, add that unknown and do not infer an owner conflict.

Add an unknown when an important detail raised by the interaction remains unresolved. Remove an
unknown only when this interaction actually resolved it. If it was partly answered, remove the old
unknown and add one narrower question for the remaining uncertainty. Evidence is append-only:
replacing current state or resolving an unknown never removes evidence history.

Use ProcessFlow as the canonical process representation. Add ProcessStep objects with stable,
descriptive IDs. Create a ProcessDecision only when evidence explicitly establishes conditional
routing; "sometimes" alone is not a supported branch condition. Each branch condition must point
to the ordered IDs of its branch-specific steps when those steps are known. When evidence
establishes decision outcomes with materially different subsequent paths, represent each supported
outcome as an explicit branch rather than leaving the decision embedded in a free-text step. Do not
create branches merely from "or" or "if": the evidence must establish distinct outcomes that affect
routing. If an outcome is established but its next action is not, include that branch with no
next_step_ids and add an unknown for its unresolved next action; never invent a destination. Use
after_step_id when the preceding step is known. Do not flatten branch-specific activities into a
sequence that implies every case executes them.

You may propose a concise process_name after several coherent steps or actors establish the process,
or earlier only when the name is explicit and obvious. Do not name from one vague sentence. Preserve
an existing sensible name. Use process_name_to_replace only when new evidence clearly shows the old
name is misleading. Never attempt to erase or replace the complete process state."""


class MCPClientBoundary(Protocol):
    def discover_tools(self) -> list[dict[str, Any]]: ...

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class RAGServiceBoundary(Protocol):
    def search_company_knowledge(self, query: str, top_k: int = 3) -> SearchResult: ...

    def status(self) -> RAGStatus: ...


class DiscoverySession:
    """Shared discovery engine operating on an explicit ``SessionContext``.

    A private default context is retained for unit-level use and backward-compatible
    embedding. The FastAPI application disables it and always supplies a context from
    ``SessionStore``.
    """

    def __init__(
        self,
        mcp_client: MCPClientBoundary,
        recent_message_limit: int = RECENT_MESSAGE_LIMIT,
        model: str = OPENAI_MODEL,
        rag_service: RAGServiceBoundary | None = None,
        initial_state: ProcessState | None = None,
        create_default_context: bool = True,
    ):
        self.mcp_client = mcp_client
        self.model = model
        self.rag_service = rag_service
        self._context = (
            SessionContext(
                process_state=(initial_state or ProcessState()).model_copy(deep=True),
                recent_messages=RecentConversation(recent_message_limit),
            )
            if create_default_context
            else None
        )

    def _resolve_context(self, context: SessionContext | None) -> SessionContext:
        resolved = context or self._context
        if resolved is None:
            raise RuntimeError("an explicit SessionContext is required")
        return resolved

    @property
    def _state(self) -> ProcessState:
        return self._resolve_context(None).process_state

    @_state.setter
    def _state(self, state: ProcessState) -> None:
        self._resolve_context(None).process_state = state

    @property
    def recent_conversation(self) -> RecentConversation:
        return self._resolve_context(None).recent_messages

    def state_snapshot(self, context: SessionContext | None = None) -> ProcessState:
        return self._resolve_context(context).state_snapshot()

    def reset(self, context: SessionContext | None = None) -> ProcessState:
        return self._resolve_context(context).reset()

    def event_stream(
        self,
        message: str,
        client: OpenAI,
        context: SessionContext | None = None,
    ) -> Iterator[dict[str, Any]]:
        session = self._resolve_context(context)
        with session.state_lock:
            yield from self._run_turn(session, message, client)

    def _run_turn(
        self, session: SessionContext, message: str, client: OpenAI
    ) -> Iterator[dict[str, Any]]:
        state_before = session.process_state.model_copy(deep=True)
        context_before = session.recent_messages.snapshot()
        discovery_input = build_discovery_input(context_before, message)

        try:
            mcp_tools = self.mcp_client.discover_tools()
            openai_tools = mcp_tools_to_openai(mcp_tools) + [copy.deepcopy(RAG_TOOL)]
        except MCPClientError:
            yield from turn_error_events(
                code="mcp_discovery_unavailable",
                stage="mcp_tool_discovery",
                message="Northstar MCP capabilities are currently unavailable.",
                assistant_message=(
                    "I couldn't reach the business-system integration. Please try again."
                ),
            )
            return

        yield llm_started_event(
            self.model,
            "discovery",
            "Interpret process context and decide next action",
        )
        try:
            response = client.responses.create(
                model=self.model,
                instructions=build_discovery_instructions(state_before),
                input=discovery_input,
                tools=openai_tools,
                parallel_tool_calls=False,
            )
        except Exception:
            yield llm_completed_event(self.model, "discovery", status="error")
            yield from turn_error_events(
                code="llm_api_failure",
                stage="initial_llm_call",
                message="The language model request could not be completed.",
                assistant_message=(
                    "I couldn't process that request because the language model service "
                    "is currently unavailable. Please try again."
                ),
            )
            return

        yield llm_completed_event(self.model, "discovery")
        tool_call = next(
            (item for item in response.output if item.type == "function_call"), None
        )
        tool_context: list[dict[str, Any]] = []
        rag_context: list[dict[str, Any]] = []

        if tool_call is None:
            assistant_text = response.output_text
            if not assistant_text:
                yield from turn_error_events(
                    code="empty_model_response",
                    stage="initial_llm_call",
                    message="The language model returned no response.",
                    assistant_message=(
                        "I couldn't produce a response for that message. Please try again."
                    ),
                )
                return
        else:
            if tool_call.name == RAG_TOOL_NAME:
                rag_outcome = yield from self._execute_rag_call(tool_call)
                if rag_outcome is None:
                    return
                arguments, search_result, grounding_output = rag_outcome
                rag_context.append(search_result.model_dump(mode="json"))
                grounded_purpose = "Generate grounded response from retrieved company evidence"
            else:
                tool_outcome = yield from self._execute_mcp_tool_call(tool_call, mcp_tools)
                if tool_outcome is None:
                    return
                arguments, tool_result = tool_outcome
                tool_context.append(
                    {"tool": tool_call.name, "arguments": arguments, "result": tool_result}
                )
                grounding_output = json.dumps(tool_result)
                grounded_purpose = "Generate grounded response from MCP tool result"
            yield llm_started_event(
                self.model,
                "grounded_response",
                grounded_purpose,
            )
            try:
                grounded_request: dict[str, Any] = {
                    "model": self.model,
                    "tools": openai_tools,
                    "parallel_tool_calls": False,
                    "previous_response_id": response.id,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": tool_call.call_id,
                            "output": grounding_output,
                        }
                    ],
                }
                if tool_call.name == RAG_TOOL_NAME:
                    grounded_request["instructions"] = (
                        build_discovery_instructions(state_before)
                        + "\n\n"
                        + RAG_GROUNDING_INSTRUCTIONS
                    )
                final_response = client.responses.create(
                    **grounded_request,
                )
            except Exception:
                yield llm_completed_event(
                    self.model, "grounded_response", status="error"
                )
                yield from turn_error_events(
                    code="llm_api_failure",
                    stage="grounded_llm_call",
                    message="The grounded language model response could not be completed.",
                    assistant_message=(
                        "The business-system lookup completed, but I couldn't generate the final "
                        "response. Please try again."
                    ),
                )
                return

            yield llm_completed_event(self.model, "grounded_response")
            if any(item.type == "function_call" for item in final_response.output):
                yield from turn_error_events(
                    code="tool_call_limit_reached",
                    stage="grounded_llm_call",
                    message="The agent requested another tool after reaching this turn's limit.",
                    assistant_message=(
                        "I couldn't finish that request within the current one-tool limit. "
                        "Please try a more specific question."
                    ),
                )
                return
            assistant_text = final_response.output_text
            if not assistant_text:
                yield from turn_error_events(
                    code="empty_model_response",
                    stage="grounded_llm_call",
                    message="The language model returned no grounded response.",
                    assistant_message=(
                        "The business-system lookup completed, but I couldn't produce a final answer."
                    ),
                )
                return

        yield {"type": "assistant_message", "content": assistant_text}
        session.recent_messages.append_turn(message, assistant_text)

        yield llm_started_event(
            self.model,
            "state_extraction",
            "Extract structured process knowledge from completed turn",
        )
        try:
            extraction_response = client.responses.parse(
                model=self.model,
                instructions=EXTRACTION_INSTRUCTIONS,
                input=build_extraction_input(
                    state_before,
                    context_before,
                    message,
                    assistant_text,
                    tool_context,
                    rag_context,
                ),
                text_format=ProcessStatePatch,
            )
            patch = ProcessStatePatch.model_validate(extraction_response.output_parsed)
            candidate_state = apply_process_state_patch(session.process_state, patch)
        except (ValidationError, ValueError, TypeError, AttributeError):
            yield llm_completed_event(self.model, "state_extraction", status="error")
            yield {
                "type": "agent_error",
                "code": "state_extraction_invalid",
                "stage": "state_extraction",
                "message": (
                    "Structured process knowledge could not be validated; existing state "
                    "was preserved."
                ),
            }
            yield interaction_completed_event("partial")
            return
        except Exception:
            yield llm_completed_event(self.model, "state_extraction", status="error")
            yield {
                "type": "agent_error",
                "code": "state_extraction_failure",
                "stage": "state_extraction",
                "message": (
                    "Process knowledge extraction could not be completed; existing state "
                    "was preserved."
                ),
            }
            yield interaction_completed_event("partial")
            return

        yield llm_completed_event(self.model, "state_extraction")
        if candidate_state != session.process_state:
            applied = effective_patch(session.process_state, candidate_state)
            session.process_state = candidate_state
            yield {
                "type": "process_state_updated",
                "technology": "Pydantic",
                "validation_owner": "Python application",
                "state_owner": "Application-owned ProcessState",
                "validation_status": "validated",
                "patch": applied.model_dump(mode="json"),
                "state": session.process_state.model_dump(mode="json"),
            }

        yield interaction_completed_event("success")

    def _execute_mcp_tool_call(
        self, tool_call: Any, mcp_tools: list[dict[str, Any]]
    ) -> Iterator[Any]:
        try:
            arguments = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError):
            yield {
                "type": "tool_call_requested",
                "technology": "OpenAI Responses API",
                "model": self.model,
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
            }
            yield from turn_error_events(
                code="malformed_tool_arguments",
                stage="tool_dispatch",
                message="The requested tool arguments were not valid JSON.",
                assistant_message=(
                    "I couldn't safely run the requested lookup because its arguments "
                    "were invalid. Please rephrase your request."
                ),
            )
            return None

        yield {
            "type": "tool_call_requested",
            "technology": "OpenAI Responses API",
            "model": self.model,
            "tool": tool_call.name,
            "arguments": arguments,
        }
        tool = next((item for item in mcp_tools if item["name"] == tool_call.name), None)
        if tool is None:
            yield from turn_error_events(
                code="mcp_tool_unavailable",
                stage="mcp_tool_dispatch",
                message=f"The requested MCP tool '{tool_call.name}' is not available.",
                assistant_message=(
                    "I couldn't complete that request because the required business-system "
                    "capability is not available."
                ),
            )
            return None

        argument_error = validate_tool_arguments(tool["inputSchema"], arguments)
        if argument_error:
            yield from turn_error_events(
                code="invalid_tool_arguments",
                stage="mcp_tool_dispatch",
                message=argument_error,
                assistant_message=(
                    "I couldn't safely run the requested lookup because its arguments "
                    "were invalid. Please rephrase your request."
                ),
            )
            return None

        yield {
            "type": "mcp_tool_call_started",
            "technology": "Model Context Protocol",
            "server": SERVER_NAME,
            "transport": TRANSPORT,
            "tool": tool_call.name,
            "arguments": arguments,
        }
        try:
            tool_result = self.mcp_client.call_tool(tool_call.name, arguments)
        except MCPToolUnavailableError:
            yield mcp_call_completed_event(tool_call.name, "error", "Tool unavailable")
            yield from turn_error_events(
                code="mcp_tool_unavailable",
                stage="mcp_tool_execution",
                message="The requested MCP capability is no longer available.",
                assistant_message=(
                    "I couldn't complete that request because the required business-system "
                    "capability became unavailable."
                ),
            )
            return None
        except MCPMalformedResultError:
            yield mcp_call_completed_event(tool_call.name, "error", "Malformed MCP result")
            yield from turn_error_events(
                code="mcp_malformed_result",
                stage="mcp_tool_execution",
                message="The MCP server returned an unusable result.",
                assistant_message="The business-system lookup returned an invalid result. Please try again.",
            )
            return None
        except (MCPToolCallError, MCPClientError):
            yield mcp_call_completed_event(tool_call.name, "error", "MCP call failed")
            yield from turn_error_events(
                code="mcp_tool_call_failure",
                stage="mcp_tool_execution",
                message="The MCP business-system lookup could not be completed.",
                assistant_message="I couldn't complete the business-system lookup. Please try again.",
            )
            return None

        yield mcp_call_completed_event(
            tool_call.name,
            "success",
            summarize_tool_result(arguments, tool_result),
            result_metadata(tool_result),
        )
        return arguments, tool_result

    def _execute_rag_call(self, tool_call: Any) -> Iterator[Any]:
        try:
            arguments = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError):
            yield {
                "type": "tool_call_requested",
                "technology": "OpenAI Responses API",
                "model": self.model,
                "tool": tool_call.name,
                "arguments": tool_call.arguments,
            }
            yield from turn_error_events(
                code="malformed_tool_arguments",
                stage="rag_dispatch",
                message="The requested knowledge-search arguments were not valid JSON.",
                assistant_message="I couldn't safely search the company documentation because the request was invalid.",
            )
            return None

        yield {
            "type": "tool_call_requested",
            "technology": "OpenAI Responses API",
            "model": self.model,
            "tool": tool_call.name,
            "arguments": arguments,
        }
        argument_error = validate_tool_arguments(RAG_TOOL["parameters"], arguments)
        top_k = arguments.get("top_k", 3) if isinstance(arguments, dict) else 3
        if argument_error or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            yield from turn_error_events(
                code="invalid_tool_arguments",
                stage="rag_dispatch",
                message=argument_error or "Knowledge-search top_k must be between 1 and 5.",
                assistant_message="I couldn't safely run the requested company-knowledge search because its arguments were invalid.",
            )
            return None
        if self.rag_service is None:
            yield from turn_error_events(
                code="rag_unavailable",
                stage="rag_retrieval",
                message="Company knowledge is not configured for this agent session.",
                assistant_message="The company-knowledge index is currently unavailable. Please build the index and try again.",
            )
            return None

        query = arguments["query"].strip()
        if not query:
            yield from turn_error_events(
                code="invalid_tool_arguments",
                stage="rag_dispatch",
                message="The company-knowledge query cannot be blank.",
                assistant_message="I couldn't run the company-document search because its query was empty.",
            )
            return None
        yield {
            "type": "rag_retrieval_started",
            "technology": "Application-owned RAG",
            "query": query,
            "top_k": top_k,
        }
        try:
            result = self.rag_service.search_company_knowledge(query, top_k)
        except IndexStaleError:
            yield rag_retrieval_completed_event(query, top_k, "error", "Index is stale")
            yield from turn_error_events(
                code="rag_index_stale",
                stage="rag_retrieval",
                message="The company-knowledge index is stale and must be rebuilt.",
                assistant_message="The company-knowledge index is out of date. Please rebuild it before relying on document retrieval.",
            )
            return None
        except IndexUnavailableError:
            yield rag_retrieval_completed_event(query, top_k, "error", "Index is unavailable")
            yield from turn_error_events(
                code="rag_unavailable",
                stage="rag_retrieval",
                message="The company-knowledge index has not been built or cannot be loaded.",
                assistant_message="The company-knowledge index is currently unavailable. Please build it and try again.",
            )
            return None
        except (ValueError, OSError):
            yield rag_retrieval_completed_event(query, top_k, "error", "Retrieval failed")
            yield from turn_error_events(
                code="rag_retrieval_failure",
                stage="rag_retrieval",
                message="The company-knowledge search could not be completed.",
                assistant_message="I couldn't complete the company-document search. Please try again.",
            )
            return None

        yield {
            "type": "rag_query_embedded",
            "technology": "OpenAI Embeddings API",
            "embedding_model": result.embedding_model,
            "query": result.query,
        }
        hit_details = [
            {
                "document_title": hit.document_title,
                "source": hit.source,
                "chunk_id": hit.chunk_id,
                "chunk_number": hit.chunk_number,
                "similarity_score": hit.similarity_score,
                "text": hit.text,
            }
            for hit in result.hits
        ]
        yield {
            "type": "rag_retrieval_completed",
            "technology": "Plain Python cosine similarity",
            "status": "success",
            "query": result.query,
            "embedding_model": result.embedding_model,
            "index_chunk_count": result.index_chunk_count,
            "top_k": result.top_k,
            "results": hit_details,
        }
        grounding_output = format_retrieved_evidence(result)
        yield {
            "type": "rag_evidence_supplied",
            "technology": "OpenAI Responses API",
            "model": self.model,
            "documents": list(dict.fromkeys(hit.document_title for hit in result.hits)),
            "chunk_ids": [hit.chunk_id for hit in result.hits],
        }
        return arguments, result, grounding_output


def mcp_tools_to_openai(mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate MCP discovery metadata to Responses API function definitions."""
    return [
        {
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": copy.deepcopy(tool["inputSchema"]),
        }
        for tool in mcp_tools
    ]


def build_discovery_instructions(state: ProcessState) -> str:
    return (
        f"{DISCOVERY_INSTRUCTIONS}\n\nCurrent validated ProcessState (read-only):\n"
        f"{state.model_dump_json(indent=2)}"
    )


def build_discovery_input(
    recent_context: list[ConversationMessage], current_message: str
) -> list[dict[str, str]]:
    messages = [message.model_dump() for message in recent_context]
    messages.append({"role": "user", "content": current_message})
    return messages


def build_extraction_input(
    state_before: ProcessState,
    recent_context: list[ConversationMessage],
    user_message: str,
    assistant_message: str,
    tool_context: list[dict[str, Any]],
    rag_context: list[dict[str, Any]] | None = None,
) -> str:
    payload = {
        "process_state_before_patch": state_before.model_dump(mode="json"),
        "recent_conversation_before_current_turn": [
            message.model_dump() for message in recent_context
        ],
        "completed_turn": {"user": user_message, "assistant": assistant_message},
        "actual_tool_calls": tool_context,
        "actual_rag_retrievals": rag_context or [],
    }
    return json.dumps(payload, ensure_ascii=False)


def format_retrieved_evidence(result: SearchResult) -> str:
    if not result.hits:
        return "Retrieved company evidence:\n\nNo relevant document chunks were retrieved."
    sections = []
    for hit in result.hits:
        label = (
            f"{hit.document_title} | {hit.source} | chunk {hit.chunk_number} "
            f"({hit.chunk_id})"
        )
        sections.append(f"[{label}]\n{hit.text}")
    return "Retrieved company evidence:\n\n" + "\n\n".join(sections)


def rag_retrieval_completed_event(
    query: str, top_k: int, status: str, result_summary: str
) -> dict[str, Any]:
    return {
        "type": "rag_retrieval_completed",
        "technology": "Application-owned RAG",
        "status": status,
        "query": query,
        "top_k": top_k,
        "result_summary": result_summary,
        "results": [],
    }


def validate_tool_arguments(schema: dict[str, Any], arguments: Any) -> str | None:
    if not isinstance(arguments, dict):
        return "Tool arguments must be a JSON object."
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    missing = [name for name in required if name not in arguments]
    if missing:
        return f"Missing required tool argument: {missing[0]}."
    unexpected = [name for name in arguments if name not in properties]
    if unexpected and schema.get("additionalProperties") is False:
        return f"Unexpected tool argument: {unexpected[0]}."
    json_type_map = {
        "string": str,
        "number": (int, float),
        "integer": int,
        "boolean": bool,
        "object": dict,
        "array": list,
    }
    for name, value in arguments.items():
        expected_json_type = properties.get(name, {}).get("type")
        expected_python_type = json_type_map.get(expected_json_type)
        if expected_python_type and not isinstance(value, expected_python_type):
            return f"Tool argument '{name}' must be {expected_json_type}."
    return None


def summarize_tool_result(arguments: dict[str, Any], result: Any) -> str:
    if result is None:
        lookup_value = next(iter(arguments.values()), "requested value")
        return f"No matching record found for {lookup_value}"
    if isinstance(result, dict):
        identifier = next(
            (result[key] for key in ("customer_id", "sku", "id") if key in result),
            None,
        )
        if identifier is not None:
            return f"Record {identifier} found"
    return "MCP tool completed successfully"


def result_metadata(result: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "result_type": type(result).__name__,
        "record_found": result is not None,
    }
    if isinstance(result, dict):
        for key in ("customer_id", "sku", "id"):
            if key in result:
                metadata["record_identifier"] = result[key]
                break
    return metadata


def mcp_call_completed_event(
    tool: str,
    status: str,
    result_summary: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "type": "mcp_tool_call_completed",
        "technology": "Model Context Protocol",
        "server": SERVER_NAME,
        "transport": TRANSPORT,
        "tool": tool,
        "status": status,
        "result_summary": result_summary,
    }
    if metadata:
        event["result_metadata"] = metadata
    return event


def llm_started_event(model: str, operation: str, purpose: str) -> dict[str, str]:
    return {
        "type": "llm_call_started",
        "technology": "OpenAI Responses API",
        "model": model,
        "operation": operation,
        "purpose": purpose,
    }


def llm_completed_event(
    model: str, operation: str, status: str = "success"
) -> dict[str, str]:
    return {
        "type": "llm_call_completed",
        "technology": "OpenAI Responses API",
        "model": model,
        "operation": operation,
        "status": status,
    }


def interaction_completed_event(status: str) -> dict[str, str]:
    return {"type": "interaction_completed", "status": status}


def turn_error_events(
    *, code: str, stage: str, message: str, assistant_message: str
) -> Iterator[dict[str, Any]]:
    yield {"type": "agent_error", "code": code, "stage": stage, "message": message}
    yield {"type": "assistant_message", "content": assistant_message, "error": True}
    yield interaction_completed_event("error")
