import json
from typing import Any

from openai import OpenAI

from backend.orchestration.models import ProcessAnalysis
from backend.orchestration.shared import evidence_catalog
from backend.state import ProcessState


ANALYST_INSTRUCTIONS = """You are the Process Analyst in an application-controlled workflow.
Analyze only the supplied validated AS-IS ProcessState snapshot. Return concise structured output.
Use the evidence catalog's explicit provenance metadata. Distinguish user_reported_practice,
documented_policy, mcp_system_fact, inference, and unresolved uncertainty in both the basis field and
wording. A user claim is "reported", never "documented", "policy", or "system-confirmed" unless a
separate document or MCP item supports that wording. A document describes documented intent, not
proof of employee practice. An MCP result is a system fact, not policy or observed human practice.
Cite only supplied evidence IDs and ensure their provenance matches the claim's wording and basis.
A reasonable possibility such as duplicate entry may be identified from the flow, but label it as an
inference (for example, "may duplicate") rather than confirmed work. Do not silently resolve
evidence conflicts. ProcessState.flow is canonical: preserve decision branches and identify
branch-specific issues without implying that every case executes every branch. When a finding has
related steps, copy their description strings exactly from ProcessState.flow.steps.
Never invent volumes, financial savings, FTE counts, SLA impact, ROI, or facts absent from the input.
Keep unresolved information unresolved. Finding IDs must be unique and stable within this response."""


def build_analyst_input(snapshot: ProcessState) -> str:
    return json.dumps(
        {
            "process_state_snapshot": snapshot.model_dump(mode="json"),
            "evidence_catalog": evidence_catalog(snapshot),
        },
        ensure_ascii=False,
    )


def analyze_process(client: OpenAI, model: str, snapshot: ProcessState) -> ProcessAnalysis:
    response: Any = client.responses.parse(
        model=model,
        instructions=ANALYST_INSTRUCTIONS,
        input=build_analyst_input(snapshot),
        text_format=ProcessAnalysis,
    )
    return ProcessAnalysis.model_validate(response.output_parsed)
