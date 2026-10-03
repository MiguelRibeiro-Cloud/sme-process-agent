import json
from typing import Any

from openai import OpenAI

from backend.orchestration.models import AutomationProposal, ProcessAnalysis
from backend.orchestration.shared import evidence_catalog
from backend.state import ProcessState


DESIGNER_INSTRUCTIONS = """You are the Automation Designer in an application-controlled workflow.
Design a realistic TO-BE process from only the supplied ProcessState and ProcessAnalysis. Every
recommendation must address at least one supplied finding ID. Cite only supplied evidence IDs.
Use the evidence catalog's provenance field: call user claims reported practice, document evidence
documented policy/procedure, and MCP evidence a system result. Never call a reported threshold
"documented" or "policy" without document evidence, and never describe document-only evidence as
actual current practice. Mark unsupported design possibilities as assumptions or dependencies.
Preserve human judgment and explicit control where appropriate. Do not treat unresolved policy or
practice conflicts as solved; record missing organizational decisions as dependencies or assumptions.
ProcessState.flow is canonical. Preserve legitimate AS-IS branches in the TO-BE design unless a
supported recommendation intentionally changes them; never flatten branch-specific work into a path
that appears universal.
Use qualitative benefits unless the input contains quantitative evidence. Never invent savings,
volumes, ROI, FTE reductions, or implementation facts. Do not call tools or request new evidence."""


def build_designer_input(snapshot: ProcessState, analysis: ProcessAnalysis) -> str:
    return json.dumps(
        {
            "process_state_snapshot": snapshot.model_dump(mode="json"),
            "evidence_catalog": evidence_catalog(snapshot),
            "process_analysis": analysis.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )


def design_automation(
    client: OpenAI,
    model: str,
    snapshot: ProcessState,
    analysis: ProcessAnalysis,
) -> AutomationProposal:
    response: Any = client.responses.parse(
        model=model,
        instructions=DESIGNER_INSTRUCTIONS,
        input=build_designer_input(snapshot, analysis),
        text_format=AutomationProposal,
    )
    return AutomationProposal.model_validate(response.output_parsed)
