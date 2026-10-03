import json
from typing import Any

from openai import OpenAI

from backend.orchestration.models import (
    AutomationProposal,
    ProcessAnalysis,
    VerifiedProposal,
)
from backend.orchestration.shared import evidence_catalog
from backend.state import ProcessState


VERIFIER_INSTRUCTIONS = """You are the Evidence Verifier in an application-controlled workflow.
Verify every supplied recommendation against the discovery evidence and analysis. Do not design new
automation. Do not accept a recommendation merely because it sounds reasonable. Mark it unsupported
or blocked_by_unknown when its factual basis or a required dependency is missing. Use
partially_supported when only part of the recommendation is evidenced. Cite only supplied evidence
IDs and never invent support. Audit provenance as part of support quality. A claim presented as
documented policy must cite document evidence; user evidence alone cannot support that wording. A
claim about actual current practice must cite user-reported evidence; document evidence alone cannot
prove behavior. A system-confirmed claim must cite MCP evidence. Inference must remain qualified.
Downgrade an otherwise plausible recommendation to partially_supported, unsupported, or
blocked_by_unknown when its cited source type does not support how the claim is characterized.
Respect ProcessState.flow branches and reject support that incorrectly generalizes one branch to all
cases. Preserve rejected recommendations in the verification output. Produce
exactly one verification per recommendation and put IDs into the matching status buckets:
supported=approved, unsupported=rejected, partially_supported/blocked_by_unknown=unresolved."""


def build_verifier_input(
    snapshot: ProcessState,
    analysis: ProcessAnalysis,
    proposal: AutomationProposal,
) -> str:
    return json.dumps(
        {
            "process_state_snapshot": snapshot.model_dump(mode="json"),
            "evidence_catalog": evidence_catalog(snapshot),
            "process_analysis": analysis.model_dump(mode="json"),
            "automation_proposal": proposal.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )


def verify_proposal(
    client: OpenAI,
    model: str,
    snapshot: ProcessState,
    analysis: ProcessAnalysis,
    proposal: AutomationProposal,
) -> VerifiedProposal:
    response: Any = client.responses.parse(
        model=model,
        instructions=VERIFIER_INSTRUCTIONS,
        input=build_verifier_input(snapshot, analysis, proposal),
        text_format=VerifiedProposal,
    )
    return VerifiedProposal.model_validate(response.output_parsed)
