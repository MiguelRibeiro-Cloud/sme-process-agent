from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from threading import RLock
from time import perf_counter
from typing import TYPE_CHECKING, Any

from openai import OpenAI

from backend.orchestration.analyst import analyze_process
from backend.orchestration.designer import design_automation
from backend.orchestration.models import (
    AutomationProposal,
    OrchestrationResult,
    ProcessAnalysis,
    VerifiedProposal,
)
from backend.orchestration.shared import evidence_catalog, process_state_fingerprint
from backend.orchestration.verifier import verify_proposal
from backend.state import ProcessState

if TYPE_CHECKING:
    from backend.sessions import SessionContext


Analyst = Callable[[OpenAI, str, ProcessState], ProcessAnalysis]
Designer = Callable[
    [OpenAI, str, ProcessState, ProcessAnalysis], AutomationProposal
]
Verifier = Callable[
    [OpenAI, str, ProcessState, ProcessAnalysis, AutomationProposal], VerifiedProposal
]


class OrchestrationService:
    """Own the fixed analyst -> designer -> verifier sequence and derived result."""

    def __init__(
        self,
        model: str,
        analyst: Analyst = analyze_process,
        designer: Designer = design_automation,
        verifier: Verifier = verify_proposal,
        retain_default_result: bool = True,
    ):
        self.model = model
        self._analyst = analyst
        self._designer = designer
        self._verifier = verifier
        self._retain_default_result = retain_default_result
        self._latest: OrchestrationResult | None = None
        self._lock = RLock()

    def latest(
        self,
        current_state: ProcessState,
        session: "SessionContext | None" = None,
    ) -> OrchestrationResult | None:
        lock = session.state_lock if session is not None else self._lock
        with lock:
            latest = session.orchestration_result if session is not None else self._latest
            if latest is None:
                return None
            stale = (
                latest.process_state_fingerprint
                != process_state_fingerprint(current_state)
            )
            return latest.model_copy(update={"stale": stale}, deep=True)

    def clear(self, session: "SessionContext | None" = None) -> None:
        lock = session.state_lock if session is not None else self._lock
        with lock:
            if session is not None:
                session.orchestration_result = None
            else:
                self._latest = None

    def event_stream(
        self,
        snapshot: ProcessState,
        client: OpenAI,
        session: "SessionContext | None" = None,
    ) -> Iterator[dict[str, Any]]:
        # Deep-copy once at the boundary. Specialists never receive discovery-owned state.
        frozen_snapshot = snapshot.model_copy(deep=True)
        fingerprint = process_state_fingerprint(frozen_snapshot)
        catalog = evidence_catalog(frozen_snapshot)
        evidence_ids = {item["evidence_id"] for item in catalog}
        started_at = perf_counter()
        yield {
            "type": "orchestration_started",
            "model": self.model,
            "stage": "orchestration",
            "process_state_fingerprint": fingerprint,
            "snapshot_counts": _snapshot_counts(frozen_snapshot),
        }

        analysis_start = perf_counter()
        yield {
            "type": "analysis_started",
            "model": self.model,
            "stage": "analysis",
            "process_state_fingerprint": fingerprint,
        }
        try:
            analysis = self._analyst(client, self.model, frozen_snapshot.model_copy(deep=True))
            _validate_analysis_references(analysis, catalog, frozen_snapshot)
        except Exception:
            yield _stage_failed("analysis")
            yield _orchestration_failed("analysis", started_at)
            return
        yield {
            "type": "analysis_completed",
            "model": self.model,
            "stage": "analysis",
            "status": "success",
            "duration_ms": _duration_ms(analysis_start),
            "finding_count": len(analysis.findings),
            "referenced_evidence_ids": _analysis_evidence_ids(analysis),
            "structured_output": analysis.model_dump(mode="json"),
        }

        design_start = perf_counter()
        yield {
            "type": "automation_design_started",
            "model": self.model,
            "stage": "automation_design",
            "process_state_fingerprint": fingerprint,
        }
        try:
            proposal = self._designer(
                client,
                self.model,
                frozen_snapshot.model_copy(deep=True),
                analysis.model_copy(deep=True),
            )
            _validate_proposal_references(proposal, analysis, evidence_ids)
        except Exception:
            yield _stage_failed("automation_design")
            yield _orchestration_failed("automation_design", started_at)
            return
        yield {
            "type": "automation_design_completed",
            "model": self.model,
            "stage": "automation_design",
            "status": "success",
            "duration_ms": _duration_ms(design_start),
            "recommendation_count": len(proposal.recommendations),
            "retained_human_decision_count": len(proposal.retained_human_decisions),
            "referenced_evidence_ids": _proposal_evidence_ids(proposal),
            "structured_output": proposal.model_dump(mode="json"),
        }

        verification_start = perf_counter()
        yield {
            "type": "verification_started",
            "model": self.model,
            "stage": "verification",
            "process_state_fingerprint": fingerprint,
        }
        try:
            verification = self._verifier(
                client,
                self.model,
                frozen_snapshot.model_copy(deep=True),
                analysis.model_copy(deep=True),
                proposal.model_copy(deep=True),
            )
            verification = _apply_provenance_guard(
                verification, proposal, analysis, catalog
            )
            _validate_verification_references(verification, proposal, evidence_ids)
        except Exception:
            yield _stage_failed("verification")
            yield _orchestration_failed("verification", started_at)
            return
        status_counts = _verification_counts(verification)
        yield {
            "type": "verification_completed",
            "model": self.model,
            "stage": "verification",
            "status": "success",
            "duration_ms": _duration_ms(verification_start),
            "status_counts": status_counts,
            "referenced_evidence_ids": _verification_evidence_ids(verification),
            "structured_output": verification.model_dump(mode="json"),
        }

        result = OrchestrationResult(
            process_state_fingerprint=fingerprint,
            model=self.model,
            created_at=datetime.now(UTC),
            analysis=analysis,
            proposal=proposal,
            verification=verification,
        )
        if session is not None:
            with session.state_lock:
                session.orchestration_result = result.model_copy(deep=True)
        elif self._retain_default_result:
            with self._lock:
                self._latest = result.model_copy(deep=True)
        yield {
            "type": "orchestration_completed",
            "model": self.model,
            "stage": "orchestration",
            "status": "success",
            "duration_ms": _duration_ms(started_at),
            "process_state_fingerprint": fingerprint,
            "result": result.model_dump(mode="json"),
        }


def _duration_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


def _snapshot_counts(snapshot: ProcessState) -> dict[str, int]:
    return {
        "actors": len(snapshot.actors),
        "systems": len(snapshot.systems),
        "steps": len(snapshot.steps),
        "decisions": len(snapshot.decisions),
        "pain_points": len(snapshot.pain_points),
        "unknowns": len(snapshot.unknowns),
        "evidence": len(snapshot.evidence),
        "conflicts": len(snapshot.conflicts),
    }


def _stage_failed(stage: str) -> dict[str, str]:
    return {
        "type": f"{stage}_failed",
        "stage": stage,
        "status": "error",
        "message": "The structured stage output could not be generated or validated.",
    }


def _orchestration_failed(stage: str, started_at: float) -> dict[str, Any]:
    return {
        "type": "orchestration_failed",
        "stage": stage,
        "status": "error",
        "duration_ms": _duration_ms(started_at),
        "message": f"Orchestration stopped during {stage}; later stages were not run.",
    }


def _validate_analysis_references(
    analysis: ProcessAnalysis,
    catalog: list[dict[str, object]],
    snapshot: ProcessState,
) -> None:
    evidence_types = {
        item["evidence_id"]: item["source_type"] for item in catalog
    }
    referenced = set(_analysis_evidence_ids(analysis))
    if not referenced <= set(evidence_types):
        raise ValueError("analysis references unknown evidence IDs")
    state_steps = set(snapshot.steps)
    related_steps = {
        step for finding in analysis.findings for step in finding.related_steps
    }
    if not related_steps <= state_steps:
        raise ValueError("analysis references a step absent from ProcessState")
    required_source_type = {
        "user_reported_practice": "user",
        "documented_policy": "document",
        "mcp_system_fact": "mcp",
    }
    for finding in analysis.findings:
        required = required_source_type.get(finding.basis)
        cited_types = {
            evidence_types[evidence_id]
            for evidence_id in finding.supporting_evidence_ids
        }
        if required is not None and required not in cited_types:
            raise ValueError("analysis finding basis does not match cited provenance")


def _apply_provenance_guard(
    verification: VerifiedProposal,
    proposal: AutomationProposal,
    analysis: ProcessAnalysis,
    catalog: list[dict[str, object]],
) -> VerifiedProposal:
    """Deterministically prevent a supported verdict when claim wording relabels its source."""
    source_types = {
        str(item["evidence_id"]): str(item["source_type"]) for item in catalog
    }
    findings = {finding.finding_id: finding for finding in analysis.findings}
    recommendations = {
        recommendation.recommendation_id: recommendation
        for recommendation in proposal.recommendations
    }
    guarded = []
    for result in verification.verifications:
        recommendation = recommendations[result.recommendation_id]
        finding_ids = recommendation.addresses_finding_ids
        evidence_ids = set(result.supporting_evidence_ids)
        evidence_ids.update(recommendation.supporting_evidence_ids)
        for finding_id in finding_ids:
            evidence_ids.update(findings[finding_id].supporting_evidence_ids)
        cited_types = {source_types[item] for item in evidence_ids if item in source_types}
        text = " ".join(
            [
                recommendation.title,
                recommendation.description,
                recommendation.proposed_automation,
                *[findings[item].description for item in finding_ids],
            ]
        ).casefold()
        mismatch = None
        if any(term in text for term in ("documented", "policy says", "policy requires")) and "document" not in cited_types:
            mismatch = "Documented-policy wording is not supported by document evidence."
        elif any(term in text for term in ("current practice", "reported practice", "employees currently")) and "user" not in cited_types:
            mismatch = "Current-practice wording is not supported by user-reported evidence."
        elif any(term in text for term in ("system-confirmed", "system result", "system shows")) and "mcp" not in cited_types:
            mismatch = "System-fact wording is not supported by MCP evidence."
        if mismatch and result.status == "supported":
            result = result.model_copy(
                update={
                    "status": "unsupported",
                    "explanation": f"{result.explanation} {mismatch}",
                    "missing_information": [*result.missing_information, mismatch],
                },
                deep=True,
            )
        guarded.append(result)

    return VerifiedProposal(
        verifications=guarded,
        approved_recommendation_ids=[
            item.recommendation_id for item in guarded if item.status == "supported"
        ],
        rejected_recommendation_ids=[
            item.recommendation_id for item in guarded if item.status == "unsupported"
        ],
        unresolved_recommendation_ids=[
            item.recommendation_id
            for item in guarded
            if item.status in {"partially_supported", "blocked_by_unknown"}
        ],
    )


def _validate_proposal_references(
    proposal: AutomationProposal,
    analysis: ProcessAnalysis,
    evidence_ids: set[object],
) -> None:
    finding_ids = {item.finding_id for item in analysis.findings}
    for recommendation in proposal.recommendations:
        if not set(recommendation.addresses_finding_ids) <= finding_ids:
            raise ValueError("recommendation references an unknown finding ID")
    if not set(_proposal_evidence_ids(proposal)) <= evidence_ids:
        raise ValueError("proposal references unknown evidence IDs")


def _validate_verification_references(
    verification: VerifiedProposal,
    proposal: AutomationProposal,
    evidence_ids: set[object],
) -> None:
    recommendation_ids = {
        item.recommendation_id for item in proposal.recommendations
    }
    verified_ids = {item.recommendation_id for item in verification.verifications}
    if verified_ids != recommendation_ids:
        raise ValueError("verifier must return exactly one result per recommendation")
    if not set(_verification_evidence_ids(verification)) <= evidence_ids:
        raise ValueError("verification references unknown evidence IDs")


def _analysis_evidence_ids(analysis: ProcessAnalysis) -> list[str]:
    return sorted(
        {
            evidence_id
            for finding in analysis.findings
            for evidence_id in finding.supporting_evidence_ids
        }
    )


def _proposal_evidence_ids(proposal: AutomationProposal) -> list[str]:
    return sorted(
        {
            evidence_id
            for item in proposal.recommendations
            for evidence_id in item.supporting_evidence_ids
        }
    )


def _verification_evidence_ids(verification: VerifiedProposal) -> list[str]:
    return sorted(
        {
            evidence_id
            for item in verification.verifications
            for evidence_id in item.supporting_evidence_ids
        }
    )


def _verification_counts(verification: VerifiedProposal) -> dict[str, int]:
    statuses = (
        "supported",
        "partially_supported",
        "unsupported",
        "blocked_by_unknown",
    )
    return {
        status: sum(item.status == status for item in verification.verifications)
        for status in statuses
    }
