from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


OrchestrationText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000)
]
Identifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)
]


class ProcessFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: Identifier
    category: Literal[
        "bottleneck",
        "manual_handoff",
        "duplicate_work",
        "policy_practice_gap",
        "automation_candidate",
        "other",
    ]
    description: OrchestrationText
    basis: Literal[
        "user_reported_practice",
        "documented_policy",
        "mcp_system_fact",
        "inference",
        "unresolved",
    ]
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    related_steps: list[OrchestrationText] = Field(default_factory=list)

    @field_validator("basis", mode="before")
    @classmethod
    def migrate_reported_fact(cls, value: str) -> str:
        # Read old persisted/demo results while emitting the precise v2 vocabulary.
        return "user_reported_practice" if value == "reported_fact" else value


class ProcessAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: OrchestrationText
    findings: list[ProcessFinding] = Field(default_factory=list)
    bottlenecks: list[OrchestrationText] = Field(default_factory=list)
    manual_handoffs: list[OrchestrationText] = Field(default_factory=list)
    duplicate_work: list[OrchestrationText] = Field(default_factory=list)
    policy_practice_gaps: list[OrchestrationText] = Field(default_factory=list)
    automation_candidates: list[OrchestrationText] = Field(default_factory=list)
    unresolved_questions: list[OrchestrationText] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_finding_ids(self) -> "ProcessAnalysis":
        identifiers = [finding.finding_id for finding in self.findings]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("finding_id values must be unique")
        return self


class AutomationRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: Identifier
    title: OrchestrationText
    description: OrchestrationText
    addresses_finding_ids: list[Identifier] = Field(min_length=1)
    proposed_automation: OrchestrationText
    human_control: OrchestrationText | None = None
    dependencies: list[OrchestrationText] = Field(default_factory=list)
    risks: list[OrchestrationText] = Field(default_factory=list)
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)


class ToBeStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order: int = Field(ge=1)
    description: OrchestrationText
    automation_level: Literal["automated", "human_controlled", "hybrid"]
    human_owner: OrchestrationText | None = None


class AutomationProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objective: OrchestrationText
    recommendations: list[AutomationRecommendation] = Field(default_factory=list)
    to_be_steps: list[ToBeStep] = Field(default_factory=list)
    retained_human_decisions: list[OrchestrationText] = Field(default_factory=list)
    assumptions: list[OrchestrationText] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_identifiers_and_step_order(self) -> "AutomationProposal":
        identifiers = [item.recommendation_id for item in self.recommendations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("recommendation_id values must be unique")
        orders = [step.order for step in self.to_be_steps]
        if len(orders) != len(set(orders)):
            raise ValueError("TO-BE step order values must be unique")
        return self


VerificationStatus = Literal[
    "supported", "partially_supported", "unsupported", "blocked_by_unknown"
]


class RecommendationVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    recommendation_id: Identifier
    status: VerificationStatus
    explanation: OrchestrationText
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    missing_information: list[OrchestrationText] = Field(default_factory=list)


class VerifiedProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verifications: list[RecommendationVerification] = Field(default_factory=list)
    approved_recommendation_ids: list[Identifier] = Field(default_factory=list)
    rejected_recommendation_ids: list[Identifier] = Field(default_factory=list)
    unresolved_recommendation_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_buckets(self) -> "VerifiedProposal":
        identifiers = [item.recommendation_id for item in self.verifications]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("each recommendation may be verified only once")

        expected_approved = {
            item.recommendation_id
            for item in self.verifications
            if item.status == "supported"
        }
        expected_rejected = {
            item.recommendation_id
            for item in self.verifications
            if item.status == "unsupported"
        }
        expected_unresolved = {
            item.recommendation_id
            for item in self.verifications
            if item.status in {"partially_supported", "blocked_by_unknown"}
        }
        supplied = (
            set(self.approved_recommendation_ids),
            set(self.rejected_recommendation_ids),
            set(self.unresolved_recommendation_ids),
        )
        if supplied != (expected_approved, expected_rejected, expected_unresolved):
            raise ValueError("verification status buckets must match recommendation statuses")
        return self


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_state_fingerprint: Identifier
    model: Identifier
    created_at: datetime
    stale: bool = False
    analysis: ProcessAnalysis
    proposal: AutomationProposal
    verification: VerifiedProposal
