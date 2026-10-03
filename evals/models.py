from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.orchestration.models import AutomationProposal, ProcessAnalysis, OrchestrationResult
from backend.state import ProcessState


EvaluatorKind = Literal["deterministic", "semantic"]


class EvalTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_message: str = Field(min_length=1)


class EvalCriterionExpectation(BaseModel):
    """A serializable, human-authored behavioral expectation."""

    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    evaluator: EvaluatorKind
    check: str
    metric: str
    expected: Any = None
    artifact: str | None = None
    rubric: str | None = None
    ground_truth: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def semantic_criteria_have_a_rubric(self) -> "EvalCriterionExpectation":
        if self.evaluator == "semantic" and not self.rubric:
            raise ValueError("semantic criteria require a rubric")
        return self


class VerifierFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    analysis: ProcessAnalysis
    proposal: AutomationProposal


class EvalScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    name: str
    description: str
    category: str
    turns: list[EvalTurn] = Field(default_factory=list)
    run_orchestration: bool = False
    initial_state: ProcessState = Field(default_factory=ProcessState)
    verifier_fixture: VerifierFixture | None = None
    expectations: list[EvalCriterionExpectation] = Field(min_length=1)

    @model_validator(mode="after")
    def has_an_execution_input(self) -> "EvalScenario":
        if not self.turns and self.verifier_fixture is None and not self.run_orchestration:
            raise ValueError("scenario requires turns, orchestration, or a verifier fixture")
        return self


class RetrievedChunkArtifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_title: str
    source: str
    chunk_id: str
    text: str
    similarity_score: float | None = None


class EvalArtifacts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assistant_messages: list[str] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)
    capabilities_requested: list[str] = Field(default_factory=list)
    mcp_tools_invoked: list[str] = Field(default_factory=list)
    rag_queries: list[str] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunkArtifact] = Field(default_factory=list)
    state_before: ProcessState = Field(default_factory=ProcessState)
    state_after: ProcessState = Field(default_factory=ProcessState)
    orchestration_result: OrchestrationResult | None = None
    model_call_attempted: bool = False
    errors: list[str] = Field(default_factory=list)


class CriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion_id: str
    description: str
    metric: str
    passed: bool
    evaluator: EvaluatorKind
    explanation: str
    skipped: bool = False


class ScenarioRunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    scenario_name: str
    category: str
    run_number: int
    passed: bool
    criteria: list[CriterionResult]
    duration_ms: int = Field(ge=0)
    errors: list[str] = Field(default_factory=list)
    artifacts: EvalArtifacts


class MetricResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    passed: int = Field(ge=0)
    total: int = Field(ge=0)
    pass_rate: float = Field(ge=0, le=1)


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    model: str
    eval_model: str
    runs_per_scenario: int = Field(ge=1)
    selected_scenario_count: int = Field(ge=0)
    scenario_execution_count: int = Field(ge=0)
    model_backed_scenario_executions: int = Field(ge=0)
    semantic_judge_calls: int = Field(ge=0)
    deterministic_only: bool = False
    results: list[ScenarioRunResult] = Field(default_factory=list)
    metrics: list[MetricResult] = Field(default_factory=list)
    overall_criteria_passed: int = Field(ge=0)
    overall_criteria_total: int = Field(ge=0)
    scenario_runs_passed: int = Field(ge=0)
    scenario_runs_total: int = Field(ge=0)


class SemanticCriterionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    criterion: str
    passed: bool
    explanation: str
