from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from evals.models import (
    CriterionResult,
    EvalArtifacts,
    EvalCriterionExpectation,
    SemanticCriterionResult,
)


JUDGE_INSTRUCTIONS = """You are a constrained evaluator of an AI process-discovery system.
Evaluate only the supplied criterion against the supplied artifact and human-authored ground truth.
Do not reward fluency. Do not assume facts that are absent. Return a concise, evidence-based
explanation; do not provide hidden reasoning or chain-of-thought."""


def select_artifact(name: str | None, artifacts: EvalArtifacts) -> Any:
    if name == "assistant_messages":
        return artifacts.assistant_messages
    if name == "process_state":
        return artifacts.state_after.model_dump(mode="json")
    if name == "state_and_messages":
        return {
            "assistant_messages": artifacts.assistant_messages,
            "process_state": artifacts.state_after.model_dump(mode="json"),
        }
    if name == "analysis_and_proposal":
        result = artifacts.orchestration_result
        return None if result is None else {
            "analysis": result.analysis.model_dump(mode="json"),
            "proposal": result.proposal.model_dump(mode="json"),
        }
    raise ValueError(f"Unknown semantic artifact selector: {name}")


class SemanticEvaluator:
    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self.call_count = 0

    def evaluate(
        self, expectation: EvalCriterionExpectation, artifacts: EvalArtifacts
    ) -> CriterionResult:
        payload = {
            "criterion_id": expectation.criterion_id,
            "criterion": expectation.description,
            "rubric": expectation.rubric,
            "known_ground_truth": expectation.ground_truth,
            "actual_artifact": select_artifact(expectation.artifact, artifacts),
        }
        self.call_count += 1
        response: Any = self.client.responses.parse(
            model=self.model,
            instructions=JUDGE_INSTRUCTIONS,
            input=json.dumps(payload, ensure_ascii=False),
            text_format=SemanticCriterionResult,
        )
        parsed = SemanticCriterionResult.model_validate(response.output_parsed)
        return CriterionResult(
            criterion_id=expectation.criterion_id,
            description=expectation.description,
            metric=expectation.metric,
            passed=parsed.passed,
            evaluator="semantic",
            explanation=parsed.explanation,
        )
