from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI

from backend.agent import DiscoverySession
from backend.config import (
    OPENAI_EMBEDDING_MODEL,
    OPENAI_EVAL_MODEL,
    OPENAI_MODEL,
    PROJECT_ROOT,
)
from backend.mcp_client import MCPClientError, NorthstarMCPClient
from backend.orchestration.models import OrchestrationResult, VerifiedProposal
from backend.orchestration.service import OrchestrationService
from backend.orchestration.shared import process_state_fingerprint
from backend.orchestration.verifier import verify_proposal
from backend.rag.embeddings import OpenAIEmbeddingProvider
from backend.rag.service import RAGService
from evals.evaluators import SemanticEvaluator, evaluate_deterministic
from evals.models import (
    CriterionResult,
    EvalArtifacts,
    EvalScenario,
    RetrievedChunkArtifact,
    ScenarioRunResult,
)
from evals.report import build_report, persist_report
from evals.scenarios import load_scenarios


class EvaluationRunner:
    def __init__(
        self,
        *,
        client: OpenAI,
        mcp_client: NorthstarMCPClient,
        rag_service: RAGService,
        model: str,
        eval_model: str,
        deterministic_only: bool = False,
    ):
        self.client = client
        self.mcp_client = mcp_client
        self.rag_service = rag_service
        self.model = model
        self.eval_model = eval_model
        self.deterministic_only = deterministic_only
        self.semantic_evaluator = SemanticEvaluator(client, eval_model)

    def execute_scenario(self, scenario: EvalScenario, run_number: int) -> ScenarioRunResult:
        started = perf_counter()
        artifacts = EvalArtifacts(
            state_before=scenario.initial_state.model_copy(deep=True),
            state_after=scenario.initial_state.model_copy(deep=True),
        )
        errors: list[str] = []
        session = DiscoverySession(
            self.mcp_client,
            model=self.model,
            rag_service=self.rag_service,
            initial_state=scenario.initial_state,
        )
        orchestration = OrchestrationService(model=self.model)

        try:
            for turn in scenario.turns:
                self._capture_events(
                    list(session.event_stream(turn.user_message, self.client)), artifacts
                )
            artifacts.state_after = session.state_snapshot()

            if scenario.run_orchestration:
                self._capture_events(
                    list(orchestration.event_stream(artifacts.state_after, self.client)),
                    artifacts,
                )

            if scenario.verifier_fixture is not None:
                fixture = scenario.verifier_fixture
                artifacts.model_call_attempted = True
                verification = verify_proposal(
                    self.client,
                    self.model,
                    artifacts.state_after.model_copy(deep=True),
                    fixture.analysis.model_copy(deep=True),
                    fixture.proposal.model_copy(deep=True),
                )
                artifacts.orchestration_result = OrchestrationResult(
                    process_state_fingerprint=process_state_fingerprint(artifacts.state_after),
                    model=self.model,
                    created_at=datetime.now(UTC),
                    analysis=fixture.analysis,
                    proposal=fixture.proposal,
                    verification=VerifiedProposal.model_validate(verification),
                )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

        errors.extend(artifacts.errors)
        criteria: list[CriterionResult] = []
        for expectation in scenario.expectations:
            try:
                if expectation.evaluator == "deterministic":
                    criteria.append(evaluate_deterministic(expectation, artifacts))
                elif self.deterministic_only:
                    criteria.append(
                        CriterionResult(
                            criterion_id=expectation.criterion_id,
                            description=expectation.description,
                            metric=expectation.metric,
                            passed=True,
                            evaluator="semantic",
                            explanation="Skipped by --deterministic-only.",
                            skipped=True,
                        )
                    )
                else:
                    criteria.append(self.semantic_evaluator.evaluate(expectation, artifacts))
            except Exception as exc:
                criteria.append(
                    CriterionResult(
                        criterion_id=expectation.criterion_id,
                        description=expectation.description,
                        metric=expectation.metric,
                        passed=False,
                        evaluator=expectation.evaluator,
                        explanation=f"Evaluator error: {type(exc).__name__}: {exc}",
                    )
                )
        evaluated = [criterion for criterion in criteria if not criterion.skipped]
        passed = not errors and all(item.passed for item in evaluated)
        return ScenarioRunResult(
            scenario_id=scenario.scenario_id,
            scenario_name=scenario.name,
            category=scenario.category,
            run_number=run_number,
            passed=passed,
            criteria=criteria,
            duration_ms=max(0, round((perf_counter() - started) * 1000)),
            errors=errors,
            artifacts=artifacts,
        )

    def run(self, scenarios: list[EvalScenario], runs: int) -> list[ScenarioRunResult]:
        results: list[ScenarioRunResult] = []
        for scenario in scenarios:
            for run_number in range(1, runs + 1):
                print(f"Running {scenario.scenario_id} ({run_number}/{runs})...", flush=True)
                try:
                    results.append(self.execute_scenario(scenario, run_number))
                except Exception as exc:
                    # A harness-level failure is recorded and the remaining scenarios continue.
                    artifacts = EvalArtifacts(
                        state_before=scenario.initial_state,
                        state_after=scenario.initial_state,
                        errors=[f"Harness error: {type(exc).__name__}: {exc}"],
                    )
                    results.append(
                        ScenarioRunResult(
                            scenario_id=scenario.scenario_id,
                            scenario_name=scenario.name,
                            category=scenario.category,
                            run_number=run_number,
                            passed=False,
                            criteria=[],
                            duration_ms=0,
                            errors=artifacts.errors,
                            artifacts=artifacts,
                        )
                    )
        return results

    @staticmethod
    def _capture_events(events: list[dict[str, Any]], artifacts: EvalArtifacts) -> None:
        for event in events:
            artifacts.events.append(event)
            event_type = event.get("type")
            if event_type == "assistant_message":
                artifacts.assistant_messages.append(str(event.get("content", "")))
            elif event_type == "tool_call_requested":
                tool = str(event.get("tool", ""))
                if tool:
                    artifacts.capabilities_requested.append(tool)
            elif event_type in {
                "llm_call_started",
                "analysis_started",
                "automation_design_started",
                "verification_started",
            }:
                artifacts.model_call_attempted = True
            elif event_type == "mcp_tool_call_started":
                artifacts.mcp_tools_invoked.append(str(event.get("tool", "")))
            elif event_type == "rag_retrieval_started":
                artifacts.rag_queries.append(str(event.get("query", "")))
                artifacts.capabilities_requested.append("search_company_knowledge")
            elif event_type == "rag_retrieval_completed":
                for item in event.get("results", []):
                    artifacts.retrieved_chunks.append(
                        RetrievedChunkArtifact.model_validate(item)
                    )
            elif event_type == "process_state_updated":
                artifacts.state_after = artifacts.state_after.model_validate(event["state"])
            elif event_type == "orchestration_completed":
                artifacts.orchestration_result = OrchestrationResult.model_validate(event["result"])
            elif event_type in {"agent_error", "orchestration_failed"}:
                artifacts.errors.append(str(event.get("message", event_type)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scenario-based AI behavior evaluations.")
    parser.add_argument("--scenario", action="append", help="Scenario ID (repeatable).")
    parser.add_argument("--category", action="append", help="Category name (repeatable).")
    parser.add_argument("--runs", type=int, default=1, help="Runs per scenario (default: 1).")
    parser.add_argument("--dry-run", action="store_true", help="Show the execution plan without API calls.")
    parser.add_argument("--deterministic-only", action="store_true", help="Skip semantic judge criteria; product-model scenarios still run.")
    parser.add_argument("--list", action="store_true", help="List available scenarios and exit.")
    return parser


def select_scenarios(
    scenarios: list[EvalScenario], scenario_ids: list[str] | None, categories: list[str] | None
) -> list[EvalScenario]:
    selected = scenarios
    if scenario_ids:
        wanted = set(scenario_ids)
        selected = [scenario for scenario in selected if scenario.scenario_id in wanted]
        missing = wanted - {scenario.scenario_id for scenario in selected}
        if missing:
            raise ValueError(f"Unknown scenario ID(s): {', '.join(sorted(missing))}")
    if categories:
        wanted_categories = set(categories)
        selected = [scenario for scenario in selected if scenario.category in wanted_categories]
        known_categories = {scenario.category for scenario in scenarios}
        missing = wanted_categories - known_categories
        if missing:
            raise ValueError(f"Unknown category/categories: {', '.join(sorted(missing))}")
    return selected


def print_plan(scenarios: list[EvalScenario], runs: int, deterministic_only: bool) -> None:
    print(f"Scenarios selected: {len(scenarios)}")
    print(f"Runs per scenario: {runs}")
    print(f"Maximum scenario executions: {len(scenarios) * runs}")
    print(f"Model: {OPENAI_MODEL}")
    print(f"Eval model: {OPENAI_EVAL_MODEL}")
    print(f"Semantic judge checks: {'skipped' if deterministic_only else 'enabled'}")
    print("No dollar estimate is shown because token pricing is not configured.")
    for scenario in scenarios:
        print(f"- {scenario.scenario_id} [{scenario.category}]: {scenario.name}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runs < 1:
        print("--runs must be at least 1", file=sys.stderr)
        return 2
    scenarios = load_scenarios()
    if args.list:
        for scenario in scenarios:
            print(f"{scenario.scenario_id}\t{scenario.category}\t{scenario.name}")
        return 0
    try:
        selected = select_scenarios(scenarios, args.scenario, args.category)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not selected:
        print("No scenarios matched the supplied filters.", file=sys.stderr)
        return 2
    print_plan(selected, args.runs, args.deterministic_only)
    if args.dry_run:
        print("Dry run complete. No OpenAI, embedding, MCP, or semantic-judge calls were made.")
        return 0

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    mcp_client = NorthstarMCPClient()
    try:
        mcp_client.start()
    except MCPClientError as exc:
        print(f"Could not start Northstar MCP client: {exc}", file=sys.stderr)
        return 1
    try:
        runner = EvaluationRunner(
            client=client,
            mcp_client=mcp_client,
            rag_service=RAGService(OpenAIEmbeddingProvider(OPENAI_EMBEDDING_MODEL)),
            model=OPENAI_MODEL,
            eval_model=OPENAI_EVAL_MODEL,
            deterministic_only=args.deterministic_only,
        )
        results = runner.run(selected, args.runs)
        report = build_report(
            model=OPENAI_MODEL,
            eval_model=OPENAI_EVAL_MODEL,
            runs_per_scenario=args.runs,
            selected_scenario_count=len(selected),
            results=results,
            model_backed_scenario_executions=sum(
                result.artifacts.model_call_attempted for result in results
            ),
            semantic_judge_calls=runner.semantic_evaluator.call_count,
            deterministic_only=args.deterministic_only,
        )
        json_path, markdown_path = persist_report(report, PROJECT_ROOT)
        print(f"Scenario runs passed: {report.scenario_runs_passed}/{report.scenario_runs_total}")
        print(f"Criteria passed: {report.overall_criteria_passed}/{report.overall_criteria_total}")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {markdown_path}")
        return 0 if report.scenario_runs_passed == report.scenario_runs_total else 1
    finally:
        mcp_client.stop()


if __name__ == "__main__":
    raise SystemExit(main())
