import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import get_evaluation_report
from backend.orchestration.models import VerifiedProposal
from backend.state import Evidence, ProcessState, ProcessStep
from backend.config import (
    DEFAULT_EVALUATION_REPORT_PATH,
    resolve_openai_eval_model,
    resolve_project_path,
)
from evals.evaluators.deterministic import (
    evaluate_deterministic,
    find_suspicious_performance_metrics,
)
from evals.evaluators.semantic import SemanticEvaluator
from evals.models import (
    CriterionResult,
    EvalArtifacts,
    EvalCriterionExpectation,
    EvalReport,
    EvalScenario,
    EvalTurn,
    ScenarioRunResult,
    SemanticCriterionResult,
)
from evals.report import aggregate_metrics, build_report, load_latest_report, markdown_report, persist_report
from evals.runner import EvaluationRunner, main
from evals.scenarios import load_scenarios


def criterion(metric="Capability routing accuracy", passed=True, evaluator="deterministic"):
    return CriterionResult(
        criterion_id="criterion",
        description="A criterion",
        metric=metric,
        passed=passed,
        evaluator=evaluator,
        explanation="Evidence-based result.",
    )


def run_result(scenario_id="scenario", run_number=1, criteria=None, passed=True):
    artifacts = EvalArtifacts()
    return ScenarioRunResult(
        scenario_id=scenario_id,
        scenario_name=scenario_id.title(),
        category="category",
        run_number=run_number,
        passed=passed,
        criteria=criteria or [criterion()],
        duration_ms=10,
        artifacts=artifacts,
    )


class ScenarioModelTests(unittest.TestCase):
    def test_scenario_loading_is_stable_unique_and_inspectable(self):
        scenarios = load_scenarios()
        self.assertEqual(12, len(scenarios))
        self.assertEqual(len(scenarios), len({item.scenario_id for item in scenarios}))
        self.assertTrue(all(item.expectations for item in scenarios))
        json.dumps([item.model_dump(mode="json") for item in scenarios])

    def test_scenario_state_isolation_uses_fresh_deep_copies(self):
        first = load_scenarios()
        target = next(item for item in first if item.scenario_id == "orchestration-manual-work")
        target.initial_state.flow.steps.append(
            ProcessStep(step_id="mutation", description="Mutation from prior run")
        )
        second = load_scenarios()
        reloaded = next(item for item in second if item.scenario_id == target.scenario_id)
        self.assertNotIn("Mutation from prior run", reloaded.initial_state.steps)

    def test_eval_model_defaults_to_product_model_and_accepts_override(self):
        self.assertEqual("product", resolve_openai_eval_model({"OPENAI_MODEL": "product"}))
        self.assertEqual("judge", resolve_openai_eval_model({"OPENAI_MODEL": "product", "OPENAI_EVAL_MODEL": " judge "}))


class DeterministicEvaluatorTests(unittest.TestCase):
    def test_objective_tool_and_retrieval_checks_use_captured_artifacts(self):
        artifacts = EvalArtifacts(mcp_tools_invoked=["crm_get_customer"])
        expected = EvalCriterionExpectation(
            criterion_id="crm", description="CRM called", evaluator="deterministic",
            check="tool_called", metric="Capability routing accuracy", expected="crm_get_customer",
        )
        self.assertTrue(evaluate_deterministic(expected, artifacts).passed)

    def test_quantitative_hallucination_check_ignores_supported_thresholds(self):
        scenario = next(item for item in load_scenarios() if item.scenario_id == "verifier-blocks-unsupported")
        artifacts = EvalArtifacts(state_after=scenario.initial_state)
        # The fixture's unsupported 40% claim is in design output, not the source state.
        from backend.orchestration.models import OrchestrationResult
        artifacts.orchestration_result = OrchestrationResult(
            process_state_fingerprint="fingerprint", model="model", created_at=datetime.now(UTC),
            analysis=scenario.verifier_fixture.analysis,
            proposal=scenario.verifier_fixture.proposal,
            verification=VerifiedProposal(verifications=[], approved_recommendation_ids=[], rejected_recommendation_ids=[], unresolved_recommendation_ids=[]),
        )
        self.assertTrue(any("40%" in item for item in find_suspicious_performance_metrics(artifacts)))
        artifacts.state_after.evidence.append(Evidence(source_type="user", source="fixture", claim="A documented threshold is 40%."))
        self.assertFalse(any("40%" in item for item in find_suspicious_performance_metrics(artifacts)))

    def test_price_check_accepts_equivalent_number_and_currency_formatting(self):
        expected = EvalCriterionExpectation(
            criterion_id="price", description="Price communicated", evaluator="deterministic",
            check="assistant_price_equivalent", metric="Capability routing accuracy",
            expected={"amount": 1250, "currency": "EUR"},
        )
        for message in ("The standard price is EUR 1,250.", "It costs €1 250.", "The price is 1250 euros."):
            with self.subTest(message=message):
                artifacts = EvalArtifacts(assistant_messages=[message])
                self.assertTrue(evaluate_deterministic(expected, artifacts).passed)

    def test_price_check_keeps_value_and_currency_strict(self):
        expected = EvalCriterionExpectation(
            criterion_id="price", description="Price communicated", evaluator="deterministic",
            check="assistant_price_equivalent", metric="Capability routing accuracy",
            expected={"amount": 1250, "currency": "EUR"},
        )
        self.assertFalse(evaluate_deterministic(expected, EvalArtifacts(assistant_messages=["EUR 1,200"])).passed)
        self.assertFalse(evaluate_deterministic(expected, EvalArtifacts(assistant_messages=["USD 1,250"])).passed)

    def test_unknown_check_accepts_renegotiation_word_forms_through_the_shared_root(self):
        expected = EvalCriterionExpectation(
            criterion_id="post-renegotiation",
            description="Post-renegotiation uncertainty remains",
            evaluator="deterministic",
            check="state_unknown_contains",
            metric="State reconciliation",
            expected=["after", "renegotiat"],
        )

        for form in ("renegotiate", "renegotiates", "renegotiated", "renegotiation"):
            with self.subTest(form=form):
                result = evaluate_deterministic(
                    expected,
                    EvalArtifacts(
                        state_after=ProcessState(
                            unknowns=[f"What happens after the seller {form} with the customer?"]
                        )
                    ),
                )

                self.assertTrue(result.passed)


class SemanticEvaluatorTests(unittest.TestCase):
    def test_semantic_judge_uses_structured_parsing(self):
        calls = []

        class Responses:
            def parse(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(output_parsed={"criterion": "meaning", "passed": True, "explanation": "The actor is present."})

        evaluator = SemanticEvaluator(SimpleNamespace(responses=Responses()), "judge-model")
        expectation = EvalCriterionExpectation(
            criterion_id="meaning", description="Meaning captured", evaluator="semantic",
            check="semantic", metric="Discovery extraction success", artifact="process_state",
            rubric="Pass when the actor is present.", ground_truth={"actor": "Sales"},
        )
        result = evaluator.evaluate(expectation, EvalArtifacts(state_after=ProcessState(actors=["Sales"])))
        self.assertTrue(result.passed)
        self.assertIs(calls[0]["text_format"], SemanticCriterionResult)
        self.assertEqual("judge-model", calls[0]["model"])


class AggregationAndReportTests(unittest.TestCase):
    def test_evaluation_report_path_is_configurable_and_project_relative(self):
        self.assertEqual(
            resolve_project_path(
                "EVALUATION_REPORT_PATH", DEFAULT_EVALUATION_REPORT_PATH, {}
            ),
            DEFAULT_EVALUATION_REPORT_PATH,
        )
        self.assertEqual(
            resolve_project_path(
                "EVALUATION_REPORT_PATH",
                DEFAULT_EVALUATION_REPORT_PATH,
                {"EVALUATION_REPORT_PATH": "deployment/evaluation-report.json"},
            ),
            DEFAULT_EVALUATION_REPORT_PATH.parents[1]
            / "deployment"
            / "evaluation-report.json",
        )

    def test_repeated_runs_remain_separate_and_aggregate(self):
        results = [run_result("same", 1), run_result("same", 2, [criterion(passed=False)], False)]
        report = build_report(
            model="m", eval_model="j", runs_per_scenario=2, selected_scenario_count=1,
            results=results, model_backed_scenario_executions=2, semantic_judge_calls=0,
            deterministic_only=False,
        )
        self.assertEqual(2, report.scenario_runs_total)
        self.assertEqual(1, report.scenario_runs_passed)
        self.assertEqual([1, 2], [item.run_number for item in report.results])

    def test_category_metric_aggregation_reports_counts_and_percentages(self):
        metrics = aggregate_metrics([
            run_result(criteria=[criterion("Conflict preservation", True)]),
            run_result("two", criteria=[criterion("Conflict preservation", False)], passed=False),
        ])
        metric = next(item for item in metrics if item.metric == "Conflict preservation")
        self.assertEqual((1, 2, 0.5), (metric.passed, metric.total, metric.pass_rate))

    def test_verifier_calibration_is_correctness_not_rejection_rate(self):
        metrics = aggregate_metrics([run_result(criteria=[criterion("Verifier calibration", True)])])
        metric = next(item for item in metrics if item.metric == "Verifier calibration")
        self.assertEqual(1, metric.passed)
        self.assertNotIn("rejection", metric.metric.casefold())

    def test_report_json_serialization_and_markdown_failure_detail(self):
        failed = run_result("missing-knowledge", criteria=[criterion("Unsupported-claim avoidance", False, "semantic")], passed=False)
        report = build_report(
            model="m", eval_model="j", runs_per_scenario=1, selected_scenario_count=1,
            results=[failed], model_backed_scenario_executions=1, semantic_judge_calls=1,
            deterministic_only=False,
        )
        restored = EvalReport.model_validate_json(report.model_dump_json())
        self.assertEqual(report.overall_criteria_total, restored.overall_criteria_total)
        markdown = markdown_report(report)
        self.assertIn("missing-knowledge", markdown.casefold())
        self.assertIn("semantic", markdown.casefold())
        self.assertIn("1 run per scenario", markdown)

    def test_report_persistence_refreshes_latest_json_and_markdown(self):
        report = build_report(
            model="m", eval_model="j", runs_per_scenario=1, selected_scenario_count=1,
            results=[run_result()], model_backed_scenario_executions=1, semantic_judge_calls=0,
            deterministic_only=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            json_path, markdown_path = persist_report(report, Path(directory))
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())
            self.assertIsNotNone(load_latest_report(Path(directory)))

    def test_report_loader_accepts_explicit_release_path_and_missing_artifact(self):
        report = build_report(
            model="m", eval_model="j", runs_per_scenario=1,
            selected_scenario_count=1, results=[run_result()],
            model_backed_scenario_executions=1, semantic_judge_calls=0,
            deterministic_only=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release_path = root / "deployment" / "evaluation-report.json"
            release_path.parent.mkdir()
            release_path.write_text(report.model_dump_json(), encoding="utf-8")
            loaded = load_latest_report(root, report_path=release_path)
            missing = load_latest_report(
                root, report_path=root / "deployment" / "missing.json"
            )

        self.assertEqual(loaded, report)
        self.assertIsNone(missing)


class RunnerBehaviorTests(unittest.TestCase):
    def test_dry_run_constructs_no_openai_client(self):
        with patch("evals.runner.OpenAI", side_effect=AssertionError("OpenAI must not be constructed")):
            self.assertEqual(0, main(["--dry-run", "--scenario", "crm-routing"]))

    def test_deterministic_only_makes_no_semantic_judge_call(self):
        formats = []

        class Responses:
            def parse(self, **kwargs):
                formats.append(kwargs["text_format"])
                return SimpleNamespace(output_parsed={
                    "verifications": [{
                        "recommendation_id": "recommendation-001", "status": "unsupported",
                        "explanation": "The claimed result is unsupported.", "supporting_evidence_ids": [],
                        "missing_information": ["Measured cost and outcome data"],
                    }],
                    "approved_recommendation_ids": [], "rejected_recommendation_ids": ["recommendation-001"],
                    "unresolved_recommendation_ids": [],
                })

        scenario = next(item for item in load_scenarios() if item.scenario_id == "verifier-blocks-unsupported")
        scenario.expectations.append(EvalCriterionExpectation(
            criterion_id="semantic-extra", description="Semantic extra", evaluator="semantic",
            check="semantic", metric="Verifier calibration", artifact="analysis_and_proposal",
            rubric="Pass if relevant.",
        ))
        runner = EvaluationRunner(
            client=SimpleNamespace(responses=Responses()), mcp_client=SimpleNamespace(),
            rag_service=SimpleNamespace(), model="m", eval_model="j", deterministic_only=True,
        )
        result = runner.execute_scenario(scenario, 1)
        self.assertNotIn(SemanticCriterionResult, formats)
        self.assertTrue(next(item for item in result.criteria if item.criterion_id == "semantic-extra").skipped)

    def test_failed_scenario_does_not_abort_remaining_scenarios(self):
        class FaultTolerantRunner:
            def __init__(self):
                self.calls = 0

            def execute_scenario(self, scenario, run_number):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("first failed")
                return run_result(scenario.scenario_id, run_number)

        scenarios = [
            EvalScenario(scenario_id="one", name="One", description="One", category="c", turns=[EvalTurn(user_message="one")], expectations=[EvalCriterionExpectation(criterion_id="c", description="c", evaluator="deterministic", check="no_mcp_tools", metric="m")]),
            EvalScenario(scenario_id="two", name="Two", description="Two", category="c", turns=[EvalTurn(user_message="two")], expectations=[EvalCriterionExpectation(criterion_id="c", description="c", evaluator="deterministic", check="no_mcp_tools", metric="m")]),
        ]
        results = EvaluationRunner.run(FaultTolerantRunner(), scenarios, 1)
        self.assertEqual(2, len(results))
        self.assertFalse(results[0].passed)
        self.assertTrue(results[1].passed)


class EvaluationViewTests(unittest.TestCase):
    def test_latest_report_endpoint_and_ui_handle_missing_report(self):
        with patch("backend.app.load_latest_report", return_value=None):
            self.assertEqual({"status": "not_available", "report": None}, get_evaluation_report())
        html = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
        script = (Path(__file__).parents[1] / "frontend" / "app.js").read_text(encoding="utf-8")
        self.assertIn("No evaluation run available yet.", html)
        self.assertIn('/evaluation-report', script)


if __name__ == "__main__":
    unittest.main()
