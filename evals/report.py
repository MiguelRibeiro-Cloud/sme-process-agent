from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from evals.models import EvalReport, MetricResult, ScenarioRunResult


METRIC_ORDER = [
    "Capability routing accuracy",
    "RAG grounding/retrieval success",
    "Conflict preservation",
    "Unsupported-claim avoidance",
    "Discovery extraction success",
    "Automation finding support",
    "Verifier calibration",
]


def aggregate_metrics(results: list[ScenarioRunResult]) -> list[MetricResult]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for result in results:
        for criterion in result.criteria:
            if criterion.skipped:
                continue
            counts[criterion.metric][1] += 1
            counts[criterion.metric][0] += int(criterion.passed)
    ordered = [*METRIC_ORDER, *sorted(set(counts) - set(METRIC_ORDER))]
    return [
        MetricResult(metric=name, passed=counts[name][0], total=counts[name][1], pass_rate=counts[name][0] / counts[name][1])
        for name in ordered
        if counts[name][1]
    ]


def build_report(
    *, model: str, eval_model: str, runs_per_scenario: int,
    selected_scenario_count: int, results: list[ScenarioRunResult],
    model_backed_scenario_executions: int, semantic_judge_calls: int,
    deterministic_only: bool,
) -> EvalReport:
    evaluated = [criterion for result in results for criterion in result.criteria if not criterion.skipped]
    return EvalReport(
        created_at=datetime.now(UTC), model=model, eval_model=eval_model,
        runs_per_scenario=runs_per_scenario,
        selected_scenario_count=selected_scenario_count,
        scenario_execution_count=len(results),
        model_backed_scenario_executions=model_backed_scenario_executions,
        semantic_judge_calls=semantic_judge_calls,
        deterministic_only=deterministic_only,
        results=results, metrics=aggregate_metrics(results),
        overall_criteria_passed=sum(item.passed for item in evaluated),
        overall_criteria_total=len(evaluated),
        scenario_runs_passed=sum(item.passed for item in results),
        scenario_runs_total=len(results),
    )


def markdown_report(report: EvalReport) -> str:
    criteria_rate = 0 if not report.overall_criteria_total else report.overall_criteria_passed / report.overall_criteria_total
    lines = [
        "# Evaluation Report", "",
        f"Generated: {report.created_at.isoformat()}",
        f"Model: `{report.model}`  ", f"Eval model: `{report.eval_model}`  ",
        f"Scenarios: {report.selected_scenario_count}  ",
        f"Scenario executions: {report.scenario_execution_count} ({report.runs_per_scenario} run{'s' if report.runs_per_scenario != 1 else ''} per scenario)  ",
        f"Passed scenario runs: {report.scenario_runs_passed}/{report.scenario_runs_total}  ",
        f"Criteria pass rate: {report.overall_criteria_passed}/{report.overall_criteria_total} ({criteria_rate:.0%})", "",
        "## Metrics", "",
    ]
    for metric in report.metrics:
        lines.append(f"- {metric.metric}: {metric.passed}/{metric.total} ({metric.pass_rate:.0%})")
    lines.extend(["", "## Scenario results", ""])
    grouped: dict[str, list[ScenarioRunResult]] = defaultdict(list)
    for result in report.results:
        grouped[result.scenario_id].append(result)
    for scenario_results in grouped.values():
        first = scenario_results[0]
        passed = sum(item.passed for item in scenario_results)
        lines.extend([f"### {'✓' if passed == len(scenario_results) else '✕'} {first.scenario_name}", "", f"Passed {passed}/{len(scenario_results)} runs.", ""])
        for result in scenario_results:
            failures = [item for item in result.criteria if not item.passed and not item.skipped]
            if result.errors:
                lines.append(f"- Run {result.run_number} errors: {'; '.join(result.errors)}")
            for failure in failures:
                lines.append(f"- Run {result.run_number} — `{failure.criterion_id}` ({failure.evaluator}): {failure.explanation}")
            if not failures and not result.errors:
                lines.append(f"- Run {result.run_number}: passed")
        lines.append("")
    lines.extend([
        "## Evaluation transparency", "",
        f"- Deterministic checks: {sum(c.evaluator == 'deterministic' and not c.skipped for r in report.results for c in r.criteria)}",
        f"- Semantic judge checks: {sum(c.evaluator == 'semantic' and not c.skipped for r in report.results for c in r.criteria)}",
        f"- Semantic judge API calls: {report.semantic_judge_calls}",
        f"- Model-backed scenario executions: {report.model_backed_scenario_executions}",
        "- Results from small run counts are regression signals, not statistically robust estimates.", "",
    ])
    return "\n".join(lines)


def persist_report(report: EvalReport, root: Path) -> tuple[Path, Path]:
    runs_dir = root / ".evals" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = report.created_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    json_text = report.model_dump_json(indent=2)
    markdown = markdown_report(report)
    run_json = runs_dir / f"{timestamp}.json"
    run_markdown = runs_dir / f"{timestamp}.md"
    run_json.write_text(json_text + "\n", encoding="utf-8")
    run_markdown.write_text(markdown, encoding="utf-8")
    latest_json = root / ".evals" / "latest.json"
    latest_markdown = root / ".evals" / "latest.md"
    latest_json.write_text(json_text + "\n", encoding="utf-8")
    latest_markdown.write_text(markdown, encoding="utf-8")
    return latest_json, latest_markdown


def load_latest_report(
    root: Path,
    *,
    report_path: Path | None = None,
) -> EvalReport | None:
    path = report_path if report_path is not None else root / ".evals" / "latest.json"
    if not path.is_file():
        return None
    try:
        return EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
