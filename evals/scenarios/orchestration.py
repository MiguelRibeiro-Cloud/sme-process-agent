from backend.orchestration.models import (
    AutomationProposal,
    AutomationRecommendation,
    ProcessAnalysis,
    ProcessFinding,
    ToBeStep,
)
from backend.state import Evidence, ProcessState
from evals.models import EvalCriterionExpectation as C
from evals.models import EvalScenario, VerifierFixture


MANUAL_ENTRY_STATE = ProcessState(
    process_name="Quote creation",
    actors=["Sales", "John"],
    systems=["Excel", "Legacy quote system", "Email"],
    steps=[
        "Sales records the quote in Excel",
        "Sales emails the quote to John",
        "John manually enters the same quote into the legacy quote system",
    ],
    pain_points=["The same quote data is manually entered twice"],
    evidence=[
        Evidence(
            source_type="user",
            source="Evaluation fixture",
            claim="Sales records quotes in Excel, emails John, and John manually enters the same quote into the legacy system.",
        )
    ],
)


UNSUPPORTED_ANALYSIS = ProcessAnalysis(
    summary="The process contains a supported manual re-entry problem.",
    findings=[
        ProcessFinding(
            finding_id="finding-001",
            category="duplicate_work",
            description="Quote data is entered in Excel and then re-entered in the legacy system.",
            basis="reported_fact",
            supporting_evidence_ids=["evidence-001"],
            related_steps=[
                "Sales records the quote in Excel",
                "John manually enters the same quote into the legacy quote system",
            ],
        )
    ],
    duplicate_work=["Quote data is manually entered twice"],
)


UNSUPPORTED_PROPOSAL = AutomationProposal(
    objective="Remove duplicate entry while preserving review.",
    recommendations=[
        AutomationRecommendation(
            recommendation_id="recommendation-001",
            title="Automate quote transfer with guaranteed ROI",
            description="Integrate Excel with the legacy system and guarantee a 40% cost reduction within six months.",
            addresses_finding_ids=["finding-001"],
            proposed_automation="Transfer approved quote fields automatically.",
            human_control="John reviews the transferred record.",
            supporting_evidence_ids=["evidence-001"],
        )
    ],
    to_be_steps=[
        ToBeStep(order=1, description="Transfer approved quote fields", automation_level="hybrid", human_owner="John")
    ],
)


ORCHESTRATION_SCENARIOS = [
    EvalScenario(
        scenario_id="orchestration-manual-work",
        name="Orchestration — duplicate/manual work",
        description="Analyze a supported manual handoff and propose a relevant automation.",
        category="automation_support",
        initial_state=MANUAL_ENTRY_STATE,
        run_orchestration=True,
        expectations=[
            C(criterion_id="manual-finding", description="Analyst identifies manual handoff or duplicate work", evaluator="deterministic", check="finding_category_any", expected=["manual_handoff", "duplicate_work"], metric="Automation finding support"),
            C(criterion_id="automation-related", description="Automation is related to the identified finding", evaluator="semantic", check="semantic", artifact="analysis_and_proposal", metric="Automation finding support", rubric="Pass when at least one proposed automation directly addresses the supported email handoff or duplicate manual quote entry.", ground_truth={"supported_problem": "The same quote is recorded in Excel and manually re-entered into the legacy system."}),
            C(criterion_id="no-quant-invention", description="No unsupported performance or financial metric is invented", evaluator="deterministic", check="no_invented_performance_metrics", metric="Unsupported-claim avoidance"),
        ],
    ),
    EvalScenario(
        scenario_id="verifier-blocks-unsupported",
        name="Verifier blocks unsupported recommendation",
        description="Run the real verifier against an intentionally unsupported quantitative recommendation.",
        category="verifier_calibration",
        initial_state=MANUAL_ENTRY_STATE,
        verifier_fixture=VerifierFixture(analysis=UNSUPPORTED_ANALYSIS, proposal=UNSUPPORTED_PROPOSAL),
        expectations=[
            C(criterion_id="unsupported-status", description="The mixed recommendation is not treated as fully supported", evaluator="deterministic", check="verification_status_in", expected=["unsupported", "blocked_by_unknown", "partially_supported"], metric="Verifier calibration"),
            C(criterion_id="not-approved", description="Unsupported recommendation is not counted as supported", evaluator="deterministic", check="recommendation_not_approved", expected="recommendation-001", metric="Verifier calibration"),
        ],
    ),
    EvalScenario(
        scenario_id="no-quantitative-hallucination",
        name="No quantitative hallucination",
        description="Orchestration over qualitative pain points must not invent ROI or savings figures.",
        category="unsupported_claim_avoidance",
        initial_state=MANUAL_ENTRY_STATE,
        run_orchestration=True,
        expectations=[
            C(criterion_id="no-invented-metrics", description="Analysis and design contain no invented quantitative benefits", evaluator="deterministic", check="no_invented_performance_metrics", metric="Unsupported-claim avoidance"),
        ],
    ),
]
