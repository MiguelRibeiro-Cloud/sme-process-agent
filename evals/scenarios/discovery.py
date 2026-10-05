from backend.state import Evidence, ProcessState
from evals.models import EvalCriterionExpectation as C
from evals.models import EvalScenario, EvalTurn


DISCOVERY_SCENARIOS = [
    EvalScenario(
        scenario_id="out-of-scope-general-assistant-request",
        name="Out-of-scope general-assistant request",
        description="Redirect a general medical request without using capabilities or changing ProcessState.",
        category="scope_adherence",
        turns=[EvalTurn(user_message="In what interval should I take paracetamol 1000mg for a nasty headache and fever?")],
        expectations=[
            C(criterion_id="no-mcp", description="No MCP capability is invoked", evaluator="deterministic", check="no_mcp_tools", metric="Scope adherence"),
            C(criterion_id="no-rag", description="No company knowledge is retrieved", evaluator="deterministic", check="rag_not_used", metric="Scope adherence"),
            C(criterion_id="state-unchanged", description="The unrelated request does not change ProcessState", evaluator="deterministic", check="state_unchanged", metric="Scope adherence"),
            C(criterion_id="redirected", description="The request is redirected to business-process discovery without answering it", evaluator="semantic", check="semantic", artifact="assistant_messages", metric="Scope adherence", rubric="Pass when the assistant does not provide medication timing, dosage, or other substantive medical advice; communicates its business-process discovery scope; and invites a workflow, manual task, approval, handoff, bottleneck, system interaction, exception, or automation opportunity.", ground_truth={"expected_behavior": "scope redirect without medical advice"}),
        ],
    ),
    EvalScenario(
        scenario_id="missing-knowledge",
        name="Missing knowledge",
        description="Do not invent quote volume when neither the user nor Northstar sources establish it.",
        category="unsupported_claim_avoidance",
        turns=[EvalTurn(user_message="How many quotes do we process every month?")],
        expectations=[
            C(criterion_id="no-invented-volume-state", description="No unsupported quantitative claim enters ProcessState", evaluator="deterministic", check="state_has_no_numbers", metric="Unsupported-claim avoidance"),
            C(criterion_id="unknown-acknowledged", description="Missing information is acknowledged", evaluator="semantic", check="semantic", artifact="state_and_messages", metric="Unsupported-claim avoidance", rubric="Pass when the artifact says the monthly quote volume is unknown or unavailable and does not assert a number.", ground_truth={"monthly_quote_volume": "unknown"}),
        ],
    ),
    EvalScenario(
        scenario_id="discovery-extraction",
        name="Discovery extraction",
        description="Extract actors, systems, handoff, and duplicate manual entry from one concise process statement.",
        category="discovery_extraction",
        turns=[EvalTurn(user_message="Sales writes quotes in Excel, emails them to John, and John manually enters them into the old quote system.")],
        expectations=[
            C(criterion_id="actors", description="Sales and John are captured as actors", evaluator="semantic", check="semantic", artifact="process_state", metric="Discovery extraction success", rubric="Pass when both Sales and John are represented as process actors, allowing semantically equivalent labels.", ground_truth={"actors": ["Sales", "John"]}),
            C(criterion_id="systems", description="Excel and the legacy quote system are captured", evaluator="semantic", check="semantic", artifact="process_state", metric="Discovery extraction success", rubric="Pass when Excel and the old/legacy quote system are represented as distinct systems.", ground_truth={"systems": ["Excel", "legacy quote system"]}),
            C(criterion_id="handoff-reentry", description="Email/manual handoff and manual re-entry are captured", evaluator="semantic", check="semantic", artifact="process_state", metric="Discovery extraction success", rubric="Pass when the state captures that Sales emails the quote to John and John manually re-enters the same quote into the legacy system.", ground_truth={"steps": ["Sales writes quote in Excel", "Sales emails quote to John", "John manually re-enters quote in legacy system"]}),
        ],
    ),
    EvalScenario(
        scenario_id="branch-reconciliation",
        name="Branch-heavy quote process reconciliation",
        description="Refine a generic system, resolve an answered unknown, preserve three routing decisions, and retain user provenance.",
        category="state_reconciliation",
        initial_state=ProcessState(
            systems=["System (identity unspecified)"],
            unknowns=["What happens after Eric rejects a non-services quote?"],
            evidence=[
                Evidence(
                    source_type="user",
                    source="Evaluation fixture",
                    claim="Eric uses an unidentified system while handling some quotes.",
                )
            ],
        ),
        turns=[
            EvalTurn(user_message="This is our sales quote approval process. The seller prepares the quote. In our current practice, discounts below 15% go to Marta; discounts of 15% or more go to Eric."),
            EvalTurn(user_message="For services quotes, Eric enters the values into the system and it automatically approves the discount. For non-services quotes, Eric approves or rejects, and Marta handles the PDF and approval stamp. The unidentified system is QuoteX."),
            EvalTurn(user_message="If Eric rejects a non-services quote, he emails Marta, Marta informs the seller, and the seller renegotiates with the customer. I do not yet know what happens after that renegotiation."),
        ],
        run_orchestration=True,
        expectations=[
            C(criterion_id="specific-system", description="QuoteX replaces the generic system placeholder", evaluator="deterministic", check="state_systems_include", expected=["QuoteX"], metric="State reconciliation"),
            C(criterion_id="generic-removed", description="The generic system placeholder is removed", evaluator="deterministic", check="state_excludes_text", expected=["System (identity unspecified)"], metric="State reconciliation"),
            C(criterion_id="resolved-unknown-removed", description="The answered rejection unknown is removed", evaluator="deterministic", check="state_excludes_text", expected=["What happens after Eric rejects a non-services quote?"], metric="State reconciliation"),
            # The shared root accepts valid grammatical forms while requiring the uncertainty.
            C(criterion_id="remaining-unknown", description="The post-renegotiation uncertainty remains (matching the shared renegotiat root)", evaluator="deterministic", check="state_unknown_contains", expected=["after", "renegotiat"], metric="State reconciliation"),
            C(criterion_id="branches-preserved", description="Threshold, quote-type, and outcome routes remain distinguishable", evaluator="deterministic", check="flow_branch_conditions_include", expected=["15%", "services", "non-services", "approve", "reject"], metric="Branch preservation"),
            C(criterion_id="process-named", description="The substantial process receives a meaningful name", evaluator="deterministic", check="process_name_meaningful", metric="State reconciliation"),
            C(criterion_id="reported-not-documented", description="Orchestration retains the threshold as reported practice rather than documented policy", evaluator="semantic", check="semantic", artifact="analysis_and_proposal", metric="Provenance integrity", rubric="Pass when the 15% routing threshold is described as user-reported/current practice or otherwise clearly distinguished from documented policy. Fail if it is called documented, policy-mandated, or system-confirmed without matching source evidence.", ground_truth={"threshold_provenance": "user-reported practice"}),
        ],
    ),
]
