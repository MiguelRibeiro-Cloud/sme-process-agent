from evals.models import EvalCriterionExpectation as C
from evals.models import EvalScenario, EvalTurn


RAG_SCENARIOS = [
    EvalScenario(
        scenario_id="policy-retrieval",
        name="Policy retrieval",
        description="Retrieve and ground Northstar's discount approval threshold.",
        category="rag_grounding",
        turns=[EvalTurn(user_message="What does company policy say about a 15% discount?")],
        expectations=[
            C(criterion_id="rag-selected", description="Company knowledge retrieval is selected", evaluator="deterministic", check="rag_used", metric="Capability routing accuracy"),
            C(criterion_id="policy-document", description="Commercial Discount Policy is retrieved", evaluator="deterministic", check="retrieved_document", expected="Commercial Discount Policy", metric="RAG grounding/retrieval success"),
            C(criterion_id="threshold-chunk", description="A relevant approval-threshold chunk is retrieved", evaluator="deterministic", check="retrieved_text_contains_all", expected=["above 10%", "Sales Manager"], metric="RAG grounding/retrieval success"),
            C(criterion_id="threshold-answer", description="Answer states the supported approval threshold", evaluator="semantic", check="semantic", artifact="state_and_messages", metric="RAG grounding/retrieval success", rubric="Pass only when the assistant communicates that a 15% discount is above the documented 10% threshold and therefore requires Sales Manager approval. Semantically equivalent wording is allowed. Fail if the threshold, comparison, approver, or requirement is wrong or omitted.", ground_truth={"discount": "15%", "documented_rule": "Discounts above 10% require Sales Manager approval."}),
        ],
    ),
    EvalScenario(
        scenario_id="policy-practice-conflict",
        name="Policy/practice conflict",
        description="Preserve reported practice and documented policy as conflicting claims.",
        category="conflict_preservation",
        turns=[
            EvalTurn(user_message="We don't get any management approval unless the discount is above 15%. Marta is our Sales Manager and handles those approvals."),
            EvalTurn(user_message="Please compare that practice with company discount policy."),
        ],
        expectations=[
            C(criterion_id="rag-used", description="Policy is retrieved", evaluator="deterministic", check="rag_used", metric="RAG grounding/retrieval success"),
            C(criterion_id="conflict-created", description="An unresolved conflict is captured", evaluator="deterministic", check="unresolved_conflict_min", expected=1, metric="Conflict preservation"),
            C(criterion_id="both-sources", description="User practice and document policy remain separate evidence", evaluator="deterministic", check="evidence_source_types", expected=["user", "document"], metric="Conflict preservation"),
            C(criterion_id="claims-distinguished", description="Reported practice and policy are clearly distinguished", evaluator="semantic", check="semantic", artifact="state_and_messages", metric="Conflict preservation", rubric="Pass only if the artifact preserves the reported practice of no management approval at or below 15% and Marta handling approvals above 15%, alongside the documented above-10% Sales Manager rule, as distinct unresolved claims. Fail if either side is silently rewritten.", ground_truth={"reported_practice": "No management approval is obtained unless the discount is above 15%; Marta is the Sales Manager and handles those approvals.", "documented_policy": "Discounts above 10% require Sales Manager approval."}),
        ],
    ),
]
