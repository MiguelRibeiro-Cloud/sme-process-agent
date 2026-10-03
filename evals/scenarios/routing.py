from evals.models import EvalCriterionExpectation as C
from evals.models import EvalScenario, EvalTurn


ROUTING_SCENARIOS = [
    EvalScenario(
        scenario_id="crm-routing",
        name="CRM routing",
        description="Route an account-owner question to CRM and ground the answer.",
        category="capability_routing",
        turns=[EvalTurn(user_message="Who owns Acme Ltd?")],
        expectations=[
            C(criterion_id="crm-called", description="CRM capability is selected", evaluator="deterministic", check="tool_called", expected="crm_get_customer", metric="Capability routing accuracy"),
            C(criterion_id="pricing-not-called", description="Pricing is not used", evaluator="deterministic", check="tool_not_called", expected="pricing_get_product", metric="Capability routing accuracy"),
            C(criterion_id="rag-not-used", description="RAG is not used", evaluator="deterministic", check="rag_not_used", metric="Capability routing accuracy"),
            C(criterion_id="owner-grounded", description="Answer contains the actual account owner", evaluator="deterministic", check="assistant_contains", expected="Sarah Chen", metric="Capability routing accuracy"),
        ],
    ),
    EvalScenario(
        scenario_id="pricing-routing",
        name="Pricing routing",
        description="Route a product-price question to pricing and ground the answer.",
        category="capability_routing",
        turns=[EvalTurn(user_message="What is the standard price of NX-440?")],
        expectations=[
            C(criterion_id="pricing-called", description="Pricing capability is selected", evaluator="deterministic", check="tool_called", expected="pricing_get_product", metric="Capability routing accuracy"),
            C(criterion_id="crm-not-called", description="CRM is not used", evaluator="deterministic", check="tool_not_called", expected="crm_get_customer", metric="Capability routing accuracy"),
            C(criterion_id="rag-not-used", description="RAG is not used", evaluator="deterministic", check="rag_not_used", metric="Capability routing accuracy"),
            C(criterion_id="price-grounded", description="Answer communicates the synthetic price and currency", evaluator="deterministic", check="assistant_price_equivalent", expected={"amount": 1250, "currency": "EUR"}, metric="Capability routing accuracy"),
        ],
    ),
    EvalScenario(
        scenario_id="no-unnecessary-capability",
        name="No unnecessary capability",
        description="A process pain point should remain a conversational discovery turn.",
        category="capability_routing",
        turns=[EvalTurn(user_message="Our quote process takes too long.")],
        expectations=[
            C(criterion_id="no-mcp", description="No MCP lookup is made", evaluator="deterministic", check="no_mcp_tools", metric="Capability routing accuracy"),
            C(criterion_id="no-rag", description="No arbitrary RAG lookup is made", evaluator="deterministic", check="rag_not_used", metric="Capability routing accuracy"),
            C(criterion_id="continues-discovery", description="Response continues process discovery", evaluator="semantic", check="semantic", artifact="assistant_messages", metric="Discovery extraction success", rubric="Pass when the response acknowledges the process pain point and asks or invites a focused follow-up that advances discovery.", ground_truth={"known_fact": "The quote process takes too long."}),
        ],
    ),
]
