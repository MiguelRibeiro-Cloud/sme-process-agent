from evals.models import EvalScenario
from evals.scenarios.discovery import DISCOVERY_SCENARIOS
from evals.scenarios.orchestration import ORCHESTRATION_SCENARIOS
from evals.scenarios.rag import RAG_SCENARIOS
from evals.scenarios.routing import ROUTING_SCENARIOS


def load_scenarios() -> list[EvalScenario]:
    """Return fresh validated copies so runs cannot mutate source fixtures."""

    scenarios = [
        *ROUTING_SCENARIOS,
        *RAG_SCENARIOS,
        *DISCOVERY_SCENARIOS,
        *ORCHESTRATION_SCENARIOS,
    ]
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("evaluation scenario IDs must be unique")
    return [scenario.model_copy(deep=True) for scenario in scenarios]


__all__ = ["load_scenarios"]
