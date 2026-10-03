import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from pydantic import ValidationError

from backend import app
from backend.agent import DiscoverySession
from backend.orchestration.analyst import ANALYST_INSTRUCTIONS, analyze_process, build_analyst_input
from backend.orchestration.designer import DESIGNER_INSTRUCTIONS, build_designer_input, design_automation
from backend.orchestration.models import (
    AutomationProposal,
    AutomationRecommendation,
    ProcessAnalysis,
    ProcessFinding,
    RecommendationVerification,
    ToBeStep,
    VerifiedProposal,
)
from backend.orchestration.service import OrchestrationService
from backend.orchestration.shared import evidence_catalog, process_state_fingerprint
from backend.orchestration.verifier import VERIFIER_INSTRUCTIONS, build_verifier_input, verify_proposal
from backend.state import Evidence, ProcessState, ProcessStep
from backend.sessions import SessionStore


def sample_state() -> ProcessState:
    return ProcessState(
        process_name="Customer quote preparation",
        actors=["Sales", "John", "Marta"],
        systems=["Excel", "Email", "Legacy quoting system"],
        steps=[
            "Sales records quote data in Excel",
            "Sales emails quote information to John",
            "John enters the quote in the legacy quoting system",
            "Marta approves exceptional discounts",
        ],
        decisions=["Marta reviews discounts reported as above 15%"],
        pain_points=["Approval waiting time delays quotes"],
        unknowns=["Which approval threshold should govern the TO-BE process?"],
        evidence=[
            Evidence(
                source_type="user",
                source="User interview",
                claim="Sales emails quote details to John for manual entry",
            ),
            Evidence(
                source_type="document",
                source="discount_policy.md",
                document_title="Commercial Discount Policy",
                chunk_id="discount-policy-001",
                claim="Discounts above 10% require approval",
            ),
        ],
    )


def sample_analysis() -> ProcessAnalysis:
    return ProcessAnalysis(
        summary="Quote preparation includes a supported manual handoff.",
        findings=[
            ProcessFinding(
                finding_id="finding-1",
                category="manual_handoff",
                description="Sales emails quote details to John for re-entry.",
                basis="reported_fact",
                supporting_evidence_ids=["evidence-001"],
                related_steps=["Sales emails quote information to John"],
            )
        ],
        bottlenecks=["Approval waiting time"],
        manual_handoffs=["Sales to John by email"],
        duplicate_work=["Excel data is re-entered in the legacy system"],
        policy_practice_gaps=["Reported and documented thresholds differ"],
        automation_candidates=["Structured quote data transfer"],
        unresolved_questions=["Which threshold should govern routing?"],
    )


def sample_proposal() -> AutomationProposal:
    return AutomationProposal(
        objective="Reduce manual transfer while retaining approval control.",
        recommendations=[
            AutomationRecommendation(
                recommendation_id="recommendation-1",
                title="Transfer structured quote data",
                description="Send captured quote data directly to the quoting system.",
                addresses_finding_ids=["finding-1"],
                proposed_automation="Map validated quote fields into the legacy system.",
                human_control="Sales confirms data before submission.",
                dependencies=["Confirm system write interface"],
                risks=["Incorrect field mapping"],
                supporting_evidence_ids=["evidence-001"],
            )
        ],
        to_be_steps=[
            ToBeStep(
                order=1,
                description="Sales enters quote details once.",
                automation_level="human_controlled",
                human_owner="Sales",
            ),
            ToBeStep(
                order=2,
                description="The system transfers validated fields.",
                automation_level="automated",
            ),
        ],
        retained_human_decisions=["Approve exceptional discounts"],
        assumptions=["A supported write interface exists"],
    )


def verification(status: str = "supported") -> VerifiedProposal:
    approved = ["recommendation-1"] if status == "supported" else []
    rejected = ["recommendation-1"] if status == "unsupported" else []
    unresolved = (
        ["recommendation-1"]
        if status in {"partially_supported", "blocked_by_unknown"}
        else []
    )
    return VerifiedProposal(
        verifications=[
            RecommendationVerification(
                recommendation_id="recommendation-1",
                status=status,
                explanation="The status follows from the supplied discovery evidence.",
                supporting_evidence_ids=(
                    ["evidence-001"] if status != "blocked_by_unknown" else []
                ),
                missing_information=(
                    ["Confirm the integration interface"]
                    if status == "blocked_by_unknown"
                    else []
                ),
            )
        ],
        approved_recommendation_ids=approved,
        rejected_recommendation_ids=rejected,
        unresolved_recommendation_ids=unresolved,
    )


class SpecialistContractTests(unittest.TestCase):
    def test_analyst_receives_complete_process_state_snapshot_and_catalog(self):
        state = sample_state()
        payload = json.loads(build_analyst_input(state))

        self.assertEqual(payload["process_state_snapshot"], state.model_dump(mode="json"))
        self.assertEqual(payload["evidence_catalog"], evidence_catalog(state))
        self.assertEqual(payload["evidence_catalog"][0]["evidence_id"], "evidence-001")
        self.assertEqual(
            payload["evidence_catalog"][0]["provenance"],
            "user_reported_practice",
        )
        self.assertIn("flow", payload["process_state_snapshot"])
        self.assertNotIn("steps", payload["process_state_snapshot"])

    def test_specialist_instructions_define_end_to_end_provenance_discipline(self):
        self.assertIn("never \"documented\"", ANALYST_INSTRUCTIONS)
        self.assertIn("Never call a reported threshold", DESIGNER_INSTRUCTIONS)
        self.assertIn("must cite document evidence", VERIFIER_INSTRUCTIONS)

    def test_structured_specialist_parsing_uses_expected_models_and_inputs(self):
        state = sample_state()
        analysis = sample_analysis()
        proposal = sample_proposal()
        verified = verification()
        client = Mock()
        client.responses.parse.side_effect = [
            SimpleNamespace(output_parsed=analysis.model_dump()),
            SimpleNamespace(output_parsed=proposal.model_dump()),
            SimpleNamespace(output_parsed=verified.model_dump()),
        ]

        self.assertEqual(analyze_process(client, "test-model", state), analysis)
        self.assertEqual(design_automation(client, "test-model", state, analysis), proposal)
        self.assertEqual(
            verify_proposal(client, "test-model", state, analysis, proposal), verified
        )
        formats = [call.kwargs["text_format"] for call in client.responses.parse.call_args_list]
        self.assertEqual(formats, [ProcessAnalysis, AutomationProposal, VerifiedProposal])
        self.assertEqual(
            json.loads(client.responses.parse.call_args_list[1].kwargs["input"])[
                "process_analysis"
            ],
            analysis.model_dump(mode="json"),
        )
        verifier_payload = json.loads(
            client.responses.parse.call_args_list[2].kwargs["input"]
        )
        self.assertEqual(verifier_payload["automation_proposal"], proposal.model_dump(mode="json"))
        self.assertEqual(verifier_payload["evidence_catalog"], evidence_catalog(state))

    def test_invalid_structured_analysis_is_rejected(self):
        with self.assertRaises(ValidationError):
            ProcessAnalysis.model_validate({"summary": "Valid", "findings": [{"bad": True}]})

    def test_verification_statuses_include_supported_unsupported_and_blocked(self):
        for status in (
            "supported",
            "partially_supported",
            "unsupported",
            "blocked_by_unknown",
        ):
            self.assertEqual(verification(status).verifications[0].status, status)

    def test_verification_buckets_must_match_status(self):
        with self.assertRaises(ValidationError):
            VerifiedProposal(
                verifications=[
                    RecommendationVerification(
                        recommendation_id="recommendation-1",
                        status="unsupported",
                        explanation="No evidence supports the dependency.",
                    )
                ],
                approved_recommendation_ids=["recommendation-1"],
            )


class OrchestrationServiceTests(unittest.TestCase):
    def build_service(self, *, analyst=None, designer=None, verifier=None):
        calls = []

        def default_analyst(client, model, state):
            calls.append("analysis")
            return sample_analysis()

        def default_designer(client, model, state, analysis):
            calls.append("design")
            return sample_proposal()

        def default_verifier(client, model, state, analysis, proposal):
            calls.append("verification")
            return verification()

        service = OrchestrationService(
            "test-model",
            analyst=analyst or default_analyst,
            designer=designer or default_designer,
            verifier=verifier or default_verifier,
        )
        return service, calls

    def test_stages_run_in_fixed_sequence_and_events_are_truthful(self):
        service, calls = self.build_service()
        events = list(service.event_stream(sample_state(), Mock()))

        self.assertEqual(calls, ["analysis", "design", "verification"])
        self.assertEqual(
            [event["type"] for event in events],
            [
                "orchestration_started",
                "analysis_started",
                "analysis_completed",
                "automation_design_started",
                "automation_design_completed",
                "verification_started",
                "verification_completed",
                "orchestration_completed",
            ],
        )
        self.assertEqual(events[2]["finding_count"], 1)
        self.assertEqual(events[4]["recommendation_count"], 1)
        self.assertEqual(events[6]["status_counts"]["supported"], 1)

    def test_specialist_cannot_mutate_discovery_state(self):
        state = sample_state()
        before = state.model_copy(deep=True)

        def mutating_analyst(client, model, snapshot):
            snapshot.actors.append("Injected actor")
            snapshot.flow.steps.clear()
            return sample_analysis()

        service, _ = self.build_service(analyst=mutating_analyst)
        list(service.event_stream(state, Mock()))

        self.assertEqual(state, before)
        self.assertNotIn("Injected actor", service.latest(state).analysis.summary)

    def test_each_specialist_receives_prior_outputs_and_snapshot_copies(self):
        state = sample_state()
        seen = {}

        def analyst(client, model, snapshot):
            seen["analyst_state"] = snapshot.model_copy(deep=True)
            return sample_analysis()

        def designer(client, model, snapshot, analysis):
            seen["designer_state"] = snapshot.model_copy(deep=True)
            seen["analysis"] = analysis
            return sample_proposal()

        def verifier_fn(client, model, snapshot, analysis, proposal):
            seen["verifier_state"] = snapshot.model_copy(deep=True)
            seen["proposal"] = proposal
            return verification()

        service, _ = self.build_service(
            analyst=analyst, designer=designer, verifier=verifier_fn
        )
        list(service.event_stream(state, Mock()))

        self.assertEqual(seen["analyst_state"], state)
        self.assertEqual(seen["designer_state"], state)
        self.assertEqual(seen["verifier_state"], state)
        self.assertEqual(seen["analysis"], sample_analysis())
        self.assertEqual(seen["proposal"], sample_proposal())

    def test_invalid_recommendation_finding_reference_stops_before_verifier(self):
        calls = []

        def designer(client, model, state, analysis):
            calls.append("design")
            proposal = sample_proposal()
            proposal.recommendations[0].addresses_finding_ids = ["missing-finding"]
            return proposal

        def verifier_fn(*args):
            calls.append("verification")
            return verification()

        service, _ = self.build_service(designer=designer, verifier=verifier_fn)
        events = list(service.event_stream(sample_state(), Mock()))

        self.assertEqual(calls, ["design"])
        self.assertEqual(events[-2]["type"], "automation_design_failed")
        self.assertEqual(events[-1]["type"], "orchestration_failed")
        self.assertIsNone(service.latest(sample_state()))

    def test_failure_in_analysis_stops_all_later_stages(self):
        designer = Mock()
        verifier_fn = Mock()

        def fail(*args):
            raise RuntimeError("private failure")

        service = OrchestrationService(
            "test-model", analyst=fail, designer=designer, verifier=verifier_fn
        )
        events = list(service.event_stream(sample_state(), Mock()))

        designer.assert_not_called()
        verifier_fn.assert_not_called()
        self.assertEqual(
            [event["type"] for event in events],
            [
                "orchestration_started",
                "analysis_started",
                "analysis_failed",
                "orchestration_failed",
            ],
        )
        self.assertNotIn("private failure", json.dumps(events))

    def test_rejected_recommendation_remains_in_final_result(self):
        service, _ = self.build_service(
            verifier=lambda *args: verification("unsupported")
        )
        events = list(service.event_stream(sample_state(), Mock()))
        result = events[-1]["result"]

        self.assertEqual(
            result["proposal"]["recommendations"][0]["recommendation_id"],
            "recommendation-1",
        )
        self.assertEqual(
            result["verification"]["rejected_recommendation_ids"],
            ["recommendation-1"],
        )

    def test_provenance_guard_rejects_documented_wording_supported_only_by_user(self):
        proposal = sample_proposal()
        proposal.recommendations[0].description = (
            "Route approvals using the documented threshold."
        )
        service, _ = self.build_service(
            designer=lambda *args: proposal,
            verifier=lambda *args: verification("supported"),
        )

        events = list(service.event_stream(sample_state(), Mock()))
        result = events[-1]["result"]

        self.assertEqual(
            result["verification"]["verifications"][0]["status"], "unsupported"
        )
        self.assertEqual(
            result["verification"]["rejected_recommendation_ids"],
            ["recommendation-1"],
        )

    def test_analysis_basis_must_match_cited_evidence_source(self):
        invalid = sample_analysis()
        invalid.findings[0].basis = "documented_policy"
        service, _ = self.build_service(analyst=lambda *args: invalid)

        events = list(service.event_stream(sample_state(), Mock()))

        self.assertEqual(events[-2]["type"], "analysis_failed")
        self.assertEqual(events[-1]["type"], "orchestration_failed")

    def test_result_is_separate_and_becomes_stale_when_state_changes(self):
        state = sample_state()
        service, _ = self.build_service()
        list(service.event_stream(state, Mock()))

        fresh = service.latest(state)
        changed = state.model_copy(deep=True)
        changed.flow.steps.append(
            ProcessStep(step_id="new-step", description="A newly discovered step")
        )
        stale = service.latest(changed)

        self.assertFalse(fresh.stale)
        self.assertTrue(stale.stale)
        self.assertEqual(
            fresh.process_state_fingerprint, process_state_fingerprint(state)
        )
        self.assertFalse(hasattr(state, "proposal"))

    def test_unknown_evidence_reference_stops_stage(self):
        bad_analysis = sample_analysis()
        bad_analysis.findings[0].supporting_evidence_ids = ["evidence-999"]
        service, _ = self.build_service(analyst=lambda *args: bad_analysis)

        events = list(service.event_stream(sample_state(), Mock()))

        self.assertEqual(events[-2]["type"], "analysis_failed")
        self.assertEqual(events[-1]["stage"], "analysis")


class OrchestrationEndpointStateTests(unittest.TestCase):
    def test_reset_clears_derived_orchestration_result(self):
        session_id = "a" * 43
        store = SessionStore(id_factory=lambda: session_id)
        context = store.get_or_create(None).context
        context.process_state = sample_state()
        service = OrchestrationService(
            "test-model",
            analyst=lambda *args: sample_analysis(),
            designer=lambda *args: sample_proposal(),
            verifier=lambda *args: verification(),
        )
        list(service.event_stream(context.state_snapshot(), Mock(), context))
        self.assertIsNotNone(service.latest(context.state_snapshot(), context))

        client = TestClient(app.app)
        client.cookies.set(app.SESSION_COOKIE_NAME, session_id)
        with patch.object(app, "session_store", store):
            result = client.post("/reset-discovery").json()

        self.assertTrue(result["analysis_cleared"])
        self.assertIsNone(context.orchestration_result)

    def test_empty_process_state_is_rejected_before_an_openai_call(self):
        session_id = "a" * 43
        store = SessionStore(id_factory=lambda: session_id)
        store.get_or_create(None)
        service = Mock()
        client = TestClient(app.app)
        client.cookies.set(app.SESSION_COOKIE_NAME, session_id)
        with (
            patch.object(app, "session_store", store),
            patch.object(app, "orchestration_service", service),
        ):
            response = client.post("/analyze-process")

        self.assertEqual(response.status_code, 409)
        service.event_stream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
