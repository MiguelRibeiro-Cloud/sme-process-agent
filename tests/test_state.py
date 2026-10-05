import unittest

from pydantic import ValidationError

from backend.state import (
    Evidence,
    EvidenceConflict,
    ProcessBranch,
    ProcessDecision,
    ProcessFlow,
    ProcessState,
    ProcessStatePatch,
    ProcessStep,
    ProcessStepReplacement,
    TextReplacement,
    apply_process_state_patch,
    effective_patch,
)


class ProcessStatePatchTests(unittest.TestCase):
    def test_additions_apply_and_existing_knowledge_survives(self):
        state = ProcessState(actors=["Sales"], steps=["Prepare quote"])

        updated = apply_process_state_patch(
            state,
            ProcessStatePatch(
                actors_to_add=["Marta"],
                systems_to_add=["CRM"],
                decisions_to_add=[
                    ProcessDecision(
                        decision_id="approval-required",
                        question="Approval required",
                    )
                ],
                pain_points_to_add=["Approvals are slow"],
            ),
        )

        self.assertEqual(updated.actors, ["Sales", "Marta"])
        self.assertEqual(updated.steps, ["Prepare quote"])
        self.assertEqual(updated.systems, ["CRM"])
        self.assertEqual(state.actors, ["Sales"])

    def test_duplicates_are_case_insensitive_for_strings_and_evidence(self):
        evidence = Evidence(source_type="user", source="conversation", claim="Sales prepares quotes")
        state = ProcessState(actors=["Sales"], evidence=[evidence])

        updated = apply_process_state_patch(
            state,
            ProcessStatePatch(
                actors_to_add=["sales", "SALES"],
                evidence_to_add=[
                    Evidence(
                        source_type="USER",
                        source="Conversation",
                        claim="sales prepares quotes",
                    )
                ],
            ),
        )

        self.assertEqual(updated.actors, ["Sales"])
        self.assertEqual(updated.evidence, [evidence])

    def test_unknowns_can_be_resolved_case_insensitively(self):
        state = ProcessState(
            unknowns=["What requires Marta's approval?", "Where is price stored?"]
        )

        updated = apply_process_state_patch(
            state,
            ProcessStatePatch(
                unknowns_to_remove=["WHAT REQUIRES MARTA'S APPROVAL?"]
            ),
        )

        self.assertEqual(updated.unknowns, ["Where is price stored?"])

    def test_malformed_patch_cannot_mutate_or_corrupt_existing_state(self):
        state = ProcessState(actors=["Sales"])

        with self.assertRaises(ValidationError):
            apply_process_state_patch(
                state,
                {"actors_to_add": ["Marta"], "unknown_mutation": "erase everything"},
            )

        self.assertEqual(state, ProcessState(actors=["Sales"]))

    def test_blank_values_are_rejected(self):
        with self.assertRaises(ValidationError):
            ProcessStatePatch(
                steps_to_add=[{"step_id": "step-new", "description": "   "}]
            )

    def test_specific_system_replaces_generic_system_and_preserves_evidence(self):
        original = Evidence(
            source_type="user",
            source="User interview",
            claim="The quote is entered into a system whose identity is not yet known.",
        )
        identifying = Evidence(
            source_type="user",
            source="User interview",
            claim="The previously unidentified system is QuoteX.",
        )
        updated = apply_process_state_patch(
            ProcessState(systems=["System (identity unspecified)"], evidence=[original]),
            ProcessStatePatch(
                systems_to_replace=[
                    TextReplacement(
                        old_value="System (identity unspecified)", new_value="QuoteX"
                    )
                ],
                evidence_to_add=[identifying],
            ),
        )
        self.assertEqual(updated.systems, ["QuoteX"])
        self.assertEqual(updated.evidence, [original, identifying])
        applied = effective_patch(
            ProcessState(systems=["System (identity unspecified)"], evidence=[original]),
            updated,
        )
        self.assertEqual(applied.systems_to_add, [])
        self.assertEqual(applied.systems_to_replace[0].new_value, "QuoteX")

    def test_replacement_requires_an_exact_existing_target_and_is_atomic(self):
        state = ProcessState(systems=["System (identity unspecified)"], actors=["Sales"])
        with self.assertRaises(ValueError):
            apply_process_state_patch(
                state,
                ProcessStatePatch(
                    actors_to_add=["Marta"],
                    systems_to_replace=[
                        TextReplacement(old_value="Unknown application", new_value="QuoteX")
                    ],
                ),
            )
        self.assertEqual(state, ProcessState(systems=["System (identity unspecified)"], actors=["Sales"]))

    def test_partly_answered_unknown_is_replaced_while_evidence_is_retained(self):
        original = "What happens after Eric rejects the quote?"
        evidence = Evidence(
            source_type="user",
            source="User interview",
            claim="After rejection Marta tells the seller, who renegotiates with the customer.",
        )
        updated = apply_process_state_patch(
            ProcessState(unknowns=[original]),
            ProcessStatePatch(
                unknowns_to_remove=[original],
                unknowns_to_add=["What happens after customer renegotiation?"],
                evidence_to_add=[evidence],
            ),
        )
        self.assertEqual(updated.unknowns, ["What happens after customer renegotiation?"])
        self.assertEqual(updated.evidence, [evidence])

    def test_typed_flow_preserves_explicit_branch_routes(self):
        state = ProcessState(
            flow=ProcessFlow(
                steps=[
                    ProcessStep(step_id="prepare", description="Seller prepares quote", actor="Seller"),
                    ProcessStep(step_id="marta", description="Marta approves", actor="Marta"),
                    ProcessStep(step_id="eric", description="Eric reviews", actor="Eric"),
                ],
                decisions=[
                    ProcessDecision(
                        decision_id="discount-route",
                        question="Which discount approval route applies?",
                        after_step_id="prepare",
                        branches=[
                            ProcessBranch(condition="discount < 15%", next_step_ids=["marta"]),
                            ProcessBranch(condition="discount >= 15%", next_step_ids=["eric"]),
                        ],
                    )
                ],
            )
        )
        dumped = state.model_dump(mode="json")
        self.assertNotIn("steps", dumped)
        self.assertEqual(
            dumped["flow"]["decisions"][0]["branches"][1]["next_step_ids"],
            ["eric"],
        )

    def test_flow_rejects_unknown_branch_step_and_step_replacement_preserves_id(self):
        with self.assertRaises(ValidationError):
            ProcessState(
                flow={
                    "steps": [{"step_id": "prepare", "description": "Prepare"}],
                    "decisions": [{
                        "decision_id": "route",
                        "question": "Route?",
                        "branches": [
                            {"condition": "A", "next_step_ids": ["missing"]},
                            {"condition": "B", "next_step_ids": ["prepare"]},
                        ],
                    }],
                }
            )
        with self.assertRaises(ValidationError):
            ProcessStepReplacement(
                target_step_id="prepare",
                replacement=ProcessStep(step_id="other", description="Prepare in QuoteX"),
            )

    def test_partially_discovered_decision_can_represent_an_outcome_with_unknown_next_action(self):
        state = ProcessState(
            flow={
                "steps": [
                    {"step_id": "review", "description": "Eric reviews"},
                    {"step_id": "renegotiate", "description": "Seller renegotiates"},
                ],
                "decisions": [{
                    "decision_id": "approval-outcome",
                    "question": "What is Eric's decision?",
                    "after_step_id": "review",
                    "branches": [
                        {"condition": "approved", "next_step_ids": []},
                        {"condition": "rejected", "next_step_ids": ["renegotiate"]}
                    ],
                }],
            },
            unknowns=["What happens after approval?"],
        )

        self.assertEqual(
            state.flow.decisions[0].branches[0].next_step_ids, []
        )
        self.assertEqual(
            state.flow.decisions[0].branches[1].next_step_ids, ["renegotiate"]
        )
        self.assertEqual(state.unknowns, ["What happens after approval?"])

    def test_process_name_is_inferred_once_and_sensible_name_is_stable(self):
        named = apply_process_state_patch(
            ProcessState(process_name="Not named yet"),
            ProcessStatePatch(process_name="Quote preparation and approval"),
        )
        unchanged = apply_process_state_patch(
            named, ProcessStatePatch(process_name="Another plausible label")
        )
        self.assertEqual(named.process_name, "Quote preparation and approval")
        self.assertEqual(unchanged.process_name, "Quote preparation and approval")

    def test_document_evidence_and_user_evidence_remain_distinguishable(self):
        user = Evidence(source_type="user", source="User interview", claim="Approval above 15%")
        document = Evidence(
            source_type="document",
            source="discount_policy.md",
            document_title="Commercial Discount Policy",
            chunk_id="discount-policy-abc",
            claim="Approval above 10%",
        )

        updated = apply_process_state_patch(
            ProcessState(evidence=[user]),
            ProcessStatePatch(evidence_to_add=[document]),
        )

        self.assertEqual([item.source_type for item in updated.evidence], ["user", "document"])
        self.assertEqual(updated.evidence[1].chunk_id, "discount-policy-abc")

    def test_conflicts_apply_and_reverse_order_duplicates_are_deduplicated(self):
        conflict = EvidenceConflict(
            topic="Discount approval threshold",
            first_claim="Approval above 15%",
            first_source="User interview",
            second_claim="Approval above 10%",
            second_source="Commercial Discount Policy",
        )
        reverse = EvidenceConflict(
            topic="discount approval threshold",
            first_claim="approval above 10%",
            first_source="commercial discount policy",
            second_claim="approval above 15%",
            second_source="user interview",
        )

        updated = apply_process_state_patch(
            ProcessState(conflicts=[conflict]),
            ProcessStatePatch(conflicts_to_add=[reverse]),
        )

        self.assertEqual(updated.conflicts, [conflict])

    def test_malformed_document_evidence_and_conflict_cannot_corrupt_state(self):
        state = ProcessState(actors=["Sales"])
        malformed = {
            "evidence_to_add": [
                {"source_type": "document", "source": "policy.md", "claim": "A rule"}
            ],
            "conflicts_to_add": [
                {
                    "topic": "Threshold",
                    "first_claim": "10%",
                    "first_source": "Policy",
                    "second_claim": "15%",
                    "second_source": "Interview",
                    "status": "maybe",
                }
            ],
        }

        with self.assertRaises(ValidationError):
            apply_process_state_patch(state, malformed)

        self.assertEqual(state, ProcessState(actors=["Sales"]))


if __name__ == "__main__":
    unittest.main()
