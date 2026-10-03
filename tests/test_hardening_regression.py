import unittest

from backend.state import (
    Evidence,
    ProcessBranch,
    ProcessDecision,
    ProcessState,
    ProcessStatePatch,
    ProcessStep,
    TextReplacement,
    apply_process_state_patch,
)


class BranchHeavyQuoteRegressionTests(unittest.TestCase):
    def test_reconciliation_preserves_branching_and_evidence_history(self):
        answered_unknown = "What happens after Eric rejects a non-services quote?"
        original_evidence = Evidence(
            source_type="user",
            source="User interview",
            claim="A system is used after Eric reviews services quotes, but its identity is unknown.",
        )
        state = ProcessState(
            systems=["System (identity unspecified)"],
            unknowns=[answered_unknown],
            evidence=[original_evidence],
        )
        threshold_evidence = Evidence(
            source_type="user",
            source="User interview",
            claim="Below 15% Marta approves; 15% and above goes to Eric.",
        )
        resolution_evidence = Evidence(
            source_type="user",
            source="User interview",
            claim="After Eric rejects, Marta informs the seller and the seller renegotiates with the customer.",
        )
        steps = [
            ProcessStep(step_id="prepare", description="Seller prepares quote", actor="Seller"),
            ProcessStep(step_id="marta-approve", description="Marta approves quote", actor="Marta"),
            ProcessStep(step_id="eric-review", description="Eric reviews quote", actor="Eric"),
            ProcessStep(step_id="quotex-entry", description="Eric enters values into QuoteX", actor="Eric", system="QuoteX"),
            ProcessStep(step_id="auto-approve", description="QuoteX automatically approves the discount", system="QuoteX"),
            ProcessStep(step_id="pdf-stamp", description="Marta creates the PDF and applies the approval stamp", actor="Marta"),
            ProcessStep(step_id="renegotiate", description="Seller renegotiates with the customer", actor="Seller"),
        ]
        decisions = [
            ProcessDecision(
                decision_id="discount-threshold",
                question="Which discount approval route applies?",
                after_step_id="prepare",
                branches=[
                    ProcessBranch(condition="discount < 15%", next_step_ids=["marta-approve"]),
                    ProcessBranch(condition="discount >= 15%", next_step_ids=["eric-review"]),
                ],
            ),
            ProcessDecision(
                decision_id="quote-type",
                question="Is this a services quote?",
                after_step_id="eric-review",
                branches=[
                    ProcessBranch(condition="services", next_step_ids=["quotex-entry", "auto-approve"]),
                    ProcessBranch(condition="non-services", next_step_ids=["pdf-stamp"]),
                ],
            ),
            ProcessDecision(
                decision_id="approval-outcome",
                question="Does Eric approve the non-services quote?",
                after_step_id="eric-review",
                branches=[
                    ProcessBranch(condition="approved", next_step_ids=["pdf-stamp"]),
                    ProcessBranch(condition="rejected", next_step_ids=["renegotiate"]),
                ],
            ),
        ]

        updated = apply_process_state_patch(
            state,
            ProcessStatePatch(
                process_name="Sales quote preparation and approval",
                systems_to_replace=[
                    TextReplacement(
                        old_value="System (identity unspecified)", new_value="QuoteX"
                    )
                ],
                steps_to_add=steps,
                decisions_to_add=decisions,
                unknowns_to_remove=[answered_unknown],
                unknowns_to_add=["What happens after customer renegotiation?"],
                evidence_to_add=[threshold_evidence, resolution_evidence],
            ),
        )

        self.assertEqual(updated.process_name, "Sales quote preparation and approval")
        self.assertEqual(updated.systems, ["QuoteX"])
        self.assertNotIn(answered_unknown, updated.unknowns)
        self.assertEqual(updated.unknowns, ["What happens after customer renegotiation?"])
        self.assertEqual(updated.evidence, [original_evidence, threshold_evidence, resolution_evidence])
        self.assertEqual(len(updated.flow.decisions), 3)
        self.assertEqual(
            updated.flow.decisions[1].branches[0].next_step_ids,
            ["quotex-entry", "auto-approve"],
        )
        self.assertEqual(threshold_evidence.source_type, "user")

    def test_newly_supported_single_route_resolves_old_unknown_atomically(self):
        answered = "What happens after Eric rejects a non-services quote?"
        state = ProcessState(
            flow={
                "steps": [
                    {
                        "step_id": "handle-non-services",
                        "description": "Eric approves or rejects the non-services quote.",
                        "actor": "Eric",
                    }
                ]
            },
            unknowns=[answered],
        )
        evidence = Evidence(
            source_type="user",
            source="User interview",
            claim="After rejection, Marta informs the seller and the seller renegotiates.",
        )

        updated = apply_process_state_patch(
            state,
            ProcessStatePatch(
                steps_to_add=[
                    ProcessStep(
                        step_id="renegotiate",
                        description="Seller renegotiates with the customer.",
                        actor="Seller",
                    )
                ],
                decisions_to_add=[
                    ProcessDecision(
                        decision_id="rejection-route",
                        question="What is Eric's decision?",
                        after_step_id="handle-non-services",
                        branches=[
                            ProcessBranch(
                                condition="rejected",
                                next_step_ids=["renegotiate"],
                            )
                        ],
                    )
                ],
                unknowns_to_remove=[answered],
                unknowns_to_add=["What happens after customer renegotiation?"],
                evidence_to_add=[evidence],
            ),
        )

        self.assertNotIn(answered, updated.unknowns)
        self.assertEqual(updated.unknowns, ["What happens after customer renegotiation?"])
        self.assertEqual(updated.flow.decisions[0].branches[0].condition, "rejected")
        self.assertEqual(updated.evidence, [evidence])


if __name__ == "__main__":
    unittest.main()
