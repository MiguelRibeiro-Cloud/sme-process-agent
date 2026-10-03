# Release evaluation audit

This audit was completed before the final paid release-candidate evaluation. It reviews 11 scenarios and 37 criteria. The suite now contains 28 deterministic criteria and 9 semantic-judge criteria.

## Classification

- Objective deterministic behavior (28): capability selection/non-selection, RAG use and retrieved evidence, conflict/evidence structure, state reconciliation and branches, finding categories, unsupported-number detection, verifier status/approval, and concrete answer facts that Python can inspect reliably.
- Semantic behavioral expectations (9): conversational continuation, substantive policy answer, preserved conflict meaning, acknowledgment of unknown information, discovery extraction, provenance framing, and proposal relevance.
- Valid strict requirements retained: MCP and RAG routing, absence of unnecessary calls, retrieved policy facts, unresolved conflict and separate provenance, no invented quantitative claims, generic-system reconciliation, resolved/remaining unknown handling, branch distinction, and non-approval of the mixed unsupported recommendation.

## Justified pre-run corrections

1. `pricing-routing / price-grounded`
   - Old: required literal `1250` and `EUR` substrings.
   - Problem: rejected equivalent public answers such as `EUR 1,250`, `€1 250`, or `1,250 euros`.
   - New: deterministic numeric/currency-equivalence check for exactly 1250 EUR.
   - Why better: formatting is flexible while the amount and currency remain strict; the independent tool-routing and no-extra-capability checks are unchanged.

2. `policy-retrieval / threshold-answer`
   - Old: required literal `above 10%` and `Sales Manager` substrings.
   - Problem: tested one phrasing of the rule rather than its complete meaning.
   - New: semantic rubric requires the 15% discount to be identified as above the documented 10% threshold and therefore requiring Sales Manager approval.
   - Why better: the answer meaning is judged flexibly, while deterministic retrieval of the exact governing document and threshold chunk remains strict.

3. `policy-practice-conflict` fixture and semantic ground truth
   - Old: “We only ask Marta for approval above 15%.”
   - Problem: allowed the reasonable interpretation that another manager approves discounts from above 10% through 15%, so a conflict was not guaranteed.
   - New: states that no management approval occurs unless the discount is above 15%, and identifies Marta as the Sales Manager who handles those approvals.
   - Why better: it expresses the original intended contradiction without hinting how the system should resolve it.

4. `verifier-blocks-unsupported / unsupported-status`
   - Old: accepted only `unsupported` or `blocked_by_unknown`.
   - Problem: the fixture deliberately combines a supported transfer automation with an unsupported 40% cost-reduction guarantee, so `partially_supported` is semantically valid.
   - New: also accepts `partially_supported`, but the separate deterministic criterion still requires the recommendation ID not to be approved.
   - Why better: it tests that the unsupported guarantee prevents full approval without falsely requiring the supported portion to be discarded.

No product prompt, model setting, orchestration, state, MCP, or RAG behavior was changed. The criteria are frozen after this audit for the final live run.
