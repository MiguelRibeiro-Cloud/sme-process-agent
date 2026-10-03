from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from evals.models import CriterionResult, EvalArtifacts, EvalCriterionExpectation


def _result(
    expectation: EvalCriterionExpectation, passed: bool, explanation: str
) -> CriterionResult:
    return CriterionResult(
        criterion_id=expectation.criterion_id,
        description=expectation.description,
        metric=expectation.metric,
        passed=passed,
        evaluator="deterministic",
        explanation=explanation,
    )


def _assistant_text(artifacts: EvalArtifacts) -> str:
    return "\n".join(artifacts.assistant_messages)


def _tool_called(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    tool = str(e.expected)
    return _result(e, tool in a.mcp_tools_invoked, f"MCP tools invoked: {a.mcp_tools_invoked or 'none'}.")


def _tool_not_called(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    tool = str(e.expected)
    return _result(e, tool not in a.mcp_tools_invoked, f"MCP tools invoked: {a.mcp_tools_invoked or 'none'}.")


def _no_mcp_tools(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    return _result(e, not a.mcp_tools_invoked, f"MCP tools invoked: {a.mcp_tools_invoked or 'none'}.")


def _rag_used(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    return _result(e, bool(a.rag_queries), f"RAG queries: {a.rag_queries or 'none'}.")


def _rag_not_used(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    return _result(e, not a.rag_queries, f"RAG queries: {a.rag_queries or 'none'}.")


def _assistant_contains(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    expected = str(e.expected)
    passed = expected.casefold() in _assistant_text(a).casefold()
    return _result(e, passed, f"Expected assistant output to contain {expected!r}.")


def _assistant_contains_all(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    text = _assistant_text(a).casefold()
    values = [str(value) for value in e.expected]
    missing = [value for value in values if value.casefold() not in text]
    return _result(e, not missing, f"Missing expected answer facts: {missing or 'none'}.")


def _assistant_price_equivalent(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    expected = dict(e.expected)
    expected_amount = float(expected["amount"])
    expected_currency = str(expected["currency"]).upper()
    text = _assistant_text(a)
    normalized_text = text.casefold()
    amount_candidates = re.findall(
        r"(?<![\w])(?:[€$£]\s*)?(\d{1,3}(?:[ ,.']\d{3})+|\d+)(?:[.,](\d{1,2}))?(?![\w])",
        text,
    )
    parsed_amounts: list[float] = []
    for whole, fractional in amount_candidates:
        compact_whole = re.sub(r"[ ,.']", "", whole)
        parsed_amounts.append(float(f"{compact_whole}.{fractional or '0'}"))
    currency_aliases = {
        "EUR": ("eur", "euro", "euros", "€"),
        "USD": ("usd", "dollar", "dollars", "$"),
        "GBP": ("gbp", "pound", "pounds", "£"),
    }
    aliases = currency_aliases.get(expected_currency, (expected_currency.casefold(),))
    amount_present = any(amount == expected_amount for amount in parsed_amounts)
    currency_present = any(alias in normalized_text for alias in aliases)
    return _result(
        e,
        amount_present and currency_present,
        (
            f"Expected amount {expected_amount:g} {expected_currency}; "
            f"numeric values found: {parsed_amounts or 'none'}; "
            f"currency represented: {currency_present}."
        ),
    )


def _retrieved_document(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    titles = [chunk.document_title for chunk in a.retrieved_chunks]
    expected = str(e.expected).casefold()
    passed = any(expected == title.casefold() for title in titles)
    return _result(e, passed, f"Retrieved documents: {titles or 'none'}.")


def _retrieved_text_contains_all(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    text = "\n".join(chunk.text for chunk in a.retrieved_chunks).casefold()
    values = [str(value) for value in e.expected]
    missing = [value for value in values if value.casefold() not in text]
    return _result(e, not missing, f"Missing expected retrieved facts: {missing or 'none'}.")


def _unresolved_conflict_min(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    count = sum(conflict.status == "unresolved" for conflict in a.state_after.conflicts)
    minimum = int(e.expected)
    return _result(e, count >= minimum, f"Found {count} unresolved conflict(s); expected at least {minimum}.")


def _evidence_source_types(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    actual = {item.source_type for item in a.state_after.evidence}
    expected = {str(value) for value in e.expected}
    missing = sorted(expected - actual)
    return _result(e, not missing, f"Evidence source types: {sorted(actual) or 'none'}; missing: {missing or 'none'}.")


def _state_has_no_numbers(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    state = a.state_after
    claims = [
        *(state.actors + state.systems + state.steps + state.decisions + state.pain_points + state.unknowns),
        *(item.claim for item in state.evidence),
        *(claim for conflict in state.conflicts for claim in (conflict.first_claim, conflict.second_claim)),
    ]
    state_text = "\n".join(claims)
    numbers = re.findall(r"(?<![A-Za-z])\d+(?:[.,]\d+)?", state_text)
    return _result(e, not numbers, f"Numeric tokens in ProcessState: {numbers or 'none'}.")


def _state_systems_include(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    actual = {value.casefold() for value in a.state_after.systems}
    expected = {str(value).casefold() for value in e.expected}
    missing = sorted(expected - actual)
    return _result(e, not missing, f"Systems: {a.state_after.systems or 'none'}; missing: {missing or 'none'}.")


def _state_excludes_text(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    state_text = a.state_after.model_dump_json().casefold()
    excluded = [str(value) for value in e.expected]
    present = [value for value in excluded if value.casefold() in state_text]
    return _result(e, not present, f"Excluded text still present: {present or 'none'}.")


def _state_unknown_contains(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    unknowns = "\n".join(a.state_after.unknowns).casefold()
    expected = [str(value) for value in e.expected]
    missing = [value for value in expected if value.casefold() not in unknowns]
    return _result(e, not missing, f"Unknown fragments missing: {missing or 'none'}.")


def _flow_branch_conditions_include(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    conditions = [
        branch.condition
        for decision in a.state_after.flow.decisions
        for branch in decision.branches
    ]
    condition_text = "\n".join(conditions).casefold()
    expected = [str(value) for value in e.expected]
    missing = [value for value in expected if value.casefold() not in condition_text]
    return _result(e, not missing, f"Branch conditions: {conditions or 'none'}; missing: {missing or 'none'}.")


def _process_name_meaningful(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    name = (a.state_after.process_name or "").strip()
    generic = {"", "not named yet", "unnamed process", "unknown process", "process"}
    return _result(e, name.casefold() not in generic, f"Process name: {name or 'none'}.")


def _finding_category_any(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    result = a.orchestration_result
    categories = [] if result is None else [item.category for item in result.analysis.findings]
    expected = {str(value) for value in e.expected}
    passed = bool(expected.intersection(categories))
    return _result(e, passed, f"Finding categories: {categories or 'none'}.")


_SUSPICIOUS_PATTERNS = [
    re.compile(r"(?:€|\$|£)\s?\d[\d,.]*|\d[\d,.]*\s?(?:EUR|USD|GBP)\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s?%\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s+(?:hours?|hrs?|FTEs?|headcount|employees?|people|quotes?|transactions?)\b", re.I),
    re.compile(r"\b(?:ROI|payback|savings?|cost reduction|time saved|hours? saved)\b[^.\n]{0,50}\d", re.I),
    re.compile(r"\b\d+[^.\n]{0,30}\b(?:ROI|payback|savings?|cost reduction|time saved)\b", re.I),
]


def find_suspicious_performance_metrics(artifacts: EvalArtifacts) -> list[str]:
    result = artifacts.orchestration_result
    if result is None:
        return []
    # Verification may quote a bad claim while rejecting it; only analysis/design claims count.
    text = json.dumps(
        {
            "analysis": result.analysis.model_dump(mode="json"),
            "proposal": result.proposal.model_dump(mode="json"),
        },
        ensure_ascii=False,
    )
    source_text = json.dumps(artifacts.state_after.model_dump(mode="json"), ensure_ascii=False)
    supported_numbers = {
        token.replace(",", "").strip().casefold()
        for token in re.findall(r"(?:€|\$|£)?\s?\d+(?:[.,]\d+)?\s?(?:%|EUR|USD|GBP)?", source_text, re.I)
        if re.search(r"\d", token)
    }
    matches: list[str] = []
    for pattern in _SUSPICIOUS_PATTERNS:
        for match in pattern.finditer(text):
            claim = match.group(0)
            numbers = {
                token.replace(",", "").strip().casefold()
                for token in re.findall(r"(?:€|\$|£)?\s?\d+(?:[.,]\d+)?\s?(?:%|EUR|USD|GBP)?", claim, re.I)
                if re.search(r"\d", token)
            }
            if numbers and numbers <= supported_numbers:
                continue
            matches.append(claim)
    return list(dict.fromkeys(matches))


def _no_invented_performance_metrics(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    matches = find_suspicious_performance_metrics(a)
    return _result(e, not matches, f"Suspicious quantitative analysis/design claims: {matches or 'none'}.")


def _verification_status_in(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    result = a.orchestration_result
    statuses = [] if result is None else [item.status for item in result.verification.verifications]
    expected = {str(value) for value in e.expected}
    return _result(e, bool(expected.intersection(statuses)), f"Verification statuses: {statuses or 'none'}.")


def _recommendation_not_approved(e: EvalCriterionExpectation, a: EvalArtifacts) -> CriterionResult:
    result = a.orchestration_result
    approved = [] if result is None else result.verification.approved_recommendation_ids
    identifier = str(e.expected)
    return _result(e, identifier not in approved, f"Approved recommendation IDs: {approved or 'none'}.")


_CHECKS: dict[str, Callable[[EvalCriterionExpectation, EvalArtifacts], CriterionResult]] = {
    "tool_called": _tool_called,
    "tool_not_called": _tool_not_called,
    "no_mcp_tools": _no_mcp_tools,
    "rag_used": _rag_used,
    "rag_not_used": _rag_not_used,
    "assistant_contains": _assistant_contains,
    "assistant_contains_all": _assistant_contains_all,
    "assistant_price_equivalent": _assistant_price_equivalent,
    "retrieved_document": _retrieved_document,
    "retrieved_text_contains_all": _retrieved_text_contains_all,
    "unresolved_conflict_min": _unresolved_conflict_min,
    "evidence_source_types": _evidence_source_types,
    "state_has_no_numbers": _state_has_no_numbers,
    "state_systems_include": _state_systems_include,
    "state_excludes_text": _state_excludes_text,
    "state_unknown_contains": _state_unknown_contains,
    "flow_branch_conditions_include": _flow_branch_conditions_include,
    "process_name_meaningful": _process_name_meaningful,
    "finding_category_any": _finding_category_any,
    "no_invented_performance_metrics": _no_invented_performance_metrics,
    "verification_status_in": _verification_status_in,
    "recommendation_not_approved": _recommendation_not_approved,
}


def evaluate_deterministic(
    expectation: EvalCriterionExpectation, artifacts: EvalArtifacts
) -> CriterionResult:
    try:
        evaluator = _CHECKS[expectation.check]
    except KeyError as exc:
        raise ValueError(f"Unknown deterministic check: {expectation.check}") from exc
    return evaluator(expectation, artifacts)
