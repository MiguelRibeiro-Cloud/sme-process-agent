from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator


StateText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1_000)]
StateIdentifier = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
]


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_type: Literal["user", "mcp", "document"]
    source: StateText
    claim: StateText
    document_title: StateText | None = None
    chunk_id: StateText | None = None

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, value: Any) -> Any:
        if isinstance(value, str):
            normalized = value.strip().casefold()
            aliases = {"tool": "mcp", "system": "mcp", "mcp/system": "mcp"}
            return aliases.get(normalized, normalized)
        return value

    @model_validator(mode="after")
    def validate_document_provenance(self) -> "Evidence":
        if self.source_type == "document" and not (self.document_title and self.chunk_id):
            raise ValueError("Document evidence requires document_title and chunk_id")
        if self.source_type != "document" and (
            self.document_title is not None or self.chunk_id is not None
        ):
            raise ValueError("Document metadata is only valid for document evidence")
        return self


class EvidenceConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topic: StateText
    first_claim: StateText
    first_source: StateText
    second_claim: StateText
    second_source: StateText
    status: Literal["unresolved", "resolved"] = "unresolved"


class ProcessStep(BaseModel):
    """One supported activity. Ordering is the order in ProcessFlow.steps."""

    model_config = ConfigDict(extra="forbid")

    step_id: StateIdentifier
    description: StateText
    actor: StateText | None = None
    system: StateText | None = None


class ProcessBranch(BaseModel):
    """An explicit route and its ordered step sequence."""

    model_config = ConfigDict(extra="forbid")

    condition: StateText
    next_step_ids: list[StateIdentifier] = Field(min_length=1)

    @field_validator("next_step_ids")
    @classmethod
    def unique_step_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("branch next_step_ids must be unique")
        return values


class ProcessDecision(BaseModel):
    """A supported decision and the routes known to follow it."""

    model_config = ConfigDict(extra="forbid")

    decision_id: StateIdentifier
    question: StateText
    after_step_id: StateIdentifier | None = None
    branches: list[ProcessBranch] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_branches(self) -> "ProcessDecision":
        conditions = [branch.condition.casefold() for branch in self.branches]
        if len(conditions) != len(set(conditions)):
            raise ValueError("decision branch conditions must be unique")
        return self


class ProcessFlow(BaseModel):
    """Canonical lightweight AS-IS flow; no parallel flat step/decision representation exists."""

    model_config = ConfigDict(extra="forbid")

    steps: list[ProcessStep] = Field(default_factory=list)
    decisions: list[ProcessDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "ProcessFlow":
        step_ids = [step.step_id for step in self.steps]
        decision_ids = [decision.decision_id for decision in self.decisions]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("step_id values must be unique")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("decision_id values must be unique")
        known_steps = set(step_ids)
        for decision in self.decisions:
            if decision.after_step_id is not None and decision.after_step_id not in known_steps:
                raise ValueError("decision after_step_id must reference a known step")
            referenced = {
                step_id for branch in decision.branches for step_id in branch.next_step_ids
            }
            if not referenced <= known_steps:
                raise ValueError("decision branches must reference known steps")
        return self


class TextReplacement(BaseModel):
    """An exact, evidence-supported replacement; Python never performs fuzzy matching."""

    model_config = ConfigDict(extra="forbid")

    old_value: StateText
    new_value: StateText


class ProcessStepReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_step_id: StateIdentifier
    replacement: ProcessStep

    @model_validator(mode="after")
    def preserve_identifier(self) -> "ProcessStepReplacement":
        if self.replacement.step_id != self.target_step_id:
            raise ValueError("a replacement must preserve its target step_id")
        return self


class ProcessDecisionReplacement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_decision_id: StateIdentifier
    replacement: ProcessDecision

    @model_validator(mode="after")
    def preserve_identifier(self) -> "ProcessDecisionReplacement":
        if self.replacement.decision_id != self.target_decision_id:
            raise ValueError("a replacement must preserve its target decision_id")
        return self


def _deduplicate_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(value)
    return unique


def _legacy_step(value: str, index: int) -> dict[str, Any]:
    return {"step_id": f"step-{index:03d}", "description": value}


def _legacy_decision(value: str, index: int) -> dict[str, Any]:
    return {"decision_id": f"decision-{index:03d}", "question": value, "branches": []}


class ProcessState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_name: StateText | None = None
    actors: list[StateText] = Field(default_factory=list)
    systems: list[StateText] = Field(default_factory=list)
    flow: ProcessFlow = Field(default_factory=ProcessFlow)
    pain_points: list[StateText] = Field(default_factory=list)
    unknowns: list[StateText] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_flat_flow(cls, value: Any) -> Any:
        """Accept persisted v1 states, then expose only the canonical v2 flow."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_steps = data.pop("steps", None)
        legacy_decisions = data.pop("decisions", None)
        if "flow" not in data and (legacy_steps is not None or legacy_decisions is not None):
            data["flow"] = {
                "steps": [
                    item if isinstance(item, dict) else _legacy_step(item, index)
                    for index, item in enumerate(legacy_steps or [], start=1)
                ],
                "decisions": [
                    item if isinstance(item, dict) else _legacy_decision(item, index)
                    for index, item in enumerate(legacy_decisions or [], start=1)
                ],
            }
        return data

    @field_validator("actors", "systems", "pain_points", "unknowns")
    @classmethod
    def deduplicate_entries(cls, values: list[str]) -> list[str]:
        return _deduplicate_strings(values)

    @property
    def steps(self) -> list[str]:
        """Read-only compatibility projection for v1 consumers."""
        return [step.description for step in self.flow.steps]

    @property
    def decisions(self) -> list[str]:
        """Read-only compatibility projection for v1 consumers."""
        return [decision.question for decision in self.flow.decisions]


class ProcessStatePatch(BaseModel):
    """A typed model-proposed change set. Python remains the sole state owner."""

    model_config = ConfigDict(extra="forbid")

    process_name: StateText | None = None
    process_name_to_replace: TextReplacement | None = None
    actors_to_add: list[StateText] = Field(default_factory=list)
    actors_to_replace: list[TextReplacement] = Field(default_factory=list)
    systems_to_add: list[StateText] = Field(default_factory=list)
    systems_to_replace: list[TextReplacement] = Field(default_factory=list)
    steps_to_add: list[ProcessStep] = Field(default_factory=list)
    steps_to_replace: list[ProcessStepReplacement] = Field(default_factory=list)
    decisions_to_add: list[ProcessDecision] = Field(default_factory=list)
    decisions_to_replace: list[ProcessDecisionReplacement] = Field(default_factory=list)
    pain_points_to_add: list[StateText] = Field(default_factory=list)
    unknowns_to_add: list[StateText] = Field(default_factory=list)
    unknowns_to_remove: list[StateText] = Field(default_factory=list)
    evidence_to_add: list[Evidence] = Field(default_factory=list)
    conflicts_to_add: list[EvidenceConflict] = Field(default_factory=list)

    @field_validator(
        "actors_to_add", "systems_to_add", "pain_points_to_add", "unknowns_to_add", "unknowns_to_remove"
    )
    @classmethod
    def deduplicate_entries(cls, values: list[str]) -> list[str]:
        return _deduplicate_strings(values)

    @model_validator(mode="after")
    def unique_operation_targets(self) -> "ProcessStatePatch":
        target_groups = (
            [item.old_value.casefold() for item in self.actors_to_replace],
            [item.old_value.casefold() for item in self.systems_to_replace],
            [item.target_step_id for item in self.steps_to_replace],
            [item.target_decision_id for item in self.decisions_to_replace],
        )
        if any(len(targets) != len(set(targets)) for targets in target_groups):
            raise ValueError("a patch may replace each target only once")
        return self


def _merge_strings(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing)
    seen = {value.casefold() for value in existing}
    for value in additions:
        if value.casefold() not in seen:
            seen.add(value.casefold())
            merged.append(value)
    return merged


def _replace_strings(existing: list[str], replacements: list[TextReplacement]) -> list[str]:
    result = list(existing)
    for replacement in replacements:
        matches = [
            index for index, value in enumerate(result)
            if value.casefold() == replacement.old_value.casefold()
        ]
        if len(matches) != 1:
            raise ValueError(f"replacement target not found exactly once: {replacement.old_value}")
        result[matches[0]] = replacement.new_value
    return _deduplicate_strings(result)


def _evidence_key(evidence: Evidence) -> tuple[str, str, str]:
    return (evidence.source_type.casefold(), evidence.source.casefold(), evidence.claim.casefold())


def _conflict_key(conflict: EvidenceConflict) -> tuple[str, tuple[tuple[str, str], ...]]:
    sides = sorted(
        (
            (conflict.first_claim.casefold(), conflict.first_source.casefold()),
            (conflict.second_claim.casefold(), conflict.second_source.casefold()),
        )
    )
    return conflict.topic.casefold(), tuple(sides)


_GENERIC_PROCESS_NAMES = {"not named yet", "unnamed process", "unknown process", "process"}


def apply_process_state_patch(
    state: ProcessState, patch: ProcessStatePatch | dict[str, Any]
) -> ProcessState:
    """Validate and atomically apply exact operations without mutating the supplied state."""

    validated_state = ProcessState.model_validate(state)
    validated_patch = ProcessStatePatch.model_validate(patch)
    data = validated_state.model_dump()

    current_name = validated_state.process_name
    if validated_patch.process_name_to_replace is not None:
        replacement = validated_patch.process_name_to_replace
        if current_name is None or current_name.casefold() != replacement.old_value.casefold():
            raise ValueError("process-name replacement target does not match current state")
        data["process_name"] = replacement.new_value
    elif validated_patch.process_name is not None and (
        current_name is None or current_name.casefold() in _GENERIC_PROCESS_NAMES
    ):
        data["process_name"] = validated_patch.process_name

    data["actors"] = _merge_strings(
        _replace_strings(data["actors"], validated_patch.actors_to_replace),
        validated_patch.actors_to_add,
    )
    data["systems"] = _merge_strings(
        _replace_strings(data["systems"], validated_patch.systems_to_replace),
        validated_patch.systems_to_add,
    )
    data["pain_points"] = _merge_strings(data["pain_points"], validated_patch.pain_points_to_add)

    resolved_unknowns = {value.casefold() for value in validated_patch.unknowns_to_remove}
    remaining_unknowns = [
        value for value in data["unknowns"] if value.casefold() not in resolved_unknowns
    ]
    data["unknowns"] = _merge_strings(remaining_unknowns, validated_patch.unknowns_to_add)

    steps = [ProcessStep.model_validate(item) for item in data["flow"]["steps"]]
    step_indexes = {step.step_id: index for index, step in enumerate(steps)}
    for operation in validated_patch.steps_to_replace:
        if operation.target_step_id not in step_indexes:
            raise ValueError(f"step replacement target does not exist: {operation.target_step_id}")
        steps[step_indexes[operation.target_step_id]] = operation.replacement
    known_step_ids = {step.step_id for step in steps}
    for step in validated_patch.steps_to_add:
        if step.step_id in known_step_ids:
            raise ValueError(f"step_id already exists: {step.step_id}")
        known_step_ids.add(step.step_id)
        steps.append(step)

    decisions = [ProcessDecision.model_validate(item) for item in data["flow"]["decisions"]]
    decision_indexes = {
        decision.decision_id: index for index, decision in enumerate(decisions)
    }
    for operation in validated_patch.decisions_to_replace:
        if operation.target_decision_id not in decision_indexes:
            raise ValueError(
                f"decision replacement target does not exist: {operation.target_decision_id}"
            )
        decisions[decision_indexes[operation.target_decision_id]] = operation.replacement
    known_decision_ids = {decision.decision_id for decision in decisions}
    for decision in validated_patch.decisions_to_add:
        if decision.decision_id in known_decision_ids:
            raise ValueError(f"decision_id already exists: {decision.decision_id}")
        known_decision_ids.add(decision.decision_id)
        decisions.append(decision)
    data["flow"] = {
        "steps": [step.model_dump() for step in steps],
        "decisions": [decision.model_dump() for decision in decisions],
    }

    evidence = [Evidence.model_validate(item) for item in data["evidence"]]
    evidence_keys = {_evidence_key(item) for item in evidence}
    for item in validated_patch.evidence_to_add:
        if _evidence_key(item) not in evidence_keys:
            evidence_keys.add(_evidence_key(item))
            evidence.append(item)
    data["evidence"] = [item.model_dump() for item in evidence]

    conflicts = [EvidenceConflict.model_validate(item) for item in data["conflicts"]]
    conflict_keys = {_conflict_key(item) for item in conflicts}
    for item in validated_patch.conflicts_to_add:
        if _conflict_key(item) not in conflict_keys:
            conflict_keys.add(_conflict_key(item))
            conflicts.append(item)
    data["conflicts"] = [item.model_dump() for item in conflicts]

    return ProcessState.model_validate(data)


def effective_patch(before: ProcessState, after: ProcessState) -> ProcessStatePatch:
    """Describe additions/removals and same-ID refinements that were actually applied."""

    def additions(old: list[str], new: list[str]) -> list[str]:
        old_keys = {value.casefold() for value in old}
        return [value for value in new if value.casefold() not in old_keys]

    def text_changes(old: list[str], new: list[str]) -> tuple[list[TextReplacement], list[str]]:
        old_keys = {value.casefold() for value in old}
        new_keys = {value.casefold() for value in new}
        replacements = [
            TextReplacement(old_value=old_value, new_value=new_value)
            for old_value, new_value in zip(old, new)
            if old_value.casefold() != new_value.casefold()
            and old_value.casefold() not in new_keys
            and new_value.casefold() not in old_keys
        ]
        replacement_values = {item.new_value.casefold() for item in replacements}
        added = [
            value
            for value in additions(old, new)
            if value.casefold() not in replacement_values
        ]
        return replacements, added

    before_steps = {step.step_id: step for step in before.flow.steps}
    after_steps = {step.step_id: step for step in after.flow.steps}
    before_decisions = {item.decision_id: item for item in before.flow.decisions}
    after_decisions = {item.decision_id: item for item in after.flow.decisions}
    before_evidence = {_evidence_key(item) for item in before.evidence}
    before_conflicts = {_conflict_key(item) for item in before.conflicts}
    after_unknowns = {value.casefold() for value in after.unknowns}
    actor_replacements, actor_additions = text_changes(before.actors, after.actors)
    system_replacements, system_additions = text_changes(before.systems, after.systems)
    return ProcessStatePatch(
        process_name=(after.process_name if before.process_name is None else None),
        process_name_to_replace=(
            TextReplacement(old_value=before.process_name, new_value=after.process_name)
            if before.process_name is not None
            and after.process_name is not None
            and before.process_name != after.process_name
            else None
        ),
        actors_to_add=actor_additions,
        actors_to_replace=actor_replacements,
        systems_to_add=system_additions,
        systems_to_replace=system_replacements,
        steps_to_add=[step for key, step in after_steps.items() if key not in before_steps],
        steps_to_replace=[
            ProcessStepReplacement(target_step_id=key, replacement=step)
            for key, step in after_steps.items()
            if key in before_steps and step != before_steps[key]
        ],
        decisions_to_add=[item for key, item in after_decisions.items() if key not in before_decisions],
        decisions_to_replace=[
            ProcessDecisionReplacement(target_decision_id=key, replacement=item)
            for key, item in after_decisions.items()
            if key in before_decisions and item != before_decisions[key]
        ],
        pain_points_to_add=additions(before.pain_points, after.pain_points),
        unknowns_to_add=additions(before.unknowns, after.unknowns),
        unknowns_to_remove=[
            value for value in before.unknowns if value.casefold() not in after_unknowns
        ],
        evidence_to_add=[item for item in after.evidence if _evidence_key(item) not in before_evidence],
        conflicts_to_add=[item for item in after.conflicts if _conflict_key(item) not in before_conflicts],
    )
