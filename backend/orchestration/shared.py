import hashlib

from backend.state import ProcessState


def process_state_fingerprint(snapshot: ProcessState) -> str:
    canonical = snapshot.model_dump_json()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evidence_catalog(snapshot: ProcessState) -> list[dict[str, object]]:
    """Assign deterministic IDs and explicit specialist-facing provenance labels."""
    provenance_labels = {
        "user": "user_reported_practice",
        "document": "documented_policy_or_procedure",
        "mcp": "mcp_system_fact",
    }
    return [
        {
            "evidence_id": f"evidence-{index:03d}",
            "provenance": provenance_labels[item.source_type],
            **item.model_dump(mode="json"),
        }
        for index, item in enumerate(snapshot.evidence, start=1)
    ]


def has_process_knowledge(snapshot: ProcessState) -> bool:
    return bool(
        snapshot.flow.steps
        or snapshot.flow.decisions
        or snapshot.pain_points
        or snapshot.evidence
        or snapshot.conflicts
    )
