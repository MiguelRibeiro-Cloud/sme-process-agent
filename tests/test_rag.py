import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import app
from backend.agent import (
    DiscoverySession,
    EXTRACTION_INSTRUCTIONS,
    RAG_GROUNDING_INSTRUCTIONS,
    RAG_TOOL_NAME,
)
from backend.config import (
    DEFAULT_OPENAI_EMBEDDING_MODEL,
    DEFAULT_RAG_INDEX_PATH,
    resolve_openai_embedding_model,
    resolve_project_path,
)
from backend.rag.chunking import chunk_markdown
from backend.rag.index import (
    IndexStaleError,
    build_index,
    export_release_index,
    load_index,
    stale_reason,
)
from backend.rag.models import SearchHit, SearchResult
from backend.rag.service import RAGService
from backend.state import Evidence, EvidenceConflict, ProcessState, ProcessStatePatch


MCP_TOOLS = [
    {
        "name": "crm_get_customer",
        "description": "Look up a customer.",
        "inputSchema": {
            "type": "object",
            "properties": {"customer_name": {"type": "string"}},
            "required": ["customer_name"],
            "additionalProperties": False,
        },
    }
]


class FakeEmbeddingProvider:
    def __init__(self, model="test-embedding-model", query_vector=None):
        self.model = model
        self.query_vector = query_vector or [1.0, 0.0]
        self.calls = []

    def embed(self, texts):
        values = list(texts)
        self.calls.append(values)
        vectors = []
        for value in values:
            if value == "alpha query":
                vectors.append(self.query_vector)
            elif "Alpha" in value:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def write_corpus(directory: Path) -> None:
    (directory / "alpha.md").write_text(
        "# Alpha Handbook\n\n## Approval\n\nAlpha approval guidance.\n",
        encoding="utf-8",
    )
    (directory / "beta.md").write_text(
        "# Beta Handbook\n\n## Delivery\n\nBeta delivery guidance.\n",
        encoding="utf-8",
    )


def rag_result() -> SearchResult:
    return SearchResult(
        query="discount approval thresholds",
        embedding_model="text-embedding-3-small",
        index_chunk_count=12,
        top_k=3,
        hits=[
            SearchHit(
                chunk_id="discount-policy-abc",
                document_id="discount_policy",
                source="discount_policy.md",
                document_title="Commercial Discount Policy",
                chunk_number=2,
                heading="Sales Manager approval",
                text="Discounts above 10% require Sales Manager approval.",
                similarity_score=0.923,
            )
        ],
    )


def function_call(name=RAG_TOOL_NAME, arguments=None):
    return SimpleNamespace(
        type="function_call",
        name=name,
        arguments=arguments or json.dumps({"query": "discount approval thresholds"}),
        call_id="rag-call-1",
    )


def response(response_id="response-1", text="", output=None):
    return SimpleNamespace(id=response_id, output_text=text, output=output or [])


class ChunkingAndIndexTests(unittest.TestCase):
    def test_runtime_index_path_is_configurable_and_project_relative(self):
        self.assertEqual(
            resolve_project_path("RAG_INDEX_PATH", DEFAULT_RAG_INDEX_PATH, {}),
            DEFAULT_RAG_INDEX_PATH,
        )
        self.assertEqual(
            resolve_project_path(
                "RAG_INDEX_PATH",
                DEFAULT_RAG_INDEX_PATH,
                {"RAG_INDEX_PATH": "deployment/northstar-index.json"},
            ),
            DEFAULT_RAG_INDEX_PATH.parents[1] / "deployment" / "northstar-index.json",
        )

    def test_embedding_model_configuration_has_a_trimmed_default(self):
        self.assertEqual(resolve_openai_embedding_model({}), DEFAULT_OPENAI_EMBEDDING_MODEL)
        self.assertEqual(
            resolve_openai_embedding_model({"OPENAI_EMBEDDING_MODEL": "  test-model  "}),
            "test-model",
        )

    def test_markdown_chunking_is_stable_and_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "policy.md"
            path.write_text(
                "# Policy Title\n\n## Standard\n\nFirst paragraph.\n\nSecond paragraph.\n\n"
                "## Exception\n\nException paragraph.\n",
                encoding="utf-8",
            )
            first = chunk_markdown(path)
            second = chunk_markdown(path)

        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertEqual(len(first), 2)
        self.assertEqual(first[0].document_title, "Policy Title")
        self.assertEqual(first[0].source, "policy.md")
        self.assertEqual(first[1].chunk_number, 2)
        self.assertIn("First paragraph.\n\nSecond paragraph.", first[0].text)

    def test_index_persists_loads_and_detects_document_or_model_staleness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            documents.mkdir()
            write_corpus(documents)
            index_path = root / "index.json"
            provider = FakeEmbeddingProvider()
            built = build_index(provider, documents_dir=documents, index_path=index_path)
            loaded = load_index(index_path)

            self.assertEqual(loaded, built)
            self.assertEqual(stale_reason(loaded, provider.model, documents), None)
            self.assertEqual(stale_reason(loaded, "different-model", documents), "embedding model changed")
            (documents / "alpha.md").write_text("# Changed", encoding="utf-8")
            self.assertEqual(stale_reason(loaded, provider.model, documents), "source documents changed")

    def test_release_export_validates_existing_index_without_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            documents.mkdir()
            write_corpus(documents)
            source = root / ".rag" / "index.json"
            destination = root / "deployment" / "index.json"
            provider = FakeEmbeddingProvider()
            build_index(provider, documents_dir=documents, index_path=source)
            provider.calls.clear()

            exported = export_release_index(
                source_path=source,
                destination_path=destination,
                embedding_model=provider.model,
                documents_dir=documents,
            )

            self.assertEqual(load_index(destination), exported)
            self.assertEqual(provider.calls, [])

    def test_missing_runtime_index_reports_unavailable_without_auto_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            provider = FakeEmbeddingProvider()
            service = RAGService(
                provider,
                documents_dir=Path(temporary),
                index_path=Path(temporary) / "missing.json",
            )
            with patch("backend.rag.index.build_index") as auto_build:
                status = service.status()

        self.assertFalse(status.available)
        self.assertFalse(status.stale)
        self.assertEqual(provider.calls, [])
        auto_build.assert_not_called()

    def test_cosine_search_ranks_fabricated_vectors_and_preserves_similarity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            documents.mkdir()
            write_corpus(documents)
            index_path = root / "index.json"
            provider = FakeEmbeddingProvider()
            build_index(provider, documents_dir=documents, index_path=index_path)
            service = RAGService(provider, documents_dir=documents, index_path=index_path)

            result = service.search_company_knowledge("alpha query", top_k=2)

        self.assertEqual(result.hits[0].document_title, "Alpha Handbook")
        self.assertEqual(result.hits[0].similarity_score, 1.0)
        self.assertEqual(result.hits[1].similarity_score, 0.0)
        self.assertNotIn("confidence", result.model_dump_json().lower())

    def test_stale_index_is_rejected_before_paid_query_embedding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "documents"
            documents.mkdir()
            write_corpus(documents)
            index_path = root / "index.json"
            build_index(FakeEmbeddingProvider("old-model"), documents_dir=documents, index_path=index_path)
            current = FakeEmbeddingProvider("new-model")
            service = RAGService(current, documents_dir=documents, index_path=index_path)

            with self.assertRaises(IndexStaleError):
                service.search_company_knowledge("alpha query")

        self.assertEqual(current.calls, [])


class NorthstarKnowledgeCatalogTests(unittest.TestCase):
    def test_catalog_is_derived_from_index_without_exposing_content_or_vectors(self):
        documents = app.rag_service.knowledge_catalog()
        payload = [document.model_dump() for document in documents]

        self.assertEqual(
            [(item["title"], item["filename"], item["chunk_count"]) for item in payload],
            [
                ("Commercial Discount Policy", "discount_policy.md", 4),
                ("Product Operations Guidelines", "product_guidelines.md", 5),
                ("Northstar Quote Process", "quote_process.md", 5),
            ],
        )
        serialized = json.dumps(payload)
        self.assertNotIn("embedding", serialized.casefold())
        self.assertNotIn("10%", serialized)
        self.assertNotIn("Standard customer quotes", serialized)

    def test_catalog_endpoint_returns_only_safe_document_metadata(self):
        result = app.get_knowledge_catalog()

        self.assertTrue(result["available"])
        self.assertEqual(len(result["documents"]), 3)
        for document in result["documents"]:
            self.assertEqual(
                set(document.model_dump()),
                {"title", "filename", "chunk_count", "topics"},
            )


class AgentRAGRoutingTests(unittest.TestCase):
    def setUp(self):
        self.mcp = Mock()
        self.mcp.discover_tools.return_value = MCP_TOOLS
        self.rag = Mock()
        self.rag.search_company_knowledge.return_value = rag_result()

    def openai(self, outputs, extraction_patch=None):
        client = Mock()
        client.responses.create.side_effect = outputs
        client.responses.parse.return_value = SimpleNamespace(
            output_parsed=extraction_patch or ProcessStatePatch()
        )
        return client

    def test_rag_schema_is_supplied_alongside_mcp_and_no_retrieval_when_unselected(self):
        client = self.openai([response(text="What happens next?")])
        session = DiscoverySession(self.mcp, rag_service=self.rag)

        list(session.event_stream("Quotes take time", client))

        tools = client.responses.create.call_args.kwargs["tools"]
        self.assertEqual([tool["name"] for tool in tools], ["crm_get_customer", RAG_TOOL_NAME])
        self.rag.search_company_knowledge.assert_not_called()
        self.mcp.call_tool.assert_not_called()

    def test_luna_can_select_rag_and_receives_explicit_grounded_evidence(self):
        client = self.openai(
            [
                response(output=[function_call()]),
                response(response_id="response-2", text="Policy requires approval above 10%."),
            ]
        )
        session = DiscoverySession(self.mcp, rag_service=self.rag)

        events = list(session.event_stream("What is the discount policy?", client))

        self.rag.search_company_knowledge.assert_called_once_with("discount approval thresholds", 3)
        self.mcp.call_tool.assert_not_called()
        grounded = client.responses.create.call_args_list[1].kwargs["input"][0]["output"]
        self.assertIn("Retrieved company evidence", grounded)
        self.assertIn("Commercial Discount Policy | discount_policy.md | chunk 2", grounded)
        extraction = json.loads(client.responses.parse.call_args.kwargs["input"])
        self.assertEqual(extraction["actual_rag_retrievals"][0]["hits"][0]["chunk_id"], "discount-policy-abc")
        self.assertEqual(extraction["actual_tool_calls"], [])

        event_types = [event["type"] for event in events]
        self.assertLess(event_types.index("rag_retrieval_started"), event_types.index("rag_query_embedded"))
        self.assertLess(event_types.index("rag_query_embedded"), event_types.index("rag_retrieval_completed"))
        completed = next(event for event in events if event["type"] == "rag_retrieval_completed")
        self.assertEqual(completed["embedding_model"], "text-embedding-3-small")
        self.assertEqual(completed["index_chunk_count"], 12)
        self.assertEqual(completed["results"][0]["similarity_score"], 0.923)
        self.assertNotIn("confidence", json.dumps(completed).lower())
        self.assertNotIn("embedding", completed["results"][0])

    def test_policy_grounding_instructions_require_general_rule_then_case_implication(self):
        client = self.openai([
            response(output=[function_call()]),
            response(response_id="response-2", text="A grounded policy answer."),
        ])
        session = DiscoverySession(self.mcp, rag_service=self.rag)

        list(session.event_stream("What does policy say about this case?", client))

        instructions = client.responses.create.call_args_list[1].kwargs["instructions"]
        self.assertIn("governing documented rule", instructions)
        self.assertIn("user's specific case", instructions)
        self.assertIn("approval limits", RAG_GROUNDING_INSTRUCTIONS)

    def test_supported_threshold_disagreement_persists_both_evidence_and_conflict(self):
        user_evidence = Evidence(
            source_type="user",
            source="user",
            claim="Approval is requested only for discounts above 15%.",
        )
        document_evidence = Evidence(
            source_type="document",
            source="discount_policy.md",
            claim="Discounts above 10% require Sales Manager approval.",
            document_title="Commercial Discount Policy",
            chunk_id="discount-policy-abc",
        )
        patch_value = ProcessStatePatch(
            evidence_to_add=[document_evidence],
            conflicts_to_add=[
                EvidenceConflict(
                    topic="Discount approval threshold",
                    first_claim=user_evidence.claim,
                    first_source="user",
                    second_claim=document_evidence.claim,
                    second_source="Commercial Discount Policy",
                )
            ],
        )
        client = self.openai(
            [
                response(output=[function_call()]),
                response(response_id="response-2", text="The threshold rules differ."),
            ],
            extraction_patch=patch_value,
        )
        session = DiscoverySession(
            self.mcp,
            rag_service=self.rag,
            initial_state=ProcessState(evidence=[user_evidence]),
        )

        list(session.event_stream("Compare our practice with policy.", client))

        state = session.state_snapshot()
        self.assertEqual([item.source_type for item in state.evidence], ["user", "document"])
        self.assertEqual(1, len(state.conflicts))
        self.assertEqual("unresolved", state.conflicts[0].status)
        self.assertIn("different thresholds", EXTRACTION_INSTRUCTIONS)

    def test_possible_owner_compatibility_remains_unknown_not_conflict(self):
        user_evidence = Evidence(
            source_type="user",
            source="user",
            claim="Marta approves discounts.",
        )
        document_evidence = Evidence(
            source_type="document",
            source="discount_policy.md",
            claim="The Sales Manager must approve discounts above the policy threshold.",
            document_title="Commercial Discount Policy",
            chunk_id="discount-policy-abc",
        )
        patch_value = ProcessStatePatch(
            evidence_to_add=[document_evidence],
            unknowns_to_add=["Whether Marta is the Sales Manager"],
        )
        client = self.openai(
            [
                response(output=[function_call()]),
                response(response_id="response-2", text="Marta's role is not established."),
            ],
            extraction_patch=patch_value,
        )
        session = DiscoverySession(
            self.mcp,
            rag_service=self.rag,
            initial_state=ProcessState(evidence=[user_evidence]),
        )

        list(session.event_stream("Compare our practice with policy.", client))

        state = session.state_snapshot()
        self.assertEqual([], state.conflicts)
        self.assertIn("Whether Marta is the Sales Manager", state.unknowns)
        self.assertIn("add an unknown instead of a conflict", EXTRACTION_INSTRUCTIONS)


class RAGStatusEndpointTests(unittest.TestCase):
    def test_rag_status_exposes_safe_metadata_without_secrets(self):
        fake_service = Mock()
        fake_service.status.return_value = {
            "available": True,
            "embedding_model": "text-embedding-3-small",
            "documents": 3,
            "chunks": 12,
            "stale": False,
            "reason": None,
        }
        with patch.object(app, "rag_service", fake_service):
            result = app.get_rag_status()

        serialized = json.dumps(result)
        self.assertEqual(result["documents"], 3)
        self.assertNotIn("api_key", serialized.lower())
        self.assertNotIn("credential", serialized.lower())


if __name__ == "__main__":
    unittest.main()
