import math
from pathlib import Path

from backend.config import OPENAI_EMBEDDING_MODEL, RAG_INDEX_PATH
from backend.rag.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from backend.rag.index import (
    DOCUMENTS_DIR,
    IndexStaleError,
    IndexUnavailableError,
    load_compatible_index,
    load_index,
    source_documents,
    stale_reason,
)
from backend.rag.models import (
    KnowledgeCatalogDocument,
    RAGStatus,
    SearchHit,
    SearchResult,
)


DOCUMENT_TOPICS = {
    "discount_policy.md": "discount approval rules, approver roles, required controls",
    "quote_process.md": "quote preparation, approvals, issuance and process recording",
    "product_guidelines.md": "operational product guidance and quote-related handling",
}


def cosine_similarity(first: list[float], second: list[float]) -> float:
    if len(first) != len(second):
        raise ValueError("Embedding dimensions do not match")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


class RAGService:
    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        documents_dir: Path = DOCUMENTS_DIR,
        index_path: Path = RAG_INDEX_PATH,
    ):
        self.provider = provider
        self.documents_dir = documents_dir
        self.index_path = index_path

    def status(self) -> RAGStatus:
        try:
            index = load_index(self.index_path)
        except IndexUnavailableError as exc:
            return RAGStatus(
                available=False,
                embedding_model=self.provider.model,
                documents=len(source_documents(self.documents_dir)),
                chunks=0,
                stale=False,
                reason=str(exc),
            )
        reason = stale_reason(index, self.provider.model, self.documents_dir)
        return RAGStatus(
            available=reason is None,
            embedding_model=self.provider.model,
            documents=len(index.metadata.documents),
            chunks=len(index.chunks),
            stale=reason is not None,
            reason=reason,
        )

    def knowledge_catalog(self) -> list[KnowledgeCatalogDocument]:
        """Return safe catalogue metadata derived from the current persisted index."""
        index = load_index(self.index_path)
        documents: dict[str, dict[str, str | int]] = {}
        for chunk in index.chunks:
            entry = documents.setdefault(
                chunk.source,
                {
                    "title": chunk.document_title,
                    "filename": chunk.source,
                    "chunk_count": 0,
                    "topics": DOCUMENT_TOPICS.get(
                        chunk.source,
                        "indexed Northstar company knowledge",
                    ),
                },
            )
            entry["chunk_count"] = int(entry["chunk_count"]) + 1
        return [
            KnowledgeCatalogDocument.model_validate(documents[source])
            for source in sorted(documents, key=str.casefold)
        ]

    def search_company_knowledge(self, query: str, top_k: int = 3) -> SearchResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Search query cannot be blank")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k must be between 1 and 10")
        # Compatibility is checked before the paid query-embedding call.
        index = load_compatible_index(
            self.provider.model,
            documents_dir=self.documents_dir,
            index_path=self.index_path,
        )
        vectors = self.provider.embed([normalized_query])
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("Embedding provider did not return one query vector")
        query_vector = vectors[0]
        ranked = sorted(
            (
                (cosine_similarity(query_vector, chunk.embedding), chunk)
                for chunk in index.chunks
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]
        hits = [
            SearchHit(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                source=chunk.source,
                document_title=chunk.document_title,
                chunk_number=chunk.chunk_number,
                heading=chunk.heading,
                text=chunk.text,
                similarity_score=score,
                metadata=chunk.metadata,
            )
            for score, chunk in ranked
        ]
        return SearchResult(
            query=normalized_query,
            embedding_model=index.metadata.embedding_model,
            index_chunk_count=len(index.chunks),
            top_k=top_k,
            hits=hits,
        )


_default_service = RAGService(
    OpenAIEmbeddingProvider(OPENAI_EMBEDDING_MODEL), index_path=RAG_INDEX_PATH
)


def search_company_knowledge(query: str, top_k: int = 3) -> SearchResult:
    return _default_service.search_company_knowledge(query, top_k)
