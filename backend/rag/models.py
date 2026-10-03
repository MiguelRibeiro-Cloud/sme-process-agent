from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    sha256: str


class IndexedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    source: str
    document_title: str
    chunk_number: int = Field(ge=1)
    heading: str | None = None
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index_version: int
    embedding_model: str
    built_at: datetime
    documents: list[DocumentFingerprint]
    chunk_count: int = Field(ge=0)


class VectorIndex(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metadata: IndexMetadata
    chunks: list[IndexedChunk]


class SearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    source: str
    document_title: str
    chunk_number: int
    heading: str | None = None
    text: str
    similarity_score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    embedding_model: str
    index_chunk_count: int
    top_k: int
    hits: list[SearchHit]


class RAGStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool
    embedding_model: str
    documents: int
    chunks: int
    stale: bool
    reason: str | None = None


class KnowledgeCatalogDocument(BaseModel):
    """Safe, presentation-only metadata for one indexed synthetic document."""

    model_config = ConfigDict(extra="forbid")

    title: str
    filename: str
    chunk_count: int = Field(ge=0)
    topics: str
