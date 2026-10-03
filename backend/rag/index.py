import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from backend.config import OPENAI_EMBEDDING_MODEL, PROJECT_ROOT
from backend.rag.chunking import chunk_markdown
from backend.rag.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from backend.rag.models import DocumentFingerprint, IndexMetadata, VectorIndex


INDEX_VERSION = 1
DOCUMENTS_DIR = PROJECT_ROOT / "backend" / "synthetic_company" / "documents"
LOCAL_INDEX_PATH = PROJECT_ROOT / ".rag" / "northstar-index.json"
RELEASE_INDEX_PATH = PROJECT_ROOT / "deployment" / "northstar-index.json"
# Backwards-compatible name for the explicit local developer build path.
INDEX_PATH = LOCAL_INDEX_PATH


class IndexUnavailableError(RuntimeError):
    pass


class IndexStaleError(RuntimeError):
    pass


def source_documents(documents_dir: Path = DOCUMENTS_DIR) -> list[Path]:
    return sorted(documents_dir.glob("*.md"), key=lambda path: path.name.casefold())


def document_fingerprints(
    documents_dir: Path = DOCUMENTS_DIR,
) -> list[DocumentFingerprint]:
    return [
        DocumentFingerprint(
            source=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in source_documents(documents_dir)
    ]


def build_index(
    provider: EmbeddingProvider,
    *,
    documents_dir: Path = DOCUMENTS_DIR,
    index_path: Path = INDEX_PATH,
) -> VectorIndex:
    chunks = [
        chunk
        for path in source_documents(documents_dir)
        for chunk in chunk_markdown(path)
    ]
    vectors = provider.embed([chunk.text for chunk in chunks])
    if len(vectors) != len(chunks):
        raise ValueError("Embedding provider returned a different number of vectors than chunks")
    for chunk, vector in zip(chunks, vectors, strict=True):
        if not vector:
            raise ValueError(f"Embedding provider returned an empty vector for {chunk.chunk_id}")
        chunk.embedding = vector
    index = VectorIndex(
        metadata=IndexMetadata(
            index_version=INDEX_VERSION,
            embedding_model=provider.model,
            built_at=datetime.now(timezone.utc),
            documents=document_fingerprints(documents_dir),
            chunk_count=len(chunks),
        ),
        chunks=chunks,
    )
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index.model_dump_json(indent=2), encoding="utf-8")
    return index


def load_index(index_path: Path = INDEX_PATH) -> VectorIndex:
    if not index_path.exists():
        raise IndexUnavailableError("The company-knowledge index has not been built")
    try:
        index = VectorIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, json.JSONDecodeError) as exc:
        raise IndexUnavailableError("The company-knowledge index cannot be loaded") from exc
    if index.metadata.chunk_count != len(index.chunks):
        raise IndexUnavailableError("The company-knowledge index has inconsistent metadata")
    return index


def stale_reason(
    index: VectorIndex,
    embedding_model: str,
    documents_dir: Path = DOCUMENTS_DIR,
) -> str | None:
    if index.metadata.index_version != INDEX_VERSION:
        return "index version changed"
    if index.metadata.embedding_model != embedding_model:
        return "embedding model changed"
    current = [item.model_dump() for item in document_fingerprints(documents_dir)]
    indexed = [item.model_dump() for item in index.metadata.documents]
    if current != indexed:
        return "source documents changed"
    return None


def load_compatible_index(
    embedding_model: str,
    *,
    documents_dir: Path = DOCUMENTS_DIR,
    index_path: Path = INDEX_PATH,
) -> VectorIndex:
    index = load_index(index_path)
    reason = stale_reason(index, embedding_model, documents_dir)
    if reason:
        raise IndexStaleError(reason)
    return index


def export_release_index(
    *,
    source_path: Path = LOCAL_INDEX_PATH,
    destination_path: Path = RELEASE_INDEX_PATH,
    embedding_model: str = OPENAI_EMBEDDING_MODEL,
    documents_dir: Path = DOCUMENTS_DIR,
) -> VectorIndex:
    """Validate and export an existing index without requesting embeddings."""
    index = load_compatible_index(
        embedding_model,
        documents_dir=documents_dir,
        index_path=source_path,
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text(
        index.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or export the Northstar RAG index.")
    parser.add_argument(
        "--export-release",
        action="store_true",
        help="Validate the local index and export the deployment artifact without API calls.",
    )
    args = parser.parse_args()
    if args.export_release:
        index = export_release_index()
        print(
            f"Exported {RELEASE_INDEX_PATH.relative_to(PROJECT_ROOT)} with "
            f"{len(index.metadata.documents)} documents and {len(index.chunks)} chunks "
            f"using {index.metadata.embedding_model}."
        )
        return
    provider = OpenAIEmbeddingProvider(OPENAI_EMBEDDING_MODEL)
    index = build_index(provider)
    print(
        f"Built {INDEX_PATH.relative_to(PROJECT_ROOT)} with "
        f"{len(index.metadata.documents)} documents and {len(index.chunks)} chunks "
        f"using {index.metadata.embedding_model}."
    )


if __name__ == "__main__":
    main()
