"""Small Markdown-aware chunking with deterministic source identities.

Headings define semantic sections and adjacent paragraphs within a section are
combined up to ``max_chars``. This keeps policy clauses together without making
an entire document one opaque chunk or splitting every sentence independently.
"""

import hashlib
import re
from pathlib import Path

from backend.rag.models import IndexedChunk


HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_markdown(path: Path, *, max_chars: int = 900) -> list[IndexedChunk]:
    text = path.read_text(encoding="utf-8").strip()
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    title = path.stem.replace("_", " ").title()
    heading: str | None = None
    sections: list[tuple[str | None, list[str]]] = []
    paragraphs: list[str] = []

    def finish_section() -> None:
        nonlocal paragraphs
        if paragraphs:
            sections.append((heading, paragraphs))
            paragraphs = []

    for block in blocks:
        match = HEADING_PATTERN.match(block)
        if match:
            level = len(match.group(1))
            value = match.group(2).strip()
            if level == 1:
                title = value
                continue
            finish_section()
            heading = value
        else:
            paragraphs.append(block)
    finish_section()

    source = path.name
    chunks: list[IndexedChunk] = []
    for section_heading, section_paragraphs in sections:
        grouped: list[str] = []
        length = 0
        for paragraph in section_paragraphs:
            added_length = len(paragraph) + (2 if grouped else 0)
            if grouped and length + added_length > max_chars:
                chunks.append(
                    _make_chunk(path, source, title, section_heading, grouped, len(chunks) + 1)
                )
                grouped = []
                length = 0
            grouped.append(paragraph)
            length += added_length
        if grouped:
            chunks.append(
                _make_chunk(path, source, title, section_heading, grouped, len(chunks) + 1)
            )
    return chunks


def _make_chunk(
    path: Path,
    source: str,
    title: str,
    heading: str | None,
    paragraphs: list[str],
    chunk_number: int,
) -> IndexedChunk:
    body = "\n\n".join(paragraphs)
    chunk_text = f"## {heading}\n\n{body}" if heading else body
    identity = f"{source}\0{heading or ''}\0{chunk_text}".encode("utf-8")
    chunk_id = f"{path.stem}-{hashlib.sha256(identity).hexdigest()[:16]}"
    return IndexedChunk(
        chunk_id=chunk_id,
        document_id=path.stem,
        source=source,
        document_title=title,
        chunk_number=chunk_number,
        heading=heading,
        text=chunk_text,
        embedding=[],
        metadata={"format": "markdown"},
    )
