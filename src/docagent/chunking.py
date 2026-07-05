from __future__ import annotations

from docagent.models import Chunk, Document


def estimate_tokens(text: str) -> int:
    # Conservative approximation that works without tokenizer-specific deps.
    return max(1, int(len(text) / 4))


def chunk_document(document: Document, chunk_size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens cannot be negative")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")

    chunk_chars = chunk_size_tokens * 4
    overlap_chars = overlap_tokens * 4
    text = document.text
    chunks: list[Chunk] = []
    start = 0
    sequence = 1

    while start < len(text):
        end = min(len(text), start + chunk_chars)
        if end < len(text):
            boundary = _find_boundary(text, start, end)
            if boundary > start:
                end = boundary
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}_chunk_{sequence:04d}",
                    sequence=sequence,
                    text=chunk_text,
                    char_start=start,
                    char_end=end,
                )
            )
            sequence += 1
        if end >= len(text):
            break
        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _find_boundary(text: str, start: int, end: int) -> int:
    window_start = max(start, end - 800)
    for marker in ("\n\n", "\n", ". "):
        pos = text.rfind(marker, window_start, end)
        if pos > start:
            return pos + len(marker)
    return end
