"""Configurable, offset-preserving document chunking."""

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class Chunk:
    doc_path: str
    start: int
    end: int
    text: str


def _fixed(text: str, doc_path: str, size: int, overlap: int) -> Iterable[Chunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk_size must be positive and overlap must be in [0, chunk_size)")
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        yield Chunk(doc_path, start, end, text[start:end])
        if end == len(text):
            break
        start = end - overlap


def _recursive(text: str, doc_path: str, size: int, overlap: int) -> Iterable[Chunk]:
    if size <= 0:
        raise ValueError("chunk_size must be positive")
    boundaries = [m.end() for m in re.finditer(r"\n\s*\n|(?<=[.!?])\s+", text)]
    boundaries = sorted(set([0, *boundaries, len(text)]))
    start = 0
    while start < len(text):
        limit = min(start + size, len(text))
        candidates = [b for b in boundaries if start < b <= limit]
        end = max(candidates) if candidates else limit
        yield Chunk(doc_path, start, end, text[start:end])
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        start = next_start


def chunk_documents(
    documents: dict[str, str], config: dict
) -> list[Chunk]:
    """Chunk documents according to ``config['strategy']``."""
    strategy = config.get("strategy", "fixed")
    size = int(config.get("chunk_size", 500))
    overlap = int(config.get("overlap", 0))
    chunks: list[Chunk] = []
    for path, text in documents.items():
        maker = _fixed if strategy == "fixed" else _recursive if strategy == "recursive" else None
        if maker is None:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
        chunks.extend(maker(text, path, size, overlap))
    if any(chunk.text != documents[chunk.doc_path][chunk.start:chunk.end] for chunk in chunks):
        raise AssertionError("Chunk text does not match its document offsets")
    return chunks
