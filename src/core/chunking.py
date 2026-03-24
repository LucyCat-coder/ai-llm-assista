"""Разбиение текста на чанки (MVP для RAG).

Идея:
- длинный документ делим на куски фиксированной длины (по словам)
- делаем overlap, чтобы смысл не рвался на границах

В дипломе можно заменить на более умный чанкинг (по заголовкам/абзацам).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def chunk_text(doc_id: str, text: str, *, chunk_size_words: int = 120, overlap_words: int = 30) -> List[Chunk]:
    words = text.split()
    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be < chunk_size_words")

    chunks: List[Chunk] = []
    start = 0
    cid = 0
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        chunk_text_str = " ".join(chunk_words).strip()
        if chunk_text_str:
            chunks.append(Chunk(doc_id=doc_id, chunk_id=cid, text=chunk_text_str))
            cid += 1
        if end == len(words):
            break
        start = end - overlap_words
    return chunks
