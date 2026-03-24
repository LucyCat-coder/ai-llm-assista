"""RAG MVP пайплайн: документы → чанки → индекс → поиск → 'ответ'.

Здесь "ответ" пока без LLM:
- показываем топ-чанки как контекст
- даём шаблонный ответ, чтобы студент видел принцип RAG
Позже можно подключить LLM и генерировать ответ по контексту.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.core.chunking import Chunk, chunk_text
from src.core.retrieval import TfidfRetriever, SearchHit


@dataclass
class RagResult:
    query: str
    hits: List[SearchHit]

    def as_text(self) -> str:
        lines = [f"Вопрос: {self.query}", ""]
        lines.append("Найденный контекст (top chunks):")
        for i, h in enumerate(self.hits, 1):
            lines.append(f"{i}) score={h.score:.3f} doc={h.chunk.doc_id} chunk={h.chunk.chunk_id}")
            lines.append(h.chunk.text)
            lines.append("")
        # MVP 'ответ' — на базе первого чанка
        if self.hits:
            lines.append("MVP-ответ (без LLM):")
            lines.append(self.hits[0].chunk.text[:400] + ("..." if len(self.hits[0].chunk.text) > 400 else ""))
        return "\n".join(lines)


class RagPipeline:
    def __init__(self) -> None:
        self.retriever = TfidfRetriever()
        self._chunks: List[Chunk] = []

    def build_index(self, docs: Dict[str, str], *, chunk_size_words: int = 120, overlap_words: int = 30) -> None:
        all_chunks: List[Chunk] = []
        for doc_id, text in docs.items():
            all_chunks.extend(chunk_text(doc_id, text, chunk_size_words=chunk_size_words, overlap_words=overlap_words))
        self._chunks = all_chunks
        self.retriever.index(self._chunks)

    def ask(self, query: str, *, top_k: int = 5) -> RagResult:
        hits = self.retriever.search(query, top_k=top_k)
        return RagResult(query=query, hits=hits)
