"""Простой retrieval для RAG: TF-IDF эмбеддинги + cosine similarity.

Почему TF-IDF?
- работает 'из коробки' без тяжёлых моделей
- хорошо подходит для MVP и обучения
- позже можно заменить на sentence-transformers + векторную БД

Интерфейс:
- index(chunks)
- search(query, top_k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.core.chunking import Chunk


@dataclass
class SearchHit:
    score: float
    chunk: Chunk


class TfidfRetriever:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(stop_words=None)
        self._chunks: List[Chunk] = []
        self._matrix = None  # type: ignore

    @property
    def is_indexed(self) -> bool:
        return self._matrix is not None and len(self._chunks) > 0

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]
        self._matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, *, top_k: int = 5) -> List[SearchHit]:
        if not self.is_indexed:
            raise RuntimeError("Retriever is not indexed. Call index(chunks) first.")
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self._matrix).ravel()
        top_idx = np.argsort(-sims)[:top_k]
        hits: List[SearchHit] = []
        for i in top_idx:
            hits.append(SearchHit(score=float(sims[i]), chunk=self._chunks[int(i)]))
        return hits
