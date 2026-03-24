from src.core.chunking import Chunk
from src.core.retrieval import TfidfRetriever


def test_retrieval_returns_hits():
    chunks = [
        Chunk("a", 0, "кот любит молоко"),
        Chunk("b", 0, "собака любит кости"),
        Chunk("c", 0, "рыба живёт в воде"),
    ]
    r = TfidfRetriever()
    r.index(chunks)
    hits = r.search("кот", top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
