from src.core.chunking import chunk_text


def test_chunking_basic():
    text = " ".join([f"w{i}" for i in range(400)])
    chunks = chunk_text("doc", text, chunk_size_words=100, overlap_words=20)
    assert len(chunks) >= 4
    assert chunks[0].doc_id == "doc"
