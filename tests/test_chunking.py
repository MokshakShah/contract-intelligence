from src.chunking import chunk_documents


def test_fixed_chunks_preserve_offsets():
    documents = {"doc": "abcdefghij"}
    chunks = chunk_documents(documents, {"strategy": "fixed", "chunk_size": 4, "overlap": 1})
    assert [(chunk.start, chunk.end, chunk.text) for chunk in chunks] == [
        (0, 4, "abcd"), (3, 7, "defg"), (6, 10, "ghij")
    ]


def test_recursive_chunks_preserve_offsets():
    documents = {"doc": "First sentence. Second sentence.\n\nThird paragraph."}
    chunks = chunk_documents(documents, {"strategy": "recursive", "chunk_size": 40, "overlap": 0})
    assert all(chunk.text == documents["doc"][chunk.start:chunk.end] for chunk in chunks)


def test_all_chunk_text_matches_source_slice():
    documents = {"doc": "One. Two.\n\nThree. Four."}
    for strategy in ("fixed", "recursive"):
        chunks = chunk_documents(
            documents,
            {"strategy": strategy, "chunk_size": 9, "overlap": 2},
        )
        assert all(chunk.text == documents[chunk.doc_path][chunk.start:chunk.end] for chunk in chunks)
