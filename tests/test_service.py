from src.retrieval import search


def test_search_returns_ranked_offset_safe_results():
    config = {
        "chunking": {"strategy": "fixed", "chunk_size": 500, "overlap": 0},
        "retrieval": {"method": "bm25", "top_k": 2},
        "abstention": {"rule": "none"},
    }
    response = search("What is the expiration date of this contract?", config, 2)
    assert len(response["results"]) == 2
    assert response["results"][0]["score"] >= response["results"][1]["score"]
    assert all(result["text"] for result in response["results"])
