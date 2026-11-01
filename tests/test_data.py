from src.data import assign_split, load_mini_queries, mini, validate_snippet_offsets


def test_assign_split_is_deterministic():
    assert assign_split("What is the agreement date?") == assign_split(
        "What is the agreement date?"
    )


def test_mini_returns_all_small_inputs():
    tests = [{"snippets": [{"file_path": "a"}]}]
    assert mini(tests) == tests


def test_mini_dataset_has_194_queries():
    assert len(load_mini_queries()) == 194


def test_validate_snippet_offsets():
    queries = [
        {"snippets": [{"file_path": "cuad/a.txt", "span": [1, 4], "answer": "ell"}]}
    ]
    assert validate_snippet_offsets(queries, {"cuad/a.txt": "hello"}) == 0
