"""Dataset loading, split assignment, and CUAD span validation."""

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


DEFAULT_DATA_ROOT = Path("data/raw")
BENCHMARK_PATH = Path("benchmarks/cuad.json")


def mini(tests: list[dict[str, Any]], n: int = 194) -> list[dict[str, Any]]:
    if len(tests) <= n:
        return tests

    def key(t: dict[str, Any]) -> float:
        import random

        random.seed(t["snippets"][0]["file_path"])
        return random.random()

    return sorted(tests, key=key)[:n]


def load_mini_queries(data_root: Path = DEFAULT_DATA_ROOT) -> list[dict[str, Any]]:
    tests = json.loads(
        (data_root / BENCHMARK_PATH).read_text(encoding="utf-8")
    )["tests"]
    return mini(tests)


def assign_split(query: str) -> str:
    value = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if value % 10 < 3 else "train"


def needed_document_paths(queries: list[dict[str, Any]]) -> set[str]:
    return {
        snippet["file_path"]
        for query in queries
        for snippet in query["snippets"]
    }


def load_corpus(
    needed_paths: set[str], data_root: Path = DEFAULT_DATA_ROOT
) -> dict[str, str]:
    corpus = {}
    for relative_path in sorted(needed_paths):
        if not relative_path.startswith("cuad/"):
            raise ValueError(f"Unexpected corpus path: {relative_path}")
        path = data_root / "corpus" / relative_path
        if not path.exists():
            target = _canonical_name(Path(relative_path).name)
            matches = [
                candidate
                for candidate in (data_root / "corpus" / "cuad").iterdir()
                if _canonical_name(candidate.name) == target
            ]
            if len(matches) != 1:
                raise FileNotFoundError(f"Could not locate {relative_path}: {matches}")
            path = matches[0]
        with open(path, encoding="utf-8") as document:
            corpus[relative_path] = document.read()
    return corpus


def validate_snippet_offsets(
    queries: list[dict[str, Any]], corpus: dict[str, str]
) -> int:
    mismatches = 0
    for query in queries:
        for snippet in query["snippets"]:
            document = corpus[snippet["file_path"]]
            start, end = snippet["span"]
            answer = snippet["answer"]
            if document[start:end] != answer:
                mismatches += 1
    return mismatches


def query_text(query: dict[str, Any]) -> str:
    """Return the benchmark's natural-language query field."""
    for field in ("query", "question"):
        if field in query:
            return query[field]
    raise KeyError("Query has neither 'query' nor 'question' field")


def _canonical_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).casefold()
    return re.sub(r"[^a-z0-9]+", "", normalized.encode("ascii", "ignore").decode())


def main() -> None:
    queries = load_mini_queries()
    corpus = load_corpus(needed_document_paths(queries))
    train_count = sum(assign_split(query_text(query)) == "train" for query in queries)
    test_count = len(queries) - train_count
    mismatches = validate_snippet_offsets(queries, corpus)
    print(f"queries: {len(queries)}")
    print(f"unique documents: {len(corpus)}")
    print(f"train: {train_count}")
    print(f"test: {test_count}")
    print(f"validation mismatches: {mismatches}")
    if mismatches:
        raise ValueError(f"Found {mismatches} snippet offset mismatches")


if __name__ == "__main__":
    main()
