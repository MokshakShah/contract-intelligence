"""Rule-based failure diagnosis for train-only retrieval experiments."""

from collections import defaultdict
from typing import Any

from .abstention import should_abstain
from .metrics import query_metrics


CLASSES = (
    "wrong_document",
    "right_doc_wrong_section",
    "right_section_partial_span",
    "false_abstention",
    "wrongly_answered",
)


def diagnose_query(
    query: dict[str, Any],
    retrieved: list[dict[str, Any]],
    abstention_config: dict[str, Any],
    unanswerable: bool = False,
) -> str | None:
    abstained = should_abstain(retrieved, abstention_config)
    if unanswerable:
        return "wrongly_answered" if not abstained else None
    if abstained:
        return "false_abstention"
    truth_docs = {snippet["file_path"] for snippet in query["snippets"]}
    retrieved_docs = {item["doc_path"] for item in retrieved}
    if not truth_docs & retrieved_docs:
        return "wrong_document"
    metrics = query_metrics(retrieved, query["snippets"], (8,))
    if metrics["span_hit"][8] == 0:
        return "right_doc_wrong_section"
    if metrics["recall"][8] < 0.5:
        return "right_section_partial_span"
    return None


def diagnose(
    benchmark: list[dict[str, Any]],
    benchmark_results: list[list[dict[str, Any]]],
    unanswerable: list[dict[str, Any]],
    unanswerable_results: list[list[dict[str, Any]]],
    abstention_config: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for query, results in zip(benchmark, benchmark_results):
        failure = diagnose_query(query, results, abstention_config)
        if failure:
            grouped[failure].append(query["query"])
    for query, results in zip(unanswerable, unanswerable_results):
        failure = diagnose_query(query, results, abstention_config, unanswerable=True)
        if failure:
            grouped[failure].append(query["query"])
    return {
        "counts": {name: len(grouped[name]) for name in CLASSES},
        "samples": {name: grouped[name][:3] for name in CLASSES},
    }
