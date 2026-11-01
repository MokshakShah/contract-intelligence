"""LegalBench-RAG character-level retrieval metrics."""

from collections.abc import Iterable


def _ranges(items: Iterable[dict], doc_key: str = "doc_path") -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for item in items:
        doc = item.get(doc_key, item.get("file_path"))
        start = item.get("start", item.get("span", (None, None))[0])
        end = item.get("end", item.get("span", (None, None))[1])
        if doc is None or start is None or end is None or end <= start:
            continue
        result.setdefault(doc, []).append((int(start), int(end)))
    return {doc: _merge(ranges) for doc, ranges in result.items()}


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _length(ranges: dict[str, list[tuple[int, int]]]) -> int:
    return sum(end - start for spans in ranges.values() for start, end in spans)


def _intersection(left: dict[str, list[tuple[int, int]]], right: dict[str, list[tuple[int, int]]]) -> int:
    total = 0
    for doc in left.keys() & right.keys():
        for a, b in left[doc]:
            total += sum(max(0, min(b, d) - max(a, c)) for c, d in right[doc])
    return total


def query_metrics(retrieved: list[dict], ground_truth: list[dict], ks=(1, 2, 4, 8, 16, 32)) -> dict[str, dict[int, float]]:
    """Compute precision, recall, document hit, and span hit for one query."""
    truth = _ranges(ground_truth, "file_path")
    output = {"precision": {}, "recall": {}, "doc_hit": {}, "span_hit": {}}
    truth_docs = set(truth)
    truth_length = _length(truth)
    for k in ks:
        selected = retrieved[:k]
        found = _ranges(selected)
        overlap = _intersection(found, truth)
        retrieved_length = _length(found)
        output["precision"][k] = overlap / retrieved_length if retrieved_length else 0.0
        output["recall"][k] = overlap / truth_length if truth_length else 0.0
        output["doc_hit"][k] = float(bool(set(found) & truth_docs))
        output["span_hit"][k] = float(overlap > 0)
    return output


def average_metrics(per_query: list[dict[str, dict[int, float]]], ks=(1, 2, 4, 8, 16, 32)) -> dict[str, dict[int, float]]:
    if not per_query:
        return {name: {k: 0.0 for k in ks} for name in ("precision", "recall", "doc_hit", "span_hit")}
    return {
        name: {k: sum(row[name][k] for row in per_query) / len(per_query) for k in ks}
        for name in ("precision", "recall", "doc_hit", "span_hit")
    }
