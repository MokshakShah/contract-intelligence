"""Run configured retrieval evaluation without using test spans at inference."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .chunking import chunk_documents
from .data import (
    assign_split,
    load_corpus,
    load_mini_queries,
    needed_document_paths,
    query_text,
)
from .metrics import average_metrics, query_metrics
from .retrieval import Retriever

KS = (1, 2, 4, 8, 16, 32)


def _load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run(config_path: Path, split: str, final: bool = False) -> dict:
    if split == "test" and not final:
        raise SystemExit("Refusing test evaluation without --final")
    config = _load_config(config_path)
    queries = [
        query for query in load_mini_queries()
        if assign_split(query_text(query)) == split
    ]
    corpus = load_corpus(needed_document_paths(load_mini_queries()))
    chunks = chunk_documents(corpus, config["chunking"])
    retriever = Retriever(chunks, config["retrieval"])
    rows = []
    for query in queries:
        retrieved = retriever.retrieve(query_text(query))
        rows.append(query_metrics(retrieved, query["snippets"], KS))
    metrics = average_metrics(rows, KS)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "split": split,
        "n_queries": len(queries),
        "metrics": metrics,
    }
    log_path = Path("experiments/log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
    if split == "test":
        counter_path = Path("experiments/test_runs.json")
        counter = {"test_run_counter": 0}
        if counter_path.exists():
            counter = json.loads(counter_path.read_text(encoding="utf-8"))
        counter["test_run_counter"] = int(counter.get("test_run_counter", 0)) + 1
        counter_path.write_text(json.dumps(counter, indent=2) + "\n", encoding="utf-8")
    print(f"split={split} n_queries={len(queries)}")
    print("k  precision  recall  doc_hit  span_hit")
    for k in KS:
        print(
            f"{k:<2} {metrics['precision'][k]:9.4f} "
            f"{metrics['recall'][k]:7.4f} {metrics['doc_hit'][k]:8.4f} "
            f"{metrics['span_hit'][k]:8.4f}"
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    run(args.config, args.split, args.final)


if __name__ == "__main__":
    main()
