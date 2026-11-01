"""Sweep score thresholds on train benchmark and unanswerable queries."""

import argparse
import json
from pathlib import Path

import yaml

from .abstention import should_abstain
from .chunking import chunk_documents
from .data import assign_split, load_corpus, load_mini_queries, needed_document_paths, query_text
from .retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("Threshold tuning is restricted to train")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    benchmark = [q for q in load_mini_queries() if assign_split(query_text(q)) == args.split]
    unanswerable = [
        q for q in json.loads(Path("data/unanswerable.json").read_text(encoding="utf-8"))
        if assign_split(q["query"]) == args.split
    ]
    corpus = load_corpus(needed_document_paths(load_mini_queries()))
    retriever = Retriever(chunk_documents(corpus, config["chunking"]), config["retrieval"])
    benchmark_results = [retriever.retrieve(query_text(q)) for q in benchmark]
    unanswerable_results = [retriever.retrieve(q["query"]) for q in unanswerable]
    print("threshold  false_abstain  correct_abstain")
    for threshold in [i / 100 for i in range(101)]:
        abstention = {"rule": "score_threshold", "threshold": threshold}
        false_count = sum(should_abstain(r, abstention) for r in benchmark_results)
        correct_count = sum(should_abstain(r, abstention) for r in unanswerable_results)
        print(
            f"{threshold:9.2f} {false_count / len(benchmark_results):14.4f} "
            f"{correct_count / len(unanswerable_results):15.4f}"
        )


if __name__ == "__main__":
    main()
