"""Command-line interface for the Contract Intelligence Assistant."""

import argparse
from pathlib import Path

import yaml

from .retrieval import search


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed CUAD contracts")
    parser.add_argument("query")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--k", type=int, default=None)
    args = parser.parse_args()
    if args.k is not None and args.k < 1:
        parser.error("--k must be positive")
    response = search(args.query, load_config(args.config), args.k)
    if response["abstained"]:
        top_score = response["results"][0]["normalized_score"] if response["results"] else 0.0
        threshold = response["threshold"]
        print("No relevant clause found")
        print(f"Top score: {top_score:.4f}; threshold: {threshold}")
        return
    for rank, result in enumerate(response["results"], 1):
        print(
            f"{rank}. {result['doc_path']} [{result['start']}:{result['end']}] "
            f"score={result['score']:.4f} normalized={result['normalized_score']:.4f}"
        )
        print(f"   {result['text'][:300].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
