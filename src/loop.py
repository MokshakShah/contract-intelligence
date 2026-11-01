"""Deterministic, train-only retrieval self-improvement loop.

Objective: maximize character-level Recall@8 on train benchmark queries.
Guardrails: false-abstention <= baseline + 3 percentage points,
correct-abstention does not drop more than 3 points, and Precision@8 does
not drop more than 20 percent relative to baseline.
"""

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .abstention import should_abstain
from .chunking import chunk_documents
from .data import assign_split, load_corpus, load_mini_queries, needed_document_paths, query_text
from .diagnose import diagnose
from .metrics import average_metrics, query_metrics
from .retrieval import Retriever

SEED = 17
KS = (8,)
MIN_DELTA = 0.001


def _train_data() -> tuple[list[dict], list[dict], dict[str, str]]:
    """Load only train benchmark queries and train unanswerable queries."""
    benchmark = [
        item for item in load_mini_queries()
        if assign_split(query_text(item)) == "train"
    ]
    unanswerable = [
        item for item in json.loads(Path("data/unanswerable.json").read_text(encoding="utf-8"))
        if assign_split(item["query"]) == "train"
    ]
    corpus = load_corpus(needed_document_paths(benchmark))
    return benchmark, unanswerable, corpus


def _evaluate(config: dict[str, Any], benchmark: list[dict], unanswerable: list[dict], corpus: dict[str, str]) -> dict[str, Any]:
    retriever = Retriever(
        chunk_documents(corpus, config["chunking"]),
        config["retrieval"],
    )
    abstention_config = config.get("abstention", {"rule": "none"})
    benchmark_results = [retriever.retrieve(query_text(item), 8) for item in benchmark]
    unanswerable_results = [retriever.retrieve(item["query"], 8) for item in unanswerable]
    rows = [query_metrics(results, item["snippets"], KS) for item, results in zip(benchmark, benchmark_results)]
    metrics = average_metrics(rows, KS)
    false_abstain = sum(should_abstain(results, abstention_config) for results in benchmark_results) / len(benchmark)
    correct_abstain = sum(should_abstain(results, abstention_config) for results in unanswerable_results) / len(unanswerable)
    return {
        "metrics": metrics,
        "false_abstention_rate": false_abstain,
        "correct_abstention_rate": correct_abstain,
        "benchmark_results": benchmark_results,
        "unanswerable_results": unanswerable_results,
        "diagnosis": diagnose(
            benchmark, benchmark_results, unanswerable, unanswerable_results, abstention_config
        ),
    }


def _propose(config: dict[str, Any], diagnosis: dict[str, Any], iteration: int) -> tuple[dict, str, str]:
    dominant = max(diagnosis["counts"], key=diagnosis["counts"].get)
    candidate = copy.deepcopy(config)
    rule = "rule"
    if dominant == "wrong_document":
        candidate["chunking"]["chunk_size"] = 250 if iteration % 2 else 750
        candidate["chunking"]["overlap"] = 50 if iteration % 2 else 0
        reason = "wrong_document dominant: test finer/coarser BM25 evidence windows"
    elif dominant == "right_doc_wrong_section":
        candidate["chunking"]["chunk_size"] = 250
        candidate["chunking"]["overlap"] = 50
        reason = "right_doc_wrong_section dominant: use smaller overlapping chunks"
    elif dominant == "right_section_partial_span":
        candidate["chunking"]["chunk_size"] = 750
        candidate["chunking"]["overlap"] = 100
        reason = "right_section_partial_span dominant: enlarge chunks with overlap"
    elif dominant == "false_abstention":
        candidate.setdefault("abstention", {})["threshold"] = max(
            0.0, float(candidate.get("abstention", {}).get("threshold", 0.75)) - 0.05
        )
        reason = "false_abstention dominant: lower abstention threshold"
    else:
        grid = [
            (250, 50, "fixed"),
            (500, 100, "fixed"),
            (750, 100, "fixed"),
            (500, 0, "recursive"),
        ]
        size, overlap, strategy = grid[iteration % len(grid)]
        candidate["chunking"].update(chunk_size=size, overlap=overlap, strategy=strategy)
        reason = "search fallback: deterministic chunking parameter grid"
        rule = "search"
    return candidate, reason, rule


def _accepted(candidate: dict, baseline: dict, current: dict) -> tuple[bool, str]:
    candidate_metrics = candidate["metrics"]
    baseline_metrics = baseline["metrics"]
    if candidate_metrics["recall"][8] < current["metrics"]["recall"][8] + MIN_DELTA:
        return False, "Recall@8 did not improve by the minimum delta"
    if candidate["false_abstention_rate"] > baseline["false_abstention_rate"] + 0.03:
        return False, "false-abstention guardrail exceeded"
    if candidate["correct_abstention_rate"] < baseline["correct_abstention_rate"] - 0.03:
        return False, "correct-abstention guardrail regressed"
    baseline_precision = baseline_metrics["precision"][8]
    if candidate_metrics["precision"][8] < baseline_precision * 0.8:
        return False, "Precision@8 guardrail regressed by more than 20 percent"
    return True, "Recall@8 improved and all guardrails hold"


def run(config_path: Path, iterations: int) -> list[dict[str, Any]]:
    random.seed(SEED)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    benchmark, unanswerable, corpus = _train_data()
    baseline = _evaluate(copy.deepcopy(config), benchmark, unanswerable, corpus)
    current = baseline
    best_config = copy.deepcopy(config)
    log_path = Path("experiments/log.jsonl")
    markdown_path = Path("experiments/LOG.md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with markdown_path.open("a", encoding="utf-8") as markdown:
        markdown.write("\n## Self-improvement loop\n")
        markdown.write("| Iteration | Change | Diagnosis | Recall@8 | Precision@8 | False abstain | Correct abstain | Decision |\n|---:|---|---|---:|---:|---:|---:|---|\n")
        records = []
        for iteration in range(1, iterations + 1):
            candidate_config, reason, proposer = _propose(best_config, current["diagnosis"], iteration)
            candidate = _evaluate(candidate_config, benchmark, unanswerable, corpus)
            accepted, decision = _accepted(candidate, baseline, current)
            if accepted:
                best_config = candidate_config
                current = candidate
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "iteration": iteration,
                "split": "train",
                "proposer": proposer,
                "change": reason,
                "diagnosis": current["diagnosis"],
                "metrics": candidate["metrics"],
                "false_abstention_rate": candidate["false_abstention_rate"],
                "correct_abstention_rate": candidate["correct_abstention_rate"],
                "decision": "accept" if accepted else "reject",
                "decision_reason": decision,
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            markdown.write(
                f"| {iteration} | {reason} | {candidate['diagnosis']['counts']} | "
                f"{candidate['metrics']['recall'][8]:.4f} | {candidate['metrics']['precision'][8]:.4f} | "
                f"{candidate['false_abstention_rate']:.4f} | {candidate['correct_abstention_rate']:.4f} | "
                f"{record['decision']}: {decision} |\n"
            )
            records.append(record)
            print(
                f"iteration={iteration} decision={record['decision']} "
                f"recall@8={candidate['metrics']['recall'][8]:.4f} "
                f"precision@8={candidate['metrics']['precision'][8]:.4f}"
            )
            print(f"diagnosis_counts={candidate['diagnosis']['counts']}")
            print(f"diagnosis_samples={candidate['diagnosis']['samples']}")
    Path("configs/final.yaml").write_text(
        yaml.safe_dump(best_config, sort_keys=False),
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    run(args.config, args.iterations)


if __name__ == "__main__":
    main()
