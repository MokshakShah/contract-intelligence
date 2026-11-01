"""Configurable post-retrieval abstention decisions."""

import math
from typing import Any


def _threshold(config: dict[str, Any], name: str, default: float) -> float:
    return float(config.get(name, default))


def should_abstain(results: list[dict[str, Any]], config: dict[str, Any]) -> bool:
    if config.get("enabled") is False:
        return False
    rule = config.get("rule", "none")
    if rule == "none" or not results:
        return False
    top = float(results[0].get("normalized_score", results[0].get("score", 0.0)))
    second = float(results[1].get("normalized_score", results[1].get("score", 0.0))) if len(results) > 1 else 0.0
    score_bad = top < _threshold(config, "threshold", 0.5)
    margin_bad = top - second < _threshold(config, "margin_threshold", 0.05)
    if rule == "score_threshold":
        return score_bad
    if rule == "margin":
        return margin_bad
    if rule == "combined":
        return score_bad or margin_bad
    raise ValueError(f"Unknown abstention rule: {rule}")


def normalized_scores(scores: list[float]) -> list[float]:
    """Map a score list to sigmoid z-scores in (0, 1)."""
    if not scores:
        return []
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / len(scores)
    scale = (math.sqrt(variance) or 1.0) * 5.0
    return [1.0 / (1.0 + math.exp(-((score - mean) / scale))) for score in scores]
