from src.abstention import normalized_scores, should_abstain


def test_normalized_scores_are_ordered_and_bounded():
    scores = normalized_scores([1.0, 2.0, 3.0])
    assert 0 < scores[0] < scores[1] < scores[2] < 1


def test_abstention_rules():
    results = [{"normalized_score": 0.4}, {"normalized_score": 0.39}]
    assert should_abstain(results, {"rule": "score_threshold", "threshold": 0.5})
    assert should_abstain(results, {"rule": "margin", "margin_threshold": 0.02})
    assert should_abstain(results, {"rule": "combined", "threshold": 0.5})
    assert not should_abstain(results, {"rule": "none"})
