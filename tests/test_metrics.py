from src.metrics import query_metrics


def test_perfect_partial_and_no_overlap_metrics():
    truth = [{"file_path": "a", "span": [10, 20]}]
    perfect = query_metrics([{"doc_path": "a", "start": 10, "end": 20}], truth, (1,))
    partial = query_metrics([{"doc_path": "a", "start": 15, "end": 25}], truth, (1,))
    none = query_metrics([{"doc_path": "a", "start": 0, "end": 5}], truth, (1,))
    assert perfect["precision"][1] == 1.0
    assert perfect["recall"][1] == 1.0
    assert partial["precision"][1] == 0.5
    assert partial["recall"][1] == 0.5
    assert none["precision"][1] == 0.0
    assert none["recall"][1] == 0.0


def test_character_metrics_merge_overlapping_retrieval_ranges():
    retrieved = [
        {"doc_path": "a", "start": 0, "end": 10},
        {"doc_path": "a", "start": 5, "end": 15},
    ]
    truth = [{"file_path": "a", "span": [7, 12]}]
    metrics = query_metrics(retrieved, truth, (1, 2))
    assert metrics["precision"][1] == 3 / 10
    assert metrics["precision"][2] == 5 / 15
    assert metrics["recall"][2] == 1.0
    assert metrics["doc_hit"][1] == 1.0
    assert metrics["span_hit"][1] == 1.0


def test_ground_truth_overlapping_spans_are_unioned():
    metrics = query_metrics(
        [{"doc_path": "a", "start": 0, "end": 15}],
        [
            {"file_path": "a", "span": [0, 10]},
            {"file_path": "a", "span": [5, 20]},
        ],
        (1,),
    )
    assert metrics["recall"][1] == 15 / 20


def test_metrics_do_not_match_ranges_from_other_documents():
    metrics = query_metrics(
        [{"doc_path": "b", "start": 0, "end": 10}],
        [{"file_path": "a", "span": [0, 10]}],
        (1,),
    )
    assert metrics["precision"][1] == 0.0
    assert metrics["recall"][1] == 0.0
    assert metrics["doc_hit"][1] == 0.0
    assert metrics["span_hit"][1] == 0.0
