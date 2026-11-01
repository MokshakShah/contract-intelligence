from src.metrics import query_metrics


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
