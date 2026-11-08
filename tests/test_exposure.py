import numpy as np

from reciprec.eval.exposure import analyze
from reciprec.eval.ranking import ranking_metrics
from reciprec.fusion import penalize


def test_sweep_reproduces_reference_and_cohorts_cover_all_queries():
    edges = np.array([[0, 1], [0, 2], [1, 0], [2, 0]])
    mutual = np.array([True, False, True, False])
    popularity = np.array([0, 100, 0, 10])
    product = np.array([0.9, 0.5, 0.9, 0.5])
    reference = penalize(product, popularity[edges[:, 1]], 0.3)
    scores = {"one_way": product, "product": product, "product_debiased": reference}
    report = analyze(edges, mutual, scores, popularity, k=1, reference_alpha=0.3)
    assert [point["alpha"] for point in report["sweep"]] == [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert report["sweep"][3] == {
        "alpha": 0.3, **ranking_metrics(edges, mutual, reference, len(popularity), 1)
    }
    assert report["sweep"][0]["mutual_hit@1"] > report["sweep"][3]["mutual_hit@1"]
    assert sum(group["users"] for group in report["cohorts"]) == 3
    assert report["cohorts"][0]["max_training_likes_received"] == 0
    assert report["cohorts"][0]["users"] == 2
