import numpy as np
import pytest

from reciprec.eval.ranking import auc, ranking_metrics, reciprocal_rows
from reciprec.fusion import fuse
from reciprec.pipeline import score_pairs
from reciprec.ranking.features import (
    FEATURE_NAMES,
    FeatureCatalog,
    pair_folds,
    user_statistics,
)


def test_fold_keeps_both_directions_together():
    edges = np.array([[0, 1], [0, 2], [1, 0], [2, 0], [1, 2], [2, 1]])
    folds = pair_folds(edges, 3, seed=42)
    assert folds[0] == folds[2] and folds[1] == folds[3] and folds[4] == folds[5]
    np.testing.assert_array_equal(folds, pair_folds(edges, 3, seed=42))


def test_pair_scoring_uses_each_orientation_and_training_statistics():
    stats, popularity = user_statistics(
        np.array([[0, 1], [1, 0]]), np.array([10, 5]), np.array([True, False]), 3
    )
    np.testing.assert_array_equal(popularity, [0, 1, 0])
    assert np.isnan(stats[2, 2])
    vectors = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    catalog = FeatureCatalog(stats, popularity, np.array(["F", "M", "F"]), vectors, vectors)

    class DirectedModel:
        def predict(self, matrix):
            return matrix[:, FEATURE_NAMES.index("src_mean_rating_given")] / 10

    scores = score_pairs(DirectedModel(), catalog, np.array([[0, 1], [1, 0]]), alpha=0.3)
    np.testing.assert_array_equal(scores["one_way"], [1.0, 0.5])
    np.testing.assert_array_equal(scores["product"], [0.5, 0.5])
    assert scores["product_debiased"][0] < scores["product_debiased"][1]
    assert catalog.dot_scores(np.array([0, 0]), np.array([1, 2]))[1] > 0


def test_zero_direction_gives_zero_fused_score():
    scores = fuse(np.array([0.0, 0.0, 0.5]), np.array([0.8, 0.0, 0.8]), np.zeros(3), alpha=0)
    np.testing.assert_allclose(scores["product"], [0.0, 0.0, 0.4])
    np.testing.assert_allclose(scores["harmonic"], [0.0, 0.0, 0.8 / 1.3])
    np.testing.assert_array_equal(scores["product"], scores["product_debiased"])


def test_ranked_metrics_include_zero_positive_queries_and_unexposed_users():
    edges = np.array([[0, 1], [1, 0], [0, 2], [2, 0], [0, 3]])
    liked = np.array([True, True, False, True, True])
    pairs, mutual = reciprocal_rows(edges, liked, n_users=4)
    np.testing.assert_array_equal(pairs, edges[:4])
    assert mutual.tolist() == [True, True, False, False]
    result = ranking_metrics(pairs, mutual, np.array([2, 1, 3, 1]), n_users=4, k=1)
    assert result["mutual_hit@1"] == pytest.approx(1 / 3)
    assert result["ndcg@1"] == pytest.approx(1 / 3)
    assert result["coverage"] == 0.5
    assert result["exposure_gini"] == pytest.approx(7 / 12)
    assert auc(np.array([True, False]), np.array([0.4, 0.4])) == 0.5
