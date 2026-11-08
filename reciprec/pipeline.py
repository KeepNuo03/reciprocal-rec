"""Shared scoring for a batch of directed candidate pairs."""

from reciprec.fusion import fuse
from reciprec.ranking.features import FEATURE_NAMES


def score_pairs(model, catalog, edges, alpha):
    matrices = pair_features(catalog, edges)
    scores, _ = score_features(model, matrices, catalog.likes_received[edges[:, 1]], alpha)
    return scores


def pair_features(catalog, edges):
    return catalog.matrix(edges), catalog.matrix(edges[:, ::-1])


def score_features(model, matrices, popularity, alpha):
    matrix, reversed_matrix = matrices
    forward = model.predict(matrix)
    reverse = model.predict(reversed_matrix)
    scores = {
        "popularity": popularity,
        "two_tower": matrix[:, FEATURE_NAMES.index("forward_dot_z")],
        **fuse(forward, reverse, popularity, alpha),
    }
    return scores, reverse
