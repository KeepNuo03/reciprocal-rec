"""Shared scoring for a batch of directed candidate pairs."""

from reciprec.fusion import fuse
from reciprec.ranking.features import FEATURE_NAMES


def score_pairs(model, catalog, edges, alpha):
    matrix = catalog.matrix(edges)
    forward = model.predict(matrix)
    reverse = model.predict(catalog.matrix(edges[:, ::-1]))
    popularity = catalog.likes_received[edges[:, 1]]
    return {
        "popularity": popularity,
        "two_tower": matrix[:, FEATURE_NAMES.index("forward_dot_z")],
        **fuse(forward, reverse, popularity, alpha),
    }
