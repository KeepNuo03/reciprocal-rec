"""Build directed pair features from a training catalog and two-tower vectors."""

from dataclasses import dataclass, field

import numpy as np

from reciprec.retrieval.data import FEATURES


FEATURE_NAMES = (
    *(f"src_{name}" for name in FEATURES),
    *(f"dst_{name}" for name in FEATURES),
    "forward_dot_z", "reverse_dot_z", "mean_given_gap", "received_like_rate_gap", "gender_pair",
)


def load_full_catalog(path, ids, genders, stats, popularity):
    with np.load(path) as vectors:
        if not np.array_equal(ids, vectors["user_ids"]) or not np.array_equal(genders, vectors["genders"]):
            raise ValueError("Retrieval artifacts do not match the prepared user catalog")
        return FeatureCatalog(stats, popularity, genders, vectors["u_out"], vectors["u_in"])


def user_statistics(edges, ratings, liked, n_users):
    src, dst = edges.T
    given = np.bincount(src, minlength=n_users)
    received = np.bincount(dst, minlength=n_users)
    likes_given = np.bincount(src[liked], minlength=n_users)
    likes_received = np.bincount(dst[liked], minlength=n_users)

    def mean(total, count):
        return np.divide(total, count, out=np.full(n_users, np.nan), where=count > 0)

    stats = np.column_stack((
        given, received,
        mean(np.bincount(src, weights=ratings, minlength=n_users), given),
        mean(np.bincount(dst, weights=ratings, minlength=n_users), received),
        mean(likes_given, given), mean(likes_received, received),
    )).astype(np.float32)
    return stats, likes_received


@dataclass(frozen=True)
class FeatureCatalog:
    stats: np.ndarray
    likes_received: np.ndarray
    genders: np.ndarray
    u_out: np.ndarray
    u_in: np.ndarray
    dot_mean: np.ndarray = field(init=False, repr=False)
    dot_scale: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Standardize each query's dot scores so independently trained towers share a scale.
        mean = self.u_in.mean(axis=0)
        centered = self.u_in - mean
        covariance = (centered.T @ centered) / len(centered)
        variance = np.einsum("ij,ij->i", self.u_out @ covariance, self.u_out)
        object.__setattr__(self, "dot_mean", self.u_out @ mean)
        object.__setattr__(self, "dot_scale", np.sqrt(np.maximum(variance, 1e-8)))

    def dot_scores(self, src, dst):
        dots = np.einsum("ij,ij->i", self.u_out[src], self.u_in[dst])
        return (dots - self.dot_mean[src]) / self.dot_scale[src]

    def matrix(self, edges):
        matrix = np.empty((len(edges), len(FEATURE_NAMES)), dtype=np.float32)
        for start in range(0, len(edges), 65536):
            block = slice(start, start + 65536)
            src, dst = edges[block].T
            source, target = self.stats[src], self.stats[dst]
            matrix[block] = np.column_stack((
                source, target,
                self.dot_scores(src, dst), self.dot_scores(dst, src),
                source[:, 2] - target[:, 2], source[:, 5] - target[:, 5],
                2 * (self.genders[src] == "M") + (self.genders[dst] == "M"),
            ))
        return matrix


def pair_folds(edges, n_users, seed):
    """Assign both directions of each training pair to the same seeded fold."""
    src, dst = edges.T
    key = np.minimum(src, dst).astype(np.int64) * n_users + np.maximum(src, dst)
    pairs, inverse = np.unique(key, return_inverse=True)
    return np.random.default_rng(seed).integers(0, 2, size=len(pairs), dtype=np.int8)[inverse]
