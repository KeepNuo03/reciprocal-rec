"""Evaluate re-ranking of held-out pairs with observed labels in both directions."""

import numpy as np
from scipy.stats import rankdata

from reciprec.eval.metrics import exposure_gini
from reciprec.pipeline import score_pairs


def reciprocal_rows(edges, liked, n_users):
    src, dst = edges.T
    keys = src.astype(np.int64) * n_users + dst
    reverse_keys = dst.astype(np.int64) * n_users + src
    order = np.argsort(keys)
    positions = np.searchsorted(keys[order], reverse_keys)
    reverse = order[np.minimum(positions, len(order) - 1)]
    keep = (src != dst) & (keys[reverse] == reverse_keys)
    return edges[keep], liked[keep] & liked[reverse[keep]]


def ranking_metrics(edges, relevance, scores, n_users, k):
    if k < 1 or not len(edges):
        raise ValueError("Ranking evaluation needs candidates and a positive k")
    src, dst = edges.T
    order = np.lexsort((dst, -scores, src))
    groups = np.r_[0, np.flatnonzero(np.diff(src[order])) + 1]
    sizes = np.diff(np.r_[groups, len(edges)])
    rank = np.arange(len(edges)) - np.repeat(groups, sizes)
    rel = relevance[order]
    top = rank < k
    positives = np.add.reduceat(rel, groups)
    hits = np.logical_or.reduceat(rel & top, groups)
    dcg = np.add.reduceat(rel * top / np.log2(rank + 2), groups)
    discount = np.r_[0, (1 / np.log2(np.arange(min(k, int(sizes.max()))) + 2)).cumsum()]
    ideal = discount[np.minimum(positives, k)]
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    exposures = np.bincount(dst[order[top]], minlength=n_users)
    histogram = {i: int(count) for i, count in enumerate(np.bincount(exposures)) if count}
    return {
        f"mutual_hit@{k}": float(hits.mean()),
        f"ndcg@{k}": float(ndcg.mean()),
        "exposure_gini": exposure_gini(histogram),
        "coverage": float(np.count_nonzero(exposures) / n_users),
    }


def auc(labels, scores):
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    return float((rankdata(scores)[labels].sum() - positives * (positives + 1) / 2)
                 / (positives * negatives))


def evaluate(model, catalog, edges, liked, k, alpha):
    probabilities = model.predict(catalog.matrix(edges))
    src, dst = edges.T
    f_to_m = (catalog.genders[src] == "F") & (catalog.genders[dst] == "M")
    m_to_f = (catalog.genders[src] == "M") & (catalog.genders[dst] == "F")
    n_users = len(catalog.genders)
    candidates, mutual = reciprocal_rows(edges, liked, n_users)
    if not len(candidates):
        raise ValueError("No reciprocally rated test pairs")
    scores = score_pairs(model, catalog, candidates, alpha)
    return {
        "directed_auc": {
            "overall": auc(liked, probabilities),
            "f_to_m": auc(liked[f_to_m], probabilities[f_to_m]),
            "m_to_f": auc(liked[m_to_f], probabilities[m_to_f]),
        },
        "evaluated_users": len(np.unique(candidates[:, 0])),
        "reciprocal_pairs": len(candidates) // 2,
        "mutual_like_pairs": int(mutual.sum()) // 2,
        "methods": {
            method: ranking_metrics(candidates, mutual, score, n_users, k)
            for method, score in scores.items()
        },
    }
