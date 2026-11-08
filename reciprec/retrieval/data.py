"""Read prepared Parquet datasets into arrays indexed by user row."""

import numpy as np
import pyarrow.parquet as pq

FEATURES = (
    "ratings_given", "ratings_received", "mean_rating_given", "mean_rating_received",
    "like_rate_given", "like_rate_received",
)


def read_users(path):
    users = pq.read_table(path, columns=["user_id", "gender", *FEATURES])
    ids = users["user_id"].to_numpy()
    order = np.argsort(ids)
    ids = ids[order]
    genders = users["gender"].to_numpy()[order]
    if len(ids) == 0 or np.any(ids[1:] == ids[:-1]) or not np.isin(genders, ["F", "M"]).all():
        raise ValueError("Expected a nonempty user table with unique IDs and F/M genders")
    features = np.column_stack([users[name].to_numpy() for name in FEATURES])
    return ids, genders.astype("U1"), scale_features(features)[order]


def scale_features(raw_features):
    features = raw_features.astype(np.float32, copy=True)
    features[:, :2] = np.log1p(features[:, :2])
    # Compute scaling from users with observed training statistics for each feature.
    for column in features.T:
        observed = np.isfinite(column)
        if observed.any():
            values = column[observed]
            column[observed] = (values - values.mean()) / max(float(values.std()), 1e-6)
        column[~observed] = 0
    return features


def read_edges(path, ids):
    edges = pq.read_table(path, columns=["src", "dst", "liked"])
    return edge_rows(edges, ids), edges["liked"].to_numpy() == 1


def edge_rows(edges, ids):
    endpoints = np.column_stack([edges[name].to_numpy() for name in ("src", "dst")])
    rows = np.searchsorted(ids, endpoints)
    if (rows >= len(ids)).any() or not np.array_equal(ids[rows], endpoints):
        raise ValueError("Found edge endpoints missing from the user table")
    return rows.astype(np.int32)


def group_targets(edges, user_count):
    """Return an array of destination rows for each source row."""
    ordered = edges[np.argsort(edges[:, 0], kind="stable")]
    stops = np.bincount(ordered[:, 0], minlength=user_count).cumsum()
    return np.split(ordered[:, 1], stops[:-1])


def evaluation_targets(edges, liked, genders, seen):
    """Build labels restricted to the same eligibility rules as candidate search."""
    source, target = edges.T
    eligible = liked & (source != target) & (genders[source] != genders[target])
    grouped = group_targets(edges[eligible], len(genders))
    targets = {}
    for user, positives in enumerate(grouped):
        if len(positives):
            remaining = positives[~np.isin(positives, seen[user])]
            if len(remaining):
                targets[user] = remaining
    return targets
