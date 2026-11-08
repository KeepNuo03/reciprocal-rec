"""Sweep the exposure penalty using a frozen ranker on known-label test candidates."""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import yaml

from reciprec.eval.ranking import ranking_metrics, reciprocal_rows
from reciprec.fusion import penalize
from reciprec.pipeline import score_pairs
from reciprec.ranking.features import load_full_catalog
from reciprec.ranking.ranker import read_training
from reciprec.retrieval.data import read_edges


def analyze(edges, mutual, scores, popularity, k, reference_alpha):
    n_users = len(popularity)
    target_popularity = popularity[edges[:, 1]]
    sweep = [
        {"alpha": alpha, **ranking_metrics(
            edges, mutual, penalize(scores["product"], target_popularity, alpha), n_users, k
        )}
        for alpha in sorted({*(step / 10 for step in range(6)), reference_alpha})
    ]
    # Keep tied counts together and define cuts on the full training catalog.
    cuts = np.unique(np.quantile(popularity, [0.25, 0.5, 0.75], method="inverted_cdf"))
    cuts = cuts[cuts < popularity.max()]
    source_popularity = popularity[edges[:, 0]]
    buckets = np.searchsorted(cuts, source_popularity, side="left")
    methods = {key: scores[key] for key in ("one_way", "product", "product_debiased")}
    cohorts = []
    for bucket in np.unique(buckets):
        mask = buckets == bucket
        cohort = edges[mask]
        comparisons = {}
        for name, values in methods.items():
            metrics = ranking_metrics(cohort, mutual[mask], values[mask], n_users, k)
            comparisons[name] = {key: metrics[key] for key in (f"mutual_hit@{k}", f"ndcg@{k}")}
        cohorts.append({
            "min_training_likes_received": 0 if bucket == 0 else int(cuts[bucket - 1]) + 1,
            "max_training_likes_received": int(cuts[bucket]) if bucket < len(cuts) else None,
            "users": len(np.unique(cohort[:, 0])),
            "methods": comparisons,
        })
    return {"k": k, "reference_alpha": reference_alpha, "sweep": sweep, "cohorts": cohorts}


def plot_tradeoff(report, path):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    sweep = report["sweep"]
    k = report["k"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, key, label in zip(axes, (f"mutual_hit@{k}", f"ndcg@{k}"),
                              (f"Mutual-Hit@{k} (%)", f"NDCG@{k}")):
        gini = [point["exposure_gini"] for point in sweep]
        values = [point[key] * (100 if key.startswith("mutual") else 1) for point in sweep]
        ax.plot(gini, values, "o-", color="#176b87")
        for x, y, point in zip(gini, values, sweep):
            ax.annotate(f'{point["alpha"]:g}', (x, y), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set(xlabel="Exposure Gini", ylabel=label)
        ax.ticklabel_format(axis="y", useOffset=False)
        ax.grid(alpha=0.2)
        ax.margins(x=0.12, y=0.25)
    fig.suptitle("Exposure penalty trade-off on LibimSeTi (points labeled by α)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--retrieval", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--ranker", type=Path, default=Path("artifacts/ranking/ranker.txt"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))["ranking"]
    ids, genders, _, _, _, stats, popularity = read_training(args.prepared)
    catalog = load_full_catalog(args.retrieval / "embeddings.npz", ids, genders, stats, popularity)
    edges, liked = read_edges(args.prepared / "test", ids)
    pairs, mutual = reciprocal_rows(edges, liked, len(ids))
    if not len(pairs):
        raise ValueError("No reciprocally rated test pairs")
    model = lgb.Booster(model_file=str(args.ranker))
    scores = score_pairs(model, catalog, pairs, settings["alpha"])
    report = analyze(pairs, mutual, scores, popularity, settings["k"], settings["alpha"])
    args.output.mkdir(parents=True, exist_ok=True)
    plot_tradeoff(report, args.output / "exposure_tradeoff.png")
    (args.output / "exposure.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
