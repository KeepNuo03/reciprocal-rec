"""Train a LightGBM ranker on pair-held-out two-tower features."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from reciprec.eval.ranking import evaluate
from reciprec.ranking.features import (
    FEATURE_NAMES,
    FeatureCatalog,
    load_full_catalog,
    pair_folds,
    user_statistics,
)
from reciprec.retrieval.data import (
    edge_rows,
    evaluation_targets,
    group_targets,
    read_edges,
    read_users,
    scale_features,
)
from reciprec.retrieval.two_tower import train_model


def cross_fit(edges, ratings, liked, genders, validation, config, seed, device):
    n_users = len(genders)
    folds = pair_folds(edges, n_users, seed)
    matrix = np.empty((len(edges), len(FEATURE_NAMES)), dtype=np.float32)
    summaries = []
    for fold in (0, 1):
        held_out = folds == fold
        if not held_out.any():
            raise ValueError("Cross-fitting requires two nonempty pair folds")
        fit_edges = edges[~held_out]
        fit_liked = liked[~held_out]
        stats, popularity = user_statistics(fit_edges, ratings[~held_out], fit_liked, n_users)
        seen = group_targets(fit_edges, n_users)
        val_targets = evaluation_targets(*validation, genders, seen)
        positives = fit_edges[fit_liked]
        positives = positives[np.lexsort((positives[:, 1], positives[:, 0]))]
        print(f"Cross-fit fold {fold + 1}/2", flush=True)
        _, (source, candidate), epoch, _, _ = train_model(
            positives, scale_features(stats), genders, seen, val_targets, config, seed + fold, device
        )
        catalog = FeatureCatalog(stats, popularity, genders, source, candidate)
        indices = np.flatnonzero(held_out)
        for start in range(0, len(indices), 65536):
            block = indices[start : start + 65536]
            matrix[block] = catalog.matrix(edges[block])
        summaries.append({"held_out_edges": len(indices), "best_tower_epoch": epoch})
    return matrix, summaries


def fit_ranker(matrix, labels, val_matrix, val_labels, settings, seed, threads):
    training = lgb.Dataset(matrix, label=labels, feature_name=list(FEATURE_NAMES),
                           categorical_feature=["gender_pair"])
    validation = lgb.Dataset(val_matrix, label=val_labels, reference=training)
    return lgb.train(
        {
            "objective": "binary", "metric": "auc", "learning_rate": settings["learning_rate"],
            "num_leaves": settings["num_leaves"], "num_threads": threads, "seed": seed,
            "deterministic": True, "force_col_wise": True, "verbosity": -1,
        },
        training, num_boost_round=settings["rounds"], valid_sets=[validation],
        callbacks=[lgb.early_stopping(settings["patience"]), lgb.log_evaluation(10)],
    )


def read_training(prepared):
    ids, genders, _ = read_users(prepared / "user_features")
    train = pq.read_table(prepared / "train", columns=["src", "dst", "rating", "liked"])
    edges = edge_rows(train, ids)
    ratings = train["rating"].to_numpy()
    liked = train["liked"].to_numpy() == 1
    del train
    stats, popularity = user_statistics(edges, ratings, liked, len(ids))
    return ids, genders, edges, ratings, liked, stats, popularity


def run(prepared, retrieval, output, config, device, threads):
    seed = config["split"]["seed"]
    settings = config["ranking"]
    ids, genders, edges, ratings, liked, stats, popularity = read_training(prepared)
    catalog = load_full_catalog(retrieval / "embeddings.npz", ids, genders, stats, popularity)
    validation = read_edges(prepared / "val", ids)
    matrix, folds = cross_fit(edges, ratings, liked, genders, validation, config["retrieval"], seed, device)
    val_edges, val_liked = validation
    model = fit_ranker(matrix, liked, catalog.matrix(val_edges), val_liked, settings, seed, threads)
    del matrix

    test_edges, test_liked = read_edges(prepared / "test", ids)
    report = evaluate(model, catalog, test_edges, test_liked, settings["k"], settings["alpha"])
    model.save_model(str(output / "ranker.txt"))
    return {
        "seed": seed,
        "alpha": settings["alpha"],
        "training_edges": len(edges),
        "validation_edges": len(val_edges),
        "cross_fit_folds": folds,
        "best_iteration": model.best_iteration,
        "validation_auc": model.best_score["valid_0"]["auc"],
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--retrieval", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/ranking"))
    parser.add_argument("--report", type=Path, default=Path("results/ranking.json"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.artifacts.exists() or args.threads < 1:
        parser.error("Choose a fresh artifacts directory and a positive thread count")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.artifacts.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ranking-", dir=args.artifacts.parent) as directory:
        staging = Path(directory) / "artifacts"
        staging.mkdir()
        report = run(args.prepared, args.retrieval, staging, config, torch.device(args.device), args.threads)
        staging.rename(args.artifacts)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    output = json.dumps(report, indent=2) + "\n"
    args.report.write_text(output, encoding="utf-8")
    print(output, end="", flush=True)


if __name__ == "__main__":
    main()
