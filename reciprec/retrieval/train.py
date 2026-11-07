"""Train the two-tower retriever, publish its indexes, and report Recall@K."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np
import torch
import yaml

from reciprec.retrieval.data import evaluation_targets, group_targets, read_edges, read_users
from reciprec.retrieval.index import CandidateIndexes
from reciprec.retrieval.two_tower import macro_recall, sample_queries, train_model


def evaluate(source, candidate, genders, seen, targets, settings, seed, output):
    k = settings["k"]
    index = CandidateIndexes.build(
        candidate, genders, hnsw_m=settings["hnsw_m"],
        ef_construction=settings["ef_construction"], ef_search=settings["ef_search"],
    )
    audit = sample_queries(targets, settings["index_audit_queries"], seed)
    approximate = index.recommend(list(targets), source, seen, k)
    exact_index = CandidateIndexes.build(candidate, genders)
    exact = exact_index.recommend(audit, source, seen, k)
    index_overlap = np.mean([
        np.isin(approximate[user], exact[user]).sum() / len(exact[user]) for user in audit
    ])
    index.save(output)
    return {
        f"test_recall@{k}": macro_recall(approximate, targets),
        "test_queries": len(targets),
        "test_eligible_positives": sum(map(len, targets.values())),
        "index_audit_queries": len(audit),
        f"hnsw_exact_overlap@{k}": float(index_overlap),
    }


def run(prepared, output, config, device):
    settings = config["retrieval"]
    seed = config["split"]["seed"]
    ids, genders, features = read_users(prepared / "user_features")
    train, liked = read_edges(prepared / "train", ids)
    seen = group_targets(train, len(ids))
    positives = train[liked]
    positives = positives[np.lexsort((positives[:, 1], positives[:, 0]))]
    del train, liked
    validation = evaluation_targets(*read_edges(prepared / "val", ids), genders, seen)
    model, (source, candidate), epoch, history, val_queries = train_model(
        positives, features, genders, seen, validation, settings, seed, device
    )

    test = evaluation_targets(*read_edges(prepared / "test", ids), genders, seen)
    report = evaluate(source, candidate, genders, seen, test, settings, seed, output)
    np.savez(output / "embeddings.npz", user_ids=ids, genders=genders, u_out=source, u_in=candidate)
    torch.save({name: tensor.cpu() for name, tensor in model.state_dict().items()}, output / "model.pt")
    return {
        "device": str(device),
        "seed": seed,
        "settings": settings,
        "users": len(ids),
        "training_positives": len(positives),
        "best_epoch": epoch,
        "validation_queries": val_queries,
        "validation_history": history,
        **report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/retrieval"))
    parser.add_argument("--report", type=Path, default=Path("results/retrieval.json"))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.artifacts.exists() or args.threads < 1:
        parser.error("Choose a fresh artifacts directory and a positive thread count")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch installation")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.artifacts.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".retrieval-", dir=args.artifacts.parent) as directory:
        staging = Path(directory) / "artifacts"
        staging.mkdir()
        report = run(args.prepared, staging, config, torch.device(device))
        staging.rename(args.artifacts)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
