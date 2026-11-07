"""Train independent source and candidate encoders with in-batch negatives."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from reciprec.retrieval.index import CandidateIndexes


class Tower(nn.Module):
    def __init__(self, user_count, feature_count, dimension):
        super().__init__()
        self.ids = nn.Embedding(user_count, dimension, sparse=True)
        self.mlp = nn.Sequential(
            nn.Linear(dimension + feature_count, dimension), nn.ReLU(),
            nn.Linear(dimension, dimension),
        )

    def forward(self, users, features):
        return self.mlp(torch.cat((self.ids(users), features[users]), dim=1))


class TwoTower(nn.Module):
    def __init__(self, user_count, feature_count, dimension):
        super().__init__()
        self.source = Tower(user_count, feature_count, dimension)
        self.candidate = Tower(user_count, feature_count, dimension)

    @torch.inference_mode()
    def encode(self, features):
        self.eval()
        users = torch.arange(len(features), device=features.device)
        return tuple(tower(users, features).cpu().numpy() for tower in (self.source, self.candidate))


def in_batch_loss(model, batch, features, target_probabilities):
    targets, labels = torch.unique(batch[:, 1], return_inverse=True)
    source = model.source(batch[:, 0], features)
    candidate = model.candidate(targets, features)
    # A target is included once after deduplication, with probability 1 - (1-p)^B.
    log_q = torch.log(-torch.expm1(len(batch) * torch.log1p(-target_probabilities[targets])))
    return F.cross_entropy(source @ candidate.T - log_q, labels)


def sample_queries(targets, limit, seed):
    if limit < 1 or not targets:
        raise ValueError("Evaluation requires a positive query limit and eligible positives")
    users = np.array(sorted(targets), dtype=np.int32)
    if len(users) > limit:
        users = np.sort(np.random.default_rng(seed).choice(users, size=limit, replace=False))
    return users


def macro_recall(results, targets):
    return float(np.mean([
        np.isin(retrieved, targets[user]).sum() / len(targets[user])
        for user, retrieved in results.items()
    ]))


def train_model(positives, features, genders, seen, validation, settings, seed, device):
    if not len(positives):
        raise ValueError("No positive training edges")
    if min(settings["epochs"], settings["patience"], settings["batch_size"]) < 1:
        raise ValueError("Epochs, patience, and batch size must be positive")
    torch.manual_seed(seed)
    model = TwoTower(len(features), features.shape[1], settings["dimension"]).to(device)
    features = torch.as_tensor(features, device=device)
    edges = torch.as_tensor(positives, dtype=torch.long, device=device)
    probabilities = torch.bincount(edges[:, 1], minlength=len(features)).float() / len(edges)
    embedding_optimizer = torch.optim.SparseAdam(
        [model.source.ids.weight, model.candidate.ids.weight], lr=settings["learning_rate"]
    )
    dense_optimizer = torch.optim.AdamW(
        [*model.source.mlp.parameters(), *model.candidate.mlp.parameters()],
        lr=settings["learning_rate"],
    )
    users = sample_queries(validation, settings["validation_queries"], seed)
    history = []
    best_recall = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        order = torch.randperm(len(edges), device=device)
        total_loss = torch.zeros((), device=device)
        for batch_rows in order.split(settings["batch_size"]):
            loss = in_batch_loss(model, edges[batch_rows], features, probabilities)
            embedding_optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            embedding_optimizer.step()
            dense_optimizer.step()
            total_loss += loss.detach() * len(batch_rows)

        source, candidate = model.encode(features)
        index = CandidateIndexes.build(candidate, genders)
        recall = macro_recall(index.recommend(users, source, seen, settings["k"]), validation)
        average_loss = float(total_loss.item() / len(edges))
        history.append({"epoch": epoch, "loss": average_loss, "recall": recall})
        print(f"Epoch {epoch}: loss={average_loss:.4f} validation recall={recall:.4f}", flush=True)
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch - best_epoch >= settings["patience"]:
            break
    model.load_state_dict(best_state)
    return model, model.encode(features), best_epoch, history, len(users)
