"""Serve recommendations from precomputed retrieval and ranking artifacts."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

import faiss
import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq
import yaml
from fastapi import FastAPI, Query

from reciprec.pipeline import pair_features, score_features
from reciprec.ranking.features import load_full_catalog
from reciprec.retrieval.data import FEATURES, group_targets, read_edges
from reciprec.retrieval.index import CandidateIndexes


class Recommender:
    def __init__(self, ids, catalog, indexes, seen, ranker, recall_k, alpha):
        self.ids = ids
        self.rows = {int(user_id): row for row, user_id in enumerate(ids)}
        self.catalog = catalog
        self.indexes = indexes
        self.seen = seen
        self.ranker = ranker
        self.recall_k = recall_k
        self.alpha = alpha

    def recommend(self, user_id, k, alpha, preferred_gender):
        start = perf_counter()
        row = self.rows.get(user_id)
        if row is None:
            eligible = np.arange(len(self.ids))
            if preferred_gender is not None:
                eligible = eligible[self.catalog.genders == preferred_gender]
            counts = self.catalog.likes_received[eligible]
            final = counts / (counts + 1.0) ** alpha
            best = np.lexsort((self.ids[eligible], -final))[:k]
            return {
                "fallback": True,
                "recommendations": [
                    {"user_id": int(self.ids[eligible[i]]), "p_forward": None, "p_reverse": None,
                     "score": float(final[i])}
                    for i in best
                ],
                "timings_ms": {"retrieval": 1000 * (perf_counter() - start), "features": 0, "ranking": 0},
            }

        candidates = self.indexes.recommend([row], self.catalog.u_out, self.seen, self.recall_k)[row]
        recalled = perf_counter()
        results = []
        features_done = recalled
        if len(candidates):
            edges = np.column_stack((np.full_like(candidates, row), candidates))
            matrices = pair_features(self.catalog, edges)
            features_done = perf_counter()
            scores, reverse = score_features(self.ranker, matrices, self.catalog.likes_received[candidates], alpha)
            final = scores["product_debiased"]
            best = np.lexsort((self.ids[candidates], -final))[:k]
            results = [
                {"user_id": int(self.ids[candidates[i]]), "p_forward": float(scores["one_way"][i]),
                 "p_reverse": float(reverse[i]), "score": float(final[i])}
                for i in best
            ]
        return {
            "fallback": False,
            "recommendations": results,
            "timings_ms": {
                "retrieval": 1000 * (recalled - start),
                "features": 1000 * (features_done - recalled),
                "ranking": 1000 * (perf_counter() - features_done),
            },
        }


def load_recommender():
    prepared = Path(os.getenv("RECIPREC_PREPARED", "data/prepared"))
    artifacts = Path(os.getenv("RECIPREC_ARTIFACTS", "artifacts"))
    config = Path(os.getenv("RECIPREC_CONFIG", "configs/default.yaml"))
    settings = yaml.safe_load(config.read_text(encoding="utf-8"))
    users = pq.read_table(prepared / "user_features", columns=["user_id", "gender", "likes_received", *FEATURES])
    order = np.argsort(users["user_id"].to_numpy())
    ids = users["user_id"].to_numpy()[order]
    genders = users["gender"].to_numpy()[order].astype("U1")
    popularity = users["likes_received"].to_numpy()[order]
    stats = np.column_stack([users[name].to_numpy() for name in FEATURES])[order].astype(np.float32)
    catalog = load_full_catalog(artifacts / "retrieval/embeddings.npz", ids, genders, stats, popularity)
    indexes = CandidateIndexes.load(artifacts / "retrieval", genders)
    edges, _ = read_edges(prepared / "train", ids)
    booster = lgb.Booster(model_file=str(artifacts / "ranking/ranker.txt"))
    faiss.omp_set_num_threads(1)
    return Recommender(ids, catalog, indexes, group_targets(edges, len(ids)),
                       SingleThreadPredictor(booster), settings["retrieval"]["k"], settings["ranking"]["alpha"])


class SingleThreadPredictor:
    def __init__(self, booster):
        self.booster = booster

    def predict(self, matrix):
        return self.booster.predict(matrix, num_threads=1)


def create_app(loader=load_recommender):
    @asynccontextmanager
    async def lifespan(app):
        app.state.recommender = loader()
        yield

    app = FastAPI(title="Reciprocal Recommendations", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/recommend/{user_id}")
    async def recommend(
        user_id: int,
        k: int = Query(default=10, ge=1, le=200),
        alpha: float | None = Query(default=None, ge=0, le=0.5),
        preferred_gender: Literal["F", "M"] | None = None,
    ):
        engine = app.state.recommender
        return engine.recommend(user_id, k, engine.alpha if alpha is None else alpha, preferred_gender)

    return app


app = create_app()
