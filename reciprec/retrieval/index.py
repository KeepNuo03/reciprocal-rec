"""Gender-partitioned FAISS indexes for inner-product retrieval."""

import faiss
import numpy as np


class CandidateIndexes:
    def __init__(self, genders, indexes):
        self.genders = genders
        self.indexes = indexes
        self.rows = {gender: np.flatnonzero(genders == gender).astype(np.int32) for gender in indexes}

    @classmethod
    def build(cls, vectors, genders, *, hnsw_m=None, ef_construction=120, ef_search=256):
        dimension = vectors.shape[1]
        indexes = {}
        for gender in ("F", "M"):
            candidates = np.ascontiguousarray(vectors[genders == gender], dtype=np.float32)
            if not len(candidates):
                continue
            if hnsw_m is None:
                index = faiss.IndexFlatIP(dimension)
            else:
                # Equalize candidate norms with an extra coordinate. Appending zero
                # to queries makes squared L2 rank candidates by the original dot product.
                norms = np.einsum("ij,ij->i", candidates, candidates)
                candidates = np.column_stack((candidates, np.sqrt(norms.max() - norms)))
                index = faiss.IndexHNSWFlat(dimension + 1, hnsw_m, faiss.METRIC_L2)
                index.hnsw.efConstruction = ef_construction
                index.hnsw.efSearch = ef_search
            index.add(candidates)
            indexes[gender] = index
        return cls(genders, indexes)

    def save(self, directory):
        for gender, index in self.indexes.items():
            faiss.write_index(index, str(directory / f"candidates_{gender}.faiss"))

    @classmethod
    def load(cls, directory, genders):
        indexes = {
            gender: faiss.read_index(str(directory / f"candidates_{gender}.faiss"))
            for gender in np.unique(genders)
        }
        result = cls(genders, indexes)
        if any(index.ntotal != len(result.rows[gender]) for gender, index in indexes.items()):
            raise ValueError("Index and user catalog sizes differ")
        return result

    def recommend(self, users, query_vectors, seen, k):
        """Return candidate rows keyed by source row, using opposite-gender eligibility."""
        if k < 1:
            raise ValueError("k must be positive")
        users = np.asarray(users, dtype=np.int32)
        result = {int(user): np.empty(0, dtype=np.int32) for user in users}
        for gender, index in self.indexes.items():
            sources = users[self.genders[users] != gender]
            for start in range(0, len(sources), 128):
                batch = sources[start : start + 128]
                self._recommend_batch(index, self.rows[gender], batch, query_vectors, seen, k, result)
        return result

    @staticmethod
    def _recommend_batch(index, rows, batch, query_vectors, seen, k, result):
        # Expand the search only for queries with too few eligible candidates.
        fetch = min(len(rows), k + 32)
        is_hnsw = isinstance(index, faiss.IndexHNSW)
        while len(batch):
            queries = np.ascontiguousarray(query_vectors[batch], dtype=np.float32)
            if is_hnsw and fetch == len(rows):
                # A graph search may still omit unreachable vectors at full depth.
                for user, query in zip(batch, queries):
                    positions = np.flatnonzero((rows != user) & ~np.isin(rows, seen[user]))
                    if len(positions):
                        scores = index.reconstruct_batch(positions)[:, : len(query)] @ query
                        best = np.argsort(-scores, kind="stable")[:k]
                        result[int(user)] = rows[positions[best]]
                return
            if is_hnsw:
                params = faiss.SearchParametersHNSW()
                params.efSearch = max(index.hnsw.efSearch, fetch)
                lifted_queries = np.pad(queries, ((0, 0), (0, 1)))
                _, matches = index.search(lifted_queries, fetch, params=params)
            else:
                _, matches = index.search(queries, fetch)
            retry = []
            for user, matches_for_user in zip(batch, matches):
                candidates = rows[matches_for_user[matches_for_user >= 0]]
                allowed = candidates[(candidates != user) & ~np.isin(candidates, seen[user])]
                result[int(user)] = allowed[:k]
                if len(allowed) < k and fetch < len(rows):
                    retry.append(user)
            batch = np.asarray(retry, dtype=np.int32)
            fetch = min(len(rows), fetch * 2)
