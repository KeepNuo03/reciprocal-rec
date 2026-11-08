import numpy as np
from fastapi.testclient import TestClient

from reciprec.ranking.features import FeatureCatalog
from reciprec.retrieval.index import CandidateIndexes
from reciprec.serving.app import Recommender, create_app


def test_recommend_filters_and_handles_cold_start():
    ids = np.array([10, 20, 30, 40])
    genders = np.array(["F", "M", "M", "F"])
    vectors = np.arange(1, 5, dtype=np.float32).reshape(-1, 1)
    catalog = FeatureCatalog(np.zeros((4, 6)), np.array([1, 8, 3, 2]), genders, vectors, vectors)

    class Model:
        def predict(self, matrix):
            return np.full(len(matrix), 0.5)

    engine = Recommender(
        ids, catalog, CandidateIndexes.build(vectors, genders, hnsw_m=8),
        [np.array([1]), [], [], []], Model(), recall_k=200, alpha=0.3,
    )
    with TestClient(create_app(lambda: engine)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        result = client.get("/recommend/10?k=1&alpha=0").json()
        assert not result["fallback"]
        assert result["recommendations"] == [
            {"user_id": 30, "p_forward": 0.5, "p_reverse": 0.5, "score": 0.25}
        ]
        assert set(result["timings_ms"]) == {"retrieval", "features", "ranking"}
        cold = client.get("/recommend/999?k=2&preferred_gender=M").json()
        assert cold["fallback"]
        assert [item["user_id"] for item in cold["recommendations"]] == [20, 30]
        assert cold["recommendations"][0]["p_forward"] is None
        assert client.get("/recommend/10?k=0").status_code == 422
        assert client.get("/recommend/10?alpha=1").status_code == 422
