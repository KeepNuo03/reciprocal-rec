import numpy as np
import pytest
import torch

from reciprec.retrieval.data import evaluation_targets, group_targets
from reciprec.retrieval.index import CandidateIndexes
from reciprec.retrieval.two_tower import TwoTower, in_batch_loss, macro_recall


@pytest.mark.parametrize("hnsw_m", [None, 8])
def test_search_refills_after_filtering_and_round_trips(tmp_path, hnsw_m):
    genders = np.array(["F", "F", *(["M"] * 80)])
    vectors = np.arange(len(genders), 0, -1, dtype=np.float32).reshape(-1, 1)
    seen = group_targets(np.array([(0, i) for i in range(2, 80)]), len(genders))
    indexes = CandidateIndexes.build(vectors, genders, hnsw_m=hnsw_m)
    indexes.save(tmp_path)
    loaded = CandidateIndexes.load(tmp_path, genders)

    results = loaded.recommend([0, 1, 2], vectors, seen, k=3)
    np.testing.assert_array_equal(results[0], [80, 81])
    np.testing.assert_array_equal(results[1], [2, 3, 4])
    np.testing.assert_array_equal(results[2], [0, 1])


def test_recall_uses_only_eligible_positive_targets():
    genders = np.array(["F", "M", "F", "M", "M"])
    seen = group_targets(np.array([[0, 1], [3, 2]]), len(genders))
    edges = np.array([[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [3, 2]])
    targets = evaluation_targets(edges, np.array([True, True, True, True, False, True]), genders, seen)
    assert list(targets) == [0]
    np.testing.assert_array_equal(targets[0], [3])
    assert macro_recall({0: np.array([3, 4])}, targets) == 1.0
    assert macro_recall({0: np.array([4])}, targets) == 0.0


def test_loss_deduplicates_batch_targets_and_applies_logq():
    model = TwoTower(4, 2, 3)
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    batch = torch.tensor([[0, 1], [2, 1], [0, 3]])
    probabilities = torch.tensor([0.0, 0.75, 0.0, 0.25])
    loss = in_batch_loss(model, batch, torch.zeros((4, 2)), probabilities)
    inclusion = 1 - (1 - probabilities[[1, 3]]) ** len(batch)
    expected = -torch.log_softmax(-torch.log(inclusion), dim=0)[[0, 0, 1]].mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.source.ids.weight.grad.is_sparse
    assert model.candidate.ids.weight.grad.is_sparse
