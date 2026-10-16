import numpy as np
import torch

from tashu_contest.models import CountGCN, GraphFlow, graph_multiply, graph_tensor, seed_everything, train_xgboost, tabular_features
from tashu_contest.evaluation import count_metrics


def test_flow_output_is_reproducible_nonnegative_integer_scenarios():
    seed_everything(5)
    base = CountGCN(12, 8, 4)
    model = GraphFlow(base, 8, 4)
    x, graph = torch.zeros(2, 3, 12), torch.eye(3)
    a = model.sample(x, graph, 5, 2, torch.Generator().manual_seed(9))
    b = model.sample(x, graph, 5, 2, torch.Generator().manual_seed(9))
    assert a.shape == (5, 2, 3, 4)
    assert torch.isfinite(a).all() and (a >= 0).all()
    torch.testing.assert_close(a, torch.floor(a))
    torch.testing.assert_close(a, b)


def test_sparse_and_dense_graph_propagation_agree():
    rng = np.random.default_rng(0)
    graph = np.eye(70, dtype=np.float32)
    graph[0, 1] = .5
    x = torch.tensor(rng.normal(size=(3, 70, 4)), dtype=torch.float32)
    torch.testing.assert_close(graph_multiply(graph_tensor(graph, "cpu"), x), torch.tensor(graph) @ x)


def test_empirical_crps_matches_brute_force_formula():
    samples = np.array([0, 1, 4], dtype=float).reshape(3, 1, 1, 1, 1)
    actual = np.array([2.]).reshape(1, 1, 1, 1)
    metrics = count_metrics(samples, actual, np.ones_like(actual, dtype=bool))
    flat = samples.ravel()
    expected = np.abs(flat - 2).mean() - .5 * np.abs(flat[:, None] - flat[None, :]).mean()
    assert abs(metrics["crps"] - expected) < 1e-9


def test_poisson_baseline_can_train_an_all_zero_channel():
    x = np.zeros((8, 2, 4), dtype=np.float32)
    y = np.zeros((8, 2, 2), dtype=np.float32)
    models = train_xgboost(x, y, np.ones_like(y, dtype=bool), np.arange(6), np.arange(6, 8), seed=1, trees=2)
    prediction = models[0].predict(tabular_features(x))
    assert np.isfinite(prediction).all() and (prediction >= 0).all()


def test_medium_has_more_capacity_without_changing_frozen_base_policy():
    from tashu_contest.pipeline import parameter_counts
    base = CountGCN(24, 48, 6)
    medium = CountGCN(24, 96, 6)
    flow = GraphFlow(medium, 96, 6)
    small_count, medium_count = parameter_counts(base), parameter_counts(medium)
    assert 3 < medium_count["total"] / small_count["total"] < 4.5
    assert parameter_counts(flow)["frozen"] == medium_count["total"]
    assert parameter_counts(flow)["trainable"] > 0
