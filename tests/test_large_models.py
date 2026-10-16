import copy

import numpy as np
import pytest
import torch

from tashu_contest.models import CountGCN, GraphFlow, graph_tensor, seed_everything, train_neural
from tashu_contest.profiles import PROFILES


def test_deep_flow_trains_while_frozen_encoder_dropout_stays_off():
    seed_everything(2)
    base = CountGCN(12, 16, 4, layers=4, dropout=.2)
    flow = GraphFlow(base, 16, 4, blocks=3, expansion=2, dropout=.1)
    flow.train()
    x, graph = torch.randn(2, 5, 12), torch.eye(5)
    assert flow.training and not flow.base.training and not flow.base.encoder.dropout.training
    torch.testing.assert_close(flow.base.encoder(x, graph), flow.base.encoder(x, graph))
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    mask[:, 0] = False
    loss = flow.objective(x, torch.ones(2, 5, 4), mask, graph)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None for p in flow.base.parameters())
    assert flow.blocks[0].ffn[0].weight.grad is not None
    assert torch.isfinite(flow.blocks[0].ffn[0].weight.grad).all()


def test_chunking_preserves_prior_and_integer_scenarios():
    seed_everything(5)
    flow = GraphFlow(CountGCN(8, 16, 4), 16, 4, blocks=3)
    x, graph = torch.ones(2, 7, 8), torch.eye(7)
    together = flow.sample(x, graph, 7, 3, torch.Generator().manual_seed(10))
    chunked = flow.sample(x, graph, 7, 3, torch.Generator().manual_seed(10), chunk_size=2)
    assert chunked.device.type == "cpu"
    torch.testing.assert_close(together, chunked, rtol=0, atol=0)


def test_accumulation_weights_observed_cells_instead_of_microbatch_means():
    seed_everything(3)
    x = np.random.normal(size=(5, 3, 8)).astype(np.float32)
    y = np.random.poisson(1, size=(5, 3, 2)).astype(np.float32)
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0:2] = False
    mask[1, 1:] = False
    a = CountGCN(8, 8, 2)
    b = copy.deepcopy(a)
    train_neural(a, x, y, mask, np.arange(4), np.array([4]), np.eye(3), 1, 4, .001, 7)
    train_neural(b, x, y, mask, np.arange(4), np.array([4]), np.eye(3), 1, 1, .001, 7, accumulate=4)
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=2e-5, atol=2e-6)


def test_rtx_capacity_is_substantial_and_microbatch_is_bounded():
    p = PROFILES["rtx3080"]
    base = CountGCN(24, p["hidden"], 6, layers=p["gcn_layers"], dropout=p["dropout"])
    flow = GraphFlow(base, p["hidden"], 6, blocks=p["flow_blocks"], expansion=p["expansion"], dropout=p["dropout"])
    assert 7_000_000 < sum(t.numel() for t in flow.parameters()) < 15_000_000
    assert p["batch_size"] * p["accumulate"] == 16
    assert p["sample_chunk"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an actual CUDA-enabled PyTorch/GPU")
def test_cuda_amp_sparse_graph_and_chunked_inference():
    seed_everything(2)
    x = np.random.normal(size=(4, 70, 8)).astype(np.float32)
    y = np.ones((4, 70, 2), dtype=np.float32)
    base = CountGCN(8, 16, 2, layers=3, dropout=.1).cuda()
    flow = GraphFlow(base, 16, 2, blocks=3, dropout=.1).cuda()
    logs = train_neural(flow, x, y, np.ones_like(y, dtype=bool), np.arange(3), np.array([3]),
                        np.eye(70, dtype=np.float32), 1, 1, .001, 2, flow=True, accumulate=2, amp=True)
    samples = flow.sample(torch.tensor(x[:1], device="cuda"), graph_tensor(np.eye(70), "cuda"),
                          3, 2, torch.Generator(device="cuda").manual_seed(3), chunk_size=1, amp=True)
    assert logs[0]["amp"] and logs[0]["peak_cuda_allocated_bytes"] > 0
    assert torch.isfinite(samples).all() and samples.device.type == "cpu"
