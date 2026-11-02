import unittest
import numpy as np
import torch
from research_code.architecture_benchmark import Flow, mix, sparse_adjacency


class FlowTests(unittest.TestCase):
    def test_unknown_future_target_cannot_change_the_training_loss(self):
        torch.set_num_threads(4)
        torch.manual_seed(4)
        model = Flow(3)
        x = torch.zeros(2, 3, 18)
        adj = sparse_adjacency(np.ones((3, 3), np.float32) / 3, 'cpu')
        valid = torch.ones(2, 3, dtype=torch.bool)
        valid[:, 1] = False
        y1 = torch.ones(2, 3)
        y2 = y1.clone()
        y2[:, 1] = 1000
        torch.manual_seed(5)
        first = model.loss(x, y1, valid, adj)
        torch.manual_seed(5)
        second = model.loss(x, y2, valid, adj)
        torch.testing.assert_close(first, second)
        first.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_gaussian_sampling_is_reproducible_and_returns_valid_counts(self):
        torch.set_num_threads(4)
        torch.manual_seed(6)
        model = Flow(3)
        x = torch.zeros(2, 3, 18)
        adj = sparse_adjacency(np.eye(3, dtype=np.float32), 'cpu')
        torch.manual_seed(7)
        first = model.generate(x, adj, 4, 2)
        torch.manual_seed(7)
        second = model.generate(x, adj, 4, 2)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
            self.assertEqual(tuple(a.shape), (2, 3))
            self.assertTrue((a >= 0).all())
        self.assertTrue((first[0] <= 1).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_graph_propagation_matches_cpu(self):
        array = np.array([[.5, .5], [.25, .75]], np.float32)
        h = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        expected = mix(sparse_adjacency(array, 'cpu'), h)
        actual = mix(sparse_adjacency(array, 'cuda'), h.cuda()).cpu()
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
