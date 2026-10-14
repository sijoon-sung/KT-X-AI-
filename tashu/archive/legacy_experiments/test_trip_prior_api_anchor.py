"""Inference contract: the prior has no inventory input; stock is an anchor."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trip_prior_api_anchor import TripPrior, stockout_probability


class TripPriorAnchorTest(unittest.TestCase):
    def test_prior_predicts_without_stock(self):
        model = TripPrior(np.array(['A', 'B']), np.array([[0, 1], [1, 0]]))
        out = model(torch.tensor([0, 1]), torch.tensor([8, 8]),
                    torch.tensor([0, 0]), torch.tensor([2, 2]))
        self.assertEqual(tuple(out.shape), (2, 2))
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out > 0).all())

    def test_current_stock_controls_simulated_stockout(self):
        rates = np.array([[4., 0.]], dtype=np.float32)
        dispersion = np.array([100., 100.])
        low = stockout_probability(rates, np.array([1.]), dispersion, draws=2048, seed=0)[0]
        high = stockout_probability(rates, np.array([20.]), dispersion, draws=2048, seed=0)[0]
        self.assertGreater(low, 0.8)
        self.assertLess(high, 0.01)


if __name__ == '__main__':
    unittest.main()
