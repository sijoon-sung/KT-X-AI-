"""Small causal inference smoke test; does not require the Tashu dataset."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integrated_st_flow import IntegratedSTFlow


class IntegratedFlowTest(unittest.TestCase):
    def test_stock_only_inference_is_finite_and_has_station_outputs(self):
        torch.manual_seed(0)
        model = IntegratedSTFlow(4, np.eye(4, dtype=np.int32), lookback=3)
        stock = torch.tensor([[[2., 1., 0., 3.], [2., 1., 0., 3.],
                               [1., 1., 1., 3.], [1., 2., 1., 2.],
                               [0., 2., 2., 2.]]])
        hist_hour = torch.tensor([[8, 9, 10]])
        hist_weekday = torch.tensor([[1, 1, 1]])
        hist_month = torch.tensor([[2, 2, 2]])
        pred, baseline = model.predict(stock, hist_hour, hist_weekday, hist_month,
                                       torch.tensor([11]), torch.tensor([1]),
                                       torch.tensor([2]))
        self.assertEqual(tuple(pred.shape), (1, 4, 2))
        self.assertEqual(tuple(baseline.shape), (1, 4, 2))
        self.assertTrue(torch.isfinite(pred).all())
        self.assertTrue((pred >= 0).all())


if __name__ == '__main__':
    unittest.main()
