import math
import unittest
import numpy as np
import torch
from research_code.architecture_benchmark import origins, station_mask, split_data, graph, sparse_adjacency, mix


class ArchitectureTests(unittest.TestCase):
    def test_future_observations_cannot_change_station_selection(self):
        times = np.arange(60) * 600
        stock = np.ones((60, 2))
        metadata = {'a': [36., 127.], 'b': [36.1, 127.1]}
        first = station_mask(stock, times, 0, 12000, ['a', 'b'], metadata)
        stock[20:] = np.nan
        second = station_mask(stock, times, 0, 12000, ['a', 'b'], metadata)
        np.testing.assert_array_equal(first, second)

    def test_origins_leave_room_for_own_history_and_target(self):
        times = np.arange(24 * 6) * 600
        selected = origins(times, 0, 86400)
        self.assertTrue((times[selected] - 3000 >= 0).all())
        self.assertTrue((times[selected] + 1800 < 86400).all())
        self.assertEqual(len(selected), 23)

    def test_missing_is_distinguishable_from_observed_zero(self):
        times = np.arange(30) * 600
        stock = np.ones((30, 2), dtype=np.float32)
        stock[6, 0] = 0
        stock[6, 1] = np.nan
        actual = np.broadcast_to(times[:, None], stock.shape).copy()
        data = split_data(stock, times, actual, np.array([6]), np.zeros((2, 2)))
        self.assertEqual(data['x'][0, 0, 0], 0)
        self.assertEqual(data['x'][0, 1, 0], 0)
        self.assertEqual(data['x'][0, 0, 6], 1)
        self.assertEqual(data['x'][0, 1, 6], 0)
        self.assertEqual(data['valid'].tolist(), [[True, False]])

    def test_request_time_tolerance_masks_the_target(self):
        times = np.arange(30) * 600
        stock = np.ones((30, 1), dtype=np.float32)
        actual = times[:, None].copy()
        actual[9, 0] += 121
        data = split_data(stock, times, actual, np.array([6]), np.zeros((1, 2)))
        self.assertFalse(data['valid'].item())

    def test_graph_control_preserves_degree_and_weight_not_geography(self):
        rng = np.random.default_rng(4)
        coords = rng.normal(size=(12, 2)) / 100 + [36, 127]
        geographic, shuffled = graph(coords, 'geo'), graph(coords, 'shuffled')
        np.testing.assert_allclose(geographic.sum(1), 1)
        np.testing.assert_allclose(shuffled.sum(1), 1)
        np.testing.assert_allclose(geographic.diagonal(), .5)
        np.testing.assert_array_equal((geographic > 0).sum(1), (shuffled > 0).sum(1))
        self.assertFalse(np.array_equal(geographic, shuffled))

    def test_sparse_mix_keeps_city_batches_separate_and_has_gradients(self):
        array = np.array([[.5, .5], [.25, .75]], np.float32)
        h = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).requires_grad_()
        result = mix(sparse_adjacency(array, 'cpu'), h)
        expected = torch.einsum('ij,bjd->bid', torch.from_numpy(array), h)
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        self.assertTrue(torch.isfinite(h.grad).all())

    def test_dequantization_zero_threshold_matches_count_bin(self):
        counts = np.array([0., 0., 1., 1., 3.])
        noise = np.array([0., .999, 0., .5, .2])
        z = np.log1p(counts + noise)
        np.testing.assert_array_equal(z < math.log(2), counts == 0)
        np.testing.assert_array_equal(np.maximum(0, np.floor(np.expm1(z))), counts)


if __name__ == '__main__':
    unittest.main()
