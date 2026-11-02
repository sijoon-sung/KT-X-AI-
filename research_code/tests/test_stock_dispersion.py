import unittest
import numpy as np
from research_code.stock_dispersion import cohort, sufficient, station_ratios, block_weights, resample, decide


class StockDispersionTests(unittest.TestCase):
    def source(self):
        times = np.arange(24 * 6 * 3) * 600
        stock = np.full((len(times), 2), 3., dtype=np.float32)
        actual = np.broadcast_to(times[:, None], stock.shape).copy()
        return stock, times, actual

    def test_future_period_cannot_change_the_cohort(self):
        stock, times, actual = self.source()
        first = cohort(stock, times, actual, 0, 86400)
        stock[times >= 86400] = 999
        second = cohort(stock, times, actual, 0, 86400)
        np.testing.assert_array_equal(first['future'], second['future'])
        self.assertTrue((times[first['origins']] + 3600 < 86400).all())

    def test_missing_one_horizon_excludes_the_common_pair(self):
        stock, times, actual = self.source()
        stock[9, 1] = np.nan
        data = cohort(stock, times, actual, 0, 86400)
        row = np.flatnonzero(data['origins'] == 6)[0]
        self.assertEqual(data['valid'][row].tolist(), [True, False])

    def test_timestamp_tolerance_is_checked_at_each_horizon(self):
        stock, times, actual = self.source()
        actual[12, 1] += 121
        data = cohort(stock, times, actual, 0, 86400)
        row = np.flatnonzero(data['origins'] == 6)[0]
        self.assertFalse(data['valid'][row, 1])

    def test_current_state_conditioning_removes_between_state_variance(self):
        n = np.array([[4., 4.]])
        total = np.array([[8., 40.]])
        squares = np.array([[16., 400.]])
        ratio = station_ratios(n, total, squares, np.ones((1, 2), dtype=bool))
        self.assertEqual(ratio.item(), 0.)
        self.assertGreater(np.var([2.] * 4 + [10.] * 4), np.mean([2.] * 4 + [10.] * 4))

    def test_weighted_variance_ratio_and_zero_mean_are_not_imputed(self):
        n = np.array([[3., 2.], [2., 0.]])
        total = np.array([[6., 10.], [0., 0.]])
        squares = np.array([[14., 50.], [0., 0.]])
        ratio = station_ratios(n, total, squares, n >= 2)
        self.assertAlmostEqual(ratio[0], 3 / 16)
        self.assertTrue(np.isnan(ratio[1]))

    def test_sufficient_statistics_keep_extreme_values(self):
        data = dict(current=np.array([[2.], [2.]]), future=np.array([[[2.],[302.]]] * 3),
                    valid=np.ones((2, 1), dtype=bool), days=np.array([0, 1]))
        n, sums, squares = sufficient(data, days=2)
        self.assertEqual(n.sum(), 2)
        self.assertEqual(sums[1].sum(), 304)
        self.assertEqual(squares[1].sum(), 4 + 302**2)

    def test_block_replicates_keep_calendar_length_and_seed(self):
        weights = block_weights(28, 7, 100)
        np.testing.assert_array_equal(weights.sum(1), np.full(100, 28))
        np.testing.assert_array_equal(weights, block_weights(28, 7, 100))

    def test_reweighted_constant_stock_has_zero_variance_interval(self):
        n = np.full((28, 2, 1), 3.)
        result = resample(n, n * 3, n * 9, np.ones((2, 1), dtype=bool),
                          np.ones(2, dtype=bool), block=3, repetitions=20)
        self.assertEqual(result['valid_replicates'], 20)
        self.assertEqual(result['interval'], [0., 0.])
        self.assertEqual(result['minimum_station_coverage'], 1.)

    def test_decision_requires_both_blocks_and_sufficient_coverage(self):
        low = dict(valid_replicates=2000, interval=[.2, .7])
        high = dict(valid_replicates=2000, interval=[.9, 1.2])
        self.assertEqual(decide(200, 12000, [low, low]), 'supported')
        self.assertEqual(decide(200, 12000, [high, high]), 'not_supported')
        self.assertEqual(decide(200, 12000, [low, high]), 'inconclusive')
        self.assertEqual(decide(99, 12000, [low, low]), 'inconclusive')


if __name__ == '__main__':
    unittest.main()
