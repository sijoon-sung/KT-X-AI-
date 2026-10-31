"""H-0005 synthetic numeric/leakage checks; run on dislab, no real API."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import history_to_stock as h


class HistoryToStockTests(unittest.TestCase):
    def test_zero_rates_and_pure_arrival(self):
        k = h.numeric_contract([0, 3])['cutoff']
        p, mean = h.transition_tables(np.array([0., 3.]), np.zeros(2), k)
        np.testing.assert_allclose(mean[0], np.arange(k+1), atol=1e-10)
        np.testing.assert_array_equal(p[0], np.arange(k+1) == 0)
        self.assertAlmostEqual(p[1, 0], np.exp(-1.5), places=10)
        np.testing.assert_allclose(p[1, 1:], 0, atol=1e-12)
        np.testing.assert_allclose(mean[1], np.arange(k+1)+1.5, atol=1e-9)

    def test_pure_departure_analytic_absorption(self):
        k = h.numeric_contract([4])['cutoff']
        p, mean = h.transition_tables(np.array([0.]), np.array([4.]), k)
        self.assertAlmostEqual(p[0, 0], 1., places=10)
        self.assertAlmostEqual(p[0, 1], 1-np.exp(-2), places=10)
        self.assertAlmostEqual(mean[0, 1], np.exp(-2), places=10)
        self.assertAlmostEqual(mean[0, 2], 4*np.exp(-2), places=10)
        self.assertAlmostEqual(mean[0, 0], 0., places=10)

    def test_two_sided_rates_against_independent_matrix_exponential(self):
        a, d, size = 2.3, 4.1, 128
        k = h.numeric_contract([a+d])['cutoff']
        p, mean = h.transition_tables(np.array([a]), np.array([d]), k)
        q = np.zeros((size, size))
        for i in range(size):
            if i < size-1:
                q[i, i+1] = a
            if i > 0:
                q[i, i-1] = d
            q[i, i] = -q[i].sum()
        transition = expm(.5*q)
        np.testing.assert_allclose(p[0, :9], transition[:9, 0], atol=1e-10)
        np.testing.assert_allclose(mean[0, :9], transition[:9]@np.arange(size), atol=1e-9)

    def test_high_stock_not_capped_and_rate_directions(self):
        rates = np.array([[2., 8.]])  # rentals then returns
        k = h.numeric_contract([10])['cutoff']
        tables = h.transition_tables(rates[:, 1], rates[:, 0], k)
        p, mean = h.forecast(np.array([[1000.]]), np.array([0]), np.array([7]),
                            np.array([0]), rates, tables, k, 'constant')
        self.assertEqual(p[0, 0], 0)
        self.assertAlmostEqual(mean[0, 0], 1003, places=8)

    def test_calendar_lookup_and_future_label_never_enters_forecast(self):
        rates = np.zeros((7, 24, 2, 2))
        rates[1, 8, 1] = [4., 0.]
        k = h.numeric_contract([4])['cutoff']
        tables = h.transition_tables(rates[:, 7:22, :, 1], rates[:, 7:22, :, 0], k)
        p, mean = h.forecast(np.array([[1.]]), np.array([1]), np.array([8]),
                            np.array([1]), rates, tables, k, 'calendar')
        self.assertAlmostEqual(p[0, 0], 1-np.exp(-2), places=10)
        self.assertAlmostEqual(mean[0, 0], np.exp(-2), places=10)

    def panel(self):
        first = int(h.datetime(2026, 9, 30, 7, tzinfo=h.KST).timestamp())
        times = np.arange(first, h.END+3600, 600, dtype=np.int64)
        stock = np.ones((len(times), 2))
        actual = np.broadcast_to(times[:, None], stock.shape).copy()+30
        return stock, times, actual, np.array(['B', 'A']), np.array(['A', 'B', 'C'])

    def test_september_only_missing_not_zero_exact_id_order_and_jitter(self):
        stock, times, actual, api, trained = self.panel()
        stock[3, 1] = np.nan
        actual[3, 1] = -1
        actual[3, 0] = times[3]+121
        data = h.pair_data(stock, times, actual, api, trained)
        self.assertEqual(data['scope']['matched_ids'], 2)
        self.assertEqual(data['scope']['unmatched_trained_ids'], 1)
        np.testing.assert_array_equal(data['model_cols'], [0, 1])
        self.assertFalse(data['valid'][0].any())
        self.assertTrue(np.isnan(data['future'][0, 0]))
        self.assertTrue((data['times']+1800 < h.END).all())
        self.assertTrue((data['hour'] <= 21).all())
        self.assertEqual(data['scope']['missing_current_or_target'], 1)
        self.assertEqual(data['scope']['invalid_timing_with_observed_values'], 1)

    def test_october_values_irrelevant_and_future_labels_do_not_change_current(self):
        stock, times, actual, api, trained = self.panel()
        before = h.pair_data(stock, times, actual, api, trained)
        stock[times >= h.END] = -999.5
        actual[times >= h.END] = -1
        after = h.pair_data(stock, times, actual, api, trained)
        for name in ('current', 'future', 'valid', 'model_cols', 'times'):
            np.testing.assert_array_equal(before[name], after[name])
        stock[3] = 40
        changed = h.pair_data(stock, times, actual, api, trained)
        np.testing.assert_array_equal(before['current'], changed['current'])
        self.assertFalse(np.array_equal(before['future'], changed['future']))

    def test_duplicate_ids_and_bad_selected_stock_stop(self):
        stock, times, actual, api, trained = self.panel()
        with self.assertRaises(ValueError):
            h.pair_data(stock, times, actual, np.array(['A', 'A']), trained)
        stock[0, 0] = -1
        with self.assertRaises(ValueError):
            h.pair_data(stock, times, actual, api, trained)

    def test_numeric_limits_and_invalid_rates(self):
        for rates in ([np.nan], [-1], [2000]):
            with self.assertRaises(ValueError):
                h.numeric_contract(rates)
        for duration in (-1, np.nan):
            with self.assertRaises(ValueError):
                h.numeric_contract([1], duration)

    def test_bootstrap_keeps_empty_calendar_dates_reproducible_and_gates(self):
        daily = np.zeros((14, 3))
        daily[3] = [20., 10., 100.]
        first = h.bootstrap(daily, 3)
        self.assertEqual(first, h.bootstrap(daily, 3))
        self.assertLess(first['valid_replicates'], 2000)
        np.testing.assert_allclose(first['interval'], [.1, .1])
        good = [dict(valid_replicates=2000, interval=[.01, .02])]*2
        self.assertEqual(h.decision(500, 10000, 10, .015, good), 'supported')
        self.assertEqual(h.decision(499, 10000, 10, .015, good), 'inconclusive')
        bad = [dict(valid_replicates=2000, interval=[-.01, .005])]*2
        self.assertEqual(h.decision(500, 10000, 10, .001, bad), 'not_supported')


if __name__ == '__main__':
    unittest.main()
