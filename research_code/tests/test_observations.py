import unittest
import numpy as np
from research_code.observation_resolution import day_shifts, bootstrap, decision


class ObservationTests(unittest.TestCase):
    def test_missing_station_day_is_not_zero_or_complete(self):
        data = np.ones((1, 144, 2))
        data[0, 0, 1] = np.nan
        shifts, eligible = day_shifts(data)
        self.assertEqual(eligible.tolist(), [[True, False]])
        self.assertTrue(np.isnan(shifts[0, 1]))

    def test_aliasing_absolute_difference_does_not_cancel(self):
        data = np.ones((1, 144, 1))
        data[:, ::3, :] = 0
        shifts, _ = day_shifts(data)
        self.assertAlmostEqual(shifts.item(), 400 / 9)

    def test_constant_states_have_no_phase_effect(self):
        shifts, _ = day_shifts(np.zeros((2, 144, 3)))
        np.testing.assert_array_equal(shifts, np.zeros((2, 3)))

    def test_borderline_or_inconsistent_intervals_are_inconclusive(self):
        self.assertEqual(decision([[4, 6], [1, 4]], 42), 'inconclusive')
        self.assertEqual(decision([[0, 4], [0, 5]], 42), 'not_supported')
        self.assertEqual(decision([[6, 7], [6, 8]], 27), 'inconclusive')

    def test_block_estimate_is_reproducible(self):
        daily = np.arange(42, dtype=float)
        daily[3] = np.nan
        np.testing.assert_array_equal(bootstrap(daily, 3), bootstrap(daily, 3))


if __name__ == '__main__':
    unittest.main()
