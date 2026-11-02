"""H-0004 synthetic contract checks; execute on dislab before the real run."""
import sys
import tempfile
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import historical_calendar as h


class HistoricalCalendarTests(unittest.TestCase):
    def test_normalized_time_and_source_month_and_duration(self):
        positions = {name: i for i, name in enumerate(h.FIELDS)}
        row = ['bike', '2025-03-05 7:16', '001', '2025-03-05 08:16:00', '002']
        value, reason = h.parse_trip(row, positions, '2025-03')
        self.assertIsNone(reason)
        self.assertEqual(value, ('bike', h.stamp('2025-03-05T07:16:00+09:00'), '001',
                                  h.stamp('2025-03-05T08:16:00+09:00'), '002'))
        self.assertEqual(h.parse_trip(row, positions, '2025-12')[1], 'filename_month_mismatch')
        bad = row.copy(); bad[1] = '2025-02-30 08:00'
        self.assertEqual(h.parse_trip(bad, positions, '2025-02')[1], 'invalid_datetime')
        bad = row.copy(); bad[3] = '2025-03-06 7:16'
        self.assertEqual(h.parse_trip(bad, positions, '2025-03')[1], 'invalid_duration')
        bad = row.copy(); bad[3] = '2025-03-05 7:15'
        self.assertEqual(h.parse_trip(bad, positions, '2025-03')[1], 'invalid_duration')
        bad = row.copy(); bad[0] = ''
        self.assertEqual(h.parse_trip(bad, positions, '2025-03')[1], 'missing_key')

    def test_exact_dedup_keeps_distinct_and_short_trips(self):
        a = (1, h.stamp('2025-08-02T08:00'), 0, h.stamp('2025-08-02T08:01'), 1)
        b = (1, h.stamp('2025-08-02T08:02'), 1, h.stamp('2025-08-02T08:03'), 0)
        self.assertEqual(len(np.unique(np.array([a, a, b], dtype=h.EVENT_DTYPE))), 2)

    def test_missing_scope_is_nan_and_large_record_count_is_kept(self):
        def event(time, origin=0, dest=1):
            return (1, h.stamp(time), origin, h.stamp(time)+60, dest)
        events = np.array([event('2025-08-02T08:00')]*120 +
                          [event('2025-08-01T08:00'), event('2025-08-02T06:00')], dtype=h.EVENT_DTYPE)
        panel, _ = h.build_panel(events, np.array(['A','B','C']), {'2025-08'},
                               h.stamp('2025-08-01'), h.stamp('2025-08-03'))
        index = int((h.stamp('2025-08-02T08:00')-h.stamp('2025-08-01'))//3600)
        self.assertEqual(panel['y'][index, 0, 0], 120)
        self.assertEqual(panel['y'][index, 2, 0], 0)  # No accepted record, explicitly conditional source scope.
        self.assertTrue(np.isnan(panel['y'][0]).all())
        self.assertTrue(np.isnan(panel['y'][index+1]).all())
        self.assertTrue(np.isnan(panel['y'][index-2]).all())
        missing, _ = h.build_panel(events, np.array(['A','B','C']), set(),
                                 h.stamp('2025-08-01'), h.stamp('2025-08-03'))
        self.assertTrue(np.isnan(missing['y']).all())

    def test_calendar_shrinkage_and_unseen_groups(self):
        y = np.array([[[0.,2.]], [[4.,6.]]])
        base, rates = h.fit_rates(y, np.array([0,0]), np.array([8,8]), 4)
        np.testing.assert_array_equal(base, (y.sum(0)+.5)/3)
        np.testing.assert_allclose(rates[0,8], (y.sum(0)+4*base)/6)
        np.testing.assert_array_equal(rates[6,21], base)

    def test_future_panel_changes_do_not_change_training_cohort_or_rates(self):
        times = np.arange(h.START, h.END, 3600, dtype=np.int64)
        axes = h.calendar_axes(times)
        valid = (axes['hour']==8) & axes['interior']
        y = np.ones((len(times),2,2)); y[~valid] = np.nan
        panel = dict(times=times, valid=valid, y=y, **axes)
        train = h.split_masks(panel)['train']
        cohort1 = h.training_cohort(panel, train)
        learned1 = h.fit_rates(y[train], axes['weekday'][train], axes['hour'][train], 4)
        y[times>=h.TRAIN_END] = 100000.
        cohort2 = h.training_cohort(panel, train)
        learned2 = h.fit_rates(y[train], axes['weekday'][train], axes['hour'][train], 4)
        np.testing.assert_array_equal(cohort1, cohort2)
        for a,b in zip(learned1,learned2): np.testing.assert_array_equal(a,b)

    def test_selection_uses_validation_and_deterministic_tie_order(self):
        y = np.ones((2,1,2)); weekdays=np.array([0,0]); hours=np.array([8,8])
        chosen, _, candidates = h.choose_alpha(y, weekdays, hours, y, np.array([6,6]), hours)
        self.assertEqual(chosen, 1)  # Unseen weekday falls back to exactly the same base for every alpha.
        self.assertTrue(all(x['validation_deviance']==candidates[0]['validation_deviance'] for x in candidates))

    def test_deviance_zero_labels_and_domain(self):
        np.testing.assert_allclose(h.deviance(np.array([0.,2.]), np.array([3.,2.])), [6.,0.])
        with self.assertRaises(ValueError): h.deviance(np.array([0.]),np.array([0.]))
        with self.assertRaises(ValueError): h.deviance(np.array([-1.]),np.array([1.]))

    def test_block_resampling_retains_calendar_gaps_and_seed(self):
        daily=np.array([[4.,2.,10.],[0.,0.,0.],[8.,4.,20.],[2.,1.,10.],[0.,0.,0.]])
        a=h.bootstrap(daily,3,200)
        self.assertEqual(a,h.bootstrap(daily,3,200))
        self.assertEqual(a['interval'],[50.,50.])

    def test_saved_model_roundtrip_and_supported_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.npz'
            rates=np.ones((7,24,1,2)); rates[0,8,0]=[2.,3.]
            h.save_model(path,'calendar',np.array(['001']),rates,4)
            result=h.load_prediction(path,'001','2026-03-02T08:00:00+09:00')
            self.assertEqual((result['expected_recorded_rentals'],result['expected_recorded_returns']),(2.,3.))
            with self.assertRaises(ValueError): h.load_prediction(path,'002','2026-03-02T08:00')
            with self.assertRaises(ValueError): h.load_prediction(path,'001','2026-03-02T06:00')
            with self.assertRaises(ValueError): h.load_prediction(path,'001','2026-03-02T08:10')

    def test_decision_coverage_and_thresholds(self):
        boot=[dict(valid_replicates=2000,interval=[2.,4.])]*2
        self.assertEqual(h.decision(500,14,21,100000,boot),'supported')
        self.assertEqual(h.decision(499,14,21,100000,boot),'inconclusive')
        boot=[dict(valid_replicates=2000,interval=[-3.,1.])]*2
        self.assertEqual(h.decision(500,14,21,100000,boot),'not_supported')


if __name__ == '__main__':
    unittest.main()
