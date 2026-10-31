"""H6 analytic/exponential/decision checks on synthetic data only, on dislab."""
import sys
from pathlib import Path
import unittest

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import history_to_stock as h
import loss_consistent_stock as s


class LossConsistentStockTests(unittest.TestCase):
    def test_zero_rates_identity_and_crps_outside_support(self):
        b = np.array([0.,1.,1000.])
        grid,p = s.state_pmf(b,np.zeros(3),np.zeros(3),16)
        outputs = s.summaries(grid,p)
        for key in ('mean','median','lower','upper'):
            np.testing.assert_array_equal(outputs[key],b)
        np.testing.assert_array_equal(outputs['pzero'],[1,0,0])
        labels = np.array([300.,0.,1.])
        np.testing.assert_allclose(s.integer_crps(grid,p,labels),np.abs(b-labels),atol=1e-12)
        self.assertTrue((p[grid < 0] == 0).all())

    def test_pure_arrival_matches_poisson(self):
        k = h.numeric_contract([3])['cutoff']
        grid,p = s.state_pmf(np.array([0.,5.]),np.array([3.,3.]),np.zeros(2),k)
        for row,b in enumerate([0,5]):
            expected = poisson.pmf(grid[row]-b,1.5)/poisson.cdf(k,1.5)
            np.testing.assert_allclose(p[row],expected,atol=1e-11)
        self.assertAlmostEqual(s.summaries(grid,p)['mean'][1],6.5,places=9)

    def test_pure_departure_absorbs_at_zero(self):
        k = h.numeric_contract([4])['cutoff']
        grid,p = s.state_pmf(np.array([0.,1.,2.]),np.zeros(3),np.full(3,4.),k)
        outputs = s.summaries(grid,p)
        np.testing.assert_allclose(outputs['pzero'],[1,poisson.sf(0,2),poisson.sf(1,2)],atol=1e-10)
        np.testing.assert_allclose(outputs['mean'],[0,np.exp(-2),4*np.exp(-2)],atol=1e-10)

    def test_forward_pmf_matches_matrix_exponential(self):
        a,d,size = 2.3,4.1,100
        k = h.numeric_contract([a+d])['cutoff']
        initial = np.array([0.,1.,5.])
        grid,p = s.state_pmf(initial,np.full(3,a),np.full(3,d),k)
        q = np.zeros((size,size))
        for i in range(size):
            if i < size-1: q[i,i+1] = a
            if i > 0: q[i,i-1] = d
            q[i,i] = -q[i].sum()
        transition = expm(.5*q)
        for row,b in enumerate(initial.astype(int)):
            nonnegative = grid[row] >= 0
            np.testing.assert_allclose(p[row,nonnegative],transition[b,grid[row,nonnegative].astype(int)],atol=1e-10)

    def test_same_pzero_and_mean_as_frozen_backward_method(self):
        b = np.array([0.,1.,10.,50.,1000.])
        a = np.array([0.,2.,3.,8.,4.]); d = np.array([2.,4.,3.,2.,1.])
        k = h.numeric_contract(a+d)['cutoff']
        grid,p = s.state_pmf(b,a,d,k)
        outputs = s.summaries(grid,p)
        old_p,old_mean = h.transition_tables(a,d,k)
        for i,initial in enumerate(b.astype(int)):
            expected_zero = old_p[i,initial] if initial <= k else 0.
            expected_mean = old_mean[i,initial] if initial <= k else h.large_state_mean(b[i:i+1],a[i:i+1],d[i:i+1],k)[0]
            self.assertAlmostEqual(outputs['pzero'][i],expected_zero,places=10)
            self.assertAlmostEqual(outputs['mean'][i],expected_mean,places=9)
        self.assertGreater(outputs['mean'][-1],1000)

    def test_median_is_absolute_loss_action_not_rounded_mean(self):
        grid = np.array([[0.,1.,2.]])
        p = np.array([[.6,0.,.4]])
        result = s.summaries(grid,p)
        self.assertAlmostEqual(result['mean'][0],.8)
        self.assertEqual(result['median'][0],0)
        self.assertGreater(result['expected_mae_advantage'][0],0)
        tie = s.summaries(np.array([[0.,1.]]),np.array([[.5,.5]]))
        self.assertEqual(tie['median'][0],0)  # fixed lower-median convention

    def test_crps_matches_independent_energy_formula_and_label_does_not_change_prediction(self):
        grid = np.array([[0.,1.,2.],[0.,1.,2.]])
        p = np.array([[.6,.1,.3],[.6,.1,.3]])
        labels = np.array([1.,20.])
        prior = s.summaries(grid,p)
        score = s.integer_crps(grid,p,labels)
        for row in range(2):
            expected = (p[row]*np.abs(grid[row]-labels[row])).sum()-.5*(p[row,:,None]*p[row,None,:]*np.abs(grid[row,:,None]-grid[row,None,:])).sum()
            self.assertAlmostEqual(score[row],expected,places=12)
        after = s.summaries(grid,p)
        for key in prior:
            np.testing.assert_array_equal(prior[key],after[key])

    def test_invalid_inputs_stop(self):
        for b,a,d in [([-1],[1],[1]),([.5],[1],[1]),([1],[-1],[1]),([1],[1],[np.nan])]:
            with self.assertRaises(ValueError):
                s.state_pmf(np.array(b),np.array(a),np.array(d),16)

    def test_predeclared_coverage_and_effect_gates(self):
        good = [dict(valid_replicates=2000,interval=[.02,.05])]*2
        self.assertEqual(s.classify(500,100000,10,.03,good),'supported')
        self.assertEqual(s.classify(500,99999,10,.03,good),'inconclusive')
        bad = [dict(valid_replicates=2000,interval=[-.02,.009])]*2
        self.assertEqual(s.classify(500,100000,10,.003,bad),'not_supported')


if __name__ == '__main__':
    unittest.main()
