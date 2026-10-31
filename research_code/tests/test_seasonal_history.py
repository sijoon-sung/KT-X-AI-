"""H10 semantic checks: masking, sufficient-statistic equivalence and real optimization."""
import sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import historical_calendar as hist
import seasonal_history_core as c
import scaled_flow_core as slots


class SeasonalHistoryTests(unittest.TestCase):
    def test_first_record_uses_only_training_events(self):
        events=np.array([(0,c.START+100,0,c.START+200,1),(1,c.TRAIN_END+10,0,c.TRAIN_END+30,1)],dtype=hist.EVENT_DTYPE)
        first=c.first_training_records(events,np.array(['A','B']),np.array(['B','A']))
        np.testing.assert_array_equal(first,[c.START+200,c.START+100])
        events['rent'][1]+=1000000;events['returned'][1]+=1000000
        np.testing.assert_array_equal(first,c.first_training_records(events,np.array(['A','B']),np.array(['B','A'])))

    def test_pre_record_zero_has_no_training_exposure(self):
        data=dict(times=np.array([c.START,c.START+1800,c.TRAIN_END,c.VALID_END]),
            counts=np.array([[[0,0]],[[2,3]],[[900,900]],[[700,700]]]),valid=np.ones(4,dtype=bool),slots=np.zeros(4,dtype=int),first=np.array([c.START+1]))
        total,n=c.sufficient_statistics(data)
        self.assertEqual(n[0,0],1);np.testing.assert_array_equal(total[0,0],[2,3])
        self.assertEqual(total.sum(),5)

    def test_missing_season_returns_to_weekly_prior(self):
        n=np.zeros((slots.SLOTS,2));total=np.zeros((slots.SLOTS,2,2))
        source=int(slots.context_slot(1,2,8,0));target=int(slots.context_slot(9,2,8,0))
        n[source]=10;total[source]=[[20,30],[40,50]]
        base,weekly,seasonal=c.profiles(total,n,8)
        np.testing.assert_allclose(seasonal[target],weekly[target])
        self.assertGreater(seasonal[source,0,0],weekly[source,0,0])
        self.assertTrue(np.isfinite(base).all())

    def test_count_outside_exposure_is_rejected(self):
        n=np.zeros((slots.SLOTS,1));total=np.zeros((slots.SLOTS,1,2));total[0,0,0]=1
        with self.assertRaises(ValueError):c.profiles(total,n,8)

    def test_raw_and_sufficient_statistic_loss_and_gradient_match(self):
        y=torch.tensor([[[0.,1.],[3.,0.]],[[2.,1.],[1.,4.]],[[0.,2.],[9.,9.]]],dtype=torch.float64)
        active=torch.tensor([[1.,1.],[1.,1.],[1.,0.]],dtype=torch.float64)
        rate=torch.tensor([[1.2,.7],[2.4,1.3]],dtype=torch.float64,requires_grad=True)
        raw=((rate-y*torch.log(rate))*active[...,None]).sum()/(2*active.sum())
        raw_grad=torch.autograd.grad(raw,rate)[0]
        aggregate=c.poisson_objective(rate,(y*active[...,None]).sum(0),active.sum(0))
        aggregate_grad=torch.autograd.grad(aggregate,rate)[0]
        torch.testing.assert_close(raw,aggregate);torch.testing.assert_close(raw_grad,aggregate_grad)

    def test_graph_changes_output_but_keeps_positive_rates(self):
        torch.manual_seed(11);model=c.SeasonalGCN()
        with torch.no_grad():model.out.weight.fill_(.03)
        x=torch.randn(1,2,32);prior=torch.ones(1,2,2)
        a=model(x,x,prior);b=model(x,x.flip(1),prior)
        self.assertGreater(float((a-b).abs().max()),1e-6)
        self.assertTrue(bool((b>0).all()))

    def test_actual_optimization_reduces_frozen_objective(self):
        torch.manual_seed(29);model=c.SeasonalGCN();optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        x=torch.randn(2,3,32);prior=torch.ones(2,3,2);n=torch.ones(2,3)*10;total=torch.ones(2,3,2)*30
        before=float(c.poisson_objective(model(x,x,prior),total,n).detach())
        for _ in range(12):
            optimizer.zero_grad();loss=c.poisson_objective(model(x,x,prior),total,n);loss.backward();optimizer.step()
        after=float(c.poisson_objective(model(x,x,prior),total,n).detach())
        self.assertLess(after,before-.2)

    def test_bootstrap_paired_relative_gain(self):
        daily=np.tile([10.,20.,8.,5.],(14,1))
        for block in (3,7):
            result=c.bootstrap(daily,block)
            self.assertEqual(result['valid_replicates'],2000)
            np.testing.assert_allclose(result['interval'],[.2,.2])


if __name__=='__main__':unittest.main()
