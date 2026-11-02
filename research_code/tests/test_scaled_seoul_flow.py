"""Meaningful H9 synthetic contracts; execute on dislab."""
import csv,itertools,math,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import scaled_flow_core as c
import historical_calendar as h


class ScaledFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_calendar_slot_roundtrip(self):
        for slot in range(c.SLOTS):self.assertEqual(int(c.context_slot(*c.decode_slot(slot))),slot)

    def test_seoul_nonfunctioning_hours_are_not_zero_demand(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'source.csv'
            with p.open('w',newline='',encoding='cp1252') as f:
                writer=csv.writer(f);writer.writerow(['Date','Hour','Rented Bike Count','Functioning Day'])
                writer.writerows([['01/01/2018',7,100,'Yes'],['02/01/2018',7,100,'Yes'],['03/01/2018',7,0,'No']])
            values,info=c.seoul_profiles(p)
            self.assertEqual(info['functional_rows'],2);self.assertEqual(info['excluded_nonfunctional'],1)
            self.assertEqual(info['source_mean_rentals'],100);self.assertTrue(np.allclose(values,0))
            with p.open('a',encoding='cp1252') as f:f.write('01/01/2018,7,100,Yes\n')
            with self.assertRaises(ValueError):c.seoul_profiles(p)

    def test_seoul_mask_and_common_features(self):
        constant=np.ones((2,2));calendar=np.ones((7,24,2,2));seoul=np.ones((c.SLOTS,4))
        absent,prior=c.feature_table(constant,calendar,seoul,False);present,_=c.feature_table(constant,calendar,seoul,True)
        self.assertTrue(np.array_equal(absent[:,:,:30],present[:,:,:30]));self.assertTrue((absent[:,:,30:]==0).all())
        self.assertTrue((present[:,:,30:]==1).all());self.assertTrue((prior==1).all())

    def test_halfhour_reconciles_completed_counts(self):
        start=h.stamp('2025-08-02T07:00:00+09:00');times=np.array([start,start+3600])
        events=np.array([(0,start+10,0,start+1900,1),(1,start+1810,1,start+2000,0)],dtype=h.EVENT_DTYPE)
        hourly=np.array([[[1,1],[1,1]],[[0,0],[0,0]]]);valid=np.array([True,False])
        panel=c.half_hour_panel(events,np.array(['A','B']),np.array(['A','B']),times,hourly,valid)
        self.assertEqual(panel['counts'][0,0,0],1);self.assertEqual(panel['counts'][1,1,0],1)
        self.assertTrue(np.array_equal(panel['valid'],[True,True,False,False]))
        hourly[0,0,0]=2
        with self.assertRaises(ValueError):c.half_hour_panel(events,np.array(['A','B']),np.array(['A','B']),times,hourly,valid)

    def test_larger_graph_encoder_and_initial_prior(self):
        small,large=c.RateEncoder(32,1),c.RateEncoder(128,3)
        self.assertGreater(sum(x.numel() for x in large.parameters()),20*sum(x.numel() for x in small.parameters()))
        x=torch.randn(2,3,34);operator=torch.tensor([[.5,.5,0],[0,.5,.5],[1.,0,0]]);prior=torch.ones(2,3,2)
        output=large(x,operator,prior);self.assertTrue(torch.equal(output,prior))
        output.sum().backward();self.assertGreater(float(large.out.weight.grad.abs().sum()),0)

    def test_flow_objective_is_actually_optimized(self):
        torch.manual_seed(7);model=c.CountFlow(4,16);optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        counts=torch.ones(32,2)*4;context=torch.zeros(32,4);center=torch.zeros(32,2)
        def loss():return c.flow_loss(model,counts,context,center,torch.Generator().manual_seed(55))
        initial=float(loss().detach())
        for _ in range(40):optimizer.zero_grad();value=loss();value.backward();optimizer.step()
        self.assertLess(float(loss().detach()),initial*.5)

    def test_heun_constant_velocity_and_integer_inverse_gate(self):
        class Constant(torch.nn.Module):
            def forward(self,z,t,context,center):return torch.ones_like(z)*2
        result=c.heun(Constant(),torch.zeros(3,2),torch.zeros(3,4),torch.zeros(3,2))
        self.assertTrue(torch.allclose(result,torch.ones(3,2)*2))
        value=c.integer_counts(np.log1p([[-.5,3.7]]),np.zeros((1,2)))
        self.assertTrue(np.array_equal(value,[[0,3]]))
        with self.assertRaises(ValueError):c.integer_counts([[math.log(70000)]],[[0]])

    def test_reflection_principle_matches_all_small_event_orders(self):
        for d in range(5):
            for a in range(5):
                depths=[]
                for down in itertools.combinations(range(a+d),d):
                    increments=np.ones(a+d,dtype=int);increments[list(down)]=-1
                    depths.append(-min(0,int(np.cumsum(increments).min()) if len(increments) else 0))
                for k in range(1,d+2):
                    expected=sum(x>=k for x in depths)/len(depths)
                    self.assertAlmostEqual(float(np.exp(c.log_depth_survival(d,a,k))),expected,places=12)

    def test_sampled_minimum_and_uncapped_stock(self):
        d=np.full(20000,2);a=np.ones(20000,dtype=int);depth=c.reflected_depth(d,a,np.random.default_rng(9))
        self.assertLess(abs(float((depth==2).mean())-1/3),.02)
        samples=c.stock_samples(np.ones(20000),np.full((20000,1),-1),depth[:,None])
        self.assertGreaterEqual(samples.min(),0)
        self.assertEqual(int(c.stock_samples(np.array([1000]),np.array([[-1]]),np.array([[2]]))[0,0]),999)

    def test_empirical_crps_matches_cdf_score(self):
        sample=np.array([[0,1,3]]);value=c.sample_summaries(sample,np.array([2]))
        self.assertAlmostEqual(float(value['crps'][0]),2/3);self.assertEqual(value['median'][0],1)
        with self.assertRaises(ValueError):c.sample_summaries([[-1,0]])


if __name__=='__main__':unittest.main()
