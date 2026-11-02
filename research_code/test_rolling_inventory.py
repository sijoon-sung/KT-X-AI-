"""H13 semantic, time-causality and distribution checks on synthetic observations."""
import unittest
from types import SimpleNamespace
import numpy as np
from scipy import sparse
import torch
import rolling_inventory_data as d
import rolling_inventory_model as m
import historical_calendar as h
import rolling_inventory_predict as p
import rolling_inventory_metrics as metrics

class Semantics(unittest.TestCase):
    def test_full_day_axes_no_clipped_weekly_slot(self):
        t=np.arange(h.stamp('2025-01-06'),h.stamp('2025-01-13'),600);a=d.axes(t)
        self.assertEqual(len(np.unique(a['weekly'])),336);self.assertEqual(a['calendar'].shape,(1008,35))
        self.assertEqual(a['phase'].shape,(1008,3));self.assertTrue((a['slot']>=0).all());self.assertLess(a['slot'].max(),4032)
    def test_recent_counts_only_previous_interval(self):
        y=np.array([[[1,4],[2,1]],[[5,0],[1,3]],[[99,99],[99,99]]],dtype='uint16');times=np.arange(3)*600
        x=d.recent_from_counts(y,times,np.ones(3,bool),np.zeros(2),sparse.eye(2,format='csr'),0,3)
        self.assertEqual(x[0,:,1].sum(),0);np.testing.assert_allclose(x[1,:,0],np.arcsinh([3,-1]));np.testing.assert_allclose(x[2,:,0],np.arcsinh([-5,2]))
        y[2]=65500;np.testing.assert_array_equal(x,d.recent_from_counts(y,times,np.ones(3,bool),np.zeros(2),sparse.eye(2,format='csr'),0,3))
    def test_source_gap_is_missing_not_zero(self):
        y=np.zeros((3,2,2),dtype='uint16');times=np.arange(3)*600;valid=np.array([False,True,True])
        x=d.recent_from_counts(y,times,valid,np.zeros(2),sparse.eye(2,format='csr'),1,3)
        self.assertEqual(x[0,:,1].sum(),0);self.assertEqual(x[1,:,1].sum(),2);self.assertEqual(x[1,:,0].sum(),0)
    def test_ten_minute_aggregation_and_boundary(self):
        start=h.stamp('2025-01-01');end=start+3*86400
        event=np.array([(start+86400+610,0,start+86400+1220,1)],dtype=[('rent','i8'),('origin','u4'),('returned','i8'),('destination','u4')])
        p,a=d.aggregate(event,np.array(['a','b']),np.array(['a','b']),{'2025-01'},start,end)
        self.assertEqual(p['counts'][145,0,0],1);self.assertEqual(p['counts'][146,1,1],1);self.assertFalse(p['valid'][0]);self.assertTrue(p['valid'][144]);self.assertTrue(all(a['count10_to30_reconciled']))
    def test_prior_never_uses_validation_labels(self):
        times=np.array([h.stamp('2025-01-02'),h.stamp('2026-02-02')]);a=d.axes(times)
        p=dict(times=times,valid=np.ones(2,bool),counts=np.ones((2,2,2),dtype='uint16'),**a);first=np.zeros(2);g=sparse.eye(2,format='csr')
        x=d.priors(p,first,g);p['counts'][1]=60000;y=d.priors(p,first,g)
        for xx,yy in zip(x,y):np.testing.assert_array_equal(xx,yy)
    def test_graph_encoder_gradient_and_feature_dimensions(self):
        torch.manual_seed(11);model=m.Forecaster();n=3
        c=SimpleNamespace(profiles=torch.randn(n,420,requires_grad=True),graph=torch.tensor([[0.,1.,0.],[0.,1.,0.],[0.,0.,1.]]),slot=torch.tensor([0]),log=torch.randn(1,n,6),neighbor=torch.randn(1,n,6),prior=torch.ones(1,n,2),calendar=torch.randn(1,35),phase=torch.randn(1,3),weather=torch.randn(1,54),recent=torch.randn(1,n,4))
        with torch.no_grad():model.output.weight.normal_(0,.05)
        model.rate(c,torch.tensor([0]),torch.tensor([0]),'recent').sum().backward();self.assertGreater(float(c.profiles.grad[1].abs().sum()),0)
    def test_integer_projection_is_not_reflected_event_order(self):
        draws=np.array([[[5,0],[0,4],[4,4]]]);np.testing.assert_array_equal(m.stock_projection(np.array([2]),draws),[[0,6,2]])
        with self.assertRaises(ValueError):m.integer_counts(np.array([[20.,0.]]),np.zeros((1,2)))
    def test_flow_head_dimension_and_heun(self):
        torch.manual_seed(11);flow=m.CountFlow(context_width=256,width=128);z=torch.randn(8,2);c=torch.randn(8,256);center=torch.ones(8,2)
        self.assertEqual(sum(p.numel() for p in flow.parameters()),67202);torch.testing.assert_close(m.heun(flow,z,c,center),z)

    def test_fair_crps_pairwise_and_pointmass(self):
        x=np.array([[0,1,3,4],[2,2,2,2]],dtype=float);y=np.array([2,0]);v=metrics.scores(x,y)
        pair=np.abs(x[:,:,None]-x[:,None,:]).sum((1,2))/(2*4*3)
        np.testing.assert_allclose(v['fair_crps'],np.abs(x-y[:,None]).mean(1)-pair);self.assertEqual(v['fair_crps'][1],2)
        self.assertEqual(v['fair_brier'][1],1)

    def test_observation_timing_missing_and_noisy_proxy(self):
        old=p.Snapshot(0,np.array([0.,3.,np.nan,1.]),np.array([1.,1.,-1.,1.]))
        new=p.Snapshot(600,np.array([2.,0.,0.,5.]),np.array([601.,601.,601.,800.]))
        proxy,trend,valid=p.proxy_and_trend(new,old,sparse.eye(4,format='csr'))
        np.testing.assert_array_equal(valid,[True,True,True,False]);np.testing.assert_array_equal(proxy[:,1],[1,1,0,0])
        np.testing.assert_allclose(proxy[:2,0],np.arcsinh([2,-3]));np.testing.assert_array_equal(trend[:3],[4,0,0])
        # A jump is an API proxy, never inferred to be a known operator action.
        self.assertEqual(proxy[2,0],0);self.assertEqual(proxy[2,3],0)

    def test_future_filtering_is_separate_from_forecast(self):
        current=p.Snapshot(600,np.array([0.,3.]),np.array([601.,601.]));graph=sparse.eye(2,format='csr')
        a=p.proxy_and_trend(current,None,graph)
        f=dict(nodes=np.array([0,1]),current=current.stock,actual=current.actual,scheduled=600,target_scheduled=1200)
        future=p.Snapshot(1200,np.array([9.,np.nan]),np.array([1202.,-1.]))
        valid,y,strict=p.score_issue(f,future);np.testing.assert_array_equal(valid,[True,False]);self.assertEqual(y[0],9)
        future.stock[:]=999;future.actual[:]=1300
        b=p.proxy_and_trend(current,None,graph)
        for aa,bb in zip(a,b):np.testing.assert_array_equal(aa,bb)
        with self.assertRaises(ValueError):p.score_issue(f,p.Snapshot(1800,np.ones(2),np.ones(2)*1800))

    def test_calendar_bootstrap_counts_empty_days(self):
        days=np.tile([2.,2.,2.,2.,1.,1000.],(14,1));days[4]=0
        r=metrics.uncertainty(days,600);self.assertEqual(r['decision'],'supported');self.assertEqual(r['intervals'][0]['valid_replicates'],2000)
        self.assertAlmostEqual(r['worst_control_fair_crps_relative_gain'],.5)

if __name__=='__main__':unittest.main()
