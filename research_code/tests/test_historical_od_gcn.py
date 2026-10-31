"""Synthetic graph, information-boundary and matched-budget checks on dislab."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import historical_od_gcn as g
import historical_calendar as h


class HistoricalODTests(unittest.TestCase):
    def test_only_training_edges_and_both_valid_endpoints(self):
        t=h.START+2*86400+8*3600
        events=np.array([(0,t,0,t+3600,1),(1,t+7200,1,t+10800,0),
            (2,h.TRAIN_END,0,h.TRAIN_END+10,1),(3,t,0,t+10,0),
            (4,t,0,t+10,2)],dtype=h.EVENT_DTYPE)
        times=np.arange(h.START,h.TRAIN_END,3600)
        valid=np.ones(len(times),dtype=bool);valid[int((t+7200-h.START)//3600)]=False
        first,scope=g.graph_counts(events,np.array(['A','B','C']),np.array(['A','B']),times,valid)
        np.testing.assert_array_equal(first,[[0,1],[0,0]])
        self.assertEqual(scope['eligible_nonself_trips'],1)
        events[2]['origin']=1;events[2]['destination']=0
        second,_=g.graph_counts(events,np.array(['A','B','C']),np.array(['A','B']),times,valid)
        np.testing.assert_array_equal(first,second)

    def test_top_union_normalization_isolated_and_stable_ties(self):
        counts=np.array([[0,2,2,0],[0,0,1,0],[0,0,0,0],[0,0,0,0]])
        operator,info=g.graph_operator(counts,top=1)
        np.testing.assert_allclose(operator.sum(1),1)
        self.assertEqual(info['isolated_nodes'],1)
        self.assertEqual(operator[3,3],1)
        self.assertEqual(operator[0,0],0)
        self.assertGreater(operator[0,1],0)

    def test_identity_and_neighbor_aggregation_no_sample_mixing(self):
        x=np.arange(2*3*2).reshape(2,3,2).astype(np.float32)
        np.testing.assert_array_equal(g.aggregate_features(x,np.eye(3)),x)
        operator=np.array([[0,1,0],[1,0,0],[0,0,1.]])
        output=g.aggregate_features(x,operator)
        np.testing.assert_array_equal(output[:,0],x[:,1])
        np.testing.assert_array_equal(output[1],g.aggregate_features(x[1:2],operator)[0])

    def test_fixed_feature_shape_no_recent_transactions(self):
        base=np.array([[1.,2.],[3.,4.]])
        calendar=np.broadcast_to(base,(7,24,2,2)).copy()
        x,prior=g.feature_grid(base,calendar)
        self.assertEqual(x.shape,(105,2,26))
        self.assertEqual(prior.shape,(105,2,2))
        np.testing.assert_array_equal(x[:,:,4:11].sum(-1),1)
        np.testing.assert_array_equal(x[:,:,11:].sum(-1),1)

    def test_same_parameters_epoch_zero_and_gradient(self):
        torch.manual_seed(11)
        model=g.RateGCN()
        x=torch.randn(2,3,26);a=torch.randn_like(x);prior=torch.ones(2,3,2)
        np.testing.assert_array_equal(model(x,a,prior).detach().numpy(),prior.numpy())
        params=sum(p.numel() for p in model.parameters())
        self.assertEqual(params,sum(p.numel() for p in g.RateGCN().parameters()))
        loss=model(x,a,prior).sum();loss.backward()
        self.assertGreater(float(model.out.bias.grad.abs().sum()),0)

    def test_graph_permutation_changes_relation_not_features(self):
        operator=np.array([[0,1,0],[.2,0,.8],[1,0,0.]])
        permutation=np.array([2,0,1])
        altered=operator[permutation][:,permutation]
        np.testing.assert_allclose(altered.sum(1),1)
        x=np.arange(6).reshape(1,3,2).astype(float)
        self.assertFalse(np.array_equal(g.aggregate_features(x,operator),g.aggregate_features(x,altered)))

    def test_worst_control_bootstrap_not_best_winner(self):
        daily=np.tile(np.array([2.,3.,4.,1.,100.]),(14,1))
        result=g.bootstraps(daily,3)
        self.assertEqual(result['valid_replicates'],2000)
        np.testing.assert_allclose(result['interval'],[.01,.01])
        self.assertEqual(result,g.bootstraps(daily,3))

    def test_numerical_gate_from_frozen_rates(self):
        self.assertLessEqual(g.numerical_contract([2.,10.])['tail_probability_bound'],1e-10)
        for values in ([-1.,5.],[np.nan],[5000.]):
            with self.assertRaises(ValueError):g.numerical_contract(values)


if __name__=='__main__':unittest.main()
