import unittest
import numpy as np
import torch
import joint_inventory_core as m
from joint_inventory_data import regions
from scipy import sparse

class JointSemantics(unittest.TestCase):
    def test_reorder_preserves_signed_margins_and_inventory(self):
        common=np.array([[-3,0,1,2],[0,0,4,5]]);template=np.array([[4.,3,2,1],[0,2,1,3]])
        out=m.reorder(common,template,np.random.default_rng(1));np.testing.assert_array_equal(np.sort(out,1),np.sort(common,1))
        stock=np.array([2,0])[:,None];np.testing.assert_array_equal(np.sort(np.maximum(0,stock+out),1),np.sort(np.maximum(0,stock+common),1))
    def test_ties_and_bad_template(self):
        x=np.arange(12).reshape(3,4);out=m.reorder(x,np.zeros((3,4)),np.random.default_rng(2));np.testing.assert_array_equal(np.sort(out,1),x)
        with self.assertRaises(ValueError):m.reorder(x,np.full((3,4),np.nan),np.random.default_rng(2))
    def test_common_margin_different_joint_zero_probability(self):
        independent=np.array([[0,0,5,5],[0,5,0,5]]);together=np.array([[0,0,5,5],[0,0,5,5]])
        np.testing.assert_array_equal(np.sort(independent,1),np.sort(together,1));self.assertEqual(((independent==0).all(0)).mean(),.25);self.assertEqual(((together==0).all(0)).mean(),.5)
    def test_gaussian_covariance_positive_and_unit_variance(self):
        torch.manual_seed(1);g=m.Gaussian(8);c=torch.randn(2,5,m.DIM);f,s=g.factors(c);cov=f@f.transpose(1,2)+torch.diag_embed(s*s)
        torch.testing.assert_close(cov.diagonal(dim1=1,dim2=2),torch.ones(2,5));torch.linalg.cholesky(cov)
        loss=g.nll(c,torch.randn(2,5),torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool));loss.backward();self.assertTrue(torch.isfinite(loss))
    def test_zero_velocity_returns_gaussian_and_masks_unknown(self):
        flow=m.GraphFlow();z=torch.randn(3,5);c=torch.randn(1,5,m.DIM);mask=torch.tensor([[1,1,1,1,0]],dtype=torch.bool);z=z*mask;operator=torch.eye(5)
        with torch.no_grad():out=flow.sample(z,flow.condition(c),mask.expand(3,-1),operator)
        torch.testing.assert_close(z,out);self.assertEqual(float(out[:,-1].abs().sum()),0)
    def test_operator_propagates_noise_not_only_context(self):
        torch.manual_seed(11);flow=m.GraphFlow()
        with torch.no_grad():flow.output.weight.normal_(0,.2)
        z=torch.randn(1,3,requires_grad=True);c=flow.condition(torch.zeros(1,3,m.DIM));mask=torch.ones(1,3,dtype=torch.bool)
        value=flow.velocity(z,.5,c,mask,torch.ones(3,3)/3)[0,0];value.backward();self.assertGreater(float(z.grad[0,1:].abs().sum()),0)
    def test_empirical_crps_ordinary_pairwise(self):
        x=np.array([[0,1,3,4],[2,2,2,2]]);y=np.array([2,0]);expected=np.abs(x-y[:,None]).mean(1)-.5*np.abs(x[:,:,None]-x[:,None,:]).mean((1,2))
        np.testing.assert_allclose(m.crps(x,y),expected)
    def test_region_partition_is_fixed_complete_disjoint(self):
        graph=sparse.eye(9,format='csr');ids=np.array([str(i) for i in range(9)]);a=regions(graph,ids,4);b=regions(graph,ids,4)
        self.assertEqual(a,b);self.assertEqual(sorted(sum(a,[])),list(range(9)));self.assertEqual([len(g) for g in a],[4,4,1])

if __name__=='__main__':unittest.main()
