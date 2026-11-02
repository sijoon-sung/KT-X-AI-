"""H11 semantic checks on synthetic data; no empirical fitting/evaluation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
import tempfile
import unittest
import csv
import numpy as np
import torch
import cross_city_data as d
import cross_city_weather as m
import historical_calendar as h

class Semantics(unittest.TestCase):
    def test_historical_manifest_resolves_hash_checked_member_catalog(self):
        source=m.historical_source_catalog()
        self.assertEqual(len(source['members']),20)
        self.assertTrue(all('header' in x and 'sha256' in x for x in source['members']))
    def test_calendar_boundaries_and_slots(self):
        a=d.axes(np.array([h.stamp(x) for x in ['2024-02-01T07:00','2024-02-28T21:30','2024-02-29T21:30','2025-03-03T07:30']]))
        self.assertEqual(a['interior'].tolist(),[False,True,False,True]);self.assertEqual(int(a['weekly'][-1]),1)
        self.assertEqual(int(a['month'][0]),1)

    def test_profile_masks_and_scale_invariance(self):
        ws=np.ones((2,210,2))*4;wn=np.ones((2,210))*2;ms=np.ones((2,12,2))*20;mn=np.ones((2,12))*10
        ms[:,5]=0;mn[:,5]=0;x=d.profile(ws,wn,ms,mn)
        self.assertEqual(x.shape,(2,456));self.assertTrue((x[:,444+5]==0).all());self.assertTrue((x[:,420+10:420+12]==0).all())
        np.testing.assert_allclose(x,d.profile(ws*7,wn,ms*7,mn))

    def test_weather_quality_and_trace(self):
        self.assertIsNone(d.accepted_weather('20','s','',0));self.assertIsNone(d.accepted_weather('','1','',3))
        self.assertIsNone(d.accepted_weather('2','2','',0));self.assertEqual(d.accepted_weather('0.1','1','T',3),(0.,1.))
        self.assertIsNone(d.accepted_weather('2','1','A',3));self.assertIsNone(d.accepted_weather('200','1','',2))

    def test_weather_asof_lag_staleness_and_future_independence(self):
        fields=['DATE']+[k+s for k in d.VARS for s in ('','_Quality_Code','_Measurement_Code')]
        def row(time,value):
            r=dict.fromkeys(fields,'');r['DATE']=time;r['temperature']=str(value);r['temperature_Quality_Code']='1';return r
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weather.psv'
            def write(rows):
                with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields,delimiter='|');w.writeheader();w.writerows(rows)
            times=np.array([h.stamp('2025-01-02T10:00'),h.stamp('2025-01-02T13:30')])
            rows=[row('2025-01-02T00:00:00',-4),row('2025-01-02T01:00:00',1000)]
            write(rows);a=d.lag_weather([p],times)
            write(rows+[row('2025-01-03T00:00:00',20)]);b=d.lag_weather([p],times)
            np.testing.assert_array_equal(a,b);self.assertEqual(a[0,0],-4);self.assertEqual(a[0,4],1);self.assertEqual(a[1,4],0)
            self.assertEqual(a[0,7],0) # Empty rain is missing, not a dry observation.

    def test_weather_scaling_uses_train_only(self):
        x=np.zeros((3,13),dtype='float32');x[:,4]=1;x[:,0]=[1,3,10000]
        a,stats=d.normalize_weather(x,np.array([1,1,0],dtype=bool));self.assertEqual(stats['mean'][0],2)
        self.assertEqual(stats['std'][0],1);self.assertTrue((a[:,1:4]==0).all())

    def test_fold_graph_and_profiles_do_not_use_future_events(self):
        times=np.arange(h.stamp('2024-08-02T07:00'),h.stamp('2025-06-01T07:00'),86400)
        a=d.axes(times);panel=dict(times=times,counts=np.ones((len(times),2,2),dtype='uint16'),valid=np.ones(len(times),dtype=bool),ids=np.array(['a','b']),**a)
        events=np.array([(0,int(t),0,int(t+1),1) for t in times],dtype=h.EVENT_DTYPE)
        f=d.fold_data(panel,events,'2025-03-01','2025-04-01','2025-06-01')
        extra=np.array([(0,h.stamp('2025-05-01'),1,h.stamp('2025-05-01')+10,0)]*100,dtype=h.EVENT_DTYPE)
        g=d.fold_data(panel,np.r_[events,extra],'2025-03-01','2025-04-01','2025-06-01')
        np.testing.assert_array_equal(f['profiles'],g['profiles']);np.testing.assert_array_equal(f['graph'].toarray(),g['graph'].toarray())

    def test_model_prior_optimization_and_serialization(self):
        torch.manual_seed(11);model=m.ForecastGCN(32,1);x=torch.randn(32,109);nb=torch.randn(32,70);prior=torch.ones(32,2)
        np.testing.assert_allclose(model(x,nb,prior).detach().numpy(),prior.numpy())
        opt=torch.optim.AdamW(model.parameters(),lr=.01);y=torch.ones(32,2)*2
        before=float(((model(x,nb,prior)-y)**2).mean().detach())
        for _ in range(5):loss=((model(x,nb,prior)-y)**2).mean();opt.zero_grad();loss.backward();opt.step()
        self.assertLess(float(((model(x,nb,prior)-y)**2).mean().detach()),before)
        clone=m.ForecastGCN(32,1);clone.load_state_dict(model.state_dict());torch.testing.assert_close(model(x,nb,prior),clone(x,nb,prior))

if __name__=='__main__':unittest.main()
