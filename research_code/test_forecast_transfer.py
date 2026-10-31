"""H12 semantic checks, synthetic records only except schema/source integrity."""
import csv,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy import sparse
import cross_city_data as old
import forecast_transfer_data as d
import forecast_transfer as m
import historical_calendar as h

class Semantics(unittest.TestCase):
    def test_daily_date_minus_two_and_dash_separate(self):
        t=np.array([h.stamp('2025-08-03T07:00'),h.stamp('2025-08-04T21:30'),h.stamp('2025-08-05T07:00')])
        day=(h.stamp('2025-08-01')+32400)//86400;r={day:'12.0',day+1:'-',day+2:'0.0',day+3:'999.0'}
        x=d.daily_features(r,t);self.assertEqual(x[0,1],1);self.assertGreater(x[0,0],0);self.assertEqual(x[1,2],1);self.assertEqual(x[1,1],0);self.assertEqual(x[2,3],1)
        r[day+3]='0';np.testing.assert_array_equal(x,d.daily_features(r,t))

    def test_latest_lag_and_staleness(self):
        t=h.stamp('2025-03-03T10:00');reports={t-3600:[3.],t:[999.]}
        x=d.latest(reports,np.array([t,t+6*3600]),1);self.assertEqual(x[0,0],3);self.assertEqual(x[0,1],1);self.assertEqual(x[1,1],0)

    def test_rain_source_and_weather_code_parser(self):
        fields=['DATE']+[v+s for v in (*old.VARS,'pres_wx_MW1','pres_wx_MW2','pres_wx_MW3') for s in ('','_Quality_Code','_Measurement_Code','_Source_Code')]
        row=dict.fromkeys(fields,'');row.update(DATE='2025-01-02T00:00:00',precipitation='2')
        # Generic amount source412 is explicitly excluded; phenomenon RA:61 remains a separate observation.
        fields.extend(k for k in ('precipitation','precipitation_Source_Code','precipitation_Quality_Code','precipitation_Measurement_Code') if k not in fields)
        row.update(precipitation_Source_Code='412',precipitation_Quality_Code='',pres_wx_MW1='RA:61',pres_wx_MW1_Quality_Code='1')
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'weather.psv'
            with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerow(row)
            # Actual archive is pipe-separated.
            with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,delimiter='|');w.writeheader();w.writerow(row)
            nominal,thermo,weather=d.extra_reports([p]);self.assertEqual(len(nominal),0);self.assertEqual(list(weather.values())[0][6],1)
            row['precipitation_Source_Code']='223';row['precipitation_Measurement_Code']='T'
            with p.open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,delimiter='|');w.writeheader();w.writerow(row)
            nominal,_,_=d.extra_reports([p]);self.assertEqual(list(nominal.values())[0],[0.,1.])

    def test_encoder_initialization_matches_objectives(self):
        torch.manual_seed(11);a=m.Autoencoder();torch.manual_seed(11);b=m.Forecaster()
        for x,y in zip(a.encoder.parameters(),b.encoder.parameters()):torch.testing.assert_close(x,y)

    def test_graph_encoder_gradient_is_live(self):
        torch.manual_seed(11);model=m.Forecaster();nnodes=3
        ctx=SimpleNamespace(profiles=torch.randn(nnodes,420,requires_grad=True),graph=torch.tensor([[0.,1.,0.],[0.,1.,0.],[0.,0.,1.]]),
            slot=torch.tensor([0]),log=torch.randn(1,nnodes,6),calendar=torch.randn(1,26),weather=torch.randn(1,54),neighbor=torch.randn(1,nnodes,6),prior=torch.ones(1,nnodes,2))
        with torch.no_grad():model.output.weight.normal_(0,.05)
        loss=model.rate(ctx,torch.tensor([0]),torch.tensor([0]),m.KINDS[-1]).sum();loss.backward()
        self.assertGreater(float(ctx.profiles.grad[1].abs().sum()),0);self.assertGreater(float(model.encoder.net[0].weight.grad.abs().sum()),0)

    def test_source_context_ignores_validation_counts(self):
        times=np.arange(h.stamp('2024-01-02T07:00'),h.stamp('2025-01-01T07:00'),86400);a=old.axes(times)
        panel=dict(times=times,counts=np.ones((len(times),2,2),dtype='uint16'),valid=np.ones(len(times),dtype=bool),ids=np.array(['1','2']),first=np.array([times[0],times[0]]),od=sparse.csr_matrix([[0,20],[30,0]]),**a)
        f=d.prepare_source_fold(panel);panel['counts'][times>=d.SOURCE_END]=999;g=d.prepare_source_fold(panel)
        np.testing.assert_array_equal(f['profiles'],g['profiles']);np.testing.assert_array_equal(f['prior'],g['prior']);np.testing.assert_array_equal(f['graph'].toarray(),g['graph'].toarray())

    def test_actual_pinned_weather_schema(self):
        manifest=json.loads((m.ROOT/'research/datasets/tashu-forecast-transfer-H0012-v1.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest['sources']),44);self.assertFalse(manifest['api_data_included'])
        raw=m.ROOT.parents[1]/'work/tashu-expanded-sources/raw'
        if raw.exists():
            x=m.weather_for(raw,'daejeon',np.array([h.stamp('2025-08-03T07:00'),h.stamp('2026-03-03T18:30')]))
            self.assertEqual(x.shape,(2,54));self.assertTrue(np.isfinite(x).all());self.assertTrue((x[:,18]+x[:,19]>0).all())

if __name__=='__main__':unittest.main()
