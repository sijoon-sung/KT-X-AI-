"""H12 frozen completed-flow forecasts; never loads inventory observations."""
import argparse,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from scipy import sparse
import forecast_transfer as m
import historical_calendar as hist

class Predictor:
    def __init__(self,folder,kind='weather_forecast_full',device='cuda'):
        self.folder=Path(folder);self.kind=kind;self.device=device
        if kind not in m.KINDS:raise ValueError('Unknown frozen candidate.')
        lock=json.loads((self.folder/'pre-test-lock.json').read_text(encoding='utf-8'));hashes={x['id']:x['sha256'] for x in lock['artifacts']}
        for name in ['context.npz','context-metadata.json']+[f'{kind}-{seed}.pt' for seed in m.SEEDS]:
            if name not in hashes or hist.digest(self.folder/name)!=hashes[name]:raise ValueError('Frozen model artifact hash mismatch.')
        self.meta=json.loads((self.folder/'context-metadata.json').read_text(encoding='utf-8'))
        if self.meta['weather_dim']!=54:raise ValueError('Forecast weather schema differs.')
        with np.load(self.folder/'context.npz',allow_pickle=False) as p:
            self.ids=p['station_ids'].copy();self.lookup={s:i for i,s in enumerate(self.ids)}
            graph=sparse.csr_matrix((p['graph_data'],p['graph_indices'],p['graph_indptr']),shape=(len(self.ids),len(self.ids)))
            self.ctx=SimpleNamespace(profiles=m.tensor(p['profiles'],device),graph=m.tensor(graph.toarray(),device),
                log=m.tensor(p['log'],device),neighbor=m.tensor(p['neighbor'],device),prior=m.tensor(p['prior'],device))
        self.models=[];self.encoded=[]
        for seed in m.SEEDS:
            saved=torch.load(self.folder/f'{kind}-{seed}.pt',map_location='cpu',weights_only=True)
            if saved['kind']!=kind or saved['seed']!=seed:raise ValueError('Checkpoint identity differs.')
            model=m.Forecaster().to(device);model.load_state_dict(saved['state_dict']);model.eval();self.models.append(model)
            with torch.no_grad():self.encoded.append(model.encode(self.ctx))
    def predict(self,station_ids,timestamps,weather_dir=None):
        if len(station_ids)!=len(timestamps) or not station_ids:raise ValueError('Equal nonempty station and timestamp lists required.')
        if any(s not in self.lookup for s in station_ids):raise ValueError('Station outside this historical training cohort.')
        times=np.array([hist.stamp(t) for t in timestamps],dtype='int64');a=m.old.axes(times)
        if np.any(times%1800) or not a['in_hours'].all() or np.any(times<self.meta['train_end']):raise ValueError('Only future half-hour starts 07:00 through 21:30 KST supported.')
        if weather_dir is None:
            weather=np.zeros((len(times),54),dtype='float32');weather[:,[8,9,10,11,15,27,28,29,41,53]]=1
        else:weather=m.weather_for(Path(weather_dir),self.meta['city'],times)
        self.ctx.weather=m.tensor(weather,self.device);self.ctx.slot=torch.as_tensor(a['month']*210+a['weekly'],device=self.device)
        cal=np.c_[np.eye(7)[a['weekday']],np.eye(15)[a['hour']-7],np.eye(2)[a['half']],np.sin(2*np.pi*a['month']/12),np.cos(2*np.pi*a['month']/12)]
        self.ctx.calendar=m.tensor(cal,self.device);rows=torch.arange(len(times),device=self.device);nodes=torch.as_tensor([self.lookup[s] for s in station_ids],device=self.device)
        with torch.no_grad():rates=torch.stack([model.rate(self.ctx,rows,nodes,self.kind,*z) for model,z in zip(self.models,self.encoded)]).mean(0).cpu().numpy()
        return [dict(hypothesis_id='H-0012',kind=self.kind,station_id=s,timestamp=t,horizon_minutes=30,
            expected_recorded_rentals=float(rate[0]),expected_recorded_returns=float(rate[1]),expected_net_recorded_flow=float(rate[1]-rate[0]),
            primary_weather_present=weather[i,4:8].astype(bool).tolist(),airport_weather_present=weather[i,24:27].astype(bool).tolist(),
            daily_lag_days=2,daily_report_status='numeric' if weather[i,18] else ('dash' if weather[i,19] else 'missing'),
            scope='historical completed-flow forecast; not externally validated measured inventory') for i,(s,t,rate) in enumerate(zip(station_ids,timestamps,rates))]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--model-dir',required=True);ap.add_argument('--station',required=True);ap.add_argument('--timestamp',required=True)
    ap.add_argument('--weather-dir');ap.add_argument('--kind',choices=m.KINDS,default=m.KINDS[-1]);ap.add_argument('--device',default='cuda');a=ap.parse_args()
    print(json.dumps(Predictor(a.model_dir,a.kind,a.device).predict([a.station],[a.timestamp],a.weather_dir)[0],ensure_ascii=False,allow_nan=False))
