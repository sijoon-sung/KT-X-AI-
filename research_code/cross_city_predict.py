"""Frozen H11 completed-flow forecasts with lagged observed weather.

This CLI does not read inventory API data or predict measured stock directly.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import cross_city_data as d
import cross_city_weather as m
import historical_calendar as h

def predict(model_dir,station_id,timestamp,weather_files,kind='large_weather_pretrained',device='cuda'):
    folder=Path(model_dir);lock=json.loads((folder/'pre-test-lock.json').read_text(encoding='utf-8'))
    expected={a['id']:a['sha256'] for a in lock['artifacts']}
    names=['context.npz','weather-scaling.json']+[f'{kind}-{s}.pt' for s in m.SEEDS]
    for name in names:
        if name not in expected or h.digest(folder/name)!=expected[name]:raise ValueError('Frozen model artifact hash mismatch.')
    p=np.load(folder/'context.npz',allow_pickle=False);matches=np.flatnonzero(p['station_ids']==station_id)
    if len(matches)!=1:raise ValueError('Station is outside this historical training cohort.')
    node=int(matches[0]);stamp=h.stamp(timestamp);a=d.axes(np.array([stamp]))
    if not a['in_hours'][0] or stamp%1800:raise ValueError('Only half-hour starts 07:00 through 21:30 KST are supported.')
    slot=int(a['month'][0]*210+a['weekly'][0]);weather=d.lag_weather(weather_files,np.array([stamp]));raw=weather.copy()
    scaling=json.loads((folder/'weather-scaling.json').read_text(encoding='utf-8'))
    for j in range(4):weather[:,j]=np.where(weather[:,4+j]>0,(weather[:,j]-scaling['mean'][j])/scaling['std'][j],0)
    if kind in m.KINDS[:2]:weather[:]=0
    embed='random' if kind.endswith('_random') else 'pretrained';use=kind.endswith(('_random','_pretrained'))
    e=p[embed+'_embedding'];g=d.sparse.csr_matrix((p['graph_data'],p['graph_indices'],p['graph_indptr']),shape=(len(e),len(e)))
    z=e[node] if use else np.zeros(64);nz=g.getrow(node).dot(e)[0] if use else np.zeros(64)
    calendar=np.r_[np.eye(7)[a['weekday'][0]],np.eye(15)[a['hour'][0]-7],np.eye(2)[a['half'][0]],np.sin(2*np.pi*a['month'][0]/12),np.cos(2*np.pi*a['month'][0]/12)]
    own=m.tensor(np.r_[p['log'][slot,node],calendar,weather[0],z][None],device)
    neighbor=m.tensor(np.r_[p['neighbor'][slot,node],nz][None],device);prior=m.tensor(p['prior'][slot,node][None],device)
    rates=[]
    with torch.no_grad():
        for seed in m.SEEDS:
            ckpt=torch.load(folder/f'{kind}-{seed}.pt',map_location='cpu',weights_only=True)
            small=kind==m.KINDS[0];model=m.ForecastGCN(32 if small else 256,1 if small else 3).to(device)
            model.load_state_dict(ckpt['state_dict']);model.eval();rates.append(model(own,neighbor,prior).cpu().numpy()[0])
    rate=np.mean(rates,axis=0)
    return dict(hypothesis_id='H-0011',kind=kind,station_id=station_id,timestamp=timestamp,horizon_minutes=30,
        expected_recorded_rentals=float(rate[0]),expected_recorded_returns=float(rate[1]),expected_net_recorded_flow=float(rate[1]-rate[0]),
        weather_present={k:bool(raw[0,4+j]) for j,k in enumerate(d.VARS)},
        weather_age_minutes={k:float(raw[0,8+j]*240) if raw[0,4+j] else None for j,k in enumerate(d.VARS)},
        scope='historical completed-flow model; not externally validated measured inventory or latent demand')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir',required=True);parser.add_argument('--station',required=True)
    parser.add_argument('--timestamp',required=True);parser.add_argument('--weather-file',action='append',default=[])
    parser.add_argument('--kind',choices=m.KINDS,default=m.KINDS[-1]);parser.add_argument('--device',default='cuda')
    args=parser.parse_args();print(json.dumps(predict(args.model_dir,args.station,args.timestamp,args.weather_file,args.kind,args.device),ensure_ascii=False,allow_nan=False))
