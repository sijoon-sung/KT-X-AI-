"""Causal API snapshot -> locked H13 forecast; future snapshot scored separately.

No credentials, HTTP request or training operation is in this public CLI. A
collector supplies private observations; predict cannot accept a future label.
"""
import argparse,json,time,hashlib
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime,timezone
import numpy as np
import torch
import historical_calendar as h
import forecast_transfer as weather
import rolling_inventory_model as m
from rolling_inventory_metrics import scores,LABELS

@dataclass
class Snapshot:
    scheduled:int
    stock:np.ndarray
    actual:np.ndarray
    total_rows:int=0
    matched_rows:int=0

def valid_observation(s):
    delay=s.actual-s.scheduled
    return np.isfinite(s.stock)&(s.stock>=0)&(s.stock==np.floor(s.stock))&np.isfinite(s.actual)&(delay>=0)&(delay<=120)

def proxy_and_trend(current,previous,graph):
    valid=valid_observation(current);present=np.zeros(len(valid),bool);net=np.zeros(len(valid),dtype='float64');trend=current.stock.copy()
    if previous is not None and current.scheduled-previous.scheduled==600:
        elapsed=current.actual-previous.actual
        present=valid&valid_observation(previous)&(elapsed>=480)&(elapsed<=720)
        net[present]=(current.stock[present]-previous.stock[present])*600/elapsed[present]
        trend[present]=np.maximum(0,np.floor(current.stock[present]+net[present]+.5))
    own=np.arcsinh(net).astype('float32');fraction=graph.dot(present.astype('float32'))
    near=graph.dot(own)/np.maximum(fraction,1e-8)
    return np.c_[own,present,near,fraction].astype('float32'),trend,valid

def read_snapshot(path,ids):
    value=json.loads(Path(path).read_text(encoding='utf-8'));scheduled=int(value['scheduled_at_unix_seconds']);actual=float(value['observed_at_unix_seconds'])
    if scheduled%600:raise ValueError('Snapshot is off the 10-minute grid.')
    payload=value['payload'];rows=payload.get('results') if isinstance(payload,dict) else payload
    if not isinstance(rows,list):raise ValueError('Snapshot rows missing.')
    stock=np.full(len(ids),np.nan);stamps=np.full(len(ids),-1.);lookup={str(s):i for i,s in enumerate(ids)};seen=set();matched=0
    for row in rows:
        station=str(row['id'])
        if station in seen:raise ValueError('Duplicate API station ID.')
        seen.add(station)
        if station not in lookup:continue
        v=row.get('parking_count')
        if isinstance(v,bool) or not isinstance(v,(int,float)):continue
        ix=lookup[station];stock[ix]=v;stamps[ix]=actual;matched+=1
    return Snapshot(scheduled,stock,stamps,len(rows),matched)

def issue(predictor,current,previous,weather_vector):
    proxy,trend,valid=proxy_and_trend(current,previous,predictor.graph)
    nodes,draws=predictor.issue(current.scheduled,current.stock,valid,proxy,weather_vector)
    draws['persistence']=np.repeat(current.stock[nodes,None].astype('int64'),m.M,1)
    draws['recent_trend']=np.repeat(trend[nodes,None].astype('int64'),m.M,1)
    return dict(nodes=nodes,current=current.stock[nodes],actual=current.actual[nodes],scheduled=current.scheduled,target_scheduled=current.scheduled+600,draws=draws,
        previous_proxy_present=int((proxy[nodes,1]>0).sum()),current_supported_valid=int(len(nodes)))

def score_issue(forecast,future):
    if future.scheduled!=forecast['target_scheduled']:raise ValueError('Future observation is not the forecast target.')
    nodes=forecast['nodes'];elapsed=future.actual[nodes]-forecast['actual'];valid=valid_observation(future)[nodes]&(elapsed>=480)&(elapsed<=720)
    return valid,future.stock[nodes][valid],np.abs(elapsed[valid]-600)<=30

def save_forecast(path,forecast,ids,issued_at,metadata):
    np.savez_compressed(path,nodes=forecast['nodes'],station_ids=ids[forecast['nodes']],current=forecast['current'],actual=forecast['actual'],scheduled=np.int64(forecast['scheduled']),target_scheduled=np.int64(forecast['target_scheduled']),issued_at_unix_seconds=np.float64(issued_at),
        metadata_json=np.array(json.dumps(metadata,ensure_ascii=False)),**forecast['draws'])

def predict_cli(a):
    if Path(a.out).exists():raise ValueError('Preserve previously issued forecast; choose a new output.')
    m.configure();predictor=m.Predictor(a.fit_dir,a.device);current=read_snapshot(a.snapshot,predictor.ids);previous=read_snapshot(a.previous,predictor.ids) if a.previous else None
    if time.time()>=current.scheduled+600:raise ValueError('Live forecast deadline passed; do not backdate a forecast.')
    wx=weather.weather_for(Path(a.weather_dir),'daejeon',np.array([current.scheduled]))[0]
    forecast=issue(predictor,current,previous,wx);issued=time.time()
    if not len(forecast['nodes']) or issued>=forecast['target_scheduled']:raise ValueError('No causal live forecast could be issued.')
    metadata=dict(hypothesis_id='H-0013',scope='live additional observation; not official September',model_lock_sha256=h.digest(Path(a.fit_dir)/'model-lock.json'),snapshot_sha256=h.digest(a.snapshot),previous_sha256=h.digest(a.previous) if a.previous else None,
        issued_at_utc=datetime.fromtimestamp(issued,timezone.utc).isoformat(),input_api_only=True,weights_updated=False,primary_weather_present=wx[4:8].tolist(),airport_weather_present=wx[24:27].tolist(),daily_report_present=bool(wx[18] or wx[19]),api_rows=current.total_rows,matched_rows=current.matched_rows,supported_valid=len(forecast['nodes']),recent_proxy_present=forecast['previous_proxy_present'])
    Path(a.out).parent.mkdir(exist_ok=True,parents=True);save_forecast(a.out,forecast,predictor.ids,issued,metadata)
    # This private artifact is a reference priority list, not a route or a learned RL action.
    x=forecast['draws']['recent_cfm'];mean=x.mean(1);pzero=(x==0).mean(1);med=np.sort(x,axis=1)[:,63]
    positive=np.flatnonzero(forecast['current']>0);rank=positive[np.lexsort((forecast['nodes'][positive],-pzero[positive]))][:10]
    admin=dict(hypothesis_id='H-0013',state='forecast reference; operator action logs unavailable',resource_conditions=dict(vehicles=None,capacity=None,depot=None),current_zero_stations=predictor.ids[forecast['nodes'][forecast['current']==0]].tolist(),positive_priority_reference=[dict(station_id=str(predictor.ids[forecast['nodes'][i]]),current_inventory=int(forecast['current'][i]),predicted_median=float(med[i]),predicted_mean=float(mean[i]),probability_zero=float(pzero[i])) for i in rank],policy_trained=False,
        required_action_log_fields=['decision_at','vehicle_id','origin_station','destination_station','actual_pickup','actual_dropoff','arrival_at','travel_cost','observed_outcome'])
    h.write(Path(a.out).with_suffix('.admin.json'),admin);print(json.dumps(metadata,ensure_ascii=False),flush=True)

def score_cli(a):
    with np.load(a.forecast,allow_pickle=False) as p:
        ids=p['station_ids'].copy();future=read_snapshot(a.snapshot,ids);issued=float(p['issued_at_unix_seconds'])
        if issued>=float(np.min(future.actual[valid_observation(future)])):raise ValueError('Forecast was not issued before future observations.')
        f=dict(nodes=np.arange(len(ids)),current=p['current'].copy(),actual=p['actual'].copy(),scheduled=int(p['scheduled']),target_scheduled=int(p['target_scheduled']),draws={k:p[k].copy() for k in LABELS})
        metadata=json.loads(str(p['metadata_json']))
    valid,y,strict=score_issue(f,future)
    if not valid.any():raise ValueError('No valid next observation to score.')
    report=dict(hypothesis_id='H-0013',scope='single live functional observation, not official accuracy test',forecast_metadata=metadata,forecast_sha256=h.digest(a.forecast),truth_snapshot_sha256=h.digest(a.snapshot),issued_before_truth=True,pairs=int(valid.sum()),strict_elapsed_pairs=int(strict.sum()),
        metrics={k:{name:float(v.mean()) for name,v in scores(f['draws'][k][valid],y).items()} for k in LABELS})
    h.write(Path(a.out),report);print(json.dumps(dict(pairs=report['pairs'],issued_before_truth=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    q=sp.add_parser('predict');q.add_argument('--fit-dir',required=True);q.add_argument('--snapshot',required=True);q.add_argument('--previous');q.add_argument('--weather-dir',required=True);q.add_argument('--out',required=True);q.add_argument('--device',default='cuda')
    q=sp.add_parser('score');q.add_argument('--forecast',required=True);q.add_argument('--snapshot',required=True);q.add_argument('--out',required=True)
    a=p.parse_args();predict_cli(a) if a.command=='predict' else score_cli(a)
