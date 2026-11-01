"""H21 frozen72resource conditions and isolated real-state integration. DISLAB ONLY."""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
import platform
import time
import numpy as np
import bulk_dispatch as engine
a=engine.base


class SyntheticRoads:
    def __init__(self,seconds,meters):self.seconds=seconds;self.meters=meters
    def table(self,locations):return self.seconds,self.meters,'synthetic-matrix'
    def query(self,*args):raise ValueError('Synthetic test has no actual road geometry.')


def run(folder,out):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True);status=a.read(folder/'status.json')
    start=datetime.fromtimestamp(a.reference_now(status),timezone.utc).isoformat();native_start=datetime.now(timezone.utc).isoformat()
    rng=np.random.default_rng(44);rows=[];failures=0;max_batch=0
    for case in range(12):
        stations=[dict(id=str(i),name='Synthetic '+str(i),current=c,samples=[c]*128,lat=36.35+i*.0001,lon=127.38,forecast={}) for i,c in enumerate([30,20,0,0])]
        state=dict(stations=stations,default_ids=[s['id'] for s in stations],default_depot_id='0',source_version='synthetic',recorded_at_utc='2099-01-01T00:00:00+00:00',target_at_utc='2099-01-01T00:10:00+00:00')
        points=rng.uniform(0,1,(5,2));meters=np.sqrt(((points[:,None]-points[None,:])**2).sum(2))*160
        seconds=meters/5+rng.uniform(0,6,(5,5));np.fill_diagonal(seconds,0)
        for mode in ('compact','bulk'):
            for handling in (10,20,40):
                body=dict(automatic=False,station_ids=state['default_ids'],handling_seconds=handling)
                if mode=='compact':body.update(capacity=6,target=2,budget=6,drop_limit=3)
                try:
                    result=a.optimize(state,body,SyntheticRoads(seconds,meters),datetime.fromisoformat(state['recorded_at_utc']).timestamp()+10)
                    cert=a.certificate(state,result);batch=max([s['bikes'] for s in result['route']['stops'] if s['action']=='deliver'] or [0])
                    # Independent unavoidable work lower bound, including both pickup and drop.
                    assert result['route']['return_minutes']+1e-7>=2*result['moved_bikes']*handling/60
                    if mode=='bulk':max_batch=max(max_batch,int(batch))
                    rows.append(dict(case=case,mode=mode,handling_seconds=handling,moved=result['moved_bikes'],largest_delivery_batch=int(batch),
                        minutes=result['route']['return_minutes'],solver_seconds=result['solver_seconds'],violations=cert['violations']))
                except Exception as exc:failures+=1;rows.append(dict(case=case,mode=mode,handling_seconds=handling,error=type(exc).__name__,violations=1))
    a.atomic(out/'engineering.json',dict(cases=rows,conditions=len(rows),feasibility_violations=failures,max_bulk_delivery_batch=int(max_batch)))
    status=a.read(folder/'status.json');path=folder/status['latest_state'];state=a.read(path);ref=datetime.fromisoformat(state['recorded_at_utc']).timestamp()+10
    replay={}
    for mode,body in [('compact',dict(capacity=6,target=2,budget=6,drop_limit=3,handling_seconds=20)),('bulk',dict())]:
        r=a.optimize(state,body,a.Roads(folder/'h21-roads'),ref);lookup={s['id']:s['name'] for s in state['stations']}
        for stop in r['route']['stops']:stop['name']=lookup[stop['station_id']]
        replay[mode]={k:r[k] for k in ('parameters','source_version','recorded_at_utc','departure_at_utc','valid_until_utc','moved_bikes','route','coverage','certificate','solver_seconds','seconds')}
    replay.update(scope='Observation-time replay with fixed marginals, NOT a live issued dispatch or new accuracy/field comparison',input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    a.atomic(out/'reference-replay.json',replay)
    current=a.read(folder/'status.json');response=engine.rpc(folder,dict(path='/api/dispatch/optimize',body=dict(source_version=current['source_version'],automatic=True)))
    if response['status']==200:
        r=response['body'];live={k:r[k] for k in ('parameters','source_version','recorded_at_utc','departure_at_utc','issued_at_utc','valid_until_utc','latest_departure_at_utc','moved_bikes','route','coverage','certificate','solver_seconds','seconds')}
    else:live=response
    a.atomic(out/'live-now.json',live)
    end=datetime.fromtimestamp(a.reference_now(a.read(folder/'status.json')),timezone.utc).isoformat()
    summary=dict(hypothesis_id='H-0021',started_at_utc=start,finished_at_utc=end,native_started_at_utc=native_start,native_finished_at_utc=datetime.now(timezone.utc).isoformat(),
        clock=dict(offset_seconds=status['clock']['offset_seconds'],uncertainty_seconds=status['clock']['uncertainty_seconds']),
        environment=dict(python=platform.python_version(),numpy=np.__version__,device='dislab CPU'),conditions=len(rows),feasibility_violations=failures,max_bulk_delivery_batch=int(max_batch),
        decision='supported' if len(rows)==72 and failures==0 and max_batch>=5 else 'not_supported',live_rpc_status=response['status'],
        live_moved=live.get('moved_bikes'),replay_compact_moved=replay['compact']['moved_bikes'],replay_bulk_moved=replay['bulk']['moved_bikes'],
        limitations=['Changed operational assumptions, not same-resource algorithm/accuracy/field gain','One vehicle and assumed parking/handling, original10min prediction cutoff','Replay is not current issuance; actual-now response separately recorded','API is input only, frozen model weights unchanged'])
    a.atomic(out/'summary.json',summary);print(summary,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--out',required=True);q=p.parse_args();run(q.folder,q.out)
