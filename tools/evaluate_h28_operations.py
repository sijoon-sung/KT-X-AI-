"""Fixed H28 engineering grid. Execute on DISLAB, never requests inventory API."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_code'))
import numpy as np
import batch_operations as model


class SyntheticRoads:
    def __init__(self,layout,seed):
        self.layout=layout;self.seed=seed
    def table(self,locations):
        n=len(locations);rng=np.random.default_rng(self.seed)
        coords={s['id']:i for i,s in enumerate(locations)}
        # Fixed small positive directed legs. Layout changes are declared, not fitted.
        scale={'compact':18,'spread':80,'long_return':25,'split_supply':22}[self.layout]
        seconds=np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                if i!=j:seconds[i,j]=scale*(1+abs(i-j)/3)+int(rng.integers(0,9))
        if self.layout=='long_return':seconds[1:,0]+=8*60
        return seconds,seconds*5,'synthetic-'+self.layout+'-'+str(self.seed)
    def query(self,*args):
        return {'routes':[{'geometry':{'type':'LineString','coordinates':[]}}]},'synthetic'


class FixedRoads(SyntheticRoads):
    def __init__(self,between=30,back=480):self.between=between;self.back=back
    def table(self,locations):
        # depot, source, receiver; all remaining directed legs conservatively fixed.
        seconds=np.array([[0,30,30],[30,0,self.between],[self.back,30,0]],float)
        return seconds,seconds*5,'known-direct-matrix'


def state(layout,seed):
    rng=np.random.default_rng(seed)
    stocks=[6,18,14,0,1,4] if layout!='split_supply' else [6,8,8,0,1,4]
    rows=[]
    for i,stock in enumerate(stocks):
        samples=np.maximum(0,stock+rng.integers(-1,2,size=128))
        if layout=='split_supply' and i in (1,2):samples=np.full(128,stock)
        rows.append(dict(id='s'+str(i),name='synthetic '+str(i),lat=36.3+i*.0002,lon=127.4+i*.0002,
                         current=stock,samples=samples.tolist(),forecast=dict(median=float(np.median(samples)),p0=float((samples<=0).mean()))))
    return dict(stations=rows,default_ids=[s['id'] for s in rows[1:]],default_depot_id='s0',
                recorded_at_utc='2026-11-11T00:00:00+00:00',target_at_utc='2026-11-11T00:10:30+00:00',
                source_version='synthetic-'+layout+'-'+str(seed),model_label='fixed synthetic samples')


def known_state(supply=14):
    value=state('compact',17);value['stations']=value['stations'][:3]
    for row,stock in zip(value['stations'],[6,supply,0]):row.update(current=stock,samples=[stock]*128)
    value['default_ids']=['s1','s2'];return value


def emit(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'research/datasets/H28-stop-batch-operations-v1.json').read_text(encoding='utf-8'))
    rows=[];failures=[];guards=[]
    common=dict(automatic=False,target=6,budget=12,handling_seconds=20,max_visits=6)
    for layout,setup,cap,deadline,minimum,seed in itertools.product(spec['synthetic']['layouts'],[60,120],[6,20],[10,30],[1,3,5],[17,29]):
        config=dict(common,stop_setup_seconds=setup,capacity=cap,deadline_minutes=deadline,minimum_transfer=minimum)
        row=dict(layout=layout,setup_seconds=setup,capacity=cap,job_deadline_minutes=deadline,minimum_transfer=minimum,seed=seed)
        try:
            result=model.optimize(state(layout,seed),config,SyntheticRoads(layout,seed),datetime(2026,11,11,tzinfo=timezone.utc).timestamp())
            row.update(status=result['status'],moved_bikes=result['moved_bikes'],distance_km=result['route']['distance_meters']/1000,
                       work_minutes=result['route']['return_minutes'],violations=result['certificate']['violations'],**result['operations'])
        except Exception as exc:
            row.update(status='error',violations=1,error=repr(exc));failures.append(row)
        rows.append(row)
        if time.perf_counter()-started>560:raise RuntimeError('H28 fixed compute budget exceeded')
    emit(out/'all-192.json',rows)
    scalar_keys=[k for k,v in rows[0].items() if not isinstance(v,(list,dict))]
    with (out/'all-192.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,scalar_keys,extrasaction='ignore');w.writeheader();w.writerows(rows)
    def guard(name,check):
        try:check();guards.append(dict(name=name,passed=True))
        except Exception as exc:guards.append(dict(name=name,passed=False,error=repr(exc)))
    base=dict(common,capacity=20,deadline_minutes=30,stop_setup_seconds=60,minimum_transfer=3)
    departure=datetime(2026,11,11,tzinfo=timezone.utc).timestamp()
    def run(roads=None,s=None,**kw):return model.optimize(s or known_state(),dict(base,**kw),roads or FixedRoads(),departure)
    def expect(condition):
        if not condition:raise AssertionError('Known case mismatch')
    guard('known feasible batch',lambda:expect(run()['moved_bikes']>=3))
    guard('minimum3 cannot collect supply2',lambda:expect(run(s=known_state(8))['moved_bikes']==0))
    guard('setup600 blocks all work',lambda:expect(run(stop_setup_seconds=600)['moved_bikes']==0))
    guard('return after forecast allowed before job30',lambda:expect(run()['route']['return_minutes']>10 and run()['moved_bikes']>=3))
    guard('same long return blocked by job10',lambda:expect(run(deadline_minutes=10)['moved_bikes']==0))
    guard('service after forecast cutoff blocked',lambda:expect(run(roads=FixedRoads(between=600))['moved_bikes']==0))
    def tamper():
        r=run();r['route']['stops'][0]['completion_minutes']-=1
        try:model.certificate(known_state(),r)
        except AssertionError:return
        raise AssertionError('Certificate accepted missing setup time')
    guard('independent certificate rejects removed setup',tamper)
    # Real CLI execution using a public, deterministic road-cache fixture.
    cli_state=out/'cli-synthetic-state.json';emit(cli_state,known_state())
    from dispatch_engine import Roads
    roads=Roads(out/'cli-road-cache');locations=[known_state()['stations'][0]]+known_state()['stations'][1:]
    coords=';'.join(f"{s['lon']:.6f},{s['lat']:.6f}" for s in locations)
    url=f'https://router.project-osrm.org/table/v1/driving/{coords}?annotations=duration,distance'
    seconds,meters,_=FixedRoads().table(locations)
    emit(roads.cache/(hashlib.sha256(url.encode()).hexdigest()+'.json'),dict(code='Ok',durations=seconds.tolist(),distances=meters.tolist()))
    # Geometry is optional. Cache every possible positive two-stop route.
    route_locs=locations+[locations[0]]
    route_coords=';'.join(f"{s['lon']:.6f},{s['lat']:.6f}" for s in route_locs)
    route_url=f'https://router.project-osrm.org/route/v1/driving/{route_coords}?overview=full&geometries=geojson&steps=false'
    emit(roads.cache/(hashlib.sha256(route_url.encode()).hexdigest()+'.json'),dict(code='Ok',routes=[dict(geometry=dict(type='LineString',coordinates=[]))]))
    command=[sys.executable,str(root/'research_code/batch_operations.py'),'--state',str(cli_state),'--roads',str(roads.cache),'--out',str(out/'cli-result.json'),'--departure-offset','0','--body',json.dumps(base)]
    proc=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=30)
    emit(out/'cli.json',dict(exit_code=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))
    guards.append(dict(name='CLI fixture',passed=proc.returncode==0 and json.loads((out/'cli-result.json').read_text(encoding='utf-8'))['moved_bikes']>=3))
    emit(out/'guards.json',guards)
    # Compare equal delivered quantities only, while retaining all excluded cases.
    comparisons=[]
    for minimum in [3,5]:
        pair_count=equal=improved=worse=tied=0;delta=[]
        for old in rows:
            if old['minimum_transfer']!=1 or old['status']=='error':continue
            new=next(r for r in rows if all(r[k]==old[k] for k in ('layout','setup_seconds','capacity','job_deadline_minutes','seed')) and r['minimum_transfer']==minimum)
            pair_count+=1
            if old['moved_bikes']==new.get('moved_bikes',-1)>0:
                equal+=1;d=new['work_minutes_per_delivered_bike']-old['work_minutes_per_delivered_bike'];delta.append(d)
                improved+=d<-1e-8;worse+=d>1e-8;tied+=abs(d)<=1e-8
        comparisons.append(dict(minimum_transfer=minimum,all_pairs=pair_count,equal_positive_count_pairs=equal,excluded_pairs=pair_count-equal,less_minutes_per_bike=improved,more_minutes_per_bike=worse,ties=tied,mean_delta_minutes_per_bike=float(np.mean(delta)) if delta else None))
    # Historical algorithm is diagnosed under its old assumptions; never compared
    # as evidence that new setup accounting improved efficiency.
    legacy=[]
    for layout,seed in itertools.product(spec['synthetic']['layouts'],[17,29]):
        s=state(layout,seed);r=model.base.optimize(s,dict(common,capacity=20,deadline_minutes=10),SyntheticRoads(layout,seed),departure)
        setup_added=len(r['route']['stops'])
        legacy.append(dict(layout=layout,seed=seed,moved_bikes=r['moved_bikes'],stops=len(r['route']['stops']),reported_minutes=r['route']['return_minutes'],minutes_with_assumed_60sec_setup=r['route']['return_minutes']+setup_added,role='physical-cost omission diagnostic, not matched baseline'))
    emit(out/'legacy-diagnostic.json',legacy)
    summary=dict(cases=len(rows),violations=sum(r['violations'] for r in rows),guards_passed=sum(r['passed'] for r in guards),guards_total=len(guards),
                 no_action=sum(r['status']=='no_action' for r in rows),errors=failures,matched_comparisons=comparisons,
                 seconds=time.perf_counter()-started,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                 execution_host='dislab',new_inventory_api_requests=0,model_training=0,field_effect=None)
    emit(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if failures or summary['guards_passed']!=len(guards):raise SystemExit(1)


if __name__=='__main__':main()
