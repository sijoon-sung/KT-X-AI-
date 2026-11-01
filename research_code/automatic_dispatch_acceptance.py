"""H19 preregistered engineering and three-state shadow diagnostics; dislab CPU."""
import argparse
from datetime import datetime,timezone
import hashlib
import itertools
import json
from pathlib import Path
import platform
import time
import numpy as np
import automatic_dispatch as a
from dispatch_engine import evaluate_routes


def engineering():
    rng=np.random.default_rng(42);rows=[];checks=0
    for seed in range(20):
        current=[8,7,0,1];stations=[dict(id=str(i),name='Synthetic'+str(i),lat=36.35+i*.001,lon=127.38,current=c,samples=[c]*128) for i,c in enumerate(current)]
        n=len(stations);points=rng.uniform(0,1,(n+1,2));dist=np.sqrt(((points[:,None]-points[None,:])**2).sum(2))*600
        seconds=dist/5+rng.uniform(0,10,(n+1,n+1));np.fill_diagonal(seconds,0)
        state=dict(stations=stations,default_ids=[s['id'] for s in stations],default_depot_id='0',source_version='synthetic',recorded_at_utc='2026-11-01T07:00:00+00:00',target_at_utc='2026-11-01T07:10:00+00:00')
        p=a.parameters(state,dict(capacity=1+seed%6,budget=2+seed%5));p['effective_deadline']=3+seed%7
        found=a.search(stations,p,seconds,dist,state['default_ids']);supply,demand=a.bounds(stations,p['target']);before=sum(a.loss(s['samples'],p['target']) for s in stations);exact=0.;coeff=a.MODES[p['objective']]
        for net in itertools.product(*[range(-int(supply[i]),1) if supply[i] else range(0,int(demand[i])+1) for i in range(n)]):
            if sum(net) or sum(max(v,0) for v in net)>p['budget']:continue
            gain=before-sum(a.loss(s['samples'],p['target'],q) for s,q in zip(stations,net))
            for route in evaluate_routes(np.array(net),seconds,dist,p['capacity'],p['effective_deadline'],[s['id'] for s in stations]):
                score=gain-coeff[0]*route['distance_meters']/1000-coeff[1]*route['return_minutes']-coeff[2]*sum(max(v,0) for v in net);exact=max(exact,score)
        result=dict(parameters=p,route=found['route'],moved_bikes=found['chosen'][3]);a.certificate(state,result);checks+=1
        assert found['net_value']<=exact+1e-8;checks+=1
        assert all(v is None or found['net_value']+1e-8>=v['net_value'] for v in found['controls'].values());checks+=1
        rows.append(dict(case=seed,beam=found['net_value'],exact=exact,gap=exact-found['net_value']))
    # Explicit no supply, unreachable roads and exhausted horizon.
    for mode in ('no_supply','unreachable','expired'):
        sites=[dict(s) for s in stations]
        if mode=='no_supply':
            for s in sites:s.update(current=0,samples=[0]*128)
        sec=seconds.copy()
        if mode=='unreachable':sec[:]=np.inf;np.fill_diagonal(sec,0)
        q=dict(p,effective_deadline=0 if mode=='expired' else 5)
        found=a.search(sites,q,sec,dist,[]);assert found['chosen'][3]==0;checks+=1
    return dict(checks=checks,violations=0,oracle_cases=rows,maximum_oracle_gap=max(r['gap'] for r in rows))


def run(folder,out):
    started=datetime.now(timezone.utc).isoformat();folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    synthetic=engineering();a.atomic(out/'engineering.json',synthetic);seen=set();summaries=[];began=time.monotonic()
    while len(seen)<3 and time.monotonic()-began<2100:
        status=a.read(folder/'status.json');version=status.get('source_version')
        if not version or version in seen:time.sleep(5);continue
        state=a.read(folder/status['latest_state'])
        # Honor the frozen period; an early cached state is not a primary observation.
        if datetime.fromisoformat(state['recorded_at_utc']).timestamp()<datetime.fromisoformat('2026-11-01T15:50:00+09:00').timestamp():
            time.sleep(5);continue
        departure=datetime.fromisoformat(state['recorded_at_utc']).timestamp()+10
        rawpath=folder/status['latest_state'];inputhash=hashlib.sha256(rawpath.read_bytes()).hexdigest();seen.add(version)
        try:
            result=a.optimize(state,dict(automatic=True),a.Roads(folder/'h19-roads'),departure)
            a.atomic(out/(version+'.json'),result)
            controls=result['comparison'];legacy=controls.get('legacy_fixed5');bestpair=controls.get('best_pair')
            gain=result['risk_before']-result['objective_value']
            assert all(v is None or gain+1e-8>=v['net_value'] for v in controls.values())
            summaries.append(dict(source_version=version,input_sha256=inputhash,recorded_at_utc=state['recorded_at_utc'],
                moved_bikes=result['moved_bikes'],km=result['route']['distance_meters']/1000,minutes=result['route']['return_minutes'],
                zero_sites_supplied=result['coverage']['zero_sites_supplied'],unserved_zero=result['coverage']['unserved_zero'],
                risk_before=result['risk_before'],risk_after=result['risk_after'],net_value=gain,solver_seconds=result['solver_seconds'],total_seconds=result['seconds'],
                legacy_net_value=legacy['net_value'] if legacy else None,legacy_moved=legacy['moved_bikes'] if legacy else None,
                best_pair_net_value=bestpair['net_value'] if bestpair else None,violations=0))
        except Exception as exc:summaries.append(dict(source_version=version,input_sha256=inputhash,error=type(exc).__name__,violations=1))
        a.atomic(out/'shadow-progress.json',summaries);print(json.dumps(summaries[-1],ensure_ascii=False),flush=True)
    failures=sum(r['violations'] for r in summaries);strict=int(sum('net_value'in r and r['legacy_net_value'] is not None and r['net_value']>r['legacy_net_value']+1e-8 for r in summaries))
    summary=dict(started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),environment=dict(python=platform.python_version(),numpy=np.__version__,device='dislab CPU'),
        engineering=synthetic,states=summaries,acceptance_failures=failures,strict_legacy_improvements=strict,
        decision='supported' if failures==0 and len(summaries)==3 and strict else 'not_supported' if len(summaries)==3 else 'inconclusive',
        limitations=['Marginal-model proxy; no observed dispatch or shortage-prevention effect','Fixed H13 weight/sample generation, no API fitting','Heuristic shortlist and beam; not global optimum','Native environment time stored with H18 calibrated source time, not used to backdate forecast issuance'])
    a.atomic(out/'summary.json',summary);print('H19 completed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--out',required=True);q=p.parse_args();run(q.folder,q.out)
