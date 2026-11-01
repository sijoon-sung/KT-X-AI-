"""H20 preregistered20x6 synthetic cases; no API learning or local execution."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import platform
import time
import numpy as np
import monotone_dispatch as m
a=m.base


def run(out):
    start=datetime.now(timezone.utc).isoformat();rng=np.random.default_rng(43);rows=[];violations=0;direct_inversions=0
    for seed in range(20):
        stocks=[8,7,0,1];stations=[dict(id=str(i),current=c,samples=[c]*128) for i,c in enumerate(stocks)]
        points=rng.uniform(0,1,(5,2));meters=np.sqrt(((points[:,None]-points[None,:])**2).sum(2))*600
        seconds=meters/5+rng.uniform(0,10,(5,5));np.fill_diagonal(seconds,0)
        state=dict(stations=stations);previous=0.;previous_direct=0.
        for cap in range(1,7):
            p=dict(capacity=cap,target=2,budget=6,objective='balanced',effective_deadline=8,deadline_minutes=8)
            found=m.capacity_search(stations,p,seconds,meters,[s['id'] for s in stations]);direct=found['controls']['direct_capacity']['net_value']
            a.certificate(state,dict(parameters=p,route=found['route'],moved_bikes=found['chosen'][3]))
            violation=found['net_value']+1e-8<previous or found['net_value']+1e-8<direct
            violations+=int(violation);direct_inversions+=int(direct+1e-8<previous_direct)
            rows.append(dict(case=seed,capacity=cap,net_value=found['net_value'],direct_net_value=direct,moved=found['chosen'][3],
                solver_seconds=found['solver_seconds'],violations=int(violation)))
            previous=found['net_value'];previous_direct=direct
    result=dict(hypothesis_id='H-0020',started_at_utc=start,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        environment=dict(python=platform.python_version(),numpy=np.__version__,device='dislab CPU'),cases=rows,
        result_count=len(rows),capacity_or_constraint_violations=violations,direct_capacity_inversions=direct_inversions,
        maximum_solver_seconds=max(r['solver_seconds'] for r in rows),median_solver_seconds=float(np.median([r['solver_seconds'] for r in rows])),
        decision='supported' if violations==0 and len(rows)==120 else 'not_supported',
        limitations=['Synthetic engineering checks, no field effect or accuracy gain','Capacity inclusion only under identical other inputs','Marginal loss and resources assumed; no global optimum'])
    a.atomic(Path(out),result);print({k:v for k,v in result.items() if k!='cases'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);q=p.parse_args();run(q.out)
