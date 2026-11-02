"""Run H17 engineering checks on dislab; individual plans stay private."""
import argparse
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
from dispatch_engine import read,write,information,parameters,exhaustive,optimize,Roads,sha,ROOT


def main(a):
    started=time.perf_counter();state=read(a.state);checks=[];cases={}
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        if not value:raise AssertionError(name)
    oracle=[dict(id='donor',current=6,samples=[6]*128),dict(id='sink',current=0,samples=[0]*128)]
    p=dict(target=2,budget=6,capacity=2,deadline_minutes=10)
    seconds=np.array([[0,10,40],[15,0,20],[30,22,0]],float);meters=seconds*10
    result=exhaustive(oracle,p,seconds,meters)['best']['balanced']
    check('tiny_independent_optimum_oracle',result['net']==[-2,2] and abs(result['objective_value']-(.03*.6+.01*2))<1e-10)
    check('directed_distance_and_handling_oracle',result['route']['distance_meters']==600 and abs(result['route']['return_minutes']-1.4)<1e-10)
    p['capacity']=1;low=exhaustive(oracle,p,seconds,meters)['best']['balanced']
    check('capacity_changes_exact_quantity',low['moved']==1 and low['route']['peak_load']==1)
    for name,body in [('unknown',dict(station_ids=['unknown'],depot_id='unknown')),('too_many',dict(station_ids=state['default_ids']+['extra'])),('unsupported_horizon',dict(deadline_minutes=30)),('nan',dict(capacity=float('nan')))]:
        try:parameters(state,body)
        except (ValueError,TypeError,OverflowError):check('reject_'+name,True)
        else:check('reject_'+name,False)
    public=information(state)
    check('future_truth_and_samples_not_in_ui_response',all('samples' not in s and 'truth' not in s for s in public['stations']) and not any(k in state for k in ['truth','live_metrics','september']))
    roads=Roads(a.cache)
    variants=[('default',{}),('capacity1',dict(capacity=1)),('deadline3',dict(deadline_minutes=3)),
              ('exclude_one',dict(station_ids=state['default_ids'][:-1])),('distance',dict(objective='distance')),('time',dict(objective='time'))]
    for name,body in variants:
        original=sha(a.state);r=optimize(state,body,roads);cases[name]=r;p=r['parameters'];route=r['route']
        check(name+'_complete_and_no_dispatch',r['proof']['complete_search'] and not r['dispatch_allowed'] and r['field_effect'] is None)
        check(name+'_distance_conservation',abs(sum(s['leg_meters'] for s in route['stops'])+route['return_leg_meters']-route['distance_meters'])<1e-7)
        elapsed=sum(s['leg_drive_minutes']+.1*s['bikes'] for s in route['stops'])+route['return_leg_minutes']
        check(name+'_time_and_load',abs(elapsed-route['return_minutes'])<1e-7 and route['return_minutes']<=p['deadline_minutes']+1e-8 and all(0<=s['vehicle_load_after']<=p['capacity'] for s in route['stops']))
        check(name+'_bike_and_source_bounds',sum(r['net'])==0 and r['moved_bikes']<=p['budget'] and all(-v<=limit for v,limit in zip(r['net'],r['source_safety_limits'])))
        check(name+'_input_file_unchanged',sha(a.state)==original)
    check('candidate_exclusion_changes_route_domain',all(s['station_id'] in cases['exclude_one']['candidate_ids'] for s in cases['exclude_one']['route']['stops']))
    text=(ROOT/'research_code/forecast_web/dispatch.html').read_text(encoding='utf-8')
    check('operational_ui_without_research_sections',not any(t in text for t in ['가설','실험','PR23','H-0013','H-0014','H-0015','H-0016','H-0017']))
    check('minimal_correct_model_attribution','Flow Matching + GCN' in text)
    elapsed=time.perf_counter()-started;stamp=datetime.fromisoformat(a.collector_start)
    summary=dict(schema_version=1,hypothesis_id='H-0017',started_at_utc=stamp.isoformat(),finished_at_utc=(stamp+timedelta(seconds=elapsed)).isoformat(),
        native_remote_finished_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=dict(device='dislab CPU',python=platform.python_version(),numpy=np.__version__),checks=checks,
        metrics=dict(acceptance_failures=sum(not c['passed'] for c in checks)),seconds=elapsed,
        cases={k:dict(moved_bikes=v['moved_bikes'],distance_meters=v['route']['distance_meters'],return_minutes=v['route']['return_minutes'],peak_load=v['route']['peak_load'],complete_search=v['proof']['complete_search']) for k,v in cases.items()},
        field_benefit_observed=False,dispatch_allowed=False)
    write(a.out,summary);write(Path(a.out).with_suffix('.private-plans.json'),cases)
    print(json.dumps(dict(checks=len(checks),metrics=summary['metrics'],cases=summary['cases'],seconds=elapsed)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--collector-start',required=True)
    main(p.parse_args())
