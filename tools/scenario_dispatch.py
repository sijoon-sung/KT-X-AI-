"""Calculate several independent, road-routed dispatch plans from frozen observations."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research_code'))
import batch_operations as planner
from dispatch_engine import Roads, read

KST=timezone(timedelta(hours=9))
LOCK='767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09'


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def local_candidates(state,center,params):
    stations=state['stations']
    supply,demand=planner.bounds(stations,params['target'],params['capacity'],params.get('drop_limit',10))
    minimum=params['minimum_transfer']
    anchor=dict(lat=center[0],lon=center[1]);radius=center[2]
    local=[(planner.base.air(anchor,s),s,int(a),int(b)) for s,a,b in zip(stations,supply,demand)]
    local=sorted((row for row in local if row[0]<=radius),key=lambda row:(row[0],row[1]['id']))
    donors=[s for _,s,a,_ in local if a>=minimum]
    receivers=[s for _,s,_,b in local if b>=minimum]
    if not donors or not receivers:return None
    return dict(automatic=False,depot_id=donors[0]['id'],station_ids=[s['id'] for s in donors[:8]+receivers[:12]])


def stamp(value):return datetime.fromtimestamp(value+.5,KST).isoformat(timespec='seconds')


def calculate(spec,stream,roads):
    at=int(spec['stamp']);path=stream/f'state-{at}.json';state=read(path)
    if state['model_lock_sha256']!=LOCK or state['source_version']!=f'H18-{at}':raise ValueError('Unexpected forecast input')
    forecast=stream/f'forecast-{at}.npz'
    import numpy as np
    with np.load(forecast,allow_pickle=False) as f:issued=float(f['issued_at_unix_seconds'])
    departure=max(at+15,math.ceil(issued+1))
    parameters=dict(capacity=20,target=6,budget=20,stop_setup_seconds=60,handling_seconds=20,minimum_transfer=3,deadline_minutes=15)
    parameters.update(spec.get('parameters',{}))
    if parameters['minimum_transfer']<3:raise ValueError('A case requires at least three bicycles per working stop')
    output=dict(id=spec['id'],name=spec['name'],source_version=state['source_version'],state_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                observation=stamp(at),target=stamp(at+600),departure=stamp(departure),cutoff=stamp(at+570),return_deadline=stamp(departure+60*parameters['deadline_minutes']))
    if spec.get('center'):
        candidates=local_candidates(state,spec['center'],parameters)
        if candidates is None:
            return dict(output,status='no_action',moved_bikes=0,stops=[],reason='권역 안에 최소 작업량을 충족하는 수거·배치 후보 쌍이 없습니다.',parameters=parameters)
        parameters.update(candidates)
    plan=planner.optimize(state,parameters,roads,departure)
    planner.certificate(state,plan)
    lookup={s['id']:s for s in state['stations']}
    stops=[]
    for stop in plan['route']['stops']:
        station=lookup[stop['station_id']]
        stops.append(dict(stop,name=station['name'],lat=station['lat'],lon=station['lon'],current=station['current'],
            probability_zero=station['forecast']['p0'],arrival=stamp(departure+60*stop['eta_minutes']),completion=stamp(departure+60*stop['completion_minutes'])))
    if plan['moved_bikes'] and plan['geometry_status']!='ok':raise ValueError('Road geometry unavailable for computed plan')
    depot=lookup[plan['parameters']['depot_id']]
    return dict(output,status=plan['status'],reason=plan['no_action_reason'],parameters=plan['parameters'],
        depot={key:depot[key] for key in ('id','name','lon','lat')},stops=stops,moved_bikes=plan['moved_bikes'],
        distance_km=round(plan['route']['distance_meters']/1000,3),work_minutes=plan['route']['return_minutes'],
        returned=stamp(departure+60*plan['route']['return_minutes']) if stops else None,
        setup_minutes=plan['operations']['setup_minutes'],handling_minutes=plan['operations']['handling_minutes'],
        work_minutes_per_bike=plan['operations']['work_minutes_per_delivered_bike'],geometry=plan['geometry'],
        certificate=plan['certificate'],road_matrix_key=plan['road_matrix_key'])


def main():
    ap=argparse.ArgumentParser()
    for key in ('stream','spec','roads','out'):ap.add_argument('--'+key,type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    specs=read(args.spec)['cases'];keys=[s['id'] for s in specs]
    if not keys or len(set(keys))!=len(keys) or any(not re.fullmatch(r'[a-z0-9-]{1,36}',k) for k in keys):raise ValueError('Invalid case IDs')
    roads=Roads(args.roads);results=[]
    for spec in specs:
        try:result=calculate(spec,args.stream,roads)
        except (ValueError,OSError,AssertionError) as exc:
            result=dict(id=spec['id'],name=spec['name'],status='error',reason=str(exc),stops=[])
        results.append(result)
        print(json.dumps({k:result[k] for k in ('id','status','moved_bikes','distance_km','reason') if k in result},ensure_ascii=False),flush=True)
        if result.get('geometry'):
            features=[dict(type='Feature',properties=dict(case=spec['id'],kind='vehicle_route'),geometry=result['geometry'])]
            for seq,stop in enumerate(result['stops'],1):
                features.append(dict(type='Feature',properties=dict(stop,sequence=seq),geometry=dict(type='Point',coordinates=[stop['lon'],stop['lat']])))
            write(args.out/(spec['id']+'.geojson'),dict(type='FeatureCollection',features=features))
    write(args.out/'plans.json',dict(schema_version=1,mode='recorded_replay',cases=results))
    fields=['case','sequence','name','action','bikes','arrival','completion','vehicle_load_after']
    with (args.out/'work-orders.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for result in results:
            for seq,stop in enumerate(result['stops'],1):writer.writerow(dict(stop,case=result['name'],sequence=seq))


if __name__=='__main__':main()
