"""H21 handling-aware capacity union. DISLAB ONLY; frozen forecasts and H19 loss.

For fixed state/candidates/resources except capacity, the search for C contains
every subproblem searched for C-1. No new dominance/optimality claim elsewhere.
"""
import argparse
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import sys
import time
import bulk_core as base

original_search=base.search
original_rpc=base.rpc


def capacity_limit(p):
    # Each transferred bicycle must be picked up and delivered; this is a physical upper bound.
    handling_bound=int(max(0,p['effective_deadline'])*60//(2*p['handling_seconds']))
    return max(1,min(p['capacity'],p['budget'],handling_bound))


def capacity_search(stations,p,seconds,meters,legacy_ids):
    started=time.perf_counter();runs=[]
    for cap in range(1,capacity_limit(p)+1):
        runs.append(original_search(stations,dict(p,capacity=cap),seconds,meters,legacy_ids))
    def key(result):
        r=result['route'];return (-result['net_value'],r['distance_meters'],r['return_minutes'],
            tuple((s['station_id'],s['action'],s['bikes']) for s in r['stops']))
    best=min(runs,key=key);result=dict(best);result['controls']=dict(runs[-1]['controls'])
    result['controls']['direct_capacity']=dict(net_value=runs[-1]['net_value'],gain=runs[-1]['gain'],moved_bikes=runs[-1]['chosen'][3],route=runs[-1]['route'])
    if len(runs)>1:
        small=min(runs[:-1],key=key);result['controls']['smaller_capacity']=dict(net_value=small['net_value'],gain=small['gain'],moved_bikes=small['chosen'][3],route=small['route'])
    result['expanded']=sum(r['expanded'] for r in runs);result['solver_seconds']=time.perf_counter()-started
    return result


base.search=capacity_search


def rpc(folder,request):
    response=original_rpc(folder,request)
    if response['status']==200 and request.get('path')!='/api/dispatch/info':
        body=response['body'];body['planner']='bulk-capacity-incumbent-union'
        body['latest_departure_at_utc']=(datetime.fromisoformat(body['valid_until_utc'])-timedelta(minutes=body['route']['return_minutes'])).isoformat()
        body['proof'].update(capacity_subproblems=list(range(1,capacity_limit(body['parameters'])+1)),
            capacity_inclusion='All other inputs fixed; every smaller-capacity feasible incumbent retained',
            method='handling-aware beam64/max8 visits at each feasible subcapacity; heuristic')
    if request.get('path')=='/api/dispatch/info' and response['status']==200:display_limits(response['body'])
    if response.get('body',{}).get('current_info'):display_limits(response['body']['current_info'])
    return response


def display_limits(info):
    info['limits'].update(capacity=40,budget=80,target=12,drop_limit=30,handling_seconds=120,visits=8)
    info['operating_defaults']=dict(capacity=20,budget=20,target=6,drop_limit=10,handling_seconds=20,max_visits=8)


base.rpc=rpc


def watch(folder,out,maximum=24*3600):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();last=None;previous=None;last_attempt=-1e9
    while time.monotonic()-began<maximum:
        status=base.read(folder/'status.json')
        if status.get('phase') in ('failed','stopped'):break
        now=base.reference_now(status);version=status.get('source_version')
        expiring=previous and previous['status']==200 and previous['body']['moved_bikes']>0 and now+15>=datetime.fromisoformat(previous['body']['latest_departure_at_utc']).timestamp()
        if version and (version!=last or (expiring and time.monotonic()-last_attempt>=30)):
            try:
                response=rpc(folder,dict(body=dict(source_version=version,automatic=True)))
                stamp=int(base.reference_now(status)*1000);base.atomic(out/(version+'-'+str(stamp)+'.json'),response);base.atomic(out/'latest.json',response)
                print(json.dumps(dict(source_version=version,status=response['status'],moved=response['body'].get('moved_bikes'),issued_at_utc=response['body'].get('issued_at_utc')),ensure_ascii=False),flush=True)
                previous=response
            except Exception as exc:
                base.atomic(out/(version+'-error.json'),dict(error=type(exc).__name__));previous=None;print(type(exc).__name__,flush=True)
            last=version;last_attempt=time.monotonic()
        time.sleep(5)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['rpc','watch']);ap.add_argument('--folder',required=True);ap.add_argument('--out');a=ap.parse_args()
    if a.command=='watch':watch(a.folder,a.out)
    else:
        try:response=rpc(a.folder,json.load(sys.stdin))
        except (ValueError,KeyError,TypeError) as exc:response=dict(status=422,body=dict(error=str(exc)))
        except Exception:response=dict(status=503,body=dict(error='도로 또는 재배치 계산을 완료하지 못했습니다.'))
        print(json.dumps(response,ensure_ascii=False,allow_nan=False))
