"""H19 automatic screening and feasible pickup/delivery search. DISLAB ONLY.

Fixed marginal operational loss, unchanged H13 forecasts, no model fitting.
Beam search is heuristic. Feasible pair/legacy/no-action fallbacks are retained.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from dispatch_engine import Roads, integer, exhaustive, read, write
from live_api_service import reference_now, atomic

MODES={'balanced':(.03,0.,.01),'distance':(.5,0.,.01),'time':(0.,.07,.01)}


def loss(samples,target,net=0):
    value=np.asarray(samples)+net
    return float(np.maximum(target-value,0).mean()+2*(value<=0).mean())


def bounds(stations,target):
    x=np.asarray([s['samples'] for s in stations],float)
    if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all() or (x<0).any():
        raise ValueError('Invalid frozen marginal samples.')
    low=np.sort(x,axis=1)[:,12];current=np.asarray([s['current'] for s in stations])
    supply=np.maximum(0,np.minimum.reduce([np.floor(current-target),np.floor(low-target),x.min(1),np.full(len(x),6)])).astype(int)
    demand=np.minimum(3,np.maximum(0,np.ceil(target-low))).astype(int)
    demand[supply>0]=0
    return supply,demand


def air(a,b):
    lat1,lat2=math.radians(a['lat']),math.radians(b['lat']);dlat=lat2-lat1;dlon=math.radians(b['lon']-a['lon'])
    return 6371*2*math.asin(min(1,math.sqrt(math.sin(dlat/2)**2+math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2)))


def parameters(state,body):
    allowed={'station_ids','depot_id','capacity','target','deadline_minutes','budget','objective','automatic'}
    if not isinstance(body,dict) or set(body)-allowed:raise ValueError('Unknown planning parameter.')
    known={s['id']:s for s in state['stations']};depot=body.get('depot_id',state['default_depot_id'])
    if depot not in known:raise ValueError('Unknown start/return position.')
    mode=body.get('objective','balanced')
    if mode not in MODES:raise ValueError('Unknown objective.')
    automatic=body.get('automatic',True)
    if not isinstance(automatic,bool):raise ValueError('Invalid automatic flag.')
    ids=body.get('station_ids',state['default_ids'])
    if not isinstance(ids,list) or not 1<=len(ids)<=29 or len(ids)!=len(set(ids)) or any(i not in known for i in ids):
        raise ValueError('Invalid manual candidates.')
    return dict(automatic=automatic,station_ids=ids,depot_id=depot,capacity=integer(body.get('capacity',6),1,12,'capacity'),
        target=integer(body.get('target',2),1,5,'target'),deadline_minutes=integer(body.get('deadline_minutes',10),1,10,'deadline'),
        budget=integer(body.get('budget',6),1,12,'budget'),objective=mode)


def screen(state,p):
    rows=state['stations'];lookup={s['id']:s for s in rows};depot=lookup[p['depot_id']]
    supply,demand=bounds(rows,p['target']);sp={s['id']:int(v) for s,v in zip(rows,supply)};dm={s['id']:int(v) for s,v in zip(rows,demand)}
    receivers=[s for s in rows if dm[s['id']]>0]
    receivers.sort(key=lambda s:(-(loss(s['samples'],p['target'])-loss(s['samples'],p['target'],dm[s['id']]))/(1+air(depot,s)),s['id']))
    top=receivers[:12]
    donors=[s for s in rows if sp[s['id']]>0]
    donors.sort(key=lambda s:(air(depot,s)+min([air(s,t) for t in top] or [0]),s['id']))
    ids=list(dict.fromkeys([s['id'] for s in top]+[s['id'] for s in donors[:12]]+state['default_ids'])) if p['automatic'] else p['station_ids']
    return [lookup[i] for i in ids],dict(scanned=len(rows),current_zero=sum(s['current']==0 for s in rows),
        receivers=len(receivers),donors=len(donors),shortlisted=len(ids),supply=sp,demand=dm,
        selection='Automatic regional shortlist from all supported stations; no citywide optimum')


def empty_route():
    return dict(stops=[],return_minutes=0.,return_leg_minutes=0.,return_leg_meters=0.,distance_meters=0.,peak_load=0)


def route_from(order,quantities,stations,seconds,meters):
    route=empty_route();last=0;clock=0.;distance=0.;load=0
    for i,q in zip(order,quantities):
        leg=float(seconds[last,i+1])/60;leg_m=float(meters[last,i+1]);arrival=clock+leg;clock=arrival+abs(q)*.1;distance+=leg_m;load-=q
        route['stops'].append(dict(station_id=stations[i]['id'],action='deliver' if q>0 else 'pickup',bikes=abs(q),
            leg_meters=leg_m,leg_drive_minutes=leg,eta_minutes=arrival,completion_minutes=clock,vehicle_load_after=load))
        last=i+1
    back=float(seconds[last,0])/60;back_m=float(meters[last,0])
    route.update(return_minutes=clock+back,return_leg_minutes=back,return_leg_meters=back_m,distance_meters=distance+back_m,
        peak_load=max([0]+[s['vehicle_load_after'] for s in route['stops']]))
    return route


def search(stations,p,seconds,meters,legacy_ids):
    started=time.perf_counter();n=len(stations);supply,demand=bounds(stations,p['target']);coeff=MODES[p['objective']]
    gains=[{q:loss(s['samples'],p['target'])-loss(s['samples'],p['target'],q) for q in range(-int(supply[i]),int(demand[i])+1)} for i,s in enumerate(stations)]
    # tuple: order, quantities, load, delivered, elapsed work minutes, distance m, marginal gain, visited bits
    origin=((),(),0,0,0.,0.,0.,0);best=(0.,(),(),origin);expanded=0
    controls={}
    def value(st):
        last=st[0][-1]+1 if st[0] else 0
        return st[6]-coeff[0]*(st[5]+meters[last,0])/1000-coeff[1]*(st[4]+seconds[last,0]/60)-coeff[2]*st[3]
    def consider(st,label=None):
        nonlocal best
        if st[2] or st[4]+seconds[st[0][-1]+1 if st[0] else 0,0]/60>p['effective_deadline']+1e-9:return
        key=(-value(st),st[0],st[1],st)
        if key[:3]<best[:3]:best=key
        if label is not None:
            old=controls.get(label)
            if old is None or value(st)>value(old):controls[label]=st
    consider(origin,'no_action')
    def extend(st,i,q):
        nonlocal expanded
        expanded+=1;last=st[0][-1]+1 if st[0] else 0
        clock=st[4]+seconds[last,i+1]/60+abs(q)*.1;distance=st[5]+meters[last,i+1]
        load=st[2]-q;delivered=st[3]+max(0,q)
        if not 0<=load<=p['capacity'] or delivered+load>p['budget'] or clock+seconds[i+1,0]/60>p['effective_deadline']+1e-9 or not np.isfinite(distance+clock):return None
        return (st[0]+(i,),st[1]+(q,),load,delivered,float(clock),float(distance),st[6]+gains[i][q],st[7]|(1<<i))
    # Exact feasible two-stop fallback, with nearest feasible pair retained separately.
    nearest=None
    for i in range(n):
        for j in range(n):
            for q in range(1,min(int(supply[i]),int(demand[j]),p['capacity'],p['budget'])+1):
                first=extend(origin,i,-q)
                if first is None:continue
                second=extend(first,j,q)
                if second is None:continue
                consider(second,'best_pair')
                near_key=(second[5]+meters[j+1,0],stations[i]['id'],stations[j]['id'])
                if nearest is None or near_key<nearest[0]:nearest=(near_key,second)
    if nearest is not None:controls['nearest_pair']=nearest[1]
    # Original bounded exact H17 policy is a comparison/fallback, never a fitted selector.
    index={s['id']:i for i,s in enumerate(stations)};legacy=[index[i] for i in legacy_ids if i in index]
    if legacy:
        subp=dict(p,station_ids=[stations[i]['id'] for i in legacy],deadline_minutes=p['effective_deadline'])
        take=[0]+[i+1 for i in legacy]
        try:
            previous=exhaustive([stations[i] for i in legacy],subp,seconds[np.ix_(take,take)],meters[np.ix_(take,take)])
            route=previous['best'][p['objective']]['route'];st=origin
            for stop in route['stops']:
                i=index[stop['station_id']];q=stop['bikes']*(1 if stop['action']=='deliver' else -1);st=extend(st,i,q)
                if st is None:break
            if st is not None:consider(st,'legacy_fixed5')
        except TimeoutError:controls['legacy_timeout']=None
    # Deterministic beam search; retain best feasible closed route at every depth.
    beam=[origin]
    max_gain=max([0.]+[gains[i].get(1,0) for i in range(n)])
    for _depth in range(6):
        states=[]
        for st in beam:
            for i in range(n):
                if st[7]&(1<<i):continue
                options=range(-min(int(supply[i]),p['capacity']-st[2],p['budget']-st[3]-st[2]),0) if supply[i]>0 else range(1,min(int(demand[i]),st[2],p['budget']-st[3])+1)
                for q in options:
                    child=extend(st,i,q)
                    if child is None:continue
                    consider(child);states.append(child)
        states.sort(key=lambda st:(-(value(st)+st[2]*max_gain),st[4],st[0],st[1]))
        beam=states[:64]
        if not beam:break
    chosen=best[3]
    return dict(chosen=chosen,route=route_from(chosen[0],chosen[1],stations,seconds,meters),gain=chosen[6],net_value=value(chosen),
        controls={name:None if st is None else dict(net_value=value(st),gain=st[6],moved_bikes=st[3],route=route_from(st[0],st[1],stations,seconds,meters)) for name,st in controls.items()},
        supply=supply.tolist(),demand=demand.tolist(),expanded=expanded,solver_seconds=time.perf_counter()-started)


def certificate(state,result):
    p=result['parameters'];lookup={s['id']:s for s in state['stations']};load=0;seen=set();moved=0
    supply,demand=bounds(state['stations'],p['target']);limits={s['id']:(int(a),int(b)) for s,a,b in zip(state['stations'],supply,demand)}
    for stop in result['route']['stops']:
        sid=stop['station_id'];q=stop['bikes']
        if sid in seen or not isinstance(q,int) or q<=0:raise AssertionError('Repeated/invalid stop')
        seen.add(sid)
        if stop['action']=='pickup':
            if q>limits[sid][0] or lookup[sid]['current']-q<p['target']:raise AssertionError('Unsafe donor')
            load+=q
        else:
            if q>limits[sid][1]:raise AssertionError('Assumed drop bound exceeded')
            load-=q;moved+=q
        if not 0<=load<=p['capacity'] or stop['vehicle_load_after']!=load:raise AssertionError('Capacity/conservation failed')
    if load or moved>p['budget'] or moved!=result['moved_bikes']:raise AssertionError('Return/budget failed')
    if result['route']['return_minutes']>p['effective_deadline']+1e-8:raise AssertionError('Forecast deadline failed')
    return dict(violations=0,bike_conservation=True,intermediate_capacity=True,donor_bounds=True,return_before_forecast=True)


def optimize(state,body,roads,departure):
    began=time.perf_counter();p=parameters(state,body);target=datetime.fromisoformat(state['target_at_utc']).timestamp()
    p['effective_deadline']=max(0.,min(p['deadline_minutes'],(target-departure-30)/60))
    lookup={s['id']:s for s in state['stations']};depot=lookup[p['depot_id']];stations,coverage=screen(state,p)
    seconds,meters,key=roads.table([depot]+stations)
    found=search(stations,p,seconds,meters,state['default_ids']);route=found['route'];chosen=found['chosen']
    ids=[s['id'] for s in stations];net=[0]*len(ids)
    for i,q in zip(chosen[0],chosen[1]):net[i]=q
    all_before=sum(loss(s['samples'],p['target']) for s in state['stations'])
    helped=[stop for stop in route['stops'] if stop['action']=='deliver'];helped_zero=sum(lookup[s['station_id']]['current']==0 for s in helped)
    coverage={k:v for k,v in coverage.items() if k not in ('supply','demand')}
    coverage.update(delivery_sites=len(helped),zero_sites_supplied=helped_zero,unserved_receivers=coverage['receivers']-len(helped),
        unserved_zero=coverage['current_zero']-helped_zero,reason='차량 1대·남은 10분 예측·적재량 안에서 가까운 공급/부족 후보를 선택합니다. 나머지는 후속 작업 대상입니다.')
    result=dict(schema_version=2,status='plan' if chosen[3] else 'no_action',parameters=p,route=route,moved_bikes=chosen[3],net=net,
        candidate_ids=ids,risk_before=all_before,risk_after=all_before-found['gain'],objective_value=all_before-found['net_value'],
        alternatives=[dict(objective=k,moved_bikes=v['moved_bikes'],distance_meters=v['route']['distance_meters'],return_minutes=v['route']['return_minutes']) for k,v in found['controls'].items() if v is not None],
        comparison=found['controls'],proof=dict(complete_search=False,method='deterministic beam64/max6 visits + feasible fallbacks',expanded=found['expanded']),
        coverage=coverage,scope='Automatic shortlist and bounded heuristic; no citywide or field optimum',recorded_at_utc=state['recorded_at_utc'],
        model_label='Flow Matching + GCN',dispatch_allowed=False,source_safety_limits=found['supply'],assumed_drop_limits=found['demand'],road_matrix_key=key,
        source_version=state['source_version'],planning_basis='Calibrated calculation departure; fixed10min forecast expiry; assumed shadow resources',
        departure_at_utc=datetime.fromtimestamp(departure,timezone.utc).isoformat(),valid_until_utc=state['target_at_utc'],
        no_action_reason='예측 유효시간 안에 공급지 보호·적재·복귀 조건을 만족하는 이득 있는 이동이 없습니다.' if not chosen[3] else None,
        geometry=None,geometry_status='not_requested',traffic_live=False,field_effect=None)
    result['certificate']=certificate(state,result)
    if chosen[3]:
        points=[depot]+[lookup[s['station_id']] for s in route['stops']]+[depot]
        try:
            geo,_=roads.query('route',points,'overview=full&geometries=geojson&steps=false');result['geometry']=geo['routes'][0]['geometry'];result['geometry_status']='ok'
        except Exception:result['geometry_status']='unavailable'
    result.update(seconds=time.perf_counter()-began,solver_seconds=found['solver_seconds'])
    return result


def rpc(folder,request):
    folder=Path(folder);status=read(folder/'status.json');state=read(folder/status['latest_state']);body=dict(request.get('body') or {})
    version=body.pop('source_version',None);now=reference_now(status);age=now-datetime.fromisoformat(state['recorded_at_utc']).timestamp()
    if version!=state['source_version']:return dict(status=409,body=dict(error='관측이 갱신됐습니다. 최신 자료로 다시 계산합니다.'))
    if not 0<=age<=720:return dict(status=409,body=dict(error='관측이 오래돼 계획을 중지했습니다.'))
    result=optimize(state,body,Roads(folder/'h19-roads'),now)
    # Never present a route whose computation/road latency consumes its remaining horizon.
    if reference_now(status)+result['route']['return_minutes']*60>datetime.fromisoformat(state['target_at_utc']).timestamp():
        return dict(status=409,body=dict(error='계산 중 예측 유효시간이 부족해졌습니다. 다음 관측으로 다시 계산하세요.'))
    result['observation_age_seconds']=age
    return dict(status=200,body=result)


def watch(folder,out,maximum=24*3600):
    folder=Path(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();last=None
    while time.monotonic()-start<maximum:
        status=read(folder/'status.json')
        if status.get('phase') in ('failed','stopped'):break
        version=status.get('source_version')
        if version and version!=last:
            try:
                response=rpc(folder,dict(body=dict(source_version=version,automatic=True)))
                atomic(out/(version+'.json'),response);atomic(out/'latest.json',response)
                print(json.dumps(dict(source_version=version,status=response['status'],moved=response['body'].get('moved_bikes')),ensure_ascii=False),flush=True)
            except Exception as exc:
                atomic(out/(version+'-error.json'),dict(error=type(exc).__name__));print(type(exc).__name__,flush=True)
            last=version
        time.sleep(5)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['rpc','watch']);ap.add_argument('--folder',required=True);ap.add_argument('--out')
    a=ap.parse_args()
    if a.command=='watch':watch(a.folder,a.out)
    else:
        try:response=rpc(a.folder,json.load(sys.stdin))
        except (ValueError,KeyError,TypeError) as exc:response=dict(status=422,body=dict(error=str(exc)))
        except Exception:response=dict(status=503,body=dict(error='도로 또는 재배치 계산을 완료하지 못했습니다.'))
        print(json.dumps(response,ensure_ascii=False,allow_nan=False))
