"""H28 separate setup/batch candidate. All planning runs on DISLAB.

No model fitting, forecast extrapolation, inventory requests or field dispatch.
All station work finishes before the fixed forecast cutoff; vehicle return has
a separate deadline. Each nonzero station visit handles at least minimum_transfer.
The beam solver is heuristic, and no action is a valid outcome.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import numpy as np
import bulk_core as base
from bulk_core import bounds, loss, screen, empty_route, MODES
from dispatch_engine import integer, Roads, read, write


def parameters(state,body):
    body=dict(body)
    minimum=integer(body.pop('minimum_transfer',3),1,10,'minimum_transfer')
    setup=integer(body.pop('stop_setup_seconds',60),0,600,'stop_setup_seconds')
    deadline=integer(body.pop('deadline_minutes',30),1,120,'deadline_minutes')
    p=base.parameters(state,dict(body,deadline_minutes=10))
    p.update(minimum_transfer=minimum,stop_setup_seconds=setup,deadline_minutes=deadline)
    return p


def route_from(order,quantities,stations,seconds,meters,handling_seconds=20, stop_setup_seconds=60):
    route=empty_route();last=0;clock=0.;distance=0.;load=0
    for i,q in zip(order,quantities):
        leg=float(seconds[last,i+1])/60;leg_m=float(meters[last,i+1]);arrival=clock+leg;clock=arrival+stop_setup_seconds/60+abs(q)*handling_seconds/60;distance+=leg_m;load-=q
        route['stops'].append(dict(station_id=stations[i]['id'],action='deliver' if q>0 else 'pickup',bikes=abs(q),
            leg_meters=leg_m,leg_drive_minutes=leg,eta_minutes=arrival,completion_minutes=clock,vehicle_load_after=load))
        last=i+1
    back=float(seconds[last,0])/60;back_m=float(meters[last,0])
    route.update(return_minutes=clock+back,return_leg_minutes=back,return_leg_meters=back_m,distance_meters=distance+back_m,
        peak_load=max([0]+[s['vehicle_load_after'] for s in route['stops']]))
    return route

def search(stations,p,seconds,meters,legacy_ids):
    started=time.perf_counter();handling_seconds=p['handling_seconds'];n=len(stations);supply,demand=bounds(stations,p['target'],p['capacity'],p['drop_limit']);coeff=MODES[p['objective']]
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
        clock=st[4]+seconds[last,i+1]/60+p['stop_setup_seconds']/60+abs(q)*handling_seconds/60;distance=st[5]+meters[last,i+1]
        load=st[2]-q;delivered=st[3]+max(0,q)
        if abs(q)<p['minimum_transfer'] or clock>p['action_deadline_minutes']+1e-9 or not 0<=load<=p['capacity'] or delivered+load>p['budget'] or clock+seconds[i+1,0]/60>p['effective_deadline']+1e-9 or not np.isfinite(distance+clock):return None
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
    # Deterministic beam search; retain best feasible closed route at every depth.
    beam=[origin]
    max_gain=max([0.]+[gains[i].get(1,0) for i in range(n)])
    for _depth in range(p['max_visits']):
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
    return dict(chosen=chosen,route=route_from(chosen[0],chosen[1],stations,seconds,meters,handling_seconds,p['stop_setup_seconds']),gain=chosen[6],net_value=value(chosen),
        controls={name:None if st is None else dict(net_value=value(st),gain=st[6],moved_bikes=st[3],route=route_from(st[0],st[1],stations,seconds,meters,handling_seconds,p['stop_setup_seconds'])) for name,st in controls.items()},
        supply=supply.tolist(),demand=demand.tolist(),expanded=expanded,solver_seconds=time.perf_counter()-started)


def certificate(state,result):
    """Recompute every constraint independently of the search's elapsed clock."""
    p=result['parameters'];lookup={s['id']:s for s in state['stations']}
    supply,demand=bounds(state['stations'],p['target'],p['capacity'],p['drop_limit'])
    limits={s['id']:(int(a),int(b)) for s,a,b in zip(state['stations'],supply,demand)}
    seen=set();load=0;moved=0;clock=0.
    for stop in result['route']['stops']:
        sid=stop['station_id'];q=stop['bikes']
        if sid in seen or not isinstance(q,int) or q<p['minimum_transfer']:raise AssertionError('Invalid/minimum batch stop')
        seen.add(sid)
        clock+=stop['leg_drive_minutes']
        if abs(clock-stop['eta_minutes'])>1e-7:raise AssertionError('Arrival clock mismatch')
        clock+=p['stop_setup_seconds']/60+q*p['handling_seconds']/60
        if abs(clock-stop['completion_minutes'])>1e-7:raise AssertionError('Stop setup/handling missing')
        if clock>p['action_deadline_minutes']+1e-8:raise AssertionError('Station work after forecast cutoff')
        if stop['action']=='pickup':
            if q>limits[sid][0] or lookup[sid]['current']-q<p['target']:raise AssertionError('Unsafe donor')
            load+=q
        elif stop['action']=='deliver':
            if q>limits[sid][1]:raise AssertionError('Assumed parking bound exceeded')
            load-=q;moved+=q
        else:raise AssertionError('Invalid action')
        if not 0<=load<=p['capacity'] or load!=stop['vehicle_load_after']:raise AssertionError('Load conservation failed')
    if len(seen)>p['max_visits'] or load or moved>p['budget'] or moved!=result['moved_bikes']:raise AssertionError('Return/budget/visits failed')
    clock+=result['route']['return_leg_minutes']
    if abs(clock-result['route']['return_minutes'])>1e-7:raise AssertionError('Return time mismatch')
    if clock>p['deadline_minutes']+1e-8:raise AssertionError('Job return deadline failed')
    return dict(violations=0,minimum_batch=True,setup_accounted=True,handling_accounted=True,
                bike_conservation=True,intermediate_capacity=True,donor_bounds=True,
                all_station_work_before_forecast=True,return_before_job_deadline=True)


def operations(state,result):
    p=result['parameters'];rows=state['stations'];supply,demand=bounds(rows,p['target'],p['capacity'],p['drop_limit'])
    stops=result['route']['stops'];moved=result['moved_bikes'];net=dict(zip(result['candidate_ids'],result['net']))
    shortage_before=sum(float(np.maximum(p['target']-np.asarray(s['samples']),0).mean()) for s in rows)
    shortage_after=sum(float(np.maximum(p['target']-np.asarray(s['samples'])-net.get(s['id'],0),0).mean()) for s in rows)
    small_receivers=[s['id'] for s,q in zip(rows,demand) if 0<q<p['minimum_transfer']]
    small_donors=[s['id'] for s,q in zip(rows,supply) if 0<q<p['minimum_transfer']]
    return dict(stops=len(stops),setup_minutes=len(stops)*p['stop_setup_seconds']/60,
                handling_minutes=2*moved*p['handling_seconds']/60,
                work_minutes_per_delivered_bike=result['route']['return_minutes']/moved if moved else None,
                handled_bikes_per_stop=2*moved/len(stops) if stops else None,
                expected_target_shortage_before=shortage_before,expected_target_shortage_after=shortage_after,
                below_minimum_receiver_count=len(small_receivers),below_minimum_donor_count=len(small_donors),
                excluded_small_receiver_ids=small_receivers,excluded_small_donor_ids=small_donors,
                note='Model-based target shortage, not measured lost customers. Setup and minimum batch are operating assumptions.')


def optimize(state,body,roads,departure):
    began=time.perf_counter();p=parameters(state,body);target=datetime.fromisoformat(state['target_at_utc']).timestamp()
    p['effective_deadline']=float(p['deadline_minutes'])
    p['action_deadline_minutes']=max(0.,(target-departure-30)/60)
    lookup={s['id']:s for s in state['stations']};depot=lookup[p['depot_id']];stations,coverage=screen(state,p)
    seconds,meters,key=roads.table([depot]+stations)
    found=search(stations,p,seconds,meters,state['default_ids']);route=found['route'];chosen=found['chosen']
    ids=[s['id'] for s in stations];net=[0]*len(ids)
    for i,q in zip(chosen[0],chosen[1]):net[i]=q
    all_before=sum(loss(s['samples'],p['target']) for s in state['stations'])
    helped=[stop for stop in route['stops'] if stop['action']=='deliver'];helped_zero=sum(lookup[s['station_id']]['current']==0 for s in helped)
    coverage={k:v for k,v in coverage.items() if k not in ('supply','demand')}
    coverage.update(delivery_sites=len(helped),zero_sites_supplied=helped_zero,unserved_receivers=coverage['receivers']-len(helped),
        unserved_zero=coverage['current_zero']-helped_zero,reason='정차 준비와 묶음 수량을 반영합니다. 수거·배치는 예측 적용시각 안에 마치고 차량 복귀는 별도 마감으로 계산합니다.')
    result=dict(schema_version=2,status='plan' if chosen[3] else 'no_action',parameters=p,route=route,moved_bikes=chosen[3],net=net,
        candidate_ids=ids,risk_before=all_before,risk_after=all_before-found['gain'],objective_value=all_before-found['net_value'],
        alternatives=[dict(objective=k,moved_bikes=v['moved_bikes'],distance_meters=v['route']['distance_meters'],return_minutes=v['route']['return_minutes']) for k,v in found['controls'].items() if v is not None],
        comparison=found['controls'],proof=dict(complete_search=False,method='setup-and-batch-aware beam64 + feasible2stop fallback; heuristic',expanded=found['expanded']),
        coverage=coverage,scope='Automatic shortlist and bounded heuristic; no citywide or field optimum',recorded_at_utc=state['recorded_at_utc'],
        model_label='Flow Matching + GCN',dispatch_allowed=False,source_safety_limits=found['supply'],assumed_drop_limits=found['demand'],road_matrix_key=key,
        source_version=state['source_version'],planning_basis='Fixed forecast action cutoff; separate vehicle return deadline; assumed setup and batch resources',
        departure_at_utc=datetime.fromtimestamp(departure,timezone.utc).isoformat(),valid_until_utc=state['target_at_utc'],
        no_action_reason='최소 묶음 수량과 정차시간을 반영하면 예측 적용시각 안에 마칠 수 있는 이동이 없습니다. 소량 후보는 다음 작업에서 확인하세요.' if not chosen[3] else None,
        geometry=None,geometry_status='not_requested',traffic_live=False,field_effect=None)
    result['certificate']=certificate(state,result)
    result['operations']=operations(state,result)
    if chosen[3]:
        points=[depot]+[lookup[s['station_id']] for s in route['stops']]+[depot]
        try:
            geo,_=roads.query('route',points,'overview=full&geometries=geojson&steps=false');result['geometry']=geo['routes'][0]['geometry'];result['geometry_status']='ok'
        except Exception:result['geometry_status']='unavailable'
    result.update(seconds=time.perf_counter()-began,solver_seconds=found['solver_seconds'])
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--state',required=True);ap.add_argument('--roads',required=True);ap.add_argument('--out',required=True);ap.add_argument('--departure-offset',type=float,default=15);ap.add_argument('--body',default='{}');a=ap.parse_args()
    state=read(a.state);departure=datetime.fromisoformat(state['recorded_at_utc']).timestamp()+a.departure_offset
    result=optimize(state,json.loads(a.body),Roads(a.roads),departure);write(a.out,result)
    print(json.dumps(dict(status=result['status'],moved_bikes=result['moved_bikes'],operations=result['operations'],certificate=result['certificate']),ensure_ascii=False))
