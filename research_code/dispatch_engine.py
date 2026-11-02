"""H17 bounded exact dispatch planning. Research computation runs on dislab.

No fitting, future labels, real dispatch or unqualified citywide optimality.
Integer quantities and all single-visit routes are searched together, <=5 sites.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
MODES=('balanced','distance','time')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(bundle,forecast,out):
    expected={r['id']:r['sha256'] for r in read(ROOT/'research/datasets/H-0017-inputs.json')['inputs']}
    if sha(bundle)!=expected['h16_bundle'] or sha(forecast)!=expected['h13_forecast']:
        raise ValueError('Frozen input digest mismatch.')
    b=read(bundle)
    with np.load(forecast,allow_pickle=False) as p:
        ids=p['station_ids'].astype(str).tolist();samples=p['recent_cfm'];lookup={s:i for i,s in enumerate(ids)}
        stations=[]
        for row in b['stations']:
            if row['lat'] is None:continue
            i=lookup[row['id']]
            # Explicit allowlist: future truth, scores and research sections never enter state.
            stations.append(dict(id=row['id'],name=row['name'],lat=row['lat'],lon=row['lon'],current=row['current'],
                                 forecast=row['forecast']['recent_cfm'],samples=samples[i].tolist()))
    state=dict(schema_version=1,model_label='Flow Matching + GCN',recorded_at_utc=b['episode']['scheduled_at_utc'],
        target_at_utc=b['episode']['target_at_utc'],model_lock_sha256=b['provenance']['model_lock_sha256'],
        source_sha256=expected,stations=stations,default_ids=[s['id'] for s in b['shadow']['selected_stations']],
        default_depot_id=b['shadow']['selected_stations'][0]['id'],dispatch_allowed=False,mode='recorded_planning')
    assert not any(k in state for k in ('truth','live_metrics','september','joint_experiment'))
    write(out,state)


def information(state):
    return dict(schema_version=1,model_label=state['model_label'],recorded_at_utc=state['recorded_at_utc'],
        target_at_utc=state['target_at_utc'],default_ids=state['default_ids'],default_depot_id=state['default_depot_id'],
        stations=[{k:s[k] for k in ('id','name','lat','lon','current','forecast')} for s in state['stations']],
        limits=dict(candidates=5,capacity=12,target=5,deadline_minutes=10,budget=12),dispatch_allowed=False)


def integer(value,lo,hi,name):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or int(value)!=value or not lo<=value<=hi:
        raise ValueError('Invalid '+name)
    return int(value)


def parameters(state,body):
    allowed={'station_ids','depot_id','capacity','target','deadline_minutes','budget','objective'}
    if not isinstance(body,dict) or set(body)-allowed:raise ValueError('Unknown planning parameter.')
    ids=body.get('station_ids',state['default_ids'])
    if not isinstance(ids,list) or not 1<=len(ids)<=5 or len(set(ids))!=len(ids):raise ValueError('Select1–5unique candidates.')
    known={s['id']:s for s in state['stations']}
    if any(s not in known for s in ids):raise ValueError('Unknown station.')
    depot=body.get('depot_id',state['default_depot_id'])
    if depot not in ids:raise ValueError('Depot must be a selected station.')
    objective=body.get('objective','balanced')
    if objective not in MODES:raise ValueError('Unknown objective.')
    return dict(station_ids=sorted(ids),depot_id=depot,capacity=integer(body.get('capacity',6),1,12,'capacity'),
        target=integer(body.get('target',2),1,5,'target'),deadline_minutes=integer(body.get('deadline_minutes',10),1,10,'deadline'),
        budget=integer(body.get('budget',6),1,12,'budget'),objective=objective)


class Roads:
    def __init__(self,cache):
        self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=True)

    def query(self,service,locations,options):
        coords=';'.join(f"{s['lon']:.6f},{s['lat']:.6f}" for s in locations)
        url=f'https://router.project-osrm.org/{service}/v1/driving/{coords}?{options}'
        key=hashlib.sha256(url.encode()).hexdigest();path=self.cache/(key+'.json')
        if path.is_file():return read(path),key
        req=Request(url,headers={'User-Agent':'TashuOperator-H17/1.0 (+https://github.com/sijoon-sung/KT-X-AI-)','Accept':'application/json'})
        with urlopen(req,timeout=20) as response:data=json.loads(response.read(4_000_000))
        if data.get('code')!='Ok':raise ValueError('Road service unavailable; no fabricated distances.')
        write(path,data)
        return data,key

    def table(self,locations):
        data,key=self.query('table',locations,'annotations=duration,distance')
        n=len(locations)
        seconds=np.array([[np.inf if v is None else v for v in row] for row in data['durations']],dtype=float)
        meters=np.array([[np.inf if v is None else v for v in row] for row in data['distances']],dtype=float)
        if seconds.shape!=(n,n) or meters.shape!=(n,n) or np.isnan(seconds).any() or np.isnan(meters).any() or (seconds<0).any() or (meters<0).any():
            raise ValueError('Invalid directed road matrix.')
        return seconds,meters,key


def risk(samples,target,net):
    loss=np.maximum(target-(samples+net[:,None]),0).sum(axis=0)
    eta=np.sort(loss)[int(math.ceil(.9*len(loss)))-1]
    return float(loss.mean()+eta+np.maximum(loss-eta,0).mean()/.1)


def evaluate_routes(net,seconds,meters,capacity,deadline,ids):
    active=np.flatnonzero(net).tolist()
    for order in itertools.permutations(active):
        load=0;clock=0.;distance=0.;last=0;stops=[]
        for i in order:
            leg=seconds[last,i+1];leg_meters=meters[last,i+1]
            if not np.isfinite(leg+leg_meters):break
            load-=int(net[i])
            if not 0<=load<=capacity:break
            arrival=clock+leg/60;clock=arrival+abs(int(net[i]))*.1;distance+=leg_meters
            stops.append(dict(station_id=ids[i],action='pickup' if net[i]<0 else 'deliver',bikes=abs(int(net[i])),
                leg_meters=float(leg_meters),leg_drive_minutes=float(leg/60),eta_minutes=float(arrival),
                completion_minutes=float(clock),vehicle_load_after=load))
            last=i+1
        else:
            if load!=0:continue
            returning=seconds[last,0]/60;return_meters=meters[last,0]
            if not np.isfinite(returning+return_meters) or clock+returning>deadline+1e-8:continue
            yield dict(stops=stops,return_minutes=float(clock+returning),return_leg_minutes=float(returning),
                return_leg_meters=float(return_meters),distance_meters=float(distance+return_meters),
                peak_load=max([0]+[s['vehicle_load_after'] for s in stops]))


def exhaustive(stations,p,seconds,meters):
    started=time.perf_counter();ids=[s['id'] for s in stations];x=np.asarray([s['samples'] for s in stations],float)
    if x.ndim!=2 or x.shape[1]!=128 or not np.isfinite(x).all() or (x<0).any():raise ValueError('Frozen forecast samples invalid.')
    lower=np.sort(x,axis=1)[:,12];current=np.array([s['current'] for s in stations]);target=p['target']
    supply=np.maximum(0,np.minimum.reduce([np.floor(current-target),np.floor(lower-target),x.min(1),np.full(len(ids),6)])).astype(int)
    demand=np.minimum(3,np.maximum(0,np.ceil(target-lower))).astype(int)
    options=[range(-int(supply[i]),1) if supply[i]>0 else range(0,int(demand[i])+1) for i in range(len(ids))]
    zero=np.zeros(len(ids),dtype=int);before=risk(x,target,zero)
    empty=dict(stops=[],return_minutes=0.,return_leg_minutes=0.,return_leg_meters=0.,distance_meters=0.,peak_load=0)
    keys={m:(before,before,0.,0.,(),tuple(zero)) for m in MODES}
    best={m:dict(route=empty,net=zero.tolist(),risk_after=before,moved=0,objective_value=before) for m in MODES}
    vectors=0;routes=0
    for values in itertools.product(*options):
        if time.perf_counter()-started>15:raise TimeoutError('Complete search time limit; partial result not issued.')
        if sum(values)!=0:continue
        net=np.array(values);moved=int(net.clip(0).sum())
        if not 0<moved<=p['budget']:continue
        vectors+=1;after=risk(x,target,net)
        if after>=before:continue
        for route in evaluate_routes(net,seconds,meters,p['capacity'],p['deadline_minutes'],ids):
            routes+=1;km=route['distance_meters']/1000;minutes=route['return_minutes']
            scores={'balanced':after+.03*km+.01*moved,'distance':after+.5*km+.01*moved,'time':after+.07*minutes+.01*moved}
            order=tuple(s['station_id'] for s in route['stops'])
            for mode,score in scores.items():
                key=(score,after,route['distance_meters'],minutes,order,values)
                if key<keys[mode]:
                    keys[mode]=key;best[mode]=dict(route=route,net=net.tolist(),risk_after=after,moved=moved,objective_value=float(score))
    return dict(best=best,risk_before=before,supply=supply.tolist(),demand=demand.tolist(),quantity_vectors=vectors,
                feasible_routes=routes,complete_search=True,seconds=time.perf_counter()-started)


def optimize(state,body,roads):
    started=time.perf_counter();p=parameters(state,body);lookup={s['id']:s for s in state['stations']}
    stations=[lookup[s] for s in p['station_ids']];depot=lookup[p['depot_id']]
    seconds,meters,road_key=roads.table([depot]+stations)
    search=exhaustive(stations,p,seconds,meters);chosen=search['best'][p['objective']]
    result=dict(schema_version=1,status='plan' if chosen['moved'] else 'no_action',parameters=p,
        route=chosen['route'],moved_bikes=chosen['moved'],net=chosen['net'],objective_value=chosen['objective_value'],
        candidate_ids=p['station_ids'],risk_before=search['risk_before'],risk_after=chosen['risk_after'],
        alternatives=[dict(objective=k,moved_bikes=v['moved'],distance_meters=v['route']['distance_meters'],
                           return_minutes=v['route']['return_minutes']) for k,v in search['best'].items()],
        proof={k:search[k] for k in ('quantity_vectors','feasible_routes','complete_search')},
        scope='Selected candidates and supplied single-vehicle constraints only; no citywide or field optimum.',
        recorded_at_utc=state['recorded_at_utc'],model_label='Flow Matching + GCN',dispatch_allowed=False,
        source_safety_limits=search['supply'],assumed_drop_limits=search['demand'],road_matrix_key=road_key,
        geometry=None,geometry_status='not_requested',traffic_live=False,field_effect=None)
    if chosen['moved']:
        points=[depot]+[lookup[s['station_id']] for s in chosen['route']['stops']]+[depot]
        try:
            geo,_=roads.query('route',points,'overview=full&geometries=geojson&steps=false')
            result['geometry']=geo['routes'][0]['geometry'];result['geometry_status']='ok'
        except Exception:
            result['geometry_status']='unavailable'
    result['seconds']=time.perf_counter()-started
    return result


def rpc(state,cache,request):
    if request.get('path')=='/api/dispatch/info':return dict(status=200,body=information(state))
    if request.get('path')=='/api/dispatch/optimize':return dict(status=200,body=optimize(state,request.get('body'),Roads(cache)))
    return dict(status=404,body=dict(error='Unknown route.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    q=sp.add_parser('prepare');q.add_argument('--bundle',required=True);q.add_argument('--forecast',required=True);q.add_argument('--out',required=True)
    q=sp.add_parser('rpc');q.add_argument('--state',required=True);q.add_argument('--cache',required=True)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.bundle,a.forecast,a.out)
    else:
        import sys
        try:value=rpc(read(a.state),a.cache,json.load(sys.stdin))
        except (ValueError,TypeError,KeyError,TimeoutError) as exc:value=dict(status=422,body=dict(error=str(exc)))
        except Exception:value=dict(status=503,body=dict(error='Road or planning service unavailable. Previous plan is not current.'))
        print(json.dumps(value,ensure_ascii=False,allow_nan=False))
