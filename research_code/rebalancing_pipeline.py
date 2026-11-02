"""H15 shadow planner: scenario CVaR quantity MILP then exact small route.

No forecasting, API fitting or dispatch happens here. Forecast scenarios are an
input contract; physical demand and treatment effects are not inferred.
"""
import argparse,json,time,math,itertools,re
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy import sparse

def integer(value,name,minimum=0):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or int(value)!=value or value<minimum:raise ValueError('Confirmed integer required: '+name)
    return int(value)

def validate(request):
    if request.get('mode') not in ('synthetic_shadow','replay_shadow','live_shadow'):raise ValueError('Only explicit shadow planning modes supported.')
    f=request['forecast'];ids=f['station_ids'];n=len(ids)
    if not n or len(set(ids))!=n:raise ValueError('Unique nonempty station IDs required.')
    x=np.asarray(f['stock_scenarios'],dtype=float);current=np.asarray(f['current_inventory'],dtype=float)
    if x.ndim!=2 or x.shape[0]!=n or not 2<=x.shape[1]<=256 or current.shape!=(n,):raise ValueError('Forecast schema invalid.')
    if not np.isfinite(x).all() or not np.isfinite(current).all() or (x<0).any() or (current<0).any() or (x!=np.floor(x)).any() or (current!=np.floor(current)).any():raise ValueError('Inventory scenarios must be observed/generated nonnegative integer counts.')
    horizon=integer(f['horizon_minutes'],'horizon',1)
    if horizon not in f['supported_horizons']:raise ValueError('Requested horizon lacks validated model support.')
    if request['mode']=='synthetic_shadow':
        if f.get('source_kind')!='synthetic':raise ValueError('Synthetic source must be labeled.')
    else:
        if f.get('source_kind')!='api_forecast' or not re.fullmatch('[0-9a-f]{64}',str(f.get('model_lock_sha256') or '')):raise ValueError('Frozen forecast lineage fields required from the verified adapter.')
        if f.get('model_status') not in ('frozen_research_reference','historical_gate_passed'):raise ValueError('Unknown forecast validation status.')
        if request['mode']=='live_shadow':
            if not request.get('planner_reference_at_utc'):raise ValueError('Collector reference clock required; do not assume remote OS is synchronized.')
            reference=datetime.fromisoformat(request['planner_reference_at_utc']);observed=datetime.fromisoformat(f['observed_at_utc'])
            if reference.utcoffset() is None or observed.utcoffset() is None:raise ValueError('Timezone-aware collector timestamps required.')
            age=(reference-observed).total_seconds()
            if not 0<=age<=120:raise ValueError('Live inventory is stale or future-dated.')
    stations=request['stations']
    if [s['id'] for s in stations]!=ids:raise ValueError('Station policy and forecast order differ.')
    targets=np.array([integer(s['target_buffer'],'target') for s in stations]);pick=np.array([integer(s['pickup_limit'],'pickup limit') for s in stations]);drop=np.array([integer(s['dropoff_limit'],'dropoff limit') for s in stations]);weights=np.array([float(s['loss_weight']) for s in stations])
    if not np.isfinite(weights).all() or (weights<0).any() or not weights.sum()>0:raise ValueError('Loss weights invalid.')
    fleet=request['fleet'];count=integer(fleet['vehicle_count'],'vehicle count',1)
    if count!=1:raise ValueError('Prototype supports one vehicle; multi-vehicle VRP solver required.')
    capacity=integer(fleet['capacity'],'vehicle capacity',1);initial=integer(fleet['initial_load'],'initial vehicle load')
    if initial>capacity:raise ValueError('Initial load exceeds capacity.')
    limit=float(fleet['max_route_minutes']);policy=request['policy'];budget=integer(policy['workload_budget'],'workload budget')
    if budget>100 or not np.isfinite(limit) or limit<=0:raise ValueError('Prototype budget/route limit invalid.')
    maxstops=integer(policy['max_stops'],'max stops',1)
    if maxstops>8:raise ValueError('More than8stops requires the production routing solver.')
    alpha=float(policy['risk_alpha']);q=float(policy['source_lower_quantile']);risk=float(policy['risk_weight']);travel=float(policy['travel_unit_weight']);handling=float(policy['handling_unit_cost']);service=float(policy['handling_minutes_per_bike'])
    if not all(np.isfinite(v) for v in (alpha,q,risk,travel,handling,service)) or not 0<alpha<1 or not 0<q<1 or min(risk,travel,service)<0 or handling<=0:raise ValueError('Risk/handling policy invalid.')
    road=request['road_matrix']
    if road['ids']!=[fleet['depot']]+ids or not road.get('basis'):raise ValueError('Directed road matrix IDs/provenance required.')
    duration=np.array([[np.inf if v is None else float(v) for v in row] for row in road['duration_minutes']])
    if duration.shape!=(n+1,n+1) or np.isnan(duration).any() or (duration<0).any() or not np.all(np.diag(duration)==0):raise ValueError('Road matrix invalid; unreachable entries use null.')
    lower=np.sort(x,axis=1)[:,max(0,int(math.ceil(q*x.shape[1]))-1)]
    supply=np.minimum.reduce([pick,np.maximum(np.floor(current-targets),0).astype(int),np.maximum(np.floor(lower-targets),0).astype(int),np.floor(x.min(1)).astype(int)])
    demand=np.minimum(drop,np.maximum(np.ceil(targets-lower),0).astype(int))
    return dict(ids=ids,x=x,current=current,targets=targets,weights=weights,supply=supply,demand=demand,capacity=capacity,initial=initial,deadline=min(limit,horizon),budget=budget,maxstops=maxstops,alpha=alpha,risk=risk,travel=travel,handling=handling,service=service,duration=duration,depot=fleet['depot'],horizon=horizon)

def allocation(c,budget):
    n,m=c['x'].shape;sources=np.flatnonzero(c['supply']>0);sinks=np.flatnonzero(c['demand']>0);arcs=[(int(i),int(j)) for i in sources for j in sinks if np.isfinite(c['duration'][i+1,j+1])]
    e=len(arcs)
    if not e or budget==0:return np.zeros(n,dtype=int),[],dict(status='no feasible transfer arc or zero budget',objective=None,gap=None)
    eta=e+n*m;xi=eta+1;nv=xi+m;cost=np.zeros(nv);upper=np.full(nv,np.inf);integral=np.zeros(nv,dtype=int);integral[:e]=1
    for k,(i,j) in enumerate(arcs):cost[k]=c['travel']*c['duration'][i+1,j+1]+c['handling'];upper[k]=min(c['supply'][i],c['demand'][j])
    for i in range(n):cost[e+i*m:e+(i+1)*m]=c['weights'][i]/m
    cost[eta]=c['risk'];cost[xi:]=c['risk']/((1-c['alpha'])*m)
    nr=n*m+m+len(sources)+len(sinks)+1;matrix=sparse.lil_matrix((nr,nv));lo=np.full(nr,-np.inf);hi=np.full(nr,np.inf);row=0
    for i in range(n):
        for s in range(m):
            matrix[row,e+i*m+s]=1
            for k,(origin,dest) in enumerate(arcs):
                if dest==i:matrix[row,k]+=1
                if origin==i:matrix[row,k]-=1
            lo[row]=c['targets'][i]-c['x'][i,s];row+=1
    for s in range(m):
        matrix[row,eta]=1;matrix[row,xi+s]=1
        for i in range(n):matrix[row,e+i*m+s]=-c['weights'][i]
        lo[row]=0;row+=1
    for i in sources:
        for k,(origin,dest) in enumerate(arcs):
            if origin==i:matrix[row,k]=1
        hi[row]=c['supply'][i];row+=1
    for j in sinks:
        for k,(origin,dest) in enumerate(arcs):
            if dest==j:matrix[row,k]=1
        hi[row]=c['demand'][j];row+=1
    matrix[row,:e]=1;hi[row]=budget
    result=milp(cost,integrality=integral,bounds=Bounds(np.zeros(nv),upper),constraints=LinearConstraint(matrix.tocsc(),lo,hi),options=dict(time_limit=60,mip_rel_gap=0.))
    if result.status!=0 or result.x is None:raise RuntimeError('Allocation solver did not certify optimal small quantity problem; preserve and review.')
    raw=result.x[:e]
    if np.max(np.abs(raw-np.rint(raw)))>1e-5:raise ValueError('Noninteger transfer output.')
    quantities=np.rint(raw).astype(int);net=np.zeros(n,dtype=int);moves=[]
    for (i,j),q in zip(arcs,quantities):
        if q:net[i]-=q;net[j]+=q;moves.append((i,j,int(q)))
    if net.sum()!=0 or (-net).clip(0).sum()>budget or ((-net).clip(0)>c['supply']).any() or (net.clip(0)>c['demand']).any():raise ValueError('Quantity conservation/bounds failed.')
    return net,moves,dict(status='optimal quantity MILP only; coupled routing is a heuristic',objective=float(result.fun),gap=float(result.mip_gap))

def route(c,net,moves):
    active=np.flatnonzero(net).tolist()
    if len(active)>c['maxstops']:return None
    best=None
    for order in itertools.permutations(active):
        seen=set();load=c['initial'];clock=0.;last=0;stops=[];valid=True
        for node in order:
            if any(dest==node and origin not in seen for origin,dest,q in moves):valid=False;break
            leg=c['duration'][last,node+1]
            if not np.isfinite(leg):valid=False;break
            arrival=clock+leg;load-=int(net[node])
            if load<0 or load>c['capacity']:valid=False;break
            clock=arrival+abs(int(net[node]))*c['service'];stops.append(dict(station_id=c['ids'][node],action='pickup' if net[node]<0 else 'deliver',bikes=abs(int(net[node])),eta_minutes=float(arrival),completion_minutes=float(clock),vehicle_load_after=load));last=node+1;seen.add(node)
        if not valid:continue
        total=clock+c['duration'][last,0]
        if total>c['deadline']+1e-8 or load!=c['initial']:continue
        key=(float(total),tuple(c['ids'][i] for i in order))
        if best is None or key<best[0]:best=(key,stops)
    if best is None:return None
    return dict(sequence=[c['depot']]+[s['station_id'] for s in best[1]]+[c['depot']],stops=best[1],return_minutes=best[0][0],peak_load=max([c['initial']]+[s['vehicle_load_after'] for s in best[1]]),final_load=c['initial'])

def loss_summary(c,net):
    future=c['x']+net[:,None]
    if (future<0).any():raise ValueError('Plan overdraws a supplied terminal scenario.')
    loss=(np.maximum(c['targets'][:,None]-future,0)*c['weights'][:,None]).sum(0)
    cvar=min(float(eta+np.maximum(loss-eta,0).mean()/(1-c['alpha'])) for eta in np.unique(loss))
    return dict(mean_weighted_buffer_shortfall=float(loss.mean()),cvar_weighted_buffer_shortfall=cvar,scenario_losses=loss.tolist(),scope='terminal exogenous-scenario surrogate, not latent demand or measured relocation benefit')

def plan(request):
    started=time.perf_counter();c=validate(request);attempts=[]
    for budget in range(c['budget'],-1,-1):
        if time.perf_counter()-started>300:raise TimeoutError('Shadow planning budget exhausted without an issued plan.')
        net,moves,solver=allocation(c,budget);path=route(c,net,moves);attempts.append(dict(workload_budget=budget,moved_bikes=int(net.clip(0).sum()),route_feasible=path is not None))
        if path is not None:break
    if path is None:raise ValueError('No feasible empty route; invalid depot/road conditions.')
    actions=[dict(station_id=s,pickup=max(0,-int(q)),deliver=max(0,int(q)),net_station_change=int(q),current_inventory=int(c['current'][i]),target_buffer=int(c['targets'][i]),safe_pickup_limit=int(c['supply'][i]),allowed_dropoff_limit=int(c['demand'][i])) for i,(s,q) in enumerate(zip(c['ids'],net)) if q]
    return dict(hypothesis_id='H-0015',mode=request['mode'],forecast_id=request['forecast']['id'],status='shadow_plan' if moves else 'no_action',dispatch_allowed=False,operating_assumptions_confirmed=False if request['mode']=='synthetic_shadow' else 'caller-provided shadow inputs; field verification required',quantity_actions=actions,transfers=[dict(origin=c['ids'][i],destination=c['ids'][j],bikes=q) for i,j,q in moves],total_moved_bikes=int(net.clip(0).sum()),route=path,deadline_minutes=c['deadline'],forecast_horizon_minutes=c['horizon'],objective_before=loss_summary(c,np.zeros(len(net),dtype=int)),objective_after=loss_summary(c,net),quantity_solver=solver,attempts=attempts,road_matrix_basis=request['road_matrix']['basis'],seconds=time.perf_counter()-started,
        limitations=['One vehicle, one visit per station and at most8stops; no globally optimal combined quantity/route claim.','Endpoint stock plus planned moves assumes exogenous completed flows; not an action-dependent physical simulator.','Confirm source bikes and destination allowed space at each visit; replan on changes.','Synthetic/road-profile assumptions do not establish actual truck legality or field savings.'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--out',required=True);a=p.parse_args();value=plan(json.loads(Path(a.request).read_text(encoding='utf-8')));Path(a.out).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=value['status'],moved_bikes=value['total_moved_bikes'],route_minutes=value['route']['return_minutes'],dispatch_allowed=False)))
