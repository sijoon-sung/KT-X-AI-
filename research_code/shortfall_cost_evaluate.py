"""H25 fixed-model September shortfall evaluation, dislab only.

Targets are inventory gaps to assumed floors, NOT latent denied rentals.
No fitting, calibration, threshold selection, dispatch or API collection.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
import rolling_inventory_model as old
import large_inventory_model as large
import rolling_inventory_predict as live
import count_flow_models as compact
from forecast_transfer import now, weather_for
from rolling_inventory_metrics import scores

ROOT=Path(__file__).resolve().parents[1]
MODELS=('persistence','h13_poisson','h13_fm','h22_poisson','h22_fm','h24_poisson','h24_fm')
TARGETS=(1,3,6)
COSTS=(1,2,5,10)
SEED=20261125
DAY0=(old.h.stamp('2026-09-17')+32400)//86400

def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def emit(**value):
    print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

def cost_vectors(samples,truth,cu):
    """Generalized inverse empirical CDF, including integer ties."""
    x=np.sort(np.asarray(samples,dtype=np.float64),axis=1)
    q=x[:,int(np.ceil(x.shape[1]*cu/(cu+1)))-1]
    under=np.maximum(0,truth-q);over=np.maximum(0,q-truth)
    return {'cost':cu*under+over,'under_bikes':under,'over_bikes':over,'gap_quantile':q}

def kernel_checks():
    x=np.array([[0,1,2,5],[0,0,0,0]],dtype=float);truth=np.array([2,1.])
    a=cost_vectors(x,truth,1);b=cost_vectors(x,truth,5)
    assert np.array_equal(a['gap_quantile'],[1,0])
    assert np.array_equal(a['cost'],[1,1])
    assert np.array_equal(b['gap_quantile'],[5,0])
    assert np.array_equal(b['cost'],[3,5])
    assert np.array_equal(np.maximum(0,3-np.array([0,2,3,8])),[3,1,0,0])
    assert np.all(cost_vectors(np.ones((2,128))*3,np.array([3,3]),5)['cost']==0)
    z=scores(np.zeros((2,128)),np.zeros(2))
    assert np.all(z['fair_crps']==0)
    return {'gap_transform':True,'integer_quantile_ties':True,'asymmetric_hand_costs':True,'deterministic_perfect_forecast':True,'zero_gap_fair_crps':True}

class Predictor:
    def __init__(self,a,manifest):
        for name,folder,file in [('h13',a.h13,'model-lock.json'),('h22',a.h22,'model-lock.json'),('h24',a.h24,'pre-test-lock.json')]:
            if large.digest(Path(folder)/file)!=manifest['model_locks'][name]:
                raise ValueError('Frozen model lock changed: '+name)
        self.parent=old.Predictor(a.h13,'cuda');self.big=large.Predictor(a.h22,'cuda')
        self.c=self.parent.c;self.ids=self.parent.ids;self.graph=self.parent.graph;self.first=self.parent.first;self.n=len(self.ids)
        if not np.array_equal(self.ids,self.big.ids):raise ValueError('Station ordering differs')
        if large.digest(Path(a.h13)/'context.npz')!=large.digest(Path(a.h22)/'context.npz'):raise ValueError('Historical static context differs')
        self.lock=json.loads((Path(a.h24)/'pre-test-lock.json').read_text(encoding='utf-8'))
        if self.lock['selected_encoder']!='compact_local' or self.lock['selected_fm']!='poisson_continuous_score':raise ValueError('H24 choice changed')
        for item in self.lock['artifacts']:
            if large.digest(Path(a.h24)/item['id'])!=item['sha256']:raise ValueError('H24 checkpoint changed')
        self.models=[];self.flows=[];self.epochs=self.lock['flow_choices']['poisson_continuous_score']['epochs']
        for seed in (11,29):
            m=compact.CompactEncoder('compact_local').cuda()
            m.load_state_dict(torch.load(Path(a.h24)/f'encoder-compact_local-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);m.eval();self.models.append(m)
            f=old.CountFlow(context_width=128,width=128).cuda()
            f.load_state_dict(torch.load(Path(a.h24)/f'flow-poisson_continuous-score-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);f.eval();self.flows.append(f)

    @torch.inference_mode()
    def issue(self,scheduled,current,valid,proxy,weather):
        nodes,base=self.parent.issue(scheduled,current,valid,proxy,weather)
        bn,big=self.big.issue(scheduled,current,valid,proxy,weather)
        if not np.array_equal(nodes,bn):raise ValueError('Forecast cohorts differ')
        out={'h13_poisson':base['recent_poisson'],'h13_fm':base['recent_cfm'],
             'h22_poisson':big['recent_poisson'],'h22_fm':big['recent_cfm']}
        parts={name:[] for name in ('h24_poisson','h24_fm')}
        seed=SEED+scheduled//600
        streams={k:(np.random.default_rng(seed),torch.Generator(device='cuda').manual_seed(seed)) for k in parts}
        for begin in range(0,len(nodes),256):
            ns=nodes[begin:begin+256];r=torch.zeros(len(ns),dtype=torch.int64,device='cuda');n=torch.as_tensor(ns,device='cuda')
            pairs=[m.rate(self.c,r,n,hidden=True) for m in self.models]
            for name in parts:
                rng,g=streams[name];choice=rng.integers(0,2,(len(ns),128));values=np.empty((len(ns),128,2),np.int32)
                for i,(p,h) in enumerate(pairs):
                    rr,ss=np.where(choice==i);ix=torch.as_tensor(rr,device='cuda')
                    f=None if name=='h24_poisson' else self.flows[i]
                    kind='poisson' if f is None else 'poisson_continuous'
                    v,_=compact.sample_flow(f,kind,0 if f is None else self.epochs[i],p[ix],h[ix],g,steps=16)
                    values[rr,ss]=v
                parts[name].append(old.stock_projection(current[ns],values))
        for name,v in parts.items():out[name]=np.concatenate(v) if v else np.empty((0,128),dtype=np.int64)
        return nodes,out

class Aggregator:
    def __init__(self):
        self.entries={};self.daily={k:np.zeros((14,3),float) for k in MODELS}

    def add(self,draws,y,current,day):
        for target in TARGETS:
            truth=np.maximum(0,target-y)
            masks={'all':np.ones(len(y),bool),'current_zero':current==0,'current_positive':current>0,
                   'current1_2':(current>0)&(current<3),'current3plus':current>=3,
                   'future_gap0':truth==0,'future_gap1':truth==1,'future_gap2':truth==2,'future_gap3plus':truth>=3}
            for model,x in draws.items():
                gap=np.maximum(0,target-x);s=scores(gap,truth)
                common={'actual_gap':truth,'predicted_mean_gap':gap.mean(1),'gap_mean_abs':np.abs(gap.mean(1)-truth),
                        'gap_median_abs':s['median_abs'],'gap_fair_crps':s['fair_crps'],'gap_mean_bias':gap.mean(1)-truth}
                for cu in COSTS:
                    loss=cost_vectors(gap,truth,cu)
                    for group,mask in masks.items():
                        key=f'target{target}/cost{cu}/{group}/{model}'
                        entry=self.entries.setdefault(key,{'n':0,'sums':{}});entry['n']+=int(mask.sum())
                        for name,value in {**common,**loss}.items():entry['sums'][name]=entry['sums'].get(name,0.)+float(value[mask].sum())
                    if target==3 and cu==5:
                        mask=current>0
                        self.daily[model][day]+=[loss['cost'][mask].sum(),loss['under_bikes'][mask].sum(),mask.sum()]

    def result(self):
        return {key:{'n':v['n'],**{k:(x/v['n'] if v['n'] else None) for k,x in v['sums'].items()}} for key,v in self.entries.items()}

def paired(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if b[:,0].sum()<=0:raise ValueError('Zero control cost')
    point=float(1-a[:,0].sum()/b[:,0].sum());intervals=[]
    for block in (3,7):
        rng=np.random.default_rng(SEED);starts=rng.integers(0,14-block+1,(20000,(14+block-1)//block))
        ix=(starts[...,None]+np.arange(block)).reshape(20000,-1)[:,:14]
        den=b[ix,0].sum(1);valid=den>0;g=1-a[ix,0].sum(1)[valid]/den[valid]
        intervals.append({'block_days':block,'replicates':int(valid.sum()),'relative_reduction_95':np.percentile(g,[2.5,97.5]).tolist()})
    return {'relative_reduction':point,'intervals':intervals}

def evaluate(a):
    start=now();timer=time.perf_counter();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    if (out/'start.json').exists():raise ValueError('Preserve started attempts')
    record=json.loads((ROOT/'research/hypotheses/H-0025.json').read_text(encoding='utf-8'))
    canon=json.dumps(record['plan'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(canon).hexdigest()!=record['preregistration']['plan_sha256']:raise ValueError('Plan changed')
    mp=ROOT/'research/datasets/H25-frozen-shortfall-v1.json'
    if large.digest(mp)!=record['plan']['data']['input_manifest_sha256']:raise ValueError('Manifest changed')
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old.configure();torch.cuda.set_per_process_memory_fraction(.55)
    free,total=torch.cuda.mem_get_info()
    if free<3*1024**3:raise ValueError('Insufficient free GPU')
    env={'execution':'dislab','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
         'gpu':torch.cuda.get_device_name(),'free_gpu_bytes':free,'total_gpu_bytes':total,'threads':4,'deterministic':True,'tf32':False}
    write(out/'start.json',{'started_at_utc':start,'code_commit':code,'plan_sha256':record['preregistration']['plan_sha256'],'environment':env})
    checks=kernel_checks();write(out/'kernel-checks.json',checks)
    manifest=json.loads(mp.read_text(encoding='utf-8'))
    for item in manifest['sources']:
        path=Path(a.panel) if item['id']=='observed_panel.npz' else Path(a.weather)/item['id']
        if large.digest(path)!=item['sha256']:raise ValueError('Source hash changed: '+item['id'])
    predictor=Predictor(a,manifest)
    write(out/'models-locked.json',{'locked_at_utc':now(),'model_lock_hashes':manifest['model_locks'],'api_opened':False,
                                  'costs':COSTS,'targets':TARGETS,'primary':'target3/cost5/current_positive/h24_fm vs h24_poisson'})
    # Open the reused evaluation source only after all weights, metrics and costs are fixed.
    with np.load(a.panel,allow_pickle=False) as f:
        times=f['scheduled_at_unix_seconds'].copy();ids=f['station_ids'].astype(str);stock=f['stock'].copy();actual=f['observed_at_unix_seconds'].copy()
    if len(set(ids.tolist()))!=len(ids) or not np.all(np.diff(times)==600):raise ValueError('Corrupt grid or IDs')
    lookup={s:i for i,s in enumerate(ids)};nodes=np.array([i for i,s in enumerate(predictor.ids) if s in lookup]);ix=np.array([lookup[predictor.ids[i]] for i in nodes])
    first=old.h.stamp('2026-09-01');end=old.h.stamp('2026-10-01');selected=np.flatnonzero((times>=first)&(times<end))
    wx=weather_for(Path(a.weather),'daejeon',times[selected]);acc=Aggregator();previous=None;pending=None
    stations=set();valid_pairs=0;positive_pairs=0;issued_cells=0;issues=0;scored_origins=0;input_proxy_present=0
    with (out/'replay-ledger.private.jsonl').open('w',encoding='utf-8') as ledger:
        for j,index in enumerate(selected):
            if time.perf_counter()-timer>2700:raise TimeoutError('Frozen wall-time budget')
            values=np.full(predictor.n,np.nan);stamps=np.full(predictor.n,-1.);values[nodes]=stock[index,ix];stamps[nodes]=actual[index,ix]
            current=live.Snapshot(int(times[index]),values,stamps,len(ids),len(nodes))
            if pending is not None:
                valid,y,strict=live.score_issue(pending,current)
                if valid.any():
                    d=int((pending['scheduled']+32400)//86400-DAY0)
                    if not 0<=d<14:raise ValueError('Scored outside September source scope')
                    draws={k:v[valid] for k,v in pending['draws'].items()}
                    acc.add(draws,y,pending['current'][valid],d);valid_pairs+=len(y);positive_pairs+=int((pending['current'][valid]>0).sum())
                    stations.update(pending['nodes'][valid].tolist());scored_origins+=1
                ledger.write(json.dumps({'event':'score','target':current.scheduled,'pairs':int(valid.sum()),'issued_hash':pending['hash']})+'\n');pending=None
            if (current.scheduled+32400)%3600==0 and current.scheduled+600<end:
                forecast=live.issue(predictor,current,previous,wx[j])
                forecast['draws'].pop('recent_trend',None)
                if set(forecast['draws'])!=set(MODELS):raise ValueError('Missing model')
                forecast['hash']=hashlib.sha256(b''.join(np.ascontiguousarray(forecast['draws'][k]).tobytes() for k in MODELS)).hexdigest()
                ledger.write(json.dumps({'event':'issue','origin':current.scheduled,'cells':len(forecast['nodes']),'forecast_hash':forecast['hash'],'future_label_accessed':False})+'\n');ledger.flush()
                issues+=1;issued_cells+=len(forecast['nodes']);input_proxy_present+=forecast['previous_proxy_present'];pending=forecast
                if issues%20==0:emit(stage='H25_replay',issued_origins=issues,valid_pairs=valid_pairs,seconds=time.perf_counter()-timer)
            previous=current
    metrics=acc.result();comparison=paired(acc.daily['h24_fm'],acc.daily['h24_poisson'])
    days=int((acc.daily['h24_fm'][:,2]>0).sum());enough=valid_pairs>=10000 and positive_pairs>=5000 and len(stations)>=500 and days>=10
    under_guard=bool(acc.daily['h24_fm'][:,1].sum()<=acc.daily['h24_poisson'][:,1].sum())
    gain=comparison['relative_reduction'];decision='inconclusive'
    if enough:
        if gain<.01 or not under_guard:decision='not_supported'
        elif all(v['relative_reduction_95'][0]>0 for v in comparison['intervals']):decision='supported'
    result={'hypothesis_id':'H-0025','started_at_utc':start,'finished_at_utc':now(),'code_commit':code,
            'plan_sha256':record['preregistration']['plan_sha256'],'input_manifest_sha256':large.digest(mp),'environment':env,
            'scope':'Reused September hourly-origin +10minute observed inventory gaps; not latent denied rentals or dispatch effect',
            'coverage':{'valid_pairs':valid_pairs,'positive_pairs':positive_pairs,'scored_ids':len(stations),'scored_days':days,
                        'issued_origins':issues,'scored_origins':scored_origins,'issued_cells':issued_cells,'previous_proxy_present':input_proxy_present},
            'primary':{'decision':decision,'sufficient_data':enough,'underprediction_guard':under_guard,**comparison},
            'metrics':metrics,'daily_primary':{k:v.tolist() for k,v in acc.daily.items()},'kernel_checks':checks,
            'weights_updated':False,'api_selected':False,'api_opened_for_external_scoring':True,
            'seconds':time.perf_counter()-timer,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'limitations':['Fixed target floors and cost ratios are assumptions, not operator-confirmed quantities',
                           'Inventory gap is not unobserved denied rental demand',
                           'No causal action simulation or relocation saving is measured',
                           'Only hourly September origins, not the full H13 cohort',
                           'Reused exposed September; no winner/threshold selection or calibration',
                           'Historical completed-flow input differs from API stock-change proxy; weather is lagged archive, not live feed',
                           'Nonnegative endpoint projection does not model event order, capacity or operator moves']}
    write(out/'results.json',result);emit(stage='H25_complete',primary=result['primary'],coverage=result['coverage'],seconds=result['seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('h13','h22','h24','panel','weather','out'):p.add_argument('--'+name,required=True)
    evaluate(p.parse_args())
