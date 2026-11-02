"""H22 fixed historical comparison then locked September replay; no updates."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import large_inventory_model as m
import rolling_inventory_predict as live
from rolling_inventory_metrics import scores
from forecast_transfer import now,emit,weather_for
from large_inventory_fit import write,tensor,labels

def bootstrap(daily):
    x=np.asarray(daily);point=1-x[:,0].sum()/x[:,1].sum();intervals=[]
    for block in (3,7):
        rng=np.random.default_rng(20261103);n=len(x)
        starts=rng.integers(0,n-block+1,(2000,(n+block-1)//block))
        indices=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:n]
        sums=x[indices].sum(1);valid=sums[:,1]>0;g=1-sums[valid,0]/sums[valid,1]
        intervals.append({'block_days':block,'replicates':int(valid.sum()),'interval':np.percentile(g,[2.5,97.5]).tolist()})
    decision='inconclusive'
    if point>.01 and all(i['interval'][0]>.01 for i in intervals):decision='supported'
    elif all(i['interval'][1]<=.01 for i in intervals):decision='not_supported'
    return {'relative_gain':float(point),'intervals':intervals,'decision':decision}

def historical(a):
    start=now();timer=time.perf_counter();predictor=m.Predictor(a.fit,'cuda');c=m.historical_context(a.data,'cuda')
    models=[];flows=[]
    for seed in m.SEEDS:
        model=m.MemoryForecaster().cuda();model.load_state_dict(torch.load(Path(a.parent)/f'recent-{seed}.pt',weights_only=True,map_location='cpu')['state_dict']);model.eval();models.append(model)
        flow=m.CountFlow(context_width=256,width=128).cuda();flow.load_state_dict(torch.load(Path(a.parent)/f'flow-{seed}.pt',weights_only=True,map_location='cpu')['state_dict']);flow.eval();flows.append(flow)
    rr,nnodes=c.sample(20261103,0,16384,'test');daily=np.zeros((31,3));totals={k:{'energy':0.,'mae':0.} for k in ('large_cfm','h13_cfm','large_poisson')}
    with torch.no_grad():encoded=[x.encode(c) for x in models];large_encoded=[x.encode(c) for x in predictor.models['recent']]
    for b in range(0,len(rr),256):
        r=tensor(rr[b:b+256],c);n=tensor(nnodes[b:b+256],c);truth=labels(c,r,n).cpu().numpy()
        energies={}
        for name,mm,ff,z in [('large_cfm',predictor.models['recent'],predictor.flows,large_encoded),('h13_cfm',models,flows,encoded),('large_poisson',predictor.models['recent'],None,large_encoded)]:
            x=m.draws(mm,ff,c,r,n,20261103+b,z)
            energy=np.linalg.norm(x-truth[:,None],axis=2).mean(1)-.5*np.linalg.norm(x[:,:64]-x[:,64:],axis=2).mean(1)
            energies[name]=energy;totals[name]['energy']+=float(energy.sum());totals[name]['mae']+=float(np.abs(np.sort(x,axis=1)[:,63]-truth).mean(1).sum())
        days=((c.times[rr[b:b+256]]+32400)//86400-(m.old.d.VALID_END+32400)//86400).astype(int)
        np.add.at(daily[:,0],days,energies['large_cfm']);np.add.at(daily[:,1],days,energies['h13_cfm']);np.add.at(daily[:,2],days,1)
    for value in totals.values():
        for key in value:value[key]/=len(rr)
    result={'hypothesis_id':'H-0022','started_at_utc':start,'finished_at_utc':now(),'model_lock_sha256':m.digest(Path(a.fit)/'model-lock.json'),'scope':'March historical completed counts, not measured stock','sample':len(rr),'metrics':totals,'daily':daily.tolist(),'comparison':bootstrap(daily),'seconds':time.perf_counter()-timer,'api_opened':False}
    write(a.out,result);emit(stage='H22_historical_complete',metrics=totals,comparison=result['comparison'],seconds=result['seconds'])

def api(a):
    start=now();timer=time.perf_counter();predictor=m.Predictor(a.fit,'cuda')
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'research/datasets/tashu-rolling-api-H0013-v1.json').read_text(encoding='utf-8'))
    expected=next(x['sha256'] for x in manifest['sources'] if x['id']=='observed_panel.npz')
    if m.digest(a.panel)!=expected:raise ValueError('Official September source changed.')
    # Fitted weights and historical adoption decision exist before API is opened.
    hist=json.loads(Path(a.historical).read_text(encoding='utf-8'))
    if hist['model_lock_sha256']!=m.digest(Path(a.fit)/'model-lock.json'):raise ValueError('Historical comparison uses different model.')
    with np.load(a.panel,allow_pickle=False) as f:
        times=f['scheduled_at_unix_seconds'].copy();ids=f['station_ids'].astype(str);stock=f['stock'].copy();actual=f['observed_at_unix_seconds'].copy()
    lookup={s:i for i,s in enumerate(ids)};nodes=np.array([i for i,s in enumerate(predictor.ids) if s in lookup]);ix=np.array([lookup[predictor.ids[i]] for i in nodes])
    selected=np.flatnonzero((times>=m.old.h.stamp('2026-09-01'))&(times<m.old.h.stamp('2026-10-01')))
    wx=weather_for(Path(a.weather),'daejeon',times[selected]);previous=None;pending=None;total=0;scored=set();origins=0
    accum={g:{} for g in ('all','current_zero','current_positive','strict_elapsed_30s')};daily=np.zeros((14,5))
    for j,index in enumerate(selected):
        values=np.full(predictor.n,np.nan);stamps=np.full(predictor.n,-1.);values[nodes]=stock[index,ix];stamps[nodes]=actual[index,ix]
        current=live.Snapshot(int(times[index]),values,stamps,len(ids),len(nodes))
        if pending is not None:
            valid,y,strict=live.score_issue(pending,current)
            if valid.any():
                n=int(valid.sum());total+=n;scored.update(pending['nodes'][valid].tolist());cur=pending['current'][valid]
                masks={'all':np.ones(n,bool),'current_zero':cur==0,'current_positive':cur>0,'strict_elapsed_30s':strict}
                for k,x in pending['draws'].items():
                    s=scores(x[valid],y)
                    for g,mask in masks.items():
                        e=accum[g].setdefault(k,{'n':0,'sums':{}});e['n']+=int(mask.sum())
                        for metric,v in s.items():e['sums'][metric]=e['sums'].get(metric,0.)+float(v[mask].sum())
                day=int((pending['scheduled']+32400)//86400-(m.old.h.stamp('2026-09-17')+32400)//86400)
                for col,k in enumerate(('recent_cfm','recent_poisson','persistence','recent_trend')):daily[day,col]+=float(scores(pending['draws'][k][valid],y)['fair_crps'].sum())
                daily[day,-1]+=n;origins+=1
        pending=live.issue(predictor,current,previous,wx[j]);previous=current
        if (j+1)%100==0:emit(stage='H22_api',origins=j+1,pairs=total,seconds=time.perf_counter()-timer)
    groups={}
    for g,entries in accum.items():
        groups[g]={}
        for k,e in entries.items():
            v={name:(value/e['n'] if e['n'] else None) for name,value in e['sums'].items()}
            if v.get('mean_sq') is not None:v['mean_rmse']=float(np.sqrt(v.pop('mean_sq')))
            v['median_mae']=v.pop('median_abs');groups[g][k]={'pairs':e['n'],**v}
    reference=json.loads(Path(a.reference).read_text(encoding='utf-8'))
    if total!=reference['coverage']['valid_pairs'] or len(scored)!=reference['coverage']['scored_ids']:raise ValueError('H13 comparison cohort mismatch.')
    report={'hypothesis_id':'H-0022','started_at_utc':start,'finished_at_utc':now(),'model_lock_sha256':m.digest(Path(a.fit)/'model-lock.json'),'api_panel_sha256':expected,'scope':'September17–30 reused official API, evaluation only','historical_decision':hist['comparison']['decision'],'coverage':{'valid_pairs':total,'scored_ids':len(scored),'issued_origins':len(selected),'scored_origins':origins},'groups':groups,'daily_fair_crps':daily.tolist(),'h13_reference_sha256':m.digest(a.reference),'h13_reference_metrics':reference['groups']['all'],'seconds':time.perf_counter()-timer,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'weights_updated':False,'api_selected':False}
    write(a.out,report);emit(stage='H22_api_complete',coverage=report['coverage'],metrics=groups['all'],seconds=report['seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['historical','api']);p.add_argument('--fit',required=True);p.add_argument('--out',required=True)
    for name in ('data','parent','panel','weather','historical','reference'):p.add_argument('--'+name)
    a=p.parse_args();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    if Path(a.out).exists():raise ValueError('Preserve existing result.')
    historical(a) if a.mode=='historical' else api(a)
