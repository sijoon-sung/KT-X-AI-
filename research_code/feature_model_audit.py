"""H23 preregistered, historical-only feature/model audit. Run on dislab.

No inventory/API loader, online update, new feature source, or deployment change.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse, gc, hashlib, json, platform, subprocess, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from scipy.special import xlogy
import xgboost as xgb
import rolling_inventory_model as old
import large_inventory_model as large
from forecast_transfer import cpu_state

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(11,29); FIX=20261110
VARIANTS=('full','no_graph','shuffled_neighbors','no_weather','no_calendar','no_recent','no_embedding','no_pretrain')

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def emit(**x):print(json.dumps(x,ensure_ascii=False,allow_nan=False),flush=True)
def digest(p):return large.digest(p)
def ix(a,r,n):return large.cpu_index(a,r,n)
def labels(c,r,n):return ix(c.y,r,n)
def indices(c,r,n):return torch.as_tensor(r,device=c.device),torch.as_tensor(n,device=c.device)

class AuditForecaster(large.MemoryForecaster):
    def __init__(self,variant):super().__init__();self.variant=variant
    def features(self,c,r,n,z,nz):
        v=self.variant;near_n=c.permutation[n] if v=='shuffled_neighbors' else n
        rec=ix(c.recent,r,n);nr=ix(c.recent,r,near_n)
        calendar=c.calendar[r];phase=c.phase[r];weather=c.weather[r]
        ez=z[n];en=nz[near_n]
        if v=='no_recent':rec=rec*0;nr=nr*0
        if v=='no_weather':weather=weather*0
        if v=='no_calendar':calendar=calendar*0;phase=phase*0
        if v=='no_embedding':ez=ez*0;en=en*0
        own=torch.cat((c.log[c.slot[r],n],calendar,phase,weather,ez,rec[:,:2]),1)
        near=torch.cat((c.neighbor[c.slot[r],near_n],en,nr[:,2:]),1)
        if v=='no_graph':near=near*0
        return own,near
    def rate(self,c,r,n,kind='recent',z=None,nz=None,hidden=False):
        if z is None:z,nz=self.encode(c)
        own,near=self.features(c,r,n,z,nz)
        h=nn.functional.silu(self.own(own)+self.neighbor(near))
        for layer in self.layers:h=nn.functional.silu(layer(h))
        rate=c.prior[c.slot[r],n]*torch.exp(2*torch.tanh(self.output(h)/2))
        return (rate,h) if hidden else rate

@torch.no_grad()
def rates(c,model,rows,nodes):
    model.eval();z=model.encode(c);parts=[]
    for b in range(0,len(rows),16384):
        r,n=indices(c,rows[b:b+16384],nodes[b:b+16384]);parts.append(model.rate(c,r,n,'recent',*z).cpu().numpy())
    return np.concatenate(parts)

def dev(y,p):
    p=np.maximum(np.asarray(p,dtype=np.float64),1e-9)
    return (2*(xlogy(y,np.asarray(y)/p)-y+p)).mean(1)

def mean_fit(c,v,seed,source,out,vr,vn,vy):
    start=time.perf_counter();torch.manual_seed(seed);m=AuditForecaster(v).to(c.device)
    if v!='no_pretrain':
        saved=torch.load(source/f'source_forecast-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']
        m.encoder.load_state_dict({k[8:]:x for k,x in saved.items() if k.startswith('encoder.')})
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);best=float('inf');trace=[]
    for ep in range(17):
        if ep:
            m.train();rr,nnodes=c.sample(seed,ep,524288)
            for b in range(0,len(rr),8192):
                r,n=indices(c,rr[b:b+8192],nnodes[b:b+8192]);p=m.rate(c,r,n);y=labels(c,r,n)
                loss=(p-y*torch.log(p)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite mean loss')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        score=float(dev(vy,rates(c,m,vr,vn)).mean());trace.append({'epoch':ep,'validation_deviance':score})
        if score<best:best=score;chosen=cpu_state(m);epoch=ep
    m.load_state_dict(chosen);m.eval()
    torch.save({'state_dict':chosen,'variant':v,'seed':seed,'epoch':epoch},out/f'{v}-{seed}.pt')
    report={'variant':v,'seed':seed,'selected_epoch':epoch,'validation_deviance':best,'parameters':sum(p.numel() for p in m.parameters()),'seconds':time.perf_counter()-start,'trace':trace}
    emit(stage='mean_complete',**{k:x for k,x in report.items() if k!='trace'})
    return m,report

def flow_fit(c,m,seed,out,vr,vn):
    began=time.perf_counter();torch.manual_seed(seed);f=old.CountFlow(context_width=256,width=128).to(c.device)
    m.eval();opt=torch.optim.AdamW(f.parameters(),lr=.001,weight_decay=.0001)
    tr,tn=indices(c,vr,vn)
    with torch.no_grad():
        z=m.encode(c);p,h=m.rate(c,tr,tn,'recent',*z,hidden=True);center=torch.log1p(p);y=labels(c,tr,tn)
    best=float('inf');trace=[]
    for ep in range(21):
        if ep:
            f.train();rr,nnodes=c.sample(seed,ep,524288);g=torch.Generator(device=c.device).manual_seed(seed+ep)
            for b in range(0,len(rr),8192):
                r,n=indices(c,rr[b:b+8192],nnodes[b:b+8192])
                with torch.no_grad():p0,h0=m.rate(c,r,n,'recent',*z,hidden=True);mu=torch.log1p(p0)
                loss=old.flow_loss(f,labels(c,r,n),h0,mu,g)
                if not torch.isfinite(loss):raise ValueError('Nonfinite flow loss')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(f.parameters(),1);opt.step()
        f.eval();g=torch.Generator(device=c.device).manual_seed(FIX);total=0.
        with torch.no_grad():
            for b in range(0,len(vr),8192):total+=float(old.flow_loss(f,y[b:b+8192],h[b:b+8192],center[b:b+8192],g))*len(y[b:b+8192])
        score=total/len(vr);trace.append({'epoch':ep,'validation_flow_loss':score})
        if score<best:best=score;chosen=cpu_state(f);epoch=ep
    f.load_state_dict(chosen);f.eval();torch.save({'state_dict':chosen,'seed':seed,'epoch':epoch},out/f'flow-{seed}.pt')
    report={'seed':seed,'selected_epoch':epoch,'validation_flow_loss':best,'seconds':time.perf_counter()-began,'trace':trace}
    emit(stage='flow_complete',seed=seed,epoch=epoch,seconds=report['seconds'])
    return f,report

@torch.no_grad()
def tree_features(c,encoder,rr,nnodes):
    z=encoder(c.profiles);nz=c.graph@z;m=AuditForecaster('full').to(c.device);parts=[];pri=[]
    for b in range(0,len(rr),16384):
        r,n=indices(c,rr[b:b+16384],nnodes[b:b+16384]);own,near=m.features(c,r,n,z,nz)
        parts.append(torch.cat((own,near),1).cpu().numpy());pri.append(c.prior[c.slot[r],n].cpu().numpy())
    return np.concatenate(parts),np.concatenate(pri)

def fit_trees(c,source,out,vr,vn,vy):
    encoder=old.parent.WeeklyEncoder().to(c.device)
    state=torch.load(source/'source_forecast-11.pt',map_location='cpu',weights_only=True)['state_dict']
    encoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('encoder.')});encoder.eval()
    rr,nnodes=c.sample(FIX,0,524288,'train');r,n=indices(c,rr,nnodes);y=labels(c,r,n).cpu().numpy()
    x,p=tree_features(c,encoder,rr,nnodes);xv,pv=tree_features(c,encoder,vr,vn)
    models=[];logs=[]
    for seed in SEEDS:
        chans=[];t=time.perf_counter()
        for j in range(2):
            dt=xgb.DMatrix(x,label=y[:,j],base_margin=np.log(p[:,j]));dv=xgb.DMatrix(xv,label=vy[:,j],base_margin=np.log(pv[:,j]))
            params={'objective':'count:poisson','eval_metric':'poisson-nloglik','tree_method':'hist','device':'cuda','max_depth':6,'max_bin':128,'eta':.05,'subsample':.8,'colsample_bytree':.8,'min_child_weight':20,'lambda':1,'nthread':4,'seed':seed}
            failures=[]
            try:bst=xgb.train(params,dt,500,evals=[(dv,'validation')],early_stopping_rounds=40,verbose_eval=False)
            except xgb.core.XGBoostError as err:
                failures.append(str(err));params['device']='cpu';emit(stage='tree_cuda_failure_cpu_retry',seed=seed,channel=j)
                bst=xgb.train(params,dt,500,evals=[(dv,'validation')],early_stopping_rounds=40,verbose_eval=False)
            bst.save_model(out/f'xgb-{seed}-{j}.ubj');chans.append(bst)
            logs.append({'seed':seed,'channel':j,'best_iteration':bst.best_iteration,'validation_poisson_nloglik':float(bst.best_score),'device':params['device'],'cuda_failures':failures})
            del dt,dv;gc.collect()
        models.append(chans);emit(stage='tree_complete',seed=seed,seconds=time.perf_counter()-t)
    del x,y,xv;gc.collect();torch.cuda.empty_cache()
    return encoder,models,logs

def tree_rates(c,encoder,models,rr,nnodes):
    x,p=tree_features(c,encoder,rr,nnodes);per=[]
    for chans in models:
        channels=[]
        for j,bst in enumerate(chans):
            dm=xgb.DMatrix(x,base_margin=np.log(p[:,j]));channels.append(bst.predict(dm,iteration_range=(0,bst.best_iteration+1)))
        per.append(np.stack(channels,1))
    return per

def validate_parents(folder):
    lock=json.loads((folder/'model-lock.json').read_text(encoding='utf-8'))
    if lock.get('api_opened') is not False:raise ValueError('Parent is not historical-only')
    records=[]
    for item in lock['artifacts']:
        if digest(folder/item['id'])!=item['sha256']:raise ValueError('Changed parent artifact '+item['id'])
        records.append(item)
    return {'hypothesis_id':lock['hypothesis_id'],'lock_sha256':digest(folder/'model-lock.json'),'artifacts':records}

def load_references(a,c):
    refs={};flows={};checks={}
    for name,path in [('locked_h13',Path(a.h13)),('locked_h22',Path(a.h22))]:
        checks[name]=validate_parents(path);models=[];ff=[]
        for seed in SEEDS:
            if name=='locked_h13':
                m=large.MemoryForecaster().to(c.device);saved=torch.load(path/f'recent-{seed}.pt',map_location='cpu',weights_only=True);m.load_state_dict(saved['state_dict'])
                f=old.CountFlow(context_width=256,width=128).to(c.device);f.load_state_dict(torch.load(path/f'flow-{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
            else:
                saved=torch.load(path/f'large-{seed}.pt',map_location='cpu',weights_only=True)
                m=large.LargeForecaster().to(c.device);m.load_state_dict(saved['gcn']);f=large.CachedFlow().to(c.device);f.load_state_dict(saved['flow'])
            m.eval();f.eval();models.append(m);ff.append(f)
        refs[name]=models;flows[name]=ff
    return refs,flows,checks

def intervals(daily_candidate,daily_control):
    # Input rows contain per-day loss sum and exposure count, with identical dates.
    a=np.asarray(daily_candidate,float);b=np.asarray(daily_control,float);n=len(a);out={}
    for block in (3,7):
        rng=np.random.default_rng(FIX);starts=rng.integers(0,n-block+1,(20000,(n+block-1)//block))
        ids=(starts[...,None]+np.arange(block)).reshape(20000,-1)[:,:n]
        av=a[ids].sum(1);bv=b[ids].sum(1);gain=1-(av[:,0]/av[:,1])/(bv[:,0]/bv[:,1])
        alpha=.05/14
        out[str(block)]={'ci95':np.quantile(gain,[.025,.975]).tolist(),'feature_simultaneous_ci':np.quantile(gain,[alpha/2,1-alpha/2]).tolist()}
    out['relative_reduction']=float(1-(a[:,0].sum()/a[:,1].sum())/(b[:,0].sum()/b[:,1].sum()))
    return out

def daily(values,days):return [[float(values[days==d].sum()),int((days==d).sum())] for d in np.unique(days)]

def distribution(c,models,flows,rr,nnodes,truth):
    vals=[];point=[];t=time.perf_counter()
    with torch.no_grad():encoded=[m.encode(c) for m in models]
    for b in range(0,len(rr),256):
        r,n=indices(c,rr[b:b+256],nnodes[b:b+256])
        draws=large.draws(models,flows,c,r,n,FIX+b,encoded)
        values=np.linalg.norm(draws-truth[b:b+256,None,:],axis=2).mean(1)-.5*np.linalg.norm(draws[:,:64]-draws[:,64:],axis=2).mean(1)
        vals.append(values);point.append(np.abs(np.median(draws,axis=1)-truth[b:b+256]).mean(1))
    return np.concatenate(vals),{'energy':float(np.concatenate(vals).mean()),'median_count_mae':float(np.concatenate(point).mean()),'seconds':time.perf_counter()-t,'station_time_exposures':len(rr)}

def run(a):
    began=now();timer=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'start.json').exists():raise ValueError('Do not overwrite a started audit; preserve failed attempts')
    rec=json.loads((ROOT/'research/hypotheses/H-0023.json').read_text(encoding='utf-8'))
    canonical=json.dumps(rec['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=rec['preregistration']['plan_sha256']:raise ValueError('Changed plan')
    mp=ROOT/'research/datasets/H23-historical-feature-audit-v1.json'
    if digest(mp)!=rec['plan']['data']['input_manifest_sha256']:raise ValueError('Changed manifest')
    manifest=json.loads(mp.read_text(encoding='utf-8'));sources={}
    for item in manifest['sources']:
        p=Path(a.source)/item['id'].replace('H12-Seoul-','') if item['id'].startswith('H12-') else Path(a.data)/item['id'].replace('H13-historical-','')
        if digest(p)!=item['sha256']:raise ValueError('Input mismatch '+item['id'])
        sources[item['id']]=item['sha256']
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old.configure();free,total=torch.cuda.mem_get_info()
    if free<3*1024**3:raise ValueError('Less than3GiB free GPU memory')
    env={'execution':'dislab','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'xgboost':xgb.__version__,'gpu':torch.cuda.get_device_name(),'gpu_free_bytes':free,'gpu_total_bytes':total,'threads':4,'tf32':False,'deterministic':True}
    write(out/'start.json',{'started_at_utc':began,'code_commit':code,'plan_sha256':rec['preregistration']['plan_sha256'],'environment':env,'api_opened':False});emit(stage='audit_started',environment=env)
    c=large.historical_context(a.data,'cuda');c.permutation=torch.as_tensor(np.random.default_rng(FIX).permutation(c.n),device=c.device)
    vr,vn=c.sample(FIX,0,131072,'validation');r,n=indices(c,vr,vn);vy=labels(c,r,n).cpu().numpy()
    feature_audit={'stations':c.n,'train_time_rows':int(c.splits['train'].sum()),'validation_time_rows':int(c.splits['validation'].sum()),'test_time_rows':int(c.splits['test'].sum()),'profile_dim':int(c.profiles.shape[1]),'weather_dim':int(c.weather.shape[1])}
    # Aggregate activity over supported training times; no API observations.
    train_rows=np.flatnonzero(c.splits['train']);w=c.weather[torch.as_tensor(train_rows,device=c.device)]
    feature_audit['weather_nonzero_fraction']=float((w!=0).float().mean());feature_audit['weather_variable_columns']=int((w.max(0).values!=w.min(0).values).sum());del w
    models={};training=[];val_rates={}
    with torch.no_grad():val_rates['prior']=c.prior[c.slot[r],n].cpu().numpy()
    for v in VARIANTS:
        models[v]=[];pp=[]
        for seed in SEEDS:
            m,log=mean_fit(c,v,seed,Path(a.source),out,vr,vn,vy);models[v].append(m);training.append(log);pp.append(rates(c,m,vr,vn))
        val_rates[v]=np.mean(pp,axis=0);write(out/'training.json',training)
        if time.perf_counter()-timer>3600:raise TimeoutError('Preregistered wall time exceeded')
    full_flows=[];flow_logs=[]
    for m,seed in zip(models['full'],SEEDS):
        f,log=flow_fit(c,m,seed,out,vr,vn);full_flows.append(f);flow_logs.append(log)
    tree_encoder,trees,tree_logs=fit_trees(c,Path(a.source),out,vr,vn,vy)
    val_rates['xgboost']=np.mean(tree_rates(c,tree_encoder,trees,vr,vn),axis=0)
    validation={k:{'deviance':float(dev(vy,p).mean()),'mae':float(np.abs(vy-p).mean())} for k,p in val_rates.items()}
    control=min(['prior','no_graph','xgboost'],key=lambda k:validation[k]['deviance'])
    refs,ref_flows,ref_checks=load_references(a,c);models.update(refs)
    lock={'locked_at_utc':now(),'plan_sha256':rec['preregistration']['plan_sha256'],'input_manifest_sha256':digest(mp),'code_commit':code,'validation':validation,'selected_non_graph_control':control,'artifacts':[{'id':p.name,'sha256':digest(p)} for p in sorted(out.glob('*')) if p.suffix in ('.pt','.ubj')],'parent_checks':ref_checks,'api_opened':False,'March_predictions_opened':False}
    write(out/'pre-test-lock.json',lock);emit(stage='all_models_locked_before_march',selected_non_graph_control=control)
    # March is opened only after all February-based choices are fixed.
    tr,tn=c.sample(FIX,0,262144,'test');r,n=indices(c,tr,tn);y=labels(c,r,n).cpu().numpy();days=(c.times[tr]+32400)//86400
    predictions={'prior':c.prior[c.slot[r],n].cpu().numpy()};per_seed={}
    for k,ms in models.items():
        pp=[rates(c,m,tr,tn) for m in ms];predictions[k]=np.mean(pp,axis=0)
        per_seed[k]=[{'seed':s,'deviance':float(dev(y,p).mean()),'mae':float(np.abs(y-p).mean())} for s,p in zip(SEEDS,pp)]
    pp=tree_rates(c,tree_encoder,trees,tr,tn);predictions['xgboost']=np.mean(pp,axis=0);per_seed['xgboost']=[{'seed':s,'deviance':float(dev(y,p).mean()),'mae':float(np.abs(y-p).mean())} for s,p in zip(SEEDS,pp)]
    scores={};daydev={}
    for k,p in predictions.items():
        dv=dev(y,p);daydev[k]=daily(dv,days)
        scores[k]={'deviance':float(dv.mean()),'mean_count_mae':float(np.abs(y-p).mean()),'station_time_exposures':len(y),'gain_vs_prior':float(1-dv.mean()/dev(y,predictions['prior']).mean())}
    primary=intervals(daydev['full'],daydev[control]);gain=primary['relative_reduction']
    supported=gain>=.01 and all(primary[str(b)]['ci95'][0]>0 for b in (3,7))
    decision='supported' if supported else 'not_supported' if gain<.01 else 'inconclusive'
    effects={}
    for v in VARIANTS[1:]:
        it=intervals(daydev['full'],daydev[v]);g=it['relative_reduction'];lo=all(it[str(b)]['feature_simultaneous_ci'][0]>0 for b in (3,7));hi=all(it[str(b)]['feature_simultaneous_ci'][1]<0 for b in (3,7))
        it['decision']='supported' if g>=.01 and lo else 'harmful' if g<=-.01 and hi else 'no_demonstrated_1pct_gain';effects[v]=it
    distr={};dayenergy={};rr,nnodes=tr[:32768],tn[:32768];yy=y[:32768];dd=days[:32768]
    for k in ['full','no_graph','locked_h13','locked_h22']:
        for suffix,flows in [('poisson',None)]+([('fm',full_flows if k=='full' else ref_flows[k])] if k!='no_graph' else []):
            name=k+'_'+suffix;values,report=distribution(c,models[k],flows,rr,nnodes,yy);distr[name]=report;dayenergy[name]=daily(values,dd);emit(stage='distribution_scored',model=name,**report)
    fm=intervals(dayenergy['full_fm'],dayenergy['full_poisson']);fm['decision']='supported' if fm['relative_reduction']>=.01 and all(fm[str(b)]['ci95'][0]>0 for b in (3,7)) else 'no_demonstrated_1pct_gain'
    groups={};hours=((c.times[tr]+32400)%86400)//3600
    for name,mask in [('night',(hours<7)|(hours>=22)),('day',(hours>=7)&(hours<22)),('zero_count',(y.sum(1)==0)),('positive_count',(y.sum(1)>0))]:
        groups[name]={'exposures':int(mask.sum()),'models':{k:{'deviance':float(dev(y[mask],p[mask]).mean()),'mae':float(np.abs(y[mask]-p[mask]).mean())} for k,p in predictions.items()}}
    result={'hypothesis_id':'H-0023','started_at_utc':began,'finished_at_utc':now(),'code_commit':code,'plan_sha256':lock['plan_sha256'],'input_manifest_sha256':digest(mp),'environment':env,'api_opened':False,'scope':'Previously exposed March completed-count audit, not stock accuracy','feature_audit':feature_audit,'training':training,'flow_training':flow_logs,'tree_training':tree_logs,'validation':validation,'test':scores,'test_rank_by_deviance':sorted(scores,key=lambda k:scores[k]['deviance']),'per_seed':per_seed,'primary':{'control':control,'decision':decision,**primary},'feature_effects':effects,'distributions':distr,'fm_effect':fm,'groups':groups,'test_dates':len(np.unique(days)),'distribution_test_dates':len(np.unique(dd)),'test_target_zero_fraction':float((y==0).mean()),'daily_deviance':daydev,'daily_energy':dayenergy,'seconds':time.perf_counter()-timer,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'input_hashes':sources,'limitations':['March previously exposed; no untouched generalization claim','Only historical completed counts, no observed stock or unmet demand truth','Neural ablations have equal optimization budgets; tree uses a different fixed family-specific budget and smaller unique training sample coverage','XGBoost uses frozen source-only embeddings; not a raw-only standalone baseline','Explicit-calendar removal leaves calendar information in priors','No-weather removal is absence of all54 values/masks, not a real-time observed weather availability experiment','Runtime weather and recent-stock proxy differ from historical training signals','Only one temporal split, two seeds, fixed hyperparameters; no global-best proof','No automatic production promotion; no API opened']}
    write(out/'results.json',result);emit(stage='audit_complete',seconds=result['seconds'],primary=result['primary'],ranking=result['test_rank_by_deviance'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--source',required=True);p.add_argument('--h13',required=True);p.add_argument('--h22',required=True);p.add_argument('--out',required=True);run(p.parse_args())
