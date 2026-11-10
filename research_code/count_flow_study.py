"""H24 dislab-only historical experiment. No API path, feature expansion or deployment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,platform,subprocess,time
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from scipy.special import xlogy
import rolling_inventory_model as old
import large_inventory_model as large
from forecast_transfer import cpu_state
import count_flow_models as cf

ROOT=Path(__file__).resolve().parents[1];FIX=20261124;SEEDS=(11,29)
ENCODERS=('full_retrain','compact_local','compact_graph','compact_profile')

def now():return datetime.now(timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def emit(**x):print(json.dumps(x,ensure_ascii=False,allow_nan=False),flush=True)
def ix(c,r,n):return torch.as_tensor(r,device=c.device),torch.as_tensor(n,device=c.device)
def truth(c,r,n):return large.cpu_index(c.y,r,n)
def dev(y,p):
    p=np.maximum(np.asarray(p,float),1e-9);return (2*(xlogy(y,y/p)-y+p)).mean(1)

@torch.no_grad()
def rates(c,m,rows,nodes):
    m.eval();encoded=m.encode(c);parts=[]
    for b in range(0,len(rows),16384):
        r,n=ix(c,rows[b:b+16384],nodes[b:b+16384]);parts.append(m.rate(c,r,n,'recent',*encoded).cpu().numpy())
    return np.concatenate(parts)

def fit_rate(c,kind,seed,source,out,vr,vn,vy):
    began=time.perf_counter();torch.manual_seed(seed)
    if kind=='full_retrain':
        m=large.MemoryForecaster().to(c.device);m.hidden_width=256
        state=torch.load(source/f'source_forecast-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']
        m.encoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('encoder.')})
    else:m=cf.CompactEncoder(kind).to(c.device)
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001);best=float('inf');trace=[]
    for ep in range(17):
        if ep:
            m.train();rr,ns=c.sample(seed,ep,524288)
            for b in range(0,len(rr),8192):
                r,n=ix(c,rr[b:b+8192],ns[b:b+8192]);p=m.rate(c,r,n,'recent');y=truth(c,r,n);loss=(p-y*torch.log(p)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite encoder loss')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
        score=float(dev(vy,rates(c,m,vr,vn)).mean());trace.append({'epoch':ep,'validation_deviance':score})
        if score<best:best=score;selected=ep;chosen=cpu_state(m)
    m.load_state_dict(chosen);m.eval();torch.save({'state_dict':chosen,'kind':kind,'seed':seed,'epoch':selected},out/f'encoder-{kind}-{seed}.pt')
    log={'kind':kind,'seed':seed,'epoch':selected,'parameters':sum(p.numel() for p in m.parameters()),'validation_deviance':best,'trace':trace,'seconds':time.perf_counter()-began}
    emit(stage='encoder_done',**{k:v for k,v in log.items() if k!='trace'});return m,log

@torch.no_grad()
def distribution(c,models,flows,kind,epochs,rows,nodes,y,steps=None):
    encoded=[m.encode(c) for m in models];energies=[];mae=[];crps=[];brier=[];pred_zero=[];mean=[];observed=[];neg=0;num=0;began=time.perf_counter()
    for b in range(0,len(rows),256):
        if hasattr(c,'budget_start') and time.perf_counter()-c.budget_start>3600:raise TimeoutError('Frozen wall-time budget')
        rr,ns=rows[b:b+256],nodes[b:b+256];r,n=ix(c,rr,ns);size=len(rr)
        pairs=[m.rate(c,r,n,'recent',*z,hidden=True) for m,z in zip(models,encoded)]
        choice=np.random.default_rng(FIX+b).integers(0,len(models),(size,128));draws=np.empty((size,128,2),np.int32)
        g=torch.Generator(device=c.device).manual_seed(FIX+b)
        for i,(p,h) in enumerate(pairs):
            ir,islot=np.where(choice==i);index=torch.as_tensor(ir,device=c.device)
            values,bad=cf.sample_flow(None if flows is None else flows[i],kind,epochs[i],p[index],h[index],g,steps=steps)
            draws[ir,islot]=values;neg+=bad;num+=values.size
        yy=y[b:b+size]
        energies.append(np.linalg.norm(draws-yy[:,None],axis=2).mean(1)-.5*np.linalg.norm(draws[:,:64]-draws[:,64:],axis=2).mean(1))
        crps.append((np.abs(draws-yy[:,None]).mean(1)-.5*np.abs(draws[:,:64]-draws[:,64:]).mean(1)).mean(1))
        mae.append(np.abs(np.median(draws,axis=1)-yy).mean(1));p0=(draws==0).mean(1)
        brier.append(((p0-(yy==0))**2).mean(1));pred_zero.append(p0.mean(1));mean.append(draws.mean(1));observed.append(yy)
    e=np.concatenate(energies);pred=np.concatenate(mean);obs=np.concatenate(observed)
    report={'energy':float(e.mean()),'fair_channel_crps':float(np.concatenate(crps).mean()),'median_count_mae':float(np.concatenate(mae).mean()),'zero_count_brier':float(np.concatenate(brier).mean()),'predicted_zero_fraction':float(np.concatenate(pred_zero).mean()),'observed_zero_fraction':float((obs==0).mean()),'mean_count_bias':(pred-obs).mean(0).tolist(),'negative_continuous_raw_fraction':neg/num,'nonnegative_integer_outputs':True,'examples':len(y),'particles':128,'seconds':time.perf_counter()-began}
    return e,report

def new_flow(kind,width,device):return (cf.JumpFlow(width) if kind=='count_jump' else old.CountFlow(context_width=width,width=128)).to(device)

def fit_flow(c,m,kind,seed,out,vr,vn,vy):
    began=time.perf_counter();torch.manual_seed(seed);f=new_flow(kind,m.hidden_width,c.device);opt=torch.optim.AdamW(f.parameters(),lr=.001,weight_decay=.0001)
    r,n=ix(c,vr,vn)
    with torch.no_grad():encoded=m.encode(c);p,h=m.rate(c,r,n,'recent',*encoded,hidden=True);y=truth(c,r,n)
    best_loss=float('inf');best_energy=float('inf');trace=[];checks=[]
    for ep in range(21):
        if ep:
            f.train();rr,ns=c.sample(seed,ep,524288);g=torch.Generator(device=c.device).manual_seed(seed+ep)
            for b in range(0,len(rr),8192):
                r0,n0=ix(c,rr[b:b+8192],ns[b:b+8192])
                with torch.no_grad():p0,h0=m.rate(c,r0,n0,'recent',*encoded,hidden=True)
                loss=cf.training_loss(f,kind,truth(c,r0,n0),h0,p0,g)
                if not torch.isfinite(loss):raise ValueError('Nonfinite count/continuous flow loss')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(f.parameters(),1);opt.step()
        f.eval();g=torch.Generator(device=c.device).manual_seed(FIX);total=0.
        with torch.no_grad():
            for b in range(0,len(vr),8192):total+=float(cf.training_loss(f,kind,y[b:b+8192],h[b:b+8192],p[b:b+8192],g))*len(y[b:b+8192])
        value=total/len(vr);trace.append({'epoch':ep,'validation_loss':value})
        if value<best_loss:best_loss=value;loss_epoch=ep;loss_state=cpu_state(f)
        if ep in (0,5,10,15,20):
            _,score=distribution(c,[m],[f],kind,[ep],vr[:8192],vn[:8192],vy[:8192]);checks.append({'epoch':ep,**score})
            if score['energy']<best_energy:best_energy=score['energy'];energy_epoch=ep;energy_state=cpu_state(f)
            emit(stage='flow_checkpoint',kind=kind,seed=seed,epoch=ep,validation_loss=value,validation_energy=score['energy'])
    selected={};meta={}
    for selector,state,ep in [('score',energy_state,energy_epoch)]+([('mse',loss_state,loss_epoch)] if kind=='gaussian' else []):
        model=new_flow(kind,m.hidden_width,c.device);model.load_state_dict(state);model.eval();selected[selector]=model
        active=ep>0 and bool(torch.count_nonzero(model.net[-1].weight))
        meta[selector]={'epoch':ep,'active':active,'identity_bypass':ep==0 and kind!='gaussian','parameters':sum(x.numel() for x in model.parameters())}
        torch.save({'state_dict':state,'kind':kind,'seed':seed,'selector':selector,**meta[selector]},out/f'flow-{kind}-{selector}-{seed}.pt')
    log={'kind':kind,'seed':seed,'trace':trace,'score_checkpoints':checks,'selected':meta,'seconds':time.perf_counter()-began}
    emit(stage='flow_done',kind=kind,seed=seed,selected=meta,seconds=log['seconds']);return selected,log

def daily(x,days):return [[float(x[days==d].sum()),int((days==d).sum())] for d in np.unique(days)]
def interval(a,b):
    a=np.asarray(a);b=np.asarray(b);n=len(a);report={}
    for block in (3,7):
        rng=np.random.default_rng(FIX);starts=rng.integers(0,n-block+1,(20000,(n+block-1)//block));idx=(starts[...,None]+np.arange(block)).reshape(20000,-1)[:,:n]
        aa=a[idx].sum(1);bb=b[idx].sum(1);gain=1-(aa[:,0]/aa[:,1])/(bb[:,0]/bb[:,1]);report[str(block)]=np.quantile(gain,[.025,.975]).tolist()
    report['relative_reduction']=float(1-(a[:,0].sum()/a[:,1].sum())/(b[:,0].sum()/b[:,1].sum()));return report

def references(path,c):
    lock=json.loads((path/'model-lock.json').read_text(encoding='utf-8'))
    if lock.get('api_opened') is not False:raise ValueError('Nonhistorical parent')
    for item in lock['artifacts']:
        if large.digest(path/item['id'])!=item['sha256']:raise ValueError('Parent artifact changed')
    models=[];flows=[]
    for seed in SEEDS:
        m=large.MemoryForecaster().to(c.device);m.hidden_width=256;m.load_state_dict(torch.load(path/f'recent-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);m.eval();models.append(m)
        f=old.CountFlow(context_width=256,width=128).to(c.device);f.load_state_dict(torch.load(path/f'flow-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);f.eval();flows.append(f)
    return models,flows,{'hypothesis_id':lock['hypothesis_id'],'lock_sha256':large.digest(path/'model-lock.json')}

def run(a):
    start=now();timer=time.perf_counter();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if (out/'start.json').exists():raise ValueError('Output already started; preserve attempts')
    rec=json.loads((ROOT/'research/hypotheses/H-0024.json').read_text(encoding='utf-8'))
    encoded_plan=json.dumps(rec['plan'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(encoded_plan).hexdigest()!=rec['preregistration']['plan_sha256']:raise ValueError('Frozen plan changed')
    mp=ROOT/'research/datasets/H24-compact-count-flow-v1.json'
    if large.digest(mp)!=rec['plan']['data']['input_manifest_sha256']:raise ValueError('Input manifest changed')
    for item in json.loads(mp.read_text(encoding='utf-8'))['sources']:
        p=Path(a.source)/item['id'].replace('H12-Seoul-','') if item['id'].startswith('H12-') else Path(a.data)/item['id'].replace('H13-historical-','')
        if large.digest(p)!=item['sha256']:raise ValueError('Data/source hash mismatch '+item['id'])
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    old.configure();free,total=torch.cuda.mem_get_info()
    if free<3*1024**3:raise ValueError('Insufficient freeGPU')
    env={'execution':'dislab','python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'gpu_free_bytes':free,'gpu_total_bytes':total,'threads':4,'deterministic':True,'tf32':False}
    write(out/'start.json',{'started_at_utc':start,'code_commit':code,'plan_sha256':rec['preregistration']['plan_sha256'],'environment':env,'api_opened':False})
    kernel_checks=cf.kernel_checks('cuda');write(out/'kernel-checks.json',kernel_checks);emit(stage='kernel_checks_passed',checks=kernel_checks)
    c=large.historical_context(a.data,'cuda');c.budget_start=timer;cut=old.d.h.stamp('2026-02-15')
    c.splits['encoder_validation']=c.splits['validation']&(c.times<cut);c.splits['decoder_validation']=c.splits['validation']&(c.times>=cut)
    vr,vn=c.sample(FIX,0,131072,'encoder_validation');r,n=ix(c,vr,vn);vy=truth(c,r,n).cpu().numpy()
    models={};training=[];validation={};params={}
    for kind in ENCODERS:
        models[kind]=[];ps=[]
        for seed in SEEDS:
            m,log=fit_rate(c,kind,seed,Path(a.source),out,vr,vn,vy);models[kind].append(m);training.append(log);ps.append(rates(c,m,vr,vn))
        p=np.mean(ps,axis=0);params[kind]=sum(x.numel() for x in models[kind][0].parameters());validation[kind]={'deviance':float(dev(vy,p).mean()),'mean_count_mae':float(np.abs(vy-p).mean()),'parameters_per_seed':params[kind]};write(out/'encoder-training.json',training)
        if time.perf_counter()-timer>3600:raise TimeoutError('Frozenbudget')
    baseline=validation['full_retrain'];eligible=[k for k in ENCODERS[1:] if params[k]<=.5*params['full_retrain'] and validation[k]['deviance']<=1.01*baseline['deviance'] and validation[k]['mean_count_mae']<=1.01*baseline['mean_count_mae']]
    selected=min(eligible,key=lambda k:(params[k],validation[k]['deviance'],k)) if eligible else 'full_retrain'
    emit(stage='encoder_selected',selected=selected,eligible=eligible,validation=validation)
    er,en=c.sample(FIX,0,131072,'decoder_validation');r,n=ix(c,er,en);ey=truth(c,r,n).cpu().numpy()
    bundles={};flow_training=[]
    for kind in ('gaussian','poisson_continuous','count_jump'):
        for i,(m,seed) in enumerate(zip(models[selected],SEEDS)):
            fitted,log=fit_flow(c,m,kind,seed,out,er,en,ey);flow_training.append(log)
            for selector,f in fitted.items():
                name=kind+'_'+selector
                if name not in bundles:bundles[name]={'kind':kind,'flows':[],'epochs':[],'active':[],'meta':[]}
                b=bundles[name];b['flows'].append(f);b['epochs'].append(log['selected'][selector]['epoch']);b['active'].append(log['selected'][selector]['active']);b['meta'].append(log['selected'][selector])
            write(out/'flow-training.json',flow_training)
    val_dist={}
    for name,b in bundles.items():
        _,report=distribution(c,models[selected],b['flows'],b['kind'],b['epochs'],er[:8192],en[:8192],ey[:8192]);val_dist[name]=report
    _,val_dist['poisson']=distribution(c,models[selected],None,'poisson',[0,0],er[:8192],en[:8192],ey[:8192])
    final=min(('poisson_continuous_score','count_jump_score'),key=lambda k:(val_dist[k]['energy'],0 if k=='poisson_continuous_score' else 1))
    refmodels,refflows,refcheck=references(Path(a.h13),c)
    lock={'hypothesis_id':'H-0024','code_commit':code,'plan_sha256':rec['preregistration']['plan_sha256'],'input_manifest_sha256':large.digest(mp),'locked_at_utc':now(),'api_opened':False,'March_predictions_opened':False,'selected_encoder':selected,'eligible_compact_encoders':eligible,'encoder_validation':validation,'selected_fm':final,'decoder_validation':val_dist,'flow_choices':{k:{'epochs':b['epochs'],'active':b['active'],'kind':b['kind']} for k,b in bundles.items()},'parent':refcheck,'artifacts':[{'id':p.name,'sha256':large.digest(p)} for p in sorted(out.glob('*.pt'))]}
    write(out/'pre-test-lock.json',lock);emit(stage='all_choices_locked',encoder=selected,flow=final,flow_choices=lock['flow_choices'])
    tr,tn=c.sample(FIX,0,262144,'test');r,n=ix(c,tr,tn);y=truth(c,r,n).cpu().numpy();days=(c.times[tr]+32400)//86400
    mean_scores={};perseed={};daily_rate={}
    for kind,ms in {**models,'frozen_h13':refmodels}.items():
        pp=[rates(c,m,tr,tn) for m in ms];p=np.mean(pp,axis=0);dv=dev(y,p)
        mean_scores[kind]={'deviance':float(dv.mean()),'mean_count_mae':float(np.abs(y-p).mean()),'parameters_per_seed':sum(x.numel() for x in ms[0].parameters())};daily_rate[kind]=daily(dv,days)
        perseed[kind]=[{'seed':s,'deviance':float(dev(y,x).mean()),'mean_count_mae':float(np.abs(y-x).mean())} for s,x in zip(SEEDS,pp)]
    rr,nnodes=tr[:32768],tn[:32768];yy=y[:32768];dd=days[:32768];dists={};dayenergy={}
    for name,b in {'poisson':{'kind':'poisson','flows':None,'epochs':[0,0]},**bundles}.items():
        e,report=distribution(c,models[selected],b['flows'],b['kind'],b['epochs'],rr,nnodes,yy);dists[name]=report;dayenergy[name]=daily(e,dd);emit(stage='test_distribution',name=name,**report)
    for name,kind,ff in [('frozen_h13_poisson','poisson',None),('frozen_h13_gaussian','gaussian',refflows)]:
        e,report=distribution(c,refmodels,ff,kind,[1,1],rr,nnodes,yy);dists[name]=report;dayenergy[name]=daily(e,dd)
    primary=interval(dayenergy[final],dayenergy['poisson']);gain=primary['relative_reduction'];active=any(bundles[final]['active']);mae_ok=dists[final]['median_count_mae']<=1.01*dists['poisson']['median_count_mae']
    guarded=bool(eligible) and active and mae_ok
    decision='supported' if guarded and gain>=.01 and all(primary[str(b)][0]>0 for b in (3,7)) else 'not_supported' if (not guarded or gain<.01) else 'inconclusive'
    comparisons={name:interval(dayenergy[name],dayenergy['poisson']) for name in dists if name!='poisson'}
    comparisons['gaussian_score_vs_mse']=interval(dayenergy['gaussian_score'],dayenergy['gaussian_mse'])
    diagnostics={}
    for name,steps in [('gaussian_mse',64),('count_jump_score',256)]:
        b=bundles[name];_,report=distribution(c,models[selected],b['flows'],b['kind'],b['epochs'],rr[:8192],nnodes[:8192],yy[:8192],steps=steps)
        _,original=distribution(c,models[selected],b['flows'],b['kind'],b['epochs'],rr[:8192],nnodes[:8192],yy[:8192])
        diagnostics[name]={'extra_steps':steps,'extra':report,'primary_steps_same_prefix':original,'selected_from_test':False}
    checkpoint_bytes={p.name:p.stat().st_size for p in out.glob('*.pt')}
    result={'hypothesis_id':'H-0024','started_at_utc':start,'finished_at_utc':now(),'code_commit':code,'plan_sha256':lock['plan_sha256'],'input_manifest_sha256':large.digest(mp),'environment':env,'api_opened':False,'scope':'Previously exposed March completed counts, not stock accuracy','kernel_checks':kernel_checks,'encoder_training':training,'encoder_validation':validation,'selected_encoder':selected,'eligible_compact_encoders':eligible,'encoder_parameter_reduction':1-params[selected]/params['full_retrain'],'flow_training':flow_training,'decoder_validation':val_dist,'flow_choices':lock['flow_choices'],'selected_fm':final,'mean_scores':mean_scores,'per_seed_mean_scores':perseed,'distributions':dists,'primary':{'decision':decision,'encoder_compact_eligible':bool(eligible),'selected_flow_active':active,'median_mae_guard':mae_ok,**primary},'comparisons':comparisons,'solver_diagnostics':diagnostics,'daily_energy':dayenergy,'daily_rate_deviance':daily_rate,'test_exposures':len(y),'distribution_exposures':len(yy),'test_dates':len(np.unique(days)),'seconds':time.perf_counter()-timer,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'checkpoint_bytes':checkpoint_bytes,'limitations':['Repeatedly exposed March,not untouched independent test','Completed counts only; no observed inventory/unmet demand/dispatch truth','Single historical split,two seeds;February halves share season','Count-FM adaptation,not full paper reproduction;no OT/CFG','Poisson epoch0 is no correction;not learned FM improvement','Nonlinear smaller models have different effective capacity;no causal mechanism claim','Observed lagged weather versus runtime missing-weather and net-flow/stock-delta mismatch remains','No automatic production change or API-based selection']}
    write(out/'results.json',result);emit(stage='study_complete',seconds=result['seconds'],encoder=selected,flow=final,primary=result['primary'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--source',required=True);p.add_argument('--h13',required=True);p.add_argument('--out',required=True);run(p.parse_args())
