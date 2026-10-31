"""H11 local-GPU cross-city representation/count experiment. No inventory API path."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
import cross_city_data as data
import historical_calendar as hist

ROOT=Path(__file__).resolve().parents[1]
KINDS=('small_calendar','large_calendar','large_weather','large_weather_random','large_weather_pretrained')
SEEDS=(11,29)

def now():return datetime.now(timezone.utc).isoformat()
def emit(**kwargs):print(json.dumps(kwargs,ensure_ascii=False,allow_nan=False),flush=True)
def tensor(v,device):return torch.as_tensor(v,dtype=torch.float32,device=device)

class ProfileAE(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Sequential(nn.Linear(456,256),nn.SiLU(),nn.Linear(256,64))
        self.decoder=nn.Sequential(nn.Linear(64,256),nn.SiLU(),nn.Linear(256,444))
    def forward(self,x):return self.decoder(self.encoder(x))

def source_learning(source,out,device):
    started=time.perf_counter();s=np.load(source,allow_pickle=False);raw=s['profiles'];valid=s['validation'];train=~valid
    if train.sum()<100 or valid.sum()<20:raise ValueError('Insufficient source stations.')
    mean=raw[train].mean(0);std=np.maximum(raw[train].std(0),.1);x=tensor((raw-mean)/std,device)
    mask=tensor(np.c_[np.ones((len(raw),420)),np.repeat(raw[:,444:456],2,axis=1)],device)
    torch.manual_seed(20261101);model=ProfileAE().to(device);random=copy.deepcopy(model.encoder).eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    train_rows=np.flatnonzero(train);val_rows=torch.as_tensor(np.flatnonzero(valid),device=device);best=float('inf');trace=[];chosen=None
    for epoch in range(1,101):
        model.train();order=np.random.default_rng(20261101+epoch).permutation(train_rows)
        for start in range(0,len(order),256):
            ix=torch.as_tensor(order[start:start+256],device=device);xx=x[ix];noisy=xx*(torch.rand_like(xx)>=.2)
            loss=(((model(noisy)-xx[:,:444])**2)*mask[ix]).sum()/mask[ix].sum()
            opt.zero_grad();loss.backward();opt.step()
        model.eval()
        with torch.no_grad():score=float((((model(x[val_rows])-x[val_rows,:444])**2)*mask[val_rows]).sum()/mask[val_rows].sum())
        trace.append(dict(epoch=epoch,validation_mse=score))
        if score<best:best=score;chosen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=epoch
    model.load_state_dict(chosen);model.eval()
    encoders={};states={}
    for name,encoder in [('random',random),('pretrained',model.encoder)]:
        with torch.no_grad():z=encoder(x).cpu().numpy()
        zm=z[train].mean(0);zs=np.maximum(z[train].std(0),.1)
        encoders[name]=dict(encoder=encoder,mean=mean,std=std,zmean=zm,zstd=zs)
        states[name]=dict(state_dict={k:v.detach().cpu() for k,v in encoder.state_dict().items()},mean=mean,std=std,zmean=zm,zstd=zs)
    torch.save(states,out/'source_encoders.pt')
    report=dict(stations=len(raw),train=int(train.sum()),validation=int(valid.sum()),selected_epoch=selected,
        validation_mse=best,trace=trace,encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        autoencoder_parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started)
    hist.write(out/'source-learning.json',report);emit(stage='source_pretraining_complete',**{k:v for k,v in report.items() if k!='trace'})
    return encoders,report

def embedding(profiles,encoder,device):
    x=tensor((profiles-encoder['mean'])/encoder['std'],device)
    with torch.no_grad():z=encoder['encoder'](x).cpu().numpy()
    return ((z-encoder['zmean'])/encoder['zstd']).astype('float32')

class ForecastGCN(nn.Module):
    def __init__(self,width,depth):
        super().__init__();self.own=nn.Linear(109,width);self.neighbor=nn.Linear(70,width,bias=False)
        self.layers=nn.ModuleList(nn.Linear(width,width) for _ in range(depth-1));self.output=nn.Linear(width,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,own,neighbor,prior):
        h=nn.functional.silu(self.own(own)+self.neighbor(neighbor))
        for layer in self.layers:h=nn.functional.silu(layer(h))
        return prior*torch.exp(2*torch.tanh(self.output(h)/2))

class Context:
    def __init__(self,panel,fold,weather,embeddings,device):
        self.device=device;self.fold=fold;self.times=panel['times'];self.first=fold['first'];self.n=len(self.first)
        self.y=tensor(panel['counts'][:,fold['selected']],device)
        self.log=tensor(fold['log'],device);self.neighbor=tensor(fold['neighbor'],device);self.prior=tensor(fold['prior'],device)
        self.slot=torch.as_tensor(fold['slot'],device=device);self.weather=tensor(weather,device)
        a=data.axes(self.times)
        calendar=np.c_[np.eye(7)[a['weekday']],np.eye(15)[a['hour']-7],np.eye(2)[a['half']],
            np.sin(2*np.pi*a['month']/12),np.cos(2*np.pi*a['month']/12)].astype('float32')
        self.calendar=tensor(calendar,device);self.emb={};self.ne={}
        for k,v in embeddings.items():self.emb[k]=tensor(v,device);self.ne[k]=tensor(fold['graph'].dot(v),device)
    def features(self,rows,nodes,kind):
        slots=self.slot[rows];ownlog=self.log[slots,nodes];neighbor=self.neighbor[slots,nodes]
        weather=self.weather[rows] if kind not in KINDS[:2] else torch.zeros((len(rows),13),device=self.device)
        key='random' if kind.endswith('_random') else 'pretrained'
        use=kind.endswith(('_random','_pretrained'))
        e=self.emb[key][nodes] if use else torch.zeros((len(rows),64),device=self.device)
        ne=self.ne[key][nodes] if use else torch.zeros_like(e)
        return torch.cat((ownlog,self.calendar[rows],weather,e),1),torch.cat((neighbor,ne),1),self.prior[slots,nodes]
    def sample(self,seed,epoch):
        rng=np.random.default_rng(seed+epoch);times=np.flatnonzero(self.fold['train']);rs=[];ns=[];number=0
        while number<262144:
            size=262144-number;r=times[rng.integers(0,len(times),size+4096)];n=rng.integers(0,self.n,len(r))
            keep=self.times[r]>=self.first[n];r=r[keep][:size];n=n[keep][:size];rs.append(r);ns.append(n);number+=len(r)
        return np.concatenate(rs),np.concatenate(ns)
    def score(self,models,kind,split,daily=False):
        rows=np.flatnonzero(self.fold[split]);total=0.;absolute=0.;number=0;days={}
        for model in models:model.eval()
        # Flatten time/station pairs lazily, never materialize T*N*features.
        with torch.no_grad():
            for begin in range(0,len(rows)*self.n,32768):
                flat=np.arange(begin,min(begin+32768,len(rows)*self.n));r=rows[flat//self.n];n=flat%self.n
                keep=self.times[r]>=self.first[n];r=r[keep];n=n[keep]
                if not len(r):continue
                tr=torch.as_tensor(r,device=self.device);tn=torch.as_tensor(n,device=self.device)
                x,nb,p=self.features(tr,tn,kind);rate=torch.stack([m(x,nb,p) for m in models]).mean(0);y=self.y[tr,tn]
                dev=2*(torch.xlogy(y,y/rate)-y+rate);ab=(y-rate).abs()
                if not torch.isfinite(dev).all():raise ValueError('Nonfinite deviance.')
                total+=float(dev.sum());absolute+=float(ab.sum());number+=dev.numel()
                if daily:
                    day=(self.times[r]+32400)//86400;values=dev.sum(1).cpu().numpy()
                    for d in np.unique(day):
                        selected=day==d;entry=days.setdefault(int(d),[0.,0]);entry[0]+=float(values[selected].sum());entry[1]+=int(selected.sum())*2
        if not number:raise ValueError('Empty evaluation split.')
        return dict(deviance=total/number,mae=absolute/number,channel_exposures=number,daily=days)

def learn(ctx,kind,seed,out):
    torch.manual_seed(seed);small=kind=='small_calendar';model=ForecastGCN(32 if small else 256,1 if small else 3).to(ctx.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');trace=[];chosen=None;start=time.perf_counter()
    for epoch in range(13):
        if epoch:
            model.train();rows,nodes=ctx.sample(seed,epoch)
            for begin in range(0,len(rows),8192):
                r=torch.as_tensor(rows[begin:begin+8192],device=ctx.device);n=torch.as_tensor(nodes[begin:begin+8192],device=ctx.device)
                x,nb,p=ctx.features(r,n,kind);rate=model(x,nb,p);y=ctx.y[r,n]
                loss=(rate-y*torch.log(rate)).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        score=ctx.score([model],kind,'validation')['deviance'];trace.append(dict(epoch=epoch,validation_deviance=score))
        if score<best:best=score;chosen={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected=epoch
    model.load_state_dict(chosen);torch.save(dict(state_dict=chosen,kind=kind,seed=seed,epoch=selected),out/f'{kind}-{seed}.pt')
    report=dict(kind=kind,seed=seed,selected_epoch=selected,validation_deviance=best,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-start,trace=trace)
    emit(stage='target_model_complete',**{k:v for k,v in report.items() if k!='trace'});return model,report

def uncertainty(folds):
    rng=np.random.default_rng(20261101);controls=KINDS[1:4]
    macro={k:float(np.mean([f['test'][k]['deviance'] for f in folds])) for k in KINDS}
    gains={c:1-macro[KINDS[-1]]/macro[c] for c in controls};intervals=[]
    for block in (3,7):
        means=[]
        for f in folds:
            days=sorted(f['test'][KINDS[-1]]['daily']);n=len(days)
            if n<block:raise ValueError('Insufficient test dates.')
            values=np.array([[f['test'][k]['daily'][d][0] for k in KINDS]+[f['test'][KINDS[-1]]['daily'][d][1]] for d in days])
            starts=rng.integers(0,n-block+1,(2000,(n+block-1)//block));ix=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:n]
            sums=values[ix].sum(1);means.append(sums[:,:5]/sums[:,5,None])
        average=np.mean(means,axis=0);gain=np.min(1-average[:,4,None]/average[:,1:4],axis=1)
        intervals.append(dict(block_days=block,valid_replicates=int(np.isfinite(gain).sum()),interval=np.percentile(gain,[2.5,97.5]).tolist()))
    point=min(gains.values());enough=all(f['stations']>=500 and len(f['test'][KINDS[-1]]['daily'])>=10 for f in folds)
    decision='supported' if enough and point>.01 and all(x['interval'][0]>.01 for x in intervals) else ('not_supported' if enough and all(x['interval'][1]<=.01 for x in intervals) else 'inconclusive')
    return dict(macro_deviance=macro,relative_gain_by_control=gains,worst_control_macro_count_deviance_gain=point,intervals=intervals,decision=decision,sufficient=enough)

def run(args):
    start=now();clock=time.perf_counter();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    record=hist.load(ROOT/'research/hypotheses/H-0011.json') if hasattr(hist,'load') else json.loads((ROOT/'research/hypotheses/H-0011.json').read_text(encoding='utf-8'))
    manifest=ROOT/'research/datasets/tashu-cross-city-weather-H0011-v1.json';catalog=json.loads(manifest.read_text(encoding='utf-8'))
    if hist.digest(manifest)!=record['plan']['data']['input_manifest_sha256']:raise ValueError('Manifest changed.')
    for entry in catalog['sources']:
        path=(Path(args.history_dir) if entry['id'].startswith('historical-') else Path(args.source_dir))/entry['id']
        if path.stat().st_size!=entry['bytes'] or hist.digest(path)!=entry['sha256']:raise ValueError('Raw source hash differs.')
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA GPU required.')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise RuntimeError('Less than 2 GiB free GPU memory; no automatic budget change.')
    torch.cuda.set_per_process_memory_fraction(.55);device=torch.device('cuda')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    environment=dict(python=platform.python_version(),packages=dict(torch=torch.__version__,numpy=np.__version__,pandas=data.pd.__version__),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,initial_free_gpu_bytes=free,cpu_threads=4)
    hist.write(out/'started.json',dict(started_at_utc=start,code_commit=code,plan_sha256=record['preregistration']['plan_sha256'],environment=environment))
    # Cache entries belong to this exact manifest and implementation; never import older/API model artifacts.
    source=out/'source-profiles.npz'
    source_audit=data.source_profiles(Path(args.source_dir)/'seoul-trips-2024.zip',source)
    hist.write(out/'source-audit.json',source_audit)
    encoders,pretraining=source_learning(source,out,device)
    historical_catalog=json.loads((ROOT/'research/datasets/tashu-historical-H0004-v1.json').read_text(encoding='utf-8'))
    events,ids,months,quality=hist.read_events(Path(args.history_dir)/'historical-trips-20260331.zip',historical_catalog)
    panel=data.target_panel(events,ids,months);weather_raw=data.lag_weather([Path(args.source_dir)/f'daejeon-{y}-ghcnh.psv' for y in (2024,2025,2026)],panel['times'])
    hist.write(out/'historical-quality.json',quality);folds=[]
    for name,te,ve,ee in data.FOLDS:
        dest=out/name;dest.mkdir(exist_ok=True);fold=data.fold_data(panel,events,te,ve,ee)
        if len(fold['ids'])<500:raise ValueError('Insufficient fold training cohort.')
        weather,weather_scaling=data.normalize_weather(weather_raw,fold['train'])
        embeddings={k:embedding(fold['profiles'],v,device) for k,v in encoders.items()}
        np.savez_compressed(dest/'context.npz',station_ids=fold['ids'],first=fold['first'],prior=fold['prior'],log=fold['log'],neighbor=fold['neighbor'],random_embedding=embeddings['random'],pretrained_embedding=embeddings['pretrained'],graph_indptr=fold['graph'].indptr,graph_indices=fold['graph'].indices,graph_data=fold['graph'].data)
        hist.write(dest/'weather-scaling.json',weather_scaling)
        ctx=Context(panel,fold,weather,embeddings,device);models={};training=[]
        for kind in KINDS:
            models[kind]=[]
            for seed in SEEDS:
                model,trace=learn(ctx,kind,seed,dest);models[kind].append(model);training.append(trace)
        lock=dict(locked_at_utc=now(),artifacts=[dict(id=p.name,sha256=hist.digest(p)) for p in sorted(dest.iterdir()) if p.suffix in ('.pt','.npz','.json')])
        hist.write(dest/'pre-test-lock.json',lock)
        scores={kind:ctx.score(models[kind],kind,'test',daily=True) for kind in KINDS}
        result=dict(name=name,train_end=te,validation_end=ve,test_end=ee,stations=len(fold['ids']),training=training,test=scores,weather=weather_scaling)
        hist.write(dest/'results.json',result);folds.append(result)
        emit(stage='fold_test_complete',fold=name,stations=len(fold['ids']),deviance={k:v['deviance'] for k,v in scores.items()})
        del ctx,models,fold;torch.cuda.empty_cache()
    summary=uncertainty(folds)
    report=dict(hypothesis_id='H-0011',started_at_utc=start,finished_at_utc=now(),code_commit=code,plan_sha256=record['preregistration']['plan_sha256'],input_manifest_sha256=hist.digest(manifest),environment=environment,source_audit=source_audit,pretraining=pretraining,historical_quality=quality,folds=folds,summary=summary,seconds=time.perf_counter()-clock,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    hist.write(out/'results.json',report);emit(stage='experiment_complete',summary=summary,seconds=report['seconds'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-dir',required=True);parser.add_argument('--history-dir',required=True);parser.add_argument('--out',required=True);run(parser.parse_args())
