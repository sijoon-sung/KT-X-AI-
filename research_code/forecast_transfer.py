"""H12: supervised Seoul flow pretraining, historical-only target fine tuning."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,hashlib,json,platform,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
import cross_city_data as old
import cross_city_weather as previous
import forecast_transfer_data as data
import historical_calendar as hist

ROOT=Path(__file__).resolve().parents[1]
KINDS=('legacy_weather_scratch','weather_scratch','weather_ae','weather_forecast_encoder','weather_forecast_full')
SEEDS=(11,29)
def now():return datetime.now(timezone.utc).isoformat()
def emit(**kw):print(json.dumps(kw,ensure_ascii=False,allow_nan=False),flush=True)
def tensor(x,device):return torch.as_tensor(x,dtype=torch.float32,device=device)
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

class WeeklyEncoder(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(420,256),nn.SiLU(),nn.Linear(256,64),nn.LayerNorm(64))
    def forward(self,x):return self.net(x)

class Autoencoder(nn.Module):
    def __init__(self):super().__init__();self.encoder=WeeklyEncoder();self.decoder=nn.Sequential(nn.Linear(64,256),nn.SiLU(),nn.Linear(256,420))
    def forward(self,x):return self.decoder(self.encoder(x))

class Forecaster(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=WeeklyEncoder();self.own=nn.Linear(150,256);self.neighbor=nn.Linear(70,256,bias=False)
        self.layers=nn.ModuleList(nn.Linear(256,256) for _ in range(2));self.output=nn.Linear(256,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def encode(self,ctx):
        z=self.encoder(ctx.profiles);return z,ctx.graph@z
    def rate(self,ctx,rows,nodes,kind,z=None,nz=None):
        if z is None:z,nz=self.encode(ctx)
        slot=ctx.slot[rows];weather=ctx.weather[rows]
        if kind==KINDS[0]:weather=weather.clone();weather[:,13:]=0
        own=torch.cat((ctx.log[slot,nodes],ctx.calendar[rows],weather,z[nodes]),1)
        neighbor=torch.cat((ctx.neighbor[slot,nodes],nz[nodes]),1)
        h=nn.functional.silu(self.own(own)+self.neighbor(neighbor))
        for layer in self.layers:h=nn.functional.silu(layer(h))
        return ctx.prior[slot,nodes]*torch.exp(2*torch.tanh(self.output(h)/2))

class Context:
    def __init__(self,panel,fold,weather,device):
        self.fold=fold;self.device=device;self.times=panel['times'];self.first=fold['first'];self.n=len(self.first)
        self.y=tensor(panel['counts'][:,fold['selected']],device);self.profiles=tensor(fold['profiles'][:,:420],device)
        self.graph=tensor(fold['graph'].toarray(),device);self.log=tensor(fold['log'],device);self.neighbor=tensor(fold['neighbor'],device)
        self.prior=tensor(fold['prior'],device);self.slot=torch.as_tensor(fold['slot'],device=device);self.weather=tensor(weather,device)
        a=old.axes(self.times)
        cal=np.c_[np.eye(7)[a['weekday']],np.eye(15)[a['hour']-7],np.eye(2)[a['half']],np.sin(2*np.pi*a['month']/12),np.cos(2*np.pi*a['month']/12)]
        self.calendar=tensor(cal,device)
    def sample(self,seed,epoch,count):
        rng=np.random.default_rng(seed+epoch);times=np.flatnonzero(self.fold['train']);rs=[];ns=[];number=0
        while number<count:
            size=count-number;r=times[rng.integers(0,len(times),size+4096)];n=rng.integers(0,self.n,len(r));keep=self.times[r]>=self.first[n]
            r=r[keep][:size];n=n[keep][:size];rs.append(r);ns.append(n);number+=len(r)
        return np.concatenate(rs),np.concatenate(ns)
    def score(self,models,kind,split,daily=False):
        rows=np.flatnonzero(self.fold[split]);total=0.;ab=0.;number=0;days={}
        for model in models:model.eval()
        with torch.no_grad():
            enc=[m.encode(self) for m in models]
            for begin in range(0,len(rows)*self.n,32768):
                flat=np.arange(begin,min(begin+32768,len(rows)*self.n));r=rows[flat//self.n];n=flat%self.n;keep=self.times[r]>=self.first[n];r=r[keep];n=n[keep]
                if not len(r):continue
                tr=torch.as_tensor(r,device=self.device);tn=torch.as_tensor(n,device=self.device)
                rate=torch.stack([m.rate(self,tr,tn,kind,*z) for m,z in zip(models,enc)]).mean(0);y=self.y[tr,tn]
                dev=2*(torch.xlogy(y,y/rate)-y+rate)
                if not torch.isfinite(dev).all():raise ValueError('Nonfinite recorded-count score.')
                total+=float(dev.sum());ab+=float((y-rate).abs().sum());number+=dev.numel()
                if daily:
                    ds=(self.times[r]+32400)//86400;values=dev.sum(1).cpu().numpy()
                    for day in np.unique(ds):
                        selected=ds==day;entry=days.setdefault(int(day),[0.,0]);entry[0]+=float(values[selected].sum());entry[1]+=int(selected.sum())*2
        if not number:raise ValueError('Empty historical evaluation.')
        return dict(deviance=total/number,mae=ab/number,channel_exposures=number,daily=days)

def learn(ctx,seed,kind,folder,epochs,count,initial=None):
    torch.manual_seed(seed);model=Forecaster().to(ctx.device)
    if initial is not None:
        if kind in ('weather_ae','weather_forecast_encoder'):model.encoder.load_state_dict(initial)
        else:model.load_state_dict(initial)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');chosen=None;trace=[];timer=time.perf_counter()
    for epoch in range(epochs+1):
        if epoch:
            model.train();rows,nodes=ctx.sample(seed,epoch,count)
            for begin in range(0,count,8192):
                r=torch.as_tensor(rows[begin:begin+8192],device=ctx.device);n=torch.as_tensor(nodes[begin:begin+8192],device=ctx.device)
                rate=model.rate(ctx,r,n,kind);loss=(rate-ctx.y[r,n]*torch.log(rate)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss.')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        score=ctx.score([model],kind,'validation')['deviance'];trace.append(dict(epoch=epoch,validation_deviance=score))
        if score<best:best=score;selected=epoch;chosen=cpu_state(model)
        if epoch%4==0:emit(stage='forecast_epoch',kind=kind,seed=seed,epoch=epoch,validation_deviance=score,seconds=time.perf_counter()-timer)
    model.load_state_dict(chosen);torch.save(dict(state_dict=chosen,kind=kind,seed=seed,epoch=selected),folder/f'{kind}-{seed}.pt')
    report=dict(kind=kind,seed=seed,selected_epoch=selected,validation_deviance=best,trace=trace,parameters=sum(p.numel() for p in model.parameters()),encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),seconds=time.perf_counter()-timer)
    emit(stage='forecast_model_complete',**{k:v for k,v in report.items() if k!='trace'});return model,report

def learn_ae(profiles,ids,seed,folder,device):
    valid=np.array([int(hashlib.sha256(str(s).encode()).hexdigest(),16)%10==0 for s in ids]);train=np.flatnonzero(~valid);val=np.flatnonzero(valid)
    if len(train)<100 or len(val)<20:raise ValueError('Insufficient AE station split.')
    x=tensor(profiles[:,:420],device);vi=torch.as_tensor(val,device=device);torch.manual_seed(seed);model=Autoencoder().to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');trace=[];timer=time.perf_counter()
    for epoch in range(25):
        if epoch:
            rng=np.random.default_rng(seed+epoch);rows=train[rng.integers(0,len(train),1048576)];model.train()
            for begin in range(0,len(rows),8192):
                ix=torch.as_tensor(rows[begin:begin+8192],device=device);clean=x[ix];noisy=clean*(torch.rand_like(clean)>=.2)
                loss=((model(noisy)-clean)**2).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval()
        with torch.no_grad():score=float(((model(x[vi])-x[vi])**2).mean())
        trace.append(dict(epoch=epoch,validation_mse=score))
        if score<best:best=score;selected=epoch;chosen=cpu_state(model)
        if epoch%4==0:emit(stage='source_ae_epoch',seed=seed,epoch=epoch,validation_mse=score,seconds=time.perf_counter()-timer)
    model.load_state_dict(chosen);state=cpu_state(model.encoder);torch.save(dict(encoder=state,autoencoder=chosen,epoch=selected,seed=seed),folder/f'source_ae-{seed}.pt')
    report=dict(seed=seed,selected_epoch=selected,validation_mse=best,source_train_stations=len(train),source_validation_stations=len(val),parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-timer,trace=trace)
    return state,report

def weather_for(raw,city,times):
    if city=='seoul':primary=[raw/'seoul-2024-ghcnh.psv'];airport=[raw/'seoul-airport-2024-ghcnh.psv'];station=108
    else:primary=[raw/f'daejeon-{y}-ghcnh.psv' for y in (2024,2025,2026)];airport=[raw/f'cheongju-airport-{y}-ghcnh.psv' for y in (2024,2025,2026)];station=133
    return data.weather(primary,airport,sorted(raw.glob(f'kma-calendar-{station}-*.html')),station,times)

def coverage(x,mask):
    return dict(issue_times=int(mask.sum()),primary_thermo_present=x[mask,4:7].mean(0).tolist(),primary_3hour_precip_present=float(x[mask,7].mean()),
        primary_nominal_precip_present=float(x[mask,14].mean()),daily_numeric=float(x[mask,18].mean()),daily_dash=float(x[mask,19].mean()),
        airport_thermo_present=x[mask,24:27].mean(0).tolist(),primary_weather_code_present=float(x[mask,40].mean()),airport_weather_code_present=float(x[mask,52].mean()))

def save_context(folder,fold,city):
    g=fold['graph'];np.savez_compressed(folder/'context.npz',station_ids=fold['ids'],first=fold['first'],profiles=fold['profiles'][:,:420],
        prior=fold['prior'],log=fold['log'],neighbor=fold['neighbor'],graph_indptr=g.indptr,graph_indices=g.indices,graph_data=g.data)
    hist.write(folder/'context-metadata.json',dict(city=city,weather_dim=54,kind='recorded completed flows; not measured stock',train_end=int(fold['train_end'])))

def lock(folder):
    report=dict(locked_at_utc=now(),artifacts=[dict(id=p.name,sha256=hist.digest(p)) for p in sorted(folder.iterdir()) if p.is_file() and p.suffix in ('.pt','.npz','.json')])
    hist.write(folder/'pre-test-lock.json',report);return report

def uncertainty(folds):
    rng=np.random.default_rng(20261102);macro={k:float(np.mean([f['test'][k]['deviance'] for f in folds])) for k in KINDS}
    controls=('weather_scratch','weather_ae');gains={c:1-macro[KINDS[-1]]/macro[c] for c in controls};intervals=[]
    for block in (3,7):
        means=[]
        for f in folds:
            days=sorted(f['test'][KINDS[-1]]['daily']);n=len(days);values=np.array([[f['test'][k]['daily'][d][0] for k in (*controls,KINDS[-1])]+[f['test'][KINDS[-1]]['daily'][d][1]] for d in days])
            starts=rng.integers(0,n-block+1,(2000,(n+block-1)//block));ix=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:n]
            sums=values[ix].sum(1);means.append(sums[:,:3]/sums[:,3,None])
        average=np.mean(means,axis=0);g=np.min(1-average[:,2,None]/average[:,:2],axis=1)
        intervals.append(dict(block_days=block,valid_replicates=int(np.isfinite(g).sum()),interval=np.percentile(g,[2.5,97.5]).tolist()))
    point=min(gains.values());enough=all(f['stations']>=500 and len(f['test'][KINDS[-1]]['daily'])>=10 for f in folds)
    decision='supported' if enough and point>.01 and all(v['interval'][0]>.01 for v in intervals) else ('not_supported' if enough and all(v['interval'][1]<=.01 for v in intervals) else 'inconclusive')
    return dict(macro_deviance=macro,relative_gain_by_control=gains,worst_control_macro_count_deviance_gain=point,intervals=intervals,decision=decision,sufficient=enough)

def run(args):
    began=now();timer=time.perf_counter();raw=Path(args.source_dir);out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    record=json.loads((ROOT/'research/hypotheses/H-0012.json').read_text(encoding='utf-8'));manifest=ROOT/'research/datasets/tashu-forecast-transfer-H0012-v1.json'
    canonical=json.dumps(record['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256'] or hist.digest(manifest)!=record['plan']['data']['input_manifest_sha256']:raise ValueError('Frozen H12 contract changed.')
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    catalog=json.loads(manifest.read_text(encoding='utf-8'))
    if hist.digest(ROOT/catalog['source_catalog'])!=catalog['source_catalog_sha256']:raise ValueError('Historical member catalog differs.')
    for entry in catalog['sources']:
        folder=Path(args.history_dir) if entry['id'].startswith('historical-') else raw;path=folder/entry['id']
        if path.stat().st_size!=entry['bytes'] or hist.digest(path)!=entry['sha256']:raise ValueError('Input artifact hash differs.')
    if not torch.cuda.is_available():raise RuntimeError('Local CUDA GPU required.')
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise RuntimeError('At least 2GiB free GPU memory required.')
    torch.cuda.set_per_process_memory_fraction(.55);device=torch.device('cuda')
    environment=dict(python=platform.python_version(),packages=dict(torch=torch.__version__,numpy=np.__version__,pandas=old.pd.__version__),gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda,cpu_threads=4,initial_free_gpu_bytes=free)
    if list(out.rglob('*.pt')):raise ValueError('Preserve existing training; no implicit model rerun or checkpoint selection.')
    startfile=out/('started-resume.json' if args.resume_data else 'started.json')
    hist.write(startfile,dict(started_at_utc=began,code_commit=code,plan_sha256=record['preregistration']['plan_sha256'],environment=environment))
    cache=out/'source-months';cache.mkdir(exist_ok=True)
    cache_code={name:hist.digest(ROOT/'research_code'/name) for name in ('cross_city_data.py','forecast_transfer_data.py')}
    if args.resume_data:
        c=json.loads((out/'source-cache-lock.json').read_text(encoding='utf-8'))
        if c['input_manifest_sha256']!=hist.digest(manifest) or c['code_hashes']!=cache_code:raise ValueError('Source cache contract differs.')
        for a in c['artifacts']:
            if hist.digest(cache/a['id'])!=a['sha256']:raise ValueError('Cached source data changed.')
        quality=json.loads((out/'source-audit.json').read_text(encoding='utf-8'))
    else:
        if list(cache.glob('??.npz')):raise ValueError('Preserve existing preprocessing; use verified resume-data.')
        quality=old.source_profiles(raw/'seoul-trips-2024.zip',out/'unused-h11-profiles.npz',data.SourceCache(cache));hist.write(out/'source-audit.json',quality)
        hist.write(out/'source-cache-lock.json',dict(input_manifest_sha256=hist.digest(manifest),code_hashes=cache_code,artifacts=[dict(id=p.name,sha256=hist.digest(p)) for p in sorted(cache.glob('??.npz'))]))
    panel=data.source_panel(cache);fold=data.prepare_source_fold(panel)
    if len(fold['ids'])<500:raise ValueError('Insufficient source cohort.')
    src=out/'source';src.mkdir(exist_ok=True);wx=weather_for(raw,'seoul',panel['times']);save_context(src,fold,'seoul');ctx=Context(panel,fold,wx,device)
    ae={};states={};source_models=[];source_trace=[];ae_trace=[]
    for seed in SEEDS:
        ae[seed],trace=learn_ae(fold['profiles'],fold['ids'],seed,src,device);ae_trace.append(trace)
        model,trace=learn(ctx,seed,'source_forecast',src,24,1048576);source_models.append(model);states[seed]=cpu_state(model);source_trace.append(trace)
    source_lock=lock(src);source_test=ctx.score(source_models,'source_forecast','test',daily=True)
    if len(source_test['daily'])<10 or len(np.unique(panel['day'][fold['validation']]))<10:raise ValueError('Insufficient source evaluation dates.')
    source_report=dict(stations=len(fold['ids']),forecast_training=source_trace,ae_training=ae_trace,test=source_test,coverage={s:coverage(wx,fold[s]) for s in ('train','validation','test')})
    hist.write(src/'results.json',source_report);emit(stage='source_complete',stations=len(fold['ids']),test_deviance=source_test['deviance'])
    del ctx,panel,fold,source_models,wx;torch.cuda.empty_cache()
    historical=previous.historical_source_catalog();events,ids,months,target_quality=hist.read_events(Path(args.history_dir)/'historical-trips-20260331.zip',historical)
    panel=old.target_panel(events,ids,months);wx=weather_for(raw,'daejeon',panel['times']);folds=[]
    for name,te,ve,ee in old.FOLDS:
        dest=out/name;dest.mkdir(exist_ok=True);fold=old.fold_data(panel,events,te,ve,ee)
        if len(fold['ids'])<500:raise ValueError('Insufficient target training cohort.')
        save_context(dest,fold,'daejeon');ctx=Context(panel,fold,wx,device);models={};training=[]
        for kind in KINDS:
            models[kind]=[]
            for seed in SEEDS:
                initial=ae[seed] if kind=='weather_ae' else ({k[len('encoder.'):]:v for k,v in states[seed].items() if k.startswith('encoder.')} if kind=='weather_forecast_encoder' else (states[seed] if kind=='weather_forecast_full' else None))
                model,trace=learn(ctx,seed,kind,dest,16,524288,initial);models[kind].append(model);training.append(trace)
        pretest=lock(dest);scores={k:ctx.score(models[k],k,'test',daily=True) for k in KINDS}
        result=dict(name=name,train_end=te,validation_end=ve,test_end=ee,stations=len(fold['ids']),training=training,test=scores,coverage={s:coverage(wx,fold[s]) for s in ('train','validation','test')})
        hist.write(dest/'results.json',result);folds.append(result);emit(stage='target_fold_complete',fold=name,deviance={k:v['deviance'] for k,v in scores.items()})
        del ctx,models,fold;torch.cuda.empty_cache()
    summary=uncertainty(folds)
    report=dict(hypothesis_id='H-0012',started_at_utc=began,finished_at_utc=now(),plan_sha256=record['preregistration']['plan_sha256'],input_manifest_sha256=hist.digest(manifest),code_commit=code,environment=environment,source_quality=quality,source=source_report,target_quality=target_quality,folds=folds,summary=summary,seconds=time.perf_counter()-timer,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    hist.write(out/'results.json',report);emit(stage='experiment_complete',summary=summary,seconds=report['seconds'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-dir',required=True);ap.add_argument('--history-dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--resume-data',action='store_true');run(ap.parse_args())
