"""H14 historical-only dependence fitting; API is not an accepted input."""
import argparse,json,time,hashlib,platform,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
from torch import nn
from scipy.stats import skellam
import historical_calendar as h
import rolling_inventory_data as dates
import joint_inventory_core as m
from forecast_transfer import cpu_state

ROOT=Path(__file__).resolve().parents[1]
def emit(**v):print(json.dumps(v,ensure_ascii=False,allow_nan=False),flush=True)
def cache(ctx):
    folder=ctx.folder
    if (folder/'cache-lock.json').exists():
        lock=json.loads((folder/'cache-lock.json').read_text(encoding='utf-8'))
        for item in lock['files']:
            if h.digest(folder/item['id'])!=item['sha256']:raise ValueError('Historical cache hash mismatch.')
        ctx.rates=np.load(folder/'anchor-rates.npy',mmap_mode='r');ctx.pit=np.load(folder/'pit-bounds.npy',mmap_mode='r');return lock
    rates=np.lib.format.open_memmap(folder/'anchor-rates.npy',mode='w+',dtype='float32',shape=(len(ctx.times),ctx.n,4));bounds=np.lib.format.open_memmap(folder/'pit-bounds.npy',mode='w+',dtype='float32',shape=(len(ctx.times),ctx.n,2));rates[:]=0;bounds[:]=0
    rows=np.flatnonzero(ctx.valid);started=time.perf_counter()
    for start in range(0,len(rows),16):
        rr=rows[start:start+16];value=ctx.anchor(ctx.slot[rr],ctx.common[rr],ctx.recent(rr));rates[rr]=value
        net=ctx.y[rr,:,1].astype('int32')-ctx.y[rr,:,0].astype('int32')
        lower=.5*(skellam.cdf(net-1,value[:,:,1],value[:,:,0])+skellam.cdf(net-1,value[:,:,3],value[:,:,2]))
        upper=.5*(skellam.cdf(net,value[:,:,1],value[:,:,0])+skellam.cdf(net,value[:,:,3],value[:,:,2]))
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or (upper<lower-1e-7).any():raise ValueError('Invalid signed-count CDF.')
        bounds[rr,:,0]=lower;bounds[rr,:,1]=np.maximum(upper-lower,0)
        if start%8192==0:emit(stage='joint_historical_cache',issue_times=start,total=len(rows),seconds=time.perf_counter()-started)
    rates.flush();bounds.flush();del rates,bounds
    result=dict(api_opened=False,issue_times=len(rows),seconds=time.perf_counter()-started,files=[dict(id=name,sha256=h.digest(folder/name)) for name in ('anchor-rates.npy','pit-bounds.npy')]);h.write(folder/'cache-lock.json',result)
    ctx.rates=np.load(folder/'anchor-rates.npy',mmap_mode='r');ctx.pit=np.load(folder/'pit-bounds.npy',mmap_mode='r');return result

def evaluate(ctx,issues,gaussians,flows):
    for model in gaussians:model.eval()
    for model in flows or []:model.eval()
    total={};timer=time.perf_counter()
    for row in issues:
        c,mask=ctx.batch([row]);draws=m.generate(ctx,c,mask,ctx.rates[row].copy(),gaussians,flows,int(ctx.times[row]));truth=ctx.truth(row)
        result=m.historical_scores(ctx,draws,truth,mask[0].cpu().numpy())
        for name,score in result.items():
            entry=total.setdefault(name,dict(crps_sum=0.,region_issues=0,variogram_sum=0.,energy_sum=0.,issues=0))
            entry['crps_sum']+=score['crps_sum'];entry['region_issues']+=score['region_issues'];entry['variogram_sum']+=score['variogram'];entry['energy_sum']+=score['energy'];entry['issues']+=1
    return {k:dict(regional_outflow_crps=v['crps_sum']/v['region_issues'],variogram=v['variogram_sum']/v['issues'],energy=v['energy_sum']/v['issues'],region_issues=v['region_issues'],issue_times=v['issues']) for k,v in total.items()},time.perf_counter()-timer

def fit_gaussian(ctx,rank,seed,selection,cfg,out,check_deadline):
    torch.manual_seed(seed);model=m.Gaussian(rank).to(ctx.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');trace=[];train_rows=np.flatnonzero(ctx.valid&(ctx.times<dates.TRAIN_END));timer=time.perf_counter()
    for epoch in range(cfg['gaussian_epochs']+1):
        check_deadline()
        if epoch:
            model.train();rng=np.random.default_rng(seed+epoch);rows=train_rows[rng.integers(0,len(train_rows),cfg['issue_samples_per_epoch'])];generator=torch.Generator(device=ctx.device).manual_seed(seed+epoch)
            for b in range(0,len(rows),16):
                rr=rows[b:b+16];c,mask=ctx.batch(rr);z=ctx.targets(rr,generator);loss=model.nll(c,z,mask)
                if not torch.isfinite(loss):raise ValueError('Gaussian dependence NLL nonfinite.')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        scores,seconds=evaluate(ctx,selection,[model],None);value=scores['gaussian']['regional_outflow_crps'];trace.append(dict(epoch=epoch,selection=scores,selection_seconds=seconds))
        if value<best:best=value;chosen=cpu_state(model);selected=epoch
        if epoch%4==0:emit(stage='joint_gaussian_epoch',rank=rank,seed=seed,epoch=epoch,selection_crps=value,seconds=time.perf_counter()-timer)
    model.load_state_dict(chosen);model.eval();torch.save(dict(state_dict=chosen,rank=rank,seed=seed,epoch=selected),out/f'gaussian-{rank}-{seed}.pt')
    return model,dict(rank=rank,seed=seed,selected_epoch=selected,selection_crps=best,trace=trace,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-timer)

def fit_flow(ctx,gaussian,seed,selection,cfg,out,check_deadline):
    torch.manual_seed(seed);model=m.GraphFlow().to(ctx.device);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);gaussian.eval();best=float('inf');trace=[];train_rows=np.flatnonzero(ctx.valid&(ctx.times<dates.TRAIN_END));timer=time.perf_counter()
    for epoch in range(cfg['flow_epochs']+1):
        check_deadline()
        if epoch:
            model.train();rng=np.random.default_rng(seed+epoch);rows=train_rows[rng.integers(0,len(train_rows),cfg['issue_samples_per_epoch'])];generator=torch.Generator(device=ctx.device).manual_seed(seed+epoch)
            for b in range(0,len(rows),16):
                rr=rows[b:b+16];c,mask=ctx.batch(rr);z1=ctx.targets(rr,generator)*mask
                with torch.no_grad():z0=gaussian.sample(c,generator)*mask
                loss=model.loss(c,z0,z1,mask,ctx.operator(mask),generator)
                if not torch.isfinite(loss):raise ValueError('Joint CFM loss nonfinite.')
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if epoch in cfg['flow_checkpoint_epochs']:
            scores,seconds=evaluate(ctx,selection,[gaussian],[model]);value=scores['flow']['regional_outflow_crps'];trace.append(dict(epoch=epoch,selection=scores,selection_seconds=seconds))
            if value<best:best=value;chosen=cpu_state(model);selected=epoch
            emit(stage='joint_flow_epoch',seed=seed,epoch=epoch,selection_crps=value,seconds=time.perf_counter()-timer)
    model.load_state_dict(chosen);model.eval();torch.save(dict(state_dict=chosen,seed=seed,epoch=selected),out/f'flow-{seed}.pt')
    return model,dict(seed=seed,selected_epoch=selected,selection_crps=best,trace=trace,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-timer)

def run(a):
    native_start=datetime.now(timezone.utc).isoformat();timer=time.perf_counter();reference_start=datetime.fromisoformat(a.collector_start);now=lambda:(reference_start+timedelta(seconds=time.perf_counter()-timer)).isoformat()
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    if (out/'fit-results.json').exists():raise ValueError('Preserve completed historical fit.')
    record=json.loads((ROOT/'research/hypotheses/H-0014.json').read_text(encoding='utf-8'));cfg_path=ROOT/'research/plans/H-0014-v1.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8'));manifest=ROOT/'research/datasets/tashu-dependence-flow-H0014-design-v1.json'
    canonical=json.dumps(record['plan'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256'] or h.digest(manifest)!=record['plan']['data']['input_manifest_sha256'] or h.digest(cfg_path)!=record['plan']['evaluation']['config_sha256']:raise ValueError('Frozen H14 protocol mismatch.')
    if reference_start.isoformat()<record['preregistration']['frozen_at_utc']:raise ValueError('Collector launch precedes plan freeze.')
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=json.loads(manifest.read_text(encoding='utf-8'));folder=Path(a.data)
    for item in source['derived_files']:
        if h.digest(folder/item['id'])!=item['sha256']:raise ValueError('Audited historical tensor differs: '+item['id'])
    m.configure();free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise ValueError('Less than2GiB GPU memory free; do not kill other jobs.')
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,device='dislab',gpu=torch.cuda.get_device_name(),free_gpu_bytes=free,total_gpu_bytes=total,memory_fraction=.30,threads=4,deterministic=True,tf32=False)
    h.write(out/'fit-start.json',dict(started_at_utc=reference_start.isoformat(),native_remote_started_at_utc=native_start,clock_basis='collector launch plus remote monotonic elapsed; remote OS clock retained separately',environment=environment,code_commit=code,api_opened=False))
    def deadline():
        if time.perf_counter()-timer>4*3600:raise TimeoutError('Frozen four-hour experiment budget exhausted; preserve partial artifacts.')
    ctx=m.Features(folder);cached=cache(ctx)
    with np.load(folder/'fixed-evaluation-issues.npz',allow_pickle=False) as f:selection=f['validation'].copy();auxiliary=f['auxiliary'].copy()
    gaussians={};training=[];rank_scores={}
    for rank in cfg['gaussian_ranks']:
        gaussians[rank]=[]
        for seed in m.SEEDS:
            model,trace=fit_gaussian(ctx,rank,seed,selection,cfg,out,deadline);gaussians[rank].append(model);training.append(trace)
        score,seconds=evaluate(ctx,selection,gaussians[rank],None);rank_scores[str(rank)]=dict(scores=score,seconds=seconds)
    chosen_rank=min(cfg['gaussian_ranks'],key=lambda rank:(rank_scores[str(rank)]['scores']['gaussian']['regional_outflow_crps'],rank));models=gaussians[chosen_rank];flows=[];flow_trace=[]
    for seed,gaussian in zip(m.SEEDS,models):
        model,trace=fit_flow(ctx,gaussian,seed,selection,cfg,out,deadline);flows.append(model);flow_trace.append(trace)
    selected,seconds=evaluate(ctx,selection,models,flows);controls=('independent','gaussian');gain={k:1-selected['flow']['regional_outflow_crps']/selected[k]['regional_outflow_crps'] for k in controls};gate=min(gain.values())>.01
    lock=dict(hypothesis_id='H-0014',locked_at_utc=now(),native_remote_clock=datetime.now(timezone.utc).isoformat(),api_opened=False,plan_sha256=record['preregistration']['plan_sha256'],input_manifest_sha256=h.digest(manifest),config_sha256=h.digest(cfg_path),code_commit=code,selected_rank=chosen_rank,historical_gate_passed=bool(gate),gate_relative_gain=gain,artifacts=[dict(id=p.name,sha256=h.digest(p)) for p in sorted(out.glob('*.pt'))],data_artifacts=source['derived_files'],cache_lock=cached,
        numerical_code=[dict(id=p.name,sha256=h.digest(p)) for p in [ROOT/'research_code/joint_inventory_core.py',ROOT/'research_code/joint_inventory_fit.py',ROOT/'research_code/joint_inventory_data.py']])
    h.write(out/'model-lock.json',lock);emit(stage='joint_models_locked',selected_rank=chosen_rank,historical_gate_passed=bool(gate),gate_relative_gain=gain)
    aux,aux_seconds=evaluate(ctx,auxiliary,models,flows)
    report=dict(hypothesis_id='H-0014',started_at_utc=reference_start.isoformat(),finished_at_utc=now(),native_remote_started_at_utc=native_start,native_remote_finished_at_utc=datetime.now(timezone.utc).isoformat(),clock_basis='collector launch plus remote monotonic elapsed; not synchronized remote OS',plan_sha256=lock['plan_sha256'],input_manifest_sha256=lock['input_manifest_sha256'],code_commit=code,environment=environment,cache=cached,gaussian_training=training,rank_selection=rank_scores,selected_rank=chosen_rank,flow_training=flow_trace,validation=selected,validation_seconds=seconds,auxiliary=aux,auxiliary_seconds=aux_seconds,historical_gate_passed=bool(gate),gate_relative_gain=gain,api_opened=False,peak_gpu_bytes=torch.cuda.max_memory_allocated(),seconds=time.perf_counter()-timer)
    h.write(out/'fit-results.json',report);emit(stage='joint_historical_complete',selected_rank=chosen_rank,historical_gate_passed=bool(gate),validation=selected,auxiliary=aux,seconds=report['seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);p.add_argument('--collector-start',required=True);run(p.parse_args())
