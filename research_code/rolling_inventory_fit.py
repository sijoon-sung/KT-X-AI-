"""H13 fit CLI. Historical-only inputs; no inventory/API loader or argument."""
import argparse,json,hashlib,time,platform,shutil,subprocess
from pathlib import Path
import numpy as np
import torch
from torch import nn
import historical_calendar as h
import rolling_inventory_data as d
import rolling_inventory_model as m
from forecast_transfer import cpu_state,now,emit

ROOT=Path(__file__).resolve().parents[1]

def mean_fit(c,kind,seed,source,folder):
    torch.manual_seed(seed);model=m.Forecaster().to(c.device)
    saved=torch.load(source,map_location='cpu',weights_only=True)
    state={k[len('encoder.'):]:v for k,v in saved['state_dict'].items() if k.startswith('encoder.')};model.encoder.load_state_dict(state)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);best=float('inf');trace=[];timer=time.perf_counter()
    for epoch in range(17):
        if epoch:
            model.train();rows,nodes=c.sample(seed,epoch,524288)
            for b in range(0,len(rows),8192):
                r=torch.as_tensor(rows[b:b+8192],device=c.device);n=torch.as_tensor(nodes[b:b+8192],device=c.device)
                rate=model.rate(c,r,n,kind);loss=(rate-c.y[r,n]*torch.log(rate)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite mean training loss.')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        value=c.score(model,kind,'validation')['deviance'];trace.append(dict(epoch=epoch,validation_deviance=value))
        if value<best:best=value;selected=epoch;chosen=cpu_state(model)
        if epoch%4==0:emit(stage='rolling_mean_epoch',kind=kind,seed=seed,epoch=epoch,validation_deviance=value)
    model.load_state_dict(chosen);model.eval();torch.save(dict(state_dict=chosen,seed=seed,kind=kind,epoch=selected),folder/f'{kind}-{seed}.pt')
    return model,dict(kind=kind,seed=seed,selected_epoch=selected,validation_deviance=best,trace=trace,parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-timer)

def flow_fit(c,encoder,seed,folder):
    torch.manual_seed(seed);flow=m.CountFlow(context_width=256,width=128).to(c.device);encoder.eval()
    optimizer=torch.optim.AdamW(flow.parameters(),lr=.001,weight_decay=.0001)
    vr,vn=c.sample(20261103,0,131072,'validation');vr=torch.as_tensor(vr,device=c.device);vn=torch.as_tensor(vn,device=c.device)
    with torch.no_grad():
        z=encoder.encode(c);rates,hidden=encoder.rate(c,vr,vn,'recent',*z,hidden=True);centers=torch.log1p(rates);labels=c.y[vr,vn]
    best=float('inf');trace=[];timer=time.perf_counter()
    for epoch in range(21):
        if epoch:
            flow.train();r,n=c.sample(seed,epoch,524288);generator=torch.Generator(device=c.device).manual_seed(seed+epoch)
            for b in range(0,len(r),8192):
                rr=torch.as_tensor(r[b:b+8192],device=c.device);nnodes=torch.as_tensor(n[b:b+8192],device=c.device)
                with torch.no_grad():rate,context=encoder.rate(c,rr,nnodes,'recent',*z,hidden=True);center=torch.log1p(rate)
                loss=m.flow_loss(flow,c.y[rr,nnodes],context,center,generator)
                if not torch.isfinite(loss):raise ValueError('Nonfinite CFM training loss.')
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(flow.parameters(),1.);optimizer.step()
        flow.eval();total=0.;generator=torch.Generator(device=c.device).manual_seed(20261103)
        with torch.no_grad():
            for b in range(0,len(vr),8192):total+=float(m.flow_loss(flow,labels[b:b+8192],hidden[b:b+8192],centers[b:b+8192],generator))*len(vr[b:b+8192])
        value=total/len(vr);trace.append(dict(epoch=epoch,validation_flow_loss=value))
        if value<best:best=value;selected=epoch;chosen=cpu_state(flow)
        if epoch%4==0:emit(stage='rolling_flow_epoch',seed=seed,epoch=epoch,validation_flow_loss=value)
    flow.load_state_dict(chosen);flow.eval();torch.save(dict(state_dict=chosen,seed=seed,epoch=selected),folder/f'flow-{seed}.pt')
    return flow,dict(seed=seed,selected_epoch=selected,validation_flow_loss=best,trace=trace,parameters=sum(p.numel() for p in flow.parameters()),seconds=time.perf_counter()-timer)

def auxiliary(c,models,flows):
    rows,nodes=c.sample(20261103,0,32768,'test');truth=c.y[torch.as_tensor(rows,device=c.device),torch.as_tensor(nodes,device=c.device)].cpu().numpy()
    out={};timer=time.perf_counter()
    with torch.no_grad():encoded={k:[model.encode(c) for model in models[k]] for k in m.KINDS}
    for label,kind,ff in [('no_recent_poisson','no_recent',None),('recent_poisson','recent',None),('recent_cfm','recent',flows)]:
        dev=0.;energy=0.;rng=np.random.default_rng(20261103);generator=torch.Generator(device=c.device).manual_seed(20261103)
        for b in range(0,len(rows),256):
            rr=torch.as_tensor(rows[b:b+256],device=c.device);nnodes=torch.as_tensor(nodes[b:b+256],device=c.device)
            draws=m.iid_counts(models[kind],ff,c,rr,nnodes,kind,20261103,encoded[kind],rng,generator);y=truth[b:b+256];rate=np.maximum(draws.mean(1),1e-6)
            from scipy.special import xlogy
            dev+=float((2*(xlogy(y,y/rate)-y+rate)).sum())
            energy+=float((np.linalg.norm(draws-y[:,None],axis=2).mean(1)-.5*np.linalg.norm(draws[:,:64]-draws[:,64:],axis=2).mean(1)).sum())
        out[label]=dict(count_deviance=dev/(2*len(rows)),count_energy=energy/len(rows),station_time_exposures=len(rows))
    return dict(scope='March historical completed counts, not measured inventory',models=out,seconds=time.perf_counter()-timer)

def run(a):
    began=now();timer=time.perf_counter();folder=Path(a.out);folder.mkdir(exist_ok=True,parents=True)
    record=json.loads((ROOT/'research/hypotheses/H-0013.json').read_text(encoding='utf-8'));mp=ROOT/'research/datasets/tashu-rolling-api-H0013-v1.json';manifest=json.loads(mp.read_text(encoding='utf-8'))
    canonical=json.dumps(record['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256'] or h.digest(mp)!=record['plan']['data']['input_manifest_sha256']:raise ValueError('Frozen H13 protocol changed.')
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source_map={p.name:p for p in Path(a.weather_dir).iterdir()}
    source_map.update({'historical-trips-20260331.zip':Path(a.history_dir)/'historical-trips-20260331.zip','H12-winter-historical-context.npz':Path(a.parent_dir)/'winter/context.npz'})
    source_map.update({f'H12-Seoul-source_forecast-{s}.pt':Path(a.parent_dir)/f'source/source_forecast-{s}.pt' for s in m.SEEDS})
    for item in manifest['sources']:
        if item['id']=='observed_panel.npz':continue # Evaluation-only input is deliberately not opened here.
        if h.digest(source_map[item['id']])!=item['sha256']:raise ValueError('Historical/weather source hash differs: '+item['id'])
    m.configure();free,total=torch.cuda.mem_get_info()
    if free<2*1024**3:raise ValueError('Less than 2GiB GPU memory available.')
    environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),device='local GPU',free_gpu_bytes=free,total_gpu_bytes=total,threads=4,memory_fraction=.55,tf32=False,deterministic=True)
    h.write(folder/'fit-start.json',dict(started_at_utc=began,code_commit=code,environment=environment,api_opened=False))
    audit=d.prepare(a.history_dir,a.parent_dir,a.weather_dir,folder);emit(stage='rolling_data_ready',**audit['panel'])
    ctx=m.Context(folder,'cuda');models={};training=[];flows=[];flow_training=[]
    for kind in m.KINDS:
        models[kind]=[]
        for seed in m.SEEDS:
            model,trace=mean_fit(ctx,kind,seed,Path(a.parent_dir)/f'source/source_forecast-{seed}.pt',folder);models[kind].append(model);training.append(trace)
    for model,seed in zip(models['recent'],m.SEEDS):
        flow,trace=flow_fit(ctx,model,seed,folder);flows.append(flow);flow_training.append(trace)
    # Lock all train-derived artifacts and exact numerical feature/generation code before external evaluation.
    artifacts=[dict(id=p.name,sha256=h.digest(p)) for p in sorted(folder.iterdir()) if p.suffix in ('.pt','.npz')]
    lock=dict(hypothesis_id='H-0013',locked_at_utc=now(),api_opened=False,plan_sha256=record['preregistration']['plan_sha256'],input_manifest_sha256=h.digest(mp),code_commit=code,
        code=[dict(id=p.name,sha256=h.digest(p)) for p in sorted((ROOT/'research_code').glob('*.py'))],artifacts=artifacts,
        particles=128,steps=16,projection='max(0,current+returns-rentals)',selection='February historical-only',model_variants=list(m.KINDS))
    h.write(folder/'model-lock.json',lock);emit(stage='rolling_models_locked',locked_at_utc=lock['locked_at_utc'])
    aux=auxiliary(ctx,models,flows)
    result=dict(hypothesis_id='H-0013',started_at_utc=began,finished_at_utc=now(),code_commit=code,plan_sha256=lock['plan_sha256'],input_manifest_sha256=h.digest(mp),environment=environment,data=audit,mean_training=training,flow_training=flow_training,auxiliary=aux,api_opened=False,seconds=time.perf_counter()-timer,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    h.write(folder/'fit-results.json',result);emit(stage='rolling_fit_complete',seconds=result['seconds'],auxiliary=aux)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--history-dir',required=True);p.add_argument('--parent-dir',required=True);p.add_argument('--weather-dir',required=True);p.add_argument('--out',required=True)
    run(p.parse_args())
