"""H22 historical-only fitting. Deliberately has no API path or loader."""
import argparse,hashlib,json,platform,shutil,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import large_inventory_model as m
from forecast_transfer import cpu_state,now,emit
from scaled_flow_core import flow_loss

ROOT=Path(__file__).resolve().parents[1]
def write(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def tensor(x,c): return torch.as_tensor(x,device=c.device)
def labels(c,r,n): return m.cpu_index(c.y,r,n)

def fit_gcn(c,seed,baseline,out,plan):
    torch.manual_seed(seed);model=m.LargeForecaster().to(c.device)
    model.encoder.load_state_dict({k[8:]:v for k,v in baseline.items() if k.startswith('encoder.')})
    initial=cpu_state(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    vr,vn=c.sample(20261103,0,131072,'validation');best=float('inf');trace=[]
    batch=plan['training']['batch_size'];began=time.perf_counter()
    for epoch in range(1,17):
        model.train();rr,nnodes=c.sample(seed,epoch,262144)
        for b in range(0,len(rr),batch):
            r=tensor(rr[b:b+batch],c);n=tensor(nnodes[b:b+batch],c)
            rate=model.rate(c,r,n,'recent');loss=(rate-labels(c,r,n)*torch.log(rate)).mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite GCN loss.')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        model.eval();total=0.
        with torch.no_grad():
            z=model.encode(c)
            for b in range(0,len(vr),batch):
                r=tensor(vr[b:b+batch],c);n=tensor(vn[b:b+batch],c);rate=model.rate(c,r,n,'recent',*z);y=labels(c,r,n)
                total+=float((2*(torch.xlogy(y,y/rate)-y+rate)).sum())
        value=total/(2*len(vr));trace.append({'epoch':epoch,'deviance':value})
        if value<best: best=value;chosen=cpu_state(model);selected=epoch
        emit(stage='H22_gcn',seed=seed,epoch=epoch,deviance=value,seconds=time.perf_counter()-began)
    model.load_state_dict(chosen);model.eval()
    changed={k:not torch.equal(v,initial[k]) for k,v in chosen.items()}
    if not all(changed.values()): raise ValueError('Unused/untrained GCN tensor: '+str([k for k,v in changed.items() if not v]))
    return model,{'seed':seed,'selected_epoch':selected,'validation_deviance':best,'trace':trace,'all_tensors_changed':True,'seconds':time.perf_counter()-began}

def fit_flow(c,model,seed,plan):
    torch.manual_seed(seed);flow=m.CachedFlow().to(c.device);initial=cpu_state(flow)
    opt=torch.optim.AdamW(flow.parameters(),lr=.0003,weight_decay=.0001)
    vr,vn=c.sample(20261103,0,131072,'validation');batch=plan['training']['batch_size']
    with torch.no_grad(): z=model.encode(c)
    best=float('inf');trace=[];began=time.perf_counter()
    for epoch in range(1,21):
        flow.train();rr,nnodes=c.sample(seed,epoch,262144)
        generator=torch.Generator(device=c.device).manual_seed(seed+epoch)
        for b in range(0,len(rr),batch):
            r=tensor(rr[b:b+batch],c);n=tensor(nnodes[b:b+batch],c)
            with torch.no_grad(): rate,context=model.rate(c,r,n,'recent',*z,hidden=True)
            loss=flow_loss(flow,labels(c,r,n),context,torch.log1p(rate),generator)
            if not torch.isfinite(loss): raise ValueError('Nonfinite flow loss.')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(flow.parameters(),1.);opt.step()
        flow.eval();total=0.;generator=torch.Generator(device=c.device).manual_seed(20261103)
        with torch.no_grad():
            for b in range(0,len(vr),batch):
                r=tensor(vr[b:b+batch],c);n=tensor(vn[b:b+batch],c)
                rate,context=model.rate(c,r,n,'recent',*z,hidden=True)
                total+=float(flow_loss(flow,labels(c,r,n),context,torch.log1p(rate),generator))*len(r)
        value=total/len(vr);trace.append({'epoch':epoch,'flow_loss':value})
        if value<best: best=value;chosen=cpu_state(flow);selected=epoch
        emit(stage='H22_flow',seed=seed,epoch=epoch,loss=value,seconds=time.perf_counter()-began)
    flow.load_state_dict(chosen);flow.eval()
    if not all(not torch.equal(v,initial[k]) for k,v in chosen.items()): raise ValueError('Untrained flow tensor.')
    return flow,{'seed':seed,'selected_epoch':selected,'validation_flow_loss':best,'trace':trace,'all_tensors_changed':True,'seconds':time.perf_counter()-began}

def run(a):
    record=json.loads((ROOT/'research/hypotheses/H-0022.json').read_text(encoding='utf-8'));plan=record['plan']
    canonical=json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256']: raise ValueError('Plan changed.')
    manifest=ROOT/'research/datasets/H22-historical-scale-v1.json'
    if m.digest(manifest)!=plan['data']['input_manifest_sha256']:raise ValueError('Manifest changed.')
    for name,sha in json.loads(manifest.read_text())['historical_artifacts'].items():
        if m.digest(Path(a.data)/name)!=sha:raise ValueError('Historical artifact differs.')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.40)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);began=now()
    env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),'device':'dislab GPU','free_gpu_bytes':torch.cuda.mem_get_info()[0],'precision':'FP32','tf32':False,'deterministic':True}
    write(out/'fit-start.json',{'started_at_utc':began,'code_commit':code,'environment':env,'api_opened':False,'counts_sha256':m.digest(Path(a.data)/'counts.npy')})
    c=m.historical_context(a.data,'cuda');reports=[]
    for seed in m.SEEDS:
        baseline=torch.load(Path(a.parent)/f'recent-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']
        model,gcn_report=fit_gcn(c,seed,baseline,out,plan)
        flow,flow_report=fit_flow(c,model,seed,plan)
        torch.save({'seed':seed,'gcn':cpu_state(model),'flow':cpu_state(flow),'gcn_epoch':gcn_report['selected_epoch'],'flow_epoch':flow_report['selected_epoch']},out/f'large-{seed}.pt')
        reports.append({'gcn':gcn_report,'flow':flow_report});write(out/'training.json',reports)
        del model,flow;torch.cuda.empty_cache()
    for name in ('context.npz','time-features.npz'): shutil.copy2(Path(a.data)/name,out/name)
    lock={'hypothesis_id':'H-0022','api_opened':False,'locked_at_utc':now(),'plan_sha256':record['preregistration']['plan_sha256'],'input_manifest_sha256':m.digest(manifest),'code_commit':code,'architecture':plan['architecture'],'artifacts':[{'id':p.name,'sha256':m.digest(p)} for p in sorted(out.iterdir()) if p.suffix in ('.pt','.npz')],'code':[{'id':p.name,'sha256':m.digest(p)} for p in sorted((ROOT/'research_code').glob('*.py'))]}
    write(out/'model-lock.json',lock)
    write(out/'fit-result.json',{'started_at_utc':began,'finished_at_utc':now(),'code_commit':code,'environment':env,'training':reports,'api_opened':False,'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
    emit(stage='H22_locked',lock_sha256=m.digest(out/'model-lock.json'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--parent',required=True);p.add_argument('--out',required=True);run(p.parse_args())
