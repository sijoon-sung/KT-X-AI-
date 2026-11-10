"""Portable H24 selected count predictor. Run on dislab; no API or training.

Input own[N,100] is exactly: log-priors6,calendar35,phase3,weather54,recent2.
Input prior[N,2] is the fixed historical ten-minute rental/return prior.
This adapter does not fetch/construct features, estimate inventory or dispatch.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch import nn

class Rate(nn.Module):
    def __init__(self):
        super().__init__();self.own=nn.Linear(100,128);self.layers=nn.ModuleList(nn.Linear(128,128) for _ in range(2));self.output=nn.Linear(128,2)
    def forward(self,x,prior):
        h=nn.functional.silu(self.own(x))
        for layer in self.layers:h=nn.functional.silu(layer(h))
        return prior*torch.exp(2*torch.tanh(self.output(h)/2)),h

class Flow(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(136,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,2))
    def forward(self,z,t,h,center):
        t=t.expand(*z.shape[:-1],1);time=torch.cat((t,torch.sin(torch.pi*t),torch.cos(torch.pi*t),t*t),-1)
        return self.net(torch.cat((z,time,h,center),-1))

def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

class Predictor:
    def __init__(self,folder,device='cpu'):
        self.folder=Path(folder);self.device=torch.device(device)
        self.card=json.loads((self.folder/'model-card.json').read_text(encoding='utf-8'))
        if self.card['encoder_kind']!='compact_local' or self.card['flow_kind']!='poisson_continuous':raise ValueError('This adapter only accepts the exact H24 selected architecture')
        lock=json.loads((self.folder/'pre-test-lock.json').read_text(encoding='utf-8'))
        if lock.get('api_opened') is not False or lock.get('hypothesis_id')!='H-0024':raise ValueError('Missing historical-only lock')
        hashes={i['id']:i['sha256'] for i in lock['artifacts']};self.rates=[];self.flows=[];self.epochs=[]
        for seed in (11,29):
            rn=f'encoder-compact_local-{seed}.pt';fn=f'flow-poisson_continuous-score-{seed}.pt'
            for name in (rn,fn):
                if digest(self.folder/name)!=hashes[name]:raise ValueError('Checkpoint hash mismatch: '+name)
            saved=torch.load(self.folder/rn,map_location='cpu',weights_only=True)
            if saved['kind']!='compact_local' or saved['seed']!=seed:raise ValueError('Wrong rate checkpoint')
            m=Rate().to(self.device);m.load_state_dict(saved['state_dict']);m.eval();self.rates.append(m)
            saved=torch.load(self.folder/fn,map_location='cpu',weights_only=True)
            if saved['kind']!='poisson_continuous' or saved['seed']!=seed or saved['selector']!='score':raise ValueError('Wrong flow checkpoint')
            f=Flow().to(self.device);f.load_state_dict(saved['state_dict']);f.eval();self.flows.append(f);self.epochs.append(saved['epoch'])

    @torch.inference_mode()
    def predict(self,own,prior,seed=20261124):
        own=np.asarray(own,dtype=np.float32);prior=np.asarray(prior,dtype=np.float32)
        if own.ndim!=2 or own.shape[1]!=100 or prior.shape!=(len(own),2) or not len(own):raise ValueError('Expected own[N,100], prior[N,2], nonempty')
        if not np.isfinite(own).all() or not np.isfinite(prior).all() or (prior<=0).any():raise ValueError('Nonfinite inputs or nonpositive prior')
        outputs=[];means=[]
        for b in range(0,len(own),256):
            x=torch.as_tensor(own[b:b+256],device=self.device);p=torch.as_tensor(prior[b:b+256],device=self.device)
            pairs=[m(x,p) for m in self.rates];means.append(torch.stack([z[0] for z in pairs]).mean(0).cpu().numpy())
            choice=np.random.default_rng(seed+b).integers(0,2,(len(x),128));out=np.empty((len(x),128,2),np.int32)
            g=torch.Generator(device=self.device).manual_seed(seed+b)
            for i,(rate,h) in enumerate(pairs):
                rr,ss=np.where(choice==i);ix=torch.as_tensor(rr,device=self.device);rate=rate[ix];h=h[ix]
                counts=torch.poisson(rate,generator=g)
                if self.epochs[i]==0:
                    raw=counts.cpu().numpy()
                else:
                    center=torch.log1p(rate);u=torch.rand(rate.shape,device=self.device,generator=g);z=torch.log1p(counts+u)-center
                    for step in range(16):
                        t=z.new_tensor(step/16);v=self.flows[i](z,t,h,center);trial=z+v/16
                        z=z+(v+self.flows[i](trial,z.new_tensor((step+1)/16),h,center))/(2*16)
                    raw=np.expm1(z.cpu().numpy().astype(float)+center.cpu().numpy().astype(float))
                if not np.isfinite(raw).all() or (raw>65535).any():raise ValueError('Numerical range failure; no upper cap')
                out[rr,ss]=np.maximum(np.floor(raw),0).astype(np.int32)
            outputs.append(out)
        return {'mean_rate':np.concatenate(means),'count_samples':np.concatenate(outputs)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=20261124);a=p.parse_args()
    with np.load(a.features,allow_pickle=False) as f:own=f['own'];prior=f['prior']
    result=Predictor(a.model_dir,a.device).predict(own,prior,a.seed)
    np.savez_compressed(a.out,**result)
    print(json.dumps({'output':a.out,'stations':len(own),'particles':128,'channels':['rental','return'],'api_used':False}))

if __name__=='__main__':main()
