"""H22: trained ~100MB two-seed GCN/CFM, cached station conditioning.

Historical features and observation semantics are inherited unchanged from H13.
The learned condition MLP runs once per station, not per particle/ODE step.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import rolling_inventory_model as old
from scaled_flow_core import CountFlow, heun, integer_counts

SEEDS = (11, 29)
M = 128

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def cpu_index(value, rows, nodes):
    if value.device.type == 'cpu':
        return value[rows.cpu(), nodes.cpu()].to(rows.device)
    return value[rows, nodes]

class MemoryForecaster(old.Forecaster):
    """Same H13 state keys; CPU historical recent array, GPU feature minibatches."""
    def rate(self, c, rows, nodes, kind, z=None, nz=None, hidden=False):
        if z is None: z, nz = self.encode(c)
        slot = c.slot[rows]
        recent = cpu_index(c.recent, rows, nodes)
        if kind != 'recent': recent = recent * 0
        own = torch.cat((c.log[slot,nodes],c.calendar[rows],c.phase[rows],c.weather[rows],z[nodes],recent[:,:2]),1)
        near = torch.cat((c.neighbor[slot,nodes],nz[nodes],recent[:,2:]),1)
        value = nn.functional.silu(self.own(own)+self.neighbor(near))
        for layer in self.layers: value = nn.functional.silu(layer(value))
        rate = c.prior[slot,nodes]*torch.exp(2*torch.tanh(self.output(value)/2))
        return (rate,value) if hidden else rate

class LargeForecaster(MemoryForecaster):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = old.parent.WeeklyEncoder()
        self.own = nn.Linear(164,1280)
        self.neighbor = nn.Linear(72,1280,bias=False)
        self.layers = nn.ModuleList(nn.Linear(1280,1280) for _ in range(4))
        self.output = nn.Linear(1280,2)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

class CachedFlow(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[]
        for _ in range(3): blocks.extend((nn.Linear(1280,1280),nn.SiLU()))
        blocks.extend((nn.Linear(1280,128),nn.SiLU()))
        self.condition = nn.Sequential(*blocks)
        self.velocity = CountFlow(context_width=128,width=128)

    def forward(self,z,t,context,center):
        return self.velocity(z,t,self.condition(context),center)

def historical_context(folder,device):
    c=old.Context(folder,'cpu')
    for name in ('profiles','graph','log','neighbor','prior','calendar','phase','weather','slot'):
        setattr(c,name,getattr(c,name).to(device))
    c.device=device
    return c

def draws(models,flows,c,rows,nodes,seed,encoded=None):
    encoded=encoded or [model.encode(c) for model in models]
    rng=np.random.default_rng(seed)
    choice=rng.integers(0,len(models),(len(rows),M))
    out=np.empty((len(rows),M,2),dtype=np.int32)
    generator=torch.Generator(device=rows.device).manual_seed(seed)
    with torch.inference_mode():
        for i,(model,z) in enumerate(zip(models,encoded)):
            rate,hidden=model.rate(c,rows,nodes,'recent',*z,hidden=True)
            flow=None if flows is None else flows[i]
            if isinstance(flow,CachedFlow):
                hidden=flow.condition(hidden); flow=flow.velocity
            rr,ss=np.where(choice==i);ix=torch.as_tensor(rr,device=rows.device)
            center=torch.log1p(rate[ix])
            if flow is None: values=torch.poisson(rate[ix],generator=generator).cpu().numpy().astype(np.int32)
            else:
                noise=torch.randn((len(rr),2),device=rows.device,generator=generator)
                residual=heun(flow,noise,hidden[ix],center,steps=16)
                values=integer_counts(residual.cpu().numpy(),center.cpu().numpy())
            out[rr,ss]=values
    return out

class Predictor:
    def __init__(self,folder,device='cpu'):
        self.folder=Path(folder);self.device=device
        self.lock=json.loads((self.folder/'model-lock.json').read_text(encoding='utf-8'))
        if self.lock.get('hypothesis_id')!='H-0022' or self.lock.get('api_opened') is not False:
            raise ValueError('Historical-only H22 lock required.')
        for item in self.lock['artifacts']:
            if Path(item['id']).name!=item['id'] or digest(self.folder/item['id'])!=item['sha256']:
                raise ValueError('H22 artifact changed.')
        self.c,self.ids,self.first,self.graph=old.static_context(folder,device)
        self.n=len(self.ids);self.models={'recent':[]};self.flows=[]
        for seed in SEEDS:
            saved=torch.load(self.folder/f'large-{seed}.pt',map_location='cpu',weights_only=True)
            if saved['seed']!=seed: raise ValueError('Wrong seed.')
            model=LargeForecaster().to(device);model.load_state_dict(saved['gcn']);model.eval()
            flow=CachedFlow().to(device);flow.load_state_dict(saved['flow']);flow.eval()
            self.models['recent'].append(model);self.flows.append(flow)
        with torch.inference_mode(): self.encoded=[model.encode(self.c) for model in self.models['recent']]

    def issue(self,scheduled,current,valid,proxy,weather):
        if scheduled%600 or scheduled<old.d.VALID_END: raise ValueError('Future ten-minute boundary required.')
        if np.shape(current)!=(self.n,) or np.shape(valid)!=(self.n,) or np.shape(proxy)!=(self.n,4) or np.shape(weather)!=(54,):
            raise ValueError('Input schema mismatch.')
        if not np.isfinite(proxy).all() or not np.isfinite(weather).all(): raise ValueError('Nonfinite features.')
        valid=np.asarray(valid,bool)&(scheduled>=self.first)
        if not np.isfinite(current[valid]).all() or (current[valid]<0).any() or (current[valid]!=np.floor(current[valid])).any():
            raise ValueError('Invalid current inventory.')
        c=self.c;a=old.d.axes(np.array([scheduled]));c.slot=torch.as_tensor(a['slot'],device=self.device)
        for name in ('calendar','phase'): setattr(c,name,old.tensor(a[name],self.device))
        c.weather=old.tensor(weather[None],self.device);c.recent=old.tensor(proxy[None],self.device)
        ids=np.flatnonzero(valid);parts={k:[] for k in ('recent_poisson','recent_cfm')}
        for begin in range(0,len(ids),256):
            ns=ids[begin:begin+256];nodes=torch.as_tensor(ns,device=self.device)
            rows=torch.zeros(len(ns),dtype=torch.int64,device=self.device)
            for label,flows in (('recent_poisson',None),('recent_cfm',self.flows)):
                x=draws(self.models['recent'],flows,c,rows,nodes,20261103+scheduled//600+begin,self.encoded)
                parts[label].append(old.stock_projection(current[ns],x))
        return ids,{k:np.concatenate(v) if v else np.empty((0,M),dtype=np.int64) for k,v in parts.items()}
