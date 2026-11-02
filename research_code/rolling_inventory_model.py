"""H13 GCN count rates, required conditional Flow Matching, IID forecasts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from scipy import sparse
import historical_calendar as h
import forecast_transfer as parent
import rolling_inventory_data as d
from scaled_flow_core import CountFlow,flow_loss,heun,integer_counts

SEEDS=(11,29);KINDS=('no_recent','recent');M=128
def tensor(x,device):return torch.as_tensor(x,dtype=torch.float32,device=device)
def configure():
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.55)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)

class Forecaster(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=parent.WeeklyEncoder();self.own=nn.Linear(164,256);self.neighbor=nn.Linear(72,256,bias=False)
        self.layers=nn.ModuleList(nn.Linear(256,256) for _ in range(2));self.output=nn.Linear(256,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def encode(self,c):
        z=self.encoder(c.profiles);return z,c.graph@z
    def rate(self,c,rows,nodes,kind,z=None,nz=None,hidden=False):
        if z is None:z,nz=self.encode(c)
        slot=c.slot[rows];recent=c.recent[rows,nodes] if kind=='recent' else c.recent[rows,nodes]*0
        own=torch.cat((c.log[slot,nodes],c.calendar[rows],c.phase[rows],c.weather[rows],z[nodes],recent[:,:2]),1)
        near=torch.cat((c.neighbor[slot,nodes],nz[nodes],recent[:,2:]),1)
        value=nn.functional.silu(self.own(own)+self.neighbor(near))
        for layer in self.layers:value=nn.functional.silu(layer(value))
        rate=c.prior[slot,nodes]*torch.exp(2*torch.tanh(self.output(value)/2))
        return (rate,value) if hidden else rate

def static_context(folder,device):
    with np.load(Path(folder)/'context.npz',allow_pickle=False) as p:
        ids=p['station_ids'].copy();first=p['first'].copy();n=len(ids)
        graph=sparse.csr_matrix((p['graph_data'],p['graph_indices'],p['graph_indptr']),shape=(n,n))
        c=SimpleNamespace(profiles=tensor(p['profiles'],device),graph=tensor(graph.toarray(),device),log=tensor(p['log'],device),neighbor=tensor(p['neighbor'],device),prior=tensor(p['prior'],device))
    return c,ids,first,graph

class Context:
    def __init__(self,folder,device):
        self.device=device;c,self.ids,self.first,self.cpu_graph=static_context(folder,device);self.__dict__.update(vars(c));self.n=len(self.ids)
        with np.load(Path(folder)/'time-features.npz',allow_pickle=False) as f:
            self.times=f['times'].copy();self.valid=f['valid'].copy()
            for name in ('calendar','phase','weather'):setattr(self,name,tensor(f[name],device))
            self.slot=torch.as_tensor(f['slot'].copy(),device=device)
        raw=np.load(Path(folder)/'counts.npy',mmap_mode='r',allow_pickle=False)
        self.y=torch.empty(raw.shape,dtype=torch.float32,device=device)
        self.recent=torch.empty((len(raw),self.n,4),dtype=torch.float32,device=device)
        for begin in range(0,len(raw),256):
            end=min(len(raw),begin+256);self.y[begin:end]=tensor(raw[begin:end].copy(),device)
            self.recent[begin:end]=tensor(d.recent_from_counts(raw,self.times,self.valid,self.first,self.cpu_graph,begin,end),device)
        self.splits={s:self.valid&(self.times>=a)&(self.times<b) for s,a,b in [('train',d.START,d.TRAIN_END),('validation',d.TRAIN_END,d.VALID_END),('test',d.VALID_END,d.END)]}
    def sample(self,seed,epoch,count,split='train'):
        rng=np.random.default_rng(seed+epoch);times=np.flatnonzero(self.splits[split]);rs=[];ns=[];number=0
        while number<count:
            size=count-number;r=times[rng.integers(0,len(times),size+4096)];n=rng.integers(0,self.n,len(r));keep=self.times[r]>=self.first[n]
            r=r[keep][:size];n=n[keep][:size];rs.append(r);ns.append(n);number+=len(r)
        return np.concatenate(rs),np.concatenate(ns)
    def score(self,model,kind,split):
        model.eval();rows=np.flatnonzero(self.splits[split]);total=0.;number=0
        with torch.no_grad():
            z=model.encode(self)
            for start in range(0,len(rows)*self.n,32768):
                flat=np.arange(start,min(start+32768,len(rows)*self.n));r=rows[flat//self.n];n=flat%self.n;keep=self.times[r]>=self.first[n];r=r[keep];n=n[keep]
                if not len(r):continue
                tr=torch.as_tensor(r,device=self.device);tn=torch.as_tensor(n,device=self.device);rate=model.rate(self,tr,tn,kind,*z);y=self.y[tr,tn]
                dev=2*(torch.xlogy(y,y/rate)-y+rate)
                if not torch.isfinite(dev).all():raise ValueError('Nonfinite validation count score.')
                total+=float(dev.sum());number+=dev.numel()
        return dict(deviance=total/number,channel_exposures=number)

def iid_counts(models,flows,c,rows,nodes,kind,seed,encoded=None,rng=None,generator=None):
    """Independently choose seed model for every particle, then sample its law."""
    if encoded is None:encoded=[m.encode(c) for m in models]
    if rng is None:rng=np.random.default_rng(seed)
    choice=rng.integers(0,2,(len(rows),M));out=np.empty((len(rows),M,2),dtype='int32')
    if generator is None:generator=torch.Generator(device=rows.device).manual_seed(int(seed))
    with torch.no_grad():
        pairs=[m.rate(c,rows,nodes,kind,*z,hidden=True) for m,z in zip(models,encoded)]
        for i,(rate,hidden) in enumerate(pairs):
            rr,ss=np.where(choice==i);ix=torch.as_tensor(rr,device=rows.device);center=torch.log1p(rate[ix])
            if flows is None:
                values=torch.poisson(rate[ix],generator=generator).cpu().numpy()
                if not np.isfinite(values).all() or (values>65535).any():raise ValueError('Poisson numerical range failure.')
                values=values.astype('int32')
            else:
                z=torch.randn((len(rr),2),device=rows.device,generator=generator)
                residual=heun(flows[i],z,hidden[ix],center)
                values=integer_counts(residual.cpu().numpy(),center.cpu().numpy())
            out[rr,ss]=values
    return out

def stock_projection(current,counts):
    c=np.asarray(counts,dtype='int64');return np.maximum(0,np.asarray(current,dtype='int64')[:,None]+c[:,:,1]-c[:,:,0])

class Predictor:
    def __init__(self,folder,device='cuda'):
        self.folder=Path(folder);self.device=device
        lock=json.loads((self.folder/'model-lock.json').read_text(encoding='utf-8'))
        if lock.get('api_opened') is not False:raise ValueError('Missing historical-only fit lock.')
        for item in lock['artifacts']:
            if h.digest(self.folder/item['id'])!=item['sha256']:raise ValueError('Frozen model artifact changed.')
        self.lock=lock;self.c,self.ids,self.first,self.graph=static_context(folder,device);self.n=len(self.ids)
        self.models={};self.encoded={};self.flows=[]
        for kind in KINDS:
            self.models[kind]=[];self.encoded[kind]=[]
            for seed in SEEDS:
                saved=torch.load(self.folder/f'{kind}-{seed}.pt',map_location='cpu',weights_only=True)
                if saved['kind']!=kind or saved['seed']!=seed:raise ValueError('Mean checkpoint identity mismatch.')
                model=Forecaster().to(device);model.load_state_dict(saved['state_dict']);model.eval();self.models[kind].append(model)
                with torch.no_grad():self.encoded[kind].append(model.encode(self.c))
        for seed in SEEDS:
            saved=torch.load(self.folder/f'flow-{seed}.pt',map_location='cpu',weights_only=True);flow=CountFlow(context_width=256,width=128).to(device)
            flow.load_state_dict(saved['state_dict']);flow.eval();self.flows.append(flow)
    def issue(self,scheduled,current,valid,proxy,weather):
        if scheduled%600 or scheduled<d.VALID_END:raise ValueError('Forecast requires future 10-minute boundary.')
        if np.shape(current)!=(self.n,) or np.shape(valid)!=(self.n,) or np.shape(proxy)!=(self.n,4) or np.shape(weather)!=(54,):raise ValueError('Forecast input schema mismatch.')
        if not np.isfinite(proxy).all() or not np.isfinite(weather).all():raise ValueError('Nonfinite forecast features.')
        valid=np.asarray(valid,bool)&(scheduled>=self.first)
        if not np.isfinite(current[valid]).all() or (current[valid]<0).any() or (current[valid]!=np.floor(current[valid])).any():raise ValueError('Invalid observed current inventory.')
        a=d.axes(np.array([scheduled]));c=self.c;c.slot=torch.as_tensor(a['slot'],device=self.device)
        for name in ('calendar','phase'):setattr(c,name,tensor(a[name],self.device))
        c.weather=tensor(weather[None],self.device);c.recent=tensor(proxy[None],self.device)
        ids=np.flatnonzero(valid);parts={k:[] for k in ('no_recent_poisson','recent_poisson','recent_cfm')}
        seed=20261103+scheduled//600
        streams={k:(np.random.default_rng(seed),torch.Generator(device=self.device).manual_seed(seed)) for k in parts}
        for begin in range(0,len(ids),256):
            ns=ids[begin:begin+256];nodes=torch.as_tensor(ns,device=self.device);rows=torch.zeros(len(ns),dtype=torch.int64,device=self.device)
            for label,kind,flow in [('no_recent_poisson','no_recent',None),('recent_poisson','recent',None),('recent_cfm','recent',self.flows)]:
                rng,generator=streams[label]
                draws=iid_counts(self.models[kind],flow,c,rows,nodes,kind,seed,self.encoded[kind],rng,generator)
                parts[label].append(stock_projection(current[ns],draws))
        return ids,{k:np.concatenate(v) if v else np.empty((0,M),dtype='int64') for k,v in parts.items()}
