"""H14 whole-cohort dependence templates with exact empirical margins."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy import sparse
import torch
from torch import nn
import historical_calendar as h
import rolling_inventory_model as parent
import rolling_inventory_data as data

M=128;SEEDS=(11,29);DIM=366
def tensor(x,device):return torch.as_tensor(x,dtype=torch.float32,device=device)
def configure():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.cuda.set_per_process_memory_fraction(.30)

class Features:
    def __init__(self,folder,device='cuda'):
        self.folder=Path(folder);self.device=device;self.c,self.ids,self.first,self.graph=parent.static_context(folder,device);self.n=len(self.ids)
        with np.load(self.folder/'time-features.npz',allow_pickle=False) as f:
            self.times=f['times'].copy();self.valid=f['valid'].copy();self.slot=f['slot'].copy()
            self.common=np.c_[f['calendar'],f['phase'],f['weather']].astype('float32')
        with np.load(self.folder/'dependence-layout.npz',allow_pickle=False) as f:self.group=f['group'].copy();self.edge=f['edge'].copy();self.edge_weight=f['edge_weight'].copy();self.scale=f['scale'].copy()
        self.groups=[np.flatnonzero(self.group==i) for i in range(int(self.group.max())+1)]
        self.y=np.load(self.folder/'counts.npy',mmap_mode='r',allow_pickle=False);self.models=[];self.enc=[]
        for seed in SEEDS:
            saved=torch.load(self.folder/f'recent-{seed}.pt',map_location='cpu',weights_only=True);model=parent.Forecaster().to(device);model.load_state_dict(saved['state_dict']);model.eval()
            self.models.append(model)
            with torch.no_grad():self.enc.append(model.encode(self.c))
        self.embedding=torch.cat([z for pair in self.enc for z in pair],1).detach()
        self.rates=None;self.pit=None
        if (self.folder/'anchor-rates.npy').exists():self.rates=np.load(self.folder/'anchor-rates.npy',mmap_mode='r',allow_pickle=False)
        if (self.folder/'pit-bounds.npy').exists():self.pit=np.load(self.folder/'pit-bounds.npy',mmap_mode='r',allow_pickle=False)
    def recent(self,rows):
        rows=np.asarray(rows);prev=np.maximum(rows-1,0);mask=(rows[:,None]>0)&self.valid[prev,None]&(self.times[prev,None]>=self.first)&(self.times[rows,None]-self.times[prev,None]==600)
        net=np.arcsinh(self.y[prev,:,1].astype('float32')-self.y[prev,:,0].astype('float32'))*mask
        frac=self.graph.dot(mask.T.astype('float32')).T;near=self.graph.dot(net.T).T/np.maximum(frac,1e-8)
        return np.stack((net,mask,near,frac),2).astype('float32')
    def anchor(self,slots,common,recent):
        c=self.c;b=len(slots);c.slot=torch.as_tensor(slots,device=self.device);c.calendar=tensor(common[:,:35],self.device);c.phase=tensor(common[:,35:38],self.device);c.weather=tensor(common[:,38:],self.device);c.recent=tensor(recent,self.device)
        rows=torch.arange(b,device=self.device).repeat_interleave(self.n);nodes=torch.arange(self.n,device=self.device).repeat(b)
        with torch.no_grad():values=[model.rate(c,rows,nodes,'recent',*enc).reshape(b,self.n,2) for model,enc in zip(self.models,self.enc)]
        return torch.cat(values,2).cpu().numpy()
    def from_inputs(self,slots,common,recent,rates):
        slot=torch.as_tensor(slots,device=self.device);b=len(slots);static=self.embedding[None].expand(b,-1,-1);common=tensor(common,self.device)[:,None].expand(-1,self.n,-1)
        rate=tensor(rates,self.device).reshape(b,self.n,2,2).mean(2)
        return torch.cat((static,self.c.log[slot],self.c.neighbor[slot],common,tensor(recent,self.device),torch.log1p(rate)),2)
    def batch(self,rows):
        rows=np.asarray(rows);rates=self.rates[rows].copy();recent=self.recent(rows)
        features=self.from_inputs(self.slot[rows],self.common[rows],recent,rates)
        mask=torch.as_tensor(self.times[rows,None]>=self.first[None],device=self.device)
        return features,mask
    def targets(self,rows,generator):
        bounds=tensor(self.pit[np.asarray(rows)].copy(),self.device);u=bounds[:,:,0]+torch.rand(bounds.shape[:2],device=self.device,generator=generator)*bounds[:,:,1]
        return math.sqrt(2)*torch.erfinv(2*u.clamp(1e-5,1-1e-5)-1)
    def operator(self,mask):
        if bool(mask.all()):return self.c.graph
        g=self.c.graph[None]*mask[:,None,:];return g/g.sum(2,keepdim=True).clamp_min(1e-8)
    def truth(self,row):return self.y[row,:,1].astype('int32')-self.y[row,:,0].astype('int32')

class Gaussian(nn.Module):
    def __init__(self,rank):
        super().__init__();self.rank=rank;self.net=nn.Sequential(nn.Linear(DIM,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,rank))
    def factors(self,c):
        factor=torch.tanh(self.net(c))*(.95/math.sqrt(self.rank));diagonal=torch.sqrt(1-factor.square().sum(-1));return factor,diagonal
    def nll(self,c,z,mask):
        factor,sd=self.factors(c);inv=mask/sd.square();b=(factor*(z*inv)[:,:,None]).sum(1)
        matrix=torch.eye(self.rank,device=c.device)[None]+torch.einsum('bnk,bnl,bn->bkl',factor,factor,inv)
        chol=torch.linalg.cholesky(matrix);solution=torch.cholesky_solve(b[:,:,None],chol)[:,:,0]
        value=(z.square()*inv+2*torch.log(sd)*mask).sum(1)-(b*solution).sum(1)+2*torch.log(chol.diagonal(dim1=-2,dim2=-1)).sum(1)
        return (.5*value/mask.sum(1).clamp_min(1)).mean()
    def sample(self,c,generator):
        factor,sd=self.factors(c);global_noise=torch.randn((len(c),self.rank),device=c.device,generator=generator);local=torch.randn(sd.shape,device=c.device,generator=generator)
        return (factor*global_noise[:,None,:]).sum(2)+sd*local

class GraphFlow(nn.Module):
    def __init__(self):
        super().__init__();self.condition=nn.Sequential(nn.Linear(DIM,64),nn.SiLU(),nn.Linear(64,64));self.stem=nn.Linear(5,64)
        self.blocks=nn.ModuleList(nn.ModuleList((nn.Linear(64,64),nn.Linear(64,64,bias=False),nn.Linear(64,64,bias=False),nn.LayerNorm(64))) for _ in range(4));self.output=nn.Linear(64,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def velocity(self,z,t,condition,mask,operator):
        tt=torch.ones_like(z)*t;clock=torch.stack((tt,torch.sin(math.pi*tt),torch.cos(math.pi*tt),tt.square()),2)
        value=nn.functional.silu(self.stem(torch.cat((z[:,:,None],clock),2))+condition)*mask[:,:,None]
        for own,near,global_layer,norm in self.blocks:
            pooled=(value*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            value=norm(value+nn.functional.silu(own(value)+near(torch.matmul(operator,value))+global_layer(pooled)[:,None]))*mask[:,:,None]
        return self.output(value)[:,:,0]*mask
    def loss(self,c,z0,z1,mask,operator,generator):
        t=torch.rand((len(c),1),device=c.device,generator=generator);z=(1-t)*z0+t*z1;prediction=self.velocity(z,t,self.condition(c),mask,operator)
        return ((prediction-(z1-z0)).square()*mask).sum()/mask.sum().clamp_min(1)
    def sample(self,z0,condition,mask,operator):
        z=z0
        for step in range(16):
            v=self.velocity(z,step/16,condition,mask,operator);trial=z+v/16
            z=z+(v+self.velocity(trial,(step+1)/16,condition,mask,operator))/32
        if not torch.isfinite(z).all():raise ValueError('Nonfinite joint flow sample.')
        return z

def reorder(common,template,tie_rng):
    common=np.asarray(common);template=np.asarray(template)
    if common.shape!=template.shape or not np.isfinite(template).all():raise ValueError('Invalid dependence template.')
    order=np.lexsort((tie_rng.random(template.shape),template),axis=1);rank=np.argsort(order,axis=1)
    out=np.take_along_axis(np.sort(common,axis=1),rank,axis=1)
    if not np.array_equal(np.sort(out,axis=1),np.sort(common,axis=1)):raise ValueError('Marginal multiset invariant failed.')
    return out

def marginal_draws(rates,rng):
    n=len(rates);choice=rng.integers(0,2,(n,M));r=np.where(choice==0,rates[:,0,None],rates[:,2,None]);a=np.where(choice==0,rates[:,1,None],rates[:,3,None])
    rentals=rng.poisson(r);returns=rng.poisson(a)
    if (rentals>65535).any() or (returns>65535).any():raise ValueError('Marginal count numerical range failure.')
    return (returns-rentals).astype('int32')

def rng_for(issue,stream):return np.random.default_rng(np.random.SeedSequence([20261104,int(issue)//600,stream]))

def generate(ctx,features,mask,rates,gaussians,flows,issue,include_shuffle=True):
    """One global seed choice per scenario; never split stations into chunks."""
    common=marginal_draws(rates,rng_for(issue,0));independent=reorder(common,rng_for(issue,1).normal(size=common.shape),rng_for(issue,10))
    choice=rng_for(issue,2).integers(0,len(gaussians),M);zg=np.zeros((ctx.n,M),dtype='float32');zf=zg.copy();operator=ctx.operator(mask)
    with torch.inference_mode():
        for k,gaussian in enumerate(gaussians):
            positions=np.flatnonzero(choice==k);generator=torch.Generator(device=ctx.device).manual_seed(20261104+int(issue)//600+100003*k)
            factor,sd=gaussian.factors(features);condition=flows[k].condition(features) if flows is not None else None
            for start in range(0,len(positions),16):
                pos=positions[start:start+16];b=len(pos)
                gn=torch.randn((b,gaussian.rank),device=ctx.device,generator=generator);local=torch.randn((b,ctx.n),device=ctx.device,generator=generator)
                z=(factor*gn[:,None,:]).sum(2)+sd*local;z=z*mask
                zg[:,pos]=z.cpu().numpy().T
                if flows is not None:zf[:,pos]=flows[k].sample(z,condition,mask.expand(b,-1),operator).cpu().numpy().T
    out={'independent':independent,'gaussian':reorder(common,zg,rng_for(issue,11))}
    if flows is not None:
        out['flow']=reorder(common,zf,rng_for(issue,11))
        if include_shuffle:
            shuffle=rng_for(issue,12);out['flow_shuffle']=np.stack([row[shuffle.permutation(M)] for row in out['flow']])
        for name,value in out.items():
            if not np.array_equal(np.sort(value,axis=1),np.sort(common,axis=1)):raise ValueError('Forecast marginal invariant violated: '+name)
    return out

def crps(samples,truth):
    x=np.sort(np.asarray(samples,dtype=float),axis=1);m=x.shape[1]
    return np.abs(x-np.asarray(truth)[:,None]).mean(1)-(x*(2*np.arange(1,m+1)-m-1)).sum(1)/(m*m)

def historical_scores(ctx,draws,truth,valid):
    groups=[g for g in ctx.groups if valid[g].all()];actual=np.array([np.maximum(-truth[g],0).mean() for g in groups]);out={}
    edge=ctx.edge;keep=valid[edge].all(1);edge=edge[keep];weight=ctx.edge_weight[keep];weight=weight/weight.sum()
    y=truth/ctx.scale;yd=np.sqrt(np.abs(y[edge[:,0]]-y[edge[:,1]]))
    for name,x in draws.items():
        u=np.array([np.maximum(-x[g],0).mean(0) for g in groups]);xs=x/ctx.scale[:,None]
        vd=np.sqrt(np.abs(xs[edge[:,0]]-xs[edge[:,1]])).mean(1)
        xx=xs[valid].T;yy=y[valid];sq=(xx*xx).sum(1);dist=np.sqrt(np.maximum(sq[:,None]+sq[None,:]-2*xx@xx.T,0))
        energy=(np.linalg.norm(xx-yy,axis=1).mean()-.5*dist.mean())/math.sqrt(int(valid.sum()))
        out[name]=dict(crps_sum=float(crps(u,actual).sum()),region_issues=len(groups),variogram=float(((vd-yd)**2*weight).sum()),energy=float(energy))
    return out
