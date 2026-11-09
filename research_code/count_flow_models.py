"""H24 fixed-input compact encoders and conditional count-flow kernels.

Birth/death bridge and local sampler follow Wei & Pearson, arXiv2605.07746v3,
Eqs1/4 and Section2.2. Conditional Poisson source is a project adaptation.
No source/target minibatch OT, categorical count cap, CFG, or inventory labels.
"""
import math
import numpy as np
import torch
from torch import nn
import rolling_inventory_model as old
import large_inventory_model as large

TAU=.999

class CompactEncoder(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.hidden_width=128
        self.profile=kind=='compact_profile';self.use_graph=kind=='compact_graph'
        if self.profile:self.encoder=nn.Sequential(nn.Linear(420,128),nn.SiLU(),nn.Linear(128,64),nn.LayerNorm(64))
        self.own=nn.Linear(164 if self.profile else 100,128)
        if self.use_graph:self.neighbor=nn.Linear(8,128,bias=False)
        self.layers=nn.ModuleList(nn.Linear(128,128) for _ in range(2));self.output=nn.Linear(128,2)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def encode(self,c):return (self.encoder(c.profiles),None) if self.profile else (None,None)
    def rate(self,c,r,n,kind='recent',z=None,nz=None,hidden=False):
        recent=large.cpu_index(c.recent,r,n)
        parts=[c.log[c.slot[r],n],c.calendar[r],c.phase[r],c.weather[r]]
        if self.profile:
            if z is None:z=self.encoder(c.profiles)
            parts.append(z[n])
        parts.append(recent[:,:2]);x=torch.cat(parts,1);h=self.own(x)
        if self.use_graph:h=h+self.neighbor(torch.cat((c.neighbor[c.slot[r],n],recent[:,2:]),1))
        h=nn.functional.silu(h)
        for layer in self.layers:h=nn.functional.silu(layer(h))
        p=c.prior[c.slot[r],n]*torch.exp(2*torch.tanh(self.output(h)/2))
        return (p,h) if hidden else p

def inverse_softplus(x):return x+torch.log(-torch.expm1(-x))

class JumpFlow(nn.Module):
    def __init__(self,context_width):
        super().__init__();self.net=nn.Sequential(nn.Linear(context_width+8,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,4))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,t,h,center,prior):
        t=t.expand(*x.shape[:-1],1);tm=torch.cat((t,torch.sin(math.pi*t),torch.cos(math.pi*t),t*t),-1)
        raw=self.net(torch.cat((torch.log1p(x),tm,h,center),-1))
        birth=nn.functional.softplus(raw[:,:2]+inverse_softplus(prior))
        death=x*nn.functional.softplus(raw[:,2:]+inverse_softplus(prior.new_ones(prior.shape)))
        return birth,death

def bridge(x0,y,t,g):
    delta=y-x0
    return x0+torch.sign(delta)*torch.binomial(torch.abs(delta),t.expand_as(delta),generator=g)

def training_loss(flow,kind,y,h,p,g):
    center=torch.log1p(p)
    if kind=='gaussian':return old.flow_loss(flow,y,h,center,g)
    if kind=='poisson_continuous':
        x0=torch.poisson(p,generator=g);u0=torch.rand(y.shape,device=y.device,generator=g);u1=torch.rand(y.shape,device=y.device,generator=g)
        start=torch.log1p(x0+u0)-center;end=torch.log1p(y+u1)-center
        t=torch.rand((len(y),1),device=y.device,generator=g);z=(1-t)*start+t*end
        return ((flow(z,t,h,center)-(end-start))**2).mean()
    x0=torch.poisson(p,generator=g);t=torch.rand((len(y),1),device=y.device,generator=g)*TAU
    x=bridge(x0,y,t,g);target_birth=(y-x).clamp_min(0)/(1-t);target_death=(x-y).clamp_min(0)/(1-t)
    birth,death=flow(x,t,h,center,p)
    loss=birth-target_birth*torch.log(birth+1e-8)+death-target_death*torch.log(death+1e-8)
    return loss.mean()

@torch.no_grad()
def sample_flow(flow,kind,epoch,prior,h,g,steps=None):
    """Per-particle conditional generation. Epoch0 Poisson candidates bypass FM."""
    center=torch.log1p(prior)
    if kind=='poisson' or (kind in ('poisson_continuous','count_jump') and epoch==0):
        out=torch.poisson(prior,generator=g)
        if not torch.isfinite(out).all() or (out>65535).any():raise ValueError('Poisson numerical range')
        return out.cpu().numpy().astype(np.int32),0
    if kind=='count_jump':
        k=steps or 64;x=torch.poisson(prior,generator=g);dt=TAU/k
        for i in range(k):
            birth,death=flow(x,x.new_tensor(i*dt),h,center,prior);total=birth+death
            if not torch.isfinite(total).all():raise ValueError('Nonfinite transition rate')
            moved=-torch.expm1(-total*dt);pb=moved*birth/total.clamp_min(1e-12);pd=moved*death/total.clamp_min(1e-12)
            u=torch.rand(x.shape,device=x.device,generator=g)
            x=x+(u<pb).to(x.dtype)-((u>=pb)&(u<pb+pd)).to(x.dtype)
        if (x<0).any() or (x!=torch.floor(x)).any() or not torch.isfinite(x).all() or (x>65535).any():raise ValueError('Count jump boundary/range failure; no clipping')
        return x.cpu().numpy().astype(np.int32),0
    if kind=='gaussian':z=torch.randn(prior.shape,device=prior.device,generator=g)
    else:
        source=torch.poisson(prior,generator=g);u=torch.rand(prior.shape,device=prior.device,generator=g);z=torch.log1p(source+u)-center
    r=old.heun(flow,z,h,center,steps=steps or 16);raw=np.expm1(r.cpu().numpy().astype(float)+center.cpu().numpy().astype(float))
    if not np.isfinite(raw).all() or (raw>65535).any():raise ValueError('Continuous FM range; no cap')
    return np.maximum(np.floor(raw),0).astype(np.int32),int((raw<0).sum())

def kernel_checks(device):
    g=torch.Generator(device=device).manual_seed(20261124)
    p=torch.tensor([[.01,.2],[1.,3.]],device=device).repeat(1024,1);x=torch.poisson(p,generator=g)
    u=torch.rand(x.shape,device=device,generator=g)*.999+.0005
    restored=torch.floor(torch.expm1(torch.log1p(x+u))).clamp_min(0)
    if not torch.equal(x,restored):raise AssertionError('Poisson/dequant roundtrip')
    y=x+2
    if not torch.equal(bridge(x,y,x.new_zeros((len(x),1)),g),x):raise AssertionError('Bridge start')
    if not torch.equal(bridge(x,y,x.new_ones((len(x),1)),g),y):raise AssertionError('Bridge end')
    f=JumpFlow(128).to(device);h=torch.zeros((len(x),128),device=device)
    birth,death=f(x*0,x.new_tensor(.5),h,torch.log1p(p),p)
    if torch.count_nonzero(death)!=0 or not torch.allclose(birth,p,atol=1e-6,rtol=1e-5):raise AssertionError('Initial rates / zero death boundary')
    sampled,_=sample_flow(f,'count_jump',1,p,h,g,steps=8)
    if (sampled<0).any():raise AssertionError('Negative generated count')
    return {'poisson_dequant_integer_roundtrip':True,'bridge_both_endpoints':True,'zero_death_at_zero':True,'initial_birth_equals_poisson_rate':True,'generated_counts_nonnegative_integers':True}
