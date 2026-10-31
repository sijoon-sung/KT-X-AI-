"""H9 historical-only graph/context encoder and bivariate count Flow Matching."""
import csv,json,math
from datetime import datetime
from pathlib import Path
import numpy as np
from scipy.special import gammaln
import torch
from torch import nn
import historical_calendar as hist
import historical_od_gcn as old

SEEDS=(11,29)
KINDS=('small','large','large_seoul')
SLOTS=12*7*15*2


def context_slot(month,weekday,hour,half):
    return (((np.asarray(month)-1)*7+np.asarray(weekday))*15+np.asarray(hour)-7)*2+np.asarray(half)


def decode_slot(slot):
    half=slot%2;slot=slot//2;hour=slot%15+7;slot=slot//15
    return slot//7+1,slot%7,hour,half


def seoul_profiles(path):
    with Path(path).open(encoding='cp1252',newline='') as f:rows=list(csv.DictReader(f))
    data=[];keys=set()
    for r in rows:
        dt=datetime.strptime(r['Date'],'%d/%m/%Y');hour=int(r['Hour']);count=int(r['Rented Bike Count'])
        key=(dt.date(),hour)
        if key in keys or not 0<=hour<24 or count<0 or float(r['Rented Bike Count'])!=count:raise ValueError('Invalid Seoul source row.')
        keys.add(key)
        if dt.year>2018:raise ValueError('Unexpected contemporary Seoul observation.')
        if r['Functioning Day']=='Yes':data.append((dt.month,dt.weekday(),hour,count))
        elif r['Functioning Day']!='No':raise ValueError('Unknown source functioning flag.')
    source=np.array(data);mu=float(source[:,3].mean())
    if not mu>0:raise ValueError('Empty Seoul functioning hours.')
    dimensions=[(12,),(24,),(7,24),(12,24)];columns=[(0,),(2,),(1,2),(0,2)];taus=[24,24,8,8];tables=[]
    for shape,cols,tau in zip(dimensions,columns,taus):
        total=np.zeros(shape);number=np.zeros(shape)
        index=tuple(source[:,c]-(1 if c==0 else 0) for c in cols)
        np.add.at(total,index,source[:,3]);np.add.at(number,index,1)
        tables.append(np.log((total+tau*mu)/(number+tau)/mu))
    values=[]
    for slot in range(SLOTS):
        month,weekday,hour,half=decode_slot(slot)
        values.append([tables[0][month-1],tables[1][hour],tables[2][weekday,hour],tables[3][month-1,hour]])
    return np.asarray(values,dtype=np.float32),dict(source_rows=len(rows),functional_rows=len(data),excluded_nonfunctional=len(rows)-len(data),source_mean_rentals=mu)


def feature_table(constant,calendar,seoul,enabled):
    prior_x,prior=old.feature_grid(constant,calendar)
    out=np.empty((SLOTS,len(constant),34),dtype=np.float32);rate=np.empty((SLOTS,len(constant),2),dtype=np.float32)
    for slot in range(SLOTS):
        month,weekday,hour,half=decode_slot(slot);key=weekday*15+hour-7
        out[slot,:,:26]=prior_x[key]
        common=[math.sin(2*math.pi*(month-1)/12),math.cos(2*math.pi*(month-1)/12),float(half==0),float(half==1)]
        out[slot,:,26:30]=common;out[slot,:,30:]=seoul[slot] if enabled else 0
        rate[slot]=prior[key]
    return out,rate


class RateEncoder(nn.Module):
    def __init__(self,width=128,depth=3):
        super().__init__();self.width=width;self.depth=depth
        self.own=nn.Linear(34,width);self.neighbor=nn.Linear(34,width,bias=False)
        self.blocks=nn.ModuleList([nn.ModuleList([nn.Linear(width,width),nn.Linear(width,width,bias=False),nn.LayerNorm(width)]) for _ in range(depth-1)])
        self.out=nn.Linear(width,2);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def encode(self,x,operator):
        hidden=nn.functional.silu(self.own(x)+self.neighbor(torch.matmul(operator,x)))
        for own,neighbor,norm in self.blocks:hidden=norm(hidden+nn.functional.silu(own(hidden)+neighbor(torch.matmul(operator,hidden))))
        return hidden

    def forward(self,x,operator,prior,with_hidden=False):
        hidden=self.encode(x,operator);mean=prior*torch.exp(2*torch.tanh(self.out(hidden)/2))
        return (mean,hidden) if with_hidden else mean


class CountFlow(nn.Module):
    def __init__(self,context_width=128,width=128):
        super().__init__();self.net=nn.Sequential(nn.Linear(context_width+2+2+4,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)

    def forward(self,z,t,context,center):
        t=t.expand(*z.shape[:-1],1);time=torch.cat((t,torch.sin(math.pi*t),torch.cos(math.pi*t),t*t),dim=-1)
        return self.net(torch.cat((z,time,context,center),dim=-1))


def flow_loss(model,counts,context,center,generator):
    shape=counts.shape
    noise=torch.randn(shape,device=counts.device,generator=generator)
    u=torch.rand(shape,device=counts.device,generator=generator)
    target=torch.log1p(counts+u)-center
    t=torch.rand((*shape[:-1],1),device=counts.device,generator=generator)
    z=(1-t)*noise+t*target
    return ((model(z,t,context,center)-(target-noise))**2).mean()


def heun(model,z,context,center,steps=16):
    with torch.inference_mode():
        for step in range(steps):
            t=z.new_tensor(step/steps);velocity=model(z,t,context,center)
            trial=z+velocity/steps
            z=z+(velocity+model(trial,z.new_tensor((step+1)/steps),context,center))/(2*steps)
    return z


def integer_counts(residual,center):
    raw=np.expm1(np.asarray(residual,dtype=float)+np.asarray(center,dtype=float))
    if not np.isfinite(raw).all() or (raw>65535).any():raise ValueError('CFM generation numerical range failed; do not cap counts.')
    return np.maximum(np.floor(raw),0).astype(np.int32)


def log_depth_survival(rentals,returns,k):
    d,a=np.asarray(rentals),np.asarray(returns)
    result=gammaln(d+1.)+gammaln(a+1.)-gammaln(d-k+1.)-gammaln(a+k+1.)
    return np.where(k<=np.maximum(d-a,0),0.,np.where(k>d,-np.inf,result))


def reflected_depth(rentals,returns,rng):
    d,a=np.asarray(rentals,dtype=np.int64),np.asarray(returns,dtype=np.int64)
    if (d<0).any() or (a<0).any():raise ValueError('Negative event count.')
    low=np.maximum(d-a,0);high=d+1;logu=np.log(np.maximum(rng.random(d.shape),np.finfo(float).tiny))
    while np.any(high-low>1):
        active=high-low>1;mid=(low+high)//2;hit=logu<=log_depth_survival(d,a,mid)
        low=np.where(active&hit,mid,low);high=np.where(active&~hit,mid,high)
    return low.astype(np.int32)


def stock_samples(initial,delta,depth):
    b=np.asarray(initial)[...,None]
    return b+delta+np.maximum(depth-b,0)


def sample_summaries(samples,labels=None):
    samples=np.sort(np.asarray(samples),axis=-1);m=samples.shape[-1]
    if m<2 or not np.isfinite(samples).all() or (samples<0).any() or (samples!=np.floor(samples)).any():raise ValueError('Invalid stock ensemble.')
    quant=lambda p:samples[...,max(0,int(np.ceil(p*m))-1)]
    result=dict(mean=samples.mean(-1),median=quant(.5),lower=quant(.05),upper=quant(.95),pzero=(samples==0).mean(-1))
    if labels is not None:
        result['crps']=np.abs(samples-np.asarray(labels)[...,None]).mean(-1)-(samples*(2*np.arange(1,m+1)-m-1)).sum(-1)/(m*m)
    return result


def half_hour_panel(events,source_ids,trained_ids,hour_times,hour_y,hour_valid):
    lookup={key:i for i,key in enumerate(trained_ids)};mapping=np.array([lookup.get(x,-1) for x in source_ids])
    times=(hour_times[:,None]+np.array([0,1800])).ravel();counts=np.zeros((len(times),len(trained_ids),2),dtype=np.int32)
    for channel,tk,sk in [(0,'rent','origin'),(1,'returned','destination')]:
        index=(events[tk]-int(times[0]))//1800;col=mapping[events[sk]]
        valid=(index>=0)&(index<len(times))&(col>=0)
        np.add.at(counts,(index[valid],col[valid],channel),1)
    if not np.array_equal(counts.reshape(len(hour_times),2,len(trained_ids),2).sum(1)[hour_valid],hour_y[hour_valid]):
        raise ValueError('Half-hour completed counts do not reconcile with frozen H4 hourly counts.')
    valid=np.repeat(hour_valid,2);dates=[datetime.fromtimestamp(int(t),hist.KST) for t in times]
    slots=context_slot([d.month for d in dates],[d.weekday() for d in dates],[d.hour for d in dates],[int(d.minute>=30) for d in dates])
    return dict(counts=counts,times=times,valid=valid,slots=slots)


def load_encoder(path,device='cpu'):
    value=torch.load(path,map_location=device,weights_only=True)
    model=RateEncoder(value['width'],value['depth']).to(device);model.load_state_dict(value['state_dict']);model.eval()
    return model


def generate_counts(models,flows,x,operator,prior,slot,device='cuda',samples_per_seed=256,nodes=None):
    month,weekday,hour,half=decode_slot(int(slot));seed=20261109+month*1000+weekday*100+hour*2+half
    generator=torch.Generator(device=device).manual_seed(seed);parts=[]
    xx=torch.as_tensor(x[slot:slot+1],device=device);pp=torch.as_tensor(prior[slot:slot+1],device=device)
    with torch.inference_mode():
        for encoder,flow in zip(models,flows):
            rate,hidden=encoder(xx,operator,pp,with_hidden=True)
            context=hidden[0];center=torch.log1p(rate[0]/2)
            if nodes is not None:context,center=context[nodes],center[nodes]
            n=len(context);z=torch.randn((n,samples_per_seed,2),device=device,generator=generator).reshape(-1,2)
            result=np.empty((n*samples_per_seed,2),dtype=np.int32)
            for start in range(0,len(z),32768):
                end=min(start+32768,len(z));node=torch.arange(start,end,device=device)//samples_per_seed
                c=context[node];mu=center[node];out=heun(flow,z[start:end],c,mu)
                result[start:end]=integer_counts(out.cpu().numpy(),mu.cpu().numpy())
            parts.append(result.reshape(n,samples_per_seed,2))
    return np.concatenate(parts,axis=1),seed
