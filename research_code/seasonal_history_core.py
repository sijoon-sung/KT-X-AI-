"""H10 recorded-flow seasonality. No API loading or fitting in this module."""
from datetime import datetime
import math
import numpy as np
import torch
from torch import nn
import historical_calendar as hist
import scaled_flow_core as slots

START=hist.stamp('2024-08-01')
TRAIN_END=hist.stamp('2026-02-01')
VALID_END=hist.stamp('2026-03-01')
END=hist.stamp('2026-04-01')
ALPHAS=(8,32,128)
SEEDS=(11,29)


def first_training_records(events,source_ids,ids,end=TRAIN_END):
    lookup={key:i for i,key in enumerate(ids)}
    mapping=np.array([lookup.get(x,-1) for x in source_ids])
    first=np.full(len(ids),np.iinfo(np.int64).max,dtype=np.int64)
    for tk,sk in [('rent','origin'),('returned','destination')]:
        node=mapping[events[sk]]
        keep=(node>=0)&(events[tk]<end)
        np.minimum.at(first,node[keep],events[tk][keep])
    if (first==np.iinfo(np.int64).max).any():raise ValueError('Cohort has no historical training record.')
    return first


def make_panel(events,source_ids,ids,months):
    hours=np.arange(START,END,3600,dtype=np.int64)
    times=(hours[:,None]+[0,1800]).ravel()
    counts=np.zeros((len(times),len(ids),2),dtype=np.int32)
    city=np.zeros((len(hours),2),dtype=np.int64)
    lookup={key:i for i,key in enumerate(ids)}
    mapping=np.array([lookup.get(x,-1) for x in source_ids])
    for channel,tk,sk in [(0,'rent','origin'),(1,'returned','destination')]:
        index=(events[tk]-START)//1800
        keep=(index>=0)&(index<len(times))
        np.add.at(city,(index[keep]//2,channel),1)
        node=mapping[events[sk]];keep &= node>=0
        np.add.at(counts,(index[keep],node[keep],channel),1)
    axes=hist.calendar_axes(hours)
    valid_hour=np.isin(axes['month'],list(months))&axes['interior']&(axes['hour']>=7)&(axes['hour']<=21)&(city>0).all(1)
    # Invalid out-of-scope hours receive a safe context index and are always masked.
    dates=[datetime.fromtimestamp(int(t),hist.KST) for t in times]
    context=np.array([int(slots.context_slot(d.month,d.weekday(),d.hour,int(d.minute>=30))) if 7<=d.hour<=21 else 0 for d in dates])
    first=first_training_records(events,source_ids,ids)
    return dict(times=times,counts=counts,valid=np.repeat(valid_hour,2),slots=context,first=first),dict(
        candidate_hours=len(hours),eligible_record_hours=int(valid_hour.sum()),
        missing_source_month_hours=int((~np.isin(axes['month'],list(months))).sum()),
        recorded_months=sorted(months),first_record_is_not_measured_operating_start=True)


def masks(data):
    t,v=data['times'],data['valid']
    return dict(train=v&(t<TRAIN_END),validation=v&(t>=TRAIN_END)&(t<VALID_END),test=v&(t>=VALID_END))


def sufficient_statistics(data):
    sums=np.zeros((slots.SLOTS,data['counts'].shape[1],2),dtype=np.float64)
    exposure=np.zeros(sums.shape[:2],dtype=np.float64)
    train=masks(data)['train']
    for slot in np.unique(data['slots'][train]):
        rows=np.flatnonzero(train&(data['slots']==slot))
        active=data['times'][rows,None]>=data['first'][None,:]
        exposure[slot]=active.sum(0)
        sums[slot]=(data['counts'][rows]*active[...,None]).sum(0)
    if not (exposure.sum(0)>0).all():raise ValueError('Empty training exposure.')
    return sums,exposure


def profiles(sums,exposure,alpha):
    if alpha not in ALPHAS or not np.isfinite(sums).all() or not np.isfinite(exposure).all() or (sums<0).any() or (exposure<0).any():raise ValueError('Invalid historical sufficient statistics.')
    if ((exposure==0)[...,None]&(sums>0)).any():raise ValueError('Count outside declared exposure.')
    base=np.maximum((sums.sum(0)+.5)/(exposure.sum(0)[:,None]+1),1e-6)
    weekly_sum=sums.reshape(12,210,*base.shape).sum(0)
    weekly_n=exposure.reshape(12,210,len(base)).sum(0)
    weekly=np.maximum((weekly_sum+alpha*base)/(weekly_n[...,None]+alpha),1e-6)
    pooled=np.broadcast_to(weekly[None],(12,*weekly.shape)).reshape(sums.shape).copy()
    seasonal=np.maximum((sums+alpha*pooled)/(exposure[...,None]+alpha),1e-6)
    return base,pooled,seasonal


def features(base,pooled,seasonal):
    out=np.empty((slots.SLOTS,len(base),32),dtype=np.float32)
    for slot in range(slots.SLOTS):
        month,weekday,hour,half=slots.decode_slot(slot)
        common=np.r_[np.eye(7)[weekday],np.eye(15)[hour-7],np.eye(2)[half],
            math.sin(2*math.pi*(month-1)/12),math.cos(2*math.pi*(month-1)/12)]
        out[slot,:,:2]=np.log1p(base)
        out[slot,:,2:4]=np.log1p(pooled[slot])
        out[slot,:,4:6]=np.log1p(seasonal[slot])
        out[slot,:,6:]=common
    return out


class SeasonalGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.own=nn.Linear(32,32)
        self.neighbor=nn.Linear(32,32,bias=False)
        self.out=nn.Linear(32,2)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,neighbor,prior):
        hidden=nn.functional.silu(self.own(x)+self.neighbor(neighbor))
        return prior*torch.exp(2*torch.tanh(self.out(hidden)/2))


def poisson_objective(rate,total,exposure):
    if not torch.isfinite(rate).all() or not (rate>0).all():raise ValueError('Invalid positive count mean.')
    return (exposure[...,None]*rate-total*torch.log(rate)).sum()/(2*exposure.sum())


def count_score(data,mask,means):
    if not mask.any():raise ValueError('No historical diagnostic observations.')
    total=0.;number=0
    for start in range(0,len(data['times']),256):
        rows=np.flatnonzero(mask[start:start+256])+start
        if not len(rows):continue
        score=hist.deviance(data['counts'][rows],means[data['slots'][rows]])
        total+=score.sum();number+=score.size
    return float(total/number)


def choose_profiles(sums,exposure,data):
    trace=[];fitted=[];valid=masks(data)['validation']
    for alpha in ALPHAS:
        base,pooled,seasonal=profiles(sums,exposure,alpha)
        trace.append(dict(alpha=alpha,pooled_deviance=count_score(data,valid,pooled),seasonal_deviance=count_score(data,valid,seasonal)))
        fitted.append((base,pooled,seasonal))
    pi=min(range(3),key=lambda i:trace[i]['pooled_deviance'])
    si=min(range(3),key=lambda i:trace[i]['seasonal_deviance'])
    return fitted[pi][1],fitted[si],dict(pooled_alpha=ALPHAS[pi],seasonal_alpha=ALPHAS[si],trace=trace)


def bootstrap(daily,block):
    rng=np.random.default_rng(20261031);days=len(daily)
    starts=rng.integers(0,days-block+1,(2000,(days+block-1)//block))
    ix=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:days]
    total=daily[ix].sum(1)
    valid=(total[:,:2]>0).all(1)&(total[:,-1]>0)
    gains=1-total[valid,2,None]/total[valid,:2]
    return dict(block_days=block,requested_replicates=2000,valid_replicates=int(valid.sum()),interval=np.percentile(gains.min(1),[2.5,97.5]).tolist())
