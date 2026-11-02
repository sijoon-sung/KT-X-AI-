"""H13 historical-only 24-hour/10-minute labels and fixed past-only context.

This module accepts no inventory API path. Zero labels only have meaning inside
declared historical source support; they are not availability or latent demand.
"""
from pathlib import Path
import json,calendar
from datetime import datetime
import numpy as np
from scipy import sparse
import historical_calendar as h
import cross_city_weather as previous
import forecast_transfer as parent

START=h.stamp('2024-08-01');TRAIN_END=h.stamp('2026-02-01')
VALID_END=h.stamp('2026-03-01');END=h.stamp('2026-04-01');SLOTS=4032

def axes(times):
    dates=[datetime.fromtimestamp(int(t),h.KST) for t in times]
    month=np.array([d.month-1 for d in dates]);weekday=np.array([d.weekday() for d in dates])
    hour=np.array([d.hour for d in dates]);half=np.array([d.minute//30 for d in dates]);phase=np.array([d.minute%30//10 for d in dates])
    weekly=(weekday*24+hour)*2+half
    return dict(month=month,weekly=weekly,slot=month*336+weekly,
        calendar=np.c_[np.eye(7)[weekday],np.eye(24)[hour],np.eye(2)[half],np.sin(2*np.pi*month/12),np.cos(2*np.pi*month/12)].astype('float32'),
        phase=np.eye(3,dtype='float32')[phase],day=(np.asarray(times)+32400)//86400,
        interior=np.array([1<d.day<calendar.monthrange(d.year,d.month)[1] for d in dates]),
        month_string=np.array([d.strftime('%Y-%m') for d in dates]))

def aggregate(events,source_ids,trained_ids,months,start=START,end=END):
    times=np.arange(start,end,600,dtype='int64');n=len(trained_ids)
    counts=np.zeros((len(times),n,2),dtype='uint16');city=np.zeros((len(times)//6,2),dtype='int64')
    lookup={s:i for i,s in enumerate(trained_ids)};mapping=np.array([lookup.get(s,-1) for s in source_ids])
    reconciled=[]
    for ch,tk,sk in ((0,'rent','origin'),(1,'returned','destination')):
        ix=(events[tk]-start)//600;inside=(ix>=0)&(ix<len(times));node=mapping[events[sk]];keep=inside&(node>=0)
        # Accumulate in uint32 first so a corrupted record pile-up cannot wrap uint16.
        flat=ix[keep]*n+node[keep];b=np.bincount(flat,minlength=len(times)*n).reshape(len(times),n)
        if b.max()>65535:raise ValueError('Historical count overflows declared storage.')
        counts[:,:,ch]=b;del b
        city[:,ch]=np.bincount(ix[inside]//6,minlength=len(city))
        thirty=np.bincount((ix[keep]//3)*n+node[keep],minlength=(len(times)//3)*n).reshape(-1,n)
        reconciled.append(bool(np.array_equal(counts[:,:,ch].reshape(-1,3,n).sum(1),thirty)))
        del thirty
    a=axes(times);valid=a['interior']&np.isin(a['month_string'],sorted(months))&np.repeat((city>0).all(1),6)
    if not all(reconciled):raise ValueError('10-minute/30-minute historical counts do not reconcile.')
    return dict(times=times,counts=counts,valid=valid,**a),dict(candidate_slots=len(times),supported_slots=int(valid.sum()),count10_to30_reconciled=reconciled)

def priors(panel,first,graph):
    n=len(first);sums=np.zeros((SLOTS,n,2),dtype='float64');exposure=np.zeros((SLOTS,n),dtype='float64')
    train=panel['valid']&(panel['times']<TRAIN_END)
    for row in np.flatnonzero(train):
        present=panel['times'][row]>=first;slot=panel['slot'][row]
        sums[slot]+=panel['counts'][row]*present[:,None];exposure[slot]+=present/3
    base=(sums.sum(0)+.5)/(exposure.sum(0)[:,None]+1)
    ws=sums.reshape(12,336,n,2).sum(0);we=exposure.reshape(12,336,n).sum(0)
    weekly=(ws+32*base)/(we[:,:,None]+32)
    prior=np.maximum((sums+32*np.tile(weekly,(12,1,1)))/(exposure[:,:,None]+32),1e-6).astype('float32')
    log=np.concatenate((np.broadcast_to(np.log1p(base),prior.shape),np.tile(np.log1p(weekly),(12,1,1)),np.log1p(prior)),2).astype('float32')
    neighbor=graph.dot(log.transpose(1,0,2).reshape(n,-1)).reshape(n,SLOTS,6).transpose(1,0,2).copy()
    return prior/3,log,neighbor

def prepare(history_dir,parent_dir,weather_dir,out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    with np.load(Path(parent_dir)/'winter/context.npz',allow_pickle=False) as f:
        ids=f['station_ids'].copy();first=f['first'].copy();profiles=f['profiles'].copy()
        graph=sparse.csr_matrix((f['graph_data'],f['graph_indices'],f['graph_indptr']),shape=(len(ids),len(ids)))
    events,source_ids,months,quality=h.read_events(Path(history_dir)/'historical-trips-20260331.zip',previous.historical_source_catalog())
    panel,audit=aggregate(events,source_ids,ids,months);del events,source_ids
    prior,log,neighbor=priors(panel,first,graph)
    weather=parent.weather_for(Path(weather_dir),'daejeon',panel['times'])
    np.save(out/'counts.npy',panel['counts'],allow_pickle=False)
    np.savez_compressed(out/'context.npz',station_ids=ids,first=first,profiles=profiles,prior=prior,log=log,neighbor=neighbor,
        graph_indptr=graph.indptr,graph_indices=graph.indices,graph_data=graph.data)
    np.savez_compressed(out/'time-features.npz',times=panel['times'],valid=panel['valid'],slot=panel['slot'],calendar=panel['calendar'],phase=panel['phase'],weather=weather)
    report=dict(source=quality,panel=audit,stations=len(ids),train_end=TRAIN_END,validation_end=VALID_END,test_end=END,
        weather={split:parent.coverage(weather,panel['valid']&(panel['times']>=s)&(panel['times']<e)) for split,s,e in [('train',START,TRAIN_END),('validation',TRAIN_END,VALID_END),('auxiliary',VALID_END,END)]})
    h.write(out/'data-audit.json',report);return report

def recent_from_counts(y,times,valid,first,graph,begin,end):
    """Past counts only. The row at t labels [t,t+600), so use row t-600."""
    rows=np.arange(begin,end);prev=rows-1;safe=np.maximum(prev,0)
    present=(prev[:,None]>=0)&valid[safe,None]&(times[safe,None]>=first[None,:])&(times[rows,None]-times[safe,None]==600)
    net=y[safe,:,1].astype('float32')-y[safe,:,0].astype('float32');net=np.arcsinh(net)*present
    fraction=graph.dot(present.T.astype('float32')).T
    neighboring=graph.dot(net.T).T/np.maximum(fraction,1e-8)
    return np.stack((net,present,neighboring,fraction),axis=2).astype('float32')
