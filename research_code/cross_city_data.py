"""H11 historical/source-city data only. No inventory API loader."""
import calendar
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from scipy import sparse
import historical_calendar as hist

START=hist.stamp('2024-08-01'); END=hist.stamp('2026-04-01')
FOLDS=(('spring','2025-03-01','2025-04-01','2025-06-01'),
       ('summer','2025-07-01','2025-08-01','2025-10-01'),
       ('winter','2026-02-01','2026-03-01','2026-04-01'))
VARS=('temperature','wind_speed','relative_humidity','precipitation_3_hour')
BOUNDS=((-60,60),(0,75),(0,100),(0,500))

def axes(times):
    local=np.asarray(times,dtype=np.int64)+32400
    day=local//86400; dates=day.astype('datetime64[D]'); month=dates.astype('datetime64[M]')
    dom=(dates-month).astype(int)+1
    last=((month+1).astype('datetime64[D]')-month).astype(int)
    half=(local%86400)//1800; weekday=(day+3)%7
    return dict(day=day,month=month.astype(int)%12,month_string=month.astype(str),
        interior=(dom>1)&(dom<last),hour=half//2,half=half%2,
        weekly=weekday*30+np.clip(half-14,0,29),weekday=weekday,
        in_hours=(half>=14)&(half<44))

def profile(weekly_sum,weekly_n,monthly_sum,monthly_n):
    """Shared profile coordinates; a zero exposure month has an explicit mask."""
    base=np.maximum(weekly_sum.sum(1)/np.maximum(weekly_n.sum(1)[:,None],1),1e-6)
    weekly=np.divide(weekly_sum,weekly_n[...,None],out=np.broadcast_to(base[:,None],weekly_sum.shape).copy(),where=weekly_n[...,None]>0)
    monthly=np.divide(monthly_sum,monthly_n[...,None],out=np.zeros_like(monthly_sum),where=monthly_n[...,None]>0)
    return np.c_[np.log1p(weekly/base[:,None]).reshape(len(base),420),
        np.log1p(monthly/base[:,None]).reshape(len(base),24),(monthly_n>0)].astype('float32')

def source_profiles(archive,out):
    out=Path(out); stats={};valid_times=[];audit=[]
    dtype=np.dtype([('bike','S32'),('rent','i8'),('origin','i4'),('returned','i8'),('destination','i4')])
    with zipfile.ZipFile(archive) as z:
        members=sorted(i for i in z.namelist() if i.lower().endswith('.csv'))
        if len(members)!=12:raise ValueError('Expected 12 Seoul monthly CSVs.')
        for name in members:
            code=re.search(r'_(24\d\d)\.csv$',name)
            if not code:raise ValueError('Unknown Seoul source month.')
            month=int(code[1][-2:]); begin=hist.stamp(f'2024-{month:02}-01')
            stop=hist.stamp('2025-01-01') if month==12 else hist.stamp(f'2024-{month+1:02}-01')
            parts=[];read=0;rejected=0
            columns=['자전거번호','대여일시','대여 대여소번호','반납일시','반납대여소번호']
            with z.open(name) as f:
                for chunk in pd.read_csv(f,encoding='cp949',usecols=columns,dtype=str,keep_default_na=False,chunksize=100000,on_bad_lines='error'):
                    read+=len(chunk)
                    rent=pd.to_datetime(chunk[columns[1]],format='%Y-%m-%d %H:%M:%S',errors='coerce')
                    ret=pd.to_datetime(chunk[columns[3]],format='%Y-%m-%d %H:%M:%S',errors='coerce')
                    rt=rent.astype('int64').to_numpy()//10**9-32400;dt=ret.astype('int64').to_numpy()//10**9-32400
                    origin=pd.to_numeric(chunk[columns[2]],errors='coerce').to_numpy(dtype=float)
                    dest=pd.to_numeric(chunk[columns[4]],errors='coerce').to_numpy(dtype=float)
                    bike=chunk[columns[0]].str.strip().str.encode('utf-8')
                    good=rent.notna().to_numpy()&ret.notna().to_numpy()&(rt>=begin)&(rt<stop)&(dt>=rt)&(dt-rt<86400)
                    good &= np.isfinite(origin)&np.isfinite(dest)&(origin==np.floor(origin))&(dest==np.floor(dest))&(origin>=1)&(origin<=99999)&(dest>=1)&(dest<=99999)
                    good &= (bike.str.len().to_numpy()>0)&(bike.str.len().to_numpy()<=32)
                    rejected+=int((~good).sum());part=np.empty(int(good.sum()),dtype=dtype)
                    part['bike']=bike.to_numpy()[good].astype('S32');part['rent']=rt[good];part['returned']=dt[good]
                    part['origin']=origin[good].astype('int32');part['destination']=dest[good].astype('int32');parts.append(part)
            before=sum(len(p) for p in parts);events=np.unique(np.concatenate(parts));del parts
            ids=np.unique(np.r_[events['origin'],events['destination']]);n=len(ids)
            weekly=np.zeros((n,210,2));monthly=np.zeros((n,2));first=np.full(n,np.iinfo('i8').max,dtype='i8')
            activity=np.zeros((n,366),dtype=bool);hours=np.arange(begin,stop,3600);city=np.zeros((len(hours),2),dtype='i8')
            for ch,tk in enumerate(('rent','returned')):
                ix=(events[tk]-begin)//3600;keep=(ix>=0)&(ix<len(hours));np.add.at(city,(ix[keep],ch),1)
            ha=axes(hours);valid=ha['interior']&ha['in_hours']&(city>0).all(1)
            valid_times.extend((hours[valid,None]+[0,1800]).ravel().tolist())
            for ch,tk,sk in ((0,'rent','origin'),(1,'returned','destination')):
                node=np.searchsorted(ids,events[sk]);np.minimum.at(first,node,events[tk])
                ix=(events[tk]-begin)//3600;keep=(ix>=0)&(ix<len(hours));keep &= valid[np.clip(ix,0,len(hours)-1)]
                ea=axes(events[tk][keep]);node=node[keep]
                np.add.at(weekly,(node,ea['weekly'],ch),1);np.add.at(monthly,(node,ch),1)
                activity[node,ea['day']-(hist.stamp('2024-01-01')+32400)//86400]=True
            for i,station in enumerate(ids):
                d=stats.setdefault(int(station),dict(weekly=np.zeros((210,2)),monthly=np.zeros((12,2)),first=np.iinfo('i8').max,active=np.zeros(366,dtype=bool)))
                d['weekly']+=weekly[i];d['monthly'][month-1]+=monthly[i];d['first']=min(d['first'],int(first[i]));d['active']|=activity[i]
            audit.append(dict(month=f'2024-{month:02}',rows=read,rejected=rejected,duplicates=before-len(events),accepted=len(events),eligible_hours=int(valid.sum())))
            print(json.dumps(dict(stage='seoul_source_month',**audit[-1])),flush=True)
    ids=np.array(sorted(stats));times=np.array(sorted(valid_times),dtype='i8');a=axes(times)
    wn=np.zeros((len(ids),210));mn=np.zeros((len(ids),12))
    for i,s in enumerate(ids):
        active=times>=stats[int(s)]['first'];wn[i]=np.bincount(a['weekly'][active],minlength=210);mn[i]=np.bincount(a['month'][active],minlength=12)
    ws=np.array([stats[int(s)]['weekly'] for s in ids]);ms=np.array([stats[int(s)]['monthly'] for s in ids]);days=np.array([stats[int(s)]['active'].sum() for s in ids])
    keep=(ws.sum(1)>=100).all(1)&(days>=60)&((mn>0).sum(1)>=6)
    x=profile(ws[keep],wn[keep],ms[keep],mn[keep]);ids=ids[keep]
    validation=np.array([int(hashlib.sha256(str(int(s)).encode()).hexdigest(),16)%10==0 for s in ids])
    np.savez_compressed(out,ids=ids,profiles=x,validation=validation)
    return dict(months=audit,stations=len(ids),source_train=int((~validation).sum()),source_validation=int(validation.sum()),input_shape=list(x.shape))

def target_panel(events,ids,months):
    days=np.arange(START,END,86400);times=(days[:,None]+np.arange(14,44)*1800).ravel()
    counts=np.zeros((len(times),len(ids),2),dtype=np.uint16)
    for ch,tk,sk in ((0,'rent','origin'),(1,'returned','destination')):
        delta=events[tk]-START;half=(delta%86400)//1800;keep=(delta>=0)&(events[tk]<END)&(half>=14)&(half<44)
        ix=(delta[keep]//86400)*30+half[keep]-14
        np.add.at(counts,(ix,events[sk][keep],ch),1)
    a=axes(times);city=counts.astype('uint32').reshape(-1,2,len(ids),2).sum((1,2))
    valid=a['interior']&np.isin(a['month_string'],list(months))&np.repeat((city>0).all(1),2)
    return dict(times=times,counts=counts,valid=valid,ids=ids,**a)

def fold_data(panel,events,train_end,valid_end,test_end):
    t=panel['times'];te,ve,ee=map(hist.stamp,(train_end,valid_end,test_end))
    train=panel['valid']&(t<te);val=panel['valid']&(t>=te)&(t<ve);test=panel['valid']&(t>=ve)&(t<ee)
    y=panel['counts'];total=y[train].sum(0);active=np.zeros(y.shape[1],dtype=int)
    for day in np.unique(panel['day'][train]):active+=(y[train&(panel['day']==day)].sum((0,2))>0)
    last=y[train&(t>=te-30*86400)].sum((0,2))>0
    selected=np.flatnonzero((total>=100).all(1)&(active>=60)&last);ids=panel['ids'][selected];n=len(ids)
    mapping=np.full(y.shape[1],-1,dtype=int);mapping[selected]=np.arange(n)
    first=np.full(n,np.iinfo('i8').max,dtype='i8')
    for tk,sk in (('rent','origin'),('returned','destination')):
        node=mapping[events[sk]];keep=(node>=0)&(events[tk]<te);np.minimum.at(first,node[keep],events[tk][keep])
    slot=panel['month']*210+panel['weekly'];sums=np.zeros((2520,n,2));exposure=np.zeros((2520,n))
    for row in np.flatnonzero(train):
        present=t[row]>=first;exposure[slot[row]]+=present;sums[slot[row]]+=y[row,selected]*present[:,None]
    import seasonal_history_core as seasonal
    base,weekly,prior=seasonal.profiles(sums,exposure,32)
    p=profile(sums.reshape(12,210,n,2).sum(0).transpose(1,0,2),exposure.reshape(12,210,n).sum(0).T,
        sums.reshape(12,210,n,2).sum(1).transpose(1,0,2),exposure.reshape(12,210,n).sum(1).T)
    od=np.zeros((n,n));u=mapping[events['origin']];v=mapping[events['destination']]
    keep=(u>=0)&(v>=0)&(events['returned']<te)&(events['rent']>=START)&(u!=v)
    np.add.at(od,(u[keep],v[keep]),1);od+=od.T.copy()
    graph=np.zeros((n,n),dtype='float32')
    for i in range(n):
        ranked=np.lexsort((ids,-od[i]))[:8];ranked=ranked[od[i,ranked]>0];graph[i,ranked]=od[i,ranked];graph[i,i]=1
    graph/=graph.sum(1,keepdims=True);graph=sparse.csr_matrix(graph)
    log=np.concatenate([np.broadcast_to(np.log1p(base)[None],prior.shape),np.log1p(weekly),np.log1p(prior)],axis=2).astype('float32')
    neighbor=graph.dot(log.transpose(1,0,2).reshape(n,-1)).reshape(n,2520,6).transpose(1,0,2).copy()
    return dict(selected=selected,ids=ids,first=first,train=train,validation=val,test=test,slot=slot,
        profiles=p,prior=prior.astype('float32'),log=log,neighbor=neighbor,graph=graph,train_end=te)

def accepted_weather(value,quality,measurement,index):
    try:v=float(value)
    except (ValueError,TypeError):return None
    if quality not in ('','0','1','4','9') or not np.isfinite(v) or not BOUNDS[index][0]<=v<=BOUNDS[index][1]:return None
    if index==3 and measurement not in ('','T'):return None
    return (0. if index==3 and measurement=='T' else v),float(index==3 and measurement=='T')

def lag_weather(files,times):
    import csv
    reports=[{} for _ in VARS]
    for path in files:
        with open(path,encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f,delimiter='|'):
                stamp=int(pd.Timestamp(row['DATE'],tz='UTC').timestamp())
                for j,var in enumerate(VARS):
                    value=accepted_weather(row[var],row[var+'_Quality_Code'],row[var+'_Measurement_Code'],j)
                    if value is not None:reports[j][stamp]=value
    x=np.zeros((len(times),13),dtype='float32');x[:,8:12]=1
    for j,records in enumerate(reports):
        ordered=sorted(records);ts=np.array(ordered);values=np.array([records[k] for k in ordered]);ix=np.searchsorted(ts,times-3600,side='right')-1
        if not len(ts):continue
        safe=np.maximum(ix,0);age=times-ts[safe];good=(ix>=0)&(age>=3600)&(age<=14400)
        x[good,j]=values[safe[good],0];x[good,4+j]=1;x[good,8+j]=age[good]/14400
        if j==3:x[good,12]=values[safe[good],1]
    return x

def normalize_weather(raw,train):
    x=raw.copy();center=[];scale=[]
    for j in range(4):
        valid=train&(raw[:,4+j]>0);v=raw[valid,j]
        m=float(v.mean()) if len(v) else 0.;s=max(float(v.std()),.1) if len(v) else 1.
        x[:,j]=np.where(raw[:,4+j]>0,(raw[:,j]-m)/s,0);center.append(m);scale.append(s)
    return x,dict(mean=center,std=scale,training_present_fraction=raw[train,4:8].mean(0).tolist())
