"""H12 source forecasting panels and lagged public weather, no inventory loader."""
from pathlib import Path
from html.parser import HTMLParser
from datetime import datetime
import calendar,csv,hashlib,json,re
import numpy as np
from scipy import sparse
import cross_city_data as old
import historical_calendar as hist
import seasonal_history_core as seasonal

SOURCE_END=hist.stamp('2024-09-01')
WEATHER_DIM=54

class CalendarParser(HTMLParser):
    def __init__(self):super().__init__();self.depth=0;self.rows=[];self.row=None;self.cell=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if tag=='table' and 'table-cal' in a.get('class',''):self.depth=1
        elif self.depth and tag=='table':self.depth+=1
        if not self.depth:return
        if tag=='tr':self.row=[]
        if tag in ('td','th'):self.cell=[]
        if tag=='br' and self.cell is not None:self.cell.append(' ')
    def handle_data(self,s):
        if self.depth and self.cell is not None:self.cell.append(s)
    def handle_endtag(self,tag):
        if not self.depth:return
        if tag in ('td','th') and self.cell is not None:self.row.append(re.sub(r'\s+',' ',''.join(self.cell)).strip());self.cell=None
        if tag=='tr' and self.row is not None:self.rows.append(self.row);self.row=None
        if tag=='table':self.depth-=1

def calendar_reports(files,station):
    reports={}
    for path in files:
        text=Path(path).read_text(encoding='utf-8');match=re.fullmatch(r'kma-calendar-(\d+)-(\d{4})(\d{2})\.html',Path(path).name)
        if not match or int(match[1])!=station:raise ValueError('KMA station/file mismatch.')
        y,m=int(match[2]),int(match[3]);city='서울' if station==108 else '대전'
        if f'option value="{station}" selected="selected"' not in text or not re.search(city+r'\(유\)/\s*'+str(y)+r'년\s*'+str(m)+r'월',re.sub('<[^>]*>',' ',text)):raise ValueError('KMA station/year/month selection mismatch.')
        parser=CalendarParser();parser.feed(text);days=None;count=0
        for row in parser.rows:
            if any(re.fullmatch(r'\d+일',v) for v in row):days=[int(v[:-1]) if re.fullmatch(r'\d+일',v) else None for v in row];continue
            if days is None or len(days)!=len(row):continue
            for day,cell in zip(days,row):
                if day is None:continue
                rain=re.search(r'일강수량\s*:\s*(-|\d+(?:\.\d+)?)',cell)
                if rain and re.search(r'평균기온\s*:\s*-?\d',cell):
                    date=f'{y}-{m:02}-{day:02}';key=(hist.stamp(date)+32400)//86400
                    reports[key]=rain[1];count+=1
            days=None
        if count!=calendar.monthrange(y,m)[1]:raise ValueError('Incomplete KMA month calendar.')
    return reports

def daily_features(reports,times):
    x=np.zeros((len(times),4),dtype='float32')
    for i,day in enumerate((np.asarray(times)+32400)//86400-2):
        v=reports.get(int(day))
        if v=='-':x[i,2]=1
        elif v is not None:
            value=float(v)
            if np.isfinite(value) and 0<=value<=1000:x[i]=[np.log1p(value)/np.log(11),1,0,float(value==0)]
    return x

def latest(records,times,width):
    x=np.zeros((len(times),width+2),dtype='float32');x[:,-1]=1
    if not records:return x
    ts=np.array(sorted(records),dtype='int64');v=np.array([records[t] for t in ts],dtype='float32')
    ix=np.searchsorted(ts,np.asarray(times)-3600,side='right')-1;safe=np.maximum(ix,0);age=np.asarray(times)-ts[safe]
    keep=(ix>=0)&(age>=3600)&(age<=14400)
    x[keep,:width]=v[safe[keep]];x[keep,width]=1;x[keep,width+1]=age[keep]/14400
    return x

def extra_reports(files):
    nominal={};phenomena={};thermo=[{} for _ in range(3)]
    for path in files:
        with open(path,encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f,delimiter='|'):
                stamp=int(old.pd.Timestamp(row['DATE'],tz='UTC').timestamp())
                for j,var in enumerate(old.VARS[:3]):
                    if row[var+'_Quality_Code'] not in ('','0','1','4'):continue
                    value=old.accepted_weather(row[var],row[var+'_Quality_Code'],row[var+'_Measurement_Code'],j)
                    if value is not None:thermo[j][stamp]=[value[0]/(20,5,100)[j]]
                if row['precipitation_Source_Code']=='223' and row['precipitation_Quality_Code'] in ('','0','1','4'):
                    v=old.accepted_weather(row['precipitation'],row['precipitation_Quality_Code'],row['precipitation_Measurement_Code'],3)
                    if v is not None:nominal[stamp]=[np.log1p(v[0])/np.log(11),v[1]]
                codes=[]
                for k in ('pres_wx_MW1','pres_wx_MW2','pres_wx_MW3'):
                    if row[k+'_Quality_Code'] not in ('','0','1','4'):continue
                    value=row[k].split(':')[-1]
                    if re.fullmatch(r'\d{1,2}',value) and 0<=int(value)<=99:codes.append(int(value)//10)
                if codes:
                    groups=np.zeros(10,dtype='float32');groups[codes]=1;phenomena[stamp]=groups
    return nominal,thermo,phenomena

def weather(primary,airport,kma_files,station,times):
    prefix=old.lag_weather(primary,times)
    prefix[:,0]/=20;prefix[:,1]/=5;prefix[:,2]/=100;prefix[:,3]=np.log1p(prefix[:,3])/np.log(11)
    nominal,_,pw=extra_reports(primary);_,thermo,aw=extra_reports(airport)
    n=latest(nominal,times,2) # [amount,trace,present,age] reordered below
    nominal_x=n[:,[0,2,3,1]]
    near=[latest(v,times,1) for v in thermo]
    near_x=np.c_[*[v[:,0] for v in near],*[v[:,1] for v in near],*[v[:,2] for v in near]]
    daily=daily_features(calendar_reports(kma_files,station),times)
    x=np.c_[prefix,nominal_x,daily,near_x,latest(pw,times,10),latest(aw,times,10)].astype('float32')
    if x.shape[1]!=WEATHER_DIM or not np.isfinite(x).all():raise ValueError('Invalid weather schema.')
    return x

class SourceCache:
    def __init__(self,folder):self.folder=Path(folder);self.folder.mkdir(exist_ok=True)
    def __call__(self,month,events,ids,hours,valid,first):
        begin=int(hours[0]);stop=int(hours[-1]+3600);all_y=np.zeros((len(hours)*2,len(ids),2),dtype='uint16')
        for ch,tk,sk in ((0,'rent','origin'),(1,'returned','destination')):
            ix=(events[tk]-begin)//1800;keep=(ix>=0)&(ix<len(all_y));node=np.searchsorted(ids,events[sk][keep]);np.add.at(all_y,(ix[keep],node,ch),1)
        times=(hours[valid,None]+[0,1800]).ravel();y=all_y[np.repeat(valid,2)]
        keep=(events['rent']<SOURCE_END)&(events['returned']<SOURCE_END)&(events['origin']!=events['destination'])
        pair=events['origin'][keep].astype('int64')*100000+events['destination'][keep];pair,count=np.unique(pair,return_counts=True)
        np.savez_compressed(self.folder/f'{month:02}.npz',ids=ids,times=times,counts=y,first=first,od_origin=pair//100000,od_destination=pair%100000,od_count=count)

def source_panel(folder):
    paths=sorted(Path(folder).glob('??.npz'))
    if len(paths)!=12:raise ValueError('Source requires 12 audited cached months.')
    ids=np.unique(np.concatenate([np.load(p,allow_pickle=False)['ids'] for p in paths]));n=len(ids)
    length=sum(len(np.load(p,allow_pickle=False)['times']) for p in paths)
    times=np.zeros(length,dtype='int64');y=np.zeros((length,n,2),dtype='uint16');first=np.full(n,np.iinfo('int64').max,dtype='int64');od=sparse.csr_matrix((n,n),dtype='float64');offset=0
    for path in paths:
        with np.load(path,allow_pickle=False) as f:
            index=np.searchsorted(ids,f['ids']);size=len(f['times']);times[offset:offset+size]=f['times'];y[offset:offset+size,index]=f['counts']
            if int(path.stem)<=8:np.minimum.at(first,index,f['first'])
            u=np.searchsorted(ids,f['od_origin']);v=np.searchsorted(ids,f['od_destination']);od+=sparse.coo_matrix((f['od_count'],(u,v)),shape=(n,n)).tocsr();offset+=size
    a=old.axes(times)
    return dict(times=times,counts=y,valid=np.ones(length,dtype=bool),ids=ids.astype(str),first=first,od=od,**a)

def prepare_source_fold(panel):
    t=panel['times'];y=panel['counts'];train=t<SOURCE_END;valid=(t>=SOURCE_END)&(t<hist.stamp('2024-10-01'));test=t>=hist.stamp('2024-10-01')
    active=np.zeros(y.shape[1],dtype=int)
    for day in np.unique(panel['day'][train]):active+=(y[train&(panel['day']==day)].sum((0,2))>0)
    keep=(y[train].sum(0)>=100).all(1)&(active>=60)&(y[train&(t>=SOURCE_END-30*86400)].sum((0,2))>0)
    selected=np.flatnonzero(keep);ids=panel['ids'][selected];first=panel['first'][selected];n=len(ids);slot=panel['month']*210+panel['weekly']
    sums=np.zeros((2520,n,2));exposure=np.zeros((2520,n))
    for row in np.flatnonzero(train):
        present=t[row]>=first;exposure[slot[row]]+=present;sums[slot[row]]+=y[row,selected]*present[:,None]
    base,weekly,prior=seasonal.profiles(sums,exposure,32)
    profile=old.profile(sums.reshape(12,210,n,2).sum(0).transpose(1,0,2),exposure.reshape(12,210,n).sum(0).T,sums.reshape(12,210,n,2).sum(1).transpose(1,0,2),exposure.reshape(12,210,n).sum(1).T)[:,:420]
    od=panel['od'][selected][:,selected].toarray();od+=od.T.copy();graph=np.zeros((n,n),dtype='float32')
    for i in range(n):
        ranked=np.lexsort((ids,-od[i]))[:8];ranked=ranked[od[i,ranked]>0];graph[i,ranked]=od[i,ranked];graph[i,i]=1
    graph/=graph.sum(1,keepdims=True);graph=sparse.csr_matrix(graph)
    log=np.concatenate((np.broadcast_to(np.log1p(base)[None],prior.shape),np.log1p(weekly),np.log1p(prior)),axis=2).astype('float32')
    neighbor=graph.dot(log.transpose(1,0,2).reshape(n,-1)).reshape(n,2520,6).transpose(1,0,2).copy()
    return dict(selected=selected,ids=ids,first=first,train=train,validation=valid,test=test,slot=slot,profiles=profile,
        prior=prior.astype('float32'),log=log,neighbor=neighbor,graph=graph,train_end=SOURCE_END)
