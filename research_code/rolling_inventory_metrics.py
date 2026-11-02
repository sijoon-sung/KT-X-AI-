"""Frozen H13 finite-ensemble scores, aggregation and calendar block intervals."""
import numpy as np

LABELS=('persistence','recent_trend','no_recent_poisson','recent_poisson','recent_cfm')

def scores(samples,truth):
    x=np.sort(np.asarray(samples,dtype='float64'),axis=1);y=np.asarray(truth,dtype='float64');size=x.shape[1]
    if size<2 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Invalid scoring ensemble.')
    spread=(x*(2*np.arange(1,size+1)-size-1)).sum(1);absolute=np.abs(x-y[:,None]).mean(1)
    p=(x==0).mean(1);z=(y==0);median=x[:,int(np.ceil(.5*size))-1];mean=x.mean(1)
    low=x[:,int(np.ceil(.05*size))-1];high=x[:,int(np.ceil(.95*size))-1]
    return dict(fair_crps=absolute-spread/(size*(size-1)),crps=absolute-spread/(size*size),median_abs=np.abs(median-y),mean_sq=(mean-y)**2,
        brier=(p-z)**2,fair_brier=(p-z)**2-p*(1-p)/(size-1),coverage_5_95=((y>=low)&(y<=high)).astype(float),
        median_exact=(median==y).astype(float),median_within1=(np.abs(median-y)<=1).astype(float),observed_zero=z.astype(float),predicted_zero=p)

class Aggregator:
    def __init__(self):self.groups={g:{k:dict(n=0,sums={}) for k in LABELS} for g in ('all','current_zero','current_positive','strict_elapsed_30s')}
    def add(self,draws,truth,current,strict):
        masks={'all':np.ones(len(truth),bool),'current_zero':current==0,'current_positive':current>0,'strict_elapsed_30s':strict}
        all_scores={k:scores(draws[k],truth) for k in LABELS}
        for group,mask in masks.items():
            for label in LABELS:
                entry=self.groups[group][label];entry['n']+=int(mask.sum())
                for metric,values in all_scores[label].items():entry['sums'][metric]=entry['sums'].get(metric,0.)+float(values[mask].sum())
        return {k:float(all_scores[k]['fair_crps'].sum()) for k in LABELS}
    def result(self):
        out={}
        for group,values in self.groups.items():
            out[group]={}
            for label,e in values.items():
                value={k:(v/e['n'] if e['n'] else None) for k,v in e['sums'].items()}
                if value.get('mean_sq') is not None:value['mean_rmse']=float(np.sqrt(value.pop('mean_sq')))
                if 'median_abs' in value:value['median_mae']=value.pop('median_abs')
                out[group][label]=dict(pairs=e['n'],**value)
        return out

def uncertainty(daily,station_count):
    """14 chronological calendar days, including empty days; city-synchronized."""
    daily=np.asarray(daily,dtype='float64');total=daily.sum(0);cfm=LABELS.index('recent_cfm');controls=[LABELS.index(k) for k in ('persistence','recent_poisson')]
    if (total[controls]<=0).any() or total[-1]<=0:raise ValueError('Empty/zero control evaluation cannot support a relative claim.')
    gains=1-total[cfm]/total[controls];point=float(gains.min());intervals=[]
    for block in (3,7):
        rng=np.random.default_rng(20261103);n=len(daily);starts=rng.integers(0,n-block+1,(2000,(n+block-1)//block))
        indices=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:n];summed=daily[indices].sum(1)
        valid=(summed[:,controls]>0).all(1)&(summed[:,-1]>0);gain=np.min(1-summed[valid,cfm,None]/summed[valid][:,controls],axis=1)
        intervals.append(dict(block_days=block,valid_replicates=int(valid.sum()),interval=np.percentile(gain,[2.5,97.5]).tolist()))
    sufficient=station_count>=500 and total[-1]>=10000 and (daily[:,-1]>0).sum()>=10 and all(i['valid_replicates']>=1900 for i in intervals)
    decision='inconclusive'
    if sufficient and point>.01 and all(i['interval'][0]>.01 for i in intervals):decision='supported'
    elif sufficient and all(i['interval'][1]<=.01 for i in intervals):decision='not_supported'
    return dict(worst_control_fair_crps_relative_gain=point,relative_gain_by_control=dict(zip(('persistence','recent_poisson'),gains.tolist())),intervals=intervals,decision=decision,sufficient=bool(sufficient))
