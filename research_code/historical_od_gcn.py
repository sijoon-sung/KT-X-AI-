"""H7: history-only graph rate learning; separately locked September evaluation."""
import argparse
from datetime import datetime,timezone
import hashlib,json,os,platform,subprocess,time
from pathlib import Path

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.stats import poisson
import torch
from torch import nn

import historical_calendar as hist
import history_to_stock as stock
import loss_consistent_stock as distribution

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(11,29,47)
KINDS=('self','od','permuted')


def contract():
    path=ROOT/'research/hypotheses/H-0007.json'
    record=json.loads(path.read_text(encoding='utf-8'))
    assert record['status'] in ('preregistered','running')
    canonical=json.dumps(record['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    assert hashlib.sha256(canonical.encode()).hexdigest()==record['preregistration']['plan_sha256']
    p=ROOT/'research/datasets/tashu-historical-od-gcn-H0007-v1.json'
    assert stock.digest(p)==record['plan']['data']['input_manifest_sha256']
    inputs=json.loads(p.read_text(encoding='utf-8'))
    for rel,key in [('parent_manifest','parent_manifest_sha256'),('source_catalog','source_catalog_sha256')]:
        assert stock.digest(ROOT/inputs[rel])==inputs[key]
    assert stock.digest(ROOT/inputs['learning']['source_run_manifest'])==inputs['learning']['source_run_manifest_sha256']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    return record,inputs,code,stock.digest(p)


def graph_counts(events,source_ids,ids,times,valid):
    lookup={key:i for i,key in enumerate(ids)}
    mapped=np.array([lookup.get(x,-1) for x in source_ids])
    origin,destination=mapped[events['origin']],mapped[events['destination']]
    keep=(events['rent']>=hist.START)&(events['returned']<hist.TRAIN_END)&(origin>=0)&(destination>=0)
    selected=np.flatnonzero(keep)
    rent_index=(events['rent'][selected]-times[0])//3600
    return_index=(events['returned'][selected]-times[0])//3600
    selected=selected[valid[rent_index]&valid[return_index]]
    counts=np.zeros((len(ids),len(ids)),dtype=np.int64)
    np.add.at(counts,(origin[selected],destination[selected]),1)
    np.fill_diagonal(counts,0)
    return counts,dict(eligible_nonself_trips=int(counts.sum()),training_only=True)


def graph_operator(counts,top=16):
    weights=np.asarray(counts,dtype=float)+np.asarray(counts,dtype=float).T
    np.fill_diagonal(weights,0)
    if not np.isfinite(weights).all() or (weights<0).any():
        raise ValueError('Invalid OD weights.')
    chosen=np.zeros_like(weights,dtype=bool)
    index=np.arange(len(weights))
    for i,row in enumerate(weights):
        order=np.lexsort((index,-row))
        order=order[row[order]>0][:top]
        chosen[i,order]=True
    weights*=chosen|chosen.T
    isolated=weights.sum(1)==0
    weights[isolated,isolated]=1
    weights/=weights.sum(1,keepdims=True)
    return weights,dict(nodes=len(weights),isolated_nodes=int(isolated.sum()),nonzero_operator_entries=int((weights>0).sum()))


def feature_grid(base,calendar):
    n=len(base)
    weekday=np.repeat(np.arange(7),15)
    hour=np.tile(np.arange(7,22),7)
    rates=calendar[weekday,hour]
    shared=np.concatenate([np.eye(7)[weekday],np.eye(15)[hour-7]],axis=-1)
    x=np.concatenate([np.broadcast_to(np.log1p(base),(105,n,2)),np.log1p(rates),
                      np.broadcast_to(shared[:,None,:],(105,n,22))],axis=-1)
    return x.astype(np.float32),rates.astype(np.float32)


def aggregate_features(x,operator):
    slots,n,features=x.shape
    flat=x.transpose(1,0,2).reshape(n,-1)
    return (csr_matrix(operator)@flat).reshape(n,slots,features).transpose(1,0,2).astype(np.float32)


class RateGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.own=nn.Linear(26,32)
        self.neighbor=nn.Linear(26,32,bias=False)
        self.out=nn.Linear(32,2)
        nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)

    def forward(self,x,neighbor,prior):
        hidden=torch.relu(self.own(x)+self.neighbor(neighbor))
        correction=2*torch.tanh(self.out(hidden)/2)
        return prior*torch.exp(correction)


def train_one(kind,seed,x,ax,prior,y,slots,masks,artifacts):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=RateGCN().cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=.0001)
    features=torch.as_tensor(x,device='cuda')
    neighbors=torch.as_tensor(ax,device='cuda')
    prior_t=torch.as_tensor(prior,device='cuda')
    labels=torch.as_tensor(y[masks['train']],dtype=torch.float32,device='cuda')
    train_slots=torch.as_tensor(slots[masks['train']],device='cuda')
    valid_slots=slots[masks['validation']]
    valid_y=y[masks['validation']]
    best=float('inf');best_epoch=None;selected=None;trace=[]
    for epoch in range(21):
        if epoch:
            model.train()
            order=np.random.default_rng(seed+epoch).permutation(len(labels))
            for start in range(0,len(order),32):
                batch=torch.as_tensor(order[start:start+32],device='cuda')
                ss=train_slots[batch]
                prediction=model(features[ss],neighbors[ss],prior_t[ss])
                loss=(prediction-labels[batch]*torch.log(prediction)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite historical training loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
        model.eval()
        with torch.no_grad():
            means=model(features,neighbors,prior_t).cpu().numpy().astype(float)
        score=float(hist.deviance(valid_y,means[valid_slots]).mean())
        trace.append(dict(epoch=epoch,validation_count_deviance=score))
        if score<best:
            best,best_epoch,selected=score,epoch,means.copy()
            torch.save(model.state_dict(),artifacts/f'{kind}-seed{seed}.pt')
    del model,optimizer,features,neighbors,prior_t,labels,train_slots
    torch.cuda.empty_cache()
    return selected,dict(kind=kind,seed=seed,best_epoch=best_epoch,validation_count_deviance=best,trace=trace)


def train(args):
    timer=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    record,inputs,code,input_sha=contract()
    directory=ROOT/'research/runs'/args.run_id
    if directory.exists():raise ValueError('Preserve existing runs; choose a new ID.')
    artifacts=directory/'artifacts';artifacts.mkdir(parents=True)
    if not torch.cuda.is_available():raise ValueError('Preregistered CUDA training unavailable.')
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.15)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    history=Path(args.history_model_dir)
    assert stock.digest(history/'prepared_panel.npz')==inputs['historical_panel']['sha256']
    models={};ids=None
    for item in inputs['learning']['model_artifacts']:
        assert stock.digest(history/item['id'])==item['sha256']
        with np.load(history/item['id'],allow_pickle=False) as f:
            if ids is not None:assert np.array_equal(ids,f['station_ids'])
            ids=f['station_ids'].copy();models[item['id'].removesuffix('.npz')]=f['means'].copy()
    with np.load(history/'prepared_panel.npz',allow_pickle=False) as f:
        assert np.array_equal(ids,f['station_ids'])
        y,times,valid=f['recorded_counts'],f['timestamp_unix_seconds'],f['observed_record_hours']
    axes=hist.calendar_axes(times)
    panel=dict(times=times,valid=valid)
    masks=hist.split_masks(panel)
    slots=axes['weekday']*15+(axes['hour']-7)
    assert stock.digest(args.archive)==inputs['archive']['sha256']
    catalog=json.loads((ROOT/inputs['source_catalog']).read_text(encoding='utf-8'))
    events,source_ids,_,quality=hist.read_events(args.archive,catalog)
    counts,graph_scope=graph_counts(events,source_ids,ids,times,valid)
    del events
    operator,graph_summary=graph_operator(counts)
    permutation=np.random.default_rng(20261031).permutation(len(ids))
    operators=dict(self=np.eye(len(ids)),od=operator,permuted=operator[permutation][:,permutation])
    np.savez_compressed(artifacts/'graph.npz',station_ids=ids,counts=counts,od=operator,permutation=permutation)
    x,prior=feature_grid(models['constant'],models['calendar'])
    selections=[];rates={}
    for kind in KINDS:
        ax=aggregate_features(x,operators[kind]);seed_means=[]
        for seed in SEEDS:
            mean,selection=train_one(kind,seed,x,ax,prior,y,slots,masks,artifacts)
            selections.append(selection);seed_means.append(mean)
            print(json.dumps(dict(stage='history_checkpoint_selected',kind=kind,seed=seed,epoch=selection['best_epoch'],validation_count_deviance=selection['validation_count_deviance'])),flush=True)
        means=models['calendar'].copy()
        means[:,7:22]=np.mean(seed_means,axis=0).reshape(7,15,len(ids),2)
        rates[kind]=means
        np.savez_compressed(artifacts/f'{kind}.npz',station_ids=ids,means=means)
    rates['calendar']=models['calendar']
    np.savez_compressed(artifacts/'calendar.npz',station_ids=ids,means=rates['calendar'])
    lock=dict(code_commit=code,selected_on='historical February recorded counts only',seeds=list(SEEDS),selections=selections,
        frozen_rate_artifacts=[dict(id=f'{kind}.npz',sha256=stock.digest(artifacts/f'{kind}.npz')) for kind in (*KINDS,'calendar')],
        saved_before_march_and_api_utc=datetime.now(timezone.utc).isoformat())
    stock.write(artifacts/'selection.json',lock)
    # First March scoring happens after every candidate has been selected and saved.
    test=masks['test'];diagnostics={}
    for kind,means in rates.items():
        predictions=means[axes['weekday'][test],axes['hour'][test]]
        diagnostics[kind]=dict(recorded_count_deviance=float(hist.deviance(y[test],predictions).mean()),
                              recorded_count_mae=float(np.abs(y[test]-predictions).mean()))
    summary=dict(hypothesis_id='H-0007',phase='historical_training_complete',code_commit=code,
        input_manifest_sha256=input_sha,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-timer,selections=selections,source_quality=quality,
        graph={**graph_scope,**graph_summary},parameters=sum(p.numel() for p in RateGCN().parameters()),
        historical_scope={name:dict(hours=int(mask.sum()),dates=int(len(np.unique(axes['day'][mask])))) for name,mask in masks.items()},
        march_auxiliary_recorded_counts=diagnostics,api_opened=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=torch.cuda.get_device_name(),
            cuda_peak_bytes=int(torch.cuda.max_memory_allocated()),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG')))
    stock.write(directory/'training.json',summary)
    print(json.dumps(dict(stage='historical_training_complete',seconds=summary['elapsed_seconds'],march_auxiliary=diagnostics)),flush=True)


def numerical_contract(rate_sums):
    rate_sums=np.asarray(rate_sums)
    if not np.isfinite(rate_sums).all() or (rate_sums<0).any():raise ValueError('Invalid frozen rates.')
    maximum=float(np.max(rate_sums)*.5)
    if not np.isfinite(maximum) or maximum<0:raise ValueError('Invalid frozen rates.')
    k=max(16,int(np.ceil(poisson.isf(1e-12,maximum)))+1)
    tail=float(poisson.sf(k,maximum))
    bound=float(maximum*poisson.sf(k-1,maximum)+tail*maximum/(1-tail))
    if k>1024 or tail>1e-10 or bound>1e-8:raise ValueError('Frozen historical numerical gate failed.')
    return dict(cutoff=k,maximum_expected_attempts=maximum,tail_probability_bound=tail,mean_error_bound=bound)


def bootstraps(daily,block):
    rng=np.random.default_rng(20261031);days=len(daily)
    starts=rng.integers(0,days-block+1,(2000,(days+block-1)//block))
    index=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:days]
    sums=daily[index].sum(1);valid=sums[:,-1]>0
    gains=(sums[valid,:3]-sums[valid,3,None])/sums[valid,-1,None]
    point=gains.min(1)
    return dict(block_days=block,valid_replicates=len(point),requested_replicates=2000,interval=np.percentile(point,[2.5,97.5]).tolist() if len(point) else None)


def evaluate(args):
    timer=time.perf_counter();record,inputs,code,input_sha=contract()
    directory=ROOT/'research/runs'/args.run_id;artifacts=directory/'artifacts'
    if (directory/'summary.json').exists():raise ValueError('Preserve completed evaluation.')
    training=json.loads((directory/'training.json').read_text(encoding='utf-8'))
    lock=json.loads((artifacts/'selection.json').read_text(encoding='utf-8'))
    assert training['code_commit']==lock['code_commit']==code and training['api_opened'] is False
    rates={};ids=None
    for entry in lock['frozen_rate_artifacts']:
        assert stock.digest(artifacts/entry['id'])==entry['sha256']
        with np.load(artifacts/entry['id'],allow_pickle=False) as f:
            if ids is not None:assert np.array_equal(ids,f['station_ids'])
            ids=f['station_ids'].copy();rates[entry['id'].removesuffix('.npz')]=f['means'].copy()
    numeric=numerical_contract(np.concatenate([x.sum(-1).ravel() for x in rates.values()]))
    print(json.dumps(dict(stage='frozen_history_numeric_contract_before_api',**numeric)),flush=True)
    ev=inputs['evaluation']
    assert stock.digest(ROOT/ev['api_manifest'])==ev['api_manifest_sha256']
    assert stock.digest(args.panel)==ev['audited_panel']['sha256']
    with np.load(args.panel,allow_pickle=False) as f:
        data=stock.pair_data(f['stock'],f['scheduled_at_unix_seconds'],f['observed_at_unix_seconds'],f['station_ids'],ids)
    if ((data['day_index']<0)|(data['day_index']>=14)).any():raise ValueError('Unexpected source calendar.')
    row,col=np.nonzero(np.isfinite(data['current']));shape=data['current'].shape
    all_predictions={};groups={}
    for kind,means in rates.items():
        predictions={key:np.full(shape,np.nan) for key in ('mean','median','lower','upper','pzero','expected_mae_advantage')}
        crps=np.full(shape,np.nan)
        selected=means[data['weekday'][row],data['hour'][row],data['model_cols'][col]]
        for start in range(0,len(row),512):
            rr,cc=row[start:start+512],col[start:start+512];chosen=selected[start:start+512]
            grid,pmf=distribution.state_pmf(data['current'][rr,cc],chosen[:,1],chosen[:,0],numeric['cutoff'])
            outputs=distribution.summaries(grid,pmf)
            for key,value in outputs.items():predictions[key][rr,cc]=value
            crps[rr,cc]=distribution.integer_crps(grid,pmf,data['future'][rr,cc])
        groups[kind]=distribution.group_scores(data,predictions,crps)
        all_predictions[kind]=predictions
        np.savez_compressed(artifacts/f'predictions-{kind}.npz',**predictions,crps=crps,valid=data['valid'],initial=data['current'],target=data['future'],times=data['times'])
        print(json.dumps(dict(stage='fixed_candidate_scored',kind=kind,positive_brier=groups[kind]['positive']['distribution']['brier'])),flush=True)
    positive=data['valid']&(data['current']>0);truth=data['future']==0
    kinds=('self','permuted','calendar','od')
    losses={kind:(all_predictions[kind]['pzero']-truth)**2 for kind in kinds}
    daily=np.zeros((14,5))
    for day in range(14):
        mask=positive&(data['day_index'][:,None]==day)
        daily[day]=[*[losses[k][mask].sum() for k in kinds],int(mask.sum())]
    gains={kind:float((losses[kind][positive]-losses['od'][positive]).mean()) for kind in kinds[:3]}
    point=min(gains.values());intervals=[bootstraps(daily,b) for b in (3,7)]
    enough=data['scope']['matched_ids']>=500 and int(positive.sum())>=10000 and (daily[:,-1]>0).sum()>=10 and all(b['valid_replicates']>=1900 for b in intervals)
    verdict='inconclusive'
    if enough and point>.002 and all(b['interval'][0]>.002 for b in intervals):verdict='supported'
    elif enough and all(b['interval'][1]<=.002 for b in intervals):verdict='not_supported'
    metrics=dict(worst_control_positive_brier_gain=point,positive_pairs=int(positive.sum()),matched_training_ids=data['scope']['matched_ids'],positive_dates=int((daily[:,-1]>0).sum()))
    for b in intervals:metrics[f"ci_{b['block_days']}day_lower"],metrics[f"ci_{b['block_days']}day_upper"]=b['interval']
    for entry in lock['frozen_rate_artifacts']:assert stock.digest(artifacts/entry['id'])==entry['sha256']
    persistence=stock.group_scores(data,{'persistence':((data['current']==0).astype(float),data['current'])})[0]
    limitations=['Official September repeatedly exposed, including H5/H6; not untouched independent evaluation.',
        'Historical-only fitting and February count checkpoint selection; historical reused splits and existing calendar-prior shrinkage selection.',
        'One small static-profile/calendar graph model and one undirected OD construction; no inference on all GCNs or learned/dynamic graphs.',
        'No recent transactions at September inference; effective completed-flow rates omit exposure, capacity, operator movement and observation latency.',
        'Mean-rate three-seed ensemble; block intervals omit fitting/selection uncertainty; auxiliary March counts are not inventory truth.',
        'Fixed API evaluation cannot select winning model/threshold/calibration, establish usability or policy value, or finalize FM/architecture.']
    summary=dict(hypothesis_id='H-0007',decision=verdict,metrics=metrics,control_positive_brier_gains=gains,
        scope=data['scope'],groups=groups,persistence=persistence,bootstraps=intervals,numerical_contract=numeric,
        daily_positive_scores=[dict(day_index=i,loss_sums={k:float(x[j]) for j,k in enumerate(kinds)},pairs=int(x[-1])) for i,x in enumerate(daily)],
        api_training=False,api_selection_or_calibration=False,rate_hashes_unchanged=True,elapsed_evaluation_seconds=time.perf_counter()-timer,limitations=limitations)
    stock.write(directory/'summary.json',summary)
    public=['summary.json','training.json']
    records=[dict(id=name,sha256=stock.digest(directory/name),availability='public aggregate') for name in public]
    records += [dict(id=p.name,sha256=stock.digest(p),availability='private dislab model/graph/selection/prediction; not published') for p in sorted(artifacts.iterdir())]
    manifest=dict(schema_version=1,run_id=args.run_id,hypothesis_id='H-0007',plan_sha256=record['preregistration']['plan_sha256'],code_commit=code,input_manifest_sha256=input_sha,
        started_at_utc=training['started_at_utc'],finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=[['python','research_code/historical_od_gcn.py','train','--archive','<PRIVATE_HISTORY_ARCHIVE>','--history-model-dir','<PRIVATE_H4_MODELS>','--run-id',args.run_id],
                 ['python','research_code/historical_od_gcn.py','evaluate','--panel','<PRIVATE_AUDITED_API_PANEL>','--run-id',args.run_id]],
        environment={**training['environment'],'scipy':scipy.__version__,'compute_location':'dislab','evaluation_device':'CPU float64, 4 threads'},
        seed=list(SEEDS),evaluation_scope=data['scope'],metrics=metrics,artifacts=records,limitations=limitations)
    stock.write(directory/'manifest.json',manifest)
    print(json.dumps(dict(stage='completed',decision=verdict,metrics=metrics,gains=gains,seconds=summary['elapsed_evaluation_seconds'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='phase',required=True)
    tr=sub.add_parser('train');tr.add_argument('--archive',required=True);tr.add_argument('--history-model-dir',required=True);tr.add_argument('--run-id',required=True)
    ev=sub.add_parser('evaluate');ev.add_argument('--panel',required=True);ev.add_argument('--run-id',required=True)
    args=ap.parse_args()
    try:
        (train if args.phase=='train' else evaluate)(args)
    except Exception as error:
        directory=ROOT/'research/runs'/args.run_id
        if directory.exists() and not (directory/'summary.json').exists():stock.write(directory/f'attempt-{args.phase}.json',dict(status='failed',decision='inconclusive',reason=type(error).__name__+': '+str(error)))
        raise
