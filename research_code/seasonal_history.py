"""H10 historical fitting, locked September evaluation and fixed-model inference. Run on dislab."""
import argparse,hashlib,json,platform,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,torch
from torch import nn
import historical_calendar as hist
import historical_od_gcn as graph
import history_to_stock as stock
import loss_consistent_stock as probability
import scaled_flow_core as slots
import scaled_seoul_flow as prior_run
import seasonal_history_core as core

ROOT=Path(__file__).resolve().parents[1]
INPUT='research/datasets/tashu-seasonal-history-H0010-v1.json'
KINDS=('pooled_calendar','seasonal_profile','seasonal_self','seasonal_od')
KEYS=('mean','median','lower','upper','pzero','crps')


def contract():
    record=json.loads((ROOT/'research/hypotheses/H-0010.json').read_text(encoding='utf-8'))
    canonical=json.dumps(record['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256']:raise ValueError('Frozen plan changed.')
    if stock.digest(ROOT/INPUT)!=record['plan']['data']['input_manifest_sha256']:raise ValueError('Input manifest changed.')
    inputs=json.loads((ROOT/INPUT).read_text(encoding='utf-8'))
    for key in ('historical_source_catalog','h4_manifest','h7_manifest','h9_manifest','api_manifest'):
        if stock.digest(ROOT/inputs[key])!=inputs[key+'_sha256']:raise ValueError('Parent public evidence bytes differ.')
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    return record,inputs,code


def artifacts_for(inputs,key):
    return {x['id']:x['sha256'] for x in json.loads((ROOT/inputs[key]).read_text(encoding='utf-8'))['artifacts']}


def checked_file(folder,name,hashes):
    path=Path(folder)/name
    if stock.digest(path)!=hashes[name]:raise ValueError('Frozen private artifact differs: '+name)
    return path


def train_one(kind,seed,x,ax,prior,sums,exposure,data,artifacts):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=core.SeasonalGCN().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    fx=torch.as_tensor(x,device='cuda');neighbors=fx if ax is x else torch.as_tensor(ax,device='cuda')
    pr=torch.as_tensor(prior,dtype=torch.float32,device='cuda')
    total=torch.as_tensor(sums,dtype=torch.float32,device='cuda');n=torch.as_tensor(exposure,dtype=torch.float32,device='cuda')
    used=np.flatnonzero(exposure.sum(1)>0);valid=core.masks(data)['validation'];valid_context=np.unique(data['slots'][valid])
    best=float('inf');selected=None;trace=[];timer=time.perf_counter()
    def export(indices):
        with torch.inference_mode():
            return np.concatenate([model(fx[ss],neighbors[ss],pr[ss]).cpu().numpy() for ss in [indices[i:i+16] for i in range(0,len(indices),16)]])
    for epoch in range(21):
        if epoch:
            model.train();order=np.random.default_rng(seed+epoch).permutation(used)
            for start in range(0,len(order),32):
                ss=order[start:start+32]
                loss=core.poisson_objective(model(fx[ss],neighbors[ss],pr[ss]),total[ss],n[ss])
                if not torch.isfinite(loss):raise ValueError('Nonfinite historical objective.')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval();means=np.ones(prior.shape,dtype=np.float32);means[valid_context]=export(valid_context)
        score=core.count_score(data,valid,means);trace.append(dict(epoch=epoch,validation_recorded_count_deviance=score))
        if score<best:
            best=score;selected=epoch;torch.save(model.state_dict(),artifacts/f'{kind}-seed{seed}.pt')
        print(json.dumps(dict(stage='historical_epoch',kind=kind,seed=seed,epoch=epoch,validation_count_deviance=score,seconds=time.perf_counter()-timer)),flush=True)
    model.load_state_dict(torch.load(artifacts/f'{kind}-seed{seed}.pt',map_location='cuda',weights_only=True))
    model.eval();means=export(np.arange(slots.SLOTS))
    # Independently recompute selected outputs with the serialized model on CPU.
    cpu_model=core.SeasonalGCN();cpu_model.load_state_dict({k:v.cpu() for k,v in model.state_dict().items()});cpu_model.eval()
    chosen=np.array([0,210,slots.context_slot(9,0,8,0),slots.SLOTS-1])
    with torch.inference_mode():
        cpu=cpu_model(torch.as_tensor(x[chosen]),torch.as_tensor(ax[chosen]),torch.as_tensor(prior[chosen],dtype=torch.float32)).numpy()
    if not np.allclose(cpu,means[chosen],rtol=2e-5,atol=2e-5):raise ValueError('Saved model export gate failed.')
    info=dict(kind=kind,seed=seed,selected_epoch=selected,validation_recorded_count_deviance=best,trace=trace,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-timer,cpu_export_max_abs_error=float(np.abs(cpu-means[chosen]).max()))
    del model,cpu_model,optimizer,fx,neighbors,pr,total,n;torch.cuda.empty_cache()
    return means,info


def verify_lock(artifacts):
    lock=json.loads((Path(artifacts)/'selection.json').read_text(encoding='utf-8'))
    if lock['api_opened'] is not False:raise ValueError('Invalid fit/validation separation.')
    for item in lock['model_artifacts']:
        if stock.digest(Path(artifacts)/item['id'])!=item['sha256']:raise ValueError('Locked model changed.')
    return lock


def train(args):
    record,inputs,code=contract();started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter();prior_run.setup()
    directory=ROOT/'research/runs'/args.run_id
    if directory.exists():raise ValueError('Preserve prior training runs.')
    artifacts=directory/'artifacts';artifacts.mkdir(parents=True)
    h4=artifacts_for(inputs,'h4_manifest');h7=artifacts_for(inputs,'h7_manifest')
    with np.load(checked_file(args.h4_artifacts,'calendar.npz',h4),allow_pickle=False) as f:ids=f['station_ids'].copy()
    with np.load(checked_file(args.h7_artifacts,'graph.npz',h7),allow_pickle=False) as f:
        if not np.array_equal(ids,f['station_ids']):raise ValueError('Graph station order differs.')
        operator=f['od'].astype(np.float32)
    if stock.digest(args.archive)!=inputs['history_archive_sha256']:raise ValueError('Wrong historical archive.')
    catalog=json.loads((ROOT/inputs['historical_source_catalog']).read_text(encoding='utf-8'))
    events,source_ids,months,quality=hist.read_events(args.archive,catalog)
    data,scope=core.make_panel(events,source_ids,ids,months);del events,source_ids
    with np.load(checked_file(args.h4_artifacts,'prepared_panel.npz',h4),allow_pickle=False) as f:
        if not np.array_equal(ids,f['station_ids']):raise ValueError('Hourly reference ID order differs.')
        index=(f['timestamp_unix_seconds']-core.START)//3600;valid=f['observed_record_hours']
        hourly=data['counts'].reshape(-1,2,len(ids),2).sum(1)
        if not np.array_equal(hourly[index[valid]],f['recorded_counts'][valid]):raise ValueError('Historical hourly reconciliation failed.')
        if not np.array_equal(data['valid'].reshape(-1,2).all(1)[index],valid):raise ValueError('Historical observation-hour mask differs.')
    del hourly
    sums,exposure=core.sufficient_statistics(data)
    pooled,(base,weekly,seasonal),selected=core.choose_profiles(sums,exposure,data)
    rates=dict(pooled_calendar=2*pooled,seasonal_profile=2*seasonal);selections=[]
    x=core.features(base,weekly,seasonal)
    np.savez_compressed(artifacts/'context.npz',station_ids=ids,base=base,weekly=weekly,seasonal=seasonal,operator=operator)
    for kind in ('seasonal_self','seasonal_od'):
        ax=x if kind=='seasonal_self' else graph.aggregate_features(x,operator)
        outputs=[]
        for seed in core.SEEDS:
            values,info=train_one(kind,seed,x,ax,seasonal,sums,exposure,data,artifacts);outputs.append(values);selections.append(info)
        rates[kind]=2*np.mean(outputs,axis=0,dtype=np.float64)
        if ax is not x:del ax
        del outputs
    del x,sums
    for kind,values in rates.items():np.savez_compressed(artifacts/f'rates-{kind}.npz',station_ids=ids,means=values)
    numeric=graph.numerical_contract(np.concatenate([v.sum(-1).ravel() for v in rates.values()]))
    names=['context.npz',*[f'rates-{k}.npz' for k in KINDS],*[f'{k}-seed{s}.pt' for k in ('seasonal_self','seasonal_od') for s in core.SEEDS]]
    lock=dict(code_commit=code,selected_on='historical February completed counts only',primary_candidate='seasonal_od',api_opened=False,
        saved_before_auxiliary_march_and_api_utc=datetime.now(timezone.utc).isoformat(),profile_selection=selected,selections=selections,
        numerical_contract=numeric,model_artifacts=[dict(id=n,sha256=stock.digest(artifacts/n)) for n in names])
    stock.write(artifacts/'selection.json',lock);verify_lock(artifacts)
    mask=core.masks(data);diagnostics={k:core.count_score(data,mask['test'],v/2) for k,v in rates.items()}
    summary=dict(hypothesis_id='H-0010',phase='historical_fit_complete',started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=code,
        input_manifest_sha256=stock.digest(ROOT/INPUT),elapsed_seconds=time.perf_counter()-timer,source_quality=quality,source_scope=scope,
        cohort_stations=len(ids),training_exposure_station_halfhours=int(exposure.sum()),
        first_record_excluded_training_cells=int(mask['train'].sum()*len(ids)-exposure.sum()),historical_hourly_reconciled=True,
        historical_scope={k:dict(halfhours=int(m.sum()),recorded_count_pairs=int(m.sum()*len(ids))) for k,m in mask.items()},
        profile_selection=selected,selections=selections,march_auxiliary_recorded_count_deviance=diagnostics,numerical_contract=numeric,api_opened=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,scipy=scipy.__version__,device=torch.cuda.get_device_name(),
            cuda_peak_bytes=int(torch.cuda.max_memory_allocated()),compute_location='dislab',threads=4))
    stock.write(directory/'training.json',summary)
    print(json.dumps(dict(stage='historical_fit_complete',seconds=summary['elapsed_seconds'],profile_selection=selected,march_auxiliary=diagnostics)),flush=True)


def evaluate(args):
    record,inputs,code=contract();torch.set_num_threads(4);timer=time.perf_counter()
    directory=ROOT/'research/runs'/args.run_id;artifacts=directory/'artifacts';lock=verify_lock(artifacts)
    if (directory/'summary.json').exists():raise ValueError('Preserve completed evaluations.')
    training=json.loads((directory/'training.json').read_text(encoding='utf-8'))
    if training['api_opened'] is not False:raise ValueError('Fit phase read API.')
    api=json.loads((ROOT/inputs['api_manifest']).read_text(encoding='utf-8'))
    if stock.digest(args.panel)!=api['derived_files'][0]['sha256']:raise ValueError('Wrong audited API panel.')
    with np.load(artifacts/'context.npz',allow_pickle=False) as f:ids=f['station_ids'].copy()
    with np.load(args.panel,allow_pickle=False) as f:data=stock.pair_data(f['stock'],f['scheduled_at_unix_seconds'],f['observed_at_unix_seconds'],f['station_ids'],ids)
    row,col=np.nonzero(data['valid']);shape=data['current'].shape
    contexts=slots.context_slot(9,data['weekday'],data['hour'],0);predictions={};k=lock['numerical_contract']['cutoff']
    for kind in KINDS:
        with np.load(artifacts/f'rates-{kind}.npz',allow_pickle=False) as f:
            if not np.array_equal(ids,f['station_ids']):raise ValueError('Forecast ID order differs.')
            rates=f['means']
        out={key:np.full(shape,np.nan) for key in KEYS}
        for start in range(0,len(row),512):
            rr,cc=row[start:start+512],col[start:start+512]
            means=rates[contexts[rr],data['model_cols'][cc]]
            grid,pmf=probability.state_pmf(data['current'][rr,cc],means[:,1],means[:,0],k)
            values=probability.summaries(grid,pmf)
            for key in KEYS:
                out[key][rr,cc]=probability.integer_crps(grid,pmf,data['future'][rr,cc]) if key=='crps' else values[key]
        predictions[kind]=out
        print(json.dumps(dict(stage='official_candidate_scored',kind=kind,crps=prior_run.grouped(data,out)['all']['crps'])),flush=True)
    # Fixed benchmark predictions are admitted only after the new models are locked.
    for key,folder,name,manifest in [('h4_calendar',args.h7_artifacts,'predictions-calendar.npz','h7_manifest'),
            ('h9_full',args.h9_artifacts,'predictions-large_seoul_cfm.npz','h9_manifest')]:
        hashes=artifacts_for(inputs,manifest);path=checked_file(folder,name,hashes)
        if key=='h9_full':
            with np.load(checked_file(folder,'context.npz',hashes),allow_pickle=False) as f:
                if not np.array_equal(ids,f['station_ids']):raise ValueError('H9 IDs differ.')
        with np.load(path,allow_pickle=False) as f:
            if not np.array_equal(f['valid'],data['valid']) or not np.array_equal(f['times'],data['times']):raise ValueError('Reference evaluation alignment differs.')
            if 'initial' in f and not np.array_equal(f['initial'],data['current'],equal_nan=True):raise ValueError('Reference initial states differ.')
            predictions[key]={key:f[key].copy() for key in KEYS}
    daily=np.zeros((14,4))
    for day in range(14):
        mask=data['valid']&(data['day_index'][:,None]==day)
        daily[day]=[*[predictions[key]['crps'][mask].sum() for key in ('pooled_calendar','h9_full','seasonal_od')],int(mask.sum())]
    total=daily.sum(0);gains=1-total[2]/total[:2];point=float(gains.min());intervals=[core.bootstrap(daily,b) for b in (3,7)]
    enough=data['scope']['matched_ids']>=500 and total[-1]>=10000 and (daily[:,-1]>0).sum()>=10 and all(x['valid_replicates']>=1900 for x in intervals)
    decision='inconclusive'
    if enough and point>.01 and all(x['interval'][0]>.01 for x in intervals):decision='supported'
    elif enough and all(x['interval'][1]<=.01 for x in intervals):decision='not_supported'
    metrics=dict(worst_control_relative_crps_gain=point,valid_pairs=int(total[-1]))
    for interval in intervals:metrics[f"ci_{interval['block_days']}day_lower"],metrics[f"ci_{interval['block_days']}day_upper"]=interval['interval']
    valid=data['valid'];error=np.abs(data['current'][valid]-data['future'][valid])
    limitations=['Repeatedly exposed September official evaluation, not an untouched independent test; no API fitting/selection/calibration.',
        'February and March are completed-count data, not measured inventory; March was already used in prior research.',
        'First historical record is only a conservative record-support proxy, not physical opening date or measured supply exposure.',
        'Static calendar/OD inputs omit current weather, recent transactions, operator moves and observation process.',
        'Poisson zero-reflection has no measured physical parking cap; completed flows are not identified latent demand.',
        'Primary comparison is a seasonality-plus-GCN package; graph effect is a separate auxiliary contrast; intervals omit fit/selection uncertainty.']
    summary=dict(hypothesis_id='H-0010',decision=decision,metrics=metrics,scope=data['scope'],groups={key:prior_run.grouped(data,out) for key,out in predictions.items()},
        control_relative_crps_gains=dict(zip(('pooled_calendar','h9_full'),gains.tolist())),bootstraps=intervals,
        persistence=dict(crps=float(error.mean()),median_mae=float(error.mean())),api_training_or_selection=False,
        numerical_contract=lock['numerical_contract'],elapsed_evaluation_seconds=time.perf_counter()-timer,limitations=limitations)
    stock.write(directory/'summary.json',summary)
    for key,out in predictions.items():np.savez_compressed(artifacts/f'predictions-{key}.npz',**out,valid=data['valid'],times=data['times'])
    verify_lock(artifacts)
    files=[dict(id=n,sha256=stock.digest(directory/n),availability='public aggregate') for n in ('training.json','summary.json')]
    files.extend(dict(id=p.name,sha256=stock.digest(p),availability='private model/rates/individual predictions; not published') for p in sorted(artifacts.iterdir()))
    stock.write(directory/'manifest.json',dict(schema_version=1,run_id=args.run_id,hypothesis_id='H-0010',plan_sha256=record['preregistration']['plan_sha256'],
        code_commit=code,input_manifest_sha256=stock.digest(ROOT/INPUT),started_at_utc=training['started_at_utc'],finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=[['python','research_code/seasonal_history.py','train','--archive','<PRIVATE_HISTORY_ARCHIVE>','--h4-artifacts','<PRIVATE_H4_ARTIFACTS>','--h7-artifacts','<PRIVATE_H7_ARTIFACTS>','--run-id',args.run_id],
            ['python','research_code/seasonal_history.py','evaluate','--panel','<PRIVATE_AUDITED_API_PANEL>','--h7-artifacts','<PRIVATE_H7_ARTIFACTS>','--h9-artifacts','<PRIVATE_H9_ARTIFACTS>','--run-id',args.run_id]],
        environment=training['environment'],seed=list(core.SEEDS),evaluation_scope=data['scope'],metrics=metrics,artifacts=files,limitations=limitations))
    print(json.dumps(dict(stage='official_evaluation_complete',decision=decision,metrics=metrics,groups=summary['groups'])),flush=True)


def predict(args):
    root=Path(args.artifacts);lock=verify_lock(root)
    dt=datetime.fromtimestamp(hist.stamp(args.issued_at),hist.KST)
    if dt.minute not in (0,30) or dt.second or not 7<=dt.hour<=21:raise ValueError('Supports full halfhours 07:00 through 21:30 KST.')
    if args.parking_count<0:raise ValueError('Current state must be nonnegative.')
    with np.load(root/'rates-seasonal_od.npz',allow_pickle=False) as f:
        found=np.flatnonzero(f['station_ids']==args.station_id)
        if len(found)!=1:raise ValueError('Unsupported historical cohort station.')
        slot=int(slots.context_slot(dt.month,dt.weekday(),dt.hour,int(dt.minute>=30)));rate=f['means'][slot,found[0]]
    grid,pmf=probability.state_pmf(np.array([args.parking_count]),rate[None,1],rate[None,0],lock['numerical_contract']['cutoff'])
    out=probability.summaries(grid,pmf)
    print(json.dumps(dict(model='H-0010 seasonal_od research candidate',station_id=args.station_id,issued_at=dt.isoformat(),horizon_minutes=30,
        current_parking_count=args.parking_count,**{key:float(out[key][0]) for key in ('mean','median','lower','upper','pzero')},
        truth_used=False,model_selected_on='historical February counts; not API'),ensure_ascii=False))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    fit=sub.add_parser('train')
    for key in ('archive','h4-artifacts','h7-artifacts','run-id'):fit.add_argument('--'+key,required=True)
    ev=sub.add_parser('evaluate')
    for key in ('panel','h7-artifacts','h9-artifacts','run-id'):ev.add_argument('--'+key,required=True)
    pred=sub.add_parser('predict')
    for key in ('artifacts','station-id','issued-at'):pred.add_argument('--'+key,required=True)
    pred.add_argument('--parking-count',required=True,type=int)
    args=ap.parse_args();globals()[args.command](args)
