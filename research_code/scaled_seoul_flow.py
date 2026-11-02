"""H9 train -> lock -> cache -> September evaluate. Run on dislab only."""
import argparse,hashlib,json,os,platform,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy,torch
from torch import nn
import scaled_flow_core as c
import historical_calendar as hist
import historical_od_gcn as old
import history_to_stock as stock
import loss_consistent_stock as probability

ROOT=Path(__file__).resolve().parents[1]
INPUT='research/datasets/tashu-scaled-seoul-flow-H0009-v1.json'


def contract():
    record=json.loads((ROOT/'research/hypotheses/H-0009.json').read_text(encoding='utf-8'))
    canonical=json.dumps(record['plan'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256']:raise ValueError('Frozen plan mismatch.')
    if stock.digest(ROOT/INPUT)!=record['plan']['data']['input_manifest_sha256']:raise ValueError('Input manifest mismatch.')
    inputs=json.loads((ROOT/INPUT).read_text(encoding='utf-8'))
    for key in ('source_catalog','h4_manifest','h7_manifest','historical_source_catalog','api_manifest'):
        if stock.digest(ROOT/inputs[key])!=inputs[key+'_sha256']:raise ValueError('Parent input bytes differ.')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True)
    return record,inputs,code


def setup():
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise ValueError('Frozen CUDA budget unavailable.')
    torch.cuda.set_per_process_memory_fraction(.35);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False


def artifact_map(manifest):return {x['id']:x['sha256'] for x in manifest['artifacts']}


def load_sources(args,inputs):
    h4=artifact_map(json.loads((ROOT/inputs['h4_manifest']).read_text(encoding='utf-8')))
    h7=artifact_map(json.loads((ROOT/inputs['h7_manifest']).read_text(encoding='utf-8')))
    base=Path(args.h4_artifacts);other=Path(args.h7_artifacts);models={};ids=None
    for name in ('constant','calendar'):
        if stock.digest(base/f'{name}.npz')!=h4[f'{name}.npz']:raise ValueError('Frozen historical prior differs.')
        with np.load(base/f'{name}.npz',allow_pickle=False) as f:
            if ids is not None and not np.array_equal(ids,f['station_ids']):raise ValueError('Prior station order differs.')
            ids=f['station_ids'].copy();models[name]=f['means'].copy()
    if stock.digest(base/'prepared_panel.npz')!=inputs['historical_panel_sha256']:raise ValueError('Historical panel differs.')
    with np.load(base/'prepared_panel.npz',allow_pickle=False) as f:
        if not np.array_equal(ids,f['station_ids']):raise ValueError('Historical panel order differs.')
        hourly={k:f[name].copy() for k,name in [('y','recorded_counts'),('times','timestamp_unix_seconds'),('valid','observed_record_hours')]}
    if stock.digest(other/'graph.npz')!=h7['graph.npz']:raise ValueError('Frozen training graph differs.')
    with np.load(other/'graph.npz',allow_pickle=False) as f:
        if not np.array_equal(ids,f['station_ids']):raise ValueError('Graph station order differs.')
        operator=f['od'].astype(np.float32)
    if stock.digest(args.archive)!=inputs['history_archive_sha256'] or stock.digest(args.seoul)!=inputs['seoul_csv_sha256']:raise ValueError('Historical source differs.')
    return ids,models,hourly,operator


def batched_rates(model,features,operator,prior,slots):
    parts=[]
    with torch.inference_mode():
        for start in range(0,len(slots),8):
            chosen=slots[start:start+8];parts.append(model(features[chosen],operator,prior[chosen]).cpu().numpy())
    return np.concatenate(parts)


def train_rate(kind,seed,x,prior,operator,data,masks,artifacts):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    width,depth=(32,1) if kind=='small' else (128,3)
    model=c.RateEncoder(width,depth).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    fx=torch.as_tensor(x,device='cuda');pr=torch.as_tensor(prior,device='cuda')
    labels=torch.as_tensor(data['counts'][masks['train']],dtype=torch.float32,device='cuda')
    slots=data['slots'][masks['train']];valid_slots=data['slots'][masks['validation']];unique,inverse=np.unique(valid_slots,return_inverse=True)
    best=float('inf');trace=[];selected=None;timer=time.perf_counter()
    for epoch in range(21):
        if epoch:
            model.train();order=np.random.default_rng(seed+epoch).permutation(len(labels))
            for start in range(0,len(order),32):
                batch=order[start:start+32];ss=slots[batch]
                prediction=model(fx[ss],operator,pr[ss])/2
                loss=(prediction-labels[batch]*torch.log(prediction)).mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite historical rate loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval();means=batched_rates(model,fx,operator,pr,unique)/2
        score=float(hist.deviance(data['counts'][masks['validation']],means[inverse]).mean())
        trace.append(dict(epoch=epoch,validation_deviance=score))
        if score<best:
            best=score;selected=epoch
            torch.save(dict(width=width,depth=depth,state_dict=model.state_dict()),artifacts/f'{kind}-seed{seed}.pt')
        print(json.dumps(dict(stage='rate_epoch',kind=kind,seed=seed,epoch=epoch,validation_count_deviance=score,elapsed_seconds=time.perf_counter()-timer)),flush=True)
    model=c.load_encoder(artifacts/f'{kind}-seed{seed}.pt','cuda')
    exported=batched_rates(model,fx,operator,pr,np.arange(c.SLOTS))
    parameters=sum(p.numel() for p in model.parameters())
    del model,optimizer,fx,pr,labels;torch.cuda.empty_cache()
    return exported,dict(kind=kind,seed=seed,selected_epoch=selected,validation_deviance=best,parameters=parameters,trace=trace)


def train_flow(seed,x,prior,operator,data,masks,artifacts):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    encoder=c.load_encoder(artifacts/f'large_seoul-seed{seed}.pt','cuda')
    train_slots=data['slots'][masks['train']];valid_slots=data['slots'][masks['validation']]
    used=np.unique(np.r_[train_slots,valid_slots]);slot_map=np.zeros(c.SLOTS,dtype=int);slot_map[used]=np.arange(len(used))
    n=data['counts'].shape[1];context=torch.empty((len(used),n,128),device='cuda');center=torch.empty((len(used),n,2),device='cuda')
    with torch.inference_mode():
        for start in range(0,len(used),8):
            ss=used[start:start+8];rate,hidden=encoder(torch.as_tensor(x[ss],device='cuda'),operator,torch.as_tensor(prior[ss],device='cuda'),with_hidden=True)
            context[start:start+len(ss)]=hidden;center[start:start+len(ss)]=torch.log1p(rate/2)
    del encoder;torch.cuda.empty_cache()
    labels=torch.as_tensor(data['counts'][masks['train']].reshape(-1,2),dtype=torch.float32,device='cuda')
    ss=torch.as_tensor(slot_map[train_slots],device='cuda')
    valid_labels=data['counts'][masks['validation']].reshape(-1,2)
    chosen=np.linspace(0,len(valid_labels)-1,min(131072,len(valid_labels))).astype(int)
    vs=torch.as_tensor(slot_map[valid_slots[chosen//n]],device='cuda');vn=torch.as_tensor(chosen%n,device='cuda')
    vy=torch.as_tensor(valid_labels[chosen],dtype=torch.float32,device='cuda')
    model=c.CountFlow().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    generator=torch.Generator(device='cuda').manual_seed(seed);best=float('inf');trace=[];selected=None;timer=time.perf_counter()
    for epoch in range(21):
        if epoch:
            model.train();order=torch.randperm(len(labels),device='cuda',generator=generator)
            for start in range(0,len(order),16384):
                batch=order[start:start+16384];rr=ss[batch//n];cc=batch%n
                loss=c.flow_loss(model,labels[batch],context[rr,cc],center[rr,cc],generator)
                if not torch.isfinite(loss):raise ValueError('Nonfinite CFM historical loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        model.eval();valid_generator=torch.Generator(device='cuda').manual_seed(20261031+seed);total=0.
        with torch.inference_mode():
            for start in range(0,len(chosen),32768):
                rr,cc=vs[start:start+32768],vn[start:start+32768]
                loss=c.flow_loss(model,vy[start:start+32768],context[rr,cc],center[rr,cc],valid_generator)
                total+=float(loss)*len(rr)
        score=total/len(chosen);trace.append(dict(epoch=epoch,validation_fixed_cfm_mse=score))
        if score<best:
            best=score;selected=epoch;torch.save(model.state_dict(),artifacts/f'cfm-seed{seed}.pt')
        print(json.dumps(dict(stage='cfm_epoch',seed=seed,epoch=epoch,validation_fixed_cfm_mse=score,elapsed_seconds=time.perf_counter()-timer)),flush=True)
    parameters=sum(p.numel() for p in model.parameters())
    del model,optimizer,context,center,labels,vy,ss,vs,vn;torch.cuda.empty_cache()
    return dict(kind='large_seoul_cfm',seed=seed,selected_epoch=selected,validation_fixed_cfm_mse=best,parameters=parameters,trace=trace)


def load_context(artifacts,enabled=True):
    with np.load(Path(artifacts)/'context.npz',allow_pickle=False) as f:
        ids=f['station_ids'].copy();operator=f['operator'].copy();x,prior=c.feature_table(f['constant'],f['calendar'],f['seoul'],enabled)
    return ids,x,prior,operator


def load_fusion(artifacts,device='cuda'):
    models=[c.load_encoder(Path(artifacts)/f'large_seoul-seed{s}.pt',device) for s in c.SEEDS];flows=[]
    for seed in c.SEEDS:
        model=c.CountFlow().to(device);model.load_state_dict(torch.load(Path(artifacts)/f'cfm-seed{seed}.pt',map_location=device,weights_only=True));model.eval();flows.append(model)
    return models,flows


def verify_lock(artifacts):
    lock=json.loads((Path(artifacts)/'selection.json').read_text(encoding='utf-8'))
    for item in lock['model_artifacts']:
        if stock.digest(Path(artifacts)/item['id'])!=item['sha256']:raise ValueError('Locked historical model changed.')
    return lock


def train(args):
    started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter();record,inputs,code=contract();setup()
    directory=ROOT/'research/runs'/args.run_id
    if directory.exists():raise ValueError('Preserve existing training run.')
    artifacts=directory/'artifacts';artifacts.mkdir(parents=True)
    ids,models,hourly,operator=load_sources(args,inputs)
    catalog=json.loads((ROOT/inputs['historical_source_catalog']).read_text(encoding='utf-8'))
    events,source_ids,_,quality=hist.read_events(args.archive,catalog)
    data=c.half_hour_panel(events,source_ids,ids,hourly['times'],hourly['y'],hourly['valid']);del events,hourly
    masks=hist.split_masks(dict(times=data['times'],valid=data['valid']))
    seoul,source_info=c.seoul_profiles(args.seoul)
    np.savez_compressed(artifacts/'context.npz',station_ids=ids,operator=operator,constant=models['constant'],calendar=models['calendar'],seoul=seoul)
    np.savez_compressed(artifacts/'historical-halfhour.npz',**data)
    gpu_operator=torch.as_tensor(operator,device='cuda');selections=[];rates={}
    for kind in c.KINDS:
        x,prior=c.feature_table(models['constant'],models['calendar'],seoul,kind=='large_seoul')
        outputs=[]
        for seed in c.SEEDS:
            means,selection=train_rate(kind,seed,x,prior,gpu_operator,data,masks,artifacts);outputs.append(means);selections.append(selection)
        rates[kind]=np.mean(outputs,axis=0,dtype=np.float64)
        np.savez_compressed(artifacts/f'rates-{kind}.npz',station_ids=ids,means=rates[kind])
        del x,prior,outputs
    x,prior=c.feature_table(models['constant'],models['calendar'],seoul,True)
    for seed in c.SEEDS:selections.append(train_flow(seed,x,prior,gpu_operator,data,masks,artifacts))
    sums=[v.sum(-1).ravel() for v in rates.values()]
    for kind in ('od','calendar'):
        with np.load(Path(args.h7_artifacts)/f'{kind}.npz',allow_pickle=False) as f:sums.append(f['means'].sum(-1).ravel())
    numeric=old.numerical_contract(np.concatenate(sums))
    names=['context.npz',*[f'rates-{k}.npz' for k in c.KINDS],*[f'{k}-seed{s}.pt' for k in c.KINDS for s in c.SEEDS],*[f'cfm-seed{s}.pt' for s in c.SEEDS]]
    lock=dict(code_commit=code,selected_on='historical February only; no API/March selection',selections=selections,numerical_contract=numeric,
        saved_before_march_and_api_utc=datetime.now(timezone.utc).isoformat(),model_artifacts=[dict(id=n,sha256=stock.digest(artifacts/n)) for n in names],api_opened=False)
    stock.write(artifacts/'selection.json',lock)
    test=masks['test'];diagnostics={}
    for kind,rate in rates.items():diagnostics[kind]=dict(recorded_count_deviance=float(hist.deviance(data['counts'][test],rate[data['slots'][test]]/2).mean()))
    # A fixed March subset is a count-only diagnostic after selection is locked.
    encoders,flows=load_fusion(artifacts);test_labels=data['counts'][test].reshape(-1,2);test_slots=data['slots'][test]
    chosen=np.linspace(0,len(test_labels)-1,min(65536,len(test_labels))).astype(int);pred=np.empty((len(chosen),512,2),dtype=np.int32)
    selected_slots=test_slots[chosen//len(ids)]
    for slot in np.unique(selected_slots):
        mask=selected_slots==slot;nodes,inverse=np.unique(chosen[mask]%len(ids),return_inverse=True)
        counts,_=c.generate_counts(encoders,flows,x,gpu_operator,prior,int(slot),nodes=nodes);pred[mask]=counts[inverse]
    targets=test_labels[chosen];mae=float(np.abs(pred.mean(1)-targets).mean())
    ordered=np.sort(pred,axis=1);crps=np.abs(pred-targets[:,None,:]).mean(1)-(ordered*(2*np.arange(1,513)-513)[None,:,None]).sum(1)/(512**2)
    diagnostics['large_seoul_cfm']=dict(count_pairs=len(chosen),sample_mean_count_mae=mae,empirical_count_crps=float(crps.mean()))
    summary=dict(hypothesis_id='H-0009',phase='historical_training_complete',code_commit=code,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        input_manifest_sha256=stock.digest(ROOT/INPUT),elapsed_seconds=time.perf_counter()-timer,selections=selections,seoul_source=source_info,source_quality=quality,
        historical_scope={key:dict(halfhours=int(mask.sum()),count_pairs=int(mask.sum()*len(ids))) for key,mask in masks.items()},
        numerical_contract=numeric,march_auxiliary_recorded_counts=diagnostics,api_opened=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,device=torch.cuda.get_device_name(),cuda_peak_bytes=int(torch.cuda.max_memory_allocated()),compute_location='dislab',threads=4))
    verify_lock(artifacts);stock.write(directory/'training.json',summary)
    print(json.dumps(dict(stage='historical_training_complete',seconds=summary['elapsed_seconds'],march_auxiliary=diagnostics)),flush=True)


def cache(args):
    record,inputs,code=contract();setup();directory=ROOT/'research/runs'/args.run_id;artifacts=directory/'artifacts';lock=verify_lock(artifacts)
    if (artifacts/'cfm-september.npz').exists():raise ValueError('Preserve fixed generation cache.')
    ids,x,prior,operator=load_context(artifacts);op=torch.as_tensor(operator,device='cuda');models,flows=load_fusion(artifacts)
    slots=np.array([c.context_slot(9,w,h,0) for w in range(7) for h in range(7,22)],dtype=int)
    delta=np.empty((len(slots),len(ids),512),dtype=np.int32);depth=np.empty_like(delta);timer=time.perf_counter()
    for j,slot in enumerate(slots):
        counts,seed=c.generate_counts(models,flows,x,op,prior,int(slot));delta[j]=counts[:,:,1]-counts[:,:,0]
        depth[j]=c.reflected_depth(counts[:,:,0],counts[:,:,1],np.random.default_rng(seed))
        print(json.dumps(dict(stage='fixed_generation_context',index=j+1,total=len(slots),elapsed_seconds=time.perf_counter()-timer)),flush=True)
    np.savez_compressed(artifacts/'cfm-september.npz',station_ids=ids,slots=slots,delta=delta,depth=depth)
    verify_lock(artifacts);stock.write(artifacts/'generation-lock.json',dict(code_commit=code,api_opened=False,saved_before_api_utc=datetime.now(timezone.utc).isoformat(),
        sha256=stock.digest(artifacts/'cfm-september.npz'),elapsed_seconds=time.perf_counter()-timer,contexts=len(slots),samples=512))


def grouped(data,predictions):
    groups={}
    for name,mask in [('all',data['valid']),('zero',data['valid']&(data['current']==0)),('positive',data['valid']&(data['current']>0))]:
        target=data['future'][mask];p={key:value[mask] for key,value in predictions.items()}
        groups[name]=dict(pairs=int(mask.sum()),crps=float(p['crps'].mean()),brier=float(((p['pzero']-(target==0))**2).mean()),
            median_mae=float(np.abs(p['median']-target).mean()),mean_rmse=float(np.sqrt(((p['mean']-target)**2).mean())),coverage_5_95=float(((target>=p['lower'])&(target<=p['upper'])).mean()))
    return groups


def bootstraps(daily,block):
    rng=np.random.default_rng(20261031);days=len(daily);starts=rng.integers(0,days-block+1,(2000,(days+block-1)//block))
    index=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:days];total=daily[index].sum(1)
    valid=(total[:,:5]>0).all(1)&(total[:,-1]>0);gain=1-total[valid,5,None]/total[valid,:5]
    return dict(block_days=block,valid_replicates=int(valid.sum()),interval=np.percentile(gain.min(1),[2.5,97.5]).tolist())


def evaluate(args):
    started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter();record,inputs,code=contract();torch.set_num_threads(4)
    directory=ROOT/'research/runs'/args.run_id;artifacts=directory/'artifacts';lock=verify_lock(artifacts)
    if (directory/'summary.json').exists():raise ValueError('Preserve completed evaluation.')
    training=json.loads((directory/'training.json').read_text(encoding='utf-8'));generation=json.loads((artifacts/'generation-lock.json').read_text(encoding='utf-8'))
    if not (training['api_opened'] is False and generation['api_opened'] is False and generation['sha256']==stock.digest(artifacts/'cfm-september.npz')):raise ValueError('History/cache lock gate failed.')
    source=json.loads((ROOT/inputs['api_manifest']).read_text(encoding='utf-8'))
    if stock.digest(args.panel)!=source['derived_files'][0]['sha256']:raise ValueError('Audited API panel hash differs.')
    with np.load(artifacts/'context.npz',allow_pickle=False) as f:ids=f['station_ids'].copy()
    with np.load(args.panel,allow_pickle=False) as f:data=stock.pair_data(f['stock'],f['scheduled_at_unix_seconds'],f['observed_at_unix_seconds'],f['station_ids'],ids)
    shape=data['current'].shape;row,col=np.nonzero(np.isfinite(data['current']));slots=c.context_slot(9,data['weekday'],data['hour'],0)
    predictions={};k=lock['numerical_contract']['cutoff']
    for kind in c.KINDS:
        with np.load(artifacts/f'rates-{kind}.npz',allow_pickle=False) as f:
            if not np.array_equal(ids,f['station_ids']):raise ValueError('Rate station order differs.')
            rates=f['means']
        out={key:np.full(shape,np.nan) for key in ('mean','median','lower','upper','pzero','crps')}
        for start in range(0,len(row),512):
            rr,cc=row[start:start+512],col[start:start+512];rate=rates[slots[rr],data['model_cols'][cc]]
            grid,pmf=probability.state_pmf(data['current'][rr,cc],rate[:,1],rate[:,0],k)
            values=probability.summaries(grid,pmf)
            for key in out:
                if key!='crps':out[key][rr,cc]=values[key]
            out['crps'][rr,cc]=probability.integer_crps(grid,pmf,data['future'][rr,cc])
        predictions[kind]=out;print(json.dumps(dict(stage='official_candidate_scored',kind=kind,crps=grouped(data,out)['all']['crps'])),flush=True)
    h7=artifact_map(json.loads((ROOT/inputs['h7_manifest']).read_text(encoding='utf-8')))
    for kind in ('od','calendar'):
        path=Path(args.h7_artifacts)/f'predictions-{kind}.npz'
        if stock.digest(path)!=h7[path.name]:raise ValueError('Fixed H7 prediction reference differs.')
        with np.load(path,allow_pickle=False) as f:
            if not np.array_equal(f['initial'],data['current'],equal_nan=True) or not np.array_equal(f['times'],data['times']) or not np.array_equal(f['valid'],data['valid']):raise ValueError('Reference pair alignment differs.')
            predictions['h7_od' if kind=='od' else 'calendar']={key:f[key].copy() for key in ('mean','median','lower','upper','pzero','crps')}
    out={key:np.full(shape,np.nan) for key in ('mean','median','lower','upper','pzero','crps')}
    with np.load(artifacts/'cfm-september.npz',allow_pickle=False) as f:
        if not np.array_equal(ids,f['station_ids']):raise ValueError('Generation station order differs.')
        index={int(s):i for i,s in enumerate(f['slots'])};delta,depth=f['delta'],f['depth']
        for start in range(0,len(row),512):
            rr,cc=row[start:start+512],col[start:start+512];ci=np.array([index[int(s)] for s in slots[rr]]);node=data['model_cols'][cc]
            samples=c.stock_samples(data['current'][rr,cc],delta[ci,node],depth[ci,node]);values=c.sample_summaries(samples,data['future'][rr,cc])
            for key in out:out[key][rr,cc]=values[key]
    predictions['large_seoul_cfm']=out
    kinds=['small','large','large_seoul','h7_od','calendar','large_seoul_cfm'];daily=np.zeros((14,7))
    for day in range(14):
        mask=data['valid']&(data['day_index'][:,None]==day);daily[day]=[*[predictions[k]['crps'][mask].sum() for k in kinds],int(mask.sum())]
    totals=daily.sum(0);gains=1-totals[5]/totals[:5];point=float(gains.min());intervals=[bootstraps(daily,b) for b in (3,7)]
    enough=data['scope']['matched_ids']>=500 and int(totals[-1])>=10000 and (daily[:,-1]>0).sum()>=10 and all(x['valid_replicates']>=1900 for x in intervals)
    decision='inconclusive'
    if enough and point>.01 and all(x['interval'][0]>.01 for x in intervals):decision='supported'
    elif enough and all(x['interval'][1]<=.01 for x in intervals):decision='not_supported'
    metrics=dict(worst_control_relative_crps_gain=point,valid_pairs=int(totals[-1]))
    for b in intervals:metrics[f"ci_{b['block_days']}day_lower"],metrics[f"ci_{b['block_days']}day_upper"]=b['interval']
    persistence=np.abs(data['current']-data['future']);valid=data['valid']
    persistence_brier=float((((data['current'][valid]==0).astype(float)-(data['future'][valid]==0))**2).mean())
    limitations=['Official September repeatedly exposed, not an untouched independent test; no API training/tuning/model selection/calibration.',
        'Seoul source is 2017-18 city aggregate rental patterns, not matched Seoul stations/current conditions/Daejeon weather.',
        'Only calendar/static graph context available, no recent September transactions; counts are completed flow, not latent demand/exposure.',
        'CFM is joint rental/return per station only, not a citywide joint distribution; Gaussian residual flow/Heun16/dequantization/512 samples are approximations.',
        'Uniform event order and zero reflection omit capacity/operator moves/observation process; no February/March inventory truth.',
        'Block intervals exclude fitting, selection and Monte Carlo uncertainty; service candidate inclusion follows user request, not API ranking.']
    summary=dict(hypothesis_id='H-0009',decision=decision,metrics=metrics,scope=data['scope'],groups={key:grouped(data,value) for key,value in predictions.items()},
        persistence=dict(crps=float(persistence[valid].mean()),median_mae=float(persistence[valid].mean()),brier=persistence_brier),
        control_relative_crps_gains=dict(zip(kinds[:5],gains.tolist())),bootstraps=intervals,numerical_contract=lock['numerical_contract'],
        api_training_or_selection=False,elapsed_evaluation_seconds=time.perf_counter()-timer,limitations=limitations)
    stock.write(directory/'summary.json',summary)
    for key,out in predictions.items():np.savez_compressed(artifacts/f'predictions-{key}.npz',**out,valid=data['valid'],times=data['times'])
    verify_lock(artifacts)
    records=[dict(id=name,sha256=stock.digest(directory/name),availability='public aggregate') for name in ('training.json','summary.json')]
    records += [dict(id=p.name,sha256=stock.digest(p),availability='private weights/counts/profiles/generation/predictions; not published') for p in sorted(artifacts.iterdir())]
    stock.write(directory/'manifest.json',dict(schema_version=1,run_id=args.run_id,hypothesis_id='H-0009',plan_sha256=record['preregistration']['plan_sha256'],code_commit=code,
        input_manifest_sha256=stock.digest(ROOT/INPUT),started_at_utc=training['started_at_utc'],finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=[['python','research_code/scaled_seoul_flow.py','train','--archive','<PRIVATE_HISTORY_ARCHIVE>','--seoul','<PRIVATE_OFFICIAL_SEOUL_CSV>','--h4-artifacts','<PRIVATE_H4_ARTIFACTS>','--h7-artifacts','<PRIVATE_H7_ARTIFACTS>','--run-id',args.run_id],
            ['python','research_code/scaled_seoul_flow.py','cache','--run-id',args.run_id],
            ['python','research_code/scaled_seoul_flow.py','evaluate','--panel','<PRIVATE_AUDITED_API_PANEL>','--h7-artifacts','<PRIVATE_H7_ARTIFACTS>','--run-id',args.run_id]],
        environment={**training['environment'],'evaluation_device':'CPU float64, 4 threads','scipy':scipy.__version__},seed=list(c.SEEDS),evaluation_scope=data['scope'],metrics=metrics,artifacts=records,limitations=limitations))
    print(json.dumps(dict(stage='official_evaluation_complete',decision=decision,metrics=metrics,groups=summary['groups'])),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    trainp=sub.add_parser('train')
    for name in ('archive','seoul','h4-artifacts','h7-artifacts','run-id'):trainp.add_argument('--'+name,required=True)
    cachep=sub.add_parser('cache');cachep.add_argument('--run-id',required=True)
    ev=sub.add_parser('evaluate')
    for name in ('panel','h7-artifacts','run-id'):ev.add_argument('--'+name,required=True)
    args=ap.parse_args();globals()[args.command](args)
