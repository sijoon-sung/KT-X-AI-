"""H14 official inventory evaluation, only after historical gate and model lock."""
import argparse,json,time,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np
import torch
import historical_calendar as h
import rolling_inventory_data as dates
import rolling_inventory_predict as obs
import forecast_transfer as weather
import joint_inventory_core as m

ROOT=Path(__file__).resolve().parents[1];NAMES=('independent','gaussian','flow','flow_shuffle','persistence')
def emit(**v):print(json.dumps(v,allow_nan=False),flush=True)
def intervals(daily,regions):
    total=daily.sum(0);controls=[0,1,4];gain=1-total[2]/total[controls];point=float(gain.min());result=[]
    for block in (3,7):
        rng=np.random.default_rng(20261104);n=len(daily);starts=rng.integers(0,n-block+1,(2000,(n+block-1)//block));ix=(starts[...,None]+np.arange(block)).reshape(2000,-1)[:,:n];s=daily[ix].sum(1)
        valid=(s[:,controls]>0).all(1)&(s[:,-1]>0);value=(1-s[valid,2,None]/s[valid][:,controls]).min(1);result.append(dict(block_days=block,valid_replicates=int(valid.sum()),interval=np.percentile(value,[2.5,97.5]).tolist()))
    enough=regions>=16 and total[-1]>=10000 and (daily[:,-1]>0).sum()>=10 and all(x['valid_replicates']>=1900 for x in result)
    decision='supported' if enough and point>.01 and all(x['interval'][0]>.01 for x in result) else ('not_supported' if enough and all(x['interval'][1]<=.01 for x in result) else 'inconclusive')
    return dict(worst_control_regional_zero_fraction_crps_relative_gain=point,relative_gain_by_control=dict(zip(('independent','gaussian','persistence'),gain.tolist())),intervals=result,sufficient=bool(enough),decision=decision)

def evaluate(a):
    start=time.perf_counter();reference=datetime.fromisoformat(a.collector_start);now=lambda:(reference+timedelta(seconds=time.perf_counter()-start)).isoformat();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    if (out/'results.json').exists():raise ValueError('Preserve completed official evaluation.')
    fit=Path(a.fit);lock=json.loads((fit/'model-lock.json').read_text(encoding='utf-8'))
    if lock['api_opened'] is not False or lock['historical_gate_passed'] is not True:raise ValueError('Historical gate did not pass: API remains unopened.')
    fit_report=json.loads((fit/'fit-results.json').read_text(encoding='utf-8'));prior_elapsed=(reference-datetime.fromisoformat(fit_report['started_at_utc'])).total_seconds()
    if prior_elapsed<0 or prior_elapsed>=4*3600:raise ValueError('Whole-experiment four-hour budget exceeded or reference clock order invalid.')
    for item in lock['artifacts']:
        if h.digest(fit/item['id'])!=item['sha256']:raise ValueError('Frozen dependence model changed.')
    for item in lock['numerical_code']:
        if h.digest(ROOT/'research_code'/item['id'])!=item['sha256']:raise ValueError('Frozen numerical implementation changed.')
    cfg_path=ROOT/'research/plans/H-0014-v1.json';cfg=json.loads(cfg_path.read_text(encoding='utf-8'))
    if h.digest(cfg_path)!=lock['config_sha256'] or h.digest(a.panel)!=cfg['api']['source_sha256']:raise ValueError('Frozen config/API input differs.')
    m.configure();ctx=m.Features(a.data);gaussians=[];flows=[]
    for seed in m.SEEDS:
        g=m.Gaussian(lock['selected_rank']).to(ctx.device);g.load_state_dict(torch.load(fit/f'gaussian-{lock["selected_rank"]}-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);g.eval();gaussians.append(g)
        f=m.GraphFlow().to(ctx.device);f.load_state_dict(torch.load(fit/f'flow-{seed}.pt',map_location='cpu',weights_only=True)['state_dict']);f.eval();flows.append(f)
    with np.load(a.panel,allow_pickle=False) as f:times=f['scheduled_at_unix_seconds'].copy();api_ids=f['station_ids'].astype(str);stock=f['stock'].copy();actual=f['observed_at_unix_seconds'].copy()
    if len(set(api_ids.tolist()))!=len(api_ids) or not np.all(np.diff(times)==600):raise ValueError('API identifiers/grid invalid.')
    lookup={s:i for i,s in enumerate(api_ids)};model_nodes=np.array([i for i,s in enumerate(ctx.ids) if s in lookup]);api_nodes=np.array([lookup[ctx.ids[i]] for i in model_nodes]);selected=np.flatnonzero((times>=h.stamp('2026-09-01'))&(times<h.stamp('2026-10-01')))
    wx=weather.weather_for(Path(a.weather_dir),'daejeon',times[selected]);daily=np.zeros((14,6));day0=(h.stamp('2026-09-17')+32400)//86400
    sums={name:dict(regions=0,regional_crps=0.,pinball90=0.,pinball95=0.,coverage90=0.,coverage95=0.,node_pairs=0,stock_crps=0.,median_mae=0.,brier=0.,joint_pairs=0,joint_brier=0.) for name in NAMES}
    previous=None;pending=None;issued=0;issued_nodes=0;scored_nodes=0;scored_regions=set();latencies=[];invariant_cases=0
    ledger=out/'replay-ledger.jsonl'
    with ledger.open('w',encoding='utf-8') as log:
        for j,index in enumerate(selected):
            values=np.full(ctx.n,np.nan);stamps=np.full(ctx.n,-1.);values[model_nodes]=stock[index,api_nodes];stamps[model_nodes]=actual[index,api_nodes];current=obs.Snapshot(int(times[index]),values,stamps,len(api_ids),len(model_nodes))
            if pending is not None:
                elapsed=current.actual-pending['actual'];valid=pending['valid']&obs.valid_observation(current)&(elapsed>=480)&(elapsed<=720)
                groups=[gi for gi,g in enumerate(ctx.groups) if pending['regions'][gi] and valid[g].all() and np.ptp(current.actual[g])<=30]
                y=values[valid];truth=np.array([(values[ctx.groups[gi]]==0).mean() for gi in groups]);dd=int((pending['scheduled']+32400)//86400-day0);edges=ctx.edge
                pair_keep=valid[edges].all(1)&(np.ptp(current.actual[edges],axis=1)<=30)&(np.ptp(pending['actual'][edges],axis=1)<=30);pairs=edges[pair_keep];yt=(values[pairs]==0).all(1)
                for ki,name in enumerate(NAMES):
                    x=pending['stock'][name];r=np.array([(x[ctx.groups[gi]]==0).mean(0) for gi in groups]);entry=sums[name]
                    if groups:
                        score=m.crps(r,truth);entry['regional_crps']+=float(score.sum());entry['regions']+=len(groups);daily[dd,ki]+=score.sum()
                        for q in (90,95):
                            quant=np.sort(r,axis=1)[:,int(np.ceil(q/100*m.M))-1];err=truth-quant;entry[f'pinball{q}']+=float(np.maximum(q/100*err,(q/100-1)*err).sum());entry[f'coverage{q}']+=float((truth<=quant).sum())
                    point=np.sort(x[valid],axis=1)[:,63];p=(x[valid]==0).mean(1);entry['stock_crps']+=float(m.crps(x[valid],y).sum());entry['median_mae']+=float(np.abs(point-y).sum());entry['brier']+=float(((p-(y==0))**2).sum());entry['node_pairs']+=len(y)
                    pp=(x[pairs]==0).all(1).mean(1);entry['joint_brier']+=float(((pp-yt)**2).sum());entry['joint_pairs']+=len(pairs)
                daily[dd,-1]+=len(groups);scored_regions.update(groups);scored_nodes+=int(valid.sum());log.write(json.dumps(dict(event='score',target=current.scheduled,node_pairs=int(valid.sum()),complete_regions=len(groups)))+'\n')
            proxy,trend,valid=obs.proxy_and_trend(current,previous,ctx.graph);valid&=current.scheduled>=ctx.first
            group_valid=np.array([bool(valid[g].all() and np.ptp(current.actual[g])<=30) for g in ctx.groups]);tick=time.perf_counter()
            if valid.any():
                axes=dates.axes(np.array([current.scheduled]));common=np.c_[axes['calendar'],axes['phase'],wx[j:j+1]];rates=ctx.anchor(axes['slot'],common,proxy[None]);features=ctx.from_inputs(axes['slot'],common,proxy[None],rates);mask=torch.as_tensor(valid[None],device=ctx.device)
                draws=m.generate(ctx,features,mask,rates[0],gaussians,flows,current.scheduled);prediction={name:np.maximum(0,values[:,None]+draws[name]) for name in NAMES[:-1]}
                reference_sorted=np.sort(prediction['independent'][valid],axis=1)
                for name in NAMES[1:-1]:
                    if not np.array_equal(reference_sorted,np.sort(prediction[name][valid],axis=1)):raise ValueError('Inventory marginal invariant failed before future truth.')
                invariant_cases+=int(valid.sum())
            else:prediction={name:np.full((ctx.n,m.M),np.nan) for name in NAMES[:-1]}
            prediction['persistence']=np.repeat(values[:,None],m.M,1);latencies.append(time.perf_counter()-tick)
            pending=dict(scheduled=current.scheduled,actual=current.actual.copy(),valid=valid,regions=group_valid,stock=prediction);previous=current;issued+=1;issued_nodes+=int(valid.sum())
            log.write(json.dumps(dict(event='issue',scheduled=current.scheduled,forecast_nodes=int(valid.sum()),complete_current_regions=int(group_valid.sum()),future_accessed=False))+'\n');log.flush()
            if issued%100==0:emit(stage='joint_api_replay',origins=issued,total=len(selected),region_issues=int(daily[:,-1].sum()),seconds=time.perf_counter()-start)
            if prior_elapsed+time.perf_counter()-start>4*3600:raise TimeoutError('Whole experiment four-hour limit reached.')
    metrics={}
    for name,e in sums.items():
        metrics[name]=dict(region_issues=e['regions'],regional_zero_fraction_crps=e['regional_crps']/e['regions'],pinball90=e['pinball90']/e['regions'],pinball95=e['pinball95']/e['regions'],coverage90=e['coverage90']/e['regions'],coverage95=e['coverage95']/e['regions'],node_pairs=e['node_pairs'],stock_crps=e['stock_crps']/e['node_pairs'],median_mae=e['median_mae']/e['node_pairs'],brier=e['brier']/e['node_pairs'],joint_pairs=e['joint_pairs'],joint_zero_brier=e['joint_brier']/e['joint_pairs'])
    summary=intervals(daily,len(scored_regions));code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    result=dict(hypothesis_id='H-0014',started_at_utc=reference.isoformat(),finished_at_utc=now(),native_remote_finished_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=code,fit_code_commit=lock['code_commit'],plan_sha256=lock['plan_sha256'],input_manifest_sha256=lock['input_manifest_sha256'],model_lock_sha256=h.digest(fit/'model-lock.json'),summary=summary,metrics=metrics,
        coverage=dict(trained_nodes=ctx.n,namespace_api_ids=len(api_ids),namespace_matched_nodes=len(model_nodes),forecast_origins=issued,issued_valid_nodes=issued_nodes,scored_node_pairs=scored_nodes,scored_regions=len(scored_regions),region_issues=int(daily[:,-1].sum()),invariant_node_forecasts=invariant_cases),daily=dict(first_date='2026-09-17',labels=list(NAMES)+['region_issues'],values=daily.tolist()),latency=dict(p50=float(np.percentile(latencies,50)),p95=float(np.percentile(latencies,95)),max=float(max(latencies)),within60seconds=bool(np.percentile(latencies,95)<=60)),seconds=time.perf_counter()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),ledger_sha256=h.digest(ledger),limitations=['Repeated exposed September evaluation, not untouched test.','Displayed API inventory, not latent demand or counterfactual action outcome.','Net-flow projection does not model physical event ordering/capacity/operator moves.','Common margins intentionally fix individual predictions; gain must be joint-risk only.','No routing/vehicle/field-effect claim from this forecast evaluation.'])
    h.write(out/'results.json',result);emit(stage='joint_api_complete',summary=summary,coverage=result['coverage'],seconds=result['seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--fit',required=True);p.add_argument('--panel',required=True);p.add_argument('--weather-dir',required=True);p.add_argument('--out',required=True);p.add_argument('--collector-start',required=True);evaluate(p.parse_args())
