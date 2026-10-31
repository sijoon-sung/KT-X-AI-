"""H13 chronological September API replay after historical-only model lock.

At each timestamp: score already-issued pending forecast, then issue from the
current/previous snapshots only. Future-validity filtering is scoring-only.
"""
import argparse,json,time,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
import historical_calendar as h
import forecast_transfer as weather
import rolling_inventory_model as m
import rolling_inventory_predict as live
from rolling_inventory_metrics import Aggregator,LABELS,uncertainty
from forecast_transfer import now,emit

ROOT=Path(__file__).resolve().parents[1];START=h.stamp('2026-09-01');END=h.stamp('2026-10-01');DAY0=(h.stamp('2026-09-17')+32400)//86400

def evaluate(a):
    started=now();timer=time.perf_counter();out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    if (out/'results.json').exists():raise ValueError('Preserve completed external evaluation.')
    m.configure();predictor=m.Predictor(a.fit_dir,'cuda')
    if predictor.lock['locked_at_utc']>started:raise ValueError('Evaluation precedes frozen model lock.')
    manifest=json.loads((ROOT/'research/datasets/tashu-rolling-api-H0013-v1.json').read_text(encoding='utf-8'));expected=next(x['sha256'] for x in manifest['sources'] if x['id']=='observed_panel.npz')
    if h.digest(a.panel)!=expected:raise ValueError('Official inventory API source hash differs.')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    with np.load(a.panel,allow_pickle=False) as f:
        times=f['scheduled_at_unix_seconds'].copy();api_ids=f['station_ids'].astype(str);stock=f['stock'].copy();actual=f['observed_at_unix_seconds'].copy()
    if len(set(api_ids.tolist()))!=len(api_ids) or len(set(times.tolist()))!=len(times) or not np.all(np.diff(times)==600):raise ValueError('API grid or identifiers are damaged.')
    lookup={s:i for i,s in enumerate(api_ids)};model_nodes=np.array([i for i,s in enumerate(predictor.ids) if s in lookup],dtype=int);api_nodes=np.array([lookup[predictor.ids[i]] for i in model_nodes],dtype=int)
    selected=np.flatnonzero((times>=START)&(times<END));wx=weather.weather_for(Path(a.weather_dir),'daejeon',times[selected])
    current_values=np.full(len(predictor.ids),np.nan);current_times=np.full(len(predictor.ids),-1.)
    previous=None;pending=None;acc=Aggregator();daily=np.zeros((14,len(LABELS)+1));scored_ids=set();issues=0;issued_cells=0;valid_pairs=0;proxy_cells=0;scored_forecasts=0
    # The private ledger contains hashes/temporal joins, no public station records.
    ledger=out/'replay-ledger.jsonl'
    with ledger.open('w',encoding='utf-8') as log:
        for j,index in enumerate(selected):
            current_values.fill(np.nan);current_times.fill(-1)
            current_values[model_nodes]=stock[index,api_nodes];current_times[model_nodes]=actual[index,api_nodes]
            current=live.Snapshot(int(times[index]),current_values.copy(),current_times.copy(),len(api_ids),len(model_nodes))
            if pending is not None:
                valid,y,strict=live.score_issue(pending,current);n=int(valid.sum())
                if n:
                    draws={k:v[valid] for k,v in pending['draws'].items()};sums=acc.add(draws,y,pending['current'][valid],strict)
                    day=int((pending['scheduled']+32400)//86400-DAY0)
                    if not 0<=day<14:raise ValueError('A scored origin crossed the frozen September scope.')
                    daily[day,:-1]+=[sums[k] for k in LABELS];daily[day,-1]+=n;valid_pairs+=n;scored_ids.update(pending['nodes'][valid].tolist());scored_forecasts+=1
                log.write(json.dumps(dict(event='score',target_scheduled=current.scheduled,pairs=n,issued_hash=pending['hash']))+'\n')
            # Forecast all valid supported CURRENT cells before considering the next observation.
            forecast=live.issue(predictor,current,previous,wx[j]);forecast['hash']=hashlib.sha256(b''.join(np.ascontiguousarray(forecast['draws'][k]).tobytes() for k in LABELS)).hexdigest()
            issues+=1;issued_cells+=len(forecast['nodes']);proxy_cells+=forecast['previous_proxy_present']
            log.write(json.dumps(dict(event='issue',scheduled=current.scheduled,target_scheduled=forecast['target_scheduled'],forecast_cells=len(forecast['nodes']),virtual_issued_at=float(np.max(forecast['actual'])) if len(forecast['actual']) else None,forecast_hash=forecast['hash'],future_label_accessed=False))+'\n');log.flush()
            pending=forecast;previous=current
            if (j+1)%100==0:emit(stage='rolling_api_replay',origins=j+1,total_origins=len(selected),pairs=valid_pairs,seconds=time.perf_counter()-timer)
    summary=uncertainty(daily,len(scored_ids));groups=acc.result()
    coverage=dict(trained_ids=len(predictor.ids),api_ids=len(api_ids),matched_ids=len(model_nodes),unmatched_trained_ids=len(predictor.ids)-len(model_nodes),unsupported_api_ids=len(api_ids)-len(model_nodes),scored_ids=len(scored_ids),issued_origins=issues,scored_origins=scored_forecasts,issued_current_cells=issued_cells,valid_pairs=valid_pairs,unscored_issued_cells=issued_cells-valid_pairs,recent_proxy_present_cells=proxy_cells,
        source_dates='2026-09-17..2026-09-30 KST',official_start='2026-09-01T00:00:00+09:00',official_end_exclusive='2026-10-01T00:00:00+09:00',nominal_horizon_seconds=600,actual_elapsed_allowed=[480,720],collection_delay_allowed=[0,120])
    report=dict(hypothesis_id='H-0013',started_at_utc=started,finished_at_utc=now(),code_commit=code,fit_code_commit=predictor.lock['code_commit'],plan_sha256=predictor.lock['plan_sha256'],input_manifest_sha256=predictor.lock['input_manifest_sha256'],model_lock_sha256=h.digest(Path(a.fit_dir)/'model-lock.json'),api_panel_sha256=expected,
        coverage=coverage,groups=groups,summary=summary,daily_fair_crps=dict(labels=list(LABELS)+['pairs'],first_date='2026-09-17',values=daily.tolist()),ledger_sha256=h.digest(ledger),weather=weather.coverage(wx,np.ones(len(wx),bool)),seconds=time.perf_counter()-timer,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        limitations=['September has been exposed in earlier hypotheses; not an untouched test.','API change is a noisy recent-flow proxy containing operations/repair/reporting changes.','Historical completed-flow training is not latent demand or operational action truth.','CFM samples only station-wise rental/return pairs, not city joint scenarios.','No capacity/action/routing/reward logs; no RL or relocation effect trained.','Weather archive as-of arrival/revisions not verified.','Confidence intervals condition on fixed fitted models and sampling rule.'])
    h.write(out/'results.json',report);emit(stage='rolling_api_complete',summary=summary,coverage=coverage,seconds=report['seconds'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit-dir',required=True);p.add_argument('--panel',required=True);p.add_argument('--weather-dir',required=True);p.add_argument('--out',required=True);evaluate(p.parse_args())
