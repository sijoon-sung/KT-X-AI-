"""Engineering check against frozen H9 outputs, not a new accuracy experiment."""
import argparse,json,subprocess,time,urllib.request,urllib.error
from datetime import datetime
from pathlib import Path
import numpy as np
import history_to_stock as h
import advanced_prediction_service as advanced


def run(args):
    if args.base_url not in ('http://127.0.0.1:8769','http://localhost:8769'):raise ValueError('Use the private loopback service.')
    def request(path,value=None):
        req=urllib.request.Request(args.base_url+path,data=json.dumps(value).encode() if value is not None else None,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=45) as r:return json.loads(r.read())
    info=request('/api/advanced/info');h9=Path(args.h9_artifacts);h7=Path(args.h7_artifacts)
    with np.load(h9/'context.npz',allow_pickle=False) as f:ids=f['station_ids'].copy()
    with np.load(args.api_panel,allow_pickle=False) as f:available=set(f['station_ids'].tolist())
    matched=[x for x in ids if x in available]
    passed=0;max_pzero=0.;max_mean=0.;latencies=[]
    with np.load(h7/'predictions-od.npz',allow_pickle=False) as prior:
        index=np.flatnonzero(np.isfinite(prior['initial']).ravel());selected=index[np.linspace(0,len(index)-1,8).astype(int)]
        for kind in advanced.VARIANTS:
            with np.load(h9/f'predictions-{kind}.npz',allow_pickle=False) as reference:
                for flat in selected:
                    r,c=np.unravel_index(flat,prior['initial'].shape)
                    value=dict(station_id=matched[c],parking_count=int(prior['initial'][r,c]),issued_at=datetime.fromtimestamp(int(prior['times'][r]),h.KST).isoformat(),model_variant=kind)
                    output=request('/api/advanced/forecast',value);p=output['models'][kind];latencies.append(output['latency_ms'])
                    pzero_error=abs(p['zero_probability']-float(reference['pzero'][r,c]));mean_error=abs(p['expected_stock']-float(reference['mean'][r,c]))
                    max_pzero=max(max_pzero,pzero_error);max_mean=max(max_mean,mean_error)
                    if pzero_error>1e-10 or mean_error>1e-8 or p['point_prediction']!=reference['median'][r,c] or any(p['interval'][key]!=reference[key][r,c] for key in ('lower','upper')):
                        raise ValueError('Service output differs from the frozen candidate.')
                    passed+=1
    manual=dict(station_id=str(ids[0]),parking_count=4,issued_at='2026-10-31T15:00:00+09:00',model_variant='large_seoul_cfm')
    first=request('/api/advanced/forecast',manual);second=request('/api/advanced/forecast',manual)
    repeat=first['models']==second['models'] and first['forecast_id']==second['forecast_id']
    payload=dict(observed_at=manual['issued_at'],stations=[dict(station_id=str(ids[0]),parking_count=0),dict(station_id='UNKNOWN',parking_count=None)],model_variant='large_seoul_cfm')
    missing=[x['status'] for x in request('/api/advanced/snapshot/forecast',payload)['forecasts']]==['ok','unavailable']
    manual['future_stock']=0;blocked=False
    try:request('/api/advanced/forecast',manual)
    except urllib.error.HTTPError as e:blocked=e.code==400
    origin=request('/api/replay/origins')['origins'][-1]['value']
    station=next(x for x in request('/api/replay/stations?origin='+str(origin))['stations'] if x['supported'] and x['available'])['station_id']
    value=dict(origin=origin,station_id=station,model_variant='large_seoul_cfm')
    before=request('/api/advanced/replay/forecast',value);truth=request('/api/replay/truth',dict(origin=origin,station_id=station));after=request('/api/advanced/replay/forecast',value)
    independent=before['models']==after['models'] and 'target_parking_count' not in before
    result=dict(kind='engineering verification only',matched_cases=passed,max_absolute_pzero_error=max_pzero,max_absolute_mean_error=max_mean,
        dynamic_context_repeatable=repeat,missing_preserved=missing,future_input_rejected=blocked,separate_truth_noninterference=independent,
        latency_ms_median=float(np.median(latencies)),provenance=info['provenance'],code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    if not all((repeat,missing,blocked,independent,passed==32)):raise ValueError('A serving contract failed.')
    if args.output:h.write(Path(args.output),result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base-url',default='http://127.0.0.1:8769')
    for name in ('h9-artifacts','h7-artifacts','api-panel'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--output');run(ap.parse_args())
