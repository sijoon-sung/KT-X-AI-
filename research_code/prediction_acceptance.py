"""H8 frozen engineering acceptance, executed only on dislab."""
import argparse,json,platform,re,subprocess,sys,threading,time,urllib.request
from datetime import datetime,timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
import scipy,torch
import prediction_service as service
import history_to_stock as h

ROOT=Path(__file__).resolve().parents[1]


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()


def run(args):
    started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter()
    if git('status','--porcelain'):raise ValueError('Acceptance requires a clean published code checkout.')
    code=git('rev-parse','HEAD');record=json.loads((ROOT/'research/hypotheses/H-0008.json').read_text(encoding='utf-8'))
    canonical=json.dumps(record['plan'],ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    import hashlib
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256']:raise ValueError('Frozen plan mismatch.')
    directory=ROOT/'research/runs'/args.run_id
    if (directory/'summary.json').exists():raise ValueError('Preserve previous acceptance results.')
    artifacts=directory/'artifacts';artifacts.mkdir(parents=True,exist_ok=True)
    tests=[]
    for folder,pattern in [('research_code/tests','test_prediction_service.py'),('tools/tests','test_*.py')]:
        command=[sys.executable,'-m','unittest','discover','-s',folder,'-p',pattern,'-v']
        result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
        output=result.stdout+result.stderr;(artifacts/f'tests-{len(tests)}.txt').write_text(output,encoding='utf-8')
        n=re.search(r'Ran (\d+) tests?',output)
        tests.append(dict(suite=folder,passed=result.returncode==0,tests=int(n.group(1)) if n else 0))
    startup=time.perf_counter();engine=service.ForecastEngine.load(args.h7_artifacts,args.h4_artifacts,args.station_metadata)
    startup=time.perf_counter()-startup
    replay=service.SeptemberReplay(args.panel,engine)
    h7=json.loads((ROOT/service.H7).read_text(encoding='utf-8'))
    ref=Path(args.h7_artifacts)/'predictions-od.npz'
    expected=next(x['sha256'] for x in h7['artifacts'] if x['id']==ref.name)
    if h.digest(ref)!=expected:raise ValueError('Frozen H7 reference hash mismatch.')
    matched=[x for x in engine.ids if x in replay.columns]
    cases=[];latencies=[];errors={'pzero':0.,'mean':0.};passes=0
    with np.load(ref,allow_pickle=False) as prior:
        if prior['initial'].shape[1]!=len(matched):raise ValueError('Reference ID column mapping mismatch.')
        finite=np.flatnonzero(np.isfinite(prior['initial']).ravel())
        selected=finite[np.linspace(0,len(finite)-1,64).astype(int)] if len(finite)>=64 else []
        for index in selected:
            row,col=np.unravel_index(index,prior['initial'].shape)
            origin=int(prior['times'][row]);station=matched[col];current=int(prior['initial'][row,col])
            output=engine.predict(station,current,datetime.fromtimestamp(origin,h.KST).isoformat())
            prediction=output['models']['gcn'];latencies.append(output['latency_ms'])
            pzero_error=abs(prediction['zero_probability']-float(prior['pzero'][row,col]))
            mean_error=abs(prediction['expected_stock']-float(prior['mean'][row,col]))
            errors['pzero']=max(errors['pzero'],pzero_error);errors['mean']=max(errors['mean'],mean_error)
            exact=all(prediction[key]==int(prior[refkey][row,col]) for key,refkey in [('point_prediction','median')])
            exact=exact and all(prediction['interval'][key]==int(prior[key][row,col]) for key in ('lower','upper'))
            domain=True
            for model in output['models'].values():
                domain=domain and 0<=model['zero_probability']<=1 and abs(sum(x['probability'] for x in model['distribution'])-1)<=1e-10
                domain=domain and all(x['stock']>=0 and np.isfinite(x['probability']) for x in model['distribution'])
                domain=domain and 0<=model['interval']['lower']<=model['point_prediction']<=model['interval']['upper'] and np.isfinite(model['expected_stock'])
            passed=bool(pzero_error<=1e-10 and mean_error<=1e-8 and exact and domain);passes+=passed
            cases.append(dict(flat_reference_index=int(index),passed=passed,pzero_error=pzero_error,mean_error=mean_error,exact_quantiles=bool(exact),domain=bool(domain)))
    # A separate actual HTTP roundtrip through the production handler.
    server=ThreadingHTTPServer(('127.0.0.1',0),service.handler(engine,replay))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base='http://127.0.0.1:'+str(server.server_port)
    def request(path,value=None):
        req=urllib.request.Request(base+path,data=json.dumps(value).encode() if value is not None else None,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=30) as response:return json.loads(response.read())
    http={}
    try:
        http['health']=request('/api/health')['status']=='ok'
        origin=int(replay.origins[-1]);candidate=next(x for x in replay.stations(origin) if x['supported'] and x['available'])
        value=dict(origin=origin,station_id=candidate['station_id'])
        a=request('/api/replay/forecast',value);truth=request('/api/replay/truth',value);b=request('/api/replay/forecast',value)
        http['replay_future_label_noninterference']=a['models']==b['models'] and a['forecast_id']==b['forecast_id'] and 'target_parking_count' not in a
        manual=dict(station_id=candidate['station_id'],parking_count=4,issued_at='2026-10-31T15:00:00+09:00')
        http['manual_forecast']=request('/api/forecast',manual)['current_stock']==4
        snapshot=dict(observed_at=manual['issued_at'],stations=[dict(station_id=manual['station_id'],parking_count=0),dict(station_id='UNKNOWN',parking_count=None)])
        http['snapshot_missing_preserved']=[x['status'] for x in request('/api/snapshot/forecast',snapshot)['forecasts']]==['ok','unavailable']
        with urllib.request.urlopen(base+'/') as response:http['web_page']=response.status==200 and '<title>타슈다음' in response.read().decode('utf-8')
    finally:server.shutdown();server.server_close();thread.join()
    h.write(artifacts/'case-verification.json',cases)
    enough=len(cases)==64
    decision='supported' if enough and passes==64 and all(x['passed'] for x in tests) and all(http.values()) else ('not_supported' if enough else 'inconclusive')
    metrics=dict(acceptance_case_pass_fraction=passes/len(cases) if cases else 0.,acceptance_cases=len(cases),passed_cases=passes,
        max_absolute_pzero_error=errors['pzero'],max_absolute_mean_error=errors['mean'],checkpoint_forward_max_absolute_rate_error=engine.provenance['forward_max_absolute_rate_error'],
        model_startup_seconds=startup,prediction_latency_ms_median=float(np.median(latencies)),prediction_latency_ms_max=float(max(latencies)))
    limitations=['Engineering acceptance only; no new model accuracy evidence or GCN superiority claim.',
        'GCN is the user-requested competition demonstration path; H7 negative result and non-adoption for performance remain unchanged.',
        'September was previously exposed; February/March have no measured inventory truth.',
        'Replay/manual/current snapshot only; live API polling and operational deployment not implemented.',
        'Non-clock-hour scenarios hold current-hour rates fixed and have no empirical validation; API displayed inventory is not guaranteed usable bikes.',
        'Web UI interactions verified separately; inference and tests on dislab CPU; local browser is presentation only.']
    h.write(directory/'summary.json',dict(hypothesis_id='H-0008',decision=decision,metrics=metrics,tests=tests,http_checks=http,
        provenance=engine.provenance,new_training=False,api_model_selection_or_calibration=False,elapsed_seconds=time.perf_counter()-timer,limitations=limitations))
    public=['summary.json'];records=[dict(id=x,sha256=h.digest(directory/x),availability='public aggregate') for x in public]
    records += [dict(id=p.name,sha256=h.digest(p),availability='private engineering verification details; not published') for p in sorted(artifacts.iterdir())]
    h.write(directory/'manifest.json',dict(schema_version=1,run_id=args.run_id,hypothesis_id='H-0008',plan_sha256=record['preregistration']['plan_sha256'],code_commit=code,
        input_manifest_sha256=h.digest(ROOT/service.INPUT),started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),
        command=['python','research_code/prediction_acceptance.py','--h7-artifacts','<PRIVATE_H7_ARTIFACTS>','--h4-artifacts','<PRIVATE_H4_ARTIFACTS>',
            '--station-metadata','<PRIVATE_DISPLAY_METADATA>','--panel','<PRIVATE_AUDITED_PANEL>','--run-id',args.run_id],
        environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,compute_location='dislab',device='CPU',threads=4),
        seed=[11,29,47],evaluation_scope=dict(cases=len(cases),selection='linspace over finite current H7 cells, no future-label selection',official_month='2026-09',kind='engineering acceptance; no new accuracy comparison'),
        metrics=metrics,artifacts=records,limitations=limitations))
    print(json.dumps(dict(decision=decision,metrics=metrics,tests=tests,http_checks=http)),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('h7-artifacts','h4-artifacts','station-metadata','panel','run-id'):ap.add_argument('--'+name,required=True)
    run(ap.parse_args())
