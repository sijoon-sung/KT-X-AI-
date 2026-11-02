"""H18 capped live observation -> locked forecast -> later independent capture.

No training, credential logging, September replacement or real dispatch.
Run collection, inference, scoring and planning only on dislab.
"""
import argparse
import ctypes
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request,build_opener,HTTPRedirectHandler

ROOT=Path(__file__).resolve().parents[1]
ENDPOINT='https://bikeapp.tashu.or.kr:50041/v1/openapi/station'
CORE=('rolling_inventory_model.py','rolling_inventory_data.py','rolling_inventory_fit.py','rolling_inventory_predict.py','rolling_inventory_metrics.py','rolling_inventory_evaluate.py','forecast_transfer.py','forecast_transfer_data.py','cross_city_data.py','cross_city_weather.py','scaled_flow_core.py','historical_calendar.py')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def iso(stamp):return datetime.fromtimestamp(float(stamp),timezone.utc).isoformat()


def atomic(path,value):
    path=Path(path).resolve();tmp=path.with_suffix(path.suffix+'.tmp')
    assert tmp.parent==path.parent
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)


class Clock:
    def __init__(self,offset,uncertainty):
        if not math.isfinite(offset) or not 0<=uncertainty<=2:raise ValueError('Collector clock calibration unavailable.')
        self.utc=time.time()+offset;self.monotonic=time.monotonic();self.uncertainty=uncertainty;self.offset=offset
    def now(self):return self.utc+time.monotonic()-self.monotonic
    def metadata(self):return dict(utc_anchor=self.utc,monotonic_anchor=self.monotonic,uncertainty_seconds=self.uncertainty,offset_seconds=self.offset)


def reference_now(status):
    c=status['clock'];elapsed=time.monotonic()-c['monotonic_anchor']
    if not 0<=elapsed<=86520:raise ValueError('Collector clock anchor expired or machine restarted.')
    return c['utc_anchor']+elapsed


def credential(path):
    """Read Windows account-bound ConvertFrom-SecureString DPAPI blob."""
    if os.name!='nt':raise ValueError('DPAPI credential requires the approved Windows account.')
    class Blob(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('data',ctypes.POINTER(ctypes.c_ubyte))]
    encrypted=bytes.fromhex(Path(path).read_text(encoding='utf-8-sig').strip())
    buffer=(ctypes.c_ubyte*len(encrypted)).from_buffer_copy(encrypted);source=Blob(len(encrypted),buffer);out=Blob()
    crypt=ctypes.WinDLL('crypt32',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    if not crypt.CryptUnprotectData(ctypes.byref(source),None,None,None,None,0,ctypes.byref(out)):
        raise ValueError('Approved credential could not be decrypted; value never logged.')
    try:return ctypes.string_at(out.data,out.length).decode('utf-16-le')
    finally:
        ctypes.memset(out.data,0,out.length);kernel.LocalFree(out.data)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None


def capture(token,scheduled,clock,path):
    request=Request(ENDPOINT,headers={'api-token':token,'User-Agent':'TashuLiveVerification-H18/1.0'})
    with build_opener(NoRedirect).open(request,timeout=15) as response:
        payload=json.loads(response.read(4_000_000));status=response.status
    observed=clock.now();rows=payload.get('results')
    if not isinstance(rows,list) or not rows:raise ValueError('Station response schema invalid.')
    ids=[str(row['id']) for row in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate API station IDs.')
    if not all('parking_count' in row for row in rows):raise ValueError('Missing inventory fields.')
    atomic(path,dict(scheduled_at_unix_seconds=scheduled,observed_at_unix_seconds=observed,payload=payload))
    return dict(http_status=status,station_rows=len(rows),observed_at_utc=iso(observed),delay_seconds=observed-scheduled)


def groups(draws,truth,current,strict):
    import numpy as np
    from rolling_inventory_metrics import Aggregator
    agg=Aggregator();agg.add(draws,truth,current,strict)
    return dict(groups=agg.result(),zero_to_positive=int(((current==0)&(truth>0)).sum()),
                positive_to_zero=int(((current>0)&(truth==0)).sum()),pairs=len(truth))


def serving_state(forecast,predictor,snapshot,metadata,defaults):
    import numpy as np
    lookup={str(row['id']):row for row in snapshot['payload']['results']};x=forecast['draws']['recent_cfm'];stations=[]
    for i,node in enumerate(forecast['nodes']):
        sid=str(predictor.ids[node]);row=lookup[sid];lat=float(row['x_pos']);lon=float(row['y_pos'])
        if not np.isfinite([lat,lon]).all() or not 33<lat<39 or not 124<lon<132:continue
        ordered=np.sort(x[i]);n=len(ordered)
        stations.append(dict(id=sid,name=row.get('name',sid),lat=lat,lon=lon,current=int(forecast['current'][i]),samples=x[i].tolist(),
            forecast=dict(median=int(ordered[math.ceil(.5*n)-1]),low=int(ordered[math.ceil(.05*n)-1]),high=int(ordered[math.ceil(.95*n)-1]),
                          mean=float(x[i].mean()),p0=float((x[i]==0).mean()))))
    ids={s['id'] for s in stations};chosen=[sid for sid in defaults['default_ids'] if sid in ids]
    if not chosen:chosen=sorted(ids)[:5]
    depot=defaults['default_depot_id'] if defaults['default_depot_id'] in chosen else chosen[0]
    return dict(schema_version=1,mode='live_observation_shadow',model_label='Flow Matching + GCN',
        recorded_at_utc=iso(forecast['scheduled']),target_at_utc=iso(forecast['target_scheduled']),
        source_version=metadata['source_version'],model_lock_sha256=metadata['model_lock_sha256'],
        stations=stations,default_ids=chosen,default_depot_id=depot,dispatch_allowed=False)


def live_info(state,status):
    from dispatch_engine import information
    info=information(state);info.update(mode='live_observation_shadow',source_version=state['source_version'],
        feed={k:status.get(k) for k in ('phase','latest_observation','latest_forecast','latest_score','roundtrips','requests','expires_at_utc','stop_reason')})
    return info


def rpc(folder,request):
    from dispatch_engine import optimize,Roads
    folder=Path(folder);status=read(folder/'status.json')
    if not status.get('latest_state'):return dict(status=503,body=dict(error='최초 관측과 예측을 준비하고 있습니다.'))
    state=read(folder/status['latest_state']);now=reference_now(status);age=now-datetime.fromisoformat(state['recorded_at_utc']).timestamp()
    if request.get('path')=='/api/dispatch/info':return dict(status=200,body=live_info(state,status))
    body=dict(request.get('body') or {});version=body.pop('source_version',None)
    if version!=state['source_version']:return dict(status=409,body=dict(error='관측이 갱신됐습니다. 최신 자료를 다시 불러오세요.'))
    if not 0<=age<=720:return dict(status=409,body=dict(error='관측이 오래돼 계산을 중지했습니다. 수집 상태를 확인하세요.'))
    result=optimize(state,body,Roads(folder/'road-cache'));result.update(source_version=version,
        observation_age_seconds=age,planning_basis='observation-time shadow plan, never real dispatch')
    return dict(status=200,body=result)


def run(a):
    import numpy as np
    import torch
    import rolling_inventory_model as model
    import rolling_inventory_predict as prediction
    import forecast_transfer as weather
    from rolling_inventory_metrics import LABELS
    folder=Path(a.out).resolve();folder.mkdir(parents=True,exist_ok=True)
    if (folder/'status.json').exists():raise ValueError('Preserve an existing live run; do not reset its request budget.')
    clock=Clock(a.clock_offset,a.clock_uncertainty);start=clock.now();expires=start+86400
    status=dict(hypothesis_id='H-0018',phase='initializing',started_at_utc=iso(start),expires_at_utc=iso(expires),clock=clock.metadata(),
                requests=0,roundtrips=0,acceptance_failures=0,weights_updated=False,official_september_replaced=False)
    def publish():status['heartbeat_at_utc']=iso(clock.now());atomic(folder/'status.json',status)
    def event(value):
        with (folder/'events.jsonl').open('a',encoding='utf-8') as handle:handle.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
    publish()
    try:
        expected=read(ROOT/'research/datasets/H-0018-live-stream.json')['model_lock_sha256']
        if digest(Path(a.fit)/'model-lock.json')!=expected:raise ValueError('Frozen parent lock mismatch.')
        lock=read(Path(a.fit)/'model-lock.json');code={x['id']:x['sha256'] for x in lock['code']}
        if not all(digest(Path(__file__).parent/name)==code[name] for name in CORE):raise ValueError('Frozen numerical/feature source changed.')
        if torch.cuda.mem_get_info()[0]<1073741824:raise ValueError('Insufficient free dislab GPU memory; other jobs untouched.')
        model.configure();torch.cuda.set_per_process_memory_fraction(.30)
        predictor=model.Predictor(a.fit,'cuda');defaults=read(a.defaults);token=credential(a.credential)
        first=math.floor(clock.now()/600)*600;due=first if clock.now()-first<=90 else first+600
        previous=None;pending={};errors=0;status['phase']='waiting_for_observation'
        status['environment']=dict(device='dislab CUDA',gpu=torch.cuda.get_device_name(),torch=torch.__version__,numpy=np.__version__)
        while clock.now()<expires and status['requests']<143:
            status['next_observation_at_utc']=iso(due);publish()
            while clock.now()<due:
                time.sleep(min(15,due-clock.now()));publish()
                if (folder/'STOP').exists():status['stop_reason']='requested_stop';break
            if status.get('stop_reason'):break
            record=dict(scheduled_at_utc=iso(due));snapshot_path=folder/f'poll-{due}.json'
            try:
                status['requests']+=1;status['phase']='capturing';publish();capture_info=capture(token,due,clock,snapshot_path)
                record.update(capture_info,snapshot_sha256=digest(snapshot_path));errors=0
                if not 0<=capture_info['delay_seconds']<=120:raise ValueError('Late observation, no fabricated valid timestamp.')
                current=prediction.read_snapshot(snapshot_path,predictor.ids)
                status['latest_observation']=dict(**capture_info,scheduled_at_utc=iso(due))
                if due in pending:
                    forecast,published,forecast_path=pending.pop(due)
                    valid,y,strict=prediction.score_issue(forecast,current)
                    if valid.any():
                        if published>=float(np.min(current.actual[forecast['nodes']][valid])):raise ValueError('Issue not before future truth.')
                        report=groups({k:forecast['draws'][k][valid] for k in LABELS},y,forecast['current'][valid],strict)
                        report.update(scope='additional live diagnostic; not official or independent accuracy test',issued_before_truth=True,
                            target_at_utc=iso(due),forecast_sha256=digest(forecast_path),truth_sha256=digest(snapshot_path))
                        atomic(folder/f'score-{due}.json',report);status['roundtrips']+=1;status['latest_score']=report
                        record['causal_roundtrip_completed']=True
                status['phase']='predicting';publish();wx=weather.weather_for(Path(a.weather),'daejeon',np.array([due]))[0]
                previous_snapshot=prediction.read_snapshot(previous,predictor.ids) if previous is not None else None
                forecast=prediction.issue(predictor,current,previous_snapshot,wx);issued=clock.now()
                if not len(forecast['nodes']) or issued>=due+600:raise ValueError('No timely supported forecast; never backdate.')
                version=f'H18-{due}';metadata=dict(source_version=version,model_lock_sha256=expected,snapshot_sha256=digest(snapshot_path),
                    previous_sha256=digest(previous) if previous is not None else None,issued_at_utc=iso(issued),weights_updated=False,
                    input_api_only=True,weather_live=False,issue_clock_basis='calibrated collector monotonic UTC',
                    previous_proxy_present=forecast['previous_proxy_present'])
                forecast_path=folder/f'forecast-{due}.npz';prediction.save_forecast(forecast_path,forecast,predictor.ids,issued,metadata)
                published=clock.now()
                if published>=due+600:raise ValueError('Forecast publication deadline passed.')
                pending[due+600]=(forecast,published,forecast_path)
                state=serving_state(forecast,predictor,read(snapshot_path),metadata,defaults)
                state_name=f'state-{due}.json';atomic(folder/state_name,state)
                status['latest_state']=state_name;status['source_version']=version
                status['latest_forecast']=dict(issued_at_utc=iso(published),target_at_utc=iso(due+600),stations=len(state['stations']),
                    current_zero=sum(s['current']==0 for s in state['stations']),previous_proxy_present=forecast['previous_proxy_present'],weights_updated=False)
                info_name=f'info-{due}.json';status['latest_info']=info_name;status['phase']='waiting_for_observation'
                atomic(folder/info_name,live_info(state,status));publish();record.update(forecast_issued=True,issued_at_utc=iso(published),target_at_utc=iso(due+600),forecast_sha256=digest(forecast_path))
                previous=snapshot_path
            except HTTPError as exc:
                errors+=1;record.update(capture_failed=True,http_status=exc.code)
                if exc.code in (401,403,429):status['stop_reason']='authentication_or_rate_limit';status['phase']='stopped'
            except Exception as exc:
                errors+=1;record.update(error_type=type(exc).__name__)
            if errors>=3:status['stop_reason']='three_consecutive_failures'
            event(record);publish()
            if status.get('stop_reason'):break
            due+=600
            if due<clock.now()-120:due=math.ceil(clock.now()/600)*600
        status['phase']='stopped';status.setdefault('stop_reason','bounded_window_or_request_budget');publish()
    except Exception as exc:
        status.update(phase='failed',error_type=type(exc).__name__,acceptance_failures=1);publish()
        print(json.dumps(dict(phase='failed',error_type=type(exc).__name__)),flush=True);raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True)
    q=sp.add_parser('run');q.add_argument('--fit',required=True);q.add_argument('--weather',required=True);q.add_argument('--defaults',required=True);q.add_argument('--credential',required=True);q.add_argument('--out',required=True);q.add_argument('--clock-offset',required=True,type=float);q.add_argument('--clock-uncertainty',required=True,type=float)
    q=sp.add_parser('rpc');q.add_argument('--folder',required=True)
    a=p.parse_args()
    if a.command=='run':run(a)
    else:
        import sys
        try:value=rpc(a.folder,json.load(sys.stdin))
        except (ValueError,KeyError,TypeError) as exc:value=dict(status=422,body=dict(error=str(exc)))
        except Exception:value=dict(status=503,body=dict(error='실시간 관측 연결을 확인하세요.'))
        print(json.dumps(value,ensure_ascii=False,allow_nan=False))
