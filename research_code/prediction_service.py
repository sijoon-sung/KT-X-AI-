"""H8 frozen GCN prediction application; no fitting, tuning or future-label inputs."""
import argparse,csv,hashlib,json,mimetypes,time
from datetime import datetime,timedelta
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse,parse_qs

import numpy as np
import torch
import historical_od_gcn as graph
import history_to_stock as stock
import loss_consistent_stock as probability

ROOT=Path(__file__).resolve().parents[1]
WEB=Path(__file__).resolve().parent/'forecast_web'
H7='research/runs/H-0007-20261031-od-gcn/manifest.json'
H4='research/runs/H-0004-20261031-historical/manifest.json'
INPUT='research/datasets/tashu-prediction-system-H0008-v1.json'


def timestamp(value):
    if not isinstance(value,str):raise ValueError('시간대가 포함된 기준시각이 필요합니다.')
    try:dt=datetime.fromisoformat(value)
    except ValueError:raise ValueError('기준시각 형식을 확인하세요.') from None
    if dt.tzinfo is None or dt.utcoffset() is None:raise ValueError('기준시각에 +09:00 같은 시간대를 포함하세요.')
    return dt.astimezone(stock.KST)


def count(value):
    if isinstance(value,bool) or value is None or not isinstance(value,(int,float)):
        raise ValueError('현재 대수는 누락되지 않은 비음수 정수여야 합니다.')
    if not np.isfinite(value) or value<0 or value!=int(value) or value>2**31-1:
        raise ValueError('현재 대수의 정수 범위를 확인하세요. 값을 잘라서 사용하지 않습니다.')
    return int(value)


def check_keys(value,required,optional=()):
    if not isinstance(value,dict) or not set(required)<=value.keys() or set(value)-set(required)-set(optional):
        raise ValueError('현재 상태 입력 형식을 확인하세요. 미래 정답 필드는 입력할 수 없습니다.')


class ForecastEngine:
    """Frozen calendar/GCN flow encoder -> current-state stock distribution."""
    def __init__(self,ids,rates,cutoff,names=None,provenance=None):
        self.ids=np.asarray(ids,dtype=str)
        if len(set(self.ids.tolist()))!=len(self.ids):raise ValueError('학습 대여소 ID가 중복됐습니다.')
        self.index={x:i for i,x in enumerate(self.ids)}
        self.rates=rates;self.cutoff=cutoff;self.names=names or {};self.provenance=provenance or {}
        for rate in rates.values():
            if rate.shape!=(7,24,len(ids),2) or not np.isfinite(rate).all() or (rate<0).any():raise ValueError('고정률 모형의 형식이 손상됐습니다.')

    @classmethod
    def load(cls,h7_artifacts,h4_artifacts,station_metadata=None):
        inputs=json.loads((ROOT/INPUT).read_text(encoding='utf-8'))
        for rel,key in [('h7_manifest','h7_manifest_sha256'),('h4_manifest','h4_manifest_sha256'),('api_manifest','api_manifest_sha256')]:
            if stock.digest(ROOT/inputs[rel])!=inputs[key]:raise ValueError('동결 입력 목록 해시가 다릅니다.')
        h7=json.loads((ROOT/H7).read_text(encoding='utf-8'));h4=json.loads((ROOT/H4).read_text(encoding='utf-8'))
        expected7={x['id']:x['sha256'] for x in h7['artifacts']};expected4={x['id']:x['sha256'] for x in h4['artifacts']}
        h7_artifacts,h4_artifacts=Path(h7_artifacts),Path(h4_artifacts)
        required7=['graph.npz',*[f'{k}.npz' for k in ('od','self','permuted','calendar')],*[f'od-seed{s}.pt' for s in graph.SEEDS]]
        for folder,expected,files in [(h7_artifacts,expected7,required7),(h4_artifacts,expected4,['constant.npz','calendar.npz'])]:
            for name in files:
                if stock.digest(folder/name)!=expected[name]:raise ValueError(f'모형 자산 해시 불일치: {name}')
        rates={};ids=None
        for name in ('od','self','permuted','calendar'):
            with np.load(h7_artifacts/f'{name}.npz',allow_pickle=False) as f:
                if ids is not None and not np.array_equal(ids,f['station_ids']):raise ValueError('모형 ID 순서가 다릅니다.')
                ids=f['station_ids'].copy();rates[name]=f['means'].copy()
        prior={}
        for name in ('constant','calendar'):
            with np.load(h4_artifacts/f'{name}.npz',allow_pickle=False) as f:
                if not np.array_equal(ids,f['station_ids']):raise ValueError('과거 프로필 ID 순서가 다릅니다.')
                prior[name]=f['means'].copy()
        with np.load(h7_artifacts/'graph.npz',allow_pickle=False) as f:
            if not np.array_equal(ids,f['station_ids']):raise ValueError('그래프 ID 순서가 다릅니다.')
            operator=f['od'].copy()
        torch.set_num_threads(4)
        x,base=graph.feature_grid(prior['constant'],prior['calendar'])
        neighbor=graph.aggregate_features(x,operator)
        outputs=[]
        for seed in graph.SEEDS:
            model=graph.RateGCN()
            model.load_state_dict(torch.load(h7_artifacts/f'od-seed{seed}.pt',map_location='cpu',weights_only=True))
            model.eval()
            with torch.inference_mode():
                outputs.append(model(torch.as_tensor(x),torch.as_tensor(neighbor),torch.as_tensor(base)).numpy().astype(float))
        forward=np.mean(outputs,axis=0).reshape(7,15,len(ids),2)
        target=rates['od'][:,7:22]
        if not np.allclose(forward,target,atol=1e-6,rtol=3e-5):raise ValueError('GCN 체크포인트와 고정 출력 캐시가 일치하지 않습니다.')
        numeric=graph.numerical_contract(np.concatenate([r.sum(-1).ravel() for r in rates.values()]))
        names={}
        if station_metadata is not None:
            if stock.digest(station_metadata)!=inputs['station_metadata']['sha256']:raise ValueError('화면 대여소 목록의 해시가 다릅니다.')
            with Path(station_metadata).open(encoding='utf-8-sig',newline='') as f:
                names={row['station_id']:row['name'] for row in csv.DictReader(f)}
        provenance=dict(model_run='H-0007-20261031-od-gcn',gcn_role='competition demonstration requested by user; superiority not established',
            gcn_checkpoint_forward_verified=True,forward_max_absolute_rate_error=float(np.max(np.abs(forward-target))),
            gcn_export_sha256=expected7['od.npz'],calendar_export_sha256=expected7['calendar.npz'],numerical_contract=numeric,
            new_training=False,api_tuning_or_calibration=False,compute_location='dislab CPU')
        return cls(ids,{'gcn':rates['od'],'calendar':rates['calendar']},numeric['cutoff'],names,provenance)

    def predict(self,station_id,parking_count,issued_at,source_mode='manual_scenario',observed_at=None):
        timer=time.perf_counter();dt=timestamp(issued_at);b=count(parking_count)
        if station_id not in self.index:raise ValueError('과거 학습 범위 밖 대여소입니다. 임의 예측으로 대체하지 않습니다.')
        if not 7<=dt.hour<=21:raise ValueError('현재 모델의 입력 지원시간은 한국시간 07~21시입니다.')
        if observed_at is not None:
            observed=timestamp(observed_at)
            if not 0<=(observed-dt).total_seconds()<=120:raise ValueError('관측시각과 기준시각의 차이가 입력 계약을 벗어났습니다.')
        else:observed=dt
        outputs={}
        for name,rates in self.rates.items():
            rental,returned=rates[dt.weekday(),dt.hour,self.index[station_id]]
            grid,pmf=probability.state_pmf(np.array([b]),np.array([returned]),np.array([rental]),self.cutoff)
            summary=probability.summaries(grid,pmf)
            outputs[name]=dict(point_prediction=int(summary['median'][0]),expected_stock=float(summary['mean'][0]),
                zero_probability=float(summary['pzero'][0]),positive_probability=float(1-summary['pzero'][0]),
                interval=dict(lower=int(summary['lower'][0]),upper=int(summary['upper'][0]),quantiles=[.05,.95]),
                distribution=[dict(stock=int(k),probability=float(p)) for k,p in zip(grid[0],pmf[0]) if k>=0 and p>0])
        identity=json.dumps([station_id,b,dt.isoformat(),self.provenance.get('gcn_export_sha256')],separators=(',',':'))
        grid_verified=dt.minute==dt.second==dt.microsecond==0
        return dict(schema_version=1,forecast_id=hashlib.sha256(identity.encode()).hexdigest()[:16],
            station_id=station_id,station_name=self.names.get(station_id,station_id),current_stock=b,
            issued_at=dt.isoformat(),observed_at=observed.isoformat(),target_at=(dt+timedelta(minutes=30)).isoformat(),horizon_minutes=30,
            source_mode=source_mode,primary_model='gcn',models=outputs,
            scope=dict(observed_truth='API displayed inventory, not verified rental success',official_validation_month='2026-09',
                previously_exposed_validation=True,validated_clock_origin=grid_verified,
                time_assumption='Current clock-hour historical rates held fixed for 30 minutes; non-clock origins are unvalidated scenarios.',
                operational_accuracy_guaranteed=False),latency_ms=(time.perf_counter()-timer)*1000)

    def snapshot(self,value):
        check_keys(value,['observed_at','stations'],['schema_version'])
        timestamp(value['observed_at'])
        rows=value['stations']
        if not isinstance(rows,list) or not 1<=len(rows)<=64:raise ValueError('스냅샷에는 현재 관측 1~64개를 넣으세요.')
        ids=set();forecasts=[]
        for row in rows:
            check_keys(row,['station_id','parking_count'],['name'])
            if row['station_id'] in ids:raise ValueError('스냅샷 대여소 ID가 중복됐습니다.')
            ids.add(row['station_id'])
            if row['parking_count'] is None:
                forecasts.append(dict(station_id=row['station_id'],status='unavailable',reason='현재 관측이 누락됐습니다. 0으로 채우지 않습니다.'));continue
            try:
                prediction=self.predict(row['station_id'],row['parking_count'],value['observed_at'],'uploaded_current_snapshot')
                forecasts.append(dict(status='ok',**prediction))
            except ValueError as error:forecasts.append(dict(station_id=row['station_id'],status='unsupported',reason=str(error)))
        return dict(observed_at=timestamp(value['observed_at']).isoformat(),forecasts=forecasts,new_training=False)


class SeptemberReplay:
    """Evaluation adapter; the forecasting engine never receives this object."""
    def __init__(self,panel,engine):
        inputs=json.loads((ROOT/INPUT).read_text(encoding='utf-8'))
        source=json.loads((ROOT/inputs['api_manifest']).read_text(encoding='utf-8'))
        if stock.digest(panel)!=source['derived_files'][0]['sha256']:raise ValueError('재생 입력 해시가 다릅니다.')
        with np.load(panel,allow_pickle=False) as f:
            times=f['scheduled_at_unix_seconds'];keep=(times>=stock.START)&(times<stock.END)
            self.times=times[keep];self.values=f['stock'][keep];self.actual=f['observed_at_unix_seconds'][keep];self.ids=f['station_ids'].copy()
        if len(set(self.ids.tolist()))!=len(self.ids):raise ValueError('재생 대여소 ID가 중복됐습니다.')
        self.time_index={int(t):i for i,t in enumerate(self.times)};self.columns={x:i for i,x in enumerate(self.ids)}
        self.names=engine.names;self.supported=set(engine.ids.tolist())
        hours=((self.times+32400)//3600)%24
        self.origins=self.times[((self.times+32400)%3600==0)&(hours>=7)&(hours<=21)&(self.times+1800<stock.END)]

    def origins_list(self):return [dict(value=int(t),label=datetime.fromtimestamp(int(t),stock.KST).isoformat()) for t in self.origins]

    def _row(self,origin):
        if isinstance(origin,bool) or not isinstance(origin,(int,float)) or origin!=int(origin) or int(origin) not in self.origins:
            raise ValueError('공식9월 재생 기준시각을 선택하세요.')
        return self.time_index[int(origin)]

    def current(self,origin,station_id):
        row=self._row(origin)
        if station_id not in self.columns:raise ValueError('재생 자료에 없는 대여소입니다.')
        col=self.columns[station_id];value=self.values[row,col];actual=int(self.actual[row,col]);delay=actual-int(origin)
        if not np.isfinite(value) or not 0<=delay<=120:raise ValueError('선택 시점 현재 관측이 없습니다. 0으로 채우지 않습니다.')
        return dict(station_id=station_id,parking_count=count(float(value)),issued_at=datetime.fromtimestamp(int(origin),stock.KST).isoformat(),
                    observed_at=datetime.fromtimestamp(actual,stock.KST).isoformat(),source_mode='september_replay')

    def stations(self,origin):
        row=self._row(origin);answer=[]
        for station_id,col in self.columns.items():
            value=self.values[row,col];delay=int(self.actual[row,col])-int(origin)
            available=bool(np.isfinite(value) and 0<=delay<=120)
            answer.append(dict(station_id=station_id,name=self.names.get(station_id,station_id),parking_count=int(value) if available else None,
                available=available,supported=station_id in self.supported))
        return sorted(answer,key=lambda x:x['station_id'])

    def truth(self,origin,station_id):
        current=self.current(origin,station_id);target=int(origin)+1800
        if target not in self.time_index:return dict(available=False,reason='9월 미래 관측이 없습니다.')
        r=self.time_index[target];c=self.columns[station_id];value=self.values[r,c];actual=int(self.actual[r,c])
        elapsed=actual-int(timestamp(current['observed_at']).timestamp())
        if not np.isfinite(value) or not 0<=actual-target<=120 or abs(elapsed-1800)>120:
            return dict(available=False,reason='유효한9월 미래 관측이 없습니다.')
        return dict(available=True,station_id=station_id,target_parking_count=int(value),
            observed_at=datetime.fromtimestamp(actual,stock.KST).isoformat(),role='evaluation label only; never a forecasting input')


def handler(engine,replay):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,value,kind='application/json; charset=utf-8'):
            raw=value if isinstance(value,bytes) else json.dumps(value,ensure_ascii=False,allow_nan=False).encode('utf-8')
            self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(raw)))
            self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            try:
                url=urlparse(self.path)
                if url.path=='/api/health':return self.reply(200,dict(status='ok',service='tashu-prediction',compute_location='dislab',horizon_minutes=30))
                if url.path=='/api/info':return self.reply(200,dict(station_count=len(engine.ids),provenance=engine.provenance,
                    models=['gcn','calendar'],replay_available=replay is not None,station_ids=[dict(station_id=x,name=engine.names.get(x,x)) for x in sorted(engine.index)]))
                if url.path=='/api/replay/origins':return self.reply(200,dict(origins=replay.origins_list()))
                if url.path=='/api/replay/stations':return self.reply(200,dict(stations=replay.stations(int(parse_qs(url.query)['origin'][0]))))
                files={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css'}
                if url.path not in files:return self.reply(404,dict(error='찾을 수 없습니다.'))
                path=WEB/files[url.path];self.reply(200,path.read_bytes(),mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
            except (ValueError,KeyError,TypeError,AttributeError) as error:self.reply(400,dict(error=str(error)))
        def do_POST(self):
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:raise ValueError('입력 크기를 확인하세요.')
                value=json.loads(self.rfile.read(size))
                if self.path=='/api/forecast':
                    check_keys(value,['station_id','parking_count','issued_at'])
                    return self.reply(200,engine.predict(**value))
                if self.path=='/api/snapshot/forecast':return self.reply(200,engine.snapshot(value))
                if self.path in ('/api/replay/forecast','/api/replay/truth'):
                    check_keys(value,['origin','station_id'])
                    if replay is None:raise ValueError('재생 입력이 연결되지 않았습니다.')
                    if self.path.endswith('/truth'):return self.reply(200,replay.truth(**value))
                    return self.reply(200,engine.predict(**replay.current(**value)))
                self.reply(404,dict(error='찾을 수 없습니다.'))
            except (ValueError,KeyError,TypeError) as error:self.reply(400,dict(error=str(error)))
            except Exception:self.reply(503,dict(error='예측을 처리하지 못했습니다. 모형 연결 상태를 확인하세요.'))
    return Handler


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--h7-artifacts',required=True);ap.add_argument('--h4-artifacts',required=True)
    ap.add_argument('--station-metadata');sub=ap.add_subparsers(dest='command',required=True)
    serve=sub.add_parser('serve');serve.add_argument('--replay-panel');serve.add_argument('--port',type=int,default=8768)
    predict=sub.add_parser('predict');predict.add_argument('--station-id',required=True);predict.add_argument('--parking-count',type=int,required=True);predict.add_argument('--issued-at',required=True)
    args=ap.parse_args();engine=ForecastEngine.load(args.h7_artifacts,args.h4_artifacts,args.station_metadata)
    if args.command=='predict':print(json.dumps(engine.predict(args.station_id,args.parking_count,args.issued_at),ensure_ascii=False,allow_nan=False))
    else:
        replay=SeptemberReplay(args.replay_panel,engine) if args.replay_panel else None
        print(json.dumps(dict(status='ready',station_count=len(engine.ids),gcn_forward_verified=True,port=args.port)),flush=True)
        ThreadingHTTPServer(('127.0.0.1',args.port),handler(engine,replay)).serve_forever()
