"""H9 research candidates on the H8 forecasting interface; no fitting endpoint."""
import argparse,hashlib,json,threading,time
from functools import lru_cache
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
import numpy as np
import torch
import prediction_service as base
import scaled_flow_core as core
import scaled_seoul_flow as study
import loss_consistent_stock as probability

RUN='H-0009-20261031-scaled-seoul-flow'
VARIANTS=('large_seoul_cfm','large_seoul','large','small')
LABELS={'large_seoul_cfm':'GCN + 따릉이 + FM','large_seoul':'큰 GCN + 따릉이','large':'큰 GCN','small':'작은 GCN'}


def pmf_output(grid,mass):
    result=probability.summaries(grid,mass)
    return dict(point_prediction=int(result['median'][0]),expected_stock=float(result['mean'][0]),zero_probability=float(result['pzero'][0]),
        positive_probability=float(1-result['pzero'][0]),interval=dict(lower=int(result['lower'][0]),upper=int(result['upper'][0]),quantiles=[.05,.95]),
        distribution=[dict(stock=int(k),probability=float(p)) for k,p in zip(grid[0],mass[0]) if k>=0 and p>0])


class AdvancedEngine:
    def __init__(self,reference,ids,rates,cutoff,slots,delta,depth,provenance):
        if not np.array_equal(reference.ids,ids):raise ValueError('H8/H9 station order differs.')
        self.reference=reference;self.ids=ids;self.index=reference.index;self.names=reference.names;self.rates=rates;self.cutoff=cutoff
        self.slots={int(s):i for i,s in enumerate(slots)};self.delta=delta;self.depth=depth;self.provenance=provenance;self.compute_lock=threading.Lock()
        if delta.shape!=depth.shape or delta.shape!=(len(slots),len(ids),512) or (depth<0).any() or (depth+delta<0).any():raise ValueError('Invalid frozen reflected event cache.')
        for rate in rates.values():
            if rate.shape!=(core.SLOTS,len(ids),2) or not np.isfinite(rate).all() or (rate<=0).any():raise ValueError('Invalid frozen rate cache.')
        self.dynamic_generator=None

    @classmethod
    def load(cls,reference,artifacts):
        artifacts=Path(artifacts);record,inputs,code=study.contract();lock=study.verify_lock(artifacts)
        public=json.loads((base.ROOT/f'research/runs/{RUN}/manifest.json').read_text(encoding='utf-8'))
        expected={x['id']:x['sha256'] for x in public['artifacts']}
        if public['plan_sha256']!=record['preregistration']['plan_sha256']:raise ValueError('Public run and frozen plan differ.')
        for name in ('selection.json','generation-lock.json','cfm-september.npz'):
            if base.stock.digest(artifacts/name)!=expected[name]:raise ValueError('H9 serving artifact hash mismatch.')
        rates={}
        for kind in core.KINDS:
            with np.load(artifacts/f'rates-{kind}.npz',allow_pickle=False) as f:
                if not np.array_equal(reference.ids,f['station_ids']):raise ValueError('H9 rate IDs differ.')
                rates[kind]=f['means'].copy()
        with np.load(artifacts/'cfm-september.npz',allow_pickle=False) as f:
            ids=f['station_ids'].copy();slots=f['slots'].copy();delta=f['delta'].copy();depth=f['depth'].copy()
        summary=json.loads((base.ROOT/f'research/runs/{RUN}/summary.json').read_text(encoding='utf-8'))
        provenance=dict(model_run=RUN,compute_location='dislab GPU/CPU',new_training_at_request=False,api_tuning_or_calibration=False,
            deployment_choice='user-requested fusion research candidate; no API winner selection',accuracy_hypothesis_decision=summary['decision'],
            seoul_source_period='2017-12..2018-11 city rental calendar patterns',cfm_distribution='bivariate completed rental/return counts per station',
            monte_carlo_samples=512,heun_steps=16,weights_locked=True,selection_sha256=base.stock.digest(artifacts/'selection.json'))
        obj=cls(reference,ids,rates,lock['numerical_contract']['cutoff'],slots,delta,depth,provenance)
        study.setup();context_ids,x,prior,op=study.load_context(artifacts);operator=torch.as_tensor(op,device='cuda');models,flows=study.load_fusion(artifacts)
        slot=int(core.context_slot(9,0,7,0))
        with torch.inference_mode():
            xx=torch.as_tensor(x[slot:slot+1],device='cuda');pp=torch.as_tensor(prior[slot:slot+1],device='cuda')
            actual=np.mean([m(xx,operator,pp).cpu().numpy()[0].astype(float) for m in models],axis=0)
        if not np.allclose(actual,rates['large_seoul'][slot],atol=1e-6,rtol=3e-5):raise ValueError('H9 checkpoint/cache forward mismatch.')
        obj.provenance['checkpoint_forward_verified']=True
        obj.provenance['checkpoint_forward_max_absolute_rate_error']=float(np.max(np.abs(actual-rates['large_seoul'][slot])))
        def generate(key):
            counts,seed=core.generate_counts(models,flows,x,operator,prior,key)
            return counts[:,:,1]-counts[:,:,0],core.reflected_depth(counts[:,:,0],counts[:,:,1],np.random.default_rng(seed))
        obj.dynamic_generator=generate
        return obj

    @lru_cache(maxsize=8)
    def events(self,slot):
        if slot in self.slots:
            index=self.slots[slot];return self.delta[index],self.depth[index]
        if self.dynamic_generator is None:raise ValueError('선택한 시나리오 시간의 생성 모델이 연결되지 않았습니다.')
        with self.compute_lock:return self.dynamic_generator(slot)

    def predict(self,station_id,parking_count,issued_at,model_variant='large_seoul_cfm',source_mode='manual_scenario',observed_at=None):
        timer=time.perf_counter()
        if model_variant not in VARIANTS:raise ValueError('지원하는 연구 모델을 선택하세요.')
        output=self.reference.predict(station_id,parking_count,issued_at,source_mode,observed_at)
        dt=base.timestamp(issued_at);slot=int(core.context_slot(dt.month,dt.weekday(),dt.hour,int(dt.minute>=30)))
        node=self.index[station_id];b=output['current_stock']
        if model_variant=='large_seoul_cfm':
            delta,depth=self.events(slot);samples=core.stock_samples(np.array([b]),delta[node:node+1],depth[node:node+1])[0]
            values,number=np.unique(samples,return_counts=True);grid=values[None,:];mass=number[None,:]/len(samples)
        else:
            rental,returned=self.rates[model_variant][slot,node]
            grid,mass=probability.state_pmf(np.array([b]),np.array([returned]),np.array([rental]),self.cutoff)
        output['models'][model_variant]=pmf_output(grid,mass);output['primary_model']=model_variant;output['primary_model_label']=LABELS[model_variant]
        identity=output['forecast_id']+model_variant+self.provenance.get('selection_sha256','synthetic')
        output['forecast_id']=hashlib.sha256(identity.encode()).hexdigest()[:16]
        output['scope'].update(research_candidate=True,accuracy_hypothesis_decision=self.provenance.get('accuracy_hypothesis_decision'),
            seoul_is_historical_calendar_profile=True,distribution_sampling=model_variant=='large_seoul_cfm',
            time_assumption='Current half-hour calendar context held fixed; non-clock origins are unvalidated scenarios.')
        output['latency_ms']=(time.perf_counter()-timer)*1000
        return output

    def snapshot(self,value):
        base.check_keys(value,['observed_at','stations'],['schema_version','model_variant']);variant=value.get('model_variant','large_seoul_cfm')
        payload={key:v for key,v in value.items() if key!='model_variant'}
        checked=self.reference.snapshot(payload);rows=[]
        for row in checked['forecasts']:
            if row['status']!='ok':rows.append(row)
            else:rows.append(dict(status='ok',**self.predict(row['station_id'],row['current_stock'],checked['observed_at'],variant,'uploaded_current_snapshot')))
        return dict(observed_at=checked['observed_at'],forecasts=rows,new_training=False)


def handler(reference,replay,advanced):
    class Handler(base.handler(reference,replay)):
        def do_GET(self):
            if urlparse(self.path).path=='/api/advanced/info':return self.reply(200,dict(available=True,models=[dict(id=x,label=LABELS[x]) for x in VARIANTS],default_model='large_seoul_cfm',provenance=advanced.provenance))
            return super().do_GET()
        def do_POST(self):
            if not self.path.startswith('/api/advanced/'):return super().do_POST()
            try:
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=2_000_000:raise ValueError('입력 크기를 확인하세요.')
                value=json.loads(self.rfile.read(size))
                if self.path=='/api/advanced/forecast':
                    base.check_keys(value,['station_id','parking_count','issued_at'],['model_variant'])
                    return self.reply(200,advanced.predict(**value))
                if self.path=='/api/advanced/replay/forecast':
                    base.check_keys(value,['origin','station_id'],['model_variant'])
                    current=replay.current(value['origin'],value['station_id']);return self.reply(200,advanced.predict(**current,model_variant=value.get('model_variant','large_seoul_cfm')))
                if self.path=='/api/advanced/snapshot/forecast':return self.reply(200,advanced.snapshot(value))
                self.reply(404,dict(error='찾을 수 없습니다.'))
            except (ValueError,KeyError,TypeError) as error:self.reply(400,dict(error=str(error)))
            except Exception:self.reply(503,dict(error='연구 모델의 예측 연결 상태를 확인하세요.'))
    return Handler


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('h7-artifacts','h4-artifacts','h9-artifacts','station-metadata','replay-panel'):ap.add_argument('--'+name,required=True)
    ap.add_argument('--port',type=int,default=8769);args=ap.parse_args()
    reference=base.ForecastEngine.load(args.h7_artifacts,args.h4_artifacts,args.station_metadata)
    replay=base.SeptemberReplay(args.replay_panel,reference);advanced=AdvancedEngine.load(reference,args.h9_artifacts)
    print(json.dumps(dict(status='ready',port=args.port,advanced_models=list(VARIANTS),checkpoint_forward_verified=True)),flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),handler(reference,replay,advanced)).serve_forever()
