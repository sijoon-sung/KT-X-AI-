"""Post-fit engineering acceptance; no fitting, model selection or API labels."""
import argparse,json,tempfile,os,time
from pathlib import Path
import numpy as np
import torch
import historical_calendar as h
import rolling_inventory_model as m
import rolling_inventory_data as d
from forecast_transfer import now

CORE=('rolling_inventory_model.py','rolling_inventory_data.py','rolling_inventory_fit.py','rolling_inventory_predict.py','rolling_inventory_metrics.py','rolling_inventory_evaluate.py','forecast_transfer.py','forecast_transfer_data.py','cross_city_data.py','cross_city_weather.py','scaled_flow_core.py','historical_calendar.py')

def run(a):
    started=now();folder=Path(a.fit_dir);m.configure();lock=json.loads((folder/'model-lock.json').read_text(encoding='utf-8'));expected={v['id']:v['sha256'] for v in lock['code']}
    for name in CORE:
        if h.digest(Path(__file__).parent/name)!=expected[name]:raise ValueError('Frozen numerical/feature/evaluation source changed: '+name)
    gpu=m.Predictor(folder,'cuda');cpu=m.Predictor(folder,'cpu');max_error=0.;checks=0
    for stamp in ('2026-09-18T00:00','2026-09-18T07:00','2026-09-18T12:10','2026-09-18T23:50'):
        t=h.stamp(stamp);axes=d.axes(np.array([t]));recent=np.zeros((1,gpu.n,4),dtype='float32');recent[0,:,0]=np.arcsinh(np.arange(gpu.n)%5-2);recent[0,:,1]=1
        recent[0,:,2]=gpu.graph.dot(recent[0,:,0]);recent[0,:,3]=1
        for pred in (cpu,gpu):
            c=pred.c;c.slot=torch.as_tensor(axes['slot'],device=pred.device)
            c.calendar=m.tensor(axes['calendar'],pred.device);c.phase=m.tensor(axes['phase'],pred.device);c.weather=torch.zeros((1,54),device=pred.device);c.recent=m.tensor(recent,pred.device)
        nodes=np.array([0,min(9,gpu.n-1),min(49,gpu.n-1),gpu.n-1])
        for kind in m.KINDS:
            for seed_index in range(2):
                rates=[]
                for pred in (cpu,gpu):
                    with torch.no_grad():rates.append(pred.models[kind][seed_index].rate(pred.c,torch.zeros(4,dtype=torch.int64,device=pred.device),torch.as_tensor(nodes,device=pred.device),kind,*pred.encoded[kind][seed_index]).cpu().numpy())
                np.testing.assert_allclose(*rates,atol=2e-5,rtol=2e-5);max_error=max(max_error,float(np.max(np.abs(rates[0]-rates[1]))));checks+=1
        # Same live inputs -> same emitted ensemble; no future observation supplied.
        valid=np.zeros(gpu.n,bool);valid[nodes]=True;current=np.full(gpu.n,4.);wx=np.zeros(54,dtype='float32')
        aa=gpu.issue(t,current,valid,recent[0],wx);bb=gpu.issue(t,current,valid,recent[0],wx)
        for kind in aa[1]:np.testing.assert_array_equal(aa[1][kind],bb[1][kind])
    # Fail closed if a listed trained artifact has a different expected hash.
    with tempfile.TemporaryDirectory(prefix='h13-lock-test-') as td:
        tmp=Path(td)
        for item in lock['artifacts']:os.link(folder/item['id'],tmp/item['id'])
        altered=json.loads(json.dumps(lock));altered['artifacts'][0]['sha256']='0'*64;h.write(tmp/'model-lock.json',altered)
        try:m.Predictor(tmp,'cpu')
        except ValueError:tamper_rejected=True
        else:raise AssertionError('Corrupt artifact hash was accepted.')
    report=dict(hypothesis_id='H-0013',started_at_utc=started,finished_at_utc=now(),scope='engineering acceptance on fixed models/synthetic inputs, no empirical selection',device=torch.cuda.get_device_name(),code_hashes_verified=list(CORE),cpu_gpu_rate_checks=checks,max_absolute_rate_error=max_error,tolerance_absolute=2e-5,tolerance_relative=2e-5,repeated_generation_identical=True,tamper_rejected=tamper_rejected,model_lock_sha256=h.digest(folder/'model-lock.json'))
    h.write(Path(a.out),report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fit-dir',required=True);p.add_argument('--out',required=True);run(p.parse_args())
