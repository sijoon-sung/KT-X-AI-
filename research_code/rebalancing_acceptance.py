"""H15 fixed synthetic engineering acceptance, executed on dislab CPU."""
import argparse,json,hashlib,time,platform,subprocess,unittest,io
from pathlib import Path
from datetime import datetime,timedelta,timezone
import numpy as np,scipy
import historical_calendar as h
import test_rebalancing_pipeline as tests

ROOT=Path(__file__).resolve().parents[1]
def run(a):
    timer=time.perf_counter();reference=datetime.fromisoformat(a.collector_start);record=json.loads((ROOT/'research/hypotheses/H-0015.json').read_text(encoding='utf-8'));manifest=ROOT/'research/datasets/H-0015-input-manifest.json';inputs=json.loads(manifest.read_text(encoding='utf-8'))
    canonical=json.dumps(record['plan'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest()!=record['preregistration']['plan_sha256'] or h.digest(manifest)!=record['plan']['data']['input_manifest_sha256'] or h.digest(ROOT/inputs['fixture'])!=inputs['fixture_sha256']:raise ValueError('Frozen functional plan/input differs.')
    if reference.isoformat()<record['preregistration']['frozen_at_utc']:raise ValueError('Collector launch before frozen plan.')
    subprocess.run(['git','diff','--quiet','HEAD','--','research_code'],cwd=ROOT,check=True);code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.Acceptance);stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite);out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    (out/'acceptance-tests.log').write_text(stream.getvalue(),encoding='utf-8');violations=len(result.failures)+len(result.errors)
    report=dict(schema_version=1,hypothesis_id='H-0015',scope='synthetic engineering acceptance only; no real operating improvement or dispatch',started_at_utc=reference.isoformat(),finished_at_utc=(reference+timedelta(seconds=time.perf_counter()-timer)).isoformat(),native_remote_finished_at_utc=datetime.now(timezone.utc).isoformat(),clock_basis='collector reference plus remote monotonic elapsed',code_commit=code,plan_sha256=record['preregistration']['plan_sha256'],input_manifest_sha256=h.digest(manifest),environment=dict(device='dislab CPU',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),tests_run=result.testsRun,metrics=dict(acceptance_constraint_violations=violations),decision='supported' if result.wasSuccessful() else 'not_supported',outputs=tests.Acceptance.outputs,seconds=time.perf_counter()-timer,log=stream.getvalue())
    h.write(out/'results.json',report);print(json.dumps(dict(tests_run=result.testsRun,violations=violations,decision=report['decision'],seconds=report['seconds'])),flush=True)
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--collector-start',required=True);run(p.parse_args())
