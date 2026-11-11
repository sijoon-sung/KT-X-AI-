"""One fixed-input engineering issuance, not accuracy or parameter selection."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
import operator_pipeline

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
if (out/'recorded-demo.json').exists():raise RuntimeError('Fixed recorded issuance already exists; do not silently repeat')
config=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));source=config['source']
folder,status,state=operator_pipeline.snapshot(source)
digest=hashlib.sha256((folder/status['latest_state']).read_bytes()).hexdigest()
assert digest=='64a4ba791422308ba58ff494cd026170a249b59d1c0fb294a96fb41140f7b2e9'
assert state['source_version']=='H18-1794311400' and source['planner']=='batch'
body=dict(source_version=state['source_version'],automatic=True,capacity=20,target=6,budget=20,
          handling_seconds=20,minimum_transfer=3,stop_setup_seconds=60,deadline_minutes=30)
response=operator_pipeline.optimize(source,body)
assert response['status']==200
r=response['body'];summary={k:r[k] for k in ('status','source_version','recorded_at_utc','parameters','route','moved_bikes','coverage','certificate','geometry','geometry_status','operations')}
for key in ('excluded_small_receiver_ids','excluded_small_donor_ids'):summary['operations'].pop(key,None)
summary.update(input_sha256=digest,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
               role='fixed-default recorded-input illustration, not measured field effectiveness or API accuracy',new_inventory_api_requests=0,model_weights_updated=False)
(out/'recorded-demo.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:summary[k] for k in ('status','moved_bikes','route','operations','certificate')},ensure_ascii=False))
