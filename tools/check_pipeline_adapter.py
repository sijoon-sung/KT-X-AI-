"""Read-only lifecycle regression checks, run on dislab with private config."""
import argparse, copy, json, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from operator_pipeline import snapshot,info,context,optimize
from assistant_context import evidence

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))['source']
folder,status,state=snapshot(source);i=info(source);c=context(source);checks=[]
def check(name,value):checks.append({'id':name,'passed':bool(value)})
check('recorded_label_not_live',i['mode']=='recorded_replay' and '기록된 자료' in evidence(c)['label'])
check('same_forecast_version',c['source_version']==i['source_version']==state['source_version'])
check('real_station_count',len(c['stations'])==len(state['stations'])==len(i['stations']))
check('wrong_version_rejected',optimize(source,{'source_version':'invalid'})['status']==409)
live=copy.deepcopy(source);live['mode']='live'
check('stopped_stream_cannot_dispatch',status['phase']=='stopped' and optimize(live,{'source_version':i['source_version']})['status']==409)
check('stopped_stream_not_live_context',context(live)['mode']=='recorded_replay')
report={'checks':checks,'api_requested':False,'weights_updated':False,'plan_cache_preserved':True}
Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False));raise SystemExit(0 if all(x['passed'] for x in checks) else 1)
