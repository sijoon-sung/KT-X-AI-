"""Fixed H26 acceptance checks. Actual model generation must run on dislab."""
import argparse,copy,hashlib,json,re,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from operator_assistant import Assistant
from assistant_context import evidence,from_results

CASES=[
 ('total','자전거를 총 몇 대 옮기는 계획이야?',r'6\s*대|여섯\s*대'),
 ('order','어떤 순서로 이동하면 돼?',r'서쪽공원.*중앙역.*시청앞'),
 ('drop','중앙역에는 몇 대를 내려놓으면 돼?',r'2\s*대|두\s*대'),
 ('pickup','서쪽공원에서 몇 대를 싣는 계획이야?',r'6\s*대|여섯\s*대'),
 ('current','중앙역은 자료 시점에 자전거가 몇 대 있었어?',r'0\s*대|없'),
 ('future','시청앞은 10분 뒤 자전거가 몇 대 남을 것으로 예상해?',r'(?<!\d)1\s*대|한\s*대'),
 ('priority','중앙역은 어떤 상태로 예상하고 있어?',r'90\s*%|0\s*대|없'),
 ('distance','이 계획의 이동 거리는 얼마야?',r'3\.2'),
 ('time','이 자료는 언제 시점이고 예시인지 알려줘.',r'2026|시연|예시'),
 ('simple','예측을 어떻게 하는지 아주 쉬운 말로 설명해줘.',r'과거|기록|주변')]

def main(a):
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    if (out/'start.json').exists():raise ValueError('Preserve prior acceptance attempt')
    record=json.loads((ROOT/'research/hypotheses/H-0026.json').read_text(encoding='utf-8'))
    canonical=json.dumps(record['plan'],ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    assert hashlib.sha256(canonical).hexdigest()==record['preregistration']['plan_sha256']
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    start=datetime.now(timezone.utc).isoformat();began=time.perf_counter()
    (out/'start.json').write_text(json.dumps({'started_at_utc':start,'code_commit':code,'plan_sha256':record['preregistration']['plan_sha256']},indent=2)+'\n',encoding='utf-8')
    model=Assistant(a.model_dir);fixture=json.loads((ROOT/'examples/assistant-demo.json').read_text(encoding='utf-8'));results=[]
    for key,q,pattern in CASES:
        response=model.answer(q,fixture);semantic=bool(re.search(pattern,response['answer'],re.DOTALL))
        item={'id':key,'question':q,'response':response,'semantic_pattern_pass':semantic};results.append(item)
        print(json.dumps({'id':key,'engine':response['engine'],'semantic':semantic,'answer':response['answer']},ensure_ascii=False),flush=True)
    checks=[]
    bad=copy.deepcopy(fixture);bad['stations'][0]['current']=-1
    try:evidence(bad);ok=False
    except ValueError:ok=True
    checks.append({'id':'negative_input_rejected','passed':ok})
    missing={'mode':'demo','stations':[{'id':'TEST','name':'미확인대여소','current':None,'predicted':2}],'plan':None}
    pack=evidence(missing);text=' '.join(x['text'] for x in pack['facts'])
    checks.append({'id':'missing_not_zero','passed':'자료 시점의 자전거 수는 없음' in text and '자료 시점의 자전거 0대' not in text})
    secret=copy.deepcopy(fixture);secret['api_key']='unit-test-secret-not-a-real-key';clean=evidence(secret)
    checks.append({'id':'unlisted_field_not_in_context','passed':'unit-test-secret' not in json.dumps(clean)})
    for key,q,raw,pattern in [
        ('empty','예측 자료가 없는데 몇 대를 옮겨야 해?',{'mode':'demo','stations':[],'plan':None},r'없|알 수|확인'),
        ('recorded','지금 실시간으로 받은 자료야?',dict(fixture,mode='recorded_replay'),r'기록|실시간.*아니|확인.*없|현재.*아니'),
        ('unmet','실제로 자전거를 못 빌리고 돌아간 사람은 몇 명이야?',fixture,r'알 수 없|자료.*없|확인.*없|제공.*않')]:
        response=model.answer(q,raw);ok=bool(re.search(pattern,response['answer'],re.DOTALL))
        if key=='unmet':ok=ok and not re.search(r'\d+\s*명',response['answer'])
        checks.append({'id':key,'passed':bool(ok),'response':response});print(json.dumps(checks[-1],ensure_ascii=False),flush=True)
    llm=sum(x['response']['engine']=='llm' for x in results);semantic=sum(x['semantic_pattern_pass'] for x in results)
    report={'hypothesis_id':'H-0026','started_at_utc':start,'finished_at_utc':datetime.now(timezone.utc).isoformat(),'code_commit':code,
            'plan_sha256':record['preregistration']['plan_sha256'],'input_manifest_sha256':record['plan']['data']['input_manifest_sha256'],
            'model':'Qwen/Qwen3-1.7B','revision':'70d244cc86ccca08cf5af4e1e306ecf908b1ad5e','parameters':model.parameter_count,
            'actual_llm_answers':llm,'normal_questions':10,'semantic_pattern_passes':semantic,'checks':checks,'responses':results,
            'decision':'supported' if llm>=8 and all(x['passed'] for x in checks) else 'not_supported',
            'seconds':time.perf_counter()-began,'environment':{'execution':'dislab','torch':model.torch.__version__,'gpu':model.torch.cuda.get_device_name()},
            'weights_updated':False,'forecasting_model_changed':False,'api_panel_opened':False,
            'limitations':['Known synthetic functional suite,not independent language benchmark','No real nontechnical user comprehension study','Numeric whitelist and evidence IDs do not prove all semantic associations correct','Fallback is displayed separately and excluded from actual LLM count']}
    (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'complete':True,'decision':report['decision'],'actual_llm_answers':llm,'semantic_passes':semantic,'seconds':report['seconds']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True);p.add_argument('--out',required=True);main(p.parse_args())
