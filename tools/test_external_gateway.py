"""Known-fixture external-answer checks. Prepare, validate and score on dislab."""
import argparse,copy,json,re,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from external_assistant import Gateway
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
 ('simple','예측을 어떻게 하는지 아주 쉬운 말로 설명해줘.',r'과거|기록|주변'),
 ('empty','예측 자료가 없는데 몇 대를 옮겨야 해?',r'없|알 수|확인'),
 ('recorded','지금 실시간으로 받은 자료야?',r'기록|아니|실시간.*없'),
 ('unmet','실제로 자전거를 못 빌리고 돌아간 사람은 몇 명이야?',r'알 수 없|자료.*없|확인.*없|모르|명시.*않|제공.*않'),
 ('long_route','어떤 순서로 이동하면 돼?',r'태평동 버드내초등학교.*유천동 서대전농협 본점 주차장.*문화동 BMK컨벤션.*오류동 서대전역.*문화동 홈플러스.*문화동 세이백화점')]

def fixture(key):
    raw=json.loads((ROOT/'examples/assistant-demo.json').read_text(encoding='utf-8'))
    if key=='empty':return {'mode':'demo','stations':[],'plan':None}
    if key=='recorded':raw['mode']='recorded_replay'
    if key=='long_route':return from_results(None,json.loads((ROOT/'research/runs/H-0021-20261101-bulk/reference-replay.json').read_text(encoding='utf-8')))
    return raw

class TestGateway(Gateway):
    def __init__(self,cfg,key):self.fixture_key=key;super().__init__(cfg)
    def raw(self):return fixture(self.fixture_key)

def static_checks():
    raw=fixture('total');bad=copy.deepcopy(raw);bad['stations'][0]['current']=-1
    try:evidence(bad);negative=False
    except ValueError:negative=True
    missing=evidence({'mode':'demo','stations':[{'id':'TEST','name':'미확인대여소','current':None,'predicted':2}],'plan':None})
    text=' '.join(x['text'] for x in missing['facts'])
    secret=copy.deepcopy(raw);secret['api_key']='unit-test-not-a-real-key'
    return [{'id':'negative_input_rejected','passed':negative},{'id':'missing_not_zero','passed':'자료 시점의 자전거 수는 없음' in text and '자료 시점의 자전거 0대' not in text},{'id':'unlisted_field_not_in_context','passed':'unit-test-not-a-real-key' not in json.dumps(evidence(secret))}]

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    request=json.load(sys.stdin);cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));mode=request['action']
    if mode=='report':
        out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        if (out/'results.json').exists():raise ValueError('Preserve prior acceptance')
        items=request['items'];normal=items[:10];checks=static_checks()+[{'id':x['id'],'passed':x['pattern_pass'],'engine':x['response']['engine']} for x in items[10:13]]
        correct=sum(x['pattern_pass'] and x['response']['engine']=='external' for x in normal)
        report={'hypothesis_id':'H-0027','kind':'first known-fixture acceptance after three development attempts','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'normal_questions':10,'correct_actual_external_answers':correct,'checks':checks,'long_route':items[13],'responses':items,'decision_component':'supported' if correct>=9 and all(x['passed'] for x in checks) else 'not_supported','api_panel_opened':False,'weights_updated':False,'limitations':['Known fixtures,not independent language benchmark','Prompt/transport and server grounding differ from H26; not a pure model-size causal comparison','Regex semantic checks do not prove every claim correct','Long-route and service integration reported separately']}
        (out/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'ok':True,'result':report},ensure_ascii=False));return
    key=request['case'];entry=next(x for x in CASES if x[0]==key);gateway=TestGateway(cfg,key)
    if mode=='prepare':value=gateway.prepare(entry[1])
    elif mode=='complete':
        response=gateway.complete(request['ticket'],request.get('candidate'),request.get('metadata'))
        match=bool(re.search(entry[2],response['answer'],re.DOTALL))
        if key=='unmet':match=match and not re.search(r'\d+\s*명',response['answer'])
        value={'id':key,'question':entry[1],'response':response,'pattern_pass':bool(match)}
    else:raise ValueError('Invalid test action')
    print(json.dumps({'ok':True,'result':value},ensure_ascii=False))

if __name__=='__main__':main()
