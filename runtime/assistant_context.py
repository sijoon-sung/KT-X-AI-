"""Allowlisted display context for the post-forecast assistant. No prediction math."""
import hashlib,json,math,re
from datetime import datetime,timezone,timedelta
from pathlib import Path

H13_LOCK='767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09'
KST=timezone(timedelta(hours=9))

def clean(value,limit=100):
    return re.sub(r'[\x00-\x1f<>]',' ',str(value))[:limit].strip()

def number(value,low=0,high=100000):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError('수량과 범위가 올바른지 확인하세요.')
    return float(value)

def when(value):
    if not value:return '자료 시각 미확인'
    if isinstance(value,(int,float)):dt=datetime.fromtimestamp(value,timezone.utc)
    else:dt=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('자료 시각에 시간대가 필요합니다.')
    return dt.astimezone(KST).strftime('%Y년 %m월 %d일 %H:%M')

def from_results(forecast=None,plan=None):
    """Adapt the existing runtime/run.py summary and dispatch response schemas."""
    out={'mode':'recorded_replay','recorded_at':None,'stations':[],'plan':None}
    if forecast:
        if forecast.get('model_lock_sha256')!=H13_LOCK:raise ValueError('확정 H13 예측 결과가 필요합니다.')
        out['recorded_at']=forecast.get('scheduled_at_unix_seconds') or forecast.get('issued_at_utc')
        out['mode']='demo' if forecast.get('mode')=='synthetic_offline_demo' else 'recorded_replay'
        for r in forecast.get('stations',[]):
            m=r.get('models',{}).get('recent_cfm')
            if not m:continue
            item={'id':clean(r['station_id'],50),'name':clean(r.get('name',r['station_id']))}
            if r.get('current_stock') is not None:item['current']=number(r['current_stock'])
            for src,dst in [('median','predicted'),('probability_zero','p_zero')]:
                if m.get(src) is not None:item[dst]=number(m[src],high=1 if src=='probability_zero' else 100000)
            out['stations'].append(item)
    if plan:
        if 'body' in plan:plan=plan['body']
        if 'bulk' in plan:plan=plan['bulk']
        if forecast and plan.get('recorded_at_utc'):
            a=when(out['recorded_at']);b=when(plan['recorded_at_utc'])
            if a!=b:raise ValueError('서로 다른 시각의 예측과 계획을 합칠 수 없습니다.')
        out['recorded_at']=out['recorded_at'] or plan.get('recorded_at_utc')
        route=plan.get('route',{})
        out['plan']={'moved_bikes':number(plan.get('moved_bikes',0)),
                     'distance_km':round(number(route.get('distance_meters',0))/1000,2),
                     'minutes':round(number(route.get('return_minutes',0)),1),'stops':[]}
        for s in route.get('stops',[])[:12]:
            if s.get('action') not in ('pickup','deliver'):raise ValueError('계획 작업 종류를 확인하세요.')
            out['plan']['stops'].append({'id':clean(s['station_id'],50),'name':clean(s.get('name',s['station_id'])),
                                         'action':s['action'],'bikes':number(s['bikes'])})
    return out

def evidence(raw,question=''):
    """Generate short immutable fact sentences; never pass arbitrary source keys."""
    if not isinstance(raw,dict):raise ValueError('예측 자료 형식을 확인하세요.')
    mode=raw.get('mode','recorded_replay')
    if mode not in ('demo','recorded_replay','forecast'):raise ValueError('자료 종류를 확인하세요.')
    stamp=when(raw.get('recorded_at'));label=('시연용 예시' if mode=='demo' else '기록된 자료')+' · '+stamp
    facts=[]
    def add(key,text,**metadata):facts.append({'id':key,'text':text,**metadata})
    add('source',label+'입니다. 현재 실시간 관측이라고 확인된 자료는 아닙니다.')
    add('method','예측 모델은 과거 이용 기록과 주변 대여소의 흐름을 보고 10분 뒤 자전거 수를 예상합니다.')
    add('limits','실제로 대여를 포기한 사람 수와 실제 이동 완료 여부는 이 자료로 알 수 없습니다.')
    stations=[]
    for s in raw.get('stations',[])[:3000]:
        if not isinstance(s,dict):raise ValueError('대여소 자료 형식을 확인하세요.')
        item={'id':clean(s.get('id',''),50),'name':clean(s.get('name',s.get('id','')))}
        if not item['id'] or not item['name']:raise ValueError('대여소 이름과 번호가 필요합니다.')
        for key in ('current','predicted','p_zero'):
            if s.get(key) is not None:item[key]=number(s[key],high=1 if key=='p_zero' else 100000)
        stations.append(item)
    matched=[s for s in stations if s['id'] in question or s['name'] in question]
    ranked=sorted(stations,key=lambda x:(-x.get('p_zero',0),x.get('current',1e9),x['id']))
    chosen=[]
    for s in matched+ranked:
        if s['id'] not in [x['id'] for x in chosen]:chosen.append(s)
        if len(chosen)>=5:break
    for s in chosen:
        parts=[s['name']]
        if 'current' in s:parts.append(f"자료 시점의 자전거 {s['current']:g}대")
        else:parts.append('자료 시점의 자전거 수는 없음')
        if 'predicted' in s:parts.append(f"10분 뒤 예상 자전거 {s['predicted']:g}대")
        if 'p_zero' in s:parts.append(f"10분 뒤 자전거가 없을 가능성 {s['p_zero']*100:.0f}%")
        add('station.'+s['id'],', '.join(parts)+'.',station_name=s['name'],station_id=s['id'],current=s.get('current'),predicted=s.get('predicted'))
    plan=raw.get('plan');summary={}
    if plan:
        moved=number(plan.get('moved_bikes',0));km=number(plan.get('distance_km',0));minutes=number(plan.get('minutes',0))
        summary={'moved_bikes':moved,'distance_km':km,'minutes':minutes}
        add('plan.total',f'총 {moved:g}대를 옮기는 계획입니다. 이동 거리 {km:g}km, 예상 시간 {minutes:g}분입니다. 실제 이동 완료 기록이 아닙니다.')
        for i,s in enumerate(plan.get('stops',[])[:12],1):
            if s.get('action') not in ('pickup','deliver'):raise ValueError('작업 종류가 올바르지 않습니다.')
            action='싣기' if s['action']=='pickup' else '내려놓기'
            add('route.'+str(i),f"방문 순서 {i}: {clean(s.get('name',s.get('id','대여소')))}, {number(s['bikes']):g}대 {action}.",bikes=number(s['bikes']))
    elif stations:add('plan.none','계산된 이동 계획은 아직 없습니다. 배치 대수를 확정해서 말할 수 없습니다.')
    else:add('data.none','예측 자료와 이동 계획이 아직 없습니다. 대여소 상태나 이동 수량을 알 수 없습니다.')
    if len(json.dumps(facts,ensure_ascii=False))>6000:raise ValueError('요약 자료가 너무 깁니다.')
    digest=hashlib.sha256(json.dumps(raw,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return {'context_id':digest,'label':label,'mode':mode,'facts':facts,'summary':summary,'station_count':len(stations)}

def load_context(context=None,forecast=None,plan=None):
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig')) if p else None
    if context and (forecast or plan):raise ValueError('문맥 파일 또는 예측/계획 파일 중 하나만 지정하세요.')
    return read(context) if context else from_results(read(forecast),read(plan))

def validate_answer(raw,pack):
    if not isinstance(raw,dict) or not isinstance(raw.get('answer'),str):raise ValueError('답변 형식 오류')
    text=raw['answer'].strip();ids=raw.get('evidence_ids',[])
    allowed={x['id'] for x in pack['facts']}
    if not 1<=len(text)<=700 or not isinstance(ids,list) or not ids or any(x not in allowed for x in ids):raise ValueError('답변 근거 오류')
    if any(x in text for x in ('<think>','<script','```')):raise ValueError('답변 형식 오류')
    number_re=r'\d+(?:\.\d+)?'
    source=' '.join(x['text'] for x in pack['facts'] if x['id'] in ids)
    if re.search(r'[\u4e00-\u9fff]',text) and not re.search(r'[\u4e00-\u9fff]',source):raise ValueError('자료에 없는 한자 표기')
    # Compare numeric values, accepting harmless trailing zeros, not novel arithmetic.
    numerical_facts=[x for x in pack['facts'] if x['id'] in ids and x['id']!='source']
    numeric_source=' '.join(x['text'] for x in numerical_facts) if numerical_facts else source
    numbers={float(x) for x in re.findall(number_re,numeric_source)}
    if any(float(x) not in numbers for x in re.findall(number_re,text)):raise ValueError('근거에 없는 숫자')
    routes=[x for x in numerical_facts if x['id'].startswith('route.')]
    if len(routes)==1:
        if any(float(x)!=routes[0]['bikes'] for x in re.findall(r'(\d+(?:\.\d+)?)\s*대',text)):
            raise ValueError('이동 수량이 계획과 다릅니다.')
    return {'answer':text,'evidence_ids':ids}

def focus(pack,question):
    """Select the supplied facts for this question; no new recommendations."""
    facts=pack['facts'];station=[x for x in facts if x['id'].startswith('station.') and (x.get('station_name','\0') in question or x.get('station_id','\0') in question)]
    if any(w in question for w in ('포기','미충족','못 빌','돌아간','실제 이동','완료')):
        chosen=[x for x in facts if x['id']=='limits']
    elif any(w in question for w in ('언제','실시간','시점이고','시각')):
        chosen=[x for x in facts if x['id']=='source']
    elif any(w in question for w in ('어떻게 하는','어떻게 예측','모델','예측 방법')):
        chosen=[x for x in facts if x['id']=='method']
    elif any(w in question for w in ('순서','동선','경로')):
        chosen=[x for x in facts if x['id'].startswith('route.')]
    elif any(w in question for w in ('싣','실어','내려','채우면','채워야')):
        routes=[x for x in facts if x['id'].startswith('route.')]
        matched=[x for x in routes if x['text'].split(': ',1)[-1].split(',')[0] in question]
        if not matched and any(w in question for w in ('내려','채우')):routes=[x for x in routes if '내려놓기' in x['text']]
        chosen=matched if matched else routes
    elif station:chosen=station
    elif any(w in question for w in ('없는 곳','0대인','비어 있는')):
        key='predicted' if any(w in question for w in ('10분','예상','미래')) else 'current'
        chosen=[x for x in facts if x['id'].startswith('station.') and x.get(key)==0]
    elif any(w in question for w in ('몇 대','몇대','거리','시간','총')):
        chosen=[x for x in facts if x['id']=='plan.total']
    else:chosen=[x for x in facts if x['id'].startswith(('station.','plan.','data.'))][:6]
    if not chosen:chosen=[x for x in facts if x['id'] in ('data.none','plan.none','limits')][:1]
    ids={x['id'] for x in chosen}
    if 'source' not in ids:chosen+=[x for x in facts if x['id']=='source']
    return dict(pack,facts=chosen)

def fallback(pack,question):
    f=pack['facts'];matches=[x for x in f if x['id'].startswith('station.') and any(word in question for word in re.findall(r'[^ (),.]+',x['text']) if len(word)>2)]
    if any(w in question for w in ('포기','미충족','실제 이동','완료')):chosen=[x for x in f if x['id']=='limits']
    elif any(w in question for w in ('언제','실시간','시각','지금 자료')):chosen=[x for x in f if x['id']=='source']
    elif any(w in question for w in ('모델','예측 방법','어떻게 예측')):chosen=[x for x in f if x['id']=='method']
    elif any(w in question for w in ('순서','동선','경로','싣','실어','내려','채우')):
        routes=[x for x in f if x['id'].startswith('route.')]
        matched=[x for x in routes if x['text'].split(': ',1)[-1].split(',')[0] in question]
        chosen=matched or routes
    elif any(w in question for w in ('몇 대','몇대','거리','시간')):chosen=[x for x in f if x['id']=='plan.total'] or matches
    else:chosen=matches or [x for x in f if x['id'].startswith(('station.','plan.','data.'))][:3]
    if not chosen:chosen=[x for x in f if x['id'] in ('data.none','plan.none','limits')][:1]
    return {'answer':' '.join(x['text'] for x in chosen[:6]),'evidence_ids':[x['id'] for x in chosen[:6]]}
