"""Recheck work orders and produce a bounded LLM review request or report."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

FLEET=('taepyeong-2050','dunsan-2050','station-2050')
MODEL='gemini-3.8-flash-low'


def stamp(text):return datetime.fromisoformat(text).timestamp()
def near(a,b):return math.isclose(a,b,abs_tol=1e-6,rel_tol=0)
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate(case):
    if case['status']!='plan':
        if case['stops'] or case.get('moved_bikes',0):raise ValueError('Non-plan contains movements')
        return None
    p=case['parameters'];origin=stamp(case['departure']);work_cutoff=stamp(case['cutoff']);return_cutoff=stamp(case['return_deadline'])
    if not near(stamp(case['target'])-work_cutoff,30) or not near(return_cutoff-origin,p['deadline_minutes']*60):
        raise ValueError('Inconsistent deadline')
    clock=0.;moved=0;load=0;seen=set();distance=0.
    for stop in case['stops']:
        sid=stop['station_id'];quantity=stop['bikes'];mode=stop['action']
        if sid in seen or type(quantity)!=int or quantity<max(3,p['minimum_transfer']):raise ValueError('Invalid working stop')
        seen.add(sid)
        if not(0<=stop['leg_drive_minutes'] and 0<=stop['leg_meters']):raise ValueError('Invalid road leg')
        clock+=stop['leg_drive_minutes'];distance+=stop['leg_meters']
        if not near(clock,stop['eta_minutes']) or abs(stamp(stop['arrival'])-origin-clock*60)>.501:raise ValueError('Arrival time mismatch')
        if mode=='pickup':
            if stop['current']-quantity<p['target']:raise ValueError('Donor stock too low')
            load+=quantity
        elif mode=='deliver':
            if quantity>p['drop_limit']:raise ValueError('Parking limit exceeded')
            load-=quantity;moved+=quantity
        else:raise ValueError('Unknown action')
        if not 0<=load<=p['capacity'] or load!=stop['vehicle_load_after']:raise ValueError('Vehicle load mismatch')
        clock+=(p['stop_setup_seconds']+quantity*p['handling_seconds'])/60
        if not near(clock,stop['completion_minutes']) or abs(stamp(stop['completion'])-origin-clock*60)>.501:
            raise ValueError('Working time mismatch')
        if origin+clock*60>work_cutoff+1e-7:raise ValueError('Station work exceeds the forecast cutoff')
    if not case['geometry'] or case['geometry']['type']!='LineString' or len(case['geometry']['coordinates'])<2:raise ValueError('Road geometry missing')
    if load or moved!=case['moved_bikes'] or moved>p['budget'] or clock>case['work_minutes']+1e-7 or distance>case['distance_km']*1000+.501:
        raise ValueError('Route totals inconsistent')
    if origin+case['work_minutes']*60>return_cutoff+1e-7 or abs(stamp(case['returned'])-origin-case['work_minutes']*60)>.501:
        raise ValueError('Return cutoff or time mismatch')
    if not near(case['setup_minutes'],len(seen)*p['stop_setup_seconds']/60) or not near(case['handling_minutes'],2*moved*p['handling_seconds']/60):
        raise ValueError('Work time accounting mismatch')
    if case['certificate']['violations']!=0:raise ValueError('Planner certificate failed')
    return dict(id=case['id'],work_slack_seconds=round(work_cutoff-origin-clock*60),
                return_slack_seconds=round(return_cutoff-origin-case['work_minutes']*60),
                work_minutes_per_bike=round(case['work_minutes']/moved,2),verified=True)


def inspect(data):
    cases={case['id']:case for case in data['cases']}
    if len(cases)!=len(data['cases']):raise ValueError('Duplicate case IDs')
    checks={key:validate(case) for key,case in cases.items()}
    fleet=[cases[k] for k in FLEET if k in cases]
    touched=[s['station_id'] for c in fleet for s in c['stops']]
    simultaneous=(len(fleet)==3 and all(c['status']=='plan' for c in fleet) and len({c['source_version'] for c in fleet})==1 and len(touched)==len(set(touched)))
    facts=[]
    def add(key,category,message):facts.append(dict(id=key,category=category,text=message))
    for key,c in cases.items():
        if c['status']!='plan':
            add(key+'-status','hold',f"{c['name']}: {c['reason']}");continue
        row=checks[key];where=c['name']
        add(key+'-result','summary',f"{where}는 {len(c['stops'])}곳에서 {c['moved_bikes']}대를 작업하는 {c['distance_km']:.2f}km 경로입니다.")
        if row['work_slack_seconds']<60:
            add(key+'-cutoff','timing',f"{where}은 배치 마감까지 {row['work_slack_seconds']}초가 남습니다. 출발 전에 도로 지연과 작업 시간을 확인하세요.")
        if row['return_slack_seconds']<60:
            add(key+'-return','timing',f"{where}은 복귀 마감까지 {row['return_slack_seconds']}초가 남습니다. 지연이 예상되면 조건을 다시 확인하세요.")
    if simultaneous:
        add('fleet-disjoint','summary',f"세 차량이 각 권역에서 독립적으로 움직이는 가정에서 {len(touched)}곳의 작업 대상이 겹치지 않으며 합계 {sum(c['moved_bikes'] for c in fleet)}대를 옮기는 계획입니다.")
    if 'split-2050' in cases and cases['split-2050']['status']=='plan':
        add('parking-assumption','field',f"배치 가능 공간을 대여소당 3대로 가정한 결과는 {cases['split-2050']['moved_bikes']}대입니다. 출발 전에 실제 빈자리를 확인하세요.")
    add('work-assumptions','field','정차 준비 1분과 대당 상하차 20초를 가정했습니다. 운영 담당자가 현장의 작업 시간을 확인하세요.')
    return dict(checks=checks,simultaneous=simultaneous,fleet=list(FLEET) if simultaneous else [],facts=facts)


def preparation(verified):
    facts=verified['facts'];ids=[row['id'] for row in facts]
    schema=dict(type='object',additionalProperties=False,required=['evidence_ids'],properties={
        'evidence_ids':dict(type='array',minItems=2,maxItems=4,items=dict(type='string',enum=ids))})
    prompt=('타슈 재배치 담당자의 검토 순서를 정합니다. 아래 사실 중 출발 전에 먼저 볼 2~4개를 순서대로 선택하세요. '
            '시간 여유가 짧거나 현장 확인이 필요한 항목을 먼저 선택하세요. 새로운 경로·숫자를 만들지 말고 ID만 반환하세요.\n'+
            '\n'.join(f"[{row['id']}] {row['text']}" for row in facts))
    return dict(model=MODEL,schema=schema,prompt=prompt)


def finish(verified,candidate):
    known={row['id']:row for row in verified['facts']};ids=candidate.get('evidence_ids') if isinstance(candidate,dict) else None
    accepted=isinstance(ids,list) and 2<=len(ids)<=4 and all(isinstance(key,str) and key in known for key in ids) and len(set(ids))==len(ids)
    if not accepted:
        ids=[row['id'] for row in verified['facts'] if row['category'] in ('timing','field')][:4]
    return dict(mode='llm_priority' if accepted else 'rule_priority',selected=[known[k] for k in ids],**verified)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plans',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prepare',action='store_true');p.add_argument('--candidate',type=Path);a=p.parse_args()
    verified=inspect(json.loads(a.plans.read_text(encoding='utf-8')))
    result=preparation(verified) if a.prepare else finish(verified,json.loads(a.candidate.read_text(encoding='utf-8-sig')) if a.candidate else None)
    write(a.out,result)
