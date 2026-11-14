"""Compute the recorded fleet scenario and verify answers on dislab."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'runtime'))
from assistant_context import evidence, validate_answer
from review_dispatch_scenarios import FLEET, inspect


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def plan(config):
    from scenario_dispatch import calculate
    from dispatch_engine import Roads
    specs = {item['id']: item for item in read(Path(config['spec']))['cases']}
    roads = Roads(Path(config['roads']))
    cases = [calculate(specs[key], Path(config['stream']), roads) for key in FLEET]
    review = inspect({'cases': cases})
    if not review['simultaneous']:
        raise ValueError('작업지가 겹치거나 계산할 수 없는 권역이 있습니다.')
    fields = ('id', 'name', 'status', 'geometry', 'moved_bikes', 'distance_km', 'work_minutes', 'returned', 'depot', 'observation')
    stop_fields = ('station_id', 'name', 'action', 'bikes', 'completion', 'arrival', 'lon', 'lat')
    public = [dict({k: c[k] for k in fields}, stops=[{k:s[k] for k in stop_fields} for s in c['stops']]) for c in cases]
    review['selected'] = [f for f in review['facts'] if f['category'] == 'timing']
    return {'cases': public, 'review': review, 'mode': 'recorded_replay'}


def route_pack(case):
    raw = {'mode': 'recorded_replay', 'recorded_at': case['observation'], 'plan': {
        'moved_bikes': case['moved_bikes'], 'distance_km': case['distance_km'], 'minutes': case['work_minutes'],
        'stops': [dict(s, id=s['station_id']) for s in case['stops']]}}
    pack = evidence(raw)
    pack['facts'] = [fact for fact in pack['facts'] if fact['id'].startswith('route.')]
    pack['required_station_names'] = [s['name'] for s in case['stops']]
    return pack


def prepare(case, question):
    pack = route_pack(case)
    ids = [f['id'] for f in pack['facts']]
    schema = {'type':'object', 'additionalProperties':False, 'required':['answer','evidence_ids'], 'properties':{
        'answer': {'type':'string'}, 'evidence_ids': {'type':'array','items':{'type':'string','enum':ids}}}}
    prompt = ('타슈 배차 담당자에게 현재 작업표를 설명하세요. 질문에 맞춰 방문 순서대로 한 줄씩 답하세요. '
              '대여소 이름을 그대로 쓰고 각 장소에 수거 또는 배치와 수량을 붙이세요. '
              '거리와 시각은 생략하세요. 파일·웹·명령·MCP 등 도구를 사용하지 마세요. 모든 근거 ID를 반환하세요.\n' +
              '\n'.join(f"[{f['id']}] {f['text']}" for f in pack['facts']) + '\n질문: ' + question)
    return {'schema':schema, 'prompt':prompt, 'model':'gemini-3.8-flash-low'}


def complete(case, candidate):
    pack = route_pack(case)
    answer = validate_answer(candidate, pack)
    if set(answer['evidence_ids']) != {f['id'] for f in pack['facts']}:
        raise ValueError('답변이 작업표의 방문 순서를 모두 포함하지 않습니다.')
    text = answer['answer']
    for i, stop in enumerate(case['stops']):
        first = text.index(stop['name']) + len(stop['name'])
        last = text.index(case['stops'][i+1]['name'], first) if i+1 < len(case['stops']) else len(text)
        segment = text[first:last]
        words = ('수거','싣','실어') if stop['action']=='pickup' else ('배치','내려','놓')
        if not any(word in segment for word in words):
            raise ValueError('답변의 수거·배치를 작업표와 맞추지 못했습니다.')
    return {'answer':text, 'engine':'external', 'plan_digest':hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    request = json.load(sys.stdin)
    action = request['action']
    if action == 'plan': result = plan(read(args.config))
    elif action == 'prepare': result = prepare(request['case'], request['question'])
    elif action == 'complete': result = complete(request['case'], request['candidate'])
    else: raise ValueError('Unknown action')
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
