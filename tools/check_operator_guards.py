"""Regression checks for observed H26 quantity, route and zero-stock failures."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from assistant_context import evidence,focus,validate_answer
fixture=json.loads((ROOT/'examples/assistant-demo.json').read_text(encoding='utf-8'))
def rejected(question,answer):
    pack=focus(evidence(fixture,question),question)
    try:validate_answer({'answer':answer,'evidence_ids':[x['id'] for x in pack['facts']]},pack)
    except ValueError:return True
    return False
assert rejected('서쪽공원에서 몇 대를 싣는 계획이야?','서쪽공원에서 12대를 싣습니다.')
assert rejected('어떤 순서로 이동하면 돼?','서쪽공원 → 시청앞')
assert rejected('자전거를 총 몇 대 옮기는 계획이야?','총 2026대를 옮깁니다.')
zero=focus(evidence(fixture,'자전거가 없는 곳을 알려줘'),'자전거가 없는 곳을 알려줘')
assert [x['station_name'] for x in zero['facts'] if x['id'].startswith('station.')]==['중앙역']
assert rejected('자전거가 없는 곳을 알려줘','자전거가 없는 대여소는 없습니다.')
print(json.dumps({'regressions':5,'passed':5,'scope':'observed implementation failures; not a new model-performance test'}))
