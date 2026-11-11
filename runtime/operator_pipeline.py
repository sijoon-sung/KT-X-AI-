"""DISLAB-only adapter for locked forecasts, H21 plans and assistant context.

Replay has an explicit historical departure time. It never refreshes an expired
forecast into a live plan, requests the inventory API or updates model weights.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research_code'))
from assistant_context import H13_LOCK, from_results
from live_api_service import read, atomic, reference_now
from dispatch_engine import information, Roads
import bulk_dispatch


def snapshot(source):
    folder = Path(source['stream']).resolve()
    status = read(folder / 'status.json')
    state_path = (folder / status['latest_state']).resolve()
    if not state_path.is_relative_to(folder):
        raise ValueError('관측 파일 위치를 확인하세요.')
    state = read(state_path)
    if state.get('model_lock_sha256') != H13_LOCK:
        raise ValueError('확정 모델의 예측 자료가 필요합니다.')
    return folder, status, state


def active(source, status, state):
    if source.get('mode') != 'live' or status.get('phase') in ('failed', 'stopped'):
        return False
    now = reference_now(status)
    return 0 <= now - datetime.fromisoformat(state['recorded_at_utc']).timestamp() <= 720 and now < datetime.fromisoformat(state['target_at_utc']).timestamp()


def info(source):
    _, status, state = snapshot(source)
    value = information(state)
    value.update(source_version=state['source_version'],
                 mode='live_observation_shadow' if active(source, status, state) else 'recorded_replay',
                 feed={k: status.get(k) for k in ('phase', 'latest_observation', 'latest_forecast', 'requests', 'roundtrips', 'stop_reason')})
    value['operating_defaults'] = dict(capacity=20, budget=20, target=6, drop_limit=10, handling_seconds=20, max_visits=8)
    bulk_dispatch.display_limits(value)
    if source.get('planner') == 'batch':
        value['planner'] = 'setup-aware-batch'
        value['operating_defaults'].update(minimum_transfer=3,stop_setup_seconds=60,deadline_minutes=30)
        value['limits'].update(deadline_minutes=120,minimum_transfer=10,stop_setup_seconds=600)
        cache=Path(source['plan_cache'])
        if source.get('mode')=='recorded_replay' and cache.is_file():
            previous=read(cache)
            if previous.get('source_version')==state['source_version']:
                value['initial_plan']=previous
    return value


def optimize(source, body):
    folder, status, state = snapshot(source)
    body = dict(body or {})
    version = body.pop('source_version', None)
    if version != state['source_version']:
        return dict(status=409, body=dict(code='source_changed', error='자료가 바뀌었습니다. 다시 계산해 주세요.', current_info=info(source)))
    replay = source.get('mode') == 'recorded_replay'
    if not replay and not active(source, status, state):
        return dict(status=409, body=dict(code='forecast_expired', error='다음 관측을 기다리고 있습니다. 지난 계획은 현재 배차에 사용할 수 없습니다.'))
    if replay:
        offset = float(source.get('replay_offset_seconds', 15))
        if not 0 <= offset <= 60:
            raise ValueError('기록 재생 출발 시각을 확인하세요.')
        departure = datetime.fromisoformat(state['recorded_at_utc']).timestamp() + offset
        if source.get('planner') == 'batch':
            from batch_operations import optimize as batch_optimize
            result = batch_optimize(state, body, Roads(folder / 'h27-replay-roads'), departure)
        else:
            result = bulk_dispatch.base.optimize(state, body, Roads(folder / 'h27-replay-roads'), departure)
        result.update(mode='recorded_replay', issued_at_utc=datetime.now(timezone.utc).isoformat(), planner='bulk-capacity-incumbent-union',
                      planning_basis='Recorded forecast replay at observation + explicit offset; assumed resources; never live dispatch')
    else:
        if source.get('planner') == 'batch':
            from batch_operations import optimize as batch_optimize
            departure=reference_now(status)
            result=batch_optimize(state,body,Roads(folder / 'h27-replay-roads'),departure)
            finish=max([0.]+[s['completion_minutes'] for s in result['route']['stops']])
            if reference_now(status)+finish*60+30>datetime.fromisoformat(state['target_at_utc']).timestamp():
                return dict(status=409,body=dict(code='forecast_expired',error='계산 중 작업 가능 시간이 부족해 다음 관측을 기다립니다.'))
            response=dict(status=200,body=result)
        else:
            response = bulk_dispatch.rpc(folder, dict(body=dict(body, source_version=version)))
        if response['status'] != 200:
            return response
        result = response['body']
        result['mode'] = 'live_observation_shadow'
    if source.get('planner') == 'batch':
        result['planner']='setup-aware-batch'
    lookup = {s['id']: s for s in state['stations']}
    for stop in result['route']['stops']:
        stop['name'] = lookup[stop['station_id']]['name']
    result['source_version'] = state['source_version']
    result['assumptions'] = '차량·거점·적재량·정차·상하차 시간과 최소 묶음 수량은 화면의 가정입니다. 실제 배차 기록이 아닙니다.'
    atomic(source['plan_cache'], result)
    return dict(status=200, body=result)


def context(source):
    _, status, state = snapshot(source)
    plan_path = Path(source['plan_cache'])
    plan = read(plan_path) if plan_path.is_file() else None
    if plan and plan.get('source_version') != state['source_version']:
        plan = None
    forecast = dict(model_lock_sha256=H13_LOCK, issued_at_utc=state['recorded_at_utc'], stations=[
        dict(station_id=s['id'], name=s['name'], current_stock=s['current'], models={'recent_cfm': dict(median=s['forecast']['median'], probability_zero=s['forecast']['p0'])}) for s in state['stations']])
    value = from_results(forecast, plan)
    value['mode'] = 'forecast' if active(source, status, state) else 'recorded_replay'
    value['source_version'] = state['source_version']
    return value
