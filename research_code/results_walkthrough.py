"""H16 private evidence bundle. Run on dislab, never fit or dispatch.

The recorded future is loaded only after the replay plan is sealed. Forecast
quality is observed; relocation benefit is unobserved. Raw station records stay
in a private, loopback-served bundle, outside the public repository.
"""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from urllib.request import Request, urlopen

import numpy as np
import scipy
from rolling_inventory_metrics import LABELS, scores
from rebalancing_pipeline import plan, validate

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def iso(stamp):
    return datetime.fromtimestamp(float(stamp), timezone.utc).isoformat()


def rows(snapshot):
    values = snapshot['payload']['results']
    ids = [str(r['id']) for r in values]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate observation ID.')
    return {str(r['id']): r for r in values}


def distance(a, b):
    lat1, lon1, lat2, lon2 = np.radians([a['lat'], a['lon'], b['lat'], b['lon']])
    h = np.sin((lat2-lat1)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin((lon2-lon1)/2)**2
    return float(12742*np.arcsin(np.sqrt(np.clip(h, 0, 1))))


def select_candidates(stations):
    """Only forecast/current fields; deliberately no future observation input."""
    donors = [s for s in stations if s['current'] >= 3 and s['cfm_min'] >= 3 and s['lat'] is not None]
    if not donors:
        return []
    anchor = min(donors, key=lambda s: (-s['current'], s['id']))
    other = sorted((s for s in donors if s['id'] != anchor['id']), key=lambda s: (distance(anchor, s), s['id']))[:1]
    sinks = sorted((s for s in stations if s['current'] < 2 and s['lat'] is not None), key=lambda s: (distance(anchor, s), s['id']))[:3]
    return [anchor]+other+sinks


def road_query(service, locations, suffix, attempts):
    coordinates = ';'.join(f"{p['lon']:.6f},{p['lat']:.6f}" for p in locations)
    url = f'https://router.project-osrm.org/{service}/v1/driving/{coordinates}?{suffix}'
    error = None
    for _ in range(attempts):
        try:
            req = Request(url, headers={'User-Agent': 'TashuResearch-H16/1.0', 'Accept': 'application/json'})
            with urlopen(req, timeout=20) as response:
                raw = response.read(4_000_000)
            data = json.loads(raw)
            if data.get('code') != 'Ok':
                raise ValueError('OSRM returned '+str(data.get('code')))
            return data, dict(status='ok', service=service, response_sha256=hashlib.sha256(raw).hexdigest(), url=url,
                              scope='present OSM car estimate; no historical traffic or truck access verification')
        except Exception as exc:
            error = type(exc).__name__+': '+str(exc)
    return None, dict(status='unavailable', service=service, error=error, fallback_used=False)


def replay_plan(stations, samples, metadata):
    chosen = select_candidates(stations)
    if len(chosen) < 2:
        return dict(status='blocked', reason='Insufficient forecast/current donor or deficit support', dispatch_allowed=False)
    road, road_audit = road_query('table', [chosen[0]]+chosen, 'annotations=duration', 2)
    selected = [{k:s[k] for k in ('id','name','lat','lon','current')} for s in chosen]
    if road is None:
        return dict(status='blocked', reason='Road-time query unavailable; no invented time substituted', selected_stations=selected,
                    roads=road_audit, dispatch_allowed=False)
    req = dict(mode='replay_shadow', forecast=dict(id='H13-recorded-20261031-2130', source_kind='api_forecast',
        station_ids=[s['id'] for s in chosen], current_inventory=[s['current'] for s in chosen],
        stock_scenarios=[samples[s['index']].tolist() for s in chosen], horizon_minutes=10, supported_horizons=[10],
        model_lock_sha256=metadata['model_lock_sha256'], model_status='frozen_research_reference'),
        stations=[dict(id=s['id'],target_buffer=2,pickup_limit=6,dropoff_limit=2,loss_weight=1.) for s in chosen],
        fleet=dict(vehicle_count=1,capacity=6,initial_load=0,depot='ASSUMED_DEPOT',max_route_minutes=10),
        policy=dict(workload_budget=6,max_stops=5,risk_alpha=.9,source_lower_quantile=.1,risk_weight=1.,
                    travel_unit_weight=.01,handling_unit_cost=.01,handling_minutes_per_bike=.1),
        road_matrix=dict(ids=['ASSUMED_DEPOT']+[s['id'] for s in chosen],basis=road_audit['scope'],
                         duration_minutes=[[None if v is None else v/60 for v in row] for row in road['durations']]))
    result = plan(req)
    c = validate(req)
    net = np.zeros(len(chosen), dtype=int)
    for action in result['quantity_actions']:
        net[c['ids'].index(action['station_id'])] = action['net_station_change']
    assert int(net.sum()) == 0
    assert result['route']['peak_load'] <= 6 and result['route']['return_minutes'] <= 10+1e-8
    assert not result['dispatch_allowed']
    lookup = {s['id']:s for s in chosen}; lookup['ASSUMED_DEPOT'] = chosen[0]
    locations = [lookup[k] for k in result['route']['sequence']]
    geometry = None
    if result['total_moved_bikes']:
        route_data, route_audit = road_query('route', locations, 'overview=full&geometries=geojson&steps=false', 1)
        if route_data is not None:
            geometry = route_data['routes'][0]['geometry']
    else:
        route_audit = dict(status='not_requested', reason='No transfer selected')
    result.update(selected_stations=selected, roads=road_audit, geometry_audit=route_audit, geometry=geometry,
        assumed_depot=dict(lat=chosen[0]['lat'],lon=chosen[0]['lon'],name=chosen[0]['name']),
        assumptions=dict(vehicles=1,capacity=6,target_buffer=2,handling_minutes_per_bike=.1,
                         depot='assumed at selected donor; not real operating base',parking='unverified; assumed dropoff limit2'),
        field_outcome=None, actual_dispatch=False, dependence_scope='H13 stationwise cached CFM, not H14 learned joint dependence',
        decision_input_sha256=hashlib.sha256(json.dumps(req,sort_keys=True).encode()).hexdigest())
    return result


def build(a):
    started = time.perf_counter()
    paths = read(a.inputs)
    manifest = read(ROOT/'research/datasets/H-0016-evidence-inputs.json')
    checks = []
    def check(name, condition):
        checks.append(dict(name=name, passed=bool(condition)))
        if not condition:
            raise ValueError('Acceptance failed: '+name)
    check('frozen_input_hashes', all(digest(paths[x['id']]) == x['sha256'] for x in manifest['inputs']))
    current = read(paths['current']); lookup = rows(current)
    with np.load(paths['forecast'], allow_pickle=False) as p:
        ids = p['station_ids'].astype(str).tolist(); inventory = p['current'].copy()
        ensembles = {k:p[k].copy() for k in LABELS}
        metadata = json.loads(str(p['metadata_json']))
        issued = float(p['issued_at_unix_seconds']); target = int(p['target_scheduled']); scheduled = int(p['scheduled'])
        actual_current = p['actual'].copy()
    check('issued_before_target', scheduled <= issued < target)
    check('forecast_current_snapshot_hash', metadata['snapshot_sha256'] == digest(paths['current']))
    check('unique_forecast_ids', len(ids) == len(set(ids)))
    stations = []
    for i, sid in enumerate(ids):
        src = lookup[sid]; lat, lon = float(src['x_pos']), float(src['y_pos'])
        geographic = np.isfinite([lat,lon]).all() and 33<lat<39 and 124<lon<132
        stats = {}
        for label in LABELS:
            x = np.sort(ensembles[label][i]); size=len(x)
            stats[label] = dict(median=int(x[int(np.ceil(.5*size))-1]),low=int(x[int(np.ceil(.05*size))-1]),
                high=int(x[int(np.ceil(.95*size))-1]),mean=float(x.mean()),p0=float((x==0).mean()))
        stations.append(dict(index=i,id=sid,name=src.get('name',sid),current=int(inventory[i]),lat=lat if geographic else None,
            lon=lon if geographic else None,cfm_min=int(ensembles['recent_cfm'][i].min()),forecast=stats))
    check('cached_current_matches_observation', all(s['current'] == lookup[s['id']]['parking_count'] for s in stations))
    # This whole block intentionally runs before opening any future observation.
    shadow = replay_plan(stations, ensembles['recent_cfm'], metadata)
    selection_before = [s['id'] for s in select_candidates(stations)]
    decision_sealed_at = time.perf_counter()-started
    # The future is for observed forecast error, never counterfactual relocation.
    future = read(paths['truth']); future_lookup = rows(future)
    check('issued_before_future_capture', issued < future['observed_at_unix_seconds'] and target == future['scheduled_at_unix_seconds'])
    valid=[]; truth=[]
    for i,s in enumerate(stations):
        row = future_lookup.get(s['id']); value = row.get('parking_count') if row else None
        elapsed = future['observed_at_unix_seconds']-actual_current[i]
        ok = row is not None and not isinstance(value,bool) and isinstance(value,(float,int)) and np.isfinite(value) and value>=0 and value==int(value)
        ok = ok and 0 <= future['observed_at_unix_seconds']-target <= 120 and 480 <= elapsed <= 720
        s['truth'] = int(value) if ok else None
        if ok: valid.append(i); truth.append(value)
    original_score = read(paths['live_score'])
    calculated = {k:{name:float(v.mean()) for name,v in scores(ensembles[k][valid],truth).items()} for k in LABELS}
    check('exact_original_pair_count', len(valid) == original_score['pairs'])
    check('original_live_scores_reproduced', all(abs(v-original_score['metrics'][k][name])<1e-10 for k,metrics in calculated.items() for name,v in metrics.items()))
    modified=[dict(s,truth=-999) for s in stations]
    check('future_labels_do_not_enter_selection', selection_before == [s['id'] for s in select_candidates(modified)])
    check('no_dispatch_or_counterfactual_field_score', shadow['dispatch_allowed'] is False and shadow.get('field_outcome') is None)
    h14=read(paths['h14_fit']); h15=read(paths['h15_acceptance']); september=read(paths['september_score'])
    check('h14_gate_fail_preserved', h14['historical_gate_passed'] is False)
    check('h15_synthetic_scope_preserved', h15['tests_run']==13 and h15['metrics']['acceptance_constraint_violations']==0)
    if shadow['status'] != 'blocked':
        check('real_station_shadow_capacity_deadline', shadow['route']['peak_load']<=6 and shadow['route']['return_minutes']<=10+1e-8)
    else:
        check('unavailable_roads_explicit_no_fallback', shadow.get('roads',{}).get('status')=='unavailable' or not shadow.get('selected_stations'))
    ordered=sorted(range(len(stations)),key=lambda i:(-stations[i]['forecast']['recent_cfm']['p0'],stations[i]['id']))
    for rank,i in enumerate(ordered):stations[i]['display_rank']=rank
    output=dict(schema_version=1,hypothesis_id='H-0016',mode='recorded_replay',not_live=True,source='private verified dislab bundle',
        episode=dict(scheduled_at_utc=iso(scheduled),issued_at_utc=iso(issued),target_at_utc=iso(target),
                     observed_future_at_utc=iso(future['observed_at_unix_seconds']),pairs=len(valid),
                     issued_before_truth=True,weather_present=False,scope='single recorded live functional roundtrip, not independent accuracy evidence'),
        stations=stations,live_metrics=calculated,september=dict(coverage=september['coverage'],groups=september['groups'],summary=september['summary']),
        joint_experiment=dict(hypothesis_id='H-0014',historical_gate_passed=False,external_primary_status='unexecuted',
                              validation=h14['validation'],gate_relative_gain=h14['gate_relative_gain']),
        synthetic_example=h15['outputs']['main'],synthetic_acceptance=dict(tests=13,violations=0),shadow=shadow,
        provenance=dict(input_hashes=manifest['inputs'],original_model='H-0013',model_lock_sha256=metadata['model_lock_sha256'],
                        forecast_sha256=digest(paths['forecast']),input_api_only=True,weights_updated=False,
                        future_opened_after_shadow_sealed=True,shadow_sealed_elapsed_seconds=decision_sealed_at),checks=checks,
        limitations=['Recorded observation, not current live availability.','No real relocation was performed.','Resource and parking inputs are assumptions.',
                     'OSRM car estimates do not verify a truck route or historical traffic.','H14 FM failed its historical gate; no new official evaluation.',
                     'Shadow loss is a scenario proxy, not measured field improvement.'])
    write(a.out,output)
    elapsed=time.perf_counter()-started; start=datetime.fromisoformat(a.collector_start)
    summary=dict(schema_version=1,hypothesis_id='H-0016',started_at_utc=start.isoformat(),finished_at_utc=(start+timedelta(seconds=elapsed)).isoformat(),
        native_remote_finished_at_utc=datetime.now(timezone.utc).isoformat(),code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        environment=dict(device='dislab CPU',python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
        checks=checks,metrics=dict(acceptance_failures=sum(not c['passed'] for c in checks)),
        stations=len(stations),scored_pairs=len(valid),shadow_status=shadow['status'],shadow_moved_bikes=shadow.get('total_moved_bikes',0),
        shadow_route_minutes=shadow.get('route',{}).get('return_minutes'),shadow_assumptions=shadow.get('assumptions'),
        roads=shadow.get('roads'),real_dispatch=False,field_benefit_observed=False,seconds=elapsed,bundle_sha256=digest(a.out))
    write(Path(a.out).with_suffix('.summary.json'),summary)
    print(json.dumps({k:summary[k] for k in ('stations','scored_pairs','shadow_status','shadow_moved_bikes','shadow_route_minutes','metrics','seconds')}))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--out',required=True);p.add_argument('--collector-start',required=True)
    build(p.parse_args())
