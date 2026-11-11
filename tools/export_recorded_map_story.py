"""Extract visual evidence from one locked forecast replay on dislab."""
import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch
from urllib.request import Request, urlopen

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'research_code'))
import batch_operations
from dispatch_engine import Roads
import forecast_transfer
import rolling_inventory_model as model
import rolling_inventory_predict as prediction

STAMP = 1794311400
STATE_HASH = '64a4ba791422308ba58ff494cd026170a249b59d1c0fb294a96fb41140f7b2e9'
LOCK = '767061e696f2b72fd48a14dbd4ea1851921bd3481405f6887e48d28ee8acbe09'
BOUNDS = (127.3865, 36.3085, 127.4150, 36.3240)
KST = timezone(timedelta(hours=9))
SAMPLES = (0, 1, 2, 3)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


class CachedRoads(Roads):
    def query(self, service, locations, options):
        coords = ';'.join(f"{s['lon']:.6f},{s['lat']:.6f}" for s in locations)
        url = f'https://router.project-osrm.org/{service}/v1/driving/{coords}?{options}'
        key = hashlib.sha256(url.encode()).hexdigest()
        return batch_operations.read(self.cache / (key+'.json')), key


def visual_stations(state, plan):
    left, bottom, right, top = BOUNDS
    visible = [s for s in state['stations'] if left < s['lon'] < right and bottom < s['lat'] < top]
    endpoints = {s['station_id'] for s in plan['route']['stops']}
    endpoints.add(plan['parameters']['depot_id'])
    # Geographical selection, independent of predicted stock and sample values.
    cx, cy = (left+right)/2, (bottom+top)/2
    visible.sort(key=lambda s: (s['id'] not in endpoints, (s['lon']-cx)**2*.65+(s['lat']-cy)**2, s['id']))
    return visible[:22]


def trace_replay(predictor, current, previous, weather, selected_ids):
    original_counts = model.iid_counts
    node_ids = {i: str(s) for i, s in enumerate(predictor.ids) if str(s) in selected_ids}
    traces = {sid: {} for sid in selected_ids}
    sample_counts = {}

    def tracked_counts(models, flows, context, rows, nodes, kind, seed, encoded=None, rng=None, generator=None):
        if flows is None:
            return original_counts(models, flows, context, rows, nodes, kind, seed, encoded, rng, generator)
        names = [node_ids.get(int(i)) for i in nodes.cpu().tolist()]
        chosen = deepcopy(rng).integers(0, 2, (len(rows), model.M))
        batches = iter(np.where(chosen == i) for i in range(2))

        @torch.inference_mode()
        def traced_heun(flow, z, hidden, center, steps=16):
            row_indices, sample_indices = next(batches)
            captured = [(i, names[r], int(s)) for i,(r,s) in enumerate(zip(row_indices,sample_indices))
                        if names[r] is not None and s in SAMPLES]
            index = torch.as_tensor([p[0] for p in captured], dtype=torch.long, device=z.device)
            path = [z[index].cpu().numpy()]
            for step in range(steps):
                velocity = flow(z, z.new_tensor(step/steps), hidden, center)
                trial = z+velocity/steps
                correction = flow(trial,z.new_tensor((step+1)/steps),hidden,center)
                z = z+(velocity+correction)/(2*steps)
                path.append(z[index].cpu().numpy())
            for (_,sid,sample), points in zip(captured,np.stack(path).transpose(1,0,2)):
                traces[sid][sample] = points.tolist()
            return z

        with patch.object(model,'heun',traced_heun):
            counts = original_counts(models,flows,context,rows,nodes,kind,seed,encoded,rng,generator)
        for row,sid in enumerate(names):
            if sid is not None:
                sample_counts[sid] = counts[row,list(SAMPLES),:].tolist()
        return counts

    with patch.object(model,'iid_counts',tracked_counts):
        issued = prediction.issue(predictor,current,previous,weather)
    if any(set(traces[sid]) != set(SAMPLES) for sid in selected_ids):
        raise AssertionError('Missing a displayed integration trajectory')
    return issued,traces,sample_counts


def fetch_map(path):
    url = 'https://api.openstreetmap.org/api/0.6/map.json?bbox='+','.join(map(str,BOUNDS))
    if not path.exists():
        request = Request(url, headers={'User-Agent':'TashuMapStory/1.0 (+https://github.com/sijoon-sung/KT-X-AI-)', 'Accept':'application/json'})
        with urlopen(request,timeout=35) as response:
            data = response.read(24_000_001)
        if len(data)>24_000_000 or not json.loads(data).get('elements'):
            raise ValueError('Invalid bounded map response')
        path.write_bytes(data)
    return url


def main():
    parser = argparse.ArgumentParser()
    for name in ('stream','weights','weather','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    state_path = args.stream / f'state-{STAMP}.json'
    current_path = args.stream / f'poll-{STAMP}.json'
    previous_path = args.stream / f'poll-{STAMP-600}.json'
    forecast_path = args.stream / f'forecast-{STAMP}.npz'
    if sha(state_path)!=STATE_HASH or sha(args.weights/'model-lock.json')!=LOCK:
        raise ValueError('Recorded state or model digest mismatch')
    state = batch_operations.read(state_path)
    with np.load(forecast_path,allow_pickle=False) as archived:
        metadata = json.loads(str(archived['metadata_json']))
        if metadata['snapshot_sha256']!=sha(current_path) or metadata['previous_sha256']!=sha(previous_path):
            raise ValueError('Recorded input digest mismatch')
        issue_seconds = float(archived['issued_at_unix_seconds'])
        departure = max(STAMP+15, math.ceil(issue_seconds+1))
        plan = batch_operations.optimize(state,dict(deadline_minutes=15,capacity=20,target=6,budget=20,
            stop_setup_seconds=60,handling_seconds=20,minimum_transfer=3),CachedRoads(args.stream/'h27-replay-roads'),departure)
        if plan['status']!='plan' or plan['geometry_status']!='ok':
            raise ValueError('The fixed 15-minute replay has no cached road plan')
        selected = visual_stations(state,plan)
        model.configure()
        predictor = model.Predictor(args.weights,'cuda')
        current = prediction.read_snapshot(current_path,predictor.ids)
        previous = prediction.read_snapshot(previous_path,predictor.ids)
        weather = forecast_transfer.weather_for(args.weather,'daejeon',np.array([STAMP]))[0]
        replay,traces,counts = trace_replay(predictor,current,previous,weather,{s['id'] for s in selected})
        if not np.array_equal(predictor.ids[replay['nodes']],archived['station_ids']) or not np.array_equal(replay['draws']['recent_cfm'],archived['recent_cfm']):
            raise AssertionError('Replayed inventory samples differ from the recorded output')
        archived_pairs = int(archived['recent_cfm'].size)

    node_lookup = {str(s):i for i,s in enumerate(predictor.ids)}
    network = []
    for i,first in enumerate(selected):
        for second in selected[i+1:]:
            a,b = node_lookup[first['id']],node_lookup[second['id']]
            weight = max(float(predictor.graph[a,b]),float(predictor.graph[b,a]))
            if weight>0:
                network.append((first['id'],second['id'],weight))
    network.sort(key=lambda edge:(-edge[2],edge[0],edge[1]))
    rows = {s['id']:s for s in state['stations']}
    for stop in plan['route']['stops']:
        station=rows[stop['station_id']]
        stop.update(name=station['name'],lat=station['lat'],lon=station['lon'])
    plan['depot'] = {k:rows[plan['parameters']['depot_id']][k] for k in ('id','name','lat','lon')}
    stations = []
    for s in selected:
        station = {k:s[k] for k in ('id','name','lat','lon','current','forecast')}
        station.update(trajectory=[traces[s['id']][j] for j in SAMPLES],counts=counts[s['id']],stock_samples=[s['samples'][j] for j in SAMPLES])
        stations.append(station)
    first_work_end = max(s['completion_minutes'] for s in plan['route']['stops'])
    def clock(t):return datetime.fromtimestamp(t,KST).isoformat(timespec='milliseconds')
    map_file = args.out/'map.osm.json'
    map_url = fetch_map(map_file)
    timeline = dict(observation=clock(STAMP),forecast_issued=clock(issue_seconds),departure=clock(departure),
                    cutoff=clock(STAMP+570),target=clock(STAMP+600),work_finished=clock(departure+first_work_end*60),
                    returned=clock(departure+plan['route']['return_minutes']*60),return_deadline=clock(departure+15*60))
    summary = dict(source_version=state['source_version'],state_sha256=STATE_HASH,model_lock_sha256=LOCK,
        archived_forecast_sha256=sha(forecast_path),exact_inventory_sample_replay=True,compared_inventory_samples=archived_pairs,
        displayed_stations=len(stations),historical_graph_edges_displayed=min(24,len(network)),scenario_indices=list(SAMPLES),
        timeline_kst=timeline,displayed_batch=dict(bikes=plan['moved_bikes'],distance_km=plan['route']['distance_meters']/1000,
        visits=[{k:s[k] for k in ('name','action','bikes','eta_minutes','completion_minutes')} for s in plan['route']['stops']],
        elapsed_minutes=plan['route']['return_minutes'],expected_setup_seconds_per_stop=60,expected_handling_seconds_per_bike=20,
        minimum_bikes_per_stop=3,return_window_minutes=15,certificate=plan['certificate']),
        base_map=dict(source=map_url,sha256=sha(map_file),attribution='© OpenStreetMap contributors'),inventory_api_requests=0)
    write(args.out/'summary.json',summary)
    write(args.out/'story.private.json',dict(summary=summary,bounds=BOUNDS,stations=stations,edges=network[:24],plan=plan))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
