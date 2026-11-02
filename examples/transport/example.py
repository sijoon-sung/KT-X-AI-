"""Synthetic transport join/routing example. Run calculations on dislab."""
import argparse
from datetime import datetime
from itertools import permutations
import json
from pathlib import Path


def stamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Timezone required.')
    return parsed.timestamp()


def run(fixture):
    issue = stamp(fixture['issue_at'])
    accepted, rejected = [], []
    for event in fixture['events']:
        keys = ['observed_at', 'published_at', 'received_at']
        if event.get('window_end'):
            keys.append('window_end')
        if any(event.get(k) is None or stamp(event[k]) > issue for k in keys):
            rejected.append(dict(id=event['id'], reason='not available at issue time'))
        else:
            accepted.append(event)
    by_key = {(x['entity_id'], x['kind']): x for x in accepted}
    contexts = []
    for station in fixture['stations']:
        context = dict(station_id=station, next_bus_eta_seconds=None,
                       walk_to_bus_seconds=None, previous_day_same_hour_metro_alightings=None,
                       bus_present=False, metro_present=False)
        for edge in fixture['connections']:
            if edge['station_id'] != station:
                continue
            event = by_key.get((edge['entity_id'], edge['kind']))
            if event is None:
                continue
            if edge['kind'] == 'bus_eta':
                assert event['unit'] == 'seconds'
                context.update(next_bus_eta_seconds=event['value'], walk_to_bus_seconds=edge['walk_seconds'], bus_present=True)
            elif edge['kind'] == 'metro_alightings':
                assert event['unit'] == 'persons'
                context.update(previous_day_same_hour_metro_alightings=event['value'], metro_present=True)
        contexts.append(context)
    routing = fixture['routing']
    edges = {(x['from'], x['to']): x for x in routing['edges']}
    speed = by_key.get(('S-A', 'road_speed'))
    assert not speed or speed['unit'] == 'km_per_hour' and speed['value'] > 0
    routes = []
    for order in permutations(routing['drops']):
        path = [routing['depot'], *order, routing['depot']]
        work = (routing['pickups_at_depot'] + sum(routing['drops'].values())) * routing['handling_seconds_per_bike']
        base_time = sum(edges[a, b]['seconds'] for a, b in zip(path, path[1:]))
        traffic_time = sum(edges[a, b]['meters'] / (speed['value'] / 3.6) if (a, b) == ('S', 'A') and speed else edges[a, b]['seconds'] for a, b in zip(path, path[1:]))
        loads = [routing['pickups_at_depot']]
        for node in order:
            loads.append(loads[-1] - routing['drops'][node])
        assert sum(routing['drops'].values()) == routing['pickups_at_depot']
        assert min(loads) >= 0 and max(loads) <= routing['capacity'] and loads[-1] == 0
        routes.append(dict(path=path, meters=sum(edges[a, b]['meters'] for a, b in zip(path, path[1:])),
                           baseline_seconds=base_time + work, traffic_seconds=traffic_time + work,
                           handling_seconds=work, loads=loads,
                           baseline_feasible=base_time + work <= routing['deadline_seconds'],
                           traffic_feasible=traffic_time + work <= routing['deadline_seconds']))
    choose = lambda field, valid: min((r for r in routes if r[valid]), key=lambda r: r[field], default=None)
    return dict(scope=fixture['scope'], issue_at=fixture['issue_at'], candidate_contexts=contexts,
                rejected_events=rejected, accepted_events=[x['id'] for x in accepted],
                routes=routes, baseline_plan=choose('baseline_seconds', 'baseline_feasible'),
                traffic_plan=choose('traffic_seconds', 'traffic_feasible'),
                model_weights_changed=False, current_h13_accepts_these_features=False,
                predicted_demand_change=None, real_operator_action=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--fixture', default=str(Path(__file__).with_name('fixture.json')))
    p.add_argument('--out', required=True)
    a = p.parse_args()
    value = run(json.loads(Path(a.fixture).read_text(encoding='utf-8')))
    target = Path(a.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(dict(output_saved=True, rejected=len(value['rejected_events']),
                          baseline=value['baseline_plan']['path'], traffic=value['traffic_plan']['path'])))
