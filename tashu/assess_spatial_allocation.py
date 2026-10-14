"""Isolate the OD graph's value in allocating a known citywide return total.

The target is observed returns, so rental censoring and stock reconstruction do
not affect this experiment. Citywide total is supplied as an oracle solely to
measure spatial allocation; it is not available in a real forecast.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from direct_demand_forecast import DEV
from trip_prior_api_anchor import TripPrior, historical, rates


def load_model(path, data):
    ckpt = torch.load(path, map_location=DEV, weights_only=True)
    if ckpt['stations'] != data['stations'].astype(str).tolist():
        raise ValueError(f'Station order mismatch: {path}')
    model = TripPrior(data['stations'], data['od'],
                      use_graph=ckpt.get('use_graph', True)).to(DEV)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def period(models, data, start, end):
    hours = np.flatnonzero(data['valid'] &
                           (data['bin_start'] >= np.datetime64(start)) &
                           (data['bin_start'] < np.datetime64(end)) &
                           (data['hour'] >= 7) & (data['hour'] <= 22))
    scores = {name: {'cross_entropy': [], 'oracle_total_mae': [], 'top10_mass': []}
              for name in models}
    evaluated_days = []
    for t in hours:
        mask = data['active'][t] & data['station_ok']
        y = data['returns'][t, mask].astype(np.float64)
        total = y.sum()
        if total <= 0 or not len(y):
            continue
        evaluated_days.append(str(data['bin_start'][t].astype('datetime64[D]')))
        for name, model in models.items():
            mu = rates(model, data['bin_start'][t])[mask, 1].astype(np.float64)
            p = mu / mu.sum()
            scores[name]['cross_entropy'].append(float(-(y * np.log(p.clip(1e-12))).sum() / total))
            scores[name]['oracle_total_mae'].append(float(np.abs(total * p - y).mean()))
            k = max(1, int(np.ceil(0.1 * len(y))))
            top = np.argpartition(p, -k)[-k:]
            scores[name]['top10_mass'].append(float(y[top].sum() / total))
    result = {'hours': len(scores[next(iter(models))]['cross_entropy']), 'models': {}}
    for name, values in scores.items():
        result['models'][name] = {metric: float(np.mean(array)) for metric, array in values.items()}
    graph = np.asarray(scores['graph']['cross_entropy'])
    no_graph = np.asarray(scores['no_graph']['cross_entropy'])
    gain = no_graph - graph
    rng = np.random.default_rng(0)
    unique_days = np.unique(evaluated_days)
    daily_gain = np.array([gain[np.asarray(evaluated_days) == day].mean() for day in unique_days])
    bootstrap = rng.choice(daily_gain, size=(3000, len(daily_gain)), replace=True).mean(axis=1)
    result['paired_cross_entropy_gain'] = {'mean': float(gain.mean()),
                                           'hours_graph_better': int((gain > 0).sum()),
                                           'days': len(unique_days),
                                           'day_block_bootstrap_95pct': np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--train-od', default='data/processed/od_train_before_2026-02-01.npy')
    ap.add_argument('--graph', default='data/trip_prior_api_anchor.pt')
    ap.add_argument('--no-graph', default='data/trip_prior_no_graph.pt')
    ap.add_argument('--out', default='data/spatial_allocation_ablation.json')
    args = ap.parse_args()
    data = historical(args.data, args.train_od)
    models = {'graph': load_model(args.graph, data),
              'no_graph': load_model(args.no_graph, data)}
    result = {'question': 'Does the OD graph improve allocation of a known return total?',
              'oracle_total_used_only_for_evaluation': True,
              'february_2026': period(models, data, '2026-02-01', '2026-03-01'),
              'march_2026': period(models, data, '2026-03-01', '2026-04-01')}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
