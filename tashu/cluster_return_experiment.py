"""Held-out descriptive aggregation test for hourly returns.

The cluster map and all forecast rates use only historical trips before the
cutoff. Smoothed means are linear, so direct cluster rates equal summed station
rates by construction. This test measures aggregation and grouping effects,
not the advantage of training a separate nonlinear cluster model.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from direct_demand_forecast import calendar


NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def kmeans(x, k, seed=0, iterations=60):
    rng = np.random.default_rng(seed)
    centers = np.empty((k, x.shape[1]), np.float64)
    centers[0] = x[rng.integers(len(x))]
    dist = ((x - centers[0]) ** 2).sum(1)
    for j in range(1, k):
        p = dist / max(dist.sum(), 1e-12)
        centers[j] = x[rng.choice(len(x), p=p)]
        dist = np.minimum(dist, ((x - centers[j]) ** 2).sum(1))
    for _ in range(iterations):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2)
        labels = d.argmin(1)
        updated = centers.copy()
        for j in range(k):
            if np.any(labels == j):
                updated[j] = x[labels == j].mean(0)
            else:
                updated[j] = x[rng.integers(len(x))]
        if np.allclose(centers, updated, atol=1e-5):
            break
        centers = updated
    return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)


def station_features(y, train, hour, weekday, lat, lon):
    # One value for each weekday/weekend hour; shape and overall volume matter.
    shape = []
    for is_weekend in (False, True):
        for h in range(24):
            rows = train & ((weekday >= 5) == is_weekend) & (hour == h)
            shape.append(y[rows].mean(0))
    profile = np.stack(shape, axis=1)
    volume = profile.mean(1)
    relative = np.log1p(profile / np.maximum(volume[:, None], 0.05))
    relative = (relative - relative.mean(0)) / np.maximum(relative.std(0), 0.2)
    relative /= np.sqrt(relative.shape[1])
    geo = np.column_stack(((lat - np.mean(lat)) * 111,
                           (lon - np.mean(lon)) * 111 * np.cos(np.deg2rad(np.mean(lat)))))
    geo /= max(np.std(geo, axis=0).mean(), 1)
    return np.column_stack((relative, 0.65 * np.log1p(volume), 0.9 * geo))


def aggregate(y, labels, k):
    out = np.zeros((len(y), k), np.float32)
    for j in range(k):
        out[:, j] = y[:, labels == j].sum(1)
    return out


def predict_rates(y, train, hour, weekday, alpha=12):
    """Smoothed hour-of-week rates, shared by stations and clusters."""
    n = y.shape[1]
    rate = np.zeros((7, 24, n), np.float32)
    for h in range(24):
        h_rows = train & (hour == h)
        hourly = y[h_rows].mean(0) if h_rows.any() else np.zeros(n)
        for d in range(7):
            rows = h_rows & (weekday == d)
            count = int(rows.sum())
            rate[d, h] = (y[rows].sum(0) + alpha * hourly) / (count + alpha)
    return rate[weekday, hour]


def score(y, pred, mask):
    actual, forecast = y[mask], pred[mask]
    return {'cells': int(actual.size), 'observed': float(actual.sum()),
            'predicted': float(forecast.sum()),
            'predicted_to_observed': float(forecast.sum() / max(actual.sum(), 1)),
            'mae': float(np.abs(actual - forecast).mean()),
            'wape': float(np.abs(actual - forecast).sum() / max(actual.sum(), 1))}


def evaluate(y, station_pred, train, test, hour, weekday, features, k, seed):
    labels = kmeans(features, k, seed=seed)
    if len(np.unique(labels)) != k:
        raise RuntimeError('K-means yielded empty clusters')
    shuffled = np.random.default_rng(seed + 1000).permutation(labels)
    result = {}
    for name, assignment in [('profile_geo', labels), ('random', shuffled)]:
        observed = aggregate(y, assignment, k)
        direct = predict_rates(observed, train, hour, weekday)
        summed = aggregate(station_pred, assignment, k)
        out = {'cluster_sizes': np.bincount(assignment, minlength=k).tolist()}
        for period, mask in test.items():
            out[period] = {'direct_cluster': score(observed, direct, mask),
                           'summed_station': score(observed, summed, mask)}
        result[name] = out
    return labels, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--cutoff', default='2026-02-01')
    ap.add_argument('--train-start', default='2025-08-01')
    ap.add_argument('--clusters', type=int, nargs='+', default=[25, 50, 100])
    ap.add_argument('--out', default='data/cluster_return_experiment.json')
    ap.add_argument('--assignments', default='data/cluster_return_assignments.npz')
    args = ap.parse_args()
    with np.load(args.data) as f:
        z = {key: f[key] for key in ('returns', 'active', 'valid', 'closed',
                                     'bin_start', 'stations', 'lat', 'lon')}
    hour, weekday, _ = calendar(z['bin_start'])
    valid = z['valid'] & ~z['closed']
    train = valid & (z['bin_start'] >= np.datetime64(args.train_start)) & (z['bin_start'] < np.datetime64(args.cutoff))
    if train.sum() < 100:
        raise ValueError('Insufficient valid training hours')
    selected = (z['active'][train].mean(0) >= 0.95) & ~np.isin(z['stations'], NON_CUSTOMER)
    y = z['returns'][:, selected].astype(np.float32)
    features = station_features(y, train, hour, weekday, z['lat'][selected], z['lon'][selected])
    station_pred = predict_rates(y, train, hour, weekday)
    test = {'february_2026': valid & (z['bin_start'] >= np.datetime64('2026-02-01')) &
            (z['bin_start'] < np.datetime64('2026-03-01')),
            'march_2026': valid & (z['bin_start'] >= np.datetime64('2026-03-01')) &
            (z['bin_start'] < np.datetime64('2026-04-01'))}
    report = {'train_start': args.train_start, 'train_before': args.cutoff,
              'target': 'observed returns per hour', 'selected_stations': int(selected.sum()),
              'selection': 'at least 95% active during training; exclude non-customer stations',
              'training_inputs': 'historical trip returns, station coordinates, calendar; no API stock',
              'baseline': 'smoothed hour-of-week mean, alpha=12',
              'interpretation': 'direct cluster and summed station rates are identical by linearity; compare grouping effects only',
              'station': {p: score(y, station_pred, m) for p, m in test.items()},
              'clusters': {}}
    assignments = {'stations': z['stations'][selected]}
    for k in args.clusters:
        labels, scores = evaluate(y, station_pred, train, test, hour, weekday, features, k, seed=42 + k)
        assignments[f'k{k}'] = labels
        report['clusters'][str(k)] = scores
        print(f'k={k} ' + json.dumps({name: {p: v[p]['direct_cluster']['wape'] for p in test}
                                      for name, v in scores.items()}), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding='utf-8')
    np.savez_compressed(args.assignments, **assignments)
    print(json.dumps({'report': args.out, 'assignments': args.assignments,
                      'selected_stations': int(selected.sum())}), flush=True)


if __name__ == '__main__':
    main()
