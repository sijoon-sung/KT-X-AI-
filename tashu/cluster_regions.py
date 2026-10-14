"""Build geographic/OD station regions using only pre-cutoff trips.

Regions are spatially local, connected through nearby-station edges, and have
a minimum station count. Neither test-period trips nor API stock are read.
"""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np


EXCLUDE = ('ST0001', 'ST1220', 'ST1198')


def coordinates_km(lat, lon):
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError('Selected station coordinates contain missing values')
    return np.column_stack(((lat - lat.mean()) * 111.0,
                            (lon - lon.mean()) * 111.0 * np.cos(np.deg2rad(lat.mean()))))


def nearby_graph(xy, od, neighbors=12):
    n = len(xy)
    d2 = ((xy[:, None] - xy[None, :]) ** 2).sum(2)
    np.fill_diagonal(d2, np.inf)
    count = od + od.T
    scale = max(float(np.quantile(count[count > 0], .95)) if np.any(count > 0) else 1., 1.)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        for j in np.argpartition(d2[i], min(neighbors, n - 1))[:neighbors]:
            j = int(j)
            distance = float(np.sqrt(d2[i, j]))
            # OD affects assignment only between geographically nearby stations.
            flow = np.log1p(float(count[i, j])) / np.log1p(scale)
            weight = np.exp(-distance / .75) * (1. + .6 * flow)
            graph.add_edge(i, j, weight=float(weight))
    return graph


def split_large(group, xy, maximum):
    """Split a large region along its principal geographic axis."""
    if len(group) <= maximum:
        return [group]
    p = xy[group]
    _, _, v = np.linalg.svd(p - p.mean(0), full_matrices=False)
    order = np.argsort((p - p.mean(0)) @ v[0])
    mid = len(group) // 2
    return split_large(group[order[:mid]], xy, maximum) + split_large(group[order[mid:]], xy, maximum)


def merge_small(groups, xy, od, minimum):
    groups = [np.asarray(g, dtype=np.int32) for g in groups]
    while True:
        small = [i for i, g in enumerate(groups) if len(g) < minimum]
        if not small:
            break
        i = min(small, key=lambda j: len(groups[j]))
        a = groups[i]
        choices = []
        for j, b in enumerate(groups):
            if j == i:
                continue
            # Prefer nearby regions with a real OD connection.
            gap = float(np.sqrt(((xy[a, None] - xy[None, b]) ** 2).sum(2)).min())
            flow = float(od[np.ix_(a, b)].sum() + od[np.ix_(b, a)].sum())
            cost = gap - .15 * np.log1p(flow / max(len(a) * len(b), 1))
            choices.append((cost, j))
        j = min(choices)[1]
        groups[j] = np.concatenate((groups[j], a))
        groups.pop(i)
    return groups


def build_clusters(data, train_od, cutoff, train_start, min_active, minimum, maximum, seed):
    with np.load(data) as z:
        times = z['bin_start']
        valid = z['valid'] & ~z['closed']
        train = valid & (times >= np.datetime64(train_start)) & (times < np.datetime64(cutoff))
        if train.sum() < 100:
            raise ValueError('Too few valid training hours')
        stations = z['stations'].astype(str)
        selected = (z['active'][train].mean(0) >= min_active) & ~np.isin(stations, EXCLUDE)
        lat, lon = z['lat'][selected], z['lon'][selected]
    od_all = np.load(train_od)
    if od_all.shape != (len(stations), len(stations)):
        raise ValueError('Training OD station order/shape mismatch')
    idx = np.flatnonzero(selected)
    od = od_all[np.ix_(idx, idx)].astype(np.float64)
    xy = coordinates_km(lat, lon)
    graph = nearby_graph(xy, od)
    initial = [np.array(sorted(c), dtype=np.int32) for c in
               nx.community.louvain_communities(graph, weight='weight', resolution=1.0, seed=seed)]
    parts = [g for group in initial for g in split_large(group, xy, maximum)]
    groups = merge_small(parts, xy, od, minimum)
    groups.sort(key=lambda g: (float(xy[g, 0].mean()), float(xy[g, 1].mean())))
    labels = np.empty(len(idx), np.int32)
    for c, group in enumerate(groups):
        labels[group] = c
    sizes = np.bincount(labels)
    if sizes.min() < minimum or len(labels) != sum(len(g) for g in groups):
        raise AssertionError('Invalid cluster partition')
    centroids = np.stack([xy[labels == c].mean(0) for c in range(len(groups))])
    radius = np.sqrt(((xy - centroids[labels]) ** 2).sum(1))
    internal = float(od[labels[:, None] == labels[None, :]].sum() / max(od.sum(), 1))
    report = {'train_start': train_start, 'train_before': cutoff,
              'train_od': str(train_od), 'api_used_for_clustering': False,
              'stations': len(idx), 'clusters': len(groups),
              'size_min': int(sizes.min()), 'size_median': float(np.median(sizes)),
              'size_max': int(sizes.max()), 'sizes': sizes.tolist(),
              'mean_station_to_centroid_km': float(radius.mean()),
              'p90_station_to_centroid_km': float(np.quantile(radius, .9)),
              'within_cluster_od_share': internal}
    return stations[idx], labels, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--train-od', default='data/processed/od_train_before_2026-02-01.npy')
    ap.add_argument('--train-start', default='2025-08-01')
    ap.add_argument('--cutoff', default='2026-02-01')
    ap.add_argument('--min-active', type=float, default=.95)
    ap.add_argument('--minimum', type=int, default=12)
    ap.add_argument('--maximum', type=int, default=60)
    ap.add_argument('--seed', type=int, default=19)
    ap.add_argument('--out', default='data/regions.npz')
    ap.add_argument('--report', default='data/regions_report.json')
    a = ap.parse_args()
    stations, labels, report = build_clusters(a.data, a.train_od, a.cutoff,
                                               a.train_start, a.min_active,
                                               a.minimum, a.maximum, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, stations=stations, labels=labels,
                        train_before=a.cutoff, train_start=a.train_start)
    Path(a.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
