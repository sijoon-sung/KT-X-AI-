"""Evaluation-only use of recent API stock snapshots for integrated ST-flow."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from direct_demand_forecast import DEV
from evaluate_api_stock import auc
from integrated_st_flow import IntegratedSTFlow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--model', default='data/integrated_st_flow.pt')
    ap.add_argument('--out', default='data/integrated_st_flow_api.json')
    args = ap.parse_args()
    raw = pd.read_csv(args.stock, usecols=['ts', 'station_id', 'stock'], parse_dates=['ts'])
    panel = raw.pivot(index='ts', columns='station_id', values='stock').sort_index()
    if panel.isna().any().any():
        raise ValueError('API stock panel has missing snapshots')
    stock = panel.to_numpy(dtype=np.float32)
    times = panel.index.to_numpy(dtype='datetime64[s]')
    with np.load(args.data) as f:
        od, stations = f['od'], f['stations'].astype(str)
    checkpoint = torch.load(args.model, map_location=DEV, weights_only=True)
    if checkpoint['stations'] != stations.tolist():
        raise ValueError('Station order mismatch')
    lookback = int(checkpoint['lookback'])
    model = IntegratedSTFlow(len(stations), od, lookback).to(DEV)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    api_map = {sid: i for i, sid in enumerate(panel.columns)}
    mi = np.array([i for i, sid in enumerate(stations) if sid in api_map])
    ai = np.array([api_map[stations[i]] for i in mi])
    scores = {'flow': [], 'baseline': [], 'stock_only': []}
    labels, abs_delta, abs_delta_baseline, predicted_total, observed_total = [], [], [], 0., 0.
    starts = []
    for t in range(0, len(times) - 6, 6):
        if t < 6 * (lookback + 1):
            continue
        idx = t + 6 * np.arange(-lookback - 1, 1)
        points = times[np.r_[idx, t + 6]]
        gaps = np.diff(points).astype('timedelta64[s]').astype(int)
        future_gaps = np.diff(times[t:t + 7]).astype('timedelta64[s]').astype(int)
        if not (np.all((gaps >= 3300) & (gaps <= 3900)) and
                np.all((future_gaps >= 540) & (future_gaps <= 900))):
            continue
        timestamp = pd.Timestamp(times[t])
        if not 7 <= timestamp.hour <= 22:
            continue
        current = stock[t, ai]
        in_stock = current > 0
        if not in_stock.any():
            continue
        hist = np.zeros((lookback + 2, len(stations)), dtype=np.float32)
        hist[:, mi] = stock[idx][:, ai]
        hist_times = [pd.Timestamp(v) for v in times[idx[1:-1]]]
        hour = torch.tensor([[v.hour for v in hist_times]], device=DEV)
        weekday = torch.tensor([[v.weekday() for v in hist_times]], device=DEV)
        month = torch.tensor([[v.month - 1 for v in hist_times]], device=DEV)
        with torch.no_grad():
            pred, base = model.predict(torch.as_tensor(hist[None], device=DEV), hour, weekday,
                                       month, torch.tensor([timestamp.hour], device=DEV),
                                       torch.tensor([timestamp.weekday()], device=DEV),
                                       torch.tensor([timestamp.month - 1], device=DEV))
        pred, base = pred[0].cpu().numpy(), base[0].cpu().numpy()
        event = (stock[t + 1:t + 7, ai].min(axis=0) == 0)[in_stock]
        observed = stock[t + 6, ai] - current
        for key, value in (('flow', pred), ('baseline', base)):
            risk = value[mi, 0] - value[mi, 1] - current
            scores[key].append(risk[in_stock])
        scores['stock_only'].append(-current[in_stock])
        labels.append(event)
        flow_delta = pred[mi, 1] - pred[mi, 0]
        base_delta = base[mi, 1] - base[mi, 0]
        abs_delta.append(np.abs(flow_delta - observed))
        abs_delta_baseline.append(np.abs(base_delta - observed))
        predicted_total += float(flow_delta.sum())
        observed_total += float(observed.sum())
        starts.append(str(times[t]))
    if not starts:
        raise ValueError('No complete windows with history and future')
    y = np.concatenate(labels)
    metrics = {key: {'stockout_auc': auc(np.concatenate(value), y)} for key, value in scores.items()}
    metrics['flow']['net_stock_change_mae'] = float(np.concatenate(abs_delta).mean())
    metrics['baseline']['net_stock_change_mae'] = float(np.concatenate(abs_delta_baseline).mean())
    paired = []
    for flow, stock_score, event in zip(scores['flow'], scores['stock_only'], labels):
        a, b = auc(flow, event), auc(stock_score, event)
        if a is not None and b is not None:
            paired.append(a - b)
    result = {'source': str(Path(args.stock)), 'windows': len(starts), 'cells': len(y),
              'stockout_events': int(y.sum()), 'first_window': starts[0], 'last_window': starts[-1],
              'metrics': metrics, 'paired_hourly_auc_gain': float(np.mean(paired)),
              'hours_improved': int(np.sum(np.array(paired) > 0)),
              'predicted_net_total': predicted_total, 'observed_net_total': observed_total,
              'training_use_of_api': False}
    Path(args.out).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
