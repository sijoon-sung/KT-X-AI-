"""Predict an in-stock station reaching zero during the next hour.

Only reconstructed historical observations are used for fitting. Ambiguous
historical labels are excluded; recent API snapshots are evaluation only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from direct_demand_forecast import DEV, load_data


class StockoutModel(nn.Module):
    def __init__(self, n_stations):
        super().__init__()
        self.station = nn.Embedding(n_stations, 24)
        self.hour = nn.Embedding(24, 8)
        self.weekday = nn.Embedding(7, 4)
        self.month = nn.Embedding(12, 4)
        self.head = nn.Sequential(nn.Linear(41, 64), nn.GELU(),
                                  nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, station, hour, weekday, month, inventory):
        x = torch.cat((self.station(station), self.hour(hour), self.weekday(weekday),
                       self.month(month), torch.log1p(inventory.clamp_min(0)).unsqueeze(-1)), -1)
        return self.head(x).squeeze(-1)


def auc(scores, labels):
    import pandas as pd
    labels = np.asarray(labels, dtype=bool)
    p, n = int(labels.sum()), int((~labels).sum())
    if not p or not n:
        return None
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    return float((ranks[labels].sum() - p * (p + 1) / 2) / (p * n))


def known_mask(data, t):
    exposure = data['exposure'][t]
    upper = data['exposure_ub'][t]
    return (data['active'][t] & data['station_ok'] &
            (data['inventory'][t] > 0) &
            ((upper < 0.999) | (exposure >= 0.999)))


def train(data, cutoff, steps, batch, lr):
    torch.manual_seed(0)
    n_stations = len(data['stations'])
    model = StockoutModel(n_stations).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    inv = torch.as_tensor(data['inventory'], device=DEV, dtype=torch.float32)
    lower = torch.as_tensor(data['exposure'], device=DEV, dtype=torch.float32)
    upper = torch.as_tensor(data['exposure_ub'], device=DEV, dtype=torch.float32)
    active = torch.as_tensor(data['active'], device=DEV)
    station_ok = torch.as_tensor(data['station_ok'], device=DEV)
    cal = {k: torch.as_tensor(data[k], device=DEV, dtype=torch.long)
           for k in ('hour', 'weekday', 'month')}
    pool = np.flatnonzero(data['valid'] & (data['bin_start'] < np.datetime64(cutoff)) &
                          (data['hour'] >= 7) & (data['hour'] <= 22))
    if not len(pool):
        raise ValueError('No historical training hours')
    pool = torch.as_tensor(pool, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(0)
    for step in range(1, steps + 1):
        t = pool[torch.randint(len(pool), (batch,), device=DEV, generator=gen)]
        s = torch.randint(n_stations, (batch,), device=DEV, generator=gen)
        keep = active[t, s] & station_ok[s] & (inv[t, s] > 0) & (
            (upper[t, s] < 0.999) | (lower[t, s] >= 0.999))
        t, s = t[keep], s[keep]
        label = (upper[t, s] < 0.999).float()
        logit = model(s, cal['hour'][t], cal['weekday'][t], cal['month'][t], inv[t, s])
        loss = nn.functional.binary_cross_entropy_with_logits(logit, label)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            print(f'step={step} loss={loss.item():.4f} labeled={len(t)} rate={label.mean().item():.4f}', flush=True)
    return model


@torch.no_grad()
def evaluate(model, data, start, end):
    times = np.flatnonzero(data['valid'] &
                           (data['bin_start'] >= np.datetime64(start)) &
                           (data['bin_start'] < np.datetime64(end)) &
                           (data['hour'] >= 7) & (data['hour'] <= 22))
    # A fixed random subset keeps memory bounded and reproducible.
    rng = np.random.default_rng(42)
    times = rng.choice(times, size=min(500, len(times)), replace=False)
    scores, baseline, labels = [], [], []
    model.eval()
    for t in times:
        mask = known_mask(data, t)
        s_np = np.flatnonzero(mask)
        if not len(s_np):
            continue
        s = torch.as_tensor(s_np, device=DEV)
        inv = torch.as_tensor(data['inventory'][t, s_np], device=DEV, dtype=torch.float32)
        logits = model(s, torch.full_like(s, int(data['hour'][t])),
                       torch.full_like(s, int(data['weekday'][t])),
                       torch.full_like(s, int(data['month'][t])), inv)
        scores.append(logits.cpu().numpy())
        baseline.append(-data['inventory'][t, s_np])
        labels.append(data['exposure_ub'][t, s_np] < 0.999)
    scores, baseline, labels = map(np.concatenate, (scores, baseline, labels))
    return {'sampled_hours': len(times), 'cells': len(labels), 'positive_rate': float(labels.mean()),
            'stockout_auc': auc(scores, labels), 'stock_only_auc': auc(baseline, labels)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--out', default='data/stockout_head.pt')
    ap.add_argument('--report', default='data/stockout_head_evaluation.json')
    ap.add_argument('--train-before', default='2026-02-01')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=65536)
    ap.add_argument('--lr', type=float, default=0.003)
    args = ap.parse_args()
    data = load_data(args.data)
    # Bike-trip reconstruction supplies lower/upper inventory estimates.
    data['inventory'] = (data['inv_start'].astype(np.float32) +
                         data['inv_ub'].astype(np.float32)) / 2
    model = train(data, args.train_before, args.steps, args.batch, args.lr)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'stations': data['stations'].astype(str).tolist(),
                'train_before': args.train_before, 'inventory_input': 'mean_of_reconstructed_bounds'}, args.out)
    report = {'train_before': args.train_before,
              'historical_label': 'exposure_ub<1 positive; exposure==1 negative; ambiguous excluded',
              'inventory_input': 'mean_of_reconstructed_bounds',
              'february_2026': evaluate(model, data, '2026-02-01', '2026-03-01'),
              'march_2026': evaluate(model, data, '2026-03-01', '2026-04-01')}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
