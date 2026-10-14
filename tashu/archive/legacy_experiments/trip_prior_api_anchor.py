"""Trip-history demand prior with a current-API stock anchor.

Training inputs: station, calendar, and OD graph derived from historical trips.
Trip-derived exposure affects only rental loss. No historical inventory or API
stock is a model input. At evaluation, current API stock starts a stochastic
one-hour inventory simulation using the predicted rental/return distributions.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from censored_flow_matching import build_directed_transition_matrices
from direct_demand_forecast import DEV, calendar, nb_logpmf
from evaluate_api_stock import auc


NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


class TripPrior(nn.Module):
    def __init__(self, stations, od, d=24, use_graph=True):
        super().__init__()
        self.n = len(stations)
        self.use_graph = use_graph
        pf, pb = build_directed_transition_matrices(od, k=15)
        self.register_buffer('pf', pf)
        self.register_buffer('pb', pb)
        self.station = nn.Embedding(self.n, d)
        self.forward_graph = nn.Linear(d, d)
        self.backward_graph = nn.Linear(d, d)
        self.hour = nn.Embedding(24, 8)
        self.weekday = nn.Embedding(7, 4)
        self.month = nn.Embedding(12, 4)
        self.head = nn.Sequential(nn.Linear(d + 16, 64), nn.GELU(),
                                  nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.constant_(self.head[-1].bias, -0.5)
        self.log_r = nn.Parameter(torch.log(torch.tensor([2.0, 2.0])))

    def forward(self, station, hour, weekday, month):
        h = self.station.weight
        if self.use_graph:
            h = F.gelu(h + self.forward_graph(self.pf @ h) +
                       self.backward_graph(self.pb @ h))
        else:
            h = F.gelu(h)
        x = torch.cat((h[station], self.hour(hour), self.weekday(weekday), self.month(month)), -1)
        return torch.exp(self.head(x).clamp(-10, 7))


def historical(path, train_od):
    with np.load(path) as f:
        z = {k: f[k] for k in ('rentals', 'returns', 'exposure', 'exposure_ub',
                              'active', 'valid', 'closed', 'bin_start', 'stations')}
    z['hour'], z['weekday'], z['month'] = calendar(z['bin_start'])
    z['station_ok'] = ~np.isin(z['stations'].astype(str), NON_CUSTOMER)
    z['valid'] &= ~z['closed']
    z['od'] = np.load(train_od)
    if z['od'].shape != (len(z['stations']), len(z['stations'])):
        raise ValueError('OD station count mismatch')
    return z


def fit(z, cutoff, steps=800, batch=65536, use_graph=True, seed=0):
    torch.manual_seed(seed)
    model = TripPrior(z['stations'], z['od'], use_graph=use_graph).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    pool = np.flatnonzero(z['valid'] & (z['bin_start'] < np.datetime64(cutoff)))
    if not len(pool):
        raise ValueError('No training hours before cutoff')
    pool = torch.as_tensor(pool, device=DEV)
    fields = {k: torch.as_tensor(z[k], device=DEV) for k in
              ('rentals', 'returns', 'exposure', 'exposure_ub', 'active',
               'station_ok', 'hour', 'weekday', 'month')}
    generator = torch.Generator(device=DEV).manual_seed(seed)
    for step in range(1, steps + 1):
        t = pool[torch.randint(len(pool), (batch,), device=DEV, generator=generator)]
        s = torch.randint(model.n, (batch,), device=DEV, generator=generator)
        keep = fields['active'][t, s] & fields['station_ok'][s]
        t, s = t[keep], s[keep]
        y_r = fields['rentals'][t, s].float()
        y_b = fields['returns'][t, s].float()
        lower = fields['exposure'][t, s]
        upper = fields['exposure_ub'][t, s]
        e = torch.where((y_r > 0) & (lower < 0.1), torch.maximum(lower, upper), lower)
        weight = torch.where(lower >= 0.999, 1., torch.where(e >= 0.5, 0.2, 0.))
        mu = model(s, fields['hour'][t], fields['weekday'][t], fields['month'][t])
        r = model.log_r.exp().clamp(0.1, 100)
        rent_ll = nb_logpmf(y_r, (mu[:, 0] * e.clamp_min(1 / 60)).clamp_min(1e-6), r[0])
        return_ll = nb_logpmf(y_b, mu[:, 1].clamp_min(1e-6), r[1])
        loss = -(weight * rent_ll).sum() / weight.sum().clamp_min(1) - 0.5 * return_ll.mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            print(f'step={step} loss={loss.item():.4f}', flush=True)
    return model


@torch.no_grad()
def rates(model, timestamp):
    stamp = pd.Timestamp(timestamp)
    n = torch.arange(model.n, device=DEV)
    mu = model(n, torch.full_like(n, stamp.hour), torch.full_like(n, stamp.weekday()),
               torch.full_like(n, stamp.month - 1))
    return mu.cpu().numpy()


@torch.no_grad()
def historical_evaluate(model, z, start, end):
    times = np.flatnonzero(z['valid'] & (z['bin_start'] >= np.datetime64(start)) &
                           (z['bin_start'] < np.datetime64(end)))
    result = {'rentals': {'error': 0., 'actual': 0., 'predicted': 0., 'cells': 0},
              'returns': {'error': 0., 'actual': 0., 'predicted': 0., 'cells': 0}}
    model.eval()
    for t in times:
        mu = rates(model, z['bin_start'][t])
        for j, kind in enumerate(('rentals', 'returns')):
            mask = z['active'][t] & z['station_ok']
            if kind == 'rentals':
                mask &= z['exposure'][t] >= 0.999
            observed = z[kind][t, mask]
            predicted = mu[mask, j]
            out = result[kind]
            out['error'] += float(np.abs(predicted - observed).sum())
            out['actual'] += float(observed.sum())
            out['predicted'] += float(predicted.sum())
            out['cells'] += len(observed)
    for out in result.values():
        out['mae'] = out['error'] / max(out['cells'], 1)
        out['pred_actual_ratio'] = out['predicted'] / max(out['actual'], 1)
    return result


def stockout_probability(mu, stock, r, draws=128, seed=0):
    """Simulate ordered 10-minute rentals/returns from hourly NB intensities."""
    rng = np.random.default_rng(seed)
    # Gamma-Poisson mixtures preserve NB overdispersion for each hour.
    lam_r = rng.gamma(r[0], np.maximum(mu[:, 0], 1e-8) / r[0], size=(draws, len(stock)))
    lam_b = rng.gamma(r[1], np.maximum(mu[:, 1], 1e-8) / r[1], size=(draws, len(stock)))
    available = np.broadcast_to(stock, (draws, len(stock))).astype(np.float32).copy()
    reached_zero = np.zeros_like(available, dtype=bool)
    for _ in range(6):
        # Arrival order within a ten-minute interval is not observed; halve
        # returns at each step as a simple midpoint approximation.
        rentals = rng.poisson(lam_r / 6)
        returns = rng.poisson(lam_b / 6)
        reached_zero |= available + 0.5 * returns - rentals <= 0
        available = np.maximum(available + returns - rentals, 0)
    return reached_zero.mean(axis=0)


def evaluate_api(model, stations, stock_path):
    raw = pd.read_csv(stock_path, usecols=['ts', 'station_id', 'stock'], parse_dates=['ts'])
    panel = raw.pivot(index='ts', columns='station_id', values='stock').sort_index()
    if panel.isna().any().any():
        raise ValueError('Missing API snapshots')
    stock = panel.to_numpy(np.float32)
    times = panel.index.to_numpy(dtype='datetime64[s]')
    api_map = {sid: i for i, sid in enumerate(panel.columns)}
    mi = np.array([i for i, sid in enumerate(stations) if sid in api_map])
    ai = np.array([api_map[stations[i]] for i in mi])
    score, baseline, events, net_errors, hours = [], [], [], [], []
    r = np.exp(model.log_r.detach().cpu().numpy())
    for t in range(0, len(times) - 6, 6):
        gaps = np.diff(times[t:t + 7]).astype('timedelta64[s]').astype(int)
        duration = int((times[t + 6] - times[t]).astype('timedelta64[s]').astype(int))
        stamp = pd.Timestamp(times[t])
        if not (7 <= stamp.hour <= 22 and 3300 <= duration <= 3900 and
                np.all((gaps >= 540) & (gaps <= 900))):
            continue
        current = stock[t, ai]
        in_stock = current > 0
        if not in_stock.any():
            continue
        mu = rates(model, times[t])[mi]
        risk = stockout_probability(mu, current, r, seed=t)
        event = (stock[t + 1:t + 7, ai].min(axis=0) == 0)
        observed_delta = stock[t + 6, ai] - current
        predicted_delta = mu[:, 1] - mu[:, 0]
        score.append(risk[in_stock]); baseline.append(-current[in_stock])
        events.append(event[in_stock]); net_errors.append(np.abs(predicted_delta - observed_delta))
        hours.append(str(times[t]))
    y = np.concatenate(events)
    paired = [auc(s, e) - auc(b, e) for s, b, e in zip(score, baseline, events)
              if auc(s, e) is not None]
    return {'windows': len(hours), 'cells': len(y), 'events': int(y.sum()),
            'first_window': hours[0], 'last_window': hours[-1],
            'stockout_auc': auc(np.concatenate(score), y),
            'stock_only_auc': auc(np.concatenate(baseline), y),
            'observed_event_rate': float(y.mean()),
            'mean_predicted_event_probability': float(np.concatenate(score).mean()),
            'brier_score': float(np.mean((np.concatenate(score) - y) ** 2)),
            'paired_hourly_auc_gain': float(np.mean(paired)),
            'hours_improved': int(np.sum(np.asarray(paired) > 0)),
            'net_stock_change_mae': float(np.concatenate(net_errors).mean()),
            'api_used_for_training': False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--train-od', default='data/processed/od_train_before_2026-02-01.npy')
    ap.add_argument('--train-before', default='2026-02-01')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--out', default='data/trip_prior_api_anchor.pt')
    ap.add_argument('--report', default='data/trip_prior_api_anchor_evaluation.json')
    ap.add_argument('--api-stock', default=None)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--no-graph', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    z = historical(args.data, args.train_od)
    if args.eval_only:
        checkpoint = torch.load(args.out, map_location=DEV, weights_only=True)
        if checkpoint['stations'] != z['stations'].astype(str).tolist():
            raise ValueError('Checkpoint station order mismatch')
        model = TripPrior(z['stations'], z['od'],
                          use_graph=checkpoint.get('use_graph', True)).to(DEV)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
    else:
        model = fit(z, args.train_before, args.steps, use_graph=not args.no_graph,
                    seed=args.seed)
        checkpoint = {'state_dict': model.state_dict(), 'stations': z['stations'].astype(str).tolist(),
                      'train_before': args.train_before, 'train_od': args.train_od,
                      'use_graph': not args.no_graph,
                      'seed': args.seed,
                      'model_inputs': 'historical trips, calendar, OD; no inventory'}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.out)
    report = {'train_before': args.train_before,
              'february_2026': historical_evaluate(model, z, '2026-02-01', '2026-03-01'),
              'march_2026': historical_evaluate(model, z, '2026-03-01', '2026-04-01')}
    if args.api_stock:
        report['recent_api'] = evaluate_api(model, z['stations'].astype(str), args.api_stock)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
