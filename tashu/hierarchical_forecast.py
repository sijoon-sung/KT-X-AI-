"""City total -> region share -> station share forecasts from historical trips.

Only recorded rentals and returns are targets. The recent stock API is read
exclusively by evaluate_api(), never during fitting or cluster construction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


def calendar(times):
    day = times.astype('datetime64[D]')
    hour = (times - day).astype('timedelta64[h]').astype(np.int64)
    weekday = (day.astype(np.int64) + 3) % 7
    month = times.astype('datetime64[M]').astype(np.int64) % 12
    days = (day - np.datetime64('2025-01-01')).astype('timedelta64[D]').astype(np.float32)
    year_fraction = (day - day.astype('datetime64[Y]')).astype('timedelta64[D]').astype(np.float32) / 365.25
    trend = np.clip(days / 365.25, -1.5, 2.0)
    cyc = np.stack([np.sin(2 * np.pi * year_fraction), np.cos(2 * np.pi * year_fraction), trend], 1)
    return hour, weekday, month, cyc


def nb_logpmf(y, mu, r):
    mu = mu.clamp_min(1e-5)
    r = r.clamp(.1, 1e4)
    return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
            + r * (torch.log(r) - torch.log(r + mu))
            + y * (torch.log(mu) - torch.log(r + mu)))


class HierarchicalRates(nn.Module):
    def __init__(self, clusters):
        super().__init__()
        self.clusters = clusters
        self.hour = nn.Embedding(24, 8)
        self.weekday = nn.Embedding(7, 4)
        self.month = nn.Embedding(12, 4)
        self.net = nn.Sequential(nn.Linear(19, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU())
        self.total = nn.Linear(64, 2)
        self.share = nn.Linear(64, 2 * clusters)
        self.log_r = nn.Parameter(torch.log(torch.tensor([20., 20.])))
        nn.init.constant_(self.total.bias, 6.)

    def forward(self, hour, weekday, month, cyc):
        x = torch.cat((self.hour(hour), self.weekday(weekday),
                       self.month(month), cyc), -1)
        h = self.net(x)
        total = torch.exp(self.total(h).clamp(-2, 10))
        share = self.share(h).reshape(-1, 2, self.clusters).softmax(-1)
        return total, share


def load_history(data_path, region_path):
    with np.load(data_path) as z:
        z_stations = z['stations'].astype(str)
        times = z['bin_start']
        valid = z['valid'] & ~z['closed']
        rentals, returns = z['rentals'], z['returns']
    with np.load(region_path) as f:
        stations = f['stations'].astype(str)
        labels = f['labels'].astype(np.int32)
        cutoff = str(f['train_before'])
    lookup = {s: i for i, s in enumerate(z_stations)}
    if any(s not in lookup for s in stations):
        raise ValueError('Region stations differ from processed trip data')
    idx = np.array([lookup[s] for s in stations])
    y_station = np.stack([rentals[:, idx], returns[:, idx]], 1).astype(np.float32)
    k = int(labels.max()) + 1
    y_cluster = np.zeros((len(times), 2, k), np.float32)
    for c in range(k):
        y_cluster[:, :, c] = y_station[:, :, labels == c].sum(-1)
    return {'times': times, 'valid': valid, 'stations': stations, 'labels': labels,
            'cutoff': cutoff, 'station': y_station, 'cluster': y_cluster,
            'total': y_cluster.sum(-1), 'calendar': calendar(times)}


def station_weights(y_station, labels, train, hour, weekday, alpha=20.):
    """Smoothed within-region shares for each hour and weekend flag."""
    n = len(labels)
    k = labels.max() + 1
    weights = np.zeros((2, 2, 24, n), np.float32)
    for kind in range(2):
        for c in range(k):
            ids = np.flatnonzero(labels == c)
            base = y_station[train][:, kind, ids].sum(0) + 1.
            base = base / base.sum()
            for w in range(2):
                for h in range(24):
                    rows = train & ((weekday >= 5) == bool(w)) & (hour == h)
                    counts = y_station[rows][:, kind, ids].sum(0)
                    weights[kind, w, h, ids] = (counts + alpha * base) / (counts.sum() + alpha)
    return weights


def historical_baseline(y_cluster, train, hour, weekday, alpha=12.):
    k = y_cluster.shape[-1]
    rate = np.zeros((7, 24, 2, k), np.float32)
    for h in range(24):
        hours = train & (hour == h)
        prior = y_cluster[hours].mean(0) if hours.any() else np.zeros((2, k))
        for d in range(7):
            rows = hours & (weekday == d)
            rate[d, h] = (y_cluster[rows].sum(0) + alpha * prior) / (rows.sum() + alpha)
    return rate


def torch_fields(history, device):
    hour, weekday, month, cyc = history['calendar']
    return (torch.as_tensor(hour, device=device, dtype=torch.long),
            torch.as_tensor(weekday, device=device, dtype=torch.long),
            torch.as_tensor(month, device=device, dtype=torch.long),
            torch.as_tensor(cyc, device=device, dtype=torch.float32))


def loss_at(model, fields, y_total, y_cluster, ix):
    total, share = model(*(field[ix] for field in fields))
    yt, yc = y_total[ix], y_cluster[ix]
    r = model.log_r.exp()
    total_loss = -nb_logpmf(yt, total, r).mean()
    fraction = yc / yt.clamp_min(1).unsqueeze(-1)
    share_loss = -(fraction * share.clamp_min(1e-8).log()).sum(-1).mean()
    return total_loss + share_loss, total_loss, share_loss


def fit_one(history, train_mask, epochs, seed, device, init=None, learning_rate=.004):
    torch.manual_seed(seed)
    k = history['cluster'].shape[-1]
    model = HierarchicalRates(k).to(device)
    if init is not None:
        model.load_state_dict(init)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    fields = torch_fields(history, device)
    yt = torch.as_tensor(history['total'], device=device)
    yc = torch.as_tensor(history['cluster'], device=device)
    indices = torch.as_tensor(np.flatnonzero(train_mask), device=device)
    for epoch in range(epochs):
        order = indices[torch.randperm(len(indices), device=device)]
        model.train()
        for ix in order.split(4096):
            loss, _, _ = loss_at(model, fields, yt, yc, ix)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(f'epoch={epoch+1} loss={loss.item():.4f}', flush=True)
    return model


@torch.no_grad()
def predict(model, history, device, batch=2048):
    model.eval()
    fields = torch_fields(history, device)
    totals, shares = [], []
    for start in range(0, len(history['times']), batch):
        sl = slice(start, start + batch)
        t, s = model(*(field[sl] for field in fields))
        totals.append(t.cpu().numpy()); shares.append(s.cpu().numpy())
    totals = np.concatenate(totals)
    shares = np.concatenate(shares)
    return totals, totals[:, :, None] * shares


def scores(actual, forecast):
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    total = max(float(actual.sum()), 1.)
    return {'observed': float(actual.sum()), 'predicted': float(forecast.sum()),
            'ratio': float(forecast.sum()/total),
            'mae': float(np.abs(actual - forecast).mean()),
            'wape': float(np.abs(actual - forecast).sum()/total)}


def evaluate_historical(history, cluster_pred, base_cluster):
    times, valid = history['times'], history['valid']
    result = {}
    for name, lo, hi in [('february_2026', '2026-02-01', '2026-03-01'),
                         ('march_2026', '2026-03-01', '2026-04-01')]:
        mask = valid & (times >= np.datetime64(lo)) & (times < np.datetime64(hi))
        result[name] = {'valid_hours': int(mask.sum())}
        for kind, label in enumerate(('recorded_rentals', 'returns')):
            obs = history['cluster'][mask, kind]
            pred = cluster_pred[mask, kind]
            base = base_cluster[mask, kind]
            true_total = obs.sum(-1)
            pred_total = pred.sum(-1)
            base_total = base.sum(-1)
            result[name][label] = {
                'city_total': scores(true_total, pred_total),
                'city_baseline': scores(true_total, base_total),
                'region': scores(obs, pred),
                'region_baseline': scores(obs, base),
                'oracle_total_region_share_wape': float(
                    np.abs(obs - pred / pred_total[:, None].clip(min=1e-6) * true_total[:, None]).sum()
                    / max(obs.sum(), 1))}
    return result


def train_pipeline(a):
    torch.set_num_threads(min(4, torch.get_num_threads()))
    history = load_history(a.data, a.regions)
    cutoff = history['cutoff']
    train = history['valid'] & (history['times'] < np.datetime64(cutoff))
    if train.sum() < 1000:
        raise ValueError('Not enough training hours')
    device = torch.device(a.device)
    # Choose training length on January 2026, then fit all pre-February history.
    inner_train = train & (history['times'] < np.datetime64('2026-01-01'))
    validation = train & (history['times'] >= np.datetime64('2026-01-01'))
    if not validation.any():
        raise ValueError('January validation period missing')
    torch.manual_seed(a.seed)
    model = HierarchicalRates(history['cluster'].shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    fields = torch_fields(history, device)
    yt = torch.as_tensor(history['total'], device=device)
    yc = torch.as_tensor(history['cluster'], device=device)
    ids = torch.as_tensor(np.flatnonzero(inner_train), device=device)
    val = torch.as_tensor(np.flatnonzero(validation), device=device)
    best, best_epoch, best_state = float('inf'), 0, None
    patience = 0
    for epoch in range(1, a.max_epochs + 1):
        model.train()
        order = ids[torch.randperm(len(ids), device=device)]
        for ix in order.split(4096):
            loss, _, _ = loss_at(model, fields, yt, yc, ix)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = np.mean([float(loss_at(model, fields, yt, yc, ix)[0].item())
                                for ix in val.split(4096)])
        if val_loss < best - .001:
            best, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch == 1 or epoch % 20 == 0 or patience >= a.patience:
            print(f'select epoch={epoch} jan_loss={val_loss:.4f} best={best:.4f}', flush=True)
        if patience >= a.patience:
            break
    # Refit from same initialization for the selected number of epochs.
    torch.manual_seed(a.seed)
    final = fit_one(history, train, best_epoch, a.seed, device, learning_rate=a.lr)
    total_pred, cluster_pred = predict(final, history, device)
    hour, weekday, _, _ = history['calendar']
    base_rates = historical_baseline(history['cluster'], train, hour, weekday)
    base_cluster = base_rates[weekday, hour]
    weights = station_weights(history['station'], history['labels'], train, hour, weekday)
    report = {'model': 'NB city total + multinomial region shares + smoothed station shares',
              'target': 'recorded rentals and returns; no claim of latent unmet demand',
              'train_before': cutoff, 'train_hours': int(train.sum()),
              'selection_validation': '2026-01; final Feb and Mar excluded',
              'best_epoch': best_epoch, 'january_validation_loss': best,
              'api_used_for_training': False,
              'historical': evaluate_historical(history, cluster_pred, base_cluster)}
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': final.cpu().state_dict(), 'stations': history['stations'].tolist(),
                'labels': history['labels'].tolist(), 'cutoff': cutoff,
                'station_weights': weights, 'log_r': final.log_r.detach().numpy()}, a.model)
    Path(a.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def auc(scores_, labels):
    labels = np.asarray(labels, bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return None
    ranks = pd.Series(scores_).rank(method='average').to_numpy()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def evaluate_api(a):
    """Held-out stockout check using snapshots; no parameter updates."""
    checkpoint = torch.load(a.model, map_location='cpu', weights_only=False)
    with np.load(a.regions) as f:
        if checkpoint['stations'] != f['stations'].astype(str).tolist() or checkpoint['labels'] != f['labels'].tolist():
            raise ValueError('Model/region station order mismatch')
    model = HierarchicalRates(max(checkpoint['labels']) + 1)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    raw = pd.read_csv(a.stock, usecols=['ts', 'station_id', 'stock'], parse_dates=['ts'])
    if raw.duplicated(['ts', 'station_id']).any():
        raise ValueError('Duplicate station snapshots')
    panel = raw.pivot(index='ts', columns='station_id', values='stock').sort_index()
    stations = checkpoint['stations']
    matched = [(i, panel.columns.get_loc(s)) for i, s in enumerate(stations) if s in panel]
    mi = np.array([x[0] for x in matched]); ai = np.array([x[1] for x in matched])
    stock = panel.to_numpy(np.float32)
    times = panel.index.to_numpy(dtype='datetime64[s]')
    weights = np.asarray(checkpoint['station_weights'])
    labels = np.asarray(checkpoint['labels'])
    rng = np.random.default_rng(a.seed)
    r = np.exp(np.asarray(checkpoint['log_r']))
    risk_rows, event_rows, baseline_rows, trend_rows, net_rows = [], [], [], [], []
    starts = []
    for t in range(6, len(times) - 6, 6):
        gaps = np.diff(times[t:t+7]).astype('timedelta64[s]').astype(int)
        past = np.diff(times[t-6:t+1]).astype('timedelta64[s]').astype(int)
        if not (np.all((gaps >= 540) & (gaps <= 900)) and
                np.all((past >= 540) & (past <= 900)) and
                3300 <= (times[t+6]-times[t]).astype('timedelta64[s]').astype(int) <= 3900):
            continue
        now = stock[t, ai]
        present = np.isfinite(now) & (now > 0)
        if not present.any():
            continue
        stamp = np.array([times[t]])
        h, d, m, cyc = calendar(stamp)
        with torch.no_grad():
            total, share = model(torch.tensor(h), torch.tensor(d), torch.tensor(m), torch.tensor(cyc))
        region_rate = (total[0, :, None] * share[0]).numpy()
        rate = np.empty((2, len(stations)), np.float32)
        for kind in range(2):
            rate[kind] = region_rate[kind, labels] * weights[kind, int(d[0] >= 5), h[0]]
        mu_rent, mu_return = rate[:, mi]
        # Shared gamma shocks retain city-wide count uncertainty.
        gamma_r = rng.gamma(r[0], 1/r[0], size=(a.draws, 1))
        gamma_b = rng.gamma(r[1], 1/r[1], size=(a.draws, 1))
        inventory = np.broadcast_to(now, (a.draws, len(now))).copy()
        hit = np.zeros_like(inventory, bool)
        for _ in range(6):
            rent = rng.poisson(gamma_r * mu_rent[None] / 6)
            ret = rng.poisson(gamma_b * mu_return[None] / 6)
            hit |= inventory + .5 * ret - rent <= 0
            inventory = np.maximum(inventory + ret - rent, 0)
        future = stock[t+1:t+7, ai]
        event = np.nanmin(future, axis=0) == 0
        eligible = present & np.isfinite(future).all(0)
        if not eligible.any():
            continue
        risk_rows.append(hit.mean(0)[eligible])
        event_rows.append(event[eligible])
        baseline_rows.append(-now[eligible])
        past_stock = stock[t-6, ai]
        trend_rows.append(-(now + now - past_stock)[eligible])
        net_rows.append(np.abs((mu_return - mu_rent)[eligible] - (stock[t+6, ai] - now)[eligible]))
        starts.append(str(times[t]))
    if not starts:
        raise ValueError('No complete API one-hour windows')
    score = np.concatenate(risk_rows); event = np.concatenate(event_rows)
    result = {'source': a.stock, 'first_window': starts[0], 'last_window': starts[-1],
              'hour_windows': len(starts), 'cells': len(event), 'events': int(event.sum()),
              'event_rate': float(event.mean()), 'mean_predicted_risk': float(score.mean()),
              'risk_auc': auc(score, event), 'stock_only_auc': auc(np.concatenate(baseline_rows), event),
              'stock_trend_auc': auc(np.concatenate(trend_rows), event),
              'risk_brier': float(np.mean((score-event)**2)),
              'net_stock_change_mae': float(np.concatenate(net_rows).mean()),
              'api_used_for_training': False}
    Path(a.api_report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.api_report).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['fit', 'evaluate-api'])
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--regions', default='data/regions.npz')
    ap.add_argument('--model', default='data/hierarchical_rates.pt')
    ap.add_argument('--report', default='data/hierarchical_rates_historical.json')
    ap.add_argument('--stock', default='C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    ap.add_argument('--api-report', default='data/hierarchical_rates_api.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--max-epochs', type=int, default=160)
    ap.add_argument('--patience', type=int, default=20)
    ap.add_argument('--lr', type=float, default=.004)
    ap.add_argument('--seed', type=int, default=19)
    ap.add_argument('--draws', type=int, default=128)
    a = ap.parse_args()
    if a.command == 'fit':
        train_pipeline(a)
    else:
        evaluate_api(a)


if __name__ == '__main__':
    main()
