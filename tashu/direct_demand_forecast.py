"""대여소·달력·시작 재고로 다음 1시간의 대여 및 반납 건수를 직접 예측한다.

학습에는 과거 대여이력만 사용한다. API 재고 스냅샷은 별도 평가에서만 읽는다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from diagnose_forecast import fit_historical_rate
# ?? -> 날씨 안 넣었는데

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def calendar(bin_start):
    days = bin_start.astype('datetime64[D]')
    hour = (bin_start - days).astype('timedelta64[h]').astype(np.int64)
    weekday = (days.astype(np.int64) + 3) % 7
    month = (bin_start.astype('datetime64[M]').astype(np.int64) % 12)
    return hour, weekday, month


def nb_logpmf(y, mu, r):
    return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
            + r * (torch.log(r) - torch.log(r + mu))
            + y * (torch.log(mu) - torch.log(r + mu)))


class DirectDemandModel(nn.Module):
    def __init__(self, n_stations):
        super().__init__()
        self.station = nn.Embedding(n_stations, 24)
        self.hour = nn.Embedding(24, 8)
        self.weekday = nn.Embedding(7, 4)
        self.month = nn.Embedding(12, 4)
        self.station_bias = nn.Embedding(n_stations, 2)
        nn.init.zeros_(self.station_bias.weight)
        self.head = nn.Sequential(nn.Linear(41, 64), nn.GELU(), nn.Linear(64, 64),
                                  nn.GELU(), nn.Linear(64, 2))
        nn.init.constant_(self.head[-1].bias, -0.5)
        self.log_r = nn.Parameter(torch.log(torch.tensor([2.0, 2.0])))

    def forward(self, station, hour, weekday, month, inventory):
        x = torch.cat((self.station(station), self.hour(hour), self.weekday(weekday),
                       self.month(month), torch.log1p(inventory.clamp_min(0)).unsqueeze(-1)), -1)
        log_mu = self.head(x) + self.station_bias(station)
        return torch.exp(log_mu.clamp(-10, 8))


def load_data(path):
    with np.load(path) as f:
        z = {key: f[key] for key in ('rentals', 'returns', 'exposure', 'exposure_ub',
                                     'inv_start', 'inv_ub', 'active', 'valid', 'closed',
                                     'bin_start', 'stations')}
    z['hour'], z['weekday'], z['month'] = calendar(z['bin_start'])
    z['station_ok'] = ~np.isin(z['stations'].astype(str), NON_CUSTOMER)
    z['valid'] = z['valid'] & ~z['closed']
    return z


def train(data, cutoff, steps=800, batch=65536, lr=0.003, partial_weight=0.2):
    np.random.seed(0)
    torch.manual_seed(0)
    t_pool = np.flatnonzero(data['valid'] & (data['bin_start'] < np.datetime64(cutoff)))
    if not len(t_pool):
        raise ValueError('No training hours before cutoff')
    n_stations = data['rentals'].shape[1]
    model = DirectDemandModel(n_stations).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    matrix = {name: torch.as_tensor(data[name], device=DEV, dtype=torch.float32)
              for name in ('rentals', 'returns', 'exposure', 'exposure_ub', 'inv_start')}
    active = torch.as_tensor(data['active'], device=DEV)
    station_ok = torch.as_tensor(data['station_ok'], device=DEV)
    cal = {name: torch.as_tensor(data[name], device=DEV, dtype=torch.long)
           for name in ('hour', 'weekday', 'month')}
    t_pool = torch.as_tensor(t_pool, device=DEV)
    generator = torch.Generator(device=DEV).manual_seed(0)
    for step in range(1, steps + 1):
        t = t_pool[torch.randint(len(t_pool), (batch,), device=DEV, generator=generator)]
        n = torch.randint(n_stations, (batch,), device=DEV, generator=generator)
        keep = active[t, n] & station_ok[n]
        if not keep.any():
            continue
        t, n = t[keep], n[keep]
        y_rent, y_return = matrix['rentals'][t, n], matrix['returns'][t, n]
        e_lb, e_ub = matrix['exposure'][t, n], matrix['exposure_ub'][t, n]
        e = torch.where((y_rent > 0) & (e_lb < 0.1), torch.maximum(e_lb, e_ub), e_lb)
        mu = model(n, cal['hour'][t], cal['weekday'][t], cal['month'][t], matrix['inv_start'][t, n])
        r = torch.exp(model.log_r).clamp(0.1, 100)

        # 완전 노출은 정확한 관측; 노출 >= 0.5의 부분 관측에는 낮은 가중치를 준다.
        rental_weight = torch.where(e_lb >= 0.999, 1.0,
                                    torch.where(e >= 0.5, partial_weight, 0.0))
        e_effective = e.clamp_min(1 / 60)
        rental_ll = nb_logpmf(y_rent, (mu[:, 0] * e_effective).clamp_min(1e-6), r[0])
        return_ll = nb_logpmf(y_return, mu[:, 1].clamp_min(1e-6), r[1])
        loss_rent = -(rental_weight * rental_ll).sum() / rental_weight.sum().clamp_min(1)
        loss_return = -return_ll.mean()
        loss = loss_rent + 0.5 * loss_return
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 200 == 0 or step == steps:
            print(f'step={step} loss={loss.item():.4f} rental_nll={loss_rent.item():.4f} '
                  f'return_nll={loss_return.item():.4f}', flush=True)
    return model


@torch.no_grad()
def predict_hour(model, data, t):
    n = torch.arange(len(data['stations']), device=DEV)
    inventory = torch.as_tensor(data['inv_start'][t], device=DEV, dtype=torch.float32)
    hour = torch.full_like(n, int(data['hour'][t]))
    weekday = torch.full_like(n, int(data['weekday'][t]))
    month = torch.full_like(n, int(data['month'][t]))
    return model(n, hour, weekday, month, inventory).cpu().numpy()


def evaluate(model, data, train_before, start, end):
    train_times = np.flatnonzero(data['valid'] &
                                  (data['bin_start'] < np.datetime64(train_before)))
    test_times = np.flatnonzero(data['valid'] &
                                 (data['bin_start'] >= np.datetime64(start)) &
                                 (data['bin_start'] < np.datetime64(end)))
    hw = data['hour'] * 2 + (data['weekday'] >= 5).astype(np.int64)
    baseline = fit_historical_rate(data['rentals'], data['exposure'], data['active'],
                                   train_times, hw, data['station_ok'], full_only=True)
    result = {name: {'predicted': 0.0, 'actual': 0.0, 'abs_error': 0.0, 'cells': 0}
              for name in ('direct', 'station_hour_baseline', 'zero')}
    hourly = []
    for t in test_times:
        mask = data['active'][t] & data['station_ok'] & (data['exposure'][t] == 1)
        if not mask.any():
            continue
        y = data['rentals'][t, mask]
        pred = predict_hour(model, data, t)[mask, 0]
        base = baseline[mask, hw[t]]
        for name, p in (('direct', pred), ('station_hour_baseline', base),
                        ('zero', np.zeros_like(pred))):
            r = result[name]
            r['predicted'] += float(p.sum())
            r['actual'] += float(y.sum())
            r['abs_error'] += float(np.abs(p - y).sum())
            r['cells'] += len(y)
        hourly.append((float(pred.sum()), float(y.sum())))
    for r in result.values():
        r['mae'] = r['abs_error'] / max(r['cells'], 1)
        r['pred_actual_ratio'] = r['predicted'] / max(r['actual'], 1)
    result['hours'] = len(hourly)
    result['hourly_total_mae'] = float(np.mean([abs(p - y) for p, y in hourly])) if hourly else None
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--out', default='data/direct_demand_model.pt')
    ap.add_argument('--report', default='data/direct_demand_evaluation.json')
    ap.add_argument('--train-before', default='2026-02-01')
    ap.add_argument('--steps', type=int, default=800)
    ap.add_argument('--batch', type=int, default=65536)
    ap.add_argument('--partial-weight', type=float, default=0.2)
    args = ap.parse_args()
    data = load_data(args.data)
    model = train(data, args.train_before, steps=args.steps, batch=args.batch,
                  partial_weight=args.partial_weight)
    torch.save({'state_dict': model.state_dict(), 'stations': data['stations'].astype(str).tolist(),
                'train_before': args.train_before, 'partial_weight': args.partial_weight}, args.out)
    report = {'train_before': args.train_before, 'partial_weight': args.partial_weight,
              'february_2026': evaluate(model, data, args.train_before, '2026-02-01', '2026-03-01'),
              'march_2026': evaluate(model, data, args.train_before, '2026-03-01', '2026-04-01')}
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
