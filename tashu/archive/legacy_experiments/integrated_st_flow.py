"""Causal inventory -> estimated flows -> ST-GCN -> conditional flow matching.

The forecasting path sees only stock snapshots available at forecast time. Trip
counts supervise the proxy and target during training but are never encoder
inputs. Recent API snapshots are reserved for evaluation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from direct_demand_forecast import DEV, calendar, load_data
from direct_demand_forecast import DirectDemandModel
from censored_flow_matching import build_directed_transition_matrices


SCALE = 5.0
NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


class STEncoder(nn.Module):
    """Temporal convolutions around OD graph propagation."""
    def __init__(self, n, od, lookback=6, d=32):
        super().__init__()
        self.lookback = lookback
        pf, pb = build_directed_transition_matrices(od, k=15)
        self.register_buffer('pf', pf)
        self.register_buffer('pb', pb)
        self.in_proj = nn.Linear(4, d)
        self.temporal1 = nn.Conv2d(d, d, (3, 1), padding=(1, 0))
        self.spatial_f = nn.Linear(d, d)
        self.spatial_b = nn.Linear(d, d)
        self.temporal2 = nn.Conv2d(d, d, (3, 1), padding=(1, 0))
        self.station = nn.Embedding(n, d)
        self.hour = nn.Embedding(24, 8)
        self.weekday = nn.Embedding(7, 4)
        self.month = nn.Embedding(12, 4)

    def forward(self, stock_hist, proxy_hist, hour, weekday, month):
        # stock_hist [B,L+2,N] covers the prior state and current anchor.
        stock_past = stock_hist[:, 1:-1]
        stock_delta = stock_past - stock_hist[:, :-2]
        # Normalize proxy rates into the inferred spatial flow distribution.
        distribution = proxy_hist / proxy_hist.sum(dim=2, keepdim=True).clamp_min(1)
        x = torch.stack((distribution[..., 0], distribution[..., 1],
                         torch.log1p(stock_past.clamp_min(0)),
                         torch.asinh(stock_delta / SCALE)), dim=-1)
        h = self.in_proj(x).permute(0, 3, 1, 2)
        h = F.gelu(self.temporal1(h)).permute(0, 2, 3, 1)
        spatial = (self.spatial_f(torch.matmul(self.pf, h)) +
                   self.spatial_b(torch.matmul(self.pb, h))) * 0.5
        h = F.gelu(h + spatial).permute(0, 3, 1, 2)
        h = F.gelu(self.temporal2(h))[:, :, -1, :].permute(0, 2, 1)
        n = stock_hist.shape[-1]
        node = self.station(torch.arange(n, device=h.device))[None].expand(h.shape[0], -1, -1)
        cal = torch.cat((self.hour(hour), self.weekday(weekday), self.month(month)), -1)
        cal = cal[:, None].expand(-1, n, -1)
        current = torch.log1p(stock_hist[:, -1].clamp_min(0)).unsqueeze(-1)
        condition = torch.cat((h, node, cal, current), -1)
        return condition


class ConditionalVelocity(nn.Module):
    def __init__(self, condition_dim=81):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(condition_dim + 3, 64), nn.GELU(),
                                 nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, tau, condition):
        time = tau[:, None, None].expand(-1, x.shape[1], 1)
        return self.net(torch.cat((x, time, condition), -1))


class IntegratedSTFlow(nn.Module):
    def __init__(self, n, od, lookback=6):
        super().__init__()
        self.lookback = lookback
        self.prior = DirectDemandModel(n)
        self.encoder = STEncoder(n, od, lookback)
        self.velocity = ConditionalVelocity()

    def infer_proxy_history(self, stock_hist, hist_hour, hist_weekday, hist_month):
        b, _, n = stock_hist.shape
        s = torch.arange(n, device=stock_hist.device)[None, None].expand(b, self.lookback, -1)
        h = hist_hour[:, :, None].expand_as(s)
        w = hist_weekday[:, :, None].expand_as(s)
        m = hist_month[:, :, None].expand_as(s)
        return self.prior(s, h, w, m, stock_hist[:, 1:-1])

    def encode(self, stock_hist, hist_hour, hist_weekday, hist_month,
               target_hour, target_weekday, target_month):
        with torch.no_grad():
            proxy = self.infer_proxy_history(stock_hist, hist_hour, hist_weekday, hist_month)
            n = stock_hist.shape[-1]
            s = torch.arange(n, device=stock_hist.device)[None].expand(stock_hist.shape[0], -1)
            baseline = self.prior(s, target_hour[:, None].expand_as(s),
                                  target_weekday[:, None].expand_as(s),
                                  target_month[:, None].expand_as(s), stock_hist[:, -1])
        condition = self.encoder(stock_hist, proxy, target_hour, target_weekday, target_month)
        return condition, baseline

    @torch.no_grad()
    def predict(self, stock_hist, hist_hour, hist_weekday, hist_month,
                target_hour, target_weekday, target_month, ode_steps=8):
        self.eval()
        condition, baseline = self.encode(stock_hist, hist_hour, hist_weekday, hist_month,
                                          target_hour, target_weekday, target_month)
        x = baseline / SCALE
        for i in range(ode_steps):
            tau = torch.full((x.shape[0],), i / ode_steps, device=x.device)
            x = x + self.velocity(x, tau, condition) / ode_steps
        return (x * SCALE).clamp_min(0), baseline


def load_historical(path, train_od=None):
    data = load_data(path)
    with np.load(path) as f:
        data['od'] = f['od'].copy()
    if train_od is not None:
        data['od'] = np.load(train_od)
        if data['od'].shape != (len(data['stations']), len(data['stations'])):
            raise ValueError('Train OD graph station count does not match historical data')
    data['stock'] = (data['inv_start'].astype(np.float32) +
                     data['inv_ub'].astype(np.float32)) / 2
    return data


def valid_targets(data, start=None, end=None, lookback=6):
    valid = data['valid'].copy()
    if start is not None:
        valid &= data['bin_start'] >= np.datetime64(start)
    if end is not None:
        valid &= data['bin_start'] < np.datetime64(end)
    ok = []
    for t in np.flatnonzero(valid):
        if t >= lookback + 1 and data['valid'][t - lookback - 1:t + 1].all():
            ok.append(t)
    return np.asarray(ok, dtype=np.int64)


def tensors(data):
    names = ('stock', 'rentals', 'returns', 'exposure', 'exposure_ub', 'active')
    z = {k: torch.as_tensor(data[k], device=DEV) for k in names}
    z.update({k: torch.as_tensor(data[k], device=DEV, dtype=torch.long)
              for k in ('hour', 'weekday', 'month')})
    z['station_ok'] = torch.as_tensor(data['station_ok'], device=DEV)
    return z


def rental_exposure(z, t, s=None):
    lower = z['exposure'][t] if s is None else z['exposure'][t, s]
    upper = z['exposure_ub'][t] if s is None else z['exposure_ub'][t, s]
    rentals = z['rentals'][t] if s is None else z['rentals'][t, s]
    return torch.where((rentals > 0) & (lower < 0.1), torch.maximum(lower, upper), lower)


def sequence(z, indices, lookback):
    offsets = torch.arange(-lookback - 1, 1, device=DEV)
    hist_idx = indices[:, None] + offsets[None]
    past_idx = indices[:, None] + torch.arange(-lookback, 0, device=DEV)[None]
    return (z['stock'][hist_idx], z['hour'][past_idx], z['weekday'][past_idx],
            z['month'][past_idx], z['hour'][indices], z['weekday'][indices],
            z['month'][indices])


def fit(data, prior_checkpoint, train_before='2026-02-01', flow_steps=500,
        lookback=6, batch=8):
    torch.manual_seed(0)
    np.random.seed(0)
    z = tensors(data)
    model = IntegratedSTFlow(len(data['stations']), data['od'], lookback).to(DEV)
    prior_ckpt = torch.load(prior_checkpoint, map_location=DEV, weights_only=True)
    if prior_ckpt['stations'] != data['stations'].astype(str).tolist():
        raise ValueError('Prior model station order does not match historical data')
    if prior_ckpt['train_before'] != train_before:
        raise ValueError('Prior model must use the same training cutoff')
    model.prior.load_state_dict(prior_ckpt['state_dict'])
    model.prior.eval()
    for p in model.prior.parameters():
        p.requires_grad_(False)
    pool = valid_targets(data, end=train_before, lookback=lookback)
    if not len(pool):
        raise ValueError('No causal historical sequences before cutoff')
    pool = torch.as_tensor(pool, device=DEV)
    gen = torch.Generator(device=DEV).manual_seed(0)
    opt = torch.optim.AdamW(list(model.encoder.parameters()) + list(model.velocity.parameters()),
                            lr=0.001, weight_decay=1e-4)
    print(f'ST-GCN + conditional flow training: {flow_steps} steps', flush=True)
    for step in range(1, flow_steps + 1):
        idx = pool[torch.randint(len(pool), (batch,), device=DEV, generator=gen)]
        inputs = sequence(z, idx, lookback)
        condition, baseline = model.encode(*inputs)
        y = torch.stack((z['rentals'][idx], z['returns'][idx]), -1).float()
        e = rental_exposure(z, idx)
        full = z['exposure'][idx] >= 0.999
        active = z['active'][idx] & z['station_ok'][None]
        rent_weight = torch.where(full, 1., torch.where(e >= 0.5, 0.2, 0.)) * active
        weights = torch.stack((rent_weight, active.float()), -1)
        endpoint_rent = (y[..., 0] / e.clamp_min(0.5)).clamp_max(20)
        endpoint = torch.stack((torch.where(rent_weight > 0, endpoint_rent, baseline[..., 0].detach()),
                                y[..., 1]), -1) / SCALE
        x0 = baseline / SCALE
        tau = torch.rand((len(idx),), device=DEV)
        x_tau = (1 - tau[:, None, None]) * x0 + tau[:, None, None] * endpoint
        velocity = model.velocity(x_tau, tau, condition)
        flow_loss = ((velocity - (endpoint - x0)).square() * weights).sum() / weights.sum().clamp_min(1)
        loss = flow_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.velocity.parameters()), 5)
        opt.step()
        if step == 1 or step % 100 == 0 or step == flow_steps:
            print(f'flow step={step} loss={loss.item():.4f}', flush=True)
    return model


@torch.no_grad()
def historical_evaluate(model, data, start, end, max_hours=300, direct_model=None):
    z = tensors(data)
    pool = valid_targets(data, start, end, model.lookback)
    if len(pool) > max_hours:
        pool = np.random.default_rng(42).choice(pool, max_hours, replace=False)
    names = ('flow', 'baseline', 'zero', 'direct') if direct_model is not None else ('flow', 'baseline', 'zero')
    totals = {name: {'error': 0.0, 'actual': 0.0, 'predicted': 0.0, 'cells': 0}
              for name in names}
    channels = {name: {kind: {'absolute_error': 0.0, 'actual': 0.0,
                               'predicted': 0.0, 'cells': 0}
                       for kind in ('rentals', 'returns')}
                for name in totals}
    step_diff = []
    model.eval()
    for chunk in np.array_split(pool, max(1, int(np.ceil(len(pool) / 8)))):
        idx = torch.as_tensor(chunk, device=DEV)
        pred, base = model.predict(*sequence(z, idx, model.lookback))
        pred_one, _ = model.predict(*sequence(z, idx, model.lookback), ode_steps=1)
        step_diff.append((pred - pred_one).abs().mean().item())
        actual = torch.stack((z['rentals'][idx], z['returns'][idx]), -1).float()
        active = z['active'][idx] & z['station_ok'][None]
        masks = torch.stack((active & (z['exposure'][idx] >= 0.999), active), -1)
        estimates = [('flow', pred), ('baseline', base), ('zero', torch.zeros_like(base))]
        if direct_model is not None:
            n = torch.arange(len(data['stations']), device=DEV)[None].expand(len(idx), -1)
            direct = direct_model(n, z['hour'][idx, None].expand_as(n),
                                  z['weekday'][idx, None].expand_as(n),
                                  z['month'][idx, None].expand_as(n),
                                  torch.as_tensor(data['inv_start'][chunk], device=DEV, dtype=torch.float32))
            estimates.append(('direct', direct))
        for name, estimate in estimates:
            v = totals[name]
            v['error'] += float(((estimate - actual).abs() * masks).sum().item())
            v['actual'] += float((actual * masks).sum().item())
            v['predicted'] += float((estimate * masks).sum().item())
            v['cells'] += int(masks.sum().item())
            for j, kind in enumerate(('rentals', 'returns')):
                ch = channels[name][kind]
                mask = masks[..., j]
                ch['absolute_error'] += float(((estimate[..., j] - actual[..., j]).abs() * mask).sum())
                ch['actual'] += float((actual[..., j] * mask).sum())
                ch['predicted'] += float((estimate[..., j] * mask).sum())
                ch['cells'] += int(mask.sum())
    for v in totals.values():
        v['mae'] = v['error'] / max(v['cells'], 1)
        v['pred_actual_ratio'] = v['predicted'] / max(v['actual'], 1)
    for model_channels in channels.values():
        for ch in model_channels.values():
            ch['mae'] = ch['absolute_error'] / max(ch['cells'], 1)
            ch['pred_actual_ratio'] = ch['predicted'] / max(ch['actual'], 1)
    return {'hours': len(pool), 'metrics': totals, 'by_channel': channels,
            'ode_1_vs_8_mean_abs_difference':
            float(np.mean(step_diff)) if step_diff else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--train-od', default=None, help='OD matrix built only before --train-before')
    ap.add_argument('--out', default='data/integrated_st_flow.pt')
    ap.add_argument('--report', default='data/integrated_st_flow_historical.json')
    ap.add_argument('--train-before', default='2026-02-01')
    ap.add_argument('--prior-checkpoint', default='data/direct_demand_model.pt')
    ap.add_argument('--flow-steps', type=int, default=500)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lookback', type=int, default=6)
    ap.add_argument('--eval-only', action='store_true')
    ap.add_argument('--direct-model', default='data/direct_demand_model.pt')
    args = ap.parse_args()
    data = load_historical(args.data, args.train_od)
    if args.eval_only:
        checkpoint = torch.load(args.out, map_location=DEV, weights_only=True)
        if checkpoint['stations'] != data['stations'].astype(str).tolist():
            raise ValueError('Checkpoint station order does not match data')
        model = IntegratedSTFlow(len(data['stations']), data['od'], checkpoint['lookback']).to(DEV)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = fit(data, args.prior_checkpoint, args.train_before, args.flow_steps,
                    args.lookback, args.batch)
        checkpoint = {'state_dict': model.state_dict(), 'stations': data['stations'].astype(str).tolist(),
                      'train_before': args.train_before, 'lookback': args.lookback,
                      'train_od': args.train_od,
                      'prior_checkpoint': args.prior_checkpoint,
                      'encoder_inputs': 'reconstructed stock history only; trip counts used as training targets'}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.out)
    direct_model = None
    if args.direct_model and Path(args.direct_model).exists():
        direct_ckpt = torch.load(args.direct_model, map_location=DEV, weights_only=True)
        if direct_ckpt['stations'] != data['stations'].astype(str).tolist():
            raise ValueError('Direct comparator station order does not match')
        direct_model = DirectDemandModel(len(data['stations'])).to(DEV)
        direct_model.load_state_dict(direct_ckpt['state_dict'])
        direct_model.eval()
    report = {'train_before': args.train_before, 'train_od': args.train_od,
              'february_2026': historical_evaluate(model, data, '2026-02-01', '2026-03-01', direct_model=direct_model),
              'march_2026': historical_evaluate(model, data, '2026-03-01', '2026-04-01', direct_model=direct_model)}
    Path(args.report).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
