"""(1) 결품 예측의 천장: 그 시간의 실제 대여·반납 건수를 안다고 가정한 오라클 AUC (과거 2~3월)
(2) 하이브리드: 확률모델(축소 요율 d=0.7, κ=30 → 마르코프 p0) 출력을 입력으로 받는 신경망 vs 각각 단독.
    학습: 2025-08~2026-01 복원 라벨(exposure_ub<1). 평가: 2026-02/03 복원 라벨, 2026-09 API 실측.
"""
import json
import numpy as np
import torch
from torch import nn

import baseline_udf as B
import structured_rates as SR

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RAIN = [(0, 0.0), (0.0, 1.0), (1.0, 5.0), (5.0, 1e9)]
TEMP = [(-99, 0), (0, 10), (10, 20), (20, 28), (28, 99)]
torch.manual_seed(0); np.random.seed(0)


def p0_table(lam_grid, mu_grid):
    """1시간 뒤 재고 0 확률 표 [b, i_lam, i_mu]."""
    L, M = len(lam_grid), len(mu_grid)
    b = np.repeat(np.arange(B.K + 1), L * M)
    lam = np.tile(np.repeat(lam_grid, M), B.K + 1); mu = np.tile(mu_grid, (B.K + 1) * L)
    P = np.zeros((len(b), B.K + 1)); P[np.arange(len(b)), b] = 1
    P, _ = B.step_hour(P, lam, mu)
    return P[:, 0].reshape(B.K + 1, L, M)


class Net(nn.Module):
    def __init__(self, n_st, extra):
        super().__init__()
        self.st = nn.Embedding(n_st, 24); self.h = nn.Embedding(24, 8); self.w = nn.Embedding(7, 4); self.m = nn.Embedding(12, 4)
        self.head = nn.Sequential(nn.Linear(40 + extra, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, s, h, w, m, x):
        return self.head(torch.cat((self.st(s), self.h(h), self.w(w), self.m(m), x), -1)).squeeze(-1)


def main():
    z = SR.load(); S = SR.structure(z); W = SR.weather_by_bin(z)
    Y = z['rentals'].astype(float); key48 = S['key48']; N = Y.shape[1]
    # ---- 채택 요율: damped 0.7, κ=30 (shrink_rates 와 동일)
    recent = (z['bin_start'] >= np.datetime64('2025-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok']
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    c48 = np.zeros((48, N)); h48 = np.zeros((48, N))
    for k in range(48):
        sel = recent & (key48 == k); a = z['active'][sel] & z['cust'][None]
        c48[k] = (Y[sel] * a).sum(0); h48[k] = (a * (e[sel] + 0.7 * (1 - e[sel]))).sum(0)
    lam48 = (c48 + 30 * S['scale'][None, :] * S['tshape']) / (h48 + 30)
    _, mu48 = B.fit_rates(z, '2025-08-01', '2026-02-01'); mu48 = np.nan_to_num(mu48)
    # ---- 날씨 계수
    train_t = (z['bin_start'] >= np.datetime64('2024-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok6']
    ts = np.flatnonzero(train_t); obs = np.zeros(len(ts)); ex = np.zeros(len(ts))
    for i, t in enumerate(ts):
        ym = str(z['ym'][t])
        if ym not in SR.BIKES:
            continue
        mu = S['scale'] * S['tshape'][key48[t]] * (S['per_bike'][ym] / S['base_pb']) * (SR.BIKES[ym] / S['base_bikes'])
        m = S['full'][t]; obs[i] = Y[t][m].sum(); ex[i] = mu[m].sum()
    ok = ex > 0
    def fac(bins, v):
        return [float(obs[sel].sum() / ex[sel].sum()) for lo, hi in bins for sel in [ok & ((v <= 0) if hi == 0.0 else ((v > lo) & (v <= hi)))]]
    rf, tf = fac(RAIN, W['precipitation'][ts]), fac(TEMP, W['temperature_2m'][ts])
    def wf(r, t):
        return next(f for (lo, hi), f in zip(RAIN, rf) if (r <= 0 if hi == 0.0 else lo < r <= hi)) * next(f for (lo, hi), f in zip(TEMP, tf) if lo < t <= hi)
    month_w = {}
    for ym in np.unique(z['ym']):
        sel = np.flatnonzero((z['ym'] == ym) & z['ok6'])
        if len(sel) >= 100:
            month_w[str(ym)] = float(np.mean([wf(W['precipitation'][t], W['temperature_2m'][t]) for t in sel]))
    def mult(t):
        ym = str(z['ym'][t]); prev = f'{int(ym[:4]) - 1}-{ym[5:]}'
        bikes = SR.BIKES.get(ym, S['base_bikes'])
        return SR.season_level(S, ym, bikes) * wf(W['precipitation'][t], W['temperature_2m'][t]) / month_w.get(prev, month_w.get(ym, 1.0))
    # ---- p0 표
    lam_grid = np.concatenate([[0], np.geomspace(0.01, 20, 60)]); mu_grid = np.concatenate([[0], np.geomspace(0.01, 20, 60)])
    tab = p0_table(lam_grid, mu_grid)
    def p0_of(b, lam, mu):
        ib = np.clip(np.round(b), 0, B.K).astype(int)
        il = np.searchsorted(lam_grid, lam); il = np.clip(il, 0, len(lam_grid) - 1)
        im = np.clip(np.searchsorted(mu_grid, mu), 0, len(mu_grid) - 1)
        return tab[ib, il, im]

    inv = (z['inv_start'].astype(np.float32) + z['inv_ub']) / 2
    hour, wday = z['hour'], z['dow']; moy = z['bin_start'].astype('datetime64[M]').astype(np.int64) % 12

    def features(t, s, inventory):
        k = key48[t]; m_ = mult(t)
        lam = lam48[k, s] * m_; mu = mu48[k, s] * m_
        p0 = p0_of(inventory, lam, mu)
        r = W['precipitation'][t]; tp = W['temperature_2m'][t]
        x = np.stack([np.log1p(inventory), np.log(lam + 1e-3), np.log(mu + 1e-3), p0, np.full(len(s), r > 0), np.full(len(s), tp / 30)], 1)
        return x.astype(np.float32), p0

    def sample(t_pool, n_hours, rng):
        rows = []
        for t in rng.choice(t_pool, min(n_hours, len(t_pool)), replace=False):
            s = np.flatnonzero(z['active'][t] & z['cust'] & (inv[t] > 0) & ((z['exposure_ub'][t] < 0.999) | (z['exposure'][t] >= 0.999)))
            if not len(s):
                continue
            x, p0 = features(t, s, inv[t, s])
            lab = (z['exposure_ub'][t, s] < 0.999)
            # 오라클: 그 시간의 순변화를 안다 (재고 - 대여 + 반납 이 0 이하면 결품 쪽)
            oracle = -(inv[t, s] - z['rentals'][t, s])
            rows.append((np.full(len(s), t), s, x, lab, p0, oracle))
        return [np.concatenate(c) for c in zip(*rows)]

    rng = np.random.default_rng(0)
    pool_tr = np.flatnonzero(z['ok'] & (z['hour'] >= 7) & (z['hour'] <= 22) & (z['bin_start'] >= np.datetime64('2024-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')))
    pool_te = np.flatnonzero(z['ok'] & (z['hour'] >= 7) & (z['hour'] <= 22) & (z['bin_start'] >= np.datetime64('2026-02-01')) & (z['bin_start'] < np.datetime64('2026-04-01')))
    tr = sample(pool_tr, 3000, rng); te = sample(pool_te, 400, rng)
    R = {'train_cells': int(len(tr[0])), 'test_cells': int(len(te[0]))}
    R['hist_feb_mar'] = {'markov_p0': B.auc(te[4], te[3]), 'stock_only': B.auc(-inv[te[0], te[1]], te[3]),
                         'oracle_known_counts': B.auc(te[5], te[3])}

    def fit(cols, name, steps=1500):
        net = Net(N, len(cols)).to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        T_ = {k: torch.as_tensor(v, device=DEV) for k, v in dict(t=tr[0], s=tr[1], x=tr[2][:, cols], y=tr[3].astype(np.float32)).items()}
        H = {k: torch.as_tensor(v[T_['t'].cpu().numpy()], device=DEV, dtype=torch.long) for k, v in (('h', hour), ('w', wday), ('m', moy))}
        n = len(tr[0]); g = torch.Generator(device=DEV).manual_seed(0)
        for step in range(steps):
            i = torch.randint(n, (65536,), device=DEV, generator=g)
            logit = net(T_['s'][i], H['h'][i], H['w'][i], H['m'][i], T_['x'][i])
            loss = nn.functional.binary_cross_entropy_with_logits(logit, T_['y'][i])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            tt = torch.as_tensor(te[0], device=DEV)
            logit = net(torch.as_tensor(te[1], device=DEV), torch.as_tensor(hour[te[0]], device=DEV), torch.as_tensor(wday[te[0]], device=DEV),
                        torch.as_tensor(moy[te[0]], device=DEV), torch.as_tensor(te[2][:, cols], device=DEV)).cpu().numpy()
        R['hist_feb_mar'][name] = B.auc(logit, te[3])
        return net

    net_plain = fit([0, 4, 5], 'nn_inventory_calendar_weather')       # stockout_head + 날씨
    net_hyb = fit([0, 1, 2, 3, 4, 5], 'nn_hybrid(+lam,mu,p0)')
    print(json.dumps(R, indent=1), flush=True)

    # ---- 9월 API
    ts_api, sids, stock = B.load_api(r'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    st = z['stations'].astype(str); idx_model = {s: i for i, s in enumerate(st)}
    m_idx = np.array([idx_model[s] for s in sids if s in idx_model]); a_idx = np.array([j for j, s in enumerate(sids) if s in idx_model])
    d = json.load(open('data/external/daejeon_hourly_openmeteo.json')); h = d['hourly']
    widx = {t: i for i, t in enumerate(np.array(h['time'], 'datetime64[s]').astype('datetime64[h]').astype(np.int64))}
    pr = np.array(h['precipitation'], float); tp = np.array(h['temperature_2m'], float)
    sl = SR.season_level(S, '2026-09', S['base_bikes'])
    res = {k: ([], []) for k in ('stock_only', 'markov_p0', 'nn_inventory_calendar_weather', 'nn_hybrid(+lam,mu,p0)')}
    for t in range(len(ts_api) - 6):
        hr = int(ts_api[t].astype('datetime64[h]').astype(np.int64) % 24)
        span = (ts_api[t + 6] - ts_api[t]).astype('timedelta64[s]').astype(int)
        if not (7 <= hr <= 22 and 3300 <= span <= 3900):
            continue
        cur = stock[t, a_idx]; fut = stock[t + 1:t + 7, a_idx]
        ok_ = ~np.isnan(cur) & ~np.isnan(fut).any(0) & (cur > 0)
        if not ok_.any():
            continue
        event = (fut.min(0) == 0)[ok_]; s = m_idx[ok_]; b = np.clip(cur[ok_], 0, B.K)
        dow = int((ts_api[t].astype('datetime64[D]').astype(np.int64) + 3) % 7); k = hr * 2 + int(dow >= 5)
        i = widx.get(int(ts_api[t].astype('datetime64[h]').astype(np.int64)))
        r_, t_ = (pr[i], tp[i]) if i is not None else (0.0, 20.0)
        m_ = sl * wf(r_, t_) / month_w['2025-09']
        lam = lam48[k, s] * m_; mu = mu48[k, s] * m_; p0 = p0_of(b, lam, mu)
        x = np.stack([np.log1p(b), np.log(lam + 1e-3), np.log(mu + 1e-3), p0, np.full(len(s), r_ > 0), np.full(len(s), t_ / 30)], 1).astype(np.float32)
        with torch.no_grad():
            args = (torch.as_tensor(s, device=DEV), torch.full((len(s),), hr, device=DEV), torch.full((len(s),), dow, device=DEV), torch.full((len(s),), 8, device=DEV))
            lp = net_plain(*args, torch.as_tensor(x[:, [0, 4, 5]], device=DEV)).cpu().numpy()
            lh = net_hyb(*args, torch.as_tensor(x, device=DEV)).cpu().numpy()
        for name, sc in (('stock_only', -b), ('markov_p0', p0), ('nn_inventory_calendar_weather', lp), ('nn_hybrid(+lam,mu,p0)', lh)):
            res[name][0].append(sc); res[name][1].append(event)
    R['api_sept'] = {k: B.auc(np.concatenate(v[0]), np.concatenate(v[1])) for k, v in res.items()}
    # 창 단위 짝 비교: hybrid - markov, hybrid - nn_plain
    rng2 = np.random.default_rng(0)
    for a_, b_ in (('nn_hybrid(+lam,mu,p0)', 'markov_p0'), ('nn_hybrid(+lam,mu,p0)', 'nn_inventory_calendar_weather')):
        dif = np.array([B.auc(x1, ev) - B.auc(x2, ev) for x1, x2, ev in zip(res[a_][0], res[b_][0], res[a_][1]) if ev.any() and not ev.all()])
        bs = rng2.choice(dif, size=(5000, len(dif)), replace=True).mean(1)
        R[f'api_paired {a_} - {b_}'] = {'mean': float(dif.mean()), 'improved': int((dif > 0).sum()), 'windows': len(dif), 'ci95': np.quantile(bs, [.025, .975]).tolist()}
    print(json.dumps(R, indent=1))
    json.dump(R, open('data/hybrid_test.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
