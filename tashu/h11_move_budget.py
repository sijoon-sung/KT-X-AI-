"""H11: 06시 배분의 실행 가능성 — 밤사이 옮겨야 하는 자전거 수와, 이동 예산 M대일 때의 손실 감소.
P2 목표(300m 도보권 배분)와 현행 06시 재고의 차이 = 필요한 야간 이동량. 예산 M 이 있으면
'현행 → 목표' 이동 중 손실 감소/대 가 큰 순으로 M 대까지만 실행 (한계이득 탐욕).  실현 200개로 손실 평가.
"""
import json
import numpy as np

import baseline_udf as B
import structured_rates as SR
import h67_decision as H

NR = 200
BUDGETS = (0, 50, 100, 200, 400, 800, 10 ** 9)


def main():
    z = SR.load(); S = SR.structure(z); W = SR.weather_by_bin(z)
    N = z['rentals'].shape[1]; key48 = S['key48']
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    lam_pol, mu_pol = H.rates(z, S, e, '2025-08-01', '2026-02-01')
    lam_gen, mu_gen = H.rates(z, S, e, '2026-02-01', '2026-04-01')
    Y = z['rentals'].astype(float)
    train_t = (z['bin_start'] >= np.datetime64('2024-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok6']
    ts = np.flatnonzero(train_t); obs = np.zeros(len(ts)); ex = np.zeros(len(ts))
    for i, t in enumerate(ts):
        ym = str(z['ym'][t])
        if ym not in SR.BIKES:
            continue
        mu_ = S['scale'] * S['tshape'][key48[t]] * (S['per_bike'][ym] / S['base_pb']) * (SR.BIKES[ym] / S['base_bikes'])
        m = S['full'][t]; obs[i] = Y[t][m].sum(); ex[i] = mu_[m].sum()
    ok = ex > 0
    fac = lambda bins, v: [float(obs[s_].sum() / ex[s_].sum()) for lo, hi in bins for s_ in [ok & ((v <= 0) if hi == 0.0 else ((v > lo) & (v <= hi)))]]
    rf, tf = fac(H.RAIN, W['precipitation'][ts]), fac(H.TEMP, W['temperature_2m'][ts])
    wf = lambda r, t: next(f for (lo, hi), f in zip(H.RAIN, rf) if (r <= 0 if hi == 0.0 else lo < r <= hi)) * next(f for (lo, hi), f in zip(H.TEMP, tf) if lo < t <= hi)
    month_w = {}
    for ym in np.unique(z['ym']):
        s_ = np.flatnonzero((z['ym'] == ym) & z['ok6'])
        if len(s_) >= 100:
            month_w[str(ym)] = float(np.mean([wf(W['precipitation'][t], W['temperature_2m'][t]) for t in s_]))
    pol_mult = lambda t: SR.season_level(S, str(z['ym'][t]), SR.BIKES[str(z['ym'][t])]) * wf(W['precipitation'][t], W['temperature_2m'][t]) / month_w[f"{int(str(z['ym'][t])[:4]) - 1}-{str(z['ym'][t])[5:]}"]
    gen_mult = lambda t: wf(W['precipitation'][t], W['temperature_2m'][t]) / month_w[str(z['ym'][t])]
    with np.load('data/processed/tashu_60min.npz') as f:
        dist = f['dist_km']; lat = f['lat']
    cust = z['cust'] & (lat > 0)
    order = np.argsort(-S['scale']); group = np.full(N, -1); hubs = []
    for s in order:
        if not cust[s]:
            continue
        near = [h for h in hubs if dist[s, h] <= 0.3]
        group[s] = group[min(near, key=lambda h: dist[s, h])] if near else len(hubs)
        if not near:
            hubs.append(s)
    G = len(hubs)
    rng = np.random.default_rng(2); R = {'budgets': BUDGETS, 'NR': NR}
    for lab, (a, b) in (('feb_2026', ('2026-02-01', '2026-03-01')), ('mar_2026', ('2026-03-01', '2026-04-01'))):
        days = np.unique(z['day'][(z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b))])
        t_any = np.flatnonzero((z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b)) & z['ok6'])
        m_avg = float(np.mean([pol_mult(t) for t in t_any]))
        B._F_CACHE.clear(); f_st = {dt: B.lost_curve(lam_pol * m_avg, mu_pol * m_avg, dt, b_max=60) for dt in (0, 1)}
        lam_g = np.zeros((48, G)); mu_g = np.zeros((48, G))
        for g in range(G):
            idx = np.flatnonzero(group == g); lam_g[:, g] = lam_pol[:, idx].sum(1); mu_g[:, g] = mu_pol[:, idx].sum(1)
        B._F_CACHE.clear(); f_g = {dt: B.lost_curve(lam_g * m_avg, mu_g * m_avg, dt, b_max=150) for dt in (0, 1)}
        acc = {M: [] for M in BUDGETS}; moved = []; n_days = 0
        for d in days:
            t_idx = np.flatnonzero((z['day'] == d) & (z['hour'] >= 6) & (z['hour'] < 22))
            if len(t_idx) < 16 or not z['ok'][t_idx].all():
                continue
            t0 = t_idx[0]; dt = int(z['daytype'][t0]); act = z['active'][t0] & cust
            cur = ((z['inv_start'][t0] + z['inv_ub'][t0]) / 2).round().astype(int); cur[~act] = 0; total = int(cur.sum())
            fg = f_g[dt].copy(); act_g = np.array([act[group == g].any() for g in range(G)]); fg[~act_g] = 0
            gg = fg[:, :-1] - fg[:, 1:]; alloc_g = np.zeros(G, int); nxg = gg[:, 0].copy(); nxg[~act_g] = -1
            for _ in range(total):
                g_ = int(np.argmax(nxg))
                if nxg[g_] <= 0:
                    break
                alloc_g[g_] += 1; nxg[g_] = gg[g_, alloc_g[g_]] if alloc_g[g_] < 150 else -1
            target = np.zeros(N, int)
            for g_ in range(G):
                idx = np.flatnonzero((group == g_) & act)
                if not len(idx) or alloc_g[g_] == 0:
                    continue
                w = lam_pol[:, idx].sum(0); w = w / w.sum() if w.sum() > 0 else np.ones(len(idx)) / len(idx)
                base = np.floor(w * alloc_g[g_]).astype(int); rem = alloc_g[g_] - base.sum()
                base[np.argsort(-(w * alloc_g[g_] - base))[:rem]] += 1; target[idx] = base
            need_move = int(np.maximum(target - cur, 0).sum()); moved.append(need_move)
            # 예산 M: 대여소별 손실곡선 f_st 로 (수거 1대의 손실 증가, 배치 1대의 손실 감소) 짝을 한계이득 순으로
            f = f_st[dt]; b_max = f.shape[1] - 1
            def lost_at(vec):
                return f[np.arange(N), np.clip(vec, 0, b_max)]
            plans = {}
            for M in BUDGETS:
                I = cur.copy(); k = 0
                surplus = np.where(act, I - target, 0); deficit = np.where(act, target - I, 0)
                while k < M and (surplus > 0).any() and (deficit > 0).any():
                    gain_drop = np.where(deficit > 0, lost_at(I) - lost_at(I + 1), -1)          # 배치 1대의 감소
                    cost_pick = np.where(surplus > 0, lost_at(I - 1) - lost_at(I), 1e9)         # 수거 1대의 증가
                    j = int(np.argmax(gain_drop)); i_ = int(np.argmin(cost_pick))
                    if gain_drop[j] - cost_pick[i_] <= 0:
                        break
                    I[j] += 1; I[i_] -= 1; deficit[j] -= 1; surplus[i_] -= 1; k += 1
                plans[M] = I
            inv = {M: np.tile(v.astype(np.int64), (NR, 1)) for M, v in plans.items()}
            loss = {M: np.zeros(NR) for M in BUDGETS}
            for t in t_idx:
                k_ = key48[t]; gm = gen_mult(t)
                dem = rng.poisson(lam_gen[k_] * gm * act, (NR, N)); ret = rng.poisson(mu_gen[k_] * gm * act, (NR, N))
                for M in BUDGETS:
                    I = inv[M]; short = np.maximum(dem - I, 0); I = I - (dem - short)
                    if short.any():
                        for g_ in np.unique(group[short.any(0)]):
                            if g_ < 0:
                                continue
                            idx = np.flatnonzero(group == g_); need = short[:, idx].sum(1); avail = I[:, idx].sum(1)
                            take = np.minimum(need, avail); loss[M] += need - take
                            I[:, idx] = I[:, idx] - np.floor(I[:, idx] * (take / np.maximum(avail, 1))[:, None]).astype(np.int64)
                    inv[M] = I + ret
            for M in BUDGETS:
                acc[M].append(loss[M])
            n_days += 1
        base = np.sum(acc[0], 0)
        res = {'days': n_days, 'bikes_to_move_per_night_for_full_P2': {'mean': float(np.mean(moved)), 'p90': float(np.quantile(moved, .9))}}
        for M in BUDGETS:
            per = np.sum(acc[M], 0) / n_days; red = (base - np.sum(acc[M], 0)) / base
            res[f'budget_{M}'] = {'loss_per_day': round(float(per.mean()), 1), 'reduction': round(float(red.mean()), 3), 'ci95': [round(float(x), 3) for x in np.quantile(red, [.025, .975])]}
        R[lab] = res; print(lab, json.dumps(res, indent=1), flush=True)
    json.dump(R, open('data/h11_move_budget.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
