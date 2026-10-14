"""H10(전 단계): 낮 동안 트럭 재배치의 가치.  가상 시나리오, RL 없음.

트럭 T∈{0,1,2,3}, 적재 15, 작업창 10~13시·17~21시, 정차 5분, 20 km/h.
목표 재고 = 06시 도보권 배분(P2)의 대여소 값.  트럭 규칙(rolling 탐욕):
  적재가 있으면 → 부족(target−I)이 큰 대여소 중 (부족 × λ_now) / (이동+정차 시간) 최대인 곳에 배치
  적재가 없거나 부족 후보가 없으면 → 초과(I−target) 가 큰 곳에서 수거
비교: P0 정적 / P2 정적 / P0+트럭 / P2+트럭, 실현 NR개, 10분 단위 시뮬레이션.
수요 생성기·요율·도보권·손실 정의는 h67_decision 과 동일 (도보권 손실만 보고).
"""
import json
import sys
import numpy as np

import baseline_udf as B
import structured_rates as SR
import h67_decision as H

NR = int(sys.argv[1]) if len(sys.argv) > 1 else 40
TRUCKS = (0, 1, 2, 3); CAP = 15; STOP_MIN = 5; SPEED = 20.0
import os
WINDOWS = ((6, 8), (16, 18)) if os.environ.get('H10_WINDOW') == 'prepeak' else ((10, 13), (17, 21))
STEP = 10  # 분


def main():
    z = SR.load(); S = SR.structure(z); W = SR.weather_by_bin(z)
    N = z['rentals'].shape[1]; key48 = S['key48']
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    lam_pol, mu_pol = H.rates(z, S, e, '2025-08-01', '2026-02-01')
    lam_gen, mu_gen = H.rates(z, S, e, '2026-02-01', '2026-04-01')
    # 날씨 (h67 과 동일 절차)
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
        near = [h for h in hubs if dist[s, h] <= H.WALK_KM]
        if near:
            group[s] = group[min(near, key=lambda h: dist[s, h])]
        else:
            group[s] = len(hubs); hubs.append(s)
    G = len(hubs)
    travel_min = np.where(np.isfinite(dist), dist / SPEED * 60, 1e9)
    rng = np.random.default_rng(1)
    R = {'NR': NR, 'trucks': TRUCKS, 'cap': CAP, 'windows': WINDOWS}
    for lab, (a, b) in (('feb_2026', ('2026-02-01', '2026-03-01')), ('mar_2026', ('2026-03-01', '2026-04-01'))):
        days = np.unique(z['day'][(z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b))])
        t_any = np.flatnonzero((z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b)) & z['ok6'])
        m_avg = float(np.mean([pol_mult(t) for t in t_any]))
        lam_g = np.zeros((48, G)); mu_g = np.zeros((48, G))
        for g in range(G):
            idx = np.flatnonzero(group == g); lam_g[:, g] = lam_pol[:, idx].sum(1); mu_g[:, g] = mu_pol[:, idx].sum(1)
        GB = 150
        B._F_CACHE.clear(); f_g = {dt: B.lost_curve(lam_g * m_avg, mu_g * m_avg, dt, b_max=GB) for dt in (0, 1)}
        acc = {}
        n_days = 0
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
                alloc_g[g_] += 1; nxg[g_] = gg[g_, alloc_g[g_]] if alloc_g[g_] < GB else -1
            target = np.zeros(N, int)
            for g_ in range(G):
                idx = np.flatnonzero((group == g_) & act)
                if not len(idx) or alloc_g[g_] == 0:
                    continue
                w = lam_pol[:, idx].sum(0); w = w / w.sum() if w.sum() > 0 else np.ones(len(idx)) / len(idx)
                base = np.floor(w * alloc_g[g_]).astype(int); rem = alloc_g[g_] - base.sum()
                base[np.argsort(-(w * alloc_g[g_] - base))[:rem]] += 1; target[idx] = base
            starts = {'P0': cur, 'P2': target}
            # 10분 단위 수요·반납 (모든 실현 공통 난수 → 정책 간 짝 비교)
            steps = []
            for t in t_idx:
                k = key48[t]; gm = gen_mult(t); hh = int(z['hour'][t])
                for j in range(60 // STEP):
                    dem = rng.poisson(lam_gen[k] * gm * act / (60 / STEP), (NR, N)); ret = rng.poisson(mu_gen[k] * gm * act / (60 / STEP), (NR, N))
                    steps.append((hh, j * STEP, k, dem, ret))
            for start_name, b0 in starts.items():
                for T in TRUCKS:
                    name = f'{start_name}+{T}truck'
                    loss_c = np.zeros(NR)
                    for r in range(NR):
                        I = b0.astype(np.int64).copy()
                        trucks = [{'pos': None, 'load': 0, 'free_at': 0} for _ in range(T)]
                        for hh, mm, k, dem, ret in steps:
                            now = hh * 60 + mm
                            in_win = any(w0 <= hh < w1 for w0, w1 in WINDOWS)
                            for tr in trucks:
                                if not in_win or tr['free_at'] > now:
                                    continue
                                lam_now = lam_pol[k] * m_avg
                                deficit = np.where(act, target - I, 0); surplus = np.where(act, I - target, 0)
                                tt_ = travel_min[tr['pos']] if tr['pos'] is not None else np.zeros(N)
                                cost = tt_ + STOP_MIN
                                best = None
                                if tr['load'] > 0:
                                    q = np.minimum(deficit, tr['load']); score = np.where(q > 0, q * lam_now / cost, -1)
                                    s_ = int(np.argmax(score))
                                    if score[s_] > 0:
                                        best = ('drop', s_, int(q[s_]))
                                if best is None and tr['load'] < CAP:
                                    q = np.minimum(surplus, CAP - tr['load']); score = np.where(q > 0, q / cost, -1)
                                    s_ = int(np.argmax(score))
                                    if score[s_] > 0:
                                        best = ('pick', s_, int(q[s_]))
                                if best is None:
                                    tr['free_at'] = now + STEP; continue
                                kind, s_, q = best
                                I[s_] += q if kind == 'drop' else -q; tr['load'] += -q if kind == 'drop' else q
                                tr['pos'] = s_; tr['free_at'] = now + int(cost[s_])
                            D = dem[r]; short = np.maximum(D - I, 0); I = I - (D - short)
                            if short.any():
                                for g_ in np.unique(group[short > 0]):
                                    if g_ < 0:
                                        continue
                                    idx = np.flatnonzero(group == g_); need = short[idx].sum(); avail = I[idx].sum(); take = min(need, avail)
                                    loss_c[r] += need - take
                                    if take > 0:
                                        I[idx] -= np.floor(I[idx] * take / avail).astype(np.int64)
                            I = I + ret[r]
                    acc.setdefault(name, []).append(loss_c)
            n_days += 1
            print(lab, np.datetime64(int(d), 'D'), {k_: round(float(np.mean(v[-1])), 0) for k_, v in acc.items()}, flush=True)
        res = {'days': n_days}
        base = np.sum(acc['P0+0truck'], 0)
        for name, v in acc.items():
            per = np.sum(v, 0) / n_days; red = (base - np.sum(v, 0)) / base
            res[name] = {'catchment_loss_per_day': round(float(per.mean()), 1), 'reduction_vs_P0_static': round(float(red.mean()), 3),
                         'ci95': [round(float(x), 3) for x in np.quantile(red, [.025, .975])]}
        R[lab] = res; print(json.dumps(res, indent=1), flush=True)
    json.dump(R, open('data/h10_dynamic.json' if WINDOWS[0][0] == 10 else 'data/h10_dynamic_prepeak.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
