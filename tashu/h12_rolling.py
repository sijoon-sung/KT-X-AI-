"""H12: 연속 시뮬레이션 — 밤마다 예산 M 만큼 목표(P2) 쪽으로 옮기면 며칠 뒤 어디에 머무는가.
- 2026-02-01 ~ 03-31 을 이어서 시뮬레이션 (장애일은 건너뜀). 시작 재고 = 2/1 06시 복원값.
- 반납은 OD 로 대수 보존: 대여소 i 의 대여가 확률 OD[i,·] 로 도착 (이용시간은 1시간 구간 안에서 처리, 단순화).
- 도보권 대체: 빈 대여소 손님은 같은 300m 권역의 재고를 사용 (권역 손실만 집계).
- 밤 (22시): 예산 M 대까지 대여소 손실곡선 한계이득 순으로 이동 (h11 과 같은 규칙). 목표는 그날 요일유형의 P2 배분.
- 지표: 주차별 권역 손실/일, 밤마다 실제 옮긴 대수, 마지막 2주 평균 감소율 (M=0 대비).
"""
import json
import numpy as np

import baseline_udf as B
import structured_rates as SR
import h67_decision as H

NR = 60
BUDGETS = (0, 100, 200, 400)


def main():
    z = SR.load(); S = SR.structure(z); W = SR.weather_by_bin(z)
    N = z['rentals'].shape[1]; key48 = S['key48']
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    lam_pol, mu_pol = H.rates(z, S, e, '2025-08-01', '2026-02-01')
    lam_gen, _ = H.rates(z, S, e, '2026-02-01', '2026-04-01')
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
        dist = f['dist_km']; lat = f['lat']; od = f['od'].astype(float)
    cust = z['cust'] & (lat > 0)
    od = od * cust[:, None] * cust[None]; od[np.arange(N), np.arange(N)] += 1e-9
    P_od = od / np.maximum(od.sum(1, keepdims=True), 1e-9)
    order = np.argsort(-S['scale']); group = np.full(N, -1); hubs = []
    for s in order:
        if not cust[s]:
            continue
        near = [h for h in hubs if dist[s, h] <= 0.3]
        group[s] = group[min(near, key=lambda h: dist[s, h])] if near else len(hubs)
        if not near:
            hubs.append(s)
    G = len(hubs)
    t_all = np.flatnonzero((z['bin_start'] >= np.datetime64('2026-02-01')) & (z['bin_start'] < np.datetime64('2026-04-01')) & z['ok6'])
    m_avg = float(np.mean([pol_mult(t) for t in t_all]))
    B._F_CACHE.clear(); f_st = {dt: B.lost_curve(lam_pol * m_avg, mu_pol * m_avg, dt, b_max=60) for dt in (0, 1)}
    lam_g = np.zeros((48, G)); mu_g = np.zeros((48, G))
    for g in range(G):
        idx = np.flatnonzero(group == g); lam_g[:, g] = lam_pol[:, idx].sum(1); mu_g[:, g] = mu_pol[:, idx].sum(1)
    B._F_CACHE.clear(); f_g = {dt: B.lost_curve(lam_g * m_avg, mu_g * m_avg, dt, b_max=150) for dt in (0, 1)}

    def target_for(total, dt, act):
        fg = f_g[dt].copy(); act_g = np.array([act[group == g].any() for g in range(G)]); fg[~act_g] = 0
        gg = fg[:, :-1] - fg[:, 1:]; alloc_g = np.zeros(G, int); nxg = gg[:, 0].copy(); nxg[~act_g] = -1
        for _ in range(int(total)):
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
        return target

    def night_move(I, target, act, dt, M):
        f = f_st[dt]; b_max = f.shape[1] - 1
        lost_at = lambda v: f[np.arange(N), np.clip(v, 0, b_max)]
        I = I.copy(); k = 0
        surplus = np.where(act, I - target, 0); deficit = np.where(act, target - I, 0)
        while k < M and (surplus > 0).any() and (deficit > 0).any():
            gd = np.where(deficit > 0, lost_at(I) - lost_at(I + 1), -1); cp = np.where(surplus > 0, lost_at(I - 1) - lost_at(I), 1e9)
            j = int(np.argmax(gd)); i_ = int(np.argmin(cp))
            if gd[j] - cp[i_] <= 0:
                break
            I[j] += 1; I[i_] -= 1; deficit[j] -= 1; surplus[i_] -= 1; k += 1
        return I, k

    days = np.unique(z['day'][t_all])
    t_first = np.flatnonzero((z['day'] == days[0]) & (z['hour'] == 6))[0]
    act0 = z['active'][t_first] & cust
    cur0 = ((z['inv_start'][t_first] + z['inv_ub'][t_first]) / 2).round().astype(int); cur0[~act0] = 0
    rng = np.random.default_rng(3)
    R = {'NR': NR, 'budgets': BUDGETS, 'start_total_bikes': int(cur0.sum())}
    state = {M: np.tile(cur0.astype(np.int64), (NR, 1)) for M in BUDGETS}
    daily = {M: [] for M in BUDGETS}; moved = {M: [] for M in BUDGETS}; day_labels = []
    snap = {M: [] for M in BUDGETS}   # 06시 상태 (실현 평균) 기록
    rec = []
    for d in days:
        t_idx = np.flatnonzero((z['day'] == d) & (z['hour'] >= 6) & (z['hour'] < 22))
        if len(t_idx) < 16 or not z['ok'][t_idx].all():
            continue
        t0 = t_idx[0]; dt = int(z['daytype'][t0]); act = z['active'][t0] & cust
        loss = {M: np.zeros(NR) for M in BUDGETS}
        for M in BUDGETS:
            snap[M].append(state[M].mean(0))
        rec.append(((z['inv_start'][t0] + z['inv_ub'][t0]) / 2) * act)
        # 공통 난수: 같은 실현에 대해 정책 비교 (수요 발생은 공통, 도착지는 다항 추출)
        for t in t_idx:
            k_ = key48[t]; gm = gen_mult(t)
            dem = rng.poisson(lam_gen[k_] * gm * act, (NR, N))
            for M in BUDGETS:
                I = state[M]; short = np.maximum(dem - I, 0); served = dem - short; I = I - served
                if short.any():
                    for g_ in np.unique(group[short.any(0)]):
                        if g_ < 0:
                            continue
                        idx = np.flatnonzero(group == g_); need = short[:, idx].sum(1); avail = I[:, idx].sum(1)
                        take = np.minimum(need, avail); loss[M] += need - take
                        extra = np.floor(I[:, idx] * (take / np.maximum(avail, 1))[:, None]).astype(np.int64)
                        I[:, idx] -= extra; served[:, idx] += extra
                # 반납: 출발지별 served 를 OD 로 도착지에 배분 (기대값 라운딩으로 근사: 대수 보존)
                arrivals = served @ P_od
                ret = np.floor(arrivals).astype(np.int64); frac = arrivals - ret
                ret += (rng.random(ret.shape) < frac).astype(np.int64)
                # 대수 보존 보정 (라운딩 오차)
                diff = served.sum(1) - ret.sum(1)
                for r_ in np.flatnonzero(diff != 0):
                    j = rng.choice(N, abs(int(diff[r_])), p=P_od.sum(0) / P_od.sum())
                    np.add.at(ret[r_], j, np.sign(diff[r_]))
                state[M] = I + ret
        for M in BUDGETS:
            daily[M].append(loss[M].copy())
            # 밤 이동 (실현별로 상태가 다르므로 실현마다 수행; 목표는 총 대수 기준)
            tot = int(np.round(state[M].sum(1).mean()))
            dt_next = int(z['daytype'][t_idx[-1] + 8]) if t_idx[-1] + 8 < len(z['daytype']) else dt
            target = target_for(tot, dt_next, act)
            mv = []
            for r_ in range(NR):
                state[M][r_], k = night_move(state[M][r_], target, act, dt_next, M); mv.append(k)
            moved[M].append(float(np.mean(mv)))
        day_labels.append(str(np.datetime64(int(d), 'D')))
        print(day_labels[-1], {M: round(float(daily[M][-1].mean()), 0) for M in BUDGETS}, 'moved', {M: round(moved[M][-1]) for M in BUDGETS}, flush=True)
    for M in BUDGETS:
        L = np.array(daily[M]); base = np.array(daily[0])
        R[f'budget_{M}'] = {'loss_per_day_week1': round(float(L[:7].mean()), 1), 'loss_per_day_last14': round(float(L[-14:].mean()), 1),
                            'reduction_last14_vs_M0': round(float(1 - L[-14:].sum() / base[-14:].sum()), 3),
                            'reduction_all_vs_M0': round(float(1 - L.sum() / base.sum()), 3),
                            'moved_per_night_mean': round(float(np.mean(moved[M])), 1), 'moved_last14': round(float(np.mean(moved[M][-14:])), 1)}
    rec = np.array(rec); last = slice(-14, None)
    R['world_check_last14'] = {}
    for M in BUDGETS:
        sm = np.array(snap[M])[last]; rm = rec[last]; a = act
        R['world_check_last14'][f'budget_{M}'] = {
            'empty_share_06h_sim': float((sm[:, a] < 0.5).mean()), 'empty_share_06h_reconstructed': float((rm[:, a] < 0.5).mean()),
            'corr_station_mean_stock': float(np.corrcoef(sm[:, a].mean(0), rm[:, a].mean(0))[0, 1]),
            'gini_sim': float(np.abs(np.subtract.outer(sm[:, a].mean(0), sm[:, a].mean(0))).mean() / (2 * sm[:, a].mean())),
            'gini_reconstructed': float(np.abs(np.subtract.outer(rm[:, a].mean(0), rm[:, a].mean(0))).mean() / (2 * rm[:, a].mean()))}
    print(json.dumps({k: v for k, v in R.items() if k.startswith('budget') or k == 'world_check_last14'}, indent=1))
    json.dump(R, open('data/h12_rolling.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
