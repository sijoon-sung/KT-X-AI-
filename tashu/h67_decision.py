"""H6·H7: 수요 기반 목표 재고 vs 현행 배치, 시뮬레이션 평가 (관측 트립 재현 금지).

수요 생성기: 시험 기간(2026-02/03) 자체에서 추정한 축소 요율(d=0.7, κ=30) × 실제 날씨 → 포아송 실현 R개 (대여·반납 독립).
정책(시험 기간을 보지 않음): 학습 기간 요율 × 전년 같은 달 대당 강도 × 운영 대수 × 날씨.
  P0 현행     : 복원 재고(하한·상한 평균)의 06시 값
  P1 대여소   : 대여소별 결품 기대값 곡선의 한계이득 탐욕 배분 (총 대수 = 현행)
  P2 도보권   : 300m 도보권을 한 대여소로 묶어 배분 후 λ 비례로 나눔
손실 정의:  (a) 대여소 결품 = 그 대여소 재고 0 이면 손실
            (b) 도보권 결품 = 300m 안 어느 대여소에도 재고 없을 때만 손실 (있으면 그곳에서 대여)
"""
import json
import sys
import numpy as np

import baseline_udf as B
import structured_rates as SR

RAIN = [(0, 0.0), (0.0, 1.0), (1.0, 5.0), (5.0, 1e9)]
TEMP = [(-99, 0), (0, 10), (10, 20), (20, 28), (28, 99)]
NR = 300          # 실현 수
WALK_KM = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
EPS = (0.05, 0.10, 0.20)
BAND_H = 2   # 밴드: 앞으로 2시간 안 결품확률 <= eps 가 되는 최소 재고


def rates(z, S, e, t_start, t_end, d=0.7, kappa=30):
    Y = z['rentals'].astype(float); N = Y.shape[1]; key48 = S['key48']
    sel = (z['bin_start'] >= np.datetime64(t_start)) & (z['bin_start'] < np.datetime64(t_end)) & z['ok']
    c = np.zeros((48, N)); h = np.zeros((48, N)); cr = np.zeros((48, N)); hr = np.zeros((48, N))
    for k in range(48):
        s = sel & (key48 == k); a = z['active'][s] & z['cust'][None]
        c[k] = (Y[s] * a).sum(0); h[k] = (a * (e[s] + d * (1 - e[s]))).sum(0)
        cr[k] = (z['returns'][s] * a).sum(0); hr[k] = a.sum(0)
    prior = S['scale'][None, :] * S['tshape']
    lam = (c + kappa * prior) / (h + kappa)
    mu = (cr + kappa * prior * (cr.sum() / max(c.sum(), 1))) / (hr + kappa)
    return lam, mu


def smin_table(lam48, mu48, daytype, mult, b_max=40):
    """s_min[eps][h, s]: h시에 재고 b 로 시작해 BAND_H 시간 안에 0 에 닿을 확률 <= eps 인 최소 b."""
    N = lam48.shape[1]; out = {e_: np.zeros((24, N), int) for e_ in EPS}
    for h in range(6, 22):
        b = np.repeat(np.arange(b_max + 1), N)
        P = np.zeros((len(b), B.K + 1)); P[np.arange(len(b)), b] = 1
        for hh in range(h, min(h + BAND_H, 22)):
            k = hh * 2 + daytype
            lam = np.tile(lam48[k] * mult, b_max + 1); mu = np.tile(mu48[k] * mult, b_max + 1)
            for _ in range(60):
                down = P * (lam * B.DT)[:, None]; down[:, 0] = 0
                up = P * (mu * B.DT)[:, None]; up[:, B.K] = 0; up[:, 0] = 0      # 0 은 흡수 상태
                Pn = P - down - up; Pn[:, :-1] += down[:, 1:]; Pn[:, 1:] += up[:, :-1]; P = Pn
        p_abs = P[:, 0].reshape(b_max + 1, N)
        for e_ in EPS:
            okb = p_abs <= e_
            out[e_][h] = np.where(okb.any(0), okb.argmax(0), b_max + 1)
    return out


def main():
    z = SR.load(); S = SR.structure(z); W = SR.weather_by_bin(z)
    N = z['rentals'].shape[1]; key48 = S['key48']
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    lam_pol, mu_pol = rates(z, S, e, '2025-08-01', '2026-02-01')              # 정책용 (학습)
    GEN_D = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
    lam_gen, mu_gen = rates(z, S, e, '2026-02-01', '2026-04-01', d=GEN_D)     # 생성기 (시험 기간 자체)
    # 날씨 계수 (structured_rates 절차)
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
    fac = lambda bins, v: [float(obs[s].sum() / ex[s].sum()) for lo, hi in bins for s in [ok & ((v <= 0) if hi == 0.0 else ((v > lo) & (v <= hi)))]]
    rf, tf = fac(RAIN, W['precipitation'][ts]), fac(TEMP, W['temperature_2m'][ts])
    wf = lambda r, t: next(f for (lo, hi), f in zip(RAIN, rf) if (r <= 0 if hi == 0.0 else lo < r <= hi)) * next(f for (lo, hi), f in zip(TEMP, tf) if lo < t <= hi)
    month_w = {}
    for ym in np.unique(z['ym']):
        s_ = np.flatnonzero((z['ym'] == ym) & z['ok6'])
        if len(s_) >= 100:
            month_w[str(ym)] = float(np.mean([wf(W['precipitation'][t], W['temperature_2m'][t]) for t in s_]))
    def pol_mult(t):
        ym = str(z['ym'][t]); prev = f'{int(ym[:4]) - 1}-{ym[5:]}'
        return SR.season_level(S, ym, SR.BIKES[ym]) * wf(W['precipitation'][t], W['temperature_2m'][t]) / month_w[prev]
    def gen_mult(t):
        ym = str(z['ym'][t])
        return wf(W['precipitation'][t], W['temperature_2m'][t]) / month_w[ym]

    # 도보권: 수요 큰 대여소부터 허브, 300m 안이면 합류
    with np.load('data/processed/tashu_60min.npz') as f:
        dist = f['dist_km']; lat = f['lat']
    cust = z['cust'] & (lat > 0)
    order = np.argsort(-S['scale'])
    group = np.full(N, -1); hubs = []
    for s in order:
        if not cust[s]:
            continue
        near = [h for h in hubs if dist[s, h] <= WALK_KM]
        if near:
            group[s] = group[min(near, key=lambda h: dist[s, h])]
        else:
            group[s] = len(hubs); hubs.append(s)
    G = len(hubs)
    R = {'catchments': G, 'stations_in_multi_station_catchment': int((np.bincount(group[group >= 0])[group[group >= 0]] > 1).sum())}

    # 결품 기대값 곡선 (정책 요율, 요일유형 × 월평균 배율)
    def curves(lam48, mu48, daytype, mult, b_max=40):
        B._F_CACHE.clear()
        return B.lost_curve(lam48 * mult, mu48 * mult, daytype, b_max=b_max)
    tests = {'feb_2026': ('2026-02-01', '2026-03-01'), 'mar_2026': ('2026-03-01', '2026-04-01')}
    rng = np.random.default_rng(0)
    out = {}
    for lab, (a, b) in tests.items():
        days = np.unique(z['day'][(z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b))])
        # 월평균 정책 배율로 곡선 계산 (평일/휴일)
        t_any = np.flatnonzero((z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b)) & z['ok6'])
        m_avg = float(np.mean([pol_mult(t) for t in t_any]))
        f_st = {dt: curves(lam_pol, mu_pol, dt, m_avg) for dt in (0, 1)}
        # 도보권 집계 곡선: 그룹 합 λ, μ 를 한 대여소처럼
        lam_g = np.zeros((48, G)); mu_g = np.zeros((48, G))
        for g in range(G):
            idx = np.flatnonzero(group == g); lam_g[:, g] = lam_pol[:, idx].sum(1); mu_g[:, g] = mu_pol[:, idx].sum(1)
        B.K_saved = B.K
        GB = 150
        f_g = {dt: curves(lam_g, mu_g, dt, m_avg, b_max=GB) for dt in (0, 1)}
        smin = {dt: smin_table(lam_pol, mu_pol, dt, m_avg) for dt in (0, 1)}
        acc = {p: {'station_loss': [], 'catchment_loss': []} for p in ('P0_current', 'P1_station', 'P2_catchment')}
        band = {p: {e_: [0.0, 0] for e_ in EPS} for p in ('P0_current', 'P1_station', 'P2_catchment')}
        n_days = 0
        for d in days:
            t_idx = np.flatnonzero((z['day'] == d) & (z['hour'] >= 6) & (z['hour'] < 22))
            if len(t_idx) < 16 or not z['ok'][t_idx].all():
                continue
            t0 = t_idx[0]; dt = int(z['daytype'][t0])
            act = z['active'][t0] & cust
            cur = ((z['inv_start'][t0] + z['inv_ub'][t0]) / 2).round().astype(int); cur[~act] = 0
            total = int(cur.sum())
            # P1
            f = f_st[dt].copy(); f[~act] = 0
            gain = f[:, :-1] - f[:, 1:]; alloc1 = np.zeros(N, int); nxt = gain[:, 0].copy(); nxt[~act] = -1
            for _ in range(total):
                s = int(np.argmax(nxt))
                if nxt[s] <= 0:
                    break
                alloc1[s] += 1; nxt[s] = gain[s, alloc1[s]] if alloc1[s] < 40 else -1
            # P2: 그룹 단위 배분 (b_max=40 은 그룹에 부족 → 곡선 범위 넘으면 마지막 이득 유지)
            fg = f_g[dt].copy(); act_g = np.array([act[group == g].any() for g in range(G)]); fg[~act_g] = 0
            gg = fg[:, :-1] - fg[:, 1:]; alloc_g = np.zeros(G, int); nxg = gg[:, 0].copy(); nxg[~act_g] = -1
            for _ in range(total):
                g_ = int(np.argmax(nxg))
                if nxg[g_] <= 0:
                    break
                alloc_g[g_] += 1; nxg[g_] = gg[g_, alloc_g[g_]] if alloc_g[g_] < GB else -1
            alloc2 = np.zeros(N, int)
            for g_ in range(G):
                idx = np.flatnonzero((group == g_) & act)
                if not len(idx) or alloc_g[g_] == 0:
                    continue
                w = lam_pol[:, idx].sum(0); w = w / w.sum() if w.sum() > 0 else np.ones(len(idx)) / len(idx)
                base = np.floor(w * alloc_g[g_]).astype(int); rem = alloc_g[g_] - base.sum()
                base[np.argsort(-(w * alloc_g[g_] - base))[:rem]] += 1; alloc2[idx] = base
            # 시뮬레이션: R 실현, 시간별 포아송
            policies = {'P0_current': cur, 'P1_station': alloc1, 'P2_catchment': alloc2}
            inv = {p: np.tile(v.astype(np.int64), (NR, 1)) for p, v in policies.items()}
            loss_s = {p: np.zeros(NR) for p in policies}; loss_c = {p: np.zeros(NR) for p in policies}
            for t in t_idx:
                k = key48[t]; gm = gen_mult(t); hh = int(z['hour'][t])
                for p in policies:
                    for e_ in EPS:
                        sm = smin[dt][e_][hh]
                        band[p][e_][0] += float((inv[p][:, act] >= sm[act]).mean()); band[p][e_][1] += 1
                dem = rng.poisson(lam_gen[k] * gm * act, (NR, N)); ret = rng.poisson(mu_gen[k] * gm * act, (NR, N))
                for p in policies:
                    I = inv[p]
                    short = np.maximum(dem - I, 0)
                    loss_s[p] += short.sum(1)
                    I = I - (dem - short)
                    # 도보권 재분배: 부족분을 같은 그룹의 남은 재고에서 충당
                    if short.any():
                        for g_ in np.unique(group[short.any(0)]):
                            if g_ < 0:
                                continue
                            idx = np.flatnonzero(group == g_)
                            need = short[:, idx].sum(1); avail = I[:, idx].sum(1)
                            take = np.minimum(need, avail)
                            loss_c[p] += need - take
                            # 남은 재고에서 비례로 차감
                            tot = np.maximum(avail, 1)
                            I[:, idx] = I[:, idx] - np.floor(I[:, idx] * (take / tot)[:, None]).astype(np.int64)
                    I = I + ret
                    inv[p] = I
            for p in policies:
                acc[p]['station_loss'].append(loss_s[p]); acc[p]['catchment_loss'].append(loss_c[p])
            n_days += 1
        res = {'days': n_days, 'walk_km': WALK_KM}
        for p in band:
            res[f'{p}/band_satisfaction'] = {str(e_): round(v[0] / max(v[1], 1), 4) for e_, v in band[p].items()}
        for p in acc:
            for kind in ('station_loss', 'catchment_loss'):
                per_real = np.sum(acc[p][kind], 0) / n_days
                res[f'{p}/{kind}'] = {'per_day_mean': round(float(per_real.mean()), 1), 'ci95': [round(float(x), 1) for x in np.quantile(per_real, [.025, .975])]}
        for p in ('P1_station', 'P2_catchment'):
            for kind in ('station_loss', 'catchment_loss'):
                d_ = (np.sum(acc['P0_current'][kind], 0) - np.sum(acc[p][kind], 0)) / np.sum(acc['P0_current'][kind], 0)
                res[f'{p} vs P0 reduction/{kind}'] = {'mean': round(float(d_.mean()), 3), 'ci95': [round(float(x), 3) for x in np.quantile(d_, [.025, .975])]}
        out[lab] = res
        print(lab, json.dumps(res, indent=1), flush=True)
    R.update(out)
    R['generator_d'] = GEN_D
    json.dump(R, open(f'data/h67_decision_d{GEN_D}_w{WALK_KM}.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
