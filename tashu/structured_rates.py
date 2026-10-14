"""구조화 요율  λ(s,h,d,t) = scale[s] × shape[type(s),h,d] × season_per_bike[month] × level(t) × weather(t)

1) 날씨: 학습 기간 도시 시간별 잔차(실제/구조 예측)를 강수·기온 구간으로 설명 → 2026-02/03 검증 (M6)
2) 운영 대수 대리: 복원 재고 합계(과거) 와 API 재고 합계(9월) 를 활동 자전거 수와 비교
3) 9월 API 결품 AUC: 표 요율(앞 시험) vs 구조화 요율
"""
import json
import numpy as np
from math import lgamma

import baseline_udf as B

gammaln = np.vectorize(lgamma)
NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')
BIKES = {'2024-08': 2810, '2024-09': 2635, '2024-10': 2710, '2024-11': 2747, '2024-12': 2350, '2025-01': 2127,
         '2025-02': 2339, '2025-03': 2773, '2025-04': 3356, '2025-05': 3535, '2025-06': 3860, '2025-07': 4001,
         '2025-08': 3868, '2025-09': 4047, '2025-10': 4058, '2025-11': 3845, '2026-01': 3498, '2026-02': 4057, '2026-03': 4944}
BASE_MONTHS = ('2025-08', '2025-09', '2025-10', '2025-11', '2026-01')


def load():
    z = B.load('data/processed/tashu_60min.npz')
    z['ok6'] = z['ok'] & (z['hour'] >= 6) & (z['hour'] < 22)
    z['ym'] = z['bin_start'].astype('datetime64[M]')
    with np.load('data/processed/tashu_60min.npz') as f:
        z['exposure_full'] = f['exposure'] >= 0.999
    return z


def weather_by_bin(z):
    d = json.load(open('data/external/daejeon_hourly_openmeteo.json'))
    h = d['hourly']
    wt = np.array(h['time'], 'datetime64[s]')
    idx = {t: i for i, t in enumerate(wt.astype('datetime64[h]').astype(np.int64))}
    bh = z['bin_start'].astype('datetime64[h]').astype(np.int64)
    j = np.array([idx.get(t, -1) for t in bh])
    out = {}
    for k in ('precipitation', 'temperature_2m', 'wind_speed_10m'):
        v = np.array(h[k], float)
        out[k] = np.where(j >= 0, v[np.maximum(j, 0)], np.nan)
    return out


def structure(z):
    """규모, 유형 모양, 전년 같은 달 대당 강도(월별), 기준 수준."""
    Y = z['rentals'].astype(float)
    full = z['exposure_full'] & z['active'] & z['ok6'][:, None] & z['cust'][None]
    key48 = z['hour'] * 2 + z['daytype']
    T, N = Y.shape
    recent = (z['bin_start'] >= np.datetime64('2025-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok6']
    c48 = np.zeros((48, N)); h48 = np.zeros((48, N))
    for k in range(48):
        sel = recent & (key48 == k); m = full[sel]
        c48[k] = (Y[sel] * m).sum(0); h48[k] = m.sum(0)
    slot_ok = h48.sum(1) > 0
    scale = c48.sum(0) / np.maximum(h48.sum(0), 1)
    # 유형 모양 (pattern_rates 와 같은 k-means 4)
    rate48 = c48 / np.maximum(h48, 1)
    big = (c48.sum(0) >= 500) & (h48[slot_ok].min(0) >= 10) & z['cust']
    X = (rate48[:, big] / rate48[:, big].sum(0, keepdims=True)).T
    rng = np.random.default_rng(0); cen = X[rng.choice(len(X), 4, replace=False)]
    for _ in range(30):
        lab = np.argmin(((X[:, None] - cen[None]) ** 2).sum(2), 1)
        cen = np.array([X[lab == j].mean(0) if (lab == j).any() else cen[j] for j in range(4)])
    types = np.full(N, -1); types[np.flatnonzero(big)] = lab
    city48 = c48.sum(1) / np.maximum(h48.sum(1), 1)
    tshape = np.tile((city48 / city48[slot_ok].mean())[:, None], (1, N))
    for j in range(4):
        idx = np.flatnonzero(types == j)
        rj = c48[:, idx].sum(1) / np.maximum(h48[:, idx].sum(1), 1)
        tshape[:, idx] = (rj / rj[slot_ok].mean())[:, None]
    # 월별 대당 강도 (도시 시간당 대여 / 활동 자전거), 기준 = BASE_MONTHS 평균
    per_bike = {}
    for ym in np.unique(z['ym']):
        sel = (z['ym'] == ym) & z['ok6']
        if sel.sum() >= 100 and str(ym) in BIKES:
            per_bike[str(ym)] = Y[sel][:, z['cust']].sum(1).mean() / BIKES[str(ym)]
    base_pb = np.mean([per_bike[m] for m in BASE_MONTHS])
    base_bikes = np.mean([BIKES[m] for m in BASE_MONTHS])
    return dict(scale=scale, tshape=tshape, key48=key48, per_bike=per_bike, base_pb=base_pb, base_bikes=base_bikes, full=full, types=types)


def season_level(S, ym_str, bikes):
    """전년 같은 달 대당 강도 / 기준  ×  운영 대수 / 기준."""
    prev = f'{int(ym_str[:4]) - 1}-{ym_str[5:]}'
    s = S['per_bike'].get(prev, S['base_pb']) / S['base_pb']
    return s * (bikes / S['base_bikes'])


def main():
    z = load(); S = structure(z); W = weather_by_bin(z)
    Y = z['rentals'].astype(float); full = S['full']; key48 = S['key48']
    R = {}

    # ------------------------------------------------ 1) 날씨 계수: 학습 기간 도시 잔차
    train = (z['bin_start'] >= np.datetime64('2024-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok6']
    ts = np.flatnonzero(train)
    obs = np.zeros(len(ts)); exp_ = np.zeros(len(ts))
    for i, t in enumerate(ts):
        m = full[t]
        ym = str(z['ym'][t])
        if ym not in BIKES:
            continue
        mu = S['scale'] * S['tshape'][key48[t]] * (S['per_bike'][ym] / S['base_pb']) * (BIKES[ym] / S['base_bikes'])
        obs[i] = Y[t][m].sum(); exp_[i] = mu[m].sum()
    ok = exp_ > 0
    rain = W['precipitation'][ts]; temp = W['temperature_2m'][ts]
    rain_bins = [(0, 0.0), (0.0, 1.0), (1.0, 5.0), (5.0, 1e9)]
    temp_bins = [(-99, 0), (0, 10), (10, 20), (20, 28), (28, 99)]
    def factor(bins, v):
        f = {}
        for lo, hi in bins:
            sel = ok & ((v > lo) & (v <= hi) if lo != hi else (v <= 0))
            if lo == 0 and hi == 0.0:
                sel = ok & (v <= 0)
            f[f'{lo}-{hi}'] = (float(obs[sel].sum() / exp_[sel].sum()), int(sel.sum()))
        return f
    rf = factor(rain_bins, rain); tf = factor(temp_bins, temp)
    R['weather_factor_rain'] = rf; R['weather_factor_temp'] = tf
    # 두 요인 결합 (곱, 비 없음·기온 10-20 기준으로 정규화한 상대값): 간단히 곱셈 모델
    def wfac(t):
        r = W['precipitation'][t]; tp = W['temperature_2m'][t]
        fr = next(v[0] for (lo, hi), v in zip(rain_bins, rf.values()) if (r <= 0 if hi == 0.0 else lo < r <= hi))
        ft = next(v[0] for (lo, hi), v in zip(temp_bins, tf.values()) if lo < tp <= hi)
        return fr * ft
    # 계절 안에 이미 평균 날씨가 들어 있으므로, 같은 달 평균 날씨 계수로 나눠 상대화
    month_mean_w = {}
    for ym in np.unique(z['ym']):
        sel = np.flatnonzero((z['ym'] == ym) & z['ok6'])
        if len(sel) >= 100:
            month_mean_w[str(ym)] = float(np.mean([wfac(t) for t in sel]))
    R['month_mean_weather_factor'] = {k: round(v, 3) for k, v in month_mean_w.items()}

    # ------------------------------------------------ 검증: M4 vs M6(+날씨), 2026-02/03
    def evaluate(name, mu_fn):
        out = {}
        for lab, (a, b) in (('feb_2026', ('2026-02-01', '2026-03-01')), ('mar_2026', ('2026-03-01', '2026-04-01'))):
            sel = np.flatnonzero((z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b)) & z['ok6'])
            ll = n = pred = act = 0.0
            for t in sel:
                m = full[t]
                if not m.any():
                    continue
                mu = np.maximum(mu_fn(t)[m], 1e-6); y = Y[t][m]
                ll += (y * np.log(mu) - mu - gammaln(y + 1)).sum(); n += m.sum(); pred += mu.sum(); act += y.sum()
            out[lab] = {'ll_per_cell': round(ll / n, 4), 'pred_over_actual': round(pred / act, 3)}
        R.setdefault('models', {})[name] = out; print(name, json.dumps(out), flush=True)

    def m4(t):
        ym = str(z['ym'][t])
        return S['scale'] * S['tshape'][key48[t]] * season_level(S, ym, BIKES[ym])
    def m6(t):
        ym = str(z['ym'][t]); prev = f'{int(ym[:4]) - 1}-{ym[5:]}'
        return m4(t) * wfac(t) / month_mean_w.get(prev, 1.0)
    evaluate('M4_structure', m4)
    evaluate('M6_structure_x_weather', m6)

    # ------------------------------------------------ 2) 운영 대수 대리: 복원 재고 합계 / API 재고 합계
    docked = {}
    for ym in np.unique(z['ym']):
        sel = (z['ym'] == ym) & z['ok6']
        if sel.sum() >= 100 and str(ym) in BIKES:
            inv = (z['inv_start'][sel].astype(float) + z['inv_ub'][sel]) / 2
            docked[str(ym)] = float(inv[:, z['cust']].sum(1).mean())
    R['docked_vs_active'] = {k: {'reconstructed_docked_mean': round(v, 0), 'active_bikes': BIKES[k], 'ratio': round(v / BIKES[k], 3)} for k, v in docked.items()}
    ts_api, sids, stock = B.load_api(r'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    api_total = np.nansum(stock, 1)
    R['api_total_stock'] = {'mean': float(api_total.mean()), 'min': float(api_total.min()), 'max': float(api_total.max())}
    ratio_docked = np.mean([docked[m] / BIKES[m] for m in BASE_MONTHS])
    api_bikes = api_total.mean() / ratio_docked            # 과거 (복원 재고 / 활동 자전거) 비율로 환산
    R['api_implied_active_bikes'] = float(api_bikes)

    # ------------------------------------------------ 3) 9월 API 결품 AUC: 표 vs 구조화
    lam_tab, mu_tab = B.fit_rates(z, '2025-08-01', '2026-02-01')
    lam_tab = np.nan_to_num(lam_tab['observed']); mu_tab = np.nan_to_num(mu_tab)
    sl = season_level(S, '2026-09', api_bikes)
    lam_struct = S['scale'][None, :] * S['tshape'] * sl
    # 반납 요율도 같은 배율 (도시 반납 합 = 대여 합)
    mu_struct = mu_tab * (lam_struct.sum() / max(lam_tab.sum(), 1e-9))
    R['sept_season_level_factor'] = float(sl)
    # 날씨 시간별 배율 (API 시각)
    d = json.load(open('data/external/daejeon_hourly_openmeteo.json')); h = d['hourly']
    widx = {t: i for i, t in enumerate(np.array(h['time'], 'datetime64[s]').astype('datetime64[h]').astype(np.int64))}
    pr = np.array(h['precipitation'], float); tp = np.array(h['temperature_2m'], float)
    def wf_api(t):
        i = widx.get(int(t.astype('datetime64[h]').astype(np.int64)), None)
        if i is None:
            return 1.0
        r_, t_ = pr[i], tp[i]
        fr = next(v[0] for (lo, hi), v in zip(rain_bins, rf.values()) if (r_ <= 0 if hi == 0.0 else lo < r_ <= hi))
        ft = next(v[0] for (lo, hi), v in zip(temp_bins, tf.values()) if lo < t_ <= hi)
        return fr * ft / month_mean_w.get('2025-09', 1.0)
    st = z['stations'].astype(str); idx_model = {s: i for i, s in enumerate(st)}
    m_idx = np.array([idx_model[s] for s in sids if s in idx_model]); a_idx = np.array([j for j, s in enumerate(sids) if s in idx_model])
    res = {k: ([], []) for k in ('stock_only', 'table', 'structure', 'structure_x_weather')}
    N = len(st)
    for t in range(len(ts_api) - 6):
        hour = int(ts_api[t].astype('datetime64[h]').astype(np.int64) % 24)
        span = (ts_api[t + 6] - ts_api[t]).astype('timedelta64[s]').astype(int)
        if not (7 <= hour <= 22 and 3300 <= span <= 3900):
            continue
        cur = stock[t, a_idx]; fut = stock[t + 1:t + 7, a_idx]
        ok_ = ~np.isnan(cur) & ~np.isnan(fut).any(0) & (cur > 0)
        if not ok_.any():
            continue
        event = fut.min(0) == 0
        dow = (ts_api[t].astype('datetime64[D]').astype(np.int64) + 3) % 7; k = hour * 2 + int(dow >= 5)
        b = np.zeros(N, int); b[m_idx] = np.clip(cur, 0, B.K).astype(int)
        for name, lam_, mu_ in (('table', lam_tab[k], mu_tab[k]), ('structure', lam_struct[k], mu_struct[k]),
                                ('structure_x_weather', lam_struct[k] * wf_api(ts_api[t]), mu_struct[k] * wf_api(ts_api[t]))):
            P = np.zeros((N, B.K + 1)); P[np.arange(N), b] = 1
            P, _ = B.step_hour(P, lam_, mu_)
            res[name][0].append(P[:, 0][m_idx][ok_]); res[name][1].append(event[ok_])
        res['stock_only'][0].append(-cur[ok_]); res['stock_only'][1].append(event[ok_])
    R['api_auc'] = {k: round(B.auc(np.concatenate(v[0]), np.concatenate(v[1])), 4) for k, v in res.items()}
    print(json.dumps({k: v for k, v in R.items() if k != 'models'}, ensure_ascii=False, indent=1))
    json.dump(R, open('data/structured_rates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
