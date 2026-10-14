"""축소(shrinkage) 요율:  λ[s,k] = (count[s,k] + κ · prior[s,k]) / (hours[s,k] + κ)
  count/hours : 최근 6개월(2025-08~2026-01) 대여소×슬롯(시각×평일/휴일) 관측 (재고 충분 칸)
  prior       : 구조화 요율 scale[s] × shape[type(s), k]
  κ           : 가상 관측 시간. κ=0 → 표 그대로, κ=∞ → 구조화 요율
그 위에 계절(전년 같은 달 대당 강도) × 운영 대수 × 날씨 를 곱한다.
평가: 2026-02/03 재고 충분 칸 포아송 LL, 9월 API 1시간 결품 AUC.
"""
import json
import numpy as np
from math import lgamma

import baseline_udf as B
import structured_rates as SR

gammaln = np.vectorize(lgamma)
RAIN = [(0, 0.0), (0.0, 1.0), (1.0, 5.0), (5.0, 1e9)]
TEMP = [(-99, 0), (0, 10), (10, 20), (20, 28), (28, 99)]


def main():
    z = SR.load(); S = SR.structure(z); W = SR.weather_by_bin(z)
    Y = z['rentals'].astype(float); full = S['full']; key48 = S['key48']; N = Y.shape[1]
    recent = (z['bin_start'] >= np.datetime64('2025-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok']   # 전체 시간
    # 요율 재료 세 가지: full(재고 내내 있던 칸만) / observed(전체 활동 시간) / exposure(노출시간 보정)
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    mats = {}
    # damped: 비어 있던 시간의 수요 강도를 재고 있을 때의 d 배로 봄 → 분모 = e + d(1-e). d=1 이면 observed, d=0 이면 exposure
    for name in ('observed', 'exposure', 'damped0.7', 'damped0.5', 'damped0.85'):
        c48 = np.zeros((48, N)); h48 = np.zeros((48, N))
        dmp = {'observed': 1.0, 'exposure': 0.0}.get(name, float(name.replace('damped', '')) if name.startswith('damped') else None)
        for k in range(48):
            sel = recent & (key48 == k)
            a = z['active'][sel] & z['cust'][None]
            c48[k] = (Y[sel] * a).sum(0); h48[k] = (a * (e[sel] + dmp * (1 - e[sel]))).sum(0)
        mats[name] = (c48, h48)
    prior = S['scale'][None, :] * S['tshape']

    # 날씨 계수 (structured_rates 와 동일 절차)
    train = (z['bin_start'] >= np.datetime64('2024-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok6']
    ts = np.flatnonzero(train); obs = np.zeros(len(ts)); exp_ = np.zeros(len(ts))
    for i, t in enumerate(ts):
        ym = str(z['ym'][t])
        if ym not in SR.BIKES:
            continue
        mu = prior[key48[t]] * SR.season_level(S, ym, SR.BIKES[ym]) if False else \
            S['scale'] * S['tshape'][key48[t]] * (S['per_bike'][ym] / S['base_pb']) * (SR.BIKES[ym] / S['base_bikes'])
        m = full[t]; obs[i] = Y[t][m].sum(); exp_[i] = mu[m].sum()
    ok = exp_ > 0; rain = W['precipitation'][ts]; temp = W['temperature_2m'][ts]
    def fac(bins, v):
        out = []
        for lo, hi in bins:
            sel = ok & ((v <= 0) if hi == 0.0 else ((v > lo) & (v <= hi)))
            out.append(float(obs[sel].sum() / exp_[sel].sum()))
        return out
    rf, tf = fac(RAIN, rain), fac(TEMP, temp)
    def wf(r, t):
        fr = next(f for (lo, hi), f in zip(RAIN, rf) if (r <= 0 if hi == 0.0 else lo < r <= hi))
        ft = next(f for (lo, hi), f in zip(TEMP, tf) if lo < t <= hi)
        return fr * ft
    month_w = {}
    for ym in np.unique(z['ym']):
        sel = np.flatnonzero((z['ym'] == ym) & z['ok6'])
        if len(sel) >= 100:
            month_w[str(ym)] = float(np.mean([wf(W['precipitation'][t], W['temperature_2m'][t]) for t in sel]))

    # API 준비
    ts_api, sids, stock = B.load_api(r'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    st = z['stations'].astype(str); idx_model = {s: i for i, s in enumerate(st)}
    m_idx = np.array([idx_model[s] for s in sids if s in idx_model]); a_idx = np.array([j for j, s in enumerate(sids) if s in idx_model])
    d = json.load(open('data/external/daejeon_hourly_openmeteo.json')); h = d['hourly']
    widx = {t: i for i, t in enumerate(np.array(h['time'], 'datetime64[s]').astype('datetime64[h]').astype(np.int64))}
    pr = np.array(h['precipitation'], float); tp = np.array(h['temperature_2m'], float)
    windows = []
    for t in range(len(ts_api) - 6):
        hour = int(ts_api[t].astype('datetime64[h]').astype(np.int64) % 24)
        span = (ts_api[t + 6] - ts_api[t]).astype('timedelta64[s]').astype(int)
        if not (7 <= hour <= 22 and 3300 <= span <= 3900):
            continue
        cur = stock[t, a_idx]; fut = stock[t + 1:t + 7, a_idx]
        ok_ = ~np.isnan(cur) & ~np.isnan(fut).any(0) & (cur > 0)
        if not ok_.any():
            continue
        dow = (ts_api[t].astype('datetime64[D]').astype(np.int64) + 3) % 7
        i = widx.get(int(ts_api[t].astype('datetime64[h]').astype(np.int64)))
        w = wf(pr[i], tp[i]) / month_w['2025-09'] if i is not None else 1.0
        windows.append((hour * 2 + int(dow >= 5), cur, ok_, fut.min(0) == 0, w))
    R = {'weather_rain': rf, 'weather_temp': tf, 'api_windows': len(windows),
         'api_rain_hours_share': float(np.mean([pr[widx[int(ts_api[t].astype('datetime64[h]').astype(np.int64))]] > 0
                                                for t in range(len(ts_api)) if int(ts_api[t].astype('datetime64[h]').astype(np.int64)) in widx]))}
    lam_tab, mu_tab = B.fit_rates(z, '2025-08-01', '2026-02-01')
    mu_tab = np.nan_to_num(mu_tab)
    sl_sept = SR.season_level(S, '2026-09', S['base_bikes'])        # 수준 대리는 불안정하므로 기준 수준 유지

    def evaluate(kappa, basis):
        c48, h48 = mats[basis]
        lam = (c48 + kappa * prior) / np.maximum(h48 + kappa, 1e-9) if np.isfinite(kappa) else prior
        out = {}
        for lab, (a, b) in (('feb_2026', ('2026-02-01', '2026-03-01')), ('mar_2026', ('2026-03-01', '2026-04-01'))):
            sel = np.flatnonzero((z['bin_start'] >= np.datetime64(a)) & (z['bin_start'] < np.datetime64(b)) & z['ok6'])
            ll = n = 0.0
            for t in sel:
                m = full[t]
                if not m.any():
                    continue
                ym = str(z['ym'][t]); prev = f'{int(ym[:4]) - 1}-{ym[5:]}'
                mult = SR.season_level(S, ym, SR.BIKES[ym]) * wf(W['precipitation'][t], W['temperature_2m'][t]) / month_w[prev]
                mu = np.maximum(lam[key48[t]][m] * mult, 1e-6); y = Y[t][m]
                ll += (y * np.log(mu) - mu - gammaln(y + 1)).sum(); n += m.sum()
            out[lab] = round(ll / n, 4)
        # API
        mu_s = mu_tab                                              # 반납 요율은 관측 표 그대로 (같은 수준)
        sc, ev = [], []
        for k, cur, ok_, event, w in windows:
            b = np.zeros(N, int); b[m_idx] = np.clip(cur, 0, B.K).astype(int)
            P = np.zeros((N, B.K + 1)); P[np.arange(N), b] = 1
            P, _ = B.step_hour(P, lam[k] * sl_sept * w, mu_s[k] * sl_sept * w)
            sc.append(P[:, 0][m_idx][ok_]); ev.append(event[ok_])
        out['api_auc'] = round(B.auc(np.concatenate(sc), np.concatenate(ev)), 4)
        return out

    for basis in ('damped0.85', 'damped0.7', 'damped0.5'):
        for kappa in (0, 10, 30, 100, 300, np.inf):
            r_ = evaluate(kappa, basis); r_ = {k: float(v) for k, v in r_.items()}
            R.setdefault('by_kappa', {}).setdefault(basis, {})[str(kappa)] = r_
            print(basis, kappa, r_, flush=True)
    json.dump(R, open('data/shrink_rates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != 'by_kappa'}, indent=1))


if __name__ == '__main__':
    main()


def paired_bootstrap():
    """창(1시간) 단위 짝지은 AUC 차이: exposure κ=10 vs observed κ=30 vs stock_only."""
    import shrink_rates as me
    z = SR.load(); S = SR.structure(z)
    Y = z['rentals'].astype(float); key48 = S['key48']; N = Y.shape[1]
    recent = (z['bin_start'] >= np.datetime64('2025-08-01')) & (z['bin_start'] < np.datetime64('2026-02-01')) & z['ok']
    e_lb, e_ub = z['exposure'], z['exposure_ub']
    e = np.where((z['rentals'] > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
    prior = S['scale'][None, :] * S['tshape']
    lam = {}
    for name, kappa in (('observed', 30), ('exposure', 10)):
        c48 = np.zeros((48, N)); h48 = np.zeros((48, N))
        for k in range(48):
            sel = recent & (key48 == k); a = z['active'][sel] & z['cust'][None]
            c48[k] = (Y[sel] * a).sum(0); h48[k] = (a * (1.0 if name == 'observed' else e[sel])).sum(0)
        lam[name] = (c48 + kappa * prior) / np.maximum(h48 + kappa, 1e-9)
    _, mu_tab = B.fit_rates(z, '2025-08-01', '2026-02-01'); mu_tab = np.nan_to_num(mu_tab)
    ts_api, sids, stock = B.load_api(r'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    st = z['stations'].astype(str); idx_model = {s: i for i, s in enumerate(st)}
    m_idx = np.array([idx_model[s] for s in sids if s in idx_model]); a_idx = np.array([j for j, s in enumerate(sids) if s in idx_model])
    per = {k: [] for k in ('observed', 'exposure', 'stock_only')}
    for t in range(len(ts_api) - 6):
        hour = int(ts_api[t].astype('datetime64[h]').astype(np.int64) % 24)
        span = (ts_api[t + 6] - ts_api[t]).astype('timedelta64[s]').astype(int)
        if not (7 <= hour <= 22 and 3300 <= span <= 3900):
            continue
        cur = stock[t, a_idx]; fut = stock[t + 1:t + 7, a_idx]
        ok_ = ~np.isnan(cur) & ~np.isnan(fut).any(0) & (cur > 0)
        event = fut.min(0) == 0
        if not ok_.any() or event[ok_].all() or not event[ok_].any():
            continue
        dow = (ts_api[t].astype('datetime64[D]').astype(np.int64) + 3) % 7; k = hour * 2 + int(dow >= 5)
        b = np.zeros(N, int); b[m_idx] = np.clip(cur, 0, B.K).astype(int)
        for name in ('observed', 'exposure'):
            P = np.zeros((N, B.K + 1)); P[np.arange(N), b] = 1
            P, _ = B.step_hour(P, lam[name][k], mu_tab[k])
            per[name].append(B.auc(P[:, 0][m_idx][ok_], event[ok_]))
        per['stock_only'].append(B.auc(-cur[ok_], event[ok_]))
    rng = np.random.default_rng(0); out = {'windows': len(per['stock_only'])}
    for a, b_ in (('exposure', 'observed'), ('observed', 'stock_only'), ('exposure', 'stock_only')):
        d = np.array(per[a]) - np.array(per[b_])
        bs = rng.choice(d, size=(5000, len(d)), replace=True).mean(1)
        out[f'{a}-{b_}'] = {'mean_gain': round(float(d.mean()), 4), 'windows_improved': int((d > 0).sum()),
                            'bootstrap_95pct': [round(float(x), 4) for x in np.quantile(bs, [0.025, 0.975])]}
    print(json.dumps(out, indent=1)); json.dump(out, open('data/shrink_rates_bootstrap.json', 'w'), indent=1)
