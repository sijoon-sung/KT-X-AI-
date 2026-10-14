"""두 기준선 비교: 포아송 요율 → 재고 마르코프 체인 → 결품 기대값 최소 배분.

기준 1 (observed) : 요율 = 관측 대여 건수 / 시간            (Hulot 2018, Gammelli 2022 방식, 결품 보정 없음)
기준 2 (exposure) : 요율 = 관측 대여 건수 / 재고가 있던 시간  (Freund 2022, Mellou 2019 방식, 분모는 자전거번호 복원 노출도)
반납 요율은 두 기준 모두 관측 건수 / 시간 (용량이 제약이 아니므로 반납 절단 없음).

요율 단위: 대여소 × 시간대(24) × 평일/휴일. 학습 2025-08-01 ~ 2026-02-01.
결정: 06시 시작 재고를 도시 전체 대수 고정 아래에서 배분해 그날(06~22시) 결품 기대값을 최소화 (한계 이득 탐욕 배분, UDF 볼록성).

평가 A (과거): 2026-02, 03 각 날에 대해 06시 배분 → 관측 대여·반납 흐름을 1시간 단위로 흘려 결품 대여 수 집계.
              비교: 현행(복원 재고), 기준 1 배분, 기준 2 배분. 총 대수는 현행과 같게 맞춤.
평가 B (API) : 2026-09 API 재고에서 출발해 1시간 뒤 결품 확률(마르코프 체인) → AUC. 비교: 재고만.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path

import numpy as np

K = 80          # 재고 상태 상한 (관측 최대 이상)
DT = 1 / 60     # 체인 시간 단위: 1분
NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


# ---------------------------------------------------------------- data
def load(path):
    with np.load(path) as f:
        z = {k: f[k] for k in ('rentals', 'returns', 'exposure', 'exposure_ub', 'inv_start',
                               'inv_ub', 'active', 'valid', 'closed', 'bin_start', 'stations')}
    bs = z['bin_start']
    tb = bs.astype('datetime64[h]').astype(np.int64)
    z['hour'] = tb % 24
    z['dow'] = ((tb // 24) + 3) % 7
    z['day'] = tb // 24
    z['daytype'] = (z['dow'] >= 5).astype(np.int64)     # TODO 공휴일은 주말로 취급 안 함
    z['ok'] = z['valid'] & ~z['closed']
    z['cust'] = ~np.isin(z['stations'].astype(str), NON_CUSTOMER)
    return z


def fit_rates(z, start, end):
    sel = z['ok'] & (z['bin_start'] >= np.datetime64(start)) & (z['bin_start'] < np.datetime64(end))
    T, N = z['rentals'].shape
    key = z['hour'] * 2 + z['daytype']                       # 48 slots
    lam = {}
    for name in ('observed', 'exposure'):
        num = np.zeros((48, N)); den = np.zeros((48, N))
        e_lb, e_ub, y = z['exposure'], z['exposure_ub'], z['rentals']
        # demand.py 의 e_lbx: 대여가 있는데 하한 노출이 거의 0 → 재배치 도착 미상, 상한 사용
        e = np.where((y > 0) & (e_lb < 0.1), np.maximum(e_ub, e_lb), e_lb)
        for t in np.flatnonzero(sel):
            a = z['active'][t]
            num[key[t]] += np.where(a, y[t], 0)
            den[key[t]] += a * (1.0 if name == 'observed' else e[t])
        lam[name] = num / np.maximum(den, 1e-9)
        lam[name][den < 20] = np.nan                          # 노출 20시간 미만이면 추정 안 함
    num = np.zeros((48, N)); den = np.zeros((48, N))
    for t in np.flatnonzero(sel):
        a = z['active'][t]
        num[key[t]] += np.where(a, z['returns'][t], 0)
        den[key[t]] += a
    mu = num / np.maximum(den, 1e-9)
    mu[den < 20] = np.nan
    return lam, mu


# ---------------------------------------------------------------- Markov chain
def step_hour(P, lam, mu):
    """P[N, K+1] 상태 분포를 1시간 전진. lam, mu [N] 시간당 요율. 반환: 새 P, 이 시간의 결품 기대값."""
    lost = np.zeros(P.shape[0]); Kp = P.shape[1] - 1
    for _ in range(int(round(1 / DT))):
        p0 = P[:, 0]
        lost += lam * DT * p0                      # 재고 0에서 온 손님
        down = P * (lam * DT)[:, None]; down[:, 0] = 0          # k -> k-1
        up = P * (mu * DT)[:, None]; up[:, Kp] = 0              # k -> k+1
        Pn = P - down - up
        Pn[:, :-1] += down[:, 1:]
        Pn[:, 1:] += up[:, :-1]
        P = Pn
    return P, lost


def expected_lost(b, lam48, mu48, daytype, h0=6, h1=22):
    """시작 재고 b[N] (정수) 로 h0시에 출발, h1시까지 결품 기대값 [N]."""
    N = len(b)
    P = np.zeros((N, K + 1)); P[np.arange(N), np.clip(b, 0, K)] = 1
    total = np.zeros(N)
    for h in range(h0, h1):
        lam = lam48[h * 2 + daytype]; mu = mu48[h * 2 + daytype]
        P, lost = step_hour(P, lam, mu)
        total += lost
    return total


_F_CACHE = {}


def lost_curve(lam48, mu48, daytype, b_max=40, h0=6, h1=22):
    """대여소별 결품 기대값 곡선 f[s, b], b = 0..b_max. 모든 b 를 한 번에 전진."""
    key = (id(lam48), daytype)
    if key in _F_CACHE:
        return _F_CACHE[key]
    N = lam48.shape[1]
    b = np.repeat(np.arange(b_max + 1), N)                 # [(b_max+1)*N]
    Kp = max(K, b_max + 20)                                 # 상태 상한은 시작 재고보다 넉넉히
    P = np.zeros((len(b), Kp + 1)); P[np.arange(len(b)), b] = 1
    lam_rep = lambda h: np.tile(lam48[h * 2 + daytype], b_max + 1)
    mu_rep = lambda h: np.tile(mu48[h * 2 + daytype], b_max + 1)
    total = np.zeros(len(b))
    for h in range(h0, h1):
        P, lost = step_hour(P, lam_rep(h), mu_rep(h))
        total += lost
    f = total.reshape(b_max + 1, N).T
    _F_CACHE[key] = f
    return f


def allocate(total_bikes, lam48, mu48, daytype, active, b_max=40):
    """탐욕 한계 배분: 자전거 1대를 추가했을 때 결품 감소가 가장 큰 대여소부터. UDF 볼록 → 최적."""
    f = lost_curve(lam48, mu48, daytype, b_max).copy()
    f[~active] = 0
    N = f.shape[0]
    gain = f[:, :-1] - f[:, 1:]                   # gain[s, b] = f(b) - f(b+1) >= 0, 감소
    alloc = np.zeros(N, int)
    # 힙 없이: 각 대여소의 다음 한계이득을 반복적으로 최대 선택 (N=1425, total~4000 → 충분)
    nxt = gain[:, 0].copy(); nxt[~active] = -1
    for _ in range(int(total_bikes)):
        s = int(np.argmax(nxt))
        if nxt[s] <= 0:
            break
        alloc[s] += 1
        nxt[s] = gain[s, alloc[s]] if alloc[s] < b_max else -1
    return alloc, f


# ---------------------------------------------------------------- 평가 A: 과거 재현
def replay_day(b0, z, t_idx):
    """06시 재고 b0 에서 관측 대여·반납을 1시간 단위로 흘림. 반환: 결품으로 못 나간 대여 수 [N]."""
    inv = b0.astype(float).copy(); lost = np.zeros(len(b0))
    for t in t_idx:
        r = z['rentals'][t]; q = z['returns'][t]
        short = np.maximum(r - inv, 0)
        lost += short
        inv = inv - (r - short) + q
    return lost


def eval_history(z, lam, mu, start, end):
    days = np.unique(z['day'][(z['bin_start'] >= np.datetime64(start)) & (z['bin_start'] < np.datetime64(end))])
    out = {k: [] for k in ('current', 'observed', 'exposure', 'total_bikes', 'rentals')}
    n_days = 0
    for d in days:
        t_idx = np.flatnonzero((z['day'] == d) & (z['hour'] >= 6) & (z['hour'] < 22))
        if len(t_idx) < 16 or not z['ok'][t_idx].all():
            continue
        t0 = t_idx[0]
        act = z['active'][t0] & z['cust']
        cur = ((z['inv_start'][t0] + z['inv_ub'][t0]) / 2).round().astype(int)     # 복원 재고 중간값
        cur[~act] = 0
        total = int(cur.sum())
        daytype = int(z['daytype'][t0])
        res = {'current': cur}
        for name in ('observed', 'exposure'):
            alloc, _ = allocate(total, lam[name], mu, daytype, act)
            res[name] = alloc
        for name, b0 in res.items():
            lost = replay_day(b0, z, t_idx)
            out[name].append(float(lost[act].sum()))
        out['total_bikes'].append(total)
        out['rentals'].append(float(z['rentals'][t_idx][:, act].sum()))
        n_days += 1
        print(f'  {np.datetime64(int(d), "D")}  bikes={total}  rentals={out["rentals"][-1]:.0f}  '
              f'lost cur={out["current"][-1]:.0f} obs={out["observed"][-1]:.0f} exp={out["exposure"][-1]:.0f}', flush=True)
    return {'days': n_days,
            **{k: {'lost_total': float(np.sum(v)), 'lost_per_day': float(np.mean(v)),
                   'lost_share_of_rentals': float(np.sum(v) / max(np.sum(out['rentals']), 1))}
               for k, v in out.items() if k in ('current', 'observed', 'exposure')},
            'rentals_total': float(np.sum(out['rentals']))}


# ---------------------------------------------------------------- 평가 B: API
def auc(scores, labels):
    labels = np.asarray(labels, bool)
    p, n = labels.sum(), (~labels).sum()
    if not p or not n:
        return None
    order = np.argsort(scores, kind='mergesort')
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    # 평균 순위 (동점 처리)
    s = scores[order]; i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2
        i = j + 1
    return float((ranks[labels].sum() - p * (p + 1) / 2) / (p * n))


def load_api(path):
    rows = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for x in r:
            rows.setdefault(x['ts'], {})[x['station_id']] = float(x['stock'])
    times = sorted(rows)
    sids = sorted({s for t in times for s in rows[t]})
    stock = np.array([[rows[t].get(s, np.nan) for s in sids] for t in times], np.float32)
    ts = np.array(times, 'datetime64[s]')
    return ts, sids, stock


def eval_api(z, lam, mu, api_path):
    ts, sids, stock = load_api(api_path)
    st = z['stations'].astype(str)
    idx_model = {s: i for i, s in enumerate(st)}
    m_idx = np.array([idx_model[s] for s in sids if s in idx_model])
    a_idx = np.array([j for j, s in enumerate(sids) if s in idx_model])
    lam = dict(lam)
    lam['exposure_cap1.5'] = np.minimum(lam['exposure'], 1.5 * lam['observed'])
    ratio = np.nanmax(lam['exposure'] / np.maximum(lam['observed'], 1e-9), axis=0)   # 대여소별 최대 보정 배율
    res = {k: {'score': [], 'event': [], 'ratio': []} for k in ('stock_only', 'observed', 'exposure', 'exposure_cap1.5')}
    for t in range(0, len(ts) - 6):
        hour = int((ts[t].astype('datetime64[h]').astype(np.int64)) % 24)
        span = (ts[t + 6] - ts[t]).astype('timedelta64[s]').astype(int)
        if not (7 <= hour <= 22 and 3300 <= span <= 3900):
            continue
        cur = stock[t, a_idx]
        fut = stock[t + 1:t + 7, a_idx]
        ok = ~np.isnan(cur) & ~np.isnan(fut).any(0) & (cur > 0)
        if not ok.any():
            continue
        event = fut.min(0) == 0
        dow = ((ts[t].astype('datetime64[D]').astype(np.int64)) + 3) % 7
        daytype = int(dow >= 5)
        N = len(st)
        b = np.zeros(N, int); b[m_idx] = np.clip(cur, 0, K).astype(int)
        for name in ('observed', 'exposure', 'exposure_cap1.5'):
            P = np.zeros((N, K + 1)); P[np.arange(N), b] = 1
            P, _ = step_hour(P, lam[name][hour * 2 + daytype], mu[hour * 2 + daytype])
            p0 = P[:, 0][m_idx]                    # 1시간 뒤 재고 0일 확률 (근사: 구간 내 0 도달과 다름)
            res[name]['score'].append(p0[ok]); res[name]['event'].append(event[ok]); res[name]['ratio'].append(ratio[m_idx][ok])
        res['stock_only']['score'].append(-cur[ok]); res['stock_only']['event'].append(event[ok]); res['stock_only']['ratio'].append(ratio[m_idx][ok])
    out = {}
    for k, v in res.items():
        s = np.concatenate(v['score']); e = np.concatenate(v['event']); r = np.concatenate(v['ratio'])
        out[k] = {'auc': auc(s, e), 'cells': int(len(e)), 'event_rate': float(e.mean()),
                  'auc_by_max_ratio': {lab: auc(s[m], e[m]) for lab, m in
                                       (('<1.5', r < 1.5), ('1.5-3', (r >= 1.5) & (r < 3)), ('>=3', r >= 3))},
                  'cells_by_max_ratio': {lab: int(m.sum()) for lab, m in
                                         (('<1.5', r < 1.5), ('1.5-3', (r >= 1.5) & (r < 3)), ('>=3', r >= 3))}}
    out['windows'] = len(res['stock_only']['score'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--api', default=r'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    ap.add_argument('--train-start', default='2025-08-01')
    ap.add_argument('--train-end', default='2026-02-01')
    ap.add_argument('--out', default='data/baseline_udf.json')
    ap.add_argument('--skip-api', action='store_true')
    ap.add_argument('--skip-history', action='store_true')
    a = ap.parse_args()
    z = load(a.data)
    print('fitting rates ...', flush=True)
    lam, mu = fit_rates(z, a.train_start, a.train_end)
    ratio = lam['exposure'] / np.maximum(lam['observed'], 1e-9)
    okr = ~np.isnan(lam['observed']) & ~np.isnan(lam['exposure']) & (lam['observed'] > 0)
    summary = {'train': [a.train_start, a.train_end],
               'rate_ratio_exposure_over_observed': {q: float(np.nanquantile(ratio[okr], q)) for q in (0.1, 0.5, 0.9)},
               'city_rate_sum_observed': float(np.nansum(lam['observed'])),
               'city_rate_sum_exposure': float(np.nansum(lam['exposure']))}
    print(json.dumps(summary, indent=1), flush=True)
    # 노출 시간 부족으로 보정 요율을 못 구한 칸은 관측 요율로 대체 (0 으로 두면 '수요 없음'이 되어 버림)
    lam['exposure'] = np.where(np.isnan(lam['exposure']), lam['observed'], lam['exposure'])
    lam = {k: np.nan_to_num(v, nan=0.0) for k, v in lam.items()}
    mu = np.nan_to_num(mu, nan=0.0)
    for label, s, e in () if a.skip_history else (('feb_2026', '2026-02-01', '2026-03-01'), ('mar_2026', '2026-03-01', '2026-04-01')):
        print(f'history {label}', flush=True)
        summary[label] = eval_history(z, lam, mu, s, e)
        print(json.dumps(summary[label], indent=1), flush=True)
    if not a.skip_api:
        print('api ...', flush=True)
        summary['api_sept_2026'] = eval_api(z, lam, mu, a.api)
        print(json.dumps(summary['api_sept_2026'], indent=1), flush=True)
    Path(a.out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
