"""대여기록만으로 절단된 대여 수요를 복원하는 모델.

재고가 0 이면 손님이 와도 빌릴 수 없어서 관측 대여량 y 는 실제 수요 Y 보다 작다.
전처리에서 구한 exposure e (구간 중 재고 > 0 이었던 시간 비율)를 이용한다.

    Y[t,n] ~ NegBin(mean = μ[t,n], dispersion = r[n])        (1시간 동안 온 손님 수)
    log μ[t,n] = H[n, hour(t), weekend(t)] + g[t] + M[n, month(t)]

    H : 대여소별 시간대×요일 패턴. 대여소 임베딩을 OD 이동 그래프 위 GNN 으로 평활해
        이용이 적거나 자주 비는 대여소가 흐름으로 연결된 이웃에게서 정보를 빌려온다.
    g : 시간 구간별 도시 전체 효과 (날씨·공휴일·행사 등 공통 충격을 흡수)
    M : 대여소별 월 효과 (신설·확장 등 추세)

손실 (--loss):
    naive    : y ~ NB(μ)          절단 무시                           → 과소추정
    drop     : 절단 구간 버리고 y ~ NB(μ)                          → 수요 큰 시간대가 빠져 과소추정
    tobit    : 비절단 log p(y) + 절단 log P(Y ≥ y)
               (y 가 '재고 한도'일 때만 옳다. 재고가 잠깐 0 이 된 구간 전반에 쓰면 μ 를 무한정 키움)
    exposure : y ~ NB(μ·e)        재고가 있던 시간에 온 손님은 모두 빌렸다고 가정  ← 제안 방법

구간의 실제 수요 추정 = y + μ·(1−e)   (재고가 없던 시간에 왔을 손님의 기댓값)

사용법:
    python demand.py validate-sim    # 검증 A: 진짜 수요를 아는 가상 세계에서 방법 비교
    python demand.py validate-real   # 검증 B: 실데이터 보정 (재고 충분 구간 hold-out 예측)
    python demand.py fit --exposure mid   # 실데이터 잠재 수요 추정 → data/demand/
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(os.cpu_count())
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_MAX = 60   # 생존확률 합산 상한 (시간당 대여 60건 초과는 사실상 없음)
E_MIN = 1 / 60  # 대여가 있었는데 노출이 0 으로 복원된 셀(재고 복원 오차)의 최소 노출 = 1분
NON_CUSTOMER = ['ST0001', 'ST1220', 'ST1198']  # 타슈관제센터 정비대기 / 타슈관제센터 / 남대전 물류센터


# ---------------------------------------------------------------- data
def load(path):
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    bs = z['bin_start']
    T, N = z['rentals'].shape
    days = bs.astype('datetime64[D]')
    hour = (bs - days).astype('timedelta64[h]').astype(np.int64)
    dow = (days.astype(np.int64) + 3) % 7                          # 1970-01-01 = 목요일 → 월=0
    month = bs.astype('datetime64[M]').astype(np.int64)
    month -= month.min()
    mask = z['active'] & z['valid'][:, None]
    mask[:, np.isin(z['stations'], NON_CUSTOMER)] = False
    t, n = np.nonzero(mask)
    y = z['rentals'][t, n].astype(np.int64)
    e_lb, e_ub = z['exposure'][t, n], np.maximum(z['exposure_ub'][t, n], z['exposure'][t, n])
    # 대여가 있는데 하한 노출이 거의 0 → 트럭이 가져다 놓은 자전거(도착 시각 미상)라 하한이 틀린 셀. 상한 사용.
    e_lbx = np.where((y > 0) & (e_lb < 0.1), e_ub, e_lb)
    cells = dict(t=t, n=n, y=y,
                 e_lb=e_lb, e_lbx=e_lbx, e_ub=e_ub, e_mid=(e_lb + e_ub) / 2,
                 inv=z['inv_start'][t, n].astype(np.int64),
                 cap=(z['inv_start'][t, n] + z['returns'][t, n]).astype(np.int64))
    meta = dict(T=T, N=N, hour=hour, weekend=(dow >= 5).astype(np.int64), month=month,
                n_months=int(month.max()) + 1, od=z['od'], lat=z['lat'], lon=z['lon'],
                stations=z['stations'], bin_start=bs)
    return cells, meta


def subset(cells, idx):
    return {k: v[idx] for k, v in cells.items()}


def od_adjacency(od, k=10):
    """대칭화한 OD 이동량에서 대여소별 상위 k 이웃만 남긴 행 정규화 인접행렬 (자기 루프 포함)."""
    a = (od + od.T).astype(np.float32)
    np.fill_diagonal(a, 0)
    keep = np.argsort(-a, axis=1)[:, :k]
    A = np.zeros_like(a)
    rows = np.arange(len(a))[:, None]
    A[rows, keep] = np.log1p(a[rows, keep])
    A += np.eye(len(a), dtype=np.float32)
    return torch.from_numpy(A / A.sum(1, keepdims=True))


# ---------------------------------------------------------------- model
class DemandModel(nn.Module):
    def __init__(self, meta, d=32, init_mean=0.5):
        super().__init__()
        N, T = meta['N'], meta['T']
        self.register_buffer('A', od_adjacency(meta['od']))
        ll = np.stack([meta['lat'], meta['lon']], 1)
        ll = np.nan_to_num((ll - np.nanmean(ll, 0)) / np.nanstd(ll, 0))
        self.register_buffer('coord', torch.tensor(ll, dtype=torch.float32))
        self.emb = nn.Embedding(N, d)
        nn.init.normal_(self.emb.weight, std=0.1)
        self.inp = nn.Linear(d + 2, d)
        self.g1, self.g2 = nn.Linear(d, d), nn.Linear(d, d)
        self.hour = nn.Embedding(24, d)
        self.wk = nn.Embedding(2, d)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.constant_(self.head[-1].bias, float(np.log(init_mean)))
        self.g = nn.Parameter(torch.zeros(T))
        self.M = nn.Parameter(torch.zeros(N, meta['n_months']))
        self.log_r = nn.Parameter(torch.full((N,), float(np.log(2.0))))

    def pattern(self):
        """H[n, hour, weekend] 테이블 (N×24×2) — 셀마다 MLP 를 돌리지 않고 테이블을 조회한다."""
        x = self.inp(torch.cat([self.emb.weight, self.coord], 1))
        x = x + torch.relu(self.A @ self.g1(x))
        x = x + torch.relu(self.A @ self.g2(x))
        h = x[:, None, None, :] + self.hour.weight[None, :, None, :] + self.wk.weight[None, None, :, :]
        return self.head(h).squeeze(-1)

    def log_mu(self, H, t, n, tf):
        return H[n, tf['hour'][t], tf['weekend'][t]] + self.g[t] + self.M[n, tf['month'][t]]


def nb_logpmf(k, mu, r):
    return (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1)
            + r * (torch.log(r) - torch.log(r + mu)) + k * (torch.log(mu) - torch.log(r + mu)))


def nb_below(y, mu, r):
    """P(Y < y), E[Y · 1(Y < y)] — y 는 K_MAX 로 절단."""
    ks = torch.arange(K_MAX, dtype=mu.dtype, device=mu.device)[None]
    p = torch.exp(nb_logpmf(ks, mu[:, None], r[:, None])) * (ks < y[:, None].clamp(max=K_MAX))
    return p.sum(1), (p * ks).sum(1)


def posterior_mean(y, mu, r):
    """E[Y | Y ≥ y] = (μ − E[Y·1(Y<y)]) / P(Y ≥ y)."""
    cdf, part = nb_below(y, mu, r)
    return torch.maximum((mu - part) / (1 - cdf).clamp(min=1e-7), y.to(mu.dtype))


def time_features(meta):
    return {k: torch.from_numpy(meta[k]).to(DEV) for k in ('hour', 'weekend', 'month')}


# ---------------------------------------------------------------- train / predict
def train(cells, meta, loss, e_key='e_mid', steps=1500, batch=500_000, lr=0.03, seed=0, log_every=500):
    """loss ∈ {naive, drop, tobit, exposure}. 절단 표시는 e_key 의 노출 < 1."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y_np, e_np = cells['y'], cells[e_key]
    cens_np = e_np < 1
    if loss == 'drop':
        pool = np.nonzero(~cens_np)[0]
    elif loss == 'exposure':
        pool = np.nonzero((e_np > 0) | (y_np > 0))[0]             # 노출 0 & 대여 0 은 정보 없음
    else:
        pool = np.arange(len(y_np))
    model = DemandModel(meta, init_mean=max(y_np[pool].mean(), 1e-2)).to(DEV)
    tf = time_features(meta)
    t_all, n_all = torch.from_numpy(cells['t']).to(DEV), torch.from_numpy(cells['n']).to(DEV)
    y_all = torch.from_numpy(y_np).float().to(DEV)
    e_all = torch.from_numpy(np.where(y_np > 0, np.maximum(e_np, E_MIN), e_np)).float().to(DEV)
    c_all = torch.from_numpy(cens_np).to(DEV)
    pool_t = torch.from_numpy(pool).to(DEV)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    t0 = time.time()
    for step in range(steps):
        b = pool_t[torch.randint(0, len(pool_t), (batch,), device=DEV, generator=gen)]
        t, n, y = t_all[b], n_all[b], y_all[b]
        mu = torch.exp(model.log_mu(model.pattern(), t, n, tf)).clamp(1e-6, 1e4)
        r = torch.exp(model.log_r[n]).clamp(0.05, 100)
        if loss == 'exposure':
            ll = nb_logpmf(y, mu * e_all[b], r)
        elif loss == 'tobit':
            c = c_all[b]
            ll = torch.zeros_like(mu)
            ll[~c] = nb_logpmf(y[~c], mu[~c], r[~c])
            cy = c & (y > 0)                                       # P(Y ≥ 0) = 1 → 정보 없음
            cdf, _ = nb_below(y[cy], mu[cy], r[cy])
            ll[cy] = torch.log1p(-cdf.clamp(max=1 - 1e-7))
        else:
            ll = nb_logpmf(y, mu, r)
        obj = -ll.mean() + 1e-2 * (model.M ** 2).mean() + 1e-3 * (model.g ** 2).mean()
        opt.zero_grad(); obj.backward(); opt.step(); sched.step()
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'    [{loss}/{e_key}] step {step:5d}  nll {-ll.mean().item():.4f}  {time.time() - t0:.0f}s', flush=True)
    return model


@torch.no_grad()
def predict(model, meta, cells, chunk=2_000_000):
    tf = time_features(meta)
    H = model.pattern()
    mus, rs = [], []
    for i in range(0, len(cells['t']), chunk):
        t, n = torch.from_numpy(cells['t'][i:i + chunk]).to(DEV), torch.from_numpy(cells['n'][i:i + chunk]).to(DEV)
        mus.append(torch.exp(model.log_mu(H, t, n, tf)).clamp(1e-6, 1e4).cpu())
        rs.append(torch.exp(model.log_r[n]).clamp(0.05, 100).cpu())
    return torch.cat(mus), torch.cat(rs)


def estimate(loss, y, e, mu, r):
    """방법별 '이 구간의 실제 수요' 추정치."""
    y = y.astype(np.float64); m = mu.double().numpy()
    if loss == 'exposure':
        return y + m * (1 - e)
    if loss == 'tobit':
        out = y.copy(); c = np.nonzero(e < 1)[0]
        for i in range(0, len(c), 500_000):
            j = c[i:i + 500_000]
            out[j] = posterior_mean(torch.from_numpy(y[j]).float(), mu[j], r[j]).double().numpy()
        return out
    return np.where(e < 1, np.maximum(y, m), y)                    # naive / drop: 절단 구간은 max(관측, μ)


def historical_average(cells, meta, use, e_key):
    """규칙 기반 기준선: 같은 (대여소, 시간대, 주말여부, 월) 의 재고 충분 구간 평균 대여량."""
    t = cells['t']
    key = ((cells['n'] * 24 + meta['hour'][t]) * 2 + meta['weekend'][t]) * meta['n_months'] + meta['month'][t]
    ok = use & (cells[e_key] >= 1)
    s = np.bincount(key[ok], weights=cells['y'][ok], minlength=key.max() + 1)
    c = np.bincount(key[ok], minlength=key.max() + 1)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)[key]


def activity_tier(cells, meta):
    act = np.bincount(cells['n'], weights=cells['y'], minlength=meta['N'])
    return np.digitize(act, np.quantile(act[act > 0], [1 / 3, 2 / 3]))


# ---------------------------------------------------------------- experiments
def validate_sim(args):
    """검증 A: 진짜 수요를 아는 가상 세계.

    실데이터에서 재고가 한 번도 바닥나지 않은(노출 하한 = 1) 셀만 모으면 y = 실제 수요 Y 다.
    여기에 같은 대여소의 실제 재고 한도 분포에서 뽑은 재고 c 를 걸어 절단을 시뮬레이션한다.
    손님 Y 명이 구간 내 균등하게 도착하므로 Y ≥ c 이면 c 번째 손님 도착 시각에 재고가 0 이 되고
    노출 e = c 번째 순서통계량 ~ Beta(c, Y−c+1), 관측 y = c.  (c = 0 이면 e = 0, y = 0)
    """
    cells, meta = load(args.data)
    rng = np.random.default_rng(args.seed)
    world = subset(cells, np.nonzero(cells['e_lb'] >= 1)[0])
    Y = world['y'].copy()
    # 대여소별 실제 절단 셀의 재고 한도 분포에서 추출 (없으면 전체 분포)
    real = cells['e_lb'] < 1
    rc_n, rc_c = cells['n'][real], cells['cap'][real]
    o = np.argsort(rc_n, kind='stable'); rc_n, rc_c = rc_n[o], rc_c[o]
    lo, hi = np.searchsorted(rc_n, world['n']), np.searchsorted(rc_n, world['n'], 'right')
    has = hi > lo
    pick = np.where(has, lo + (rng.random(len(Y)) * np.maximum(hi - lo, 1)).astype(np.int64), 0)
    c = np.where(has, rc_c[np.minimum(pick, len(rc_c) - 1)], rng.choice(rc_c, len(Y)))
    c = c + rng.integers(0, args.cap_slack + 1, len(Y))           # 재고 여유 (절단 강도 조절)
    hit = Y >= c
    e = np.ones(len(Y))
    e[hit & (c == 0)] = 0.0
    k = hit & (c > 0)
    e[k] = rng.beta(c[k], Y[k] - c[k] + 1)
    world['y'] = np.where(hit, c, Y)
    for key in ('e_lb', 'e_mid', 'e_ub'):
        world[key] = e
    print(f'world {len(Y):,} cells, censored {hit.mean():.1%}  '
          f'(true demand {Y.sum():,} → observed {world["y"].sum():,}, {world["y"].sum() / Y.sum():.1%})')

    tier = activity_tier(world, meta)[world['n']]
    results = []

    def score(name, est):
        m = hit
        row = dict(method=name, recovery_censored=float(est[m].sum() / Y[m].sum()),
                   mae_censored=float(np.abs(est[m] - Y[m]).mean()))
        for ti, nm in enumerate(('low', 'mid', 'high')):
            mm = m & (tier == ti)
            row[f'recovery_{nm}'] = float(est[mm].sum() / max(Y[mm].sum(), 1))
        results.append(row)
        print(f'  {name:26s} recovery {row["recovery_censored"]:.3f}  MAE {row["mae_censored"]:.3f}  '
              f'(activity low/mid/high {row["recovery_low"]:.2f}/{row["recovery_mid"]:.2f}/{row["recovery_high"]:.2f})',
              flush=True)

    score('observed (no correction)', world['y'].astype(np.float64))
    ha = historical_average(world, meta, np.ones(len(Y), bool), 'e_mid')
    score('rule: historical avg', np.where(hit, np.maximum(world['y'], np.nan_to_num(ha)), world['y']))
    for loss in ('naive', 'drop', 'tobit', 'exposure'):
        model = train(world, meta, loss, 'e_mid', args.steps, args.batch, seed=args.seed)
        mu, r = predict(model, meta, world)
        score(loss, estimate(loss, world['y'], world['e_mid'], mu, r))
    save(args, 'validation_sim.json', dict(world_cells=int(len(Y)), censored_share=float(hit.mean()),
                                            cap_slack=args.cap_slack, steps=args.steps, results=results))


def validate_real(args):
    """검증 B: 실데이터 보정. 재고가 넉넉하게 '시작한' 낮 시간 셀 일부를 떼어 두고 나머지로 학습 → 예측합/실제합.

    '재고가 끝까지 남은 셀'로 고르면 수요가 적어서 남은 셀이 뽑혀(선택 편향) 올바른 모델도 과대로 보인다.
    구간 시작 재고는 그 구간 수요보다 먼저 정해지므로 이 편향이 없고, 시작 재고 ≥ 3 이면 절단도 드물다.
    """
    cells, meta = load(args.data)
    rng = np.random.default_rng(args.seed)
    hour = meta['hour'][cells['t']]
    cand = np.nonzero((cells['inv'] >= args.min_inv) & (hour >= 7) & (hour < 22))[0]
    hold = np.sort(rng.choice(cand, int(len(cand) * args.hold_frac), replace=False))
    keep = np.ones(len(cells['y']), bool); keep[hold] = False
    tr, te = subset(cells, np.nonzero(keep)[0]), subset(cells, hold)
    tier = activity_tier(cells, meta)[te['n']]
    print(f'train {len(tr["y"]):,}  hold-out {len(te["y"]):,} (시작 재고 ≥ {args.min_inv}, 대여 {te["y"].sum():,}, '
          f'그중 재고 소진 셀 {(te["e_lb"] < 1).mean():.1%})')
    results = []
    runs = [('naive', 'e_lb'), ('drop', 'e_lb'), ('tobit', 'e_lb'),
            ('exposure', 'e_lb'), ('exposure', 'e_lbx'), ('exposure', 'e_mid'), ('exposure', 'e_ub')]
    for loss, ek in runs:
        model = train(tr, meta, loss, ek, args.steps, args.batch, seed=args.seed)
        mu, _ = predict(model, meta, te)
        m = mu.double().numpy(); y = te['y']
        row = dict(method=f'{loss}/{ek}', calibration=float(m.sum() / y.sum()), mae=float(np.abs(m - y).mean()))
        for ti, nm in enumerate(('low', 'mid', 'high')):
            k = tier == ti
            row[f'calibration_{nm}'] = float(m[k].sum() / max(y[k].sum(), 1))
        results.append(row)
        print(f'  {row["method"]:18s} Σμ/Σy {row["calibration"]:.3f}  MAE {row["mae"]:.3f}  '
              f'(activity low/mid/high {row["calibration_low"]:.2f}/{row["calibration_mid"]:.2f}/{row["calibration_high"]:.2f})',
              flush=True)
    save(args, 'validation_real.json', dict(hold_out=int(len(hold)), steps=args.steps, results=results))


def fit(args):
    """실데이터 전체에 exposure 모델을 적합해 잠재 수요와 놓친 수요를 추정.

    수요 수준 μ 는 --exposure 노출로 학습한 모델에서, 놓친 수요 μ·(1−e) 는 '재고가 없던 시간'을
    얼마로 보느냐에 따라 달라지므로 범위로 낸다:
        high = μ·(1 − e_학습노출)   재고 하한 기준 (방치·재배치 자전거를 없는 것으로 봄)
        low  = μ·(1 − e_ub)          재고 상한 기준
    """
    cells, meta = load(args.data)
    ek = f'e_{args.exposure}'
    print(f'cells {len(cells["y"]):,}  exposure={ek}  재고 부족 셀 {(cells[ek] < 1).mean():.1%}')
    model = train(cells, meta, 'exposure', ek, args.steps, args.batch, seed=args.seed)
    mu, r = predict(model, meta, cells)
    y = cells['y'].astype(np.float64)
    m = mu.double().numpy()
    miss = {'high': m * (1 - cells[ek]), 'low': m * (1 - np.maximum(cells['e_ub'], cells[ek]))}
    T, N = meta['T'], meta['N']

    def dense(v):
        a = np.full((T, N), np.nan, np.float32); a[cells['t'], cells['n']] = v; return a

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f'demand_{args.exposure}.npz', mu=dense(m), observed=dense(y),
                        unmet_high=dense(miss['high']), unmet_low=dense(miss['low']),
                        bin_start=meta['bin_start'], stations=meta['stations'])
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / f'model_{args.exposure}.pt')

    n = cells['n']
    cnt = np.maximum(np.bincount(n, minlength=N), 1)
    obs_s = np.bincount(n, weights=y, minlength=N)
    lo_s = np.bincount(n, weights=miss['low'], minlength=N)
    hi_s = np.bincount(n, weights=miss['high'], minlength=N)
    short_s = np.bincount(n, weights=cells[ek] < 1, minlength=N) / cnt
    names = {}
    st_csv = Path(args.data).parent / 'stations.csv'
    if st_csv.exists():
        with open(st_csv, encoding='utf-8-sig') as fh:
            names = {r_['station_id']: (r_['name'], r_['gu']) for r_ in csv.DictReader(fh)}
    # 근거량: 재고가 있던 총 시간. 이게 적으면 μ 는 이웃·패턴에서 외삽한 값이라 순위에서 뒤로 보낸다.
    evid_s = np.bincount(n, weights=cells[ek], minlength=N)
    reliable = evid_s >= args.min_evidence_h
    order = np.lexsort((-lo_s, ~reliable))                         # 신뢰 대여소 먼저, 보수적(하단) 추정 순
    served = np.bincount(n, minlength=N) > 0
    with open(out / f'station_unmet_{args.exposure}.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh)
        w.writerow(['rank', 'station_id', 'name', 'gu', 'observed_rentals', 'unmet_low', 'unmet_high',
                    'unmet_share_low', 'unmet_share_high', 'stockout_hour_share', 'stocked_hours', 'reliable'])
        for rank, i in enumerate(order[served[order]], 1):
            s = meta['stations'][i]; nm, gu = names.get(s, ('', ''))
            w.writerow([rank, s, nm, gu, int(obs_s[i]), round(lo_s[i], 1), round(hi_s[i], 1),
                        round(lo_s[i] / max(obs_s[i] + lo_s[i], 1e-9), 4),
                        round(hi_s[i] / max(obs_s[i] + hi_s[i], 1e-9), 4), round(short_s[i], 4),
                        round(evid_s[i]), bool(reliable[i])])
    hour = meta['hour'][cells['t']]

    def share(k, sel=slice(None)):
        return float(miss[k][sel].sum() / (y[sel].sum() + miss[k][sel].sum()))

    by_hour = {h: [round(share('low', hour == h), 4), round(share('high', hour == h), 4)]
               for h in range(24) if (hour == h).any()}
    by_lo = np.argsort(-lo_s)
    top = by_lo[:max(1, N // 10)]
    rel_cells = reliable[n]
    summary = dict(exposure=args.exposure, steps=args.steps, observed_total=int(y.sum()),
                   reliable_stations=int(reliable.sum()), unreliable_stations=int((served & ~reliable).sum()),
                   unmet_share_reliable_only=[share('low', rel_cells), share('high', rel_cells)],
                   unmet_total=[round(float(miss['low'].sum())), round(float(miss['high'].sum()))],
                   unmet_share=[share('low'), share('high')],
                   top10pct_stations_share_of_unmet_low=float(lo_s[top].sum() / lo_s.sum()),
                   unmet_share_by_hour_low_high=by_hour)
    save(args, f'summary_{args.exposure}.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'unmet_share_by_hour_low_high'}, indent=2, ensure_ascii=False))


def save(args, name, obj):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'saved {out / name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['validate-sim', 'validate-real', 'fit'])
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--out', default='data/demand')
    ap.add_argument('--exposure', choices=['lb', 'lbx', 'mid', 'ub'], default='lbx',
                    help='fit 학습 노출: lbx = 재고 하한 + 재배치 유입 셀만 상한 (기본) / lb / mid / ub')
    ap.add_argument('--min-evidence-h', type=float, default=300,
                    help='fit: 재고가 있던 시간이 이보다 적은 대여소는 근거 부족으로 표시')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=2_000_000)
    ap.add_argument('--hold-frac', type=float, default=0.1)
    ap.add_argument('--min-inv', type=int, default=3, help='validate-real: hold-out 셀의 최소 시작 재고')
    ap.add_argument('--cap-slack', type=int, default=1, help='validate-sim: 재고 한도에 더할 최대 여유')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    {'validate-sim': validate_sim, 'validate-real': validate_real, 'fit': fit}[args.cmd](args)


if __name__ == '__main__':
    main()
