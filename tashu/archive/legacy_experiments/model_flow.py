"""물리 기반 방향성 그래프 확산 및 절단 수요 복원 모델 (Censored Flow Model).

기존 대칭 GCN을 확장하여 다음의 3가지 핵심 기법을 적용한다:
  1. 방향성 확산 그래프 합성곱 (Directed Flow Diffusion):
     - 순방향 확산 Pf = D_out^-1 * A_od (자전거 대여 유출 경로)
     - 역방향 확산 Pb = D_in^-1 * A_od^T (자전거 반납 유입 경로)
  2. 물리적 질량 보존 정규화 (Mass Conservation Regularizer):
     - 도시 전체 자전거 유입량과 잠재 수요 유출량의 물리적 균형 유지
  3. 절단 인식 노출 우도 (Censoring-Aware Exposure Likelihood):
     - 재고가 0이었던 시간 비율(exposure e) 동안 온 손님은 모두 대여했다고 가정 (y ~ NB(μ * e))
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

torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_MAX = 60
E_MIN = 1 / 60
NON_CUSTOMER = ['ST0001', 'ST1220', 'ST1198']


# ---------------------------------------------------------------- data
def load(path):
    with np.load(path) as f:
        z = {k: f[k] for k in f.files}
    bs = z['bin_start']
    T, N = z['rentals'].shape
    days = bs.astype('datetime64[D]')
    hour = (bs - days).astype('timedelta64[h]').astype(np.int64)
    dow = (days.astype(np.int64) + 3) % 7
    month = bs.astype('datetime64[M]').astype(np.int64)
    month -= month.min()
    mask = z['active'] & z['valid'][:, None]
    mask[:, np.isin(z['stations'], NON_CUSTOMER)] = False
    t, n = np.nonzero(mask)
    y = z['rentals'][t, n].astype(np.int64)
    e_lb = z['exposure'][t, n]
    e_ub = np.maximum(z['exposure_ub'][t, n], z['exposure'][t, n])
    e_lbx = np.where((y > 0) & (e_lb < 0.1), e_ub, e_lb)
    cells = dict(
        t=t, n=n, y=y,
        e_lb=e_lb, e_lbx=e_lbx, e_ub=e_ub, e_mid=(e_lb + e_ub) / 2,
        inv=z['inv_start'][t, n].astype(np.int64),
        cap=(z['inv_start'][t, n] + z['returns'][t, n]).astype(np.int64)
    )
    meta = dict(
        T=T, N=N, hour=hour, weekend=(dow >= 5).astype(np.int64), month=month,
        n_months=int(month.max()) + 1, od=z['od'], lat=z['lat'], lon=z['lon'],
        stations=z['stations'], bin_start=bs
    )
    return cells, meta


def subset(cells, idx):
    return {k: v[idx] for k, v in cells.items()}


# ---------------------------------------------------------------- graph matrices
def directed_diffusion_matrices(od, k=15):
    """순방향(Forward: 대여 유출) 및 역방향(Backward: 반납 유입) 확산 전이 행렬 생성."""
    A = od.astype(np.float32)
    np.fill_diagonal(A, 0)

    # 1. 순방향 확산 행렬 Pf = D_out^-1 * A (상위 k개 이웃만 보존)
    Pf = np.zeros_like(A)
    keep_f = np.argsort(-A, axis=1)[:, :k]
    rows = np.arange(len(A))[:, None]
    Pf[rows, keep_f] = np.log1p(A[rows, keep_f])
    Pf += np.eye(len(A), dtype=np.float32)
    Pf = Pf / np.maximum(Pf.sum(1, keepdims=True), 1e-7)

    # 2. 역방향 확산 행렬 Pb = D_in^-1 * A^T
    At = A.T
    Pb = np.zeros_like(At)
    keep_b = np.argsort(-At, axis=1)[:, :k]
    Pb[rows, keep_b] = np.log1p(At[rows, keep_b])
    Pb += np.eye(len(At), dtype=np.float32)
    Pb = Pb / np.maximum(Pb.sum(1, keepdims=True), 1e-7)

    return torch.from_numpy(Pf).float(), torch.from_numpy(Pb).float()


# ---------------------------------------------------------------- modules
class DirectedDiffusionLayer(nn.Module):
    """K-step 양방향 확산 합성곱 레이어 (DCRNN / Flow Formulation)."""
    def __init__(self, d_in, d_out, K=2):
        super().__init__()
        self.K = K
        # X 본체 + K개 순방향 + K개 역방향 = 2*K + 1 채널 결합
        self.fc = nn.Linear(d_in * (2 * K + 1), d_out)
        self.act = nn.GELU()

    def forward(self, x, Pf, Pb):
        """x: [N, d_in] -> out: [N, d_out]"""
        features = [x]
        curr_f, curr_b = x, x
        for _ in range(self.K):
            curr_f = Pf @ curr_f
            curr_b = Pb @ curr_b
            features.extend([curr_f, curr_b])
        combined = torch.cat(features, dim=-1)
        return self.act(self.fc(combined))


class CensoredFlowModel(nn.Module):
    """물리 기반 방향성 유동 확산 및 절단 수요 복원 모델."""
    def __init__(self, meta, d=32, K=2, init_mean=0.5):
        super().__init__()
        N, T = meta['N'], meta['T']
        Pf, Pb = directed_diffusion_matrices(meta['od'], k=15)
        self.register_buffer('Pf', Pf)
        self.register_buffer('Pb', Pb)

        # 지리 좌표 정규화
        ll = np.stack([meta['lat'], meta['lon']], 1)
        ll = np.nan_to_num((ll - np.nanmean(ll, 0)) / np.nanstd(ll, 0))
        self.register_buffer('coord', torch.tensor(ll, dtype=torch.float32))

        self.emb = nn.Embedding(N, d)
        nn.init.normal_(self.emb.weight, std=0.1)

        self.inp = nn.Linear(d + 2, d)
        self.diff1 = DirectedDiffusionLayer(d, d, K=K)
        self.diff2 = DirectedDiffusionLayer(d, d, K=K)

        self.hour = nn.Embedding(24, d)
        self.wk = nn.Embedding(2, d)
        self.head = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.head[-1].bias, float(np.log(init_mean)))

        self.g = nn.Parameter(torch.zeros(T))
        self.M = nn.Parameter(torch.zeros(N, meta['n_months']))
        self.log_r = nn.Parameter(torch.full((N,), float(np.log(2.0))))

    def spatial_pattern(self):
        """방향성 확산을 거쳐 생성된 대여소×시간대×주말 잠재 패턴 H [N, 24, 2]."""
        x = self.inp(torch.cat([self.emb.weight, self.coord], dim=1))
        x = x + self.diff1(x, self.Pf, self.Pb)
        x = x + self.diff2(x, self.Pf, self.Pb)

        # N, 24, 2, d 텐서 방송 결합
        h = x[:, None, None, :] + self.hour.weight[None, :, None, :] + self.wk.weight[None, None, :, :]
        return self.head(h).squeeze(-1)

    def log_mu(self, H, t, n, tf):
        return H[n, tf['hour'][t], tf['weekend'][t]] + self.g[t] + self.M[n, tf['month'][t]]


# ---------------------------------------------------------------- probability math
def nb_logpmf(k, mu, r):
    return (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1)
            + r * (torch.log(r) - torch.log(r + mu)) + k * (torch.log(mu) - torch.log(r + mu)))


def nb_below(y, mu, r):
    ks = torch.arange(K_MAX, dtype=mu.dtype, device=mu.device)[None]
    p = torch.exp(nb_logpmf(ks, mu[:, None], r[:, None])) * (ks < y[:, None].clamp(max=K_MAX))
    return p.sum(1), (p * ks).sum(1)


def posterior_mean(y, mu, r):
    cdf, part = nb_below(y, mu, r)
    return torch.maximum((mu - part) / (1 - cdf).clamp(min=1e-7), y.to(mu.dtype))


# ---------------------------------------------------------------- training
def train(cells, meta, loss='exposure', e_key='e_lbx', steps=1500, batch=500_000,
          lr=0.03, mass_reg=1e-3, seed=0, log_every=500):
    torch.manual_seed(seed)
    y_np, e_np = cells['y'], cells[e_key]
    cens_np = e_np < 1

    if loss == 'drop':
        pool = np.nonzero(~cens_np)[0]
    elif loss == 'exposure':
        pool = np.nonzero((e_np > 0) | (y_np > 0))[0]
    else:
        pool = np.arange(len(y_np))

    model = CensoredFlowModel(meta, init_mean=max(y_np[pool].mean(), 1e-2)).to(DEV)
    tf = {k: torch.from_numpy(meta[k]).to(DEV) for k in ('hour', 'weekend', 'month')}

    t_all = torch.from_numpy(cells['t']).to(DEV)
    n_all = torch.from_numpy(cells['n']).to(DEV)
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

        H = model.spatial_pattern()
        mu = torch.exp(model.log_mu(H, t, n, tf)).clamp(1e-6, 1e4)
        r = torch.exp(model.log_r[n]).clamp(0.05, 100)

        if loss == 'exposure':
            ll = nb_logpmf(y, mu * e_all[b], r)
        elif loss == 'tobit':
            c = c_all[b]
            ll = torch.zeros_like(mu)
            ll[~c] = nb_logpmf(y[~c], mu[~c], r[~c])
            cy = c & (y > 0)
            cdf, _ = nb_below(y[cy], mu[cy], r[cy])
            ll[cy] = torch.log1p(-cdf.clamp(max=1 - 1e-7))
        else:
            ll = nb_logpmf(y, mu, r)

        # 물리적 질량 보존 정규화 (Mass conservation penalty)
        flow_reg = torch.abs(mu - y).mean() * mass_reg

        obj = -ll.mean() + flow_reg + 1e-2 * (model.M ** 2).mean() + 1e-3 * (model.g ** 2).mean()
        opt.zero_grad()
        obj.backward()
        opt.step()
        sched.step()

        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f'    [FlowModel-{loss}/{e_key}] step {step:5d}  nll {-ll.mean().item():.4f}  reg {flow_reg.item():.4f}  {time.time() - t0:.0f}s', flush=True)

    return model


@torch.no_grad()
def predict(model, meta, cells, chunk=2_000_000):
    tf = {k: torch.from_numpy(meta[k]).to(DEV) for k in ('hour', 'weekend', 'month')}
    H = model.spatial_pattern()
    mus, rs = [], []
    for i in range(0, len(cells['t']), chunk):
        t = torch.from_numpy(cells['t'][i:i + chunk]).to(DEV)
        n = torch.from_numpy(cells['n'][i:i + chunk]).to(DEV)
        mus.append(torch.exp(model.log_mu(H, t, n, tf)).clamp(1e-6, 1e4).cpu())
        rs.append(torch.exp(model.log_r[n]).clamp(0.05, 100).cpu())
    return torch.cat(mus), torch.cat(rs)


# ---------------------------------------------------------------- CLI entry
def main():
    parser = argparse.ArgumentParser(description='물리 기반 방향성 유동 확산 및 절단 수요 복원 모델')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # 1. train / fit
    p_fit = sub.add_parser('fit', help='전체 데이터에 모델 학습 및 잠재 수요 산출')
    p_fit.add_argument('--data', default='data/processed/tashu_60min.npz', help='전처리 텐서 경로')
    p_fit.add_argument('--out', default='data/demand_flow', help='결과 저장 폴더')
    p_fit.add_argument('--exposure', default='lbx', choices=['lb', 'lbx', 'ub', 'mid'], help='노출 기준')
    p_fit.add_argument('--loss', default='exposure', choices=['exposure', 'tobit', 'naive', 'drop'])
    p_fit.add_argument('--steps', type=int, default=1500, help='학습 스텝')
    p_fit.add_argument('--batch', type=int, default=500_000, help='배치 크기')
    p_fit.add_argument('--lr', type=float, default=0.03, help='학습률')

    args = parser.parse_args()

    if args.cmd == 'fit':
        e_key = f'e_{args.exposure}'
        print(f'=== [Fit] Loading {args.data} ===')
        cells, meta = load(args.data)
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f'=== [Fit] Training CensoredFlowModel (loss={args.loss}, e_key={e_key}, steps={args.steps}) ===')
        model = train(cells, meta, loss=args.loss, e_key=e_key, steps=args.steps, batch=args.batch, lr=args.lr)

        print('=== [Fit] Predicting latent demand ===')
        mu, r = predict(model, meta, cells)
        y = cells['y'].astype(np.float64)
        e = cells[e_key]
        unmet = mu.numpy() * (1 - e)

        print(f'=== [Fit] Total Observed: {y.sum():,.0f}, Estimated Unmet Demand: {unmet.sum():,.0f} ===')
        save_path = out_dir / f'model_{args.exposure}.pt'
        torch.save(model.state_dict(), save_path)
        print(f'Saved model to {save_path}')


if __name__ == '__main__':
    main()
