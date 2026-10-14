"""시공간 그래프 합성곱 신경망 (Full Spatio-Temporal GCN for Bike-Sharing).

구조:
  - 입력 텐서: (배치 B, 과거 시간 L=6, 대여소 수 N=1425, 피처 채널 F=4)
  - 피처 채널 (사용자 제안 유입/유출 분포):
      채널 0: P_in  (반납된 곳의 공간 확률 분포: returns / sum(returns))
      채널 1: P_out (시작된 곳의 공간 확률 분포: rentals / sum(rentals))
      채널 2: inv   (대여소 시작 재고 스케일링)
      채널 3: exp   (노출도 exposure: 재고 > 0 시간 비율)
  - ST-Block (시공간 샌드위치 블록):
      1. Temporal Layer: Gated 1D Temporal Convolution (TCN)으로 직전 추세 학습
      2. Spatial Layer : OD 이동 행렬 기반 Graph Convolution (GCN)으로 공간 유동 전파
      3. Temporal Layer: 1D TCN으로 시계열 특징 집계
  - 출력: 다음 1시간 뒤 1,425개 대여소의 잠재 수요 μ 및 재고 순변화
  - 손실: 절단 인식 음이항 노출 우도 (y ~ NB(μ * e))
"""
import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
K_MAX = 60
E_MIN = 1 / 60
NON_CUSTOMER = ['ST0001', 'ST1220', 'ST1198']


# ---------------------------------------------------------------- data & tensor builder
def load_raw_tensors(npz_path):
    """npz에서 [T, N] 기본 배열들을 로드하고 유효 마스크를 생성한다."""
    with np.load(npz_path) as f:
        z = {k: f[k] for k in f.files}
    bs = z['bin_start']
    T, N = z['rentals'].shape
    days = bs.astype('datetime64[D]')
    hour = (bs - days).astype('timedelta64[h]').astype(np.int64)
    dow = (days.astype(np.int64) + 3) % 7
    month = bs.astype('datetime64[M]').astype(np.int64)
    month -= month.min()

    valid_t = z['valid'] & ~z['closed']
    active_mask = z['active'] & valid_t[:, None]
    active_mask[:, np.isin(z['stations'], NON_CUSTOMER)] = False

    e_lb = z['exposure']
    e_ub = np.maximum(z['exposure_ub'], z['exposure'])
    e_lbx = np.where((z['rentals'] > 0) & (e_lb < 0.1), e_ub, e_lb)

    meta = dict(
        T=T, N=N, hour=hour, weekend=(dow >= 5).astype(np.int64), month=month,
        n_months=int(month.max()) + 1, od=z['od'], lat=z['lat'], lon=z['lon'],
        stations=z['stations'], bin_start=bs, valid_t=valid_t, active_mask=active_mask
    )
    data = dict(
        rentals=z['rentals'].astype(np.float32),
        returns=z['returns'].astype(np.float32),
        inv=z['inv_start'].astype(np.float32),
        exposure=e_lbx.astype(np.float32)
    )
    return data, meta


def build_st_dataset(data, meta, L=6):
    """사용자 제안 공간 분포(Pin, Pout)를 결합한 [T, N, F] 텐서를 생성한다."""
    rentals = data['rentals']
    returns = data['returns']
    inv = data['inv']
    exp = data['exposure']

    # 1. 공간 유입/유출 확률 밀도 분포 (Spatial Flow Distribution)
    tot_ret = np.maximum(returns.sum(axis=1, keepdims=True), 1.0)
    tot_rent = np.maximum(rentals.sum(axis=1, keepdims=True), 1.0)
    p_in = returns / tot_ret     # 반납된 곳의 공간 분포
    p_out = rentals / tot_rent   # 시작된 곳의 공간 분포

    # 2. 4차원 피처 텐서 [T, N, F=4]
    inv_norm = np.log1p(inv)
    feat_tensor = np.stack([p_in, p_out, inv_norm, exp], axis=-1)  # [T, N, 4]

    # 3. 과거 L시간이 모두 valid한 유효 타임스텝 t 인덱스 추출
    valid_t = meta['valid_t']
    valid_steps = []
    for t in range(L, len(valid_t)):
        if np.all(valid_t[t - L:t + 1]):
            valid_steps.append(t)
    valid_steps = np.array(valid_steps, dtype=np.int64)

    return feat_tensor, valid_steps


def build_od_adjacency(od, k=15):
    """OD 통행량 기반 대칭 인접 행렬 및 라플라시안 정규화."""
    A = (od + od.T).astype(np.float32)
    np.fill_diagonal(A, 0)
    keep = np.argsort(-A, axis=1)[:, :k]
    A_sparse = np.zeros_like(A)
    rows = np.arange(len(A))[:, None]
    A_sparse[rows, keep] = np.log1p(A[rows, keep])
    A_sparse += np.eye(len(A), dtype=np.float32)
    deg = A_sparse.sum(axis=1, keepdims=True)
    A_norm = A_sparse / np.maximum(deg, 1e-7)
    return torch.from_numpy(A_norm).float()


# ---------------------------------------------------------------- ST-GCN layers
class TemporalConv(nn.Module):
    """Gated 1D Temporal Convolution (TCN)."""
    def __init__(self, c_in, c_out, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out * 2, kernel_size=(kernel_size, 1), padding=(kernel_size // 2, 0))

    def forward(self, x):
        # x: [B, c_in, L, N]
        h = self.conv(x)
        p, q = torch.chunk(h, 2, dim=1)
        return p * torch.sigmoid(q)  # Gated TCN 활성화


class SpatialGraphConv(nn.Module):
    """공간 그래프 합성곱 레이어 (Spatial GCN)."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.fc = nn.Linear(c_in, c_out)

    def forward(self, x, A):
        # x: [B, c_in, L, N] -> transpose -> [B, L, N, c_in]
        b, c, l, n = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # [B, L, N, c_in]
        # 그래프 전파: A @ X
        # A: [N, N], x_perm: [B, L, N, c_in]
        ax = torch.einsum('nm, blmc -> blnc', A, x_perm)
        out = self.fc(ax)  # [B, L, N, c_out]
        return F.gelu(out).permute(0, 3, 1, 2)  # [B, c_out, L, N]


class STBlock(nn.Module):
    """Temporal -> Spatial -> Temporal 샌드위치 블록."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.tcn1 = TemporalConv(c_in, c_out, kernel_size=3)
        self.gcn = SpatialGraphConv(c_out, c_out)
        self.tcn2 = TemporalConv(c_out, c_out, kernel_size=3)
        self.res = nn.Conv2d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
        self.norm = nn.BatchNorm2d(c_out)

    def forward(self, x, A):
        res = self.res(x)
        h = self.tcn1(x)
        h = self.gcn(h, A)
        h = self.tcn2(h)
        return self.norm(h + res)


class SpatioTemporalGCN(nn.Module):
    """전체 시공간 GCN 모델."""
    def __init__(self, meta, c_in=4, d=32, L=6, init_mean=0.5):
        super().__init__()
        self.L = L
        self.N = meta['N']
        self.register_buffer('A', build_od_adjacency(meta['od'], k=15))

        # 정적 노드 임베딩 (대여소 고유 특성)
        self.st_emb = nn.Embedding(self.N, d)
        nn.init.normal_(self.st_emb.weight, std=0.1)

        # ST-Block 2단계
        self.block1 = STBlock(c_in, d)
        self.block2 = STBlock(d, d)

        # 시간/캘린더 임베딩
        self.hour_emb = nn.Embedding(24, d)
        self.wk_emb = nn.Embedding(2, d)

        # 최종 예측 헤드 (잠재 수요 log μ 산출)
        self.head = nn.Sequential(
            nn.Linear(d + d + d + d, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        nn.init.constant_(self.head[-1].bias, float(np.log(init_mean)))

        self.log_r = nn.Parameter(torch.full((self.N,), float(np.log(2.0))))

    def forward(self, x_seq, hour, weekend):
        """
        x_seq: [B, L, N, F]
        hour, weekend: [B]
        out: [B, N] (log_mu)
        """
        B = x_seq.size(0)
        # 텐서 형태 변환: [B, F, L, N]
        x = x_seq.permute(0, 3, 1, 2)
        h = self.block1(x, self.A)
        h = self.block2(h, self.A)  # [B, d, L, N]

        # 최근 시점의 공간 임베딩 취합 [B, N, d]
        h_last = h[:, :, -1, :].permute(0, 2, 1)

        # 노드 임베딩 방송 결합 [B, N, d]
        node_emb = self.st_emb.weight.unsqueeze(0).expand(B, -1, -1)

        # 시간 임베딩 방송 결합 [B, N, d]
        h_hour = self.hour_emb(hour).unsqueeze(1).expand(-1, self.N, -1)
        h_wk = self.wk_emb(weekend).unsqueeze(1).expand(-1, self.N, -1)

        combined = torch.cat([h_last, node_emb, h_hour, h_wk], dim=-1)
        log_mu = self.head(combined).squeeze(-1)  # [B, N]
        return log_mu


# ---------------------------------------------------------------- loss
def nb_logpmf(k, mu, r):
    k_term = torch.where(k > 0, k * (torch.log(mu) - torch.log(r + mu)), torch.zeros_like(mu))
    return (torch.lgamma(k + r) - torch.lgamma(r) - torch.lgamma(k + 1)
            + r * (torch.log(r) - torch.log(r + mu)) + k_term)


# ---------------------------------------------------------------- training loop
def train_st_gcn(feat_tensor, valid_steps, data, meta, L=6, steps=1000, batch_size=32, lr=0.01):
    model = SpatioTemporalGCN(meta, c_in=4, d=32, L=L, init_mean=0.5).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    feat_tensor_t = torch.from_numpy(feat_tensor).to(DEV)
    rentals_t = torch.from_numpy(data['rentals']).to(DEV)
    exposure_t = torch.from_numpy(data['exposure']).to(DEV)
    active_mask_t = torch.from_numpy(meta['active_mask']).to(DEV)

    hour_t = torch.from_numpy(meta['hour']).to(DEV)
    wk_t = torch.from_numpy(meta['weekend']).to(DEV)

    print(f'=== Training ST-GCN: {len(valid_steps):,} sequences, steps={steps}, batch={batch_size} ===')
    t0 = time.time()
    for step in range(steps):
        # 무작위 타임스텝 샘플링
        batch_idx = np.random.choice(valid_steps, size=batch_size, replace=True)

        # 슬라이딩 윈도우 슬라이싱: [B, L, N, F]
        seqs = [feat_tensor_t[t - L:t] for t in batch_idx]
        x_batch = torch.stack(seqs, dim=0)

        t_tgt = torch.from_numpy(batch_idx).to(DEV)
        h_tgt = hour_t[t_tgt]
        w_tgt = wk_t[t_tgt]

        log_mu = model(x_batch, h_tgt, w_tgt)  # [B, N]
        mu = torch.exp(log_mu).clamp(1e-6, 1e4)
        r = torch.exp(model.log_r).unsqueeze(0).clamp(0.05, 100)

        # 타깃 대여량 및 노출도
        y_tgt = rentals_t[t_tgt]        # [B, N]
        e_tgt = exposure_t[t_tgt]       # [B, N]
        mask = active_mask_t[t_tgt]     # [B, N]

        # 음이항 절단 우도 (Exposure loss)
        # e가 0일 때 log(0)으로 인한 nan 방지
        e_clamped = torch.where(y_tgt > 0, torch.clamp(e_tgt, min=E_MIN), e_tgt)
        mu_eff = torch.clamp(mu * e_clamped, min=1e-6, max=1e4)
        ll = nb_logpmf(y_tgt, mu_eff, r)
        loss = -ll[mask].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 200 == 0 or step == steps - 1:
            print(f'    [ST-GCN] step {step:4d} | NLL Loss: {loss.item():.4f} | elapsed: {time.time() - t0:.1f}s', flush=True)

    return model


# ---------------------------------------------------------------- main entry
def main():
    parser = argparse.ArgumentParser(description='Full Spatio-Temporal GCN for Tashu')
    parser.add_argument('--data', default='data/processed/tashu_60min.npz', help='전처리 텐서 경로')
    parser.add_argument('--out', default='data/st_gcn_model.pt', help='모델 저장 경로')
    parser.add_argument('--steps', type=int, default=1000, help='학습 스텝')
    parser.add_argument('--batch', type=int, default=32, help='배치 사이즈')
    parser.add_argument('--lr', type=float, default=0.01, help='학습률')
    parser.add_argument('--lookback', type=int, default=6, help='과거 시간 윈도우 L')
    args = parser.parse_args()

    print(f'Loading {args.data}...')
    data, meta = load_raw_tensors(args.data)
    print(f'Building ST sequence dataset with lookback L={args.lookback}...')
    feat_tensor, valid_steps = build_st_dataset(data, meta, L=args.lookback)

    model = train_st_gcn(feat_tensor, valid_steps, data, meta, L=args.lookback, steps=args.steps, batch_size=args.batch, lr=args.lr)

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_p)
    print(f'Successfully saved ST-GCN model to {out_p}')


if __name__ == '__main__':
    main()
