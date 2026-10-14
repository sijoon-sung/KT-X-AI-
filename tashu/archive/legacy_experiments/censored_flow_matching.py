"""Censored-Flow: 재고 조건부 다음 시간 유동 예측과 절단 대여량 학습.

3대 핵심 아키텍처:
  1. Flow Conservation Prior:
     - 대여(유출 Pout)와 반납(유입 Pin)의 OD 전이 행렬(Pf, Pb)을 기반으로 한
       자전거 유동 보존 속도장(Velocity Field v_theta) 신경망 설계
  2. Censoring-Aware Flow Matching:
     - 반납은 관측 유동에 맞추고, 재고 품절(exposure < 1) 대여는 노출 우도로 학습
  3. Inventory-Conditioned Flow:
     - 학습과 ODE 추론 모두 재고 스냅샷을 시작 상태와 조건으로 사용
"""
import argparse
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
STATIONS_CSV = Path('data/processed/stations.csv')
FLOW_SCALE = 10.0  # 대여/반납 건수와 재고 앵커의 고정 스케일


def inventory_anchor(inventory):
    """학습과 추론에 공통으로 사용하는 재고 시작 상태 [B, N, 2]."""
    scaled = torch.log1p(inventory.clamp_min(0)) / FLOW_SCALE
    return torch.stack((scaled, scaled), dim=-1)


# ---------------------------------------------------------------- OD transition matrices
def build_directed_transition_matrices(od, k=15):
    """
    순방향(Forward: 대여 유출 -> 반납 유입) 및 역방향(Backward) 확산 전이 행렬 구축.
    """
    A = od.astype(np.float32)
    np.fill_diagonal(A, 0)

    # 1. 순방향 전이 행렬 Pf = D_out^-1 * A (어디서 빌려서 어디로 흘러가는가)
    Pf = np.zeros_like(A)
    keep_f = np.argsort(-A, axis=1)[:, :k]
    rows = np.arange(len(A))[:, None]
    Pf[rows, keep_f] = np.log1p(A[rows, keep_f])
    Pf += np.eye(len(A), dtype=np.float32)
    Pf = Pf / np.maximum(Pf.sum(1, keepdims=True), 1e-7)

    # 2. 역방향 전이 행렬 Pb = D_in^-1 * A^T (어디로 반납된 자전거가 어디서 왔는가)
    At = A.T
    Pb = np.zeros_like(At)
    keep_b = np.argsort(-At, axis=1)[:, :k]
    Pb[rows, keep_b] = np.log1p(At[rows, keep_b])
    Pb += np.eye(len(At), dtype=np.float32)
    Pb = Pb / np.maximum(Pb.sum(1, keepdims=True), 1e-7)

    return torch.from_numpy(Pf).float(), torch.from_numpy(Pb).float()


# ---------------------------------------------------------------- time embedding
class SinusoidalPositionalEmbedding(nn.Module):
    """연속 시간 t in [0, 1]에 대한 고주파수 Sinusoidal 임베딩."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim)
        args = t[:, None] * freqs[None, :] * 50.0
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


# ---------------------------------------------------------------- Pillar 1: Flow Conservation Prior Network
class FlowConservationVelocityNet(nn.Module):
    """
    자전거 유동 보존 물리 법칙을 내장한 GCN 속도장 신경망 v_theta(x_t, t, Pf, Pb).
    x_t 채널: [반납 건수 / FLOW_SCALE, 대여 건수 / FLOW_SCALE]
    """
    def __init__(self, n_nodes, in_channels=2, hidden_dim=64):
        super().__init__()
        self.n_nodes = n_nodes
        self.in_channels = in_channels

        # 시간 조건 임베딩
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 상태 사영
        self.in_proj = nn.Linear(in_channels, hidden_dim)
        self.inventory_proj = nn.Linear(1, hidden_dim)

        # 양방향 확산 GCN 층 (순방향 유출 + 역방향 유입)
        self.gcn_f1 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_b1 = nn.Linear(hidden_dim, hidden_dim)

        self.gcn_f2 = nn.Linear(hidden_dim, hidden_dim)
        self.gcn_b2 = nn.Linear(hidden_dim, hidden_dim)

        # 속도 벡터 출력 헤드 (v_theta: x_t와 동일한 [B, N, in_channels])
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, in_channels)
        )

    def forward(self, x_t, t, Pf, Pb, inventory):
        """
        x_t: [B, N, 2] (시간 t에서의 스케일된 유동 건수)
        t  : [B] (연속 시간)
        Pf : [N, N] (순방향 확산 전이 행렬)
        Pb : [N, N] (역방향 확산 전이 행렬)
        inventory: [B, N] (현재 재고 / FLOW_SCALE)
        out: [B, N, 2] (속도 벡터장 v_theta)
        """
        B, N, C = x_t.shape
        t_emb = self.time_mlp(t).unsqueeze(1)  # [B, 1, hidden_dim]

        # 1. 입력 사영 + 시간 주입
        h = self.in_proj(x_t) + self.inventory_proj(inventory.unsqueeze(-1)) + t_emb

        # 2. 제1차 양방향 물리 확산
        h_f1 = F.gelu(self.gcn_f1(torch.matmul(Pf, h)))
        h_b1 = F.gelu(self.gcn_b1(torch.matmul(Pb, h)))
        h = h + (h_f1 + h_b1) * 0.5

        # 3. 제2차 양방향 물리 확산
        h_f2 = F.gelu(self.gcn_f2(torch.matmul(Pf, h)))
        h_b2 = F.gelu(self.gcn_b2(torch.matmul(Pb, h)))
        h = h + (h_f2 + h_b2) * 0.5

        # 4. 물리적 질량 보존 잔차 결합 (Flow Conservation Residual)
        # x_t[..., 0]은 Pin(유입), x_t[..., 1]은 Pout(유출)
        # 유출된 자전거가 도로망 Pf를 타고 유입된 예측치 Pf^T @ Pout과 실제 Pin의 오차
        p_out = x_t[:, :, 1:2]
        predicted_flow = torch.matmul(Pf.t(), p_out)
        flow_residual = h * predicted_flow

        combined = torch.cat([h, flow_residual], dim=-1)
        v_theta = self.out_head(combined)  # [B, N, 2]
        return v_theta


# ---------------------------------------------------------------- Pillar 2: Censoring-Aware Flow Matching Loss
class CensoringAwareFlowMatching(nn.Module):
    """
    절단 인식 플로우 매칭 프레임워크 (Censored-CFM).
    - 반납 및 비절단 대여: MSE 속도장 회귀
    - 절단 대여: 관측 건수와 재고 노출도를 이용한 Poisson 우도
    """
    def __init__(self, n_nodes, od_matrix, in_channels=2, hidden_dim=64):
        super().__init__()
        self.n_nodes = n_nodes
        Pf, Pb = build_directed_transition_matrices(od_matrix, k=15)
        self.register_buffer('Pf', Pf)
        self.register_buffer('Pb', Pb)
        self.v_net = FlowConservationVelocityNet(n_nodes, in_channels=in_channels, hidden_dim=hidden_dim)

    def compute_loss(self, x_1, exposure_mask, inventory):
        """
        x_1: 다음 시간의 관측 대여/반납 건수를 FLOW_SCALE로 나눈 값 [B, N, 2]
        exposure_mask: 재고 가용 시간 비율 [B, N] (1.0이면 완전관측, <1.0이면 절단)
        inventory: 현재 시간의 시작 재고 [B, N]
        """
        B, N, C = x_1.shape
        x_0 = inventory_anchor(inventory)

        # 1. 연속 시간 t ~ Uniform(0, 1)
        t = torch.rand(B, device=x_1.device)
        t_broad = t.view(B, 1, 1)

        # 2. 최적 수송 변위 보간 궤적: x_t = (1 - t) * x_0 + t * x_1
        x_t = (1.0 - t_broad) * x_0 + t_broad * x_1
        u_t = x_1 - x_0  # 목표 속도장

        # 3. 신경망 속도장 예측
        v_pred = self.v_net(x_t, t, self.Pf, self.Pb, inventory / FLOW_SCALE)

        # 4. 속도장으로부터 도출되는 최종 도달 상태 예측: x_hat_1 = x_t + (1 - t) * v_pred
        x_hat_1 = x_t + (1.0 - t_broad) * v_pred

        # 반납은 재고 부족으로 절단되지 않으므로 모든 노드에서 정확한 유동을 학습한다.
        return_loss = F.mse_loss(v_pred[..., 0], u_t[..., 0])

        uncensored = exposure_mask >= 1.0
        rental_loss = F.mse_loss(v_pred[..., 1][uncensored], u_t[..., 1][uncensored]) if uncensored.any() else v_pred[..., 1].sum() * 0

        # 절단 대여량은 정규화된 공간 비율이 아니라 원래 건수에 대한 노출 우도를 쓴다.
        # 관측 대여 y ~ Poisson(잠재 수요 * exposure). 노출 0, 관측 0은 정보가 없다.
        censored = (exposure_mask < 1.0) & ((exposure_mask > 0) | (x_1[..., 1] > 0))
        if censored.any():
            latent_rentals = F.softplus(x_hat_1[..., 1][censored] * FLOW_SCALE)
            effective_exposure = exposure_mask[censored].clamp_min(1 / 60)
            observed = x_1[..., 1][censored] * FLOW_SCALE
            rate = (latent_rentals * effective_exposure).clamp_min(1e-6)
            censored_loss = (rate - observed * torch.log(rate)).mean() / FLOW_SCALE
        else:
            censored_loss = v_pred[..., 1].sum() * 0

        return return_loss + rental_loss + censored_loss


# ---------------------------------------------------------------- Pillar 3: Sparse Anchor Injection ODE
@torch.no_grad()
def sample_with_realtime_anchor(model, real_inventory_vector, n_steps=10):
    """
    현재 재고 스냅샷을 Boundary Condition과 조건으로 사용하여
    미래 유동 건수 및 품절 위험을 Euler ODE로 적분 추론.

    real_inventory_vector: [N] 대여소별 현재 실제 거치 자전거 대수
    """
    model.eval()
    N = model.n_nodes
    if n_steps < 1:
        raise ValueError('n_steps must be positive')
    if len(real_inventory_vector) != N:
        raise ValueError(f'Expected inventory for {N} stations')

    # 학습과 동일한 변환으로 초기 경계조건 x_0 생성
    inv_t = torch.as_tensor(real_inventory_vector, dtype=torch.float32, device=model.Pf.device)

    # 초기 상태 x_0: 현재 재고를 바탕으로 초기 유입/유출 베이스라인 구성
    x_t = inventory_anchor(inv_t).unsqueeze(0)  # [1, N, 2]

    dt = 1.0 / n_steps
    trajectory = [x_t.cpu()]

    # 2. 10-Step Euler Numerical Integration
    for step in range(n_steps):
        t_val = step * dt
        t = torch.tensor([t_val], device=x_t.device)
        # 물리 보존 속도장 계산
        v = model.v_net(x_t, t, model.Pf, model.Pb, inv_t.unsqueeze(0) / FLOW_SCALE)
        # 오일러 전진 적분
        x_t = x_t + dt * v
        trajectory.append(x_t.cpu())

    # 최종 t=1 시점의 예측 유동 건수: [N, 2] (0: 반납, 1: 대여)
    final_flow = x_t.squeeze(0).cpu().numpy()
    pred_pin = np.maximum(final_flow[:, 0] * FLOW_SCALE, 0)
    pred_pout = np.maximum(final_flow[:, 1] * FLOW_SCALE, 0)

    # 3. 품절 위험 지수 (Stockout Risk Index) 산출:
    # 현재 재고 + 예상 반납 대비 예상 대여가 많은 대여소
    stockout_risk = pred_pout - pred_pin - inv_t.cpu().numpy()

    return {
        'pred_pin': pred_pin,
        'pred_pout': pred_pout,
        'stockout_risk': stockout_risk,
        'trajectory': trajectory
    }


# ---------------------------------------------------------------- training loop
def train_censored_flow(npz_path='data/processed/tashu_60min.npz', steps=300, batch_size=32, lr=0.002,
                        train_before=None):
    np.random.seed(0)
    torch.manual_seed(0)
    with np.load(npz_path) as f:
        rentals = f['rentals'].astype(np.float32)
        returns = f['returns'].astype(np.float32)
        exposure = f['exposure'].astype(np.float32)
        inventory = f['inv_start'].astype(np.float32)
        bin_start = f['bin_start']
        od = f['od']
        N = rentals.shape[1]
        valid = f['valid'] & ~f['closed']
    if train_before is not None:
        valid &= bin_start < np.datetime64(train_before)

    # inv_start[t]는 구간 시작 재고, 유동[t]는 그 다음 1시간의 대여/반납이다.
    valid_indices = np.where(valid)[0]
    if not len(valid_indices):
        raise ValueError('No valid one-hour training bins')

    # 고정 스케일의 건수이므로 대여 관측값은 잠재 수요의 하한으로 유지된다.
    flow_tensor = np.stack([returns, rentals], axis=-1) / FLOW_SCALE

    flow_t = torch.from_numpy(flow_tensor).to(DEV)
    exp_t = torch.from_numpy(exposure).to(DEV)
    inv_t = torch.from_numpy(inventory).to(DEV)

    model = CensoringAwareFlowMatching(n_nodes=N, od_matrix=od, in_channels=2, hidden_dim=64).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print(f'=== [Pillars 1 & 2] Training Censored-Flow on Tashu: steps={steps}, batch={batch_size} ===')
    t0 = time.time()
    for step in range(1, steps + 1):
        batch_idx = np.random.choice(valid_indices, size=batch_size, replace=True)
        x_1 = flow_t[batch_idx]
        exp_mask = exp_t[batch_idx]
        current_inventory = inv_t[batch_idx]

        loss = model.compute_loss(x_1, exp_mask, current_inventory)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == steps:
            print(f'    [Censored-Flow] step {step:4d}/{steps} | Loss (CFM + exposure likelihood): {loss.item():.6f} | {time.time() - t0:.1f}s', flush=True)

    return model


# ---------------------------------------------------------------- CLI entry
def main():
    parser = argparse.ArgumentParser(description='Censored-Flow: Physics-Informed Flow Matching for Tashu')
    sub = parser.add_subparsers(dest='cmd', required=True)

    # 1. 학습
    p_fit = sub.add_parser('fit', help='Flow Conservation Prior & Censoring-Aware CFM 학습')
    p_fit.add_argument('--data', default='data/processed/tashu_60min.npz')
    p_fit.add_argument('--steps', type=int, default=200)
    p_fit.add_argument('--batch', type=int, default=32)
    p_fit.add_argument('--out', default='data/censored_flow_model.pt')
    p_fit.add_argument('--train-before', default=None,
                       help='이 날짜 전 구간만 학습 (예: 2026-02-01)')

    # 2. 저장된 재고 스냅샷을 조건으로 한 ODE 시뮬레이션
    p_sim = sub.add_parser('anchor-sim', help='저장된 재고 스냅샷 앵커 주입 후 다음 시간 유동 예측')
    p_sim.add_argument('--model', default='data/censored_flow_model.pt')
    p_sim.add_argument('--data', default='data/processed/tashu_60min.npz')
    p_sim.add_argument('--ode_steps', type=int, default=10)

    args = parser.parse_args()

    if args.cmd == 'fit':
        model = train_censored_flow(npz_path=args.data, steps=args.steps, batch_size=args.batch,
                                    train_before=args.train_before)
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_p)
        print(f'=== Successfully saved Censored-Flow model to {out_p} ===')

    elif args.cmd == 'anchor-sim':
        print('=== Running Inventory-Conditioned Forecast with Historical Snapshot ===')
        with np.load(args.data) as f:
            od = f['od']
            N = f['rentals'].shape[1]
            stations = f['stations'].astype(str)
            # 과거 재고 스냅샷 (실시간 API 데이터가 아님)
            real_inv = f['inv_start'][-1000].astype(np.float32)

        model = CensoringAwareFlowMatching(n_nodes=N, od_matrix=od, in_channels=2, hidden_dim=64).to(DEV)
        if not Path(args.model).exists():
            parser.error(f'Trained model not found: {args.model}')
        try:
            model.load_state_dict(torch.load(args.model, map_location=DEV, weights_only=True))
        except RuntimeError as exc:
            parser.error(f'Model weights do not match this version; retrain with fit. {exc}')
        print(f'Loaded trained weights from {args.model}')

        result = sample_with_realtime_anchor(model, real_inv, n_steps=args.ode_steps)
        risk = result['stockout_risk']

        # 상위 품절 위험 대여소 10곳 추출
        top_risk_idx = np.argsort(-risk)[:10]

        st_names = {}
        if STATIONS_CSV.exists():
            with open(STATIONS_CSV, 'r', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    st_names[row['station_id']] = row['name']

        print(f'\n=== Top 10 High-Risk Stockout Stations (Predicted via 10-Step Flow ODE) ===')
        for rank, idx in enumerate(top_risk_idx, 1):
            sid = stations[idx]
            name = st_names.get(sid, sid)
            # 윈도우 콘솔 한글 인코딩 안전 출력
            try:
                display_name = name[:20]
            except Exception:
                display_name = sid
            print(f'  {rank:2d}. [{sid}] {display_name:<20s} | Risk Index: {risk[idx]:+.4f} (Inv: {real_inv[idx]:.0f})')
        print(f'=== {args.ode_steps}-Step Inventory-Conditioned ODE Integration Finished ===\n')


if __name__ == '__main__':
    main()
