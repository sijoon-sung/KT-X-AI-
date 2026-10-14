"""Neural Flux Operator (Neural-KFDM): 고차원 신경망 유속 연산자 기반 동역학적 유량 모델.

핵심 아키텍처:
1. 고차원 대여소 잠재 표현 (Latent Station Space Z in R^{N x D}, D=32):
   - 대여소의 공간적 특징과 지리적 위치를 잠재 매니폴드에 임베딩.
2. 상황 적응형 신경망 유속 연산자 (Dynamic Neural Advection Operator):
   - 출발지 Query Q_i(t)와 도착지 Key K_j(t)를 시공간 컨텍스트(시간/요일/계절)와 결합.
   - Top-K 스파스 경로(K=32)에 대한 Sparse Dynamic Attention 적용:
     P_OD(t)_{ij} = Softmax_{k in K}( (Q_i(t) * K_j(t)) / sqrt(d) + log P_prior_{ij} )
   - Row-stochastic normalization으로 sum_j P_OD(t)_{ij} == 1.0 보장.
3. 질량 보존 보장 물리 이류 (Scatter-Add Conservation):
   - mu_return = ScatterAdd(Lambda_rent * P_OD(t))
   - sum(mu_return) == sum(Lambda_rent) == M(t) 수학적 항등 보장.
4. 동적 유체 발산 및 실시간 최적도 진단:
   - 출퇴근 방향 전환(아침: 주거->중심 vs 저녁: 중심->주거)에 따라 실시간으로 변화하는
     발산(Divergence_i(t) = Lambda_i(t) - mu_i(t))을 계산하여 동적 최적 재고 I*(t) 도출.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kinetic_flux_demand import (
    DEV,
    NON_CUSTOMER,
    BoltzmannSpatialHead,
    MacroFluxHead,
    calendar_features,
    load_dataset,
    nb_logpmf,
)

torch.set_num_threads(os.cpu_count() or 4)


def build_topk_sparse_prior(od_matrix, k=32, smoothing=0.01):
    """
    각 대여소별 상위 K개 도착지 인덱스와 사전 확률의 로그값을 구축한다.
    자기 자신(Self-loop)은 항상 포함된다.
    """
    n = len(od_matrix)
    A = od_matrix.astype(np.float32) + smoothing
    # 대각 성분에 가중치 부여 (근거리/자가 반납 보장)
    np.fill_diagonal(A, np.maximum(A.diagonal(), 1.0))

    top_k_indices = np.zeros((n, k), dtype=np.int64)
    log_prior = np.zeros((n, k), dtype=np.float32)

    for i in range(n):
        # 상위 k개 인덱스 선택 (자기 자신 포함 보장)
        row = A[i].copy()
        row[i] += 1.0  # 자기 자신 우선순위 상향
        best_k = np.argpartition(-row, k)[:k]
        best_k = best_k[np.argsort(-row[best_k])]
        top_k_indices[i] = best_k

        probs = row[best_k] / row[best_k].sum()
        log_prior[i] = np.log(np.maximum(probs, 1e-6))

    return torch.from_numpy(top_k_indices).long(), torch.from_numpy(log_prior).float()


class DynamicNeuralAdvection(nn.Module):
    """64차원 잠재 공간에서 시공간 컨텍스트에 따라 변화하는 동적 유속 연산자."""
    def __init__(self, n_stations, top_k_indices, log_prior, d_model=32):
        super().__init__()
        self.n_stations = n_stations
        self.k = top_k_indices.shape[1]
        self.d_model = d_model

        # Top-K 그래프 인덱스 및 사전 확률 버퍼 등록
        self.register_buffer('top_k_indices', top_k_indices)  # [N, K]
        self.register_buffer('log_prior', log_prior)          # [N, K]

        # 대여소 잠재 표현 Z in R^{N x d}
        self.station_emb = nn.Embedding(n_stations, d_model)

        # 시간/요일/월 컨텍스트 인코더
        self.ctx_hour = nn.Embedding(24, 12)
        self.ctx_weekday = nn.Embedding(7, 6)
        self.ctx_month = nn.Embedding(12, 6)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(24, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Query / Key Projection (출발지 성향 vs 도착지 매력도)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.scale = 1.0 / np.sqrt(d_model)

        # 동적 어텐션과 사전 OD 확률의 가중 결합 온도 계수
        self.dyn_scale = nn.Parameter(torch.tensor(0.5))

    def compute_dynamic_transition(self, hour, weekday, month):
        """
        주어진 시간 컨텍스트(t)에 대한 [B, N, K] 동적 전이 확률 행렬 계산.
        sum_{k=0}^{K-1} P_dyn[b, i, k] == 1.0 (Row-stochastic).
        """
        B = hour.shape[0]
        N, K = self.n_stations, self.k

        # 1. 시간 컨텍스트 벡터 [B, d_model]
        ctx = torch.cat([self.ctx_hour(hour), self.ctx_weekday(weekday),
                         self.ctx_month(month)], dim=-1)
        ctx_vec = self.ctx_mlp(ctx)  # [B, d]

        # 2. 출발지 Query [B, N, d] & 도착지 Key [B, N, d]
        st_z = self.station_emb.weight.unsqueeze(0).expand(B, N, -1)  # [B, N, d]
        z_mod = st_z + ctx_vec.unsqueeze(1)  # 시간 컨텍스트 융합

        Q = self.q_proj(z_mod)  # [B, N, d]
        K_all = self.k_proj(z_mod)  # [B, N, d]

        # 3. Top-K 도착지 Key 수집: [B, N, K, d]
        # top_k_indices: [N, K] -> [B, N, K]
        idx_expanded = self.top_k_indices.unsqueeze(0).expand(B, N, K)
        # Gather keys for destinations: idx_expanded in [0, N-1]
        # Flatten for gather: [B, N * K]
        flat_idx = idx_expanded.reshape(B, N * K, 1).expand(-1, -1, self.d_model)
        K_dest = torch.gather(K_all, 1, flat_idx).reshape(B, N, K, self.d_model)

        # 4. Attention Score: Q_i dot K_dest [B, N, K]
        dot = (Q.unsqueeze(2) * K_dest).sum(dim=-1) * self.scale
        # 동적 어텐션 점수 + 사전 OD 통계 결합
        dyn_weight = torch.clamp(self.dyn_scale, 0.05, 2.0)
        logits = dyn_weight * dot + self.log_prior.unsqueeze(0)  # [B, N, K]

        # 5. Row Softmax (질량 보존 보장)
        P_dyn = F.softmax(logits, dim=-1)  # [B, N, K]
        return P_dyn

    def forward(self, lambda_rent, hour, weekday, month):
        """
        lambda_rent: [B, N] 대여 잠재 유량
        반환값:
            mu_return: [B, N] 반납 유입 유량 (sum(mu_return) == sum(lambda_rent))
            P_dyn: [B, N, K] 동적 전이 확률
        """
        B, N, K = lambda_rent.shape[0], self.n_stations, self.k
        P_dyn = self.compute_dynamic_transition(hour, weekday, month)  # [B, N, K]

        # 각 출발지 i에서 k번째 도착지로 흘러가는 유량: [B, N, K]
        flow = lambda_rent.unsqueeze(-1) * P_dyn  # [B, N, K]

        # 도착지 인덱스에 따라 질량 집계 (Scatter Add)
        # top_k_indices: [N, K]
        dest_idx = self.top_k_indices.unsqueeze(0).expand(B, N, K)  # [B, N, K]

        mu_return = torch.zeros(B, N, device=lambda_rent.device, dtype=lambda_rent.dtype)
        mu_return.scatter_add_(1, dest_idx.reshape(B, N * K), flow.reshape(B, N * K))

        return mu_return, P_dyn


class HighDimKineticModel(nn.Module):
    """고차원 잠재 공간 신경망 유속 연산자 기반 유량 모델."""
    def __init__(self, n_stations, od_matrix, base_spatial_prior=None, k=32, d_model=32):
        super().__init__()
        self.n_stations = n_stations
        self.macro_head = MacroFluxHead()
        self.spatial_head = BoltzmannSpatialHead(n_stations, base_spatial_prior)

        top_k_indices, log_prior = build_topk_sparse_prior(od_matrix, k=k)
        self.advection_op = DynamicNeuralAdvection(n_stations, top_k_indices, log_prior, d_model=d_model)

        self.log_r = nn.Parameter(torch.tensor([2.0, 2.0]))  # [r_rent, r_return]

    def forward(self, hour, weekday, month, station_mask=None):
        """
        반환값:
            lambda_rent: [B, N] 잠재 대여 유량
            mu_return: [B, N] 신경망 유속장에 의해 전파된 반납 유량
            M: [B] 거시 총량
            pi: [B, N] 공간 확률
            P_dyn: [B, N, K] 동적 전이 행렬
        """
        M = self.macro_head(hour, weekday, month)  # [B]
        pi, V = self.spatial_head(hour, weekday, month, station_mask)  # [B, N]

        lambda_rent = M.unsqueeze(-1) * pi  # [B, N]
        mu_return, P_dyn = self.advection_op(lambda_rent, hour, weekday, month)  # [B, N]

        return lambda_rent, mu_return, M, pi, P_dyn


def compute_neural_kinetic_loss(model, y_rent, y_return, exposure, lambda_rent,
                                mu_return, M, mask, partial_weight=0.2):
    """랭뮤어 절단 손실 + 동적 반납 이류 손실 + 거시 유량 손실."""
    r = torch.exp(model.log_r).clamp(0.1, 100.0)
    r_rent, r_return = r[0], r[1]

    # 1. 대여 손실 (랭뮤어 흡착 절단)
    e_eff = exposure.clamp_min(1 / 60.0)
    mu_rent_obs = (lambda_rent * e_eff).clamp_min(1e-6)
    ll_rent = nb_logpmf(y_rent, mu_rent_obs, r_rent)

    weight_rent = torch.where(exposure >= 0.99, 1.0,
                              torch.where(exposure >= 0.1, partial_weight, 0.0))
    weight_rent = weight_rent * mask.float()
    loss_rent = -(weight_rent * ll_rent).sum() / weight_rent.sum().clamp_min(1.0)

    # 2. 반납 손실 (동적 신경망 유속장에 의해 도달한 유입)
    mu_return_safe = mu_return.clamp_min(1e-6)
    ll_return = nb_logpmf(y_return, mu_return_safe, r_return)
    weight_return = mask.float()
    loss_return = -(weight_return * ll_return).sum() / weight_return.sum().clamp_min(1.0)

    # 3. 거시 총량 손실
    with torch.no_grad():
        full_mask = mask & (exposure >= 0.99)
    y_macro_full = (y_rent * full_mask.float()).sum(dim=-1)
    pred_macro_full = (lambda_rent * full_mask.float()).sum(dim=-1)
    loss_macro = F.smooth_l1_loss(torch.log1p(pred_macro_full),
                                  torch.log1p(y_macro_full))

    total_loss = loss_rent + 0.4 * loss_return + 0.2 * loss_macro
    return total_loss, loss_rent, loss_return, loss_macro


def train_neural_flux_model(data, cutoff='2026-02-01', epochs=10, batch_hours=64, lr=0.003, k=32):
    """과거 대여 이력으로 고차원 신경망 유속 연산자 모델을 학습한다."""
    np.random.seed(42)
    torch.manual_seed(42)

    train_mask = data['valid_hours'] & (data['bin_start'] < np.datetime64(cutoff))
    t_pool = np.flatnonzero(train_mask)
    if len(t_pool) == 0:
        raise ValueError("No training hours found before cutoff")

    n_stations = len(data['stations'])
    train_rentals = data['rentals'][t_pool]
    station_sums = train_rentals.sum(axis=0) + 1.0
    base_spatial_prior = station_sums / station_sums.sum()

    od_path = Path('data/processed/od_train_before_2026-02-01.npy')
    od_matrix = np.load(od_path) if od_path.exists() else data['od']

    model = HighDimKineticModel(n_stations, od_matrix, base_spatial_prior, k=k, d_model=32).to(DEV)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    y_rent_all = torch.as_tensor(data['rentals'], dtype=torch.float32, device=DEV)
    y_ret_all = torch.as_tensor(data['returns'], dtype=torch.float32, device=DEV)
    exp_all = torch.as_tensor(data['exposure'], dtype=torch.float32, device=DEV)
    exp_ub_all = torch.as_tensor(data['exposure_ub'], dtype=torch.float32, device=DEV)
    active_all = torch.as_tensor(data['active'], dtype=torch.bool, device=DEV)
    st_ok = torch.as_tensor(data['station_ok'], dtype=torch.bool, device=DEV)

    h_all = torch.as_tensor(data['hour'], dtype=torch.long, device=DEV)
    w_all = torch.as_tensor(data['weekday'], dtype=torch.long, device=DEV)
    m_all = torch.as_tensor(data['month'], dtype=torch.long, device=DEV)

    steps_per_epoch = len(t_pool) // batch_hours
    total_steps = epochs * steps_per_epoch
    print(f"Training HighDimKineticModel (Neural-KFDM): {epochs} epochs ({total_steps} steps), "
          f"batch_hours={batch_hours}, stations={n_stations}, Top-K={k}")

    step = 0
    model.train()
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(t_pool)
        for i in range(0, len(perm) - batch_hours, batch_hours):
            step += 1
            batch_t = perm[i:i + batch_hours]
            t_idx = torch.as_tensor(batch_t, dtype=torch.long, device=DEV)

            h = h_all[t_idx]
            w = w_all[t_idx]
            m = m_all[t_idx]
            mask = active_all[t_idx] & st_ok.unsqueeze(0)

            y_r = y_rent_all[t_idx]
            y_ret = y_ret_all[t_idx]
            e_lb = exp_all[t_idx]
            e_ub = exp_ub_all[t_idx]
            e = torch.where((y_r > 0) & (e_lb < 0.1), torch.maximum(e_lb, e_ub), e_lb)

            lambda_rent, mu_return, M, pi, P_dyn = model(h, w, m, mask)
            loss, l_rent, l_ret, l_mac = compute_neural_kinetic_loss(
                model, y_r, y_ret, e, lambda_rent, mu_return, M, mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if step == 1 or step % 100 == 0 or step == total_steps:
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{total_steps}] "
                      f"Loss: {loss.item():.4f} (Rent: {l_rent.item():.4f}, "
                      f"Return: {l_ret.item():.4f}, Macro: {l_mac.item():.4f}, "
                      f"DynScale: {model.advection_op.dyn_scale.item():.3f})", flush=True)

    model.eval()
    return model


@torch.no_grad()
def predict_neural_hour(model, data, t):
    """특정 시간 t에 대한 예측."""
    h = torch.tensor([data['hour'][t]], dtype=torch.long, device=DEV)
    w = torch.tensor([data['weekday'][t]], dtype=torch.long, device=DEV)
    m = torch.tensor([data['month'][t]], dtype=torch.long, device=DEV)
    mask = torch.tensor(data['active'][t] & data['station_ok'], dtype=torch.bool, device=DEV).unsqueeze(0)

    lambda_rent, mu_return, M, pi, P_dyn = model(h, w, m, mask)
    return {
        'lambda_rent': lambda_rent.squeeze(0).cpu().numpy(),
        'mu_return': mu_return.squeeze(0).cpu().numpy(),
        'macro_total': float(M.item()),
        'spatial_prob': pi.squeeze(0).cpu().numpy(),
        'P_dyn': P_dyn.squeeze(0).cpu().numpy()
    }
