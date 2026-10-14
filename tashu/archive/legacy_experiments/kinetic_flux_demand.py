"""Kinetic Flux Demand Model (KFDM): 물리학적 유량 보존과 화학 흡착 반응론 기반의 수요 및 최적분포 진단기.

핵심 원리:
1. 거시 활동도 헤드 (Macro Thermal Activity M(t)):
   - 도시 전체의 총 유량(기온/시간대/계절 등 열역학적 거시 상태)을 예측하여 스케일 발산 방지.
2. 미시 볼츠만 공간 퍼텐셜 헤드 (Micro Boltzmann Potential pi_i(t)):
   - 각 대여소의 공간 잠재력 V_i(t)를 계산하고 Softmax 정규화하여 sum(pi_i) = 1.0 보장.
   - 잠재 수요 강도: Lambda_i(t) = M(t) * pi_i(t).
3. 랭뮤어 흡착식 절단 우도 (Censored Adsorption Likelihood):
   - 재고 노출도 e_it에 따른 관측 절단 반영: Y_rent ~ NegativeBinomial(Lambda * e, r).
4. OD 유체 이류 연산자 (Continuity Advection for Returns):
   - 반납은 독립 예측 대신 OD 전이 행렬을 통한 물리적 이류로 생성: mu_return = P_OD^T * Lambda_rent.
   - 대여 총량과 반납 총량의 완벽한 질량 보존 보장: sum(mu_return) == sum(Lambda_rent) == M(t).
5. 열역학적 분포 최적도 진단기 (Distribution Optimality & Free Energy):
   - 각 대여소의 순유량 발산(Divergence: Div_i = Lambda_i - mu_i)과 현재 재고를 비교.
   - 품절(Deficit) 및 만차(Excess) 불균형 에너지를 산출하여 "분포 최적도 지수(0~100%)" 계산.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diagnose_forecast import fit_historical_rate

torch.set_num_threads(os.cpu_count() or 4)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def calendar_features(bin_start):
    """datetime64 배열로부터 시간, 요일, 월 및 주말 여부를 추출한다."""
    days = bin_start.astype('datetime64[D]')
    hour = (bin_start - days).astype('timedelta64[h]').astype(np.int64)
    weekday = (days.astype(np.int64) + 3) % 7
    month = (bin_start.astype('datetime64[M]').astype(np.int64) % 12)
    is_weekend = (weekday >= 5).astype(np.int64)
    return hour, weekday, month, is_weekend


def nb_logpmf(y, mu, r):
    """Negative-Binomial Log PMF."""
    return (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1)
            + r * (torch.log(r) - torch.log(r + mu))
            + y * (torch.log(mu) - torch.log(r + mu)))


def build_od_transition_kernel(od_matrix, smoothing=0.05):
    """
    OD 통계로부터 행 정규화된 마르코프 유속 전이 확률 행렬 P_OD를 구축한다.
    P_OD[i, j] = 대여소 i에서 대여된 자전거가 대여소 j로 반납될 확률.
    sum_j P_OD[i, j] = 1.0 (질량 보존).
    """
    n = len(od_matrix)
    A = od_matrix.astype(np.float32)
    A_smoothed = A + smoothing
    row_sums = A_smoothed.sum(axis=1, keepdims=True)
    P_OD = A_smoothed / np.maximum(row_sums, 1e-7)
    return torch.from_numpy(P_OD).float()


class MacroFluxHead(nn.Module):
    """도시 전체의 거시적 총 잠재 유량 M(t)를 예측하는 열역학적 상태 헤드."""
    def __init__(self):
        super().__init__()
        self.hour_emb = nn.Embedding(24, 12)
        self.weekday_emb = nn.Embedding(7, 6)
        self.month_emb = nn.Embedding(12, 6)
        # 12 + 6 + 6 + 4 (cyclical sine/cosine of hour and month) = 28
        self.net = nn.Sequential(
            nn.Linear(28, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        nn.init.constant_(self.net[-1].bias, 5.8)

    def forward(self, hour, weekday, month):
        h_rad = 2.0 * np.pi * hour.float() / 24.0
        m_rad = 2.0 * np.pi * month.float() / 12.0
        cyclical = torch.stack([torch.sin(h_rad), torch.cos(h_rad),
                                torch.sin(m_rad), torch.cos(m_rad)], dim=-1)
        feat = torch.cat([self.hour_emb(hour), self.weekday_emb(weekday),
                          self.month_emb(month), cyclical], dim=-1)
        log_m = self.net(feat).squeeze(-1)
        return F.softplus(log_m) + 1.0


class BoltzmannSpatialHead(nn.Module):
    """각 대여소의 상대적 공간 잠재력 V_i(t)를 계산하고 Softmax 확률 pi_i(t)를 산출."""
    def __init__(self, n_stations, base_spatial_prior=None):
        super().__init__()
        self.n_stations = n_stations
        self.station_emb = nn.Embedding(n_stations, 24)
        self.hour_emb = nn.Embedding(24, 8)
        self.weekday_emb = nn.Embedding(7, 4)
        self.month_emb = nn.Embedding(12, 4)

        if base_spatial_prior is not None:
            prior = torch.as_tensor(base_spatial_prior, dtype=torch.float32)
            init_bias = torch.log(prior.clamp_min(1e-5))
            self.station_bias = nn.Parameter(init_bias)
        else:
            self.station_bias = nn.Parameter(torch.zeros(n_stations))

        self.interaction = nn.Sequential(
            nn.Linear(24 + 8 + 4 + 4, 48),
            nn.GELU(),
            nn.Linear(48, 1)
        )
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)

    def forward(self, hour, weekday, month, station_mask=None):
        B = hour.shape[0]
        N = self.n_stations
        time_feat = torch.cat([self.hour_emb(hour), self.weekday_emb(weekday),
                               self.month_emb(month)], dim=-1).unsqueeze(1).expand(B, N, -1)
        st_feat = self.station_emb.weight.unsqueeze(0).expand(B, N, -1)

        combined = torch.cat([st_feat, time_feat], dim=-1)
        delta_v = self.interaction(combined).squeeze(-1)  # [B, N]
        V = self.station_bias.unsqueeze(0) + delta_v  # [B, N]

        if station_mask is not None:
            V = V.masked_fill(~station_mask, -1e9)

        pi = F.softmax(V, dim=-1)
        return pi, V


class KineticFluxDemandModel(nn.Module):
    """동역학적 유량 및 연속 방정식 결합 모델."""
    def __init__(self, n_stations, od_matrix, base_spatial_prior=None):
        super().__init__()
        self.n_stations = n_stations
        self.macro_head = MacroFluxHead()
        self.spatial_head = BoltzmannSpatialHead(n_stations, base_spatial_prior)

        p_od = build_od_transition_kernel(od_matrix)
        self.register_buffer('P_OD', p_od)

        self.log_r = nn.Parameter(torch.tensor([2.0, 2.0]))  # [r_rent, r_return]

    def forward(self, hour, weekday, month, station_mask=None):
        M = self.macro_head(hour, weekday, month)  # [B]
        pi, V = self.spatial_head(hour, weekday, month, station_mask)  # [B, N]

        # 대여 잠재 수요 강도
        lambda_rent = M.unsqueeze(-1) * pi  # [B, N]

        # 물리적 유체 이류: mu_return = lambda_rent @ P_OD
        mu_return = torch.matmul(lambda_rent, self.P_OD)  # [B, N]

        return lambda_rent, mu_return, M, pi


def compute_kinetic_loss(model, y_rent, y_return, exposure, lambda_rent, mu_return,
                         M, mask, partial_weight=0.2):
    """
    랭뮤어 흡착식 절단 우도 + 거시 유량 손실 + 질량 보존 반납 손실을 계산한다.
    """
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

    # 2. 반납 손실 (OD 이류 보존 유동)
    mu_return_safe = mu_return.clamp_min(1e-6)
    ll_return = nb_logpmf(y_return, mu_return_safe, r_return)
    weight_return = mask.float()
    loss_return = -(weight_return * ll_return).sum() / weight_return.sum().clamp_min(1.0)

    # 3. 거시 총량 보정 손실 (완전 노출 지점들의 합을 통한 거시 스케일 안정화)
    with torch.no_grad():
        full_mask = mask & (exposure >= 0.99)
    y_macro_full = (y_rent * full_mask.float()).sum(dim=-1)
    pred_macro_full = (lambda_rent * full_mask.float()).sum(dim=-1)
    loss_macro = F.smooth_l1_loss(torch.log1p(pred_macro_full),
                                  torch.log1p(y_macro_full))

    total_loss = loss_rent + 0.3 * loss_return + 0.2 * loss_macro
    return total_loss, loss_rent, loss_return, loss_macro


def evaluate_distribution_optimality(lambda_rent, mu_return, current_inventory,
                                     capacities=None, safety_margin=2.0):
    """
    열역학적 불균형 및 분포 최적도(Distribution Optimality) 진단기.

    반환값:
        dict:
            divergence: [N] 순유량 발산 (Sink: 양수, Source: 음수)
            optimal_inventory: [N] 이상적인 완충 재고 분포 I*
            rebalance_priority: [N] 재배치 필요량 (양수: 투입 필요, 음수: 수거 필요)
            expected_deficit: 총 예상 품절 결손 건수
            expected_excess: 총 예상 만차 잉여 건수
            optimality_score: 분포 최적화 지수 (0 ~ 100%)
            is_optimal: 최적도 85% 이상 여부
    """
    n = len(lambda_rent)
    if capacities is None:
        capacities = np.full(n, 20.0, dtype=np.float32)

    # 1. 순유량 발산 (Divergence = Outflow - Inflow)
    div = lambda_rent - mu_return

    # 2. 이상적 완충 재고 I*_i 계산
    opt_inv = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if div[i] > 0:
            opt_inv[i] = min(capacities[i], div[i] + safety_margin)
        else:
            opt_inv[i] = max(0.0, capacities[i] - (abs(div[i]) + safety_margin))
        if abs(div[i]) < 0.5:
            opt_inv[i] = capacities[i] * 0.4

    # 3. 현재 재고 하에서의 불균형 손실 에너지 계산
    deficit = np.maximum(0.0, lambda_rent - current_inventory)
    expected_end_inv = current_inventory - lambda_rent + mu_return
    excess = np.maximum(0.0, expected_end_inv - capacities)

    total_deficit = float(deficit.sum())
    total_excess = float(excess.sum())
    total_flow = float(lambda_rent.sum() + mu_return.sum() + 1e-6)
    total_imbalance = total_deficit + total_excess

    optimality_score = max(0.0, 100.0 * (1.0 - (total_imbalance / total_flow)))
    rebalance_priority = opt_inv - current_inventory

    return {
        'divergence': div,
        'optimal_inventory': opt_inv,
        'rebalance_priority': rebalance_priority,
        'expected_deficit': total_deficit,
        'expected_excess': total_excess,
        'total_flow': total_flow,
        'optimality_score': optimality_score,
        'is_optimal': bool(optimality_score >= 85.0)
    }


def load_dataset(path='data/processed/tashu_60min.npz'):
    with np.load(path) as f:
        data = {k: f[k] for k in ('rentals', 'returns', 'exposure', 'exposure_ub',
                                  'inv_start', 'inv_ub', 'active', 'valid', 'closed',
                                  'bin_start', 'stations', 'od')}
    hour, weekday, month, is_weekend = calendar_features(data['bin_start'])
    data['hour'] = hour
    data['weekday'] = weekday
    data['month'] = month
    data['is_weekend'] = is_weekend
    data['station_ok'] = ~np.isin(data['stations'].astype(str), NON_CUSTOMER)
    data['valid_hours'] = data['valid'] & ~data['closed']
    return data


def train_kinetic_model(data, cutoff='2026-02-01', epochs=12, batch_hours=64, lr=0.003):
    """과거 대여 이력으로 동역학적 유량 모델을 학습한다."""
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
    if od_path.exists():
        od_matrix = np.load(od_path)
    else:
        od_matrix = data['od']

    model = KineticFluxDemandModel(n_stations, od_matrix, base_spatial_prior).to(DEV)
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
    print(f"Training KineticFluxDemandModel: {epochs} epochs ({total_steps} steps), "
          f"batch_hours={batch_hours}, stations={n_stations}")

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

            lambda_rent, mu_return, M, pi = model(h, w, m, mask)
            loss, l_rent, l_ret, l_mac = compute_kinetic_loss(
                model, y_r, y_ret, e, lambda_rent, mu_return, M, mask
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if step == 1 or step % 100 == 0 or step == total_steps:
                print(f"Epoch [{epoch}/{epochs}] Step [{step}/{total_steps}] "
                      f"Loss: {loss.item():.4f} (Rent: {l_rent.item():.4f}, "
                      f"Return: {l_ret.item():.4f}, Macro: {l_mac.item():.4f})", flush=True)

    model.eval()
    return model


@torch.no_grad()
def predict_kinetic_hour(model, data, t):
    """특정 시간 t에 대한 잠재 대여 수요, 반납 유입, 유량 발산 예측."""
    h = torch.tensor([data['hour'][t]], dtype=torch.long, device=DEV)
    w = torch.tensor([data['weekday'][t]], dtype=torch.long, device=DEV)
    m = torch.tensor([data['month'][t]], dtype=torch.long, device=DEV)
    mask = torch.tensor(data['active'][t] & data['station_ok'], dtype=torch.bool, device=DEV).unsqueeze(0)

    lambda_rent, mu_return, M, pi = model(h, w, m, mask)
    return {
        'lambda_rent': lambda_rent.squeeze(0).cpu().numpy(),
        'mu_return': mu_return.squeeze(0).cpu().numpy(),
        'macro_total': float(M.item()),
        'spatial_prob': pi.squeeze(0).cpu().numpy()
    }
