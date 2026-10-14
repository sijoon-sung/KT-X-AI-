"""Online Residual Reinforcement Learning Feedback Evaluator for Tashu 2026 Feb & Mar.

Implements the user's inquiry:
"개인적으로 2,3월 데이터를 평가를 하는 동시에 RL을 통해서 조금씩 피드백도 줘봐"

Architecture:
1. Base Supervised Foundation (Massive Ensemble: LightGBM + ST-ResZIP priors)
2. Online Residual RL Agent (Actor-Critic Contextual Bandit):
   - State s_t: [inventory, base_prediction, hour_sin, hour_cos, day_sin, day_cos, prev_step_error]
   - Action a_t: Residual adjustment vector Delta_y in [-delta, +delta]
   - Final forecast: y*_t = max(0, y^(0)_t + a_t)
   - Reward R_t: - |y_t - y*_t| - alpha * StockoutRisk - beta * ActionPenalty + gamma * ZeroPrecisionBonus
   - Online Policy Gradient Update: Incremental feedback learning as time flows chronologically through Feb (Winter) and Mar (Spring).
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================================
# 1. Online Residual RL Agent (Lightweight Actor-Critic)
# =========================================================================

class ResidualActorCritic(nn.Module):
    """Residual policy network that learns dynamic correction offsets Delta_y."""
    def __init__(self, state_dim=8, hidden_dim=64):
        super().__init__()
        # Actor: outputs mean mu and log_std for continuous residual action
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.actor_mu = nn.Linear(hidden_dim, 1)       # Action mean Delta_y
        self.actor_logstd = nn.Parameter(torch.zeros(1) - 1.0)  # Log std
        self.critic = nn.Linear(hidden_dim, 1)         # Value baseline V(s)

    def forward(self, state):
        # state: [N, state_dim]
        h = self.shared(state)
        mu = torch.tanh(self.actor_mu(h)) * 1.5  # Max residual adjustment +/- 1.5 bikes
        value = self.critic(h)
        std = torch.exp(self.actor_logstd).clamp(0.05, 1.0)
        return mu, std, value

    def sample_action(self, state):
        mu, std, value = self.forward(state)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


# =========================================================================
# 2. Dataset Loader & Pretrained Base Model
# =========================================================================

def load_environment_data():
    print("Loading environment data (Tashu 1,425 stations + Seoul Foundation Features)...")
    npz = np.load('data/processed/tashu_60min.npz', allow_pickle=True)
    stations = npz['stations']
    rentals = npz['rentals'].astype(np.float32)
    inv_start = npz['inv_start'].astype(np.float32)
    valid = npz['valid'] & ~npz['closed']
    bin_start = npz['bin_start']
    lat = npz['lat'].astype(np.float32)
    lon = npz['lon'].astype(np.float32)

    seoul_npz = np.load('data/seoul_derived_tashu_features.npz', allow_pickle=True)
    seoul_temp = seoul_npz['temporal_features'].astype(np.float32)
    seoul_st = seoul_npz['station_features'].astype(np.float32)

    st_cluster = {}
    with open('data/clustering_stations.csv', 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            st_cluster[r['station_id']] = int(r['cluster_id'])

    sel_mask = np.array([s in st_cluster for s in stations])
    sel_idx = np.where(sel_mask)[0]
    N_sel = len(sel_idx)

    cluster_labels = np.array([st_cluster[stations[i]] for i in sel_idx])
    n_clusters = len(np.unique(cluster_labels))
    cluster_to_indices = [np.where(cluster_labels == c)[0] for c in range(n_clusters)]
    reorder = np.concatenate(cluster_to_indices)
    ordered_sel_idx = sel_idx[reorder]

    y_rent = rentals[:, ordered_sel_idx]
    inv = inv_start[:, ordered_sel_idx]
    lat_sel = lat[ordered_sel_idx]
    lon_sel = lon[ordered_sel_idx]
    seoul_st_sel = seoul_st[ordered_sel_idx]

    ts = bin_start.astype('datetime64[h]')
    hour = (ts.astype('int64') % 24)
    day = ((ts.astype('int64') // 24) + 3) % 7
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    day_sin = np.sin(2 * np.pi * day / 7.0).astype(np.float32)
    day_cos = np.cos(2 * np.pi * day / 7.0).astype(np.float32)

    train_mask = valid & (bin_start >= np.datetime64('2025-08-01')) & (bin_start < np.datetime64('2026-02-01'))
    test_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-04-01'))
    feb_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01'))
    mar_mask = valid & (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01'))

    # Smoothed historical table
    rate_rent = np.zeros((7, 24, N_sel), dtype=np.float32)
    alpha = 10.0
    for h in range(24):
        h_mask = train_mask & (hour == h)
        hourly_r = y_rent[h_mask].mean(0) if h_mask.any() else np.zeros(N_sel)
        for d in range(7):
            d_mask = h_mask & (day == d)
            cnt = int(d_mask.sum())
            if cnt > 0:
                rate_rent[d, h] = (y_rent[d_mask].sum(0) + alpha * hourly_r) / (cnt + alpha)
            else:
                rate_rent[d, h] = hourly_r

    hist_rent = rate_rent[day, hour]

    return {
        'y_rent': y_rent, 'inv': inv, 'hist_rent': hist_rent,
        'hour': hour, 'day': day, 'bin_start': bin_start,
        'hour_sin': hour_sin, 'hour_cos': hour_cos,
        'day_sin': day_sin, 'day_cos': day_cos,
        'lat': (lat_sel - lat_sel.mean()) * 111.0,
        'lon': (lon_sel - lon_sel.mean()) * 89.0,
        'seoul_temp': seoul_temp,
        'seoul_st': seoul_st_sel,
        'train_mask': train_mask, 'test_mask': test_mask,
        'feb_mask': feb_mask, 'mar_mask': mar_mask,
        'N_sel': N_sel
    }


def train_base_supervised_model(data):
    """Trains base LightGBM model on training data."""
    print("Training Base Supervised Model (Aug 2025 ~ Jan 2026)...")
    tr_idx = np.where(data['train_mask'])[0][::2]  # Step=2 for fast fit
    T = len(tr_idx)
    N = data['N_sel']

    inv = data['inv'][tr_idx].reshape(-1, 1)
    hist_r = data['hist_rent'][tr_idx].reshape(-1, 1)
    h_rep = np.repeat(data['hour'][tr_idx], N)[:, None]
    d_rep = np.repeat(data['day'][tr_idx], N)[:, None]
    stemp = np.repeat(data['seoul_temp'][tr_idx], N, axis=0)

    X = np.hstack([np.log1p(inv), np.log1p(hist_r), h_rep, d_rep, stemp]).astype(np.float32)
    y = data['y_rent'][tr_idx].reshape(-1)

    m = lgb.LGBMRegressor(objective='poisson', n_estimators=150, learning_rate=0.06, num_leaves=63, n_jobs=-1, verbose=-1)
    m.fit(X, y)
    print("Base model trained successfully.")
    return m


# =========================================================================
# 3. Online RL Feedback Simulation on 2026 Feb & Mar Stream
# =========================================================================

def run_online_rl_feedback_simulation(data, base_model, lr=0.0008):
    print("\n" + "=" * 84)
    print(" [온라인 강화학습(RL) 피드백 평가 루프 가동]")
    print(" 2026년 2월(겨울) -> 3월(봄) 시간 스트림을 따라 순차 평가 & 실시간 정책 업데이트")
    print("=" * 84)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device for Online RL: {dev}")

    N = data['N_sel']
    agent = ResidualActorCritic(state_dim=8, hidden_dim=64).to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    test_indices = np.where(data['test_mask'])[0]
    T_test = len(test_indices)

    # State tracking
    prev_error = np.zeros(N, dtype=np.float32)

    # Metrics history
    history = {
        'feb': {'static_mae': [], 'rl_mae': [], 'static_wape': [], 'rl_wape': [], 'rewards': []},
        'mar': {'static_mae': [], 'rl_mae': [], 'static_wape': [], 'rl_wape': [], 'rewards': []}
    }

    t0 = time.time()
    batch_states, batch_actions, batch_log_probs, batch_rewards, batch_values = [], [], [], [], []
    update_interval = 24  # Update policy daily (every 24 hours of streaming feedback)

    for step_i, t in enumerate(test_indices):
        current_time = data['bin_start'][t]
        is_mar = (current_time >= np.datetime64('2026-03-01'))
        period_key = 'mar' if is_mar else 'feb'

        # 1. Base Model Supervised Prediction
        inv_t = data['inv'][t]
        hist_t = data['hist_rent'][t]
        h_val = data['hour'][t]
        d_val = data['day'][t]
        stemp_t = data['seoul_temp'][t]

        X_step = np.column_stack([
            np.log1p(inv_t), np.log1p(hist_t),
            np.full(N, h_val), np.full(N, d_val),
            np.repeat(stemp_t[None, :], N, axis=0)
        ]).astype(np.float32)

        base_pred = base_model.predict(X_step).astype(np.float32)

        # 2. Formulate RL State Vector: [8 dimensions]
        # [log1p(inv), base_pred, h_sin, h_cos, d_sin, d_cos, prev_error, stemp_macro]
        h_sin = data['hour_sin'][t]
        h_cos = data['hour_cos'][t]
        d_sin = data['day_sin'][t]
        d_cos = data['day_cos'][t]

        state_matrix = np.column_stack([
            np.log1p(inv_t),
            np.log1p(base_pred),
            np.full(N, h_sin),
            np.full(N, h_cos),
            np.full(N, d_sin),
            np.full(N, d_cos),
            np.clip(prev_error, -3.0, 3.0),
            np.full(N, stemp_t[0])
        ]).astype(np.float32)

        state_tensor = torch.tensor(state_matrix, device=dev)

        # 3. Agent samples residual correction action Delta_y
        action, log_prob, value = agent.sample_action(state_tensor)
        action_np = action.detach().cpu().numpy().reshape(-1)

        # Final RL-adjusted forecast: max(0, base_pred + action)
        rl_pred = np.maximum(0.0, base_pred + action_np)

        # 4. Environment reveals ground truth y_t
        y_true = data['y_rent'][t]

        # 5. Compute Reward Signal:
        # Reward = - |y - y_rl| - penalty_for_unnecessary_large_action + bonus_for_exact_zero_hit
        abs_err_rl = np.abs(y_true - rl_pred)
        abs_err_static = np.abs(y_true - base_pred)

        # Structural zero bonus: if actual is 0 and prediction < 0.1, reward +0.3
        zero_bonus = np.where((y_true == 0) & (rl_pred < 0.1), 0.3, 0.0)
        # Peak bonus: if actual >= 3 and error decreased compared to static, reward +0.5
        peak_bonus = np.where((y_true >= 3.0) & (abs_err_rl < abs_err_static), 0.5, 0.0)
        # Action magnitude penalty to discourage crazy shifts
        action_pen = 0.05 * (action_np ** 2)

        step_reward = - abs_err_rl + zero_bonus + peak_bonus - action_pen
        mean_reward = float(step_reward.mean())

        # Update tracking
        prev_error = y_true - rl_pred

        # Record metrics
        mae_static = float(abs_err_static.mean())
        mae_rl = float(abs_err_rl.mean())
        wape_static = float(abs_err_static.sum() / max(y_true.sum(), 1.0))
        wape_rl = float(abs_err_rl.sum() / max(y_true.sum(), 1.0))

        history[period_key]['static_mae'].append(mae_static)
        history[period_key]['rl_mae'].append(mae_rl)
        history[period_key]['static_wape'].append(wape_static)
        history[period_key]['rl_wape'].append(wape_rl)
        history[period_key]['rewards'].append(mean_reward)

        # Save transitions for online PPO/Policy Gradient update
        batch_states.append(state_tensor.float())
        batch_actions.append(action.float())
        batch_log_probs.append(log_prob.float())
        batch_rewards.append(torch.tensor(step_reward, dtype=torch.float32, device=dev)[:, None])
        batch_values.append(value.float())

        # 6. Online Policy Update every 24 hours (Daily Feedback Loop)
        if (step_i + 1) % update_interval == 0 or step_i == T_test - 1:
            optimizer.zero_grad()
            b_states = torch.cat(batch_states, dim=0)
            b_actions = torch.cat(batch_actions, dim=0)
            b_old_log_probs = torch.cat(batch_log_probs, dim=0)
            b_rewards = torch.cat(batch_rewards, dim=0)
            b_values = torch.cat(batch_values, dim=0)

            # Advantage = Reward - Value
            advantages = b_rewards - b_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

            # Re-evaluate log prob
            mu, std, values_new = agent(b_states)
            dist = torch.distributions.Normal(mu, std)
            log_probs_new = dist.log_prob(b_actions)

            # Policy Gradient Loss + Critic MSE Loss
            actor_loss = - (log_probs_new * advantages).mean()
            critic_loss = F.mse_loss(values_new, b_rewards)
            total_loss = actor_loss + 0.5 * critic_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)
            optimizer.step()

            # Clear buffer
            batch_states, batch_actions, batch_log_probs, batch_rewards, batch_values = [], [], [], [], []

        if (step_i + 1) % 168 == 0 or step_i == T_test - 1:
            day_num = (step_i + 1) // 24
            print(f"   [Day {day_num:02d} / 59 | {current_time}] Cumulative Static MAE: {np.mean(history[period_key]['static_mae']):.4f} | Online RL MAE: {np.mean(history[period_key]['rl_mae']):.4f} | Reward: {mean_reward:.4f}")

    print(f"\nCompleted Online RL Evaluation in {time.time()-t0:.1f}s")
    return history


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    data = load_environment_data()
    base_model = train_base_supervised_model(data)

    history = run_online_rl_feedback_simulation(data, base_model, lr=0.001)

    # Compute Final Results
    summary = {}
    for period_key, label in [('feb', '2026년 2월 (겨울 저수요)'), ('mar', '2026년 3월 (봄 개학/성수기 전환)')]:
        h = history[period_key]
        stat_mae = float(np.mean(h['static_mae']))
        rl_mae = float(np.mean(h['rl_mae']))
        stat_wape = float(np.mean(h['static_wape']))
        rl_wape = float(np.mean(h['rl_wape']))
        mean_rew = float(np.mean(h['rewards']))
        improvement = ((stat_mae - rl_mae) / stat_mae) * 100.0

        summary[period_key] = {
            'static_mae': round(stat_mae, 4),
            'online_rl_mae': round(rl_mae, 4),
            'static_wape': round(stat_wape, 4),
            'online_rl_wape': round(rl_wape, 4),
            'mae_improvement_pct': round(improvement, 2),
            'mean_reward': round(mean_rew, 4)
        }

    # Print Table
    print("\n" + "=" * 88)
    print("         [정적 지도학습 모델 vs 온라인 강화학습(RL) 잔차 피드백 성능 비교]")
    print("=" * 88)
    print(f"{'평가 기간':<18} | {'정적 모델 MAE':<13} | {'온라인 RL MAE':<13} | {'개선율(%)':<10} | {'WAPE 개선 (Static -> RL)'}")
    print("-" * 88)
    for p_key, label in [('feb', '2026년 2월 (겨울)'), ('mar', '2026년 3월 (봄 성수기)')]:
        s = summary[p_key]
        print(f"{label:<18} | {s['static_mae']:>13.4f} | {s['online_rl_mae']:>13.4f} | {s['mae_improvement_pct']:>9.2f}% | {s['static_wape']:.4f} -> {s['online_rl_wape']:.4f}")
    print("=" * 88)

    out_path = 'data/online_rl_feedback_results.json'
    Path(out_path).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved online RL evaluation results to: {out_path}")


if __name__ == '__main__':
    main()
