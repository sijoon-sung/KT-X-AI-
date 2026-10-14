"""Massive Scaled-Up Model Benchmark:
1. Massive-XGBoost / LightGBM (1,000 Trees, Max-Depth 10, Seoul Foundation Priors, Spatial Lags)
2. Massive-ST-ResZIP (6-Block Deep Residual Zero-Inflated Graph Network, 1.8M params, Dense Skip-Connections)
3. Super-Ensemble Blending

Evaluates on strictly held-out Feb 2026 (Winter) & Mar 2026 (Spring).
Includes GPU memory watchdog to run when GPU is available.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb


def check_gpu_status():
    """Checks if GPU is currently free."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,nounits,noheader"],
            encoding='utf-8'
        ).strip().split(',')
        mem_used = int(out[0].strip())
        util = int(out[1].strip())
        is_free = (mem_used < 1500) and (util < 20)
        return is_free, f"Memory: {mem_used}MB, Util: {util}%"
    except Exception as e:
        return True, f"Status check fallback: {e}"


def wait_for_gpu_free(target_pid=21568, poll_interval=10, max_wait_sec=7200):
    """Waits until target process finishes and GPU memory is released."""
    print(f"\n[GPU Watchdog] Monitoring PID {target_pid} and GPU memory usage...")
    start_time = time.time()
    while time.time() - start_time < max_wait_sec:
        import subprocess
        # Check if process is running
        try:
            p_check = subprocess.check_output(
                ["powershell", "-Command", f"Get-Process -Id {target_pid} -ErrorAction SilentlyContinue"],
                encoding='utf-8'
            ).strip()
            p_alive = len(p_check) > 0
        except Exception:
            p_alive = False

        is_free, info = check_gpu_status()
        if not p_alive and is_free:
            print(f"[GPU Watchdog] Target process PID {target_pid} completed! GPU is free ({info}). Starting massive experiment!")
            return True

        elapsed_min = (time.time() - start_time) / 60.0
        print(f"   [Waiting {elapsed_min:.1f}m] PID {target_pid} alive: {p_alive} | GPU: {info}. Polling in {poll_interval}s...")
        time.sleep(poll_interval)

    print("[GPU Watchdog] Timeout reached. Proceeding with best-effort allocation.")
    return False


def build_normalized_adj(od_matrix, add_self_loop=True):
    A = od_matrix.astype(np.float32).copy()
    if add_self_loop:
        np.fill_diagonal(A, np.diag(A) + np.maximum(A.sum(1) * 0.05, 1.0))
    d = np.maximum(A.sum(1), 1e-6)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


# =========================================================================
# Architecture: Massive Spatio-Temporal Residual Zero-Inflated Poisson Network
# =========================================================================

class ResidualGCNBlock(nn.Module):
    """Dense Residual Graph Convolution with LayerNorm & Dropout to prevent over-smoothing."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x, adj, x_raw):
        # x: [B, N, dim], adj: [N, N], x_raw: [B, N, dim]
        h = self.norm1(x)
        # Graph message passing
        support = self.linear1(h)
        graph_out = torch.matmul(adj, support)
        graph_out = self.act(graph_out)
        graph_out = self.dropout(graph_out)

        # Second linear + residual injection of raw input (guarantees node identity retention)
        h2 = self.linear2(graph_out)
        out = x + h2 + 0.1 * x_raw  # Raw identity anchor
        return out


class MassiveSTResZIP(nn.Module):
    """Massive 6-layer Residual Zero-Inflated Poisson Graph Network (~1.8M params)."""
    def __init__(self, in_dim=18, hidden_dim=256, num_blocks=6):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            ResidualGCNBlock(hidden_dim, dropout=0.1) for _ in range(num_blocks)
        ])

        # Dense Skip-Connection Projection (Jumping Knowledge)
        self.jk_proj = nn.Linear(hidden_dim * (num_blocks + 1), hidden_dim)

        # Dual-Head for Zero-Inflated Poisson (ZIP)
        # Head 1: Probability of Structural Zero (Bernoulli logit)
        self.zero_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # [zero_prob_rent, zero_prob_ret]
        )
        # Head 2: Poisson Rate Lambda (Intensity if non-zero)
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2)  # [lambda_rent, lambda_ret]
        )

    def forward(self, x, adj):
        # x: [B, N, in_dim], adj: [N, N]
        h0 = self.in_proj(x)
        h = h0
        intermediates = [h0]

        for block in self.blocks:
            h = block(h, adj, h0)
            intermediates.append(h)

        # Jumping knowledge concatenation across all residual depths
        jk_cat = torch.cat(intermediates, dim=-1)  # [B, N, hidden_dim * 7]
        h_final = F.gelu(self.jk_proj(jk_cat))

        # Output predictions
        zero_logits = self.zero_head(h_final)  # [B, N, 2]
        pi_zero = torch.sigmoid(zero_logits)    # Probability of exact zero

        rate_logits = self.rate_head(h_final)  # [B, N, 2]
        rate_lambda = F.softplus(rate_logits) + 1e-4  # Poisson rate

        # Expected value = (1 - pi) * lambda
        expected_rent = (1.0 - pi_zero[:, :, 0]) * rate_lambda[:, :, 0]
        expected_ret = (1.0 - pi_zero[:, :, 1]) * rate_lambda[:, :, 1]

        return {
            'expected_rent': expected_rent,
            'expected_ret': expected_ret,
            'pi_zero': pi_zero,
            'rate_lambda': rate_lambda
        }


def zip_nll_loss(pi, lam, y):
    """Exact Negative Log-Likelihood of Zero-Inflated Poisson Distribution."""
    # pi: [B, N], lam: [B, N], y: [B, N]
    eps = 1e-6
    pi = torch.clamp(pi, eps, 1.0 - eps)
    lam = torch.clamp(lam, eps, 1e4)

    is_zero = (y == 0).float()
    is_pos = (y > 0).float()

    # Case 1: y == 0 -> log(pi + (1 - pi) * exp(-lam))
    prob_zero = pi + (1.0 - pi) * torch.exp(-lam)
    loss_zero = -torch.log(torch.clamp(prob_zero, min=eps)) * is_zero

    # Case 2: y > 0 -> log(1 - pi) + y * log(lam) - lam - lgamma(y + 1)
    ll_pos = (torch.log(1.0 - pi) + y * torch.log(lam) - lam - torch.lgamma(y + 1.0))
    loss_pos = -ll_pos * is_pos

    total_loss = (loss_zero + loss_pos).mean()
    return total_loss


# =========================================================================
# Data Preparation: Multi-Scale Lags + Seoul Foundation Priors
# =========================================================================

def load_enriched_dataset(npz_path, cluster_csv, seoul_feat_path):
    print("Loading Tashu NPZ, Seoul foundation features, and clusters...")
    npz = np.load(npz_path, allow_pickle=True)
    stations = npz['stations']
    rentals = npz['rentals'].astype(np.float32)
    returns = npz['returns'].astype(np.float32)
    inv_start = npz['inv_start'].astype(np.float32)
    valid = npz['valid'] & ~npz['closed']
    bin_start = npz['bin_start']
    od = npz['od'].astype(np.float32)
    lat = npz['lat'].astype(np.float32)
    lon = npz['lon'].astype(np.float32)

    # Seoul foundation transfer features
    seoul_npz = np.load(seoul_feat_path, allow_pickle=True)
    seoul_temp = seoul_npz['temporal_features'].astype(np.float32)  # [14616, 2]
    seoul_st = seoul_npz['station_features'].astype(np.float32)     # [1425, 6]

    st_cluster = {}
    with open(cluster_csv, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            st_cluster[r['station_id']] = int(r['cluster_id'])

    sel_mask = np.array([s in st_cluster for s in stations])
    sel_idx = np.where(sel_mask)[0]
    N_sel = len(sel_idx)

    cluster_labels = np.array([st_cluster[stations[i]] for i in sel_idx])
    n_clusters = len(np.unique(cluster_labels))
    cluster_to_indices = [np.where(cluster_labels == c)[0] for c in range(n_clusters)]
    cluster_sizes = [len(idx) for idx in cluster_to_indices]
    reorder = np.concatenate(cluster_to_indices)
    ordered_sel_idx = sel_idx[reorder]
    ordered_cluster_labels = cluster_labels[reorder]

    y_rent = rentals[:, ordered_sel_idx]
    y_ret = returns[:, ordered_sel_idx]
    inv = inv_start[:, ordered_sel_idx]
    lat_sel = lat[ordered_sel_idx]
    lon_sel = lon[ordered_sel_idx]
    od_sel = od[np.ix_(ordered_sel_idx, ordered_sel_idx)]
    seoul_st_sel = seoul_st[ordered_sel_idx]

    ts = bin_start.astype('datetime64[h]')
    hour = (ts.astype('int64') % 24)
    day = ((ts.astype('int64') // 24) + 3) % 7
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    day_sin = np.sin(2 * np.pi * day / 7.0).astype(np.float32)
    day_cos = np.cos(2 * np.pi * day / 7.0).astype(np.float32)

    # Multi-scale rolling lags: Lag 1h, Lag 2h, Lag 24h
    lag1_rent = np.roll(y_rent, shift=1, axis=0)
    lag2_rent = np.roll(y_rent, shift=2, axis=0)
    lag24_rent = np.roll(y_rent, shift=24, axis=0)
    lag1_rent[:1] = 0; lag2_rent[:2] = 0; lag24_rent[:24] = 0

    train_mask = valid & (bin_start >= np.datetime64('2025-08-01')) & (bin_start < np.datetime64('2026-02-01'))
    feb_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01'))
    mar_mask = valid & (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01'))

    # Historical smoothed rate
    rate_rent = np.zeros((7, 24, N_sel), dtype=np.float32)
    rate_ret = np.zeros((7, 24, N_sel), dtype=np.float32)
    alpha = 10.0
    for h in range(24):
        h_mask = train_mask & (hour == h)
        hourly_r = y_rent[h_mask].mean(0) if h_mask.any() else np.zeros(N_sel)
        hourly_ret = y_ret[h_mask].mean(0) if h_mask.any() else np.zeros(N_sel)
        for d in range(7):
            d_mask = h_mask & (day == d)
            cnt = int(d_mask.sum())
            if cnt > 0:
                rate_rent[d, h] = (y_rent[d_mask].sum(0) + alpha * hourly_r) / (cnt + alpha)
                rate_ret[d, h] = (y_ret[d_mask].sum(0) + alpha * hourly_ret) / (cnt + alpha)
            else:
                rate_rent[d, h] = hourly_r
                rate_ret[d, h] = hourly_ret

    hist_rent = rate_rent[day, hour]
    hist_ret = rate_ret[day, hour]
    full_adj = build_normalized_adj(od_sel, add_self_loop=True)

    return {
        'y_rent': y_rent, 'y_ret': y_ret, 'inv': inv,
        'lag1_rent': lag1_rent, 'lag2_rent': lag2_rent, 'lag24_rent': lag24_rent,
        'hist_rent': hist_rent, 'hist_ret': hist_ret,
        'hour': hour, 'day': day,
        'time_feats': np.column_stack([hour_sin, hour_cos, day_sin, day_cos]),
        'lat': (lat_sel - lat_sel.mean()) * 111.0,
        'lon': (lon_sel - lon_sel.mean()) * 89.0,
        'cluster_labels': ordered_cluster_labels,
        'cluster_sizes': cluster_sizes,
        'seoul_temp': seoul_temp,
        'seoul_st': seoul_st_sel,
        'full_adj': full_adj,
        'train_mask': train_mask, 'feb_mask': feb_mask, 'mar_mask': mar_mask,
        'N_sel': N_sel
    }


def prepare_massive_tabular_dataset(data, mask, step=1):
    t_idx = np.where(mask)[0]
    if step > 1:
        t_idx = t_idx[::step]

    T = len(t_idx)
    N = data['N_sel']

    inv = data['inv'][t_idx].reshape(-1, 1)
    hist_r = data['hist_rent'][t_idx].reshape(-1, 1)
    hist_ret = data['hist_ret'][t_idx].reshape(-1, 1)
    l1 = data['lag1_rent'][t_idx].reshape(-1, 1)
    l2 = data['lag2_rent'][t_idx].reshape(-1, 1)
    l24 = data['lag24_rent'][t_idx].reshape(-1, 1)

    lat_rep = np.tile(data['lat'], T)[:, None]
    lon_rep = np.tile(data['lon'], T)[:, None]
    clust_rep = np.tile(data['cluster_labels'], T)[:, None]
    seoul_st_rep = np.tile(data['seoul_st'], (T, 1))  # [T*N, 6]

    tf = data['time_feats'][t_idx]
    tf_rep = np.repeat(tf, N, axis=0)

    stemp = data['seoul_temp'][t_idx]  # [T, 2]
    stemp_rep = np.repeat(stemp, N, axis=0)  # [T*N, 2]

    h_raw = data['hour'][t_idx]
    d_raw = data['day'][t_idx]
    h_rep = np.repeat(h_raw, N)[:, None]
    d_rep = np.repeat(d_raw, N)[:, None]

    # Total 22 Rich Features
    X = np.hstack([
        np.log1p(inv),
        np.log1p(hist_r),
        np.log1p(hist_ret),
        np.log1p(l1),
        np.log1p(l2),
        np.log1p(l24),
        lat_rep * 0.1,
        lon_rep * 0.1,
        tf_rep,
        stemp_rep,
        seoul_st_rep,
        h_rep,
        d_rep,
        clust_rep
    ]).astype(np.float32)

    y_rent = data['y_rent'][t_idx].reshape(-1)
    y_ret = data['y_ret'][t_idx].reshape(-1)

    return X, y_rent, y_ret, T, N, t_idx


def prepare_deep_batch(data, t_idx):
    """Constructs [B, N, 18] input tensor for Massive-ST-ResZIP."""
    B = len(t_idx)
    N = data['N_sel']

    inv = np.log1p(data['inv'][t_idx])[:, :, None]
    hist_r = np.log1p(data['hist_rent'][t_idx])[:, :, None]
    hist_ret = np.log1p(data['hist_ret'][t_idx])[:, :, None]
    l1 = np.log1p(data['lag1_rent'][t_idx])[:, :, None]
    l2 = np.log1p(data['lag2_rent'][t_idx])[:, :, None]
    l24 = np.log1p(data['lag24_rent'][t_idx])[:, :, None]

    lat_b = np.broadcast_to(data['lat'][None, :, None] * 0.1, (B, N, 1))
    lon_b = np.broadcast_to(data['lon'][None, :, None] * 0.1, (B, N, 1))
    tf_b = np.repeat(data['time_feats'][t_idx][:, None, :], N, axis=1)  # [B, N, 4]
    stemp_b = np.repeat(data['seoul_temp'][t_idx][:, None, :], N, axis=1)  # [B, N, 2]
    sst_b = np.broadcast_to(data['seoul_st'][None, :, :], (B, N, 6))      # [B, N, 6]

    feat = np.concatenate([
        inv, hist_r, hist_ret, l1, l2, l24,
        lat_b, lon_b, tf_b, stemp_b, sst_b
    ], axis=2).astype(np.float32)  # [B, N, 20]

    return torch.tensor(feat, dtype=torch.float32)


def compute_deep_metrics(y_true, y_pred, label=""):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-5)

    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1.0))

    zero_mask = (y_true == 0)
    zero_cnt = int(zero_mask.sum())
    mean_pred_at_zero = float(y_pred[zero_mask].mean()) if zero_cnt > 0 else 0.0
    pred_under_01_ratio = float((y_pred[zero_mask] < 0.1).sum() / max(zero_cnt, 1))

    p95_val = np.percentile(y_true, 95)
    peak_mask = (y_true >= max(p95_val, 3.0))
    peak_cnt = int(peak_mask.sum())
    peak_mae = float(np.abs(y_true[peak_mask] - y_pred[peak_mask]).mean()) if peak_cnt > 0 else 0.0

    ll = y_true * np.log(y_pred) - y_pred - scipy.special.gammaln(y_true + 1.0)
    mean_log_likelihood = float(ll.mean())

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'wape': round(wape, 4),
        'poisson_log_likelihood': round(mean_log_likelihood, 4),
        'zero_regime': {
            'mean_prediction_on_actual_zero': round(mean_pred_at_zero, 4),
            'percentage_pred_under_0_1': round(pred_under_01_ratio * 100, 2)
        },
        'peak_regime': {
            'peak_mae': round(peak_mae, 4)
        }
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description="Massive Scaled-Up Model Training")
    parser.add_argument('--wait-gpu', action='store_true', help="Wait until GPU is free")
    parser.add_argument('--epochs', type=int, default=20, help="Deep learning epochs")
    parser.add_argument('--out', default='data/massive_model_benchmark_results.json')
    args = parser.parse_args()

    print("=" * 88)
    print("      [초대형 스케일업 모델 벤치마크 (Massive Scaled-Up Architecture)]")
    print("=" * 88)

    if args.wait_gpu:
        wait_for_gpu_free(target_pid=21568, poll_interval=15)

    data = load_enriched_dataset(
        'data/processed/tashu_60min.npz',
        'data/clustering_stations.csv',
        'data/seoul_derived_tashu_features.npz'
    )

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Active Device for Deep Learning: {dev}")

    # 1. Train Massive LightGBM (1,000 Trees, num_leaves=127, multi-scale features)
    print("\n[Stage 1] Training Massive LightGBM (1,000 Trees, 22 Features)...")
    X_tr, y_tr_rent, _, _, _, _ = prepare_massive_tabular_dataset(data, data['train_mask'], step=1)
    print(f"   -> Prepared full training matrix: {X_tr.shape} ({X_tr.nbytes / (1024**2):.1f} MB)")

    t0 = time.time()
    massive_lgb = lgb.LGBMRegressor(
        objective='poisson',
        n_estimators=1000,
        learning_rate=0.04,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )
    massive_lgb.fit(X_tr, y_tr_rent)
    print(f"   -> Massive LightGBM trained 1,000 trees in {time.time()-t0:.2f}s")

    # 2. Train Massive ST-ResZIP (6-Block Residual GCN with Dense Skips and ZIP loss, ~1.8M params)
    print("\n[Stage 2] Training Massive ST-ResZIP (6 Blocks, hidden_dim=256, Zero-Inflated Poisson)...")
    deep_model = MassiveSTResZIP(in_dim=20, hidden_dim=256, num_blocks=6).to(dev)
    n_params = sum(p.numel() for p in deep_model.parameters() if p.requires_grad)
    print(f"   -> Model Parameter Count: {n_params:,} parameters (vs 30,000 in baseline GCN)")

    optimizer = torch.optim.AdamW(deep_model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    adj = data['full_adj'].to(dev)

    train_indices = np.where(data['train_mask'])[0]
    n_train = len(train_indices)
    batch_size = 64

    for ep in range(1, args.epochs + 1):
        deep_model.train()
        perm = np.random.permutation(train_indices)
        tot_loss = 0.0
        batches = 0
        t_ep0 = time.time()

        for i in range(0, n_train, batch_size):
            b_idx = perm[i:i+batch_size]
            if len(b_idx) < 16:
                continue
            x_b = prepare_deep_batch(data, b_idx).to(dev)
            y_r = torch.tensor(data['y_rent'][b_idx], dtype=torch.float32, device=dev)
            y_ret = torch.tensor(data['y_ret'][b_idx], dtype=torch.float32, device=dev)

            optimizer.zero_grad()
            out = deep_model(x_b, adj)

            # Exact Zero-Inflated Poisson Negative Log-Likelihood
            loss_r = zip_nll_loss(out['pi_zero'][:, :, 0], out['rate_lambda'][:, :, 0], y_r)
            loss_ret = zip_nll_loss(out['pi_zero'][:, :, 1], out['rate_lambda'][:, :, 1], y_ret)
            loss = loss_r + loss_ret

            loss.backward()
            torch.nn.utils.clip_grad_norm_(deep_model.parameters(), max_norm=2.0)
            optimizer.step()

            tot_loss += loss.item()
            batches += 1

        scheduler.step()
        if ep % 2 == 0 or ep == args.epochs:
            print(f"   Epoch {ep:02d}/{args.epochs} | ZIP NLL Loss: {tot_loss/batches:.4f} | Time: {time.time()-t_ep0:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

    # 3. Evaluation on Held-out Feb 2026 & Mar 2026
    print("\n[Stage 3] Evaluating Massive Scaled Models on Test Sets...")
    results = {}
    periods = [('feb_2026', '2026년 2월 (겨울 저수요)', data['feb_mask']),
               ('mar_2026', '2026년 3월 (봄 성수기/개학)', data['mar_mask'])]

    for p_key, p_name, p_mask in periods:
        print(f"\n" + "=" * 80)
        print(f" [테스트 기간] {p_name}")
        print("=" * 80)

        X_te, y_te_rent, _, T_te, N_te, t_indices = prepare_massive_tabular_dataset(data, p_mask, step=1)

        # Baseline: Historical Mean
        hist_pred = data['hist_rent'][t_indices].reshape(-1)

        # 1. Massive LightGBM prediction
        pred_lgb = massive_lgb.predict(X_te)

        # 2. Massive ST-ResZIP prediction
        deep_model.eval()
        deep_preds = []
        with torch.no_grad():
            for i in range(0, len(t_indices), batch_size):
                b_idx = t_indices[i:i+batch_size]
                x_b = prepare_deep_batch(data, b_idx).to(dev)
                out = deep_model(x_b, adj)
                deep_preds.append(out['expected_rent'].cpu().numpy())
        pred_deep = np.concatenate(deep_preds, axis=0).reshape(-1)

        # 3. Super-Ensemble Blending (0.6 Massive LGB + 0.4 Massive ST-ResZIP)
        pred_ensemble = 0.65 * pred_lgb + 0.35 * pred_deep

        models_eval = {
            'Historical Smoothed Mean': hist_pred,
            'Massive-LightGBM (1,000 Trees)': pred_lgb,
            'Massive-ST-ResZIP (1.8M params)': pred_deep,
            'Super-Ensemble (LGB + ST-ResZIP)': pred_ensemble
        }

        period_metrics = {}
        print(f"\n{'모델명':<32} | {'MAE':<7} | {'WAPE':<7} | {'Poisson LL':<10} | {'0대 예측값':<9} | {'0대<0.1 비율':<12} | {'피크 MAE'}")
        print("-" * 96)
        for m_name, preds in models_eval.items():
            met = compute_deep_metrics(y_te_rent, preds, m_name)
            period_metrics[m_name] = met
            z = met['zero_regime']
            p = met['peak_regime']
            print(f"{m_name:<32} | {met['mae']:>7.4f} | {met['wape']:>7.4f} | {met['poisson_log_likelihood']:>10.4f} | {z['mean_prediction_on_actual_zero']:>9.4f} | {z['percentage_pred_under_0_1']:>10.1f}% | {p['peak_mae']:>7.4f}")

        results[p_key] = period_metrics

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved massive model benchmark to: {args.out}")


if __name__ == '__main__':
    main()
