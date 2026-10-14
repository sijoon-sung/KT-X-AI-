"""Ultra-Scale Model Benchmark (7.36 Million Parameters):
1. Ultra-LightGBM (1,500 Trees, num_leaves=191, 23 Rich Multi-Scale Features)
2. Ultra-STGNN (7,358,980 Parameters, 8-Block Residual GCN + MLP Expansion, Dense Jumping Knowledge, Zero-Inflated Poisson Head)
3. Ultra-Ensemble Blending

Evaluates on strictly held-out Feb 2026 (Winter) & Mar 2026 (Spring).
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
import xgboost as xgb


def build_normalized_adj(od_matrix, add_self_loop=True):
    A = od_matrix.astype(np.float32).copy()
    if add_self_loop:
        np.fill_diagonal(A, np.diag(A) + np.maximum(A.sum(1) * 0.05, 1.0))
    d = np.maximum(A.sum(1), 1e-6)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


# =========================================================================
# Ultra-Scale Architecture (7.36 Million Parameters)
# =========================================================================

class UltraResidualBlock(nn.Module):
    def __init__(self, dim=384, expand_dim=768, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gcn = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, expand_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, adj, x_raw):
        h = self.norm1(x)
        g_out = torch.matmul(adj, self.gcn(h))
        x = x + F.gelu(g_out) + 0.1 * x_raw
        h2 = self.norm2(x)
        mlp_out = self.mlp(h2)
        out = x + mlp_out
        return out


class UltraSTGNN(nn.Module):
    """Ultra-Scale 7.36M Parameter Spatio-Temporal Graph Network."""
    def __init__(self, in_dim=20, hidden_dim=384, expand_dim=768, num_blocks=8):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            UltraResidualBlock(hidden_dim, expand_dim) for _ in range(num_blocks)
        ])
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_blocks + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.zero_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )
        self.rate_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2)
        )

    def forward(self, x, adj):
        h0 = self.in_proj(x)
        h = h0
        intermediates = [h0]
        for block in self.blocks:
            h = block(h, adj, h0)
            intermediates.append(h)
        jk = self.jk_proj(torch.cat(intermediates, dim=-1))
        pi_zero = torch.sigmoid(self.zero_head(jk))
        rate_lambda = F.softplus(self.rate_head(jk)) + 1e-4

        expected_rent = (1.0 - pi_zero[:, :, 0]) * rate_lambda[:, :, 0]
        expected_ret = (1.0 - pi_zero[:, :, 1]) * rate_lambda[:, :, 1]

        return {
            'expected_rent': expected_rent,
            'expected_ret': expected_ret,
            'pi_zero': pi_zero,
            'rate_lambda': rate_lambda
        }


def zip_nll_loss(pi, lam, y):
    eps = 1e-6
    pi = torch.clamp(pi, eps, 1.0 - eps)
    lam = torch.clamp(lam, eps, 1e4)

    is_zero = (y == 0).float()
    is_pos = (y > 0).float()

    prob_zero = pi + (1.0 - pi) * torch.exp(-lam)
    loss_zero = -torch.log(torch.clamp(prob_zero, min=eps)) * is_zero

    ll_pos = (torch.log(1.0 - pi) + y * torch.log(lam) - lam - torch.lgamma(y + 1.0))
    loss_pos = -ll_pos * is_pos

    return (loss_zero + loss_pos).mean()


# =========================================================================
# Data Loading & Preparation
# =========================================================================

def load_data():
    print("Loading Tashu and Seoul transfer features...")
    npz = np.load('data/processed/tashu_60min.npz', allow_pickle=True)
    stations = npz['stations']
    rentals = npz['rentals'].astype(np.float32)
    returns = npz['returns'].astype(np.float32)
    inv_start = npz['inv_start'].astype(np.float32)
    valid = npz['valid'] & ~npz['closed']
    bin_start = npz['bin_start']
    od = npz['od'].astype(np.float32)
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

    lag1_rent = np.roll(y_rent, shift=1, axis=0)
    lag2_rent = np.roll(y_rent, shift=2, axis=0)
    lag24_rent = np.roll(y_rent, shift=24, axis=0)
    lag1_rent[:1] = 0; lag2_rent[:2] = 0; lag24_rent[:24] = 0

    train_mask = valid & (bin_start >= np.datetime64('2025-08-01')) & (bin_start < np.datetime64('2026-02-01'))
    feb_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01'))
    mar_mask = valid & (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01'))

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
        'seoul_temp': seoul_temp,
        'seoul_st': seoul_st_sel,
        'full_adj': full_adj,
        'train_mask': train_mask, 'feb_mask': feb_mask, 'mar_mask': mar_mask,
        'N_sel': N_sel
    }


def prepare_deep_batch(data, t_idx):
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
    tf_b = np.repeat(data['time_feats'][t_idx][:, None, :], N, axis=1)
    stemp_b = np.repeat(data['seoul_temp'][t_idx][:, None, :], N, axis=1)
    sst_b = np.broadcast_to(data['seoul_st'][None, :, :], (B, N, 6))

    feat = np.concatenate([
        inv, hist_r, hist_ret, l1, l2, l24,
        lat_b, lon_b, tf_b, stemp_b, sst_b
    ], axis=2).astype(np.float32)

    return torch.tensor(feat, dtype=torch.float32)


def prepare_tabular_dataset(data, mask, step=1):
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
    seoul_st_rep = np.tile(data['seoul_st'], (T, 1))

    tf = data['time_feats'][t_idx]
    tf_rep = np.repeat(tf, N, axis=0)

    stemp = data['seoul_temp'][t_idx]
    stemp_rep = np.repeat(stemp, N, axis=0)

    h_raw = data['hour'][t_idx]
    d_raw = data['day'][t_idx]
    h_rep = np.repeat(h_raw, N)[:, None]
    d_rep = np.repeat(d_raw, N)[:, None]

    X = np.hstack([
        np.log1p(inv), np.log1p(hist_r), np.log1p(hist_ret),
        np.log1p(l1), np.log1p(l2), np.log1p(l24),
        lat_rep * 0.1, lon_rep * 0.1, tf_rep, stemp_rep, seoul_st_rep,
        h_rep, d_rep, clust_rep
    ]).astype(np.float32)

    y_rent = data['y_rent'][t_idx].reshape(-1)
    return X, y_rent, t_idx


def compute_metrics(y_true, y_pred):
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
    mean_ll = float(ll.mean())

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'wape': round(wape, 4),
        'poisson_log_likelihood': round(mean_ll, 4),
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
    print("=" * 88)
    print("     [울트라 스케일 7.36M 파라미터 모델 벤치마크 (Ultra-Scale STGNN)]")
    print("=" * 88)

    data = load_data()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}")

    # 1. Train Ultra-XGBoost directly on GPU (RTX 3080 CUDA)
    print("\n[Step 1] Training Ultra-XGBoost purely on NVIDIA RTX 3080 GPU (1,500 Trees, max_depth=10, CUDA)...", flush=True)
    X_tr, y_tr, _ = prepare_tabular_dataset(data, data['train_mask'], step=1)
    t0 = time.time()
    ultra_xgb = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=1500,
        learning_rate=0.035,
        max_depth=10,
        tree_method='hist',
        device='cuda',
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42
    )
    ultra_xgb.fit(X_tr, y_tr)
    print(f"   -> Ultra-XGBoost GPU trained in {time.time()-t0:.1f}s", flush=True)

    # 2. Train Ultra-STGNN (7,358,980 parameters)
    print("\n[Step 2] Training Ultra-STGNN on GPU (7,358,980 parameters, 8 blocks, 384 dim)...", flush=True)
    model = UltraSTGNN(in_dim=20, hidden_dim=384, expand_dim=768, num_blocks=8).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   -> Trainable Parameters on CUDA: {n_params:,}", flush=True)

    epochs = 20
    batch_size = 32
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    adj = data['full_adj'].to(dev)

    train_indices = np.where(data['train_mask'])[0]
    n_train = len(train_indices)

    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.permutation(train_indices)
        tot_loss = 0.0
        batches = 0
        t_ep = time.time()

        for i in range(0, n_train, batch_size):
            b_idx = perm[i:i+batch_size]
            if len(b_idx) < 16:
                continue
            x_b = prepare_deep_batch(data, b_idx).to(dev)
            y_r = torch.tensor(data['y_rent'][b_idx], dtype=torch.float32, device=dev)
            y_ret = torch.tensor(data['y_ret'][b_idx], dtype=torch.float32, device=dev)

            optimizer.zero_grad()
            out = model(x_b, adj)

            loss_r = zip_nll_loss(out['pi_zero'][:, :, 0], out['rate_lambda'][:, :, 0], y_r)
            loss_ret = zip_nll_loss(out['pi_zero'][:, :, 1], out['rate_lambda'][:, :, 1], y_ret)
            loss = loss_r + loss_ret

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.5)
            optimizer.step()

            tot_loss += loss.item()
            batches += 1

        scheduler.step()
        if ep % 2 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs} | ZIP Loss: {tot_loss/batches:.4f} | Time: {time.time()-t_ep:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}", flush=True)

    # 3. Evaluation on Held-out Feb 2026 & Mar 2026
    print("\n[Step 3] Evaluating on Held-out Feb & Mar 2026 Test Sets on GPU...", flush=True)
    results = {}
    periods = [('feb_2026', '2026년 2월 (겨울 저수요)', data['feb_mask']),
               ('mar_2026', '2026년 3월 (봄 성수기/개학)', data['mar_mask'])]

    for p_key, p_name, p_mask in periods:
        print(f"\n" + "=" * 84, flush=True)
        print(f" [테스트 기간] {p_name}", flush=True)
        print("=" * 84, flush=True)

        X_te, y_te_rent, t_indices = prepare_tabular_dataset(data, p_mask, step=1)

        # 1. Historical Mean
        hist_pred = data['hist_rent'][t_indices].reshape(-1)

        # 2. Ultra-XGBoost GPU
        pred_xgb = ultra_xgb.predict(X_te)

        # 3. Ultra-STGNN GPU
        model.eval()
        deep_preds = []
        with torch.no_grad():
            for i in range(0, len(t_indices), batch_size):
                b_idx = t_indices[i:i+batch_size]
                x_b = prepare_deep_batch(data, b_idx).to(dev)
                out = model(x_b, adj)
                deep_preds.append(out['expected_rent'].cpu().numpy())
        pred_ultra = np.concatenate(deep_preds, axis=0).reshape(-1)

        # 4. Ultra-Ensemble (0.6 Ultra-XGB + 0.4 Ultra-STGNN)
        pred_ens = 0.60 * pred_xgb + 0.40 * pred_ultra

        eval_dict = {
            'Historical Smoothed Mean': hist_pred,
            'Ultra-XGBoost (1,500 Trees, CUDA)': pred_xgb,
            'Ultra-STGNN (7.36M params, CUDA)': pred_ultra,
            'Ultra-Ensemble (XGB + STGNN, GPU)': pred_ens
        }

        period_metrics = {}
        print(f"\n{'모델명':<34} | {'MAE':<7} | {'WAPE':<7} | {'Poisson LL':<10} | {'0대 예측값':<9} | {'0대<0.1 비율':<12} | {'피크 MAE'}", flush=True)
        print("-" * 98, flush=True)
        for m_name, preds in eval_dict.items():
            met = compute_metrics(y_te_rent, preds)
            period_metrics[m_name] = met
            z = met['zero_regime']
            p = met['peak_regime']
            print(f"{m_name:<32} | {met['mae']:>7.4f} | {met['wape']:>7.4f} | {met['poisson_log_likelihood']:>10.4f} | {z['mean_prediction_on_actual_zero']:>9.4f} | {z['percentage_pred_under_0_1']:>10.1f}% | {p['peak_mae']:>7.4f}")

        results[p_key] = period_metrics

    out_file = 'data/ultra_scale_model_benchmark_results.json'
    Path(out_file).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved ultra scale benchmark results to: {out_file}")


if __name__ == '__main__':
    main()
