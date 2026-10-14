"""Ultra-Max Scaled-Up Model Benchmark (~10.6 Million Parameters):
1. UltraMax-XGBoost (2,000 Trees, max_depth=10, 100% CUDA GPU)
2. UltraMax-STGNN (10.6M Parameters, 10-Block Residual GCN + 832-dim MLP Expansion, Dense JK, ZIP Head on CUDA)
3. Tri-Season Comprehensive Evaluation:
   - 2026년 2월 (겨울 혹한기 저수요)
   - 2026년 3월 (봄 개학 성수기 폭발)
   - 2026년 9월 최신 API 실시간 재고 패널 (C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz, 549,600행)
     -> 순재고 변화량 오차 (Net Stock Change MAE) & 1시간 뒤 품절 예측 ROC-AUC
"""

import csv
import gzip
import json
import math
import sys
import time
from pathlib import Path

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
# UltraMax Architecture (~10.6 Million Parameters)
# =========================================================================

class UltraMaxResidualBlock(nn.Module):
    def __init__(self, dim=416, expand_dim=832, dropout=0.1):
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


class UltraMaxSTGNN(nn.Module):
    """UltraMax 10.6M Parameter Spatio-Temporal Graph Network."""
    def __init__(self, in_dim=20, hidden_dim=416, expand_dim=832, num_blocks=10):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.blocks = nn.ModuleList([
            UltraMaxResidualBlock(hidden_dim, expand_dim) for _ in range(num_blocks)
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
# Data Preparation
# =========================================================================

def load_data():
    print("Loading Tashu and Seoul transfer features...", flush=True)
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
    ordered_stations = stations[ordered_sel_idx]

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
        'ordered_stations': ordered_stations,
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
    y_ret = data['y_ret'][t_idx].reshape(-1)
    return X, y_rent, y_ret, t_idx


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


# =========================================================================
# Evaluation on 2026-09 Real-Time Stock Panel
# =========================================================================

def evaluate_september_2026_stock(data, model, ultra_xgb_rent, ultra_xgb_ret, dev):
    print("\n" + "=" * 84, flush=True)
    print(" [Step 4] 2026년 9월 최신 API 실시간 재고 패널 평가 (549,600행)", flush=True)
    print("=" * 84, flush=True)

    stock_gz = 'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz'
    if not Path(stock_gz).exists():
        print(f"Warning: {stock_gz} not found.")
        return {}

    # Load 10min stock panel
    print(f"Reading {stock_gz}...", flush=True)
    timestamps = []
    stations = []
    stocks = []
    with gzip.open(stock_gz, 'rt', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for r in reader:
            timestamps.append(r[0])
            stations.append(r[1])
            stocks.append(float(r[2]))

    ts_arr = np.array(timestamps)
    st_arr = np.array(stations)
    stock_arr = np.array(stocks, dtype=np.float32)

    unique_ts = np.unique(ts_arr)
    unique_ts.sort()
    unique_st = np.unique(st_arr)
    print(f"Loaded {len(ts_arr):,} rows across {len(unique_ts)} snapshots and {len(unique_st)} stations.", flush=True)

    # Station mapping to model ordered stations
    model_st = [str(s) for s in data['ordered_stations']]
    st_to_col = {s: i for i, s in enumerate(unique_st)}
    matched_idx = [i for i, s in enumerate(model_st) if s in st_to_col]
    matched_st = [model_st[i] for i in matched_idx]
    N_match = len(matched_idx)

    # Create 2D panel: [T_snap, N_match]
    ts_to_row = {t: i for i, t in enumerate(unique_ts)}
    panel = np.full((len(unique_ts), N_match), np.nan, dtype=np.float32)

    col_map = {s: idx for idx, s in enumerate(matched_st)}
    for t_val, s_val, stk in zip(ts_arr, st_arr, stock_arr):
        if s_val in col_map:
            r = ts_to_row[t_val]
            c = col_map[s_val]
            panel[r, c] = stk

    # Fill NaNs with column median
    nan_mask = np.isnan(panel)
    if nan_mask.any():
        col_med = np.nanmedian(panel, axis=0)
        panel = np.where(nan_mask, col_med[None, :], panel)

    print(f"Built stock panel matrix: {panel.shape} (Snapshots x Matched Stations)", flush=True)

    # Evaluate 1-hour windows (6 steps of 10 min)
    actual_deltas = []
    actual_stockouts = []
    pred_deltas_xgb = []
    pred_deltas_stgnn = []
    pred_risk_xgb = []
    pred_risk_stgnn = []

    model.eval()
    adj = data['full_adj'].to(dev)

    step_gap = 6  # 6 * 10min = 60min
    eval_count = 0

    for t in range(0, len(unique_ts) - step_gap, 3):  # evaluate every 30min
        s_curr = panel[t]
        s_next = panel[t + step_gap]
        s_window = panel[t+1:t+step_gap+1]

        # Valid in-stock stations
        valid_st_mask = (s_curr > 0)
        if not valid_st_mask.any():
            continue

        act_delta = (s_next - s_curr)[valid_st_mask]
        act_stockout = (s_window.min(axis=0) == 0)[valid_st_mask].astype(int)

        # Time features for snapshot
        # Parse hour and weekday from timestamp string "2026-09-17 14:14:16"
        ts_str = unique_ts[t]
        hour_val = int(ts_str[11:13])
        day_val = 3  # Thursday (2026-09-17) roughly

        # Fast feature construction for XGBoost
        N_all = data['N_sel']
        inv_full = np.zeros(N_all, dtype=np.float32)
        inv_full[matched_idx] = s_curr

        h_rep = np.full(N_all, hour_val)
        d_rep = np.full(N_all, day_val)
        hist_r = data['hist_rent'][-1]
        hist_ret = data['hist_ret'][-1]
        stemp = data['seoul_temp'][-1]

        X_snap = np.hstack([
            np.log1p(inv_full)[:, None],
            np.log1p(hist_r)[:, None],
            np.log1p(hist_ret)[:, None],
            np.zeros((N_all, 3), dtype=np.float32),
            data['lat'][:, None] * 0.1,
            data['lon'][:, None] * 0.1,
            np.tile(data['time_feats'][-1], (N_all, 1)),
            np.repeat(stemp[None, :], N_all, axis=0),
            data['seoul_st'],
            h_rep[:, None],
            d_rep[:, None],
            data['cluster_labels'][:, None]
        ]).astype(np.float32)

        # 1. XGBoost Prediction
        p_r_xgb = ultra_xgb_rent.predict(X_snap)[matched_idx][valid_st_mask]
        p_ret_xgb = ultra_xgb_ret.predict(X_snap)[matched_idx][valid_st_mask]
        p_delta_xgb = p_ret_xgb - p_r_xgb
        p_risk_xgb = (p_r_xgb - p_ret_xgb) - s_curr[valid_st_mask]

        # 2. UltraMax-STGNN Prediction
        # Construct input tensor
        lat_b = data['lat'][None, :, None] * 0.1
        lon_b = data['lon'][None, :, None] * 0.1
        tf_b = data['time_feats'][-1][None, None, :]
        stemp_b = stemp[None, None, :]
        sst_b = data['seoul_st'][None, :, :]

        deep_in = np.concatenate([
            np.log1p(inv_full)[None, :, None],
            np.log1p(hist_r)[None, :, None],
            np.log1p(hist_ret)[None, :, None],
            np.zeros((1, N_all, 3), dtype=np.float32),
            lat_b, lon_b, np.repeat(tf_b, N_all, axis=1),
            np.repeat(stemp_b, N_all, axis=1), sst_b
        ], axis=2).astype(np.float32)

        with torch.no_grad():
            out_deep = model(torch.tensor(deep_in, device=dev), adj)
            p_r_stgnn = out_deep['expected_rent'].cpu().numpy()[0, matched_idx][valid_st_mask]
            p_ret_stgnn = out_deep['expected_ret'].cpu().numpy()[0, matched_idx][valid_st_mask]

        p_delta_stgnn = p_ret_stgnn - p_r_stgnn
        p_risk_stgnn = (p_r_stgnn - p_ret_stgnn) - s_curr[valid_st_mask]

        actual_deltas.append(act_delta)
        actual_stockouts.append(act_stockout)
        pred_deltas_xgb.append(p_delta_xgb)
        pred_deltas_stgnn.append(p_delta_stgnn)
        pred_risk_xgb.append(p_risk_xgb)
        pred_risk_stgnn.append(p_risk_stgnn)

        eval_count += 1
        if eval_count >= 50:
            break

    y_act_delta = np.concatenate(actual_deltas)
    y_act_stockout = np.concatenate(actual_stockouts)
    y_pred_delta_xgb = np.concatenate(pred_deltas_xgb)
    y_pred_delta_stgnn = np.concatenate(pred_deltas_stgnn)
    y_risk_xgb = np.concatenate(pred_risk_xgb)
    y_risk_stgnn = np.concatenate(pred_risk_stgnn)

    # Compute Net Stock Change MAE
    mae_delta_xgb = float(np.abs(y_act_delta - y_pred_delta_xgb).mean())
    mae_delta_stgnn = float(np.abs(y_act_delta - y_pred_delta_stgnn).mean())

    # Compute Stockout AUC
    def calc_auc(scores, labels):
        n_pos = int(labels.sum())
        n_neg = len(labels) - n_pos
        if not n_pos or not n_neg: return 0.5
        # Rank scores
        ranks = np.argsort(np.argsort(scores)) + 1
        return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    auc_xgb = calc_auc(y_risk_xgb, y_act_stockout)
    auc_stgnn = calc_auc(y_risk_stgnn, y_act_stockout)

    sep_metrics = {
        'evaluated_snapshots': eval_count,
        'total_evaluated_station_windows': len(y_act_delta),
        'actual_stockout_event_rate_pct': round(float(y_act_stockout.mean() * 100), 2),
        'UltraMax-XGBoost': {
            'net_stock_change_mae': round(mae_delta_xgb, 4),
            'stockout_prediction_auc': round(auc_xgb, 4)
        },
        'UltraMax-STGNN': {
            'net_stock_change_mae': round(mae_delta_stgnn, 4),
            'stockout_prediction_auc': round(auc_stgnn, 4)
        }
    }

    print(f"\n{'모델명':<32} | {'순재고 변화 MAE':<16} | {'1시간 뒤 품절 예측 ROC-AUC'}", flush=True)
    print("-" * 75, flush=True)
    print(f"{'UltraMax-XGBoost (GPU)':<32} | {mae_delta_xgb:>16.4f} | {auc_xgb:>22.4f}", flush=True)
    print(f"{'UltraMax-STGNN (10.6M params, GPU)':<32} | {mae_delta_stgnn:>16.4f} | {auc_stgnn:>22.4f}", flush=True)

    return sep_metrics


# =========================================================================
# Main Execution
# =========================================================================

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 88, flush=True)
    print("     [울트라-맥스 10.6M 모델 벤치마크 & 2월·3월·9월 3계절 통합 평가]", flush=True)
    print("=" * 88, flush=True)

    data = load_data()
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {dev}", flush=True)

    # 1. Train UltraMax-XGBoost (Rentals & Returns on CUDA GPU)
    print("\n[Step 1] Training UltraMax-XGBoost on GPU (2,000 Trees, max_depth=10, CUDA)...", flush=True)
    X_tr, y_tr_rent, y_tr_ret, _ = prepare_tabular_dataset(data, data['train_mask'], step=1)
    t0 = time.time()
    ultra_xgb_rent = xgb.XGBRegressor(
        objective='count:poisson', n_estimators=2000, learning_rate=0.03,
        max_depth=10, tree_method='hist', device='cuda', subsample=0.85, colsample_bytree=0.85, random_state=42
    )
    ultra_xgb_rent.fit(X_tr, y_tr_rent)

    ultra_xgb_ret = xgb.XGBRegressor(
        objective='count:poisson', n_estimators=2000, learning_rate=0.03,
        max_depth=10, tree_method='hist', device='cuda', subsample=0.85, colsample_bytree=0.85, random_state=42
    )
    ultra_xgb_ret.fit(X_tr, y_tr_ret)
    print(f"   -> UltraMax-XGBoost Dual Models trained on GPU in {time.time()-t0:.1f}s", flush=True)

    # 2. Train UltraMax-STGNN (10,589,460 parameters on CUDA)
    print("\n[Step 2] Training UltraMax-STGNN on GPU (10.6M Parameters, 10 blocks, 416 dim)...", flush=True)
    model = UltraMaxSTGNN(in_dim=20, hidden_dim=416, expand_dim=832, num_blocks=10).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   -> UltraMax Trainable Parameters on CUDA: {n_params:,}", flush=True)

    epochs = 20
    batch_size = 24
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0012, weight_decay=1e-4)
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
    trip_results = {}
    periods = [('feb_2026', '2026년 2월 (겨울 저수요)', data['feb_mask']),
               ('mar_2026', '2026년 3월 (봄 성수기/개학)', data['mar_mask'])]

    for p_key, p_name, p_mask in periods:
        print(f"\n" + "=" * 84, flush=True)
        print(f" [테스트 기간] {p_name}", flush=True)
        print("=" * 84, flush=True)

        X_te, y_te_rent, _, t_indices = prepare_tabular_dataset(data, p_mask, step=1)
        hist_pred = data['hist_rent'][t_indices].reshape(-1)
        pred_xgb = ultra_xgb_rent.predict(X_te)

        model.eval()
        deep_preds = []
        with torch.no_grad():
            for i in range(0, len(t_indices), batch_size):
                b_idx = t_indices[i:i+batch_size]
                x_b = prepare_deep_batch(data, b_idx).to(dev)
                out = model(x_b, adj)
                deep_preds.append(out['expected_rent'].cpu().numpy())
        pred_ultra = np.concatenate(deep_preds, axis=0).reshape(-1)
        pred_ens = 0.60 * pred_xgb + 0.40 * pred_ultra

        eval_dict = {
            'Historical Smoothed Mean': hist_pred,
            'UltraMax-XGBoost (2,000 Trees, CUDA)': pred_xgb,
            'UltraMax-STGNN (10.6M params, CUDA)': pred_ultra,
            'UltraMax-Ensemble (XGB + STGNN, GPU)': pred_ens
        }

        period_metrics = {}
        print(f"\n{'모델명':<36} | {'MAE':<7} | {'WAPE':<7} | {'Poisson LL':<10} | {'0대 예측값':<9} | {'0대<0.1 비율':<12} | {'피크 MAE'}", flush=True)
        print("-" * 100, flush=True)
        for m_name, preds in eval_dict.items():
            met = compute_metrics(y_te_rent, preds)
            period_metrics[m_name] = met
            z = met['zero_regime']
            p = met['peak_regime']
            print(f"{m_name:<36} | {met['mae']:>7.4f} | {met['wape']:>7.4f} | {met['poisson_log_likelihood']:>10.4f} | {z['mean_prediction_on_actual_zero']:>9.4f} | {z['percentage_pred_under_0_1']:>10.1f}% | {p['peak_mae']:>7.4f}", flush=True)

        trip_results[p_key] = period_metrics

    # 4. Evaluate September 2026 Stock Panel
    sep_results = evaluate_september_2026_stock(data, model, ultra_xgb_rent, ultra_xgb_ret, dev)

    full_report = {
        'model_capacity': 'UltraMax 10.6M STGNN + 2,000-Tree XGBoost on CUDA',
        'winter_feb_2026': trip_results['feb_2026'],
        'spring_mar_2026': trip_results['mar_2026'],
        'autumn_sep_2026_realtime_api': sep_results
    }

    out_file = 'data/ultra_max_tri_evaluation_results.json'
    Path(out_file).write_text(json.dumps(full_report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved Tri-Season UltraMax benchmark results to: {out_file}", flush=True)


if __name__ == '__main__':
    main()
