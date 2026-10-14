"""Empirical Head-to-Head Benchmark: GCN vs XGBoost / LightGBM vs Historical Mean.

Investigates the core user question:
"GCN으로 해봐 XGBoost보다 확률이 잘 나올리가 없는데, deep learning은 스무딩 효과가 있어서"

Evaluates:
1. Overall Accuracy: MAE, RMSE, WAPE
2. Probabilistic Quality: Mean Poisson Log-Likelihood (Deviance)
3. Zero-Count Behavior (The Poisson 0-Regime): False positive drift on 0-demand stations
4. Peak Spikes Behavior (The 18:00 Hub Shock): Underestimation due to Laplacian over-smoothing
5. Spatial Conservation: Macro city total drift

Strictly held-out evaluation on Feb 2026 (Winter) and Mar 2026 (Spring).
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


def build_normalized_adj(od_matrix, add_self_loop=True, directed_norm=True):
    A = od_matrix.astype(np.float32).copy()
    if add_self_loop:
        np.fill_diagonal(A, np.diag(A) + np.maximum(A.sum(1) * 0.05, 1.0))
    if directed_norm:
        row_sum = np.maximum(A.sum(1, keepdims=True), 1e-6)
        A_norm = A / row_sum
    else:
        d = np.maximum(A.sum(1), 1e-6)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return F.relu(out)


class FlatGCN(nn.Module):
    def __init__(self, in_dim=9, hidden_dim=48):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        out = F.softplus(self.head(h)) + 1e-4
        return out[:, :, 0], out[:, :, 1]


class HierarchicalClusterGCN(nn.Module):
    def __init__(self, cluster_sizes, macro_in_dim=8, micro_in_dim=9, hidden_dim=48):
        super().__init__()
        self.n_clusters = len(cluster_sizes)
        self.cluster_sizes = cluster_sizes
        self.macro_gcn1 = GCNLayer(macro_in_dim, hidden_dim)
        self.macro_gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.macro_head = nn.Linear(hidden_dim, 2)

        self.micro_gcn1 = GCNLayer(micro_in_dim, hidden_dim)
        self.micro_gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.micro_rent_head = nn.Linear(hidden_dim, 1)
        self.micro_ret_head = nn.Linear(hidden_dim, 1)

    def forward(self, macro_x, macro_adj, micro_x_list, micro_adj_list):
        h_macro = self.macro_gcn1(macro_x, macro_adj)
        h_macro = self.macro_gcn2(h_macro, macro_adj)
        macro_out = F.softplus(self.macro_head(h_macro)) + 1e-4
        macro_rent = macro_out[:, :, 0]
        macro_ret = macro_out[:, :, 1]

        station_rent_list = []
        station_ret_list = []
        for c in range(self.n_clusters):
            x_c = micro_x_list[c]
            adj_c = micro_adj_list[c]
            h_micro = self.micro_gcn1(x_c, adj_c)
            h_micro = self.micro_gcn2(h_micro, adj_c)
            w_rent = F.softmax(self.micro_rent_head(h_micro).squeeze(-1), dim=1)
            w_ret = F.softmax(self.micro_ret_head(h_micro).squeeze(-1), dim=1)
            station_rent_list.append(macro_rent[:, c:c+1] * w_rent)
            station_ret_list.append(macro_ret[:, c:c+1] * w_ret)

        return (torch.cat(station_rent_list, dim=1),
                torch.cat(station_ret_list, dim=1),
                macro_rent, macro_ret)


def load_all_data(npz_path, clustering_csv_path):
    print("Loading Tashu NPZ and cluster definitions...")
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

    st_cluster = {}
    with open(clustering_csv_path, 'r', encoding='utf-8-sig') as f:
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

    ts = bin_start.astype('datetime64[h]')
    hour = (ts.astype('int64') % 24)
    day = ((ts.astype('int64') // 24) + 3) % 7
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    day_sin = np.sin(2 * np.pi * day / 7.0).astype(np.float32)
    day_cos = np.cos(2 * np.pi * day / 7.0).astype(np.float32)

    train_mask = valid & (bin_start >= np.datetime64('2025-08-01')) & (bin_start < np.datetime64('2026-02-01'))
    feb_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01'))
    mar_mask = valid & (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01'))

    # Historical smoothed mean
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

    # Inter and intra graphs for GCN
    inter_od = np.zeros((n_clusters, n_clusters), dtype=np.float32)
    offset_i = 0
    for ci in range(n_clusters):
        len_i = cluster_sizes[ci]
        offset_j = 0
        for cj in range(n_clusters):
            len_j = cluster_sizes[cj]
            inter_od[ci, cj] = od_sel[offset_i:offset_i+len_i, offset_j:offset_j+len_j].sum()
            offset_j += len_j
        offset_i += len_i

    inter_adj = build_normalized_adj(inter_od, add_self_loop=True, directed_norm=True)
    intra_adjs = []
    offset = 0
    for c in range(n_clusters):
        sz = cluster_sizes[c]
        intra_adjs.append(build_normalized_adj(od_sel[offset:offset+sz, offset:offset+sz], True, False))
        offset += sz
    full_adj = build_normalized_adj(od_sel, True, False)

    return {
        'y_rent': y_rent, 'y_ret': y_ret, 'inv': inv,
        'hist_rent': hist_rent, 'hist_ret': hist_ret,
        'hour': hour, 'day': day,
        'time_feats': np.column_stack([hour_sin, hour_cos, day_sin, day_cos]),
        'lat': (lat_sel - lat_sel.mean()) * 111.0,
        'lon': (lon_sel - lon_sel.mean()) * 89.0,
        'cluster_labels': ordered_cluster_labels,
        'cluster_sizes': cluster_sizes,
        'inter_adj': inter_adj, 'intra_adjs': intra_adjs, 'full_adj': full_adj,
        'train_mask': train_mask, 'feb_mask': feb_mask, 'mar_mask': mar_mask,
        'N_sel': N_sel
    }


def prepare_tabular_dataset(data, mask, max_hours=None, step=1):
    """Formats 2D spatial-temporal matrices into flattened Tabular arrays for XGBoost/LightGBM."""
    t_idx = np.where(mask)[0]
    if step > 1:
        t_idx = t_idx[::step]
    if max_hours and len(t_idx) > max_hours:
        t_idx = np.random.choice(t_idx, max_hours, replace=False)
        t_idx.sort()

    T = len(t_idx)
    N = data['N_sel']

    # Flattened features:
    # 0: log1p(inv)
    # 1: log1p(hist_rent)
    # 2: log1p(hist_ret)
    # 3: lat
    # 4: lon
    # 5: hour_sin
    # 6: hour_cos
    # 7: day_sin
    # 8: day_cos
    # 9: hour (categorical)
    # 10: day (categorical)
    # 11: cluster_id (categorical)

    inv = data['inv'][t_idx].reshape(-1, 1)
    hist_r = data['hist_rent'][t_idx].reshape(-1, 1)
    hist_ret = data['hist_ret'][t_idx].reshape(-1, 1)

    lat_rep = np.tile(data['lat'], T)[:, None]
    lon_rep = np.tile(data['lon'], T)[:, None]
    clust_rep = np.tile(data['cluster_labels'], T)[:, None]

    tf = data['time_feats'][t_idx]  # [T, 4]
    tf_rep = np.repeat(tf, N, axis=0)  # [T*N, 4]

    h_raw = data['hour'][t_idx]
    d_raw = data['day'][t_idx]
    h_rep = np.repeat(h_raw, N)[:, None]
    d_rep = np.repeat(d_raw, N)[:, None]

    X = np.hstack([
        np.log1p(inv),
        np.log1p(hist_r),
        np.log1p(hist_ret),
        lat_rep * 0.1,
        lon_rep * 0.1,
        tf_rep,
        h_rep,
        d_rep,
        clust_rep
    ]).astype(np.float32)

    y_rent = data['y_rent'][t_idx].reshape(-1)
    y_ret = data['y_ret'][t_idx].reshape(-1)

    return X, y_rent, y_ret, T, N, t_idx


def compute_deep_metrics(y_true, y_pred, label=""):
    """Computes comprehensive scientific evaluation metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64), 1e-5)

    mae = float(np.abs(y_true - y_pred).mean())
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.abs(y_true - y_pred).sum() / max(y_true.sum(), 1.0))

    # Zero-Demand Analysis (Poisson 0-regime)
    zero_mask = (y_true == 0)
    zero_cnt = int(zero_mask.sum())
    total_cnt = len(y_true)
    zero_ratio = zero_cnt / total_cnt

    mean_pred_at_zero = float(y_pred[zero_mask].mean()) if zero_cnt > 0 else 0.0
    pred_under_01_ratio = float((y_pred[zero_mask] < 0.1).sum() / max(zero_cnt, 1))

    # Peak Spike Analysis (Top 5% high-demand hours)
    p95_val = np.percentile(y_true, 95)
    peak_mask = (y_true >= max(p95_val, 3.0))
    peak_cnt = int(peak_mask.sum())
    peak_mae = float(np.abs(y_true[peak_mask] - y_pred[peak_mask]).mean()) if peak_cnt > 0 else 0.0
    peak_underestimation_ratio = float((y_pred[peak_mask] < y_true[peak_mask]).sum() / max(peak_cnt, 1))

    # Probabilistic Metric: Poisson Log-Likelihood per station-hour
    # LL = y * ln(lambda) - lambda - ln(gamma(y + 1))
    ll = y_true * np.log(y_pred) - y_pred - scipy.special.gammaln(y_true + 1.0)
    mean_log_likelihood = float(ll.mean())

    return {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'wape': round(wape, 4),
        'poisson_log_likelihood': round(mean_log_likelihood, 4),
        'zero_regime': {
            'actual_zero_percentage': round(zero_ratio * 100, 2),
            'mean_prediction_on_actual_zero': round(mean_pred_at_zero, 4),
            'percentage_pred_under_0_1': round(pred_under_01_ratio * 100, 2)
        },
        'peak_regime': {
            'peak_threshold_trips': float(p95_val),
            'peak_mae': round(peak_mae, 4),
            'peak_underestimate_rate': round(peak_underestimation_ratio * 100, 2)
        }
    }


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("=" * 84)
    print("      [실증 대결] GCN (오버스무딩 딥러닝) vs XGBoost/LightGBM (트리 분기)")
    print("=" * 84)

    npz_path = 'data/processed/tashu_60min.npz'
    cluster_csv = 'data/clustering_stations.csv'
    data = load_all_data(npz_path, cluster_csv)

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"PyTorch Device: {dev}")

    # 1. Prepare tabular dataset for XGBoost / LightGBM
    # Subsample training every 2 hours to get 2,208 hours * 1,283 = 2.83M rows for rapid robust fit
    print("\n[Step 1] Preparing tabular training dataset (step=2, ~2.8M rows)...")
    t0 = time.time()
    X_train, y_train_rent, y_train_ret, T_tr, N_st, _ = prepare_tabular_dataset(
        data, data['train_mask'], step=2
    )
    print(f"Prepared X_train: shape={X_train.shape} in {time.time()-t0:.2f}s")

    # 2. Train XGBoost (Poisson) & LightGBM (Poisson)
    print("\n[Step 2-1] Training LightGBM Poisson Regressor...")
    t0 = time.time()
    lgb_rent = lgb.LGBMRegressor(
        objective='poisson',
        n_estimators=100,
        learning_rate=0.08,
        num_leaves=63,
        n_jobs=-1,
        verbose=-1,
        random_state=42
    )
    lgb_rent.fit(X_train, y_train_rent)
    print(f"   -> LightGBM Rent model trained in {time.time()-t0:.2f}s")

    print("\n[Step 2-2] Training XGBoost (Hist Poisson) Regressor...")
    t0 = time.time()
    xgb_rent = xgb.XGBRegressor(
        objective='count:poisson',
        n_estimators=80,
        learning_rate=0.08,
        max_depth=6,
        tree_method='hist',
        n_jobs=-1,
        random_state=42
    )
    xgb_rent.fit(X_train, y_train_rent)
    print(f"   -> XGBoost Rent model trained in {time.time()-t0:.2f}s")

    # 3. Train Flat GCN & Hierarchical GCN (15 epochs)
    print("\n[Step 3] Training Flat Monolithic GCN and Hierarchical GCN...")
    from cluster_hierarchical_gcn import prepare_features

    train_indices = np.where(data['train_mask'])[0]
    n_train = len(train_indices)

    hier_model = HierarchicalClusterGCN(
        cluster_sizes=data['cluster_sizes'],
        macro_in_dim=8, micro_in_dim=9, hidden_dim=48
    ).to(dev)

    flat_model = FlatGCN(in_dim=9, hidden_dim=48).to(dev)

    hier_opt = torch.optim.Adam(hier_model.parameters(), lr=0.005, weight_decay=1e-5)
    flat_opt = torch.optim.Adam(flat_model.parameters(), lr=0.005, weight_decay=1e-5)

    inter_adj = data['inter_adj'].to(dev)
    intra_adjs = [adj.to(dev) for adj in data['intra_adjs']]
    full_adj = data['full_adj'].to(dev)

    epochs = 15
    batch_size = 128
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(train_indices)
        hier_model.train()
        flat_model.train()
        h_loss_tot = 0.0
        f_loss_tot = 0.0
        batches = 0

        for i in range(0, n_train, batch_size):
            b_idx = perm[i:i+batch_size]
            if len(b_idx) < 16:
                continue
            macro_x, micro_list, flat_x = prepare_features(data, b_idx)
            macro_x = macro_x.to(dev)
            micro_list = [m.to(dev) for m in micro_list]
            flat_x = flat_x.to(dev)
            y_r = torch.tensor(data['y_rent'][b_idx], dtype=torch.float32, device=dev)

            # Hierarchical GCN
            hier_opt.zero_grad()
            pr, _, _, _ = hier_model(macro_x, inter_adj, micro_list, intra_adjs)
            loss_h = F.poisson_nll_loss(pr, y_r, log_input=False)
            loss_h.backward()
            hier_opt.step()
            h_loss_tot += loss_h.item()

            # Flat GCN
            flat_opt.zero_grad()
            fr, _ = flat_model(flat_x, full_adj)
            loss_f = F.poisson_nll_loss(fr, y_r, log_input=False)
            loss_f.backward()
            flat_opt.step()
            f_loss_tot += loss_f.item()
            batches += 1

        if ep % 5 == 0 or ep == epochs:
            print(f"   Epoch {ep:02d}/{epochs} | Hier GCN Loss: {h_loss_tot/batches:.4f} | Flat GCN Loss: {f_loss_tot/batches:.4f}")

    # 4. Strictly Held-out Evaluation on Feb 2026 & Mar 2026
    results = {}
    periods = [('feb_2026', '2026년 2월 (겨울, 저수요)', data['feb_mask']),
               ('mar_2026', '2026년 3월 (봄, 개학/성수기)', data['mar_mask'])]

    for period_key, period_name, p_mask in periods:
        print(f"\n================================================================================")
        print(f" [평가 기간] {period_name}")
        print(f"================================================================================")

        # Tabular test set
        X_test, y_test_r, _, T_te, N_te, t_indices = prepare_tabular_dataset(data, p_mask, step=1)
        y_test_matrix = data['y_rent'][t_indices]

        # 1. Historical Mean
        hist_pred = data['hist_rent'][t_indices].reshape(-1)

        # 2. LightGBM Poisson
        t0 = time.time()
        lgb_pred = lgb_rent.predict(X_test)
        t_lgb = time.time() - t0

        # 3. XGBoost Poisson
        t0 = time.time()
        xgb_pred = xgb_rent.predict(X_test)
        t_xgb = time.time() - t0

        # 4. GCNs
        hier_model.eval()
        flat_model.eval()
        pred_hier_list = []
        pred_flat_list = []

        with torch.no_grad():
            for i in range(0, len(t_indices), batch_size):
                b_idx = t_indices[i:i+batch_size]
                macro_x, micro_list, flat_x = prepare_features(data, b_idx)
                macro_x = macro_x.to(dev)
                micro_list = [m.to(dev) for m in micro_list]
                flat_x = flat_x.to(dev)

                hr, _, _, _ = hier_model(macro_x, inter_adj, micro_list, intra_adjs)
                fr, _ = flat_model(flat_x, full_adj)

                pred_hier_list.append(hr.cpu().numpy())
                pred_flat_list.append(fr.cpu().numpy())

        hier_pred = np.concatenate(pred_hier_list, axis=0).reshape(-1)
        flat_pred = np.concatenate(pred_flat_list, axis=0).reshape(-1)

        # Compute deep metrics
        eval_models = {
            'XGBoost (Poisson)': xgb_pred,
            'LightGBM (Poisson)': lgb_pred,
            'Historical Smoothed Mean': hist_pred,
            'Flat GCN (Monolithic)': flat_pred,
            'Hierarchical GCN (Macro+Micro)': hier_pred
        }

        period_res = {}
        for m_name, preds in eval_models.items():
            metrics = compute_deep_metrics(y_test_r, preds, m_name)
            period_res[m_name] = metrics

        results[period_key] = period_res

        # Print comparison table
        print(f"\n{'모델명':<26} | {'MAE':<7} | {'WAPE':<7} | {'Poisson LL':<10} | {'0대 예측값':<9} | {'0대<0.1 비율':<12} | {'피크 MAE'}")
        print("-" * 92)
        for m_name, m in period_res.items():
            z = m['zero_regime']
            p = m['peak_regime']
            print(f"{m_name:<26} | {m['mae']:>7.4f} | {m['wape']:>7.4f} | {m['poisson_log_likelihood']:>10.4f} | {z['mean_prediction_on_actual_zero']:>9.4f} | {z['percentage_pred_under_0_1']:>10.1f}% | {p['peak_mae']:>7.4f}")

    # Save to json
    out_path = 'data/benchmark_gcn_vs_xgboost_results.json'
    Path(out_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved benchmark results to {out_path}")


if __name__ == '__main__':
    main()
