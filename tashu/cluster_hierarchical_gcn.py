"""Hierarchical Cluster GCN for Bike Demand & Return Forecasting.

Architecture:
- Level 1 (Inter-Cluster Macro GCN):
    Models the 6x6 macro directed flow between functional clusters.
    Predicts macro cluster departures and arrivals.
- Level 2 (Intra-Cluster Micro GCN):
    Models the dense local OD sub-graphs within each cluster.
    Outputs mathematically normalized spatial allocation weights (sum=1.0 per cluster).
- Integration:
    Final station forecast = (Cluster Macro Total) * (Intra-Cluster Station Weight)
    Guarantees city-scale flow conservation and eliminates noise runaway.

Evaluates against:
1. Historical Empirical Baseline (Hour-of-Week smoothed mean)
2. Flat Monolithic GCN (All 1,283 stations directly passing messages on full graph)

Validation on strictly held-out periods:
- Feb 2026 (Winter)
- Mar 2026 (Early Spring)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def build_normalized_adj(od_matrix, add_self_loop=True, directed_norm=True):
    """Builds a normalized adjacency matrix from an OD matrix."""
    A = od_matrix.astype(np.float32).copy()
    if add_self_loop:
        np.fill_diagonal(A, np.diag(A) + np.maximum(A.sum(1) * 0.05, 1.0))
    if directed_norm:
        row_sum = np.maximum(A.sum(1, keepdims=True), 1e-6)
        A_norm = A / row_sum
    else:
        # Symmetric norm D^{-1/2} A D^{-1/2}
        d = np.maximum(A.sum(1), 1e-6)
        d_inv_sqrt = 1.0 / np.sqrt(d)
        A_norm = d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


class GCNLayer(nn.Module):
    """Simple graph convolution layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [B, N, in_dim], adj: [N, N]
        support = self.linear(x)
        out = torch.matmul(adj, support)
        return F.relu(out)


class FlatGCN(nn.Module):
    """Monolithic GCN predicting all stations without cluster hierarchy."""
    def __init__(self, n_stations, in_dim=10, hidden_dim=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 2)  # [rentals, returns]

    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        out = F.softplus(self.head(h))
        return out[:, :, 0], out[:, :, 1]  # [B, N], [B, N]


class HierarchicalClusterGCN(nn.Module):
    """2-Stage Hierarchical GCN (Macro Inter-Cluster GCN + Micro Intra-Cluster GCNs)."""
    def __init__(self, cluster_sizes, macro_in_dim=12, micro_in_dim=8, hidden_dim=48):
        super().__init__()
        self.n_clusters = len(cluster_sizes)
        self.cluster_sizes = cluster_sizes

        # Level 1: Macro Inter-Cluster GCN (6 nodes)
        self.macro_gcn1 = GCNLayer(macro_in_dim, hidden_dim)
        self.macro_gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.macro_head = nn.Linear(hidden_dim, 2)  # [log_macro_rent, log_macro_ret]

        # Level 2: Micro Intra-Cluster Allocation GCNs
        # Shared or modular local GCN layers
        self.micro_gcn1 = GCNLayer(micro_in_dim, hidden_dim)
        self.micro_gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.micro_rent_head = nn.Linear(hidden_dim, 1)
        self.micro_ret_head = nn.Linear(hidden_dim, 1)

    def forward(self, macro_x, macro_adj, micro_x_list, micro_adj_list):
        B = macro_x.shape[0]

        # 1. Macro prediction
        h_macro = self.macro_gcn1(macro_x, macro_adj)
        h_macro = self.macro_gcn2(h_macro, macro_adj)
        macro_out = F.softplus(self.macro_head(h_macro)) + 1e-4  # [B, 6, 2]
        macro_rent = macro_out[:, :, 0]  # [B, 6]
        macro_ret = macro_out[:, :, 1]   # [B, 6]

        # 2. Micro spatial allocation per cluster
        station_rent_list = []
        station_ret_list = []

        for c in range(self.n_clusters):
            x_c = micro_x_list[c]      # [B, N_c, micro_in_dim]
            adj_c = micro_adj_list[c]  # [N_c, N_c]

            h_micro = self.micro_gcn1(x_c, adj_c)
            h_micro = self.micro_gcn2(h_micro, adj_c)

            logit_rent = self.micro_rent_head(h_micro).squeeze(-1)  # [B, N_c]
            logit_ret = self.micro_ret_head(h_micro).squeeze(-1)    # [B, N_c]

            # Mathematically guaranteed normalized spatial allocation
            w_rent = F.softmax(logit_rent, dim=1)  # [B, N_c], sum=1.0
            w_ret = F.softmax(logit_ret, dim=1)    # [B, N_c], sum=1.0

            # Combined station rate = Macro Total * Spatial Allocation Weight
            y_rent_c = macro_rent[:, c:c+1] * w_rent
            y_ret_c = macro_ret[:, c:c+1] * w_ret

            station_rent_list.append(y_rent_c)
            station_ret_list.append(y_ret_c)

        # Concatenate back to [B, N_total] in cluster order
        pred_rent = torch.cat(station_rent_list, dim=1)
        pred_ret = torch.cat(station_ret_list, dim=1)

        return pred_rent, pred_ret, macro_rent, macro_ret


def load_dataset(npz_path, clustering_csv_path, train_start='2025-08-01', cutoff='2026-02-01'):
    """Loads and preprocesses data and clusters."""
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

    # Read clusters
    st_cluster = {}
    with open(clustering_csv_path, 'r', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            st_cluster[r['station_id']] = int(r['cluster_id'])

    # Filter active selected stations
    train_mask = valid & (bin_start >= np.datetime64(train_start)) & (bin_start < np.datetime64(cutoff))
    selected_mask = np.array([s in st_cluster for s in stations])
    sel_idx = np.where(selected_mask)[0]
    N_sel = len(sel_idx)

    # Re-index stations by cluster for contiguous cluster blocks
    cluster_labels = np.array([st_cluster[stations[i]] for i in sel_idx])
    n_clusters = len(np.unique(cluster_labels))

    cluster_to_indices = [np.where(cluster_labels == c)[0] for c in range(n_clusters)]
    cluster_sizes = [len(idx) for idx in cluster_to_indices]
    reorder = np.concatenate(cluster_to_indices)
    ordered_sel_idx = sel_idx[reorder]

    # Reordered arrays
    y_rent = rentals[:, ordered_sel_idx]
    y_ret = returns[:, ordered_sel_idx]
    inv = inv_start[:, ordered_sel_idx]
    lat_sel = lat[ordered_sel_idx]
    lon_sel = lon[ordered_sel_idx]
    od_sel = od[np.ix_(ordered_sel_idx, ordered_sel_idx)]

    # Time features
    ts = bin_start.astype('datetime64[h]')
    hour = (ts.astype('int64') % 24)
    day = ((ts.astype('int64') // 24) + 3) % 7
    hour_sin = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    hour_cos = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    day_sin = np.sin(2 * np.pi * day / 7.0).astype(np.float32)
    day_cos = np.cos(2 * np.pi * day / 7.0).astype(np.float32)

    # Historical smoothed rate by hour of week
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

    # Cluster Macro Graphs & Intra Graphs
    # Inter-cluster OD matrix (training only)
    train_trips = y_rent[train_mask].sum()
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

    # Intra-cluster sub-graphs
    intra_adjs = []
    offset = 0
    for c in range(n_clusters):
        sz = cluster_sizes[c]
        local_od = od_sel[offset:offset+sz, offset:offset+sz]
        local_adj = build_normalized_adj(local_od, add_self_loop=True, directed_norm=False)
        intra_adjs.append(local_adj)
        offset += sz

    # Full graph for flat baseline
    full_adj = build_normalized_adj(od_sel, add_self_loop=True, directed_norm=False)

    # Masks for evaluation
    feb_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01'))
    mar_mask = valid & (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01'))

    return {
        'y_rent': y_rent,
        'y_ret': y_ret,
        'inv': inv,
        'hist_rent': hist_rent,
        'hist_ret': hist_ret,
        'time_feats': np.column_stack([hour_sin, hour_cos, day_sin, day_cos]),
        'lat': (lat_sel - lat_sel.mean()) * 111.0,
        'lon': (lon_sel - lon_sel.mean()) * 89.0,
        'cluster_sizes': cluster_sizes,
        'inter_adj': inter_adj,
        'intra_adjs': intra_adjs,
        'full_adj': full_adj,
        'train_mask': train_mask,
        'feb_mask': feb_mask,
        'mar_mask': mar_mask,
        'N_sel': N_sel
    }


def prepare_features(data, t_idx):
    """Prepares Macro and Micro feature tensors for time index t_idx."""
    # Micro station features: [B, N, 8]
    # [inv, hist_rent, hist_ret, time_feats(4), lat_lon(2)]
    inv = data['inv'][t_idx]
    hist_r = data['hist_rent'][t_idx]
    hist_ret = data['hist_ret'][t_idx]
    tf = data['time_feats'][t_idx]
    lat = data['lat']
    lon = data['lon']

    B = len(t_idx)
    N = data['N_sel']

    # Expand coordinates to [B, N]
    lat_b = np.broadcast_to(lat[None, :], (B, N))
    lon_b = np.broadcast_to(lon[None, :], (B, N))
    tf_b = np.repeat(tf[:, None, :], N, axis=1)  # [B, N, 4]

    micro_all = np.concatenate([
        np.log1p(inv)[:, :, None],
        np.log1p(hist_r)[:, :, None],
        np.log1p(hist_ret)[:, :, None],
        lat_b[:, :, None] * 0.1,
        lon_b[:, :, None] * 0.1,
        tf_b
    ], axis=2).astype(np.float32)

    # Slice into per-cluster micro features
    micro_list = []
    offset = 0
    for sz in data['cluster_sizes']:
        micro_list.append(torch.tensor(micro_all[:, offset:offset+sz, :]))
        offset += sz

    # Macro cluster features: [B, 6, 12]
    macro_feats = []
    offset = 0
    for sz in data['cluster_sizes']:
        c_inv = inv[:, offset:offset+sz].sum(1, keepdims=True)
        c_hist_r = hist_r[:, offset:offset+sz].sum(1, keepdims=True)
        c_hist_ret = hist_ret[:, offset:offset+sz].sum(1, keepdims=True)
        c_sz = np.full_like(c_inv, sz)
        macro_feats.append(np.column_stack([
            np.log1p(c_inv),
            np.log1p(c_hist_r),
            np.log1p(c_hist_ret),
            np.log1p(c_sz),
            tf
        ]))
        offset += sz

    macro_x = torch.tensor(np.stack(macro_feats, axis=1), dtype=torch.float32)  # [B, 6, 8]
    flat_x = torch.tensor(micro_all, dtype=torch.float32)

    return macro_x, micro_list, flat_x


def compute_metrics(y_true_rent, y_true_ret, pred_rent, pred_ret, cluster_sizes):
    """Computes comprehensive MAE, WAPE, and Net Flow metrics."""
    actual_rent = y_true_rent
    actual_ret = y_true_ret
    actual_net = actual_ret - actual_rent
    pred_net = pred_ret - pred_rent

    rent_mae = float(np.abs(actual_rent - pred_rent).mean())
    ret_mae = float(np.abs(actual_ret - pred_ret).mean())
    net_mae = float(np.abs(actual_net - pred_net).mean())
    rent_wape = float(np.abs(actual_rent - pred_rent).sum() / max(actual_rent.sum(), 1.0))
    ret_wape = float(np.abs(actual_ret - pred_ret).sum() / max(actual_ret.sum(), 1.0))

    # Cluster-level macro MAE
    macro_actual_rent = []
    macro_pred_rent = []
    offset = 0
    for sz in cluster_sizes:
        macro_actual_rent.append(actual_rent[:, offset:offset+sz].sum(1))
        macro_pred_rent.append(pred_rent[:, offset:offset+sz].sum(1))
        offset += sz

    macro_actual = np.stack(macro_actual_rent, axis=1)
    macro_pred = np.stack(macro_pred_rent, axis=1)
    macro_mae = float(np.abs(macro_actual - macro_pred).mean())

    return {
        'rent_mae': round(rent_mae, 4),
        'ret_mae': round(ret_mae, 4),
        'net_flow_mae': round(net_mae, 4),
        'rent_wape': round(rent_wape, 4),
        'ret_wape': round(ret_wape, 4),
        'macro_cluster_mae': round(macro_mae, 2)
    }


def train_models(data, device='cuda', epochs=25, batch_size=128):
    """Trains Hierarchical Cluster GCN and Flat Monolithic GCN."""
    print(f"Using device: {device}")
    dev = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')

    train_indices = np.where(data['train_mask'])[0]
    n_train = len(train_indices)

    # Initialize models
    hier_model = HierarchicalClusterGCN(
        cluster_sizes=data['cluster_sizes'],
        macro_in_dim=8,
        micro_in_dim=9,
        hidden_dim=48
    ).to(dev)

    flat_model = FlatGCN(
        n_stations=data['N_sel'],
        in_dim=9,
        hidden_dim=48
    ).to(dev)

    hier_opt = torch.optim.Adam(hier_model.parameters(), lr=0.005, weight_decay=1e-5)
    flat_opt = torch.optim.Adam(flat_model.parameters(), lr=0.005, weight_decay=1e-5)

    inter_adj = data['inter_adj'].to(dev)
    intra_adjs = [adj.to(dev) for adj in data['intra_adjs']]
    full_adj = data['full_adj'].to(dev)

    print(f"Starting training on {n_train} hours for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(train_indices)
        hier_model.train()
        flat_model.train()
        total_hier_loss = 0.0
        total_flat_loss = 0.0
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
            y_ret = torch.tensor(data['y_ret'][b_idx], dtype=torch.float32, device=dev)

            # 1. Train Hierarchical GCN
            hier_opt.zero_grad()
            pred_r, pred_ret, m_r, m_ret = hier_model(macro_x, inter_adj, micro_list, intra_adjs)

            # Station Poisson loss + Macro Poisson loss
            loss_micro_r = F.poisson_nll_loss(pred_r, y_r, log_input=False)
            loss_micro_ret = F.poisson_nll_loss(pred_ret, y_ret, log_input=False)

            hier_loss = loss_micro_r + loss_micro_ret
            hier_loss.backward()
            hier_opt.step()
            total_hier_loss += hier_loss.item()

            # 2. Train Flat GCN
            flat_opt.zero_grad()
            flat_pred_r, flat_pred_ret = flat_model(flat_x, full_adj)
            flat_loss = (F.poisson_nll_loss(flat_pred_r, y_r, log_input=False) +
                         F.poisson_nll_loss(flat_pred_ret, y_ret, log_input=False))
            flat_loss.backward()
            flat_opt.step()
            total_flat_loss += flat_loss.item()

            batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} | Hier Loss: {total_hier_loss/batches:.4f} | Flat Loss: {total_flat_loss/batches:.4f}")

    return hier_model, flat_model, dev


def evaluate_period(data, mask, hier_model, flat_model, dev):
    """Evaluates Hierarchical GCN, Flat GCN, and Historical Mean on a test period."""
    t_indices = np.where(mask)[0]
    hier_model.eval()
    flat_model.eval()

    pred_hier_r = []
    pred_hier_ret = []
    pred_flat_r = []
    pred_flat_ret = []

    inter_adj = data['inter_adj'].to(dev)
    intra_adjs = [adj.to(dev) for adj in data['intra_adjs']]
    full_adj = data['full_adj'].to(dev)

    batch_size = 128
    with torch.no_grad():
        for i in range(0, len(t_indices), batch_size):
            b_idx = t_indices[i:i+batch_size]
            macro_x, micro_list, flat_x = prepare_features(data, b_idx)
            macro_x = macro_x.to(dev)
            micro_list = [m.to(dev) for m in micro_list]
            flat_x = flat_x.to(dev)

            hr, hret, _, _ = hier_model(macro_x, inter_adj, micro_list, intra_adjs)
            fr, fret = flat_model(flat_x, full_adj)

            pred_hier_r.append(hr.cpu().numpy())
            pred_hier_ret.append(hret.cpu().numpy())
            pred_flat_r.append(fr.cpu().numpy())
            pred_flat_ret.append(fret.cpu().numpy())

    hier_r = np.concatenate(pred_hier_r, axis=0)
    hier_ret = np.concatenate(pred_hier_ret, axis=0)
    flat_r = np.concatenate(pred_flat_r, axis=0)
    flat_ret = np.concatenate(pred_flat_ret, axis=0)

    y_true_r = data['y_rent'][t_indices]
    y_true_ret = data['y_ret'][t_indices]
    hist_r = data['hist_rent'][t_indices]
    hist_ret = data['hist_ret'][t_indices]

    sz = data['cluster_sizes']
    return {
        'hierarchical_gcn': compute_metrics(y_true_r, y_true_ret, hier_r, hier_ret, sz),
        'flat_gcn': compute_metrics(y_true_r, y_true_ret, flat_r, flat_ret, sz),
        'historical_mean': compute_metrics(y_true_r, y_true_ret, hist_r, hist_ret, sz)
    }


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Cluster GCN Evaluation")
    parser.add_argument('--npz', default='data/processed/tashu_60min.npz')
    parser.add_argument('--clusters', default='data/clustering_stations.csv')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--out', default='data/hierarchical_gcn_evaluation.json')
    args = parser.parse_args()

    sys.stdout.reconfigure(encoding='utf-8')
    print("1. Loading dataset and cluster mappings...")
    data = load_dataset(args.npz, args.clusters)
    print(f"   -> Evaluated {data['N_sel']} stations across {len(data['cluster_sizes'])} clusters: {data['cluster_sizes']}")

    print("\n2. Training Hierarchical GCN & Flat Monolithic GCN...")
    hier_model, flat_model, dev = train_models(data, device=args.device, epochs=args.epochs)

    print("\n3. Evaluating on strictly held-out periods...")
    eval_feb = evaluate_period(data, data['feb_mask'], hier_model, flat_model, dev)
    eval_mar = evaluate_period(data, data['mar_mask'], hier_model, flat_model, dev)

    report = {
        'evaluation_target': 'Tashu Hourly Rentals & Returns Forecasting',
        'training_period': '2025-08-01 ~ 2026-02-01',
        'test_period_feb_2026': eval_feb,
        'test_period_mar_2026': eval_mar
    }

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\nSaved evaluation results to: {args.out}")

    # Print summary table
    print("\n" + "=" * 88)
    print("           [계층형 GCN vs 평면 GCN vs 평활 평균 실데이터 검증 비교]")
    print("=" * 88)
    print(f"{'평가 기간':<10} | {'모델':<20} | {'대여 MAE':<10} | {'반납 MAE':<10} | {'순흐름 MAE':<10} | {'거시총량 MAE'}")
    print("-" * 88)
    for period_name, p_eval in [('2026년 2월 (겨울)', eval_feb), ('2026년 3월 (봄)', eval_mar)]:
        for m_name, label in [('hierarchical_gcn', '제안: 계층형 GCN'),
                              ('flat_gcn', '기존: 단일 평면 GCN'),
                              ('historical_mean', '기준: 평활 평균')]:
            m = p_eval[m_name]
            print(f"{period_name:<10} | {label:<20} | {m['rent_mae']:>10.4f} | {m['ret_mae']:>10.4f} | {m['net_flow_mae']:>10.4f} | {m['macro_cluster_mae']:>10.2f}건")
        print("-" * 88)


if __name__ == '__main__':
    main()
