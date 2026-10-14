"""
train_evaluate_hybrid.py
------------------------
Trains and evaluates the Hybrid Architecture:
ST-GCN (Spatial Graph Convolution) + Continuous Flow Matching (CFM) + ZIC (Zero-Inflated Censored Head).

Multi-task Training:
1. ZIC Loss: Censored Poisson/Negative-Binomial loss with structural zero gate.
2. CFM Loss: Optimal transport vector field regression on continuous spatial flow states.
3. Micro-Timeline Evaluation: Evaluates unmet demand on Feb & Mar 2026 test weekdays.
"""

import os
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from kinetic_flux_demand import load_dataset, DEV
from hybrid_st_flow_zic import HybridFlowZICModel
from neural_zic_model import zic_loss_function
from neural_allocation_policy import NeuralPolicyAllocator
from rigorous_zero_classifier import classify_all_stations, simulate_15min_micro_timeline
from tashu_unified_kinetic_system import simulate_dynamic_peak

torch.set_num_threads(os.cpu_count() or 4)


def train_hybrid_model(data, od_matrix, dist_matrix, epochs=10, batch_size=32, lr=1e-3):
    print("\n" + "="*70)
    print("  [TRAINING] HYBRID ST-GCN + CONTINUOUS FLOW MATCHING + ZIC OPERATOR")
    print("="*70)

    n_stations = len(data['stations'])
    model = HybridFlowZICModel(n_stations, od_matrix, dist_matrix, in_channels=2, hidden_dim=48).to(DEV)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    bin_start = data['bin_start']
    is_train = bin_start < np.datetime64('2026-02-01')
    valid_train = is_train & data['valid_hours']
    train_idx = np.flatnonzero(valid_train)

    hours = data['hour']
    weekdays = data['weekday']
    months = data['month']

    rentals = torch.as_tensor(data['rentals'], device=DEV)
    returns = torch.as_tensor(data['returns'], device=DEV)
    exposure = torch.as_tensor(data['exposure'], device=DEV)
    active = torch.as_tensor(data['active'], device=DEV)
    censored = torch.as_tensor(data['exposure'] < 0.99, device=DEV)

    h_t = torch.as_tensor(hours, device=DEV)
    w_t = torch.as_tensor(weekdays, device=DEV)
    m_t = torch.as_tensor(months, device=DEV)

    # Precompute spatial flow states: [P_out, P_in] = [rentals / sum, returns / sum]
    tot_rent = np.maximum(data['rentals'].sum(axis=1, keepdims=True), 1.0)
    tot_ret = np.maximum(data['returns'].sum(axis=1, keepdims=True), 1.0)
    flow_states = np.stack([data['rentals'] / tot_rent, data['returns'] / tot_ret], axis=-1).astype(np.float32)
    flow_tensor = torch.as_tensor(flow_states, device=DEV)

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.shuffle(train_idx)
        total_loss_epoch = 0.0
        cfm_loss_epoch = 0.0
        zic_loss_epoch = 0.0
        n_batches = 0

        sample_steps = min(len(train_idx) // batch_size, 80)
        for b in range(sample_steps):
            b_idx = train_idx[b * batch_size:(b + 1) * batch_size]
            b_tensor = torch.as_tensor(b_idx, device=DEV)

            b_h = h_t[b_tensor]
            b_w = w_t[b_tensor]
            b_m = m_t[b_tensor]

            # 1. Forward through GCN & ZIC Head
            pred = model(b_h, b_w, b_m, active[b_tensor])
            loss_zic, loss_dict = zic_loss_function(
                pred,
                rentals[b_tensor],
                returns[b_tensor],
                exposure[b_tensor],
                censored[b_tensor],
                active[b_tensor]
            )

            # 2. Continuous Flow Matching (CFM) Loss:
            # x_1 is current hour flow, x_0 is a noisy prior state
            x_1 = flow_tensor[b_tensor]
            x_0 = torch.randn_like(x_1) * 0.1
            loss_cfm = model.forward_cfm_loss(x_1, x_0, pred['H_gcn'])

            # Multi-task loss: ZIC + 0.2 * CFM
            loss = loss_zic + 0.2 * loss_cfm

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss_epoch += float(loss.item())
            cfm_loss_epoch += float(loss_cfm.item())
            zic_loss_epoch += float(loss_zic.item())
            n_batches += 1

        avg_loss = total_loss_epoch / max(n_batches, 1)
        avg_zic = zic_loss_epoch / max(n_batches, 1)
        avg_cfm = cfm_loss_epoch / max(n_batches, 1)
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Hybrid Loss: {avg_loss:.4f} (ZIC: {avg_zic:.4f}, CFM-ODE: {avg_cfm:.4f})")

    Path('data').mkdir(exist_ok=True)
    torch.save(model.state_dict(), 'data/hybrid_flow_zic_model.pt')
    print("  Saved Hybrid ST-GCN-CFM model to data/hybrid_flow_zic_model.pt.")
    return model


def evaluate_hybrid(model, data, cat_stats, od_matrix_norm):
    print("\n" + "="*70)
    print("  [BENCHMARK] EVALUATING HYBRID ST-GCN-CFM-ZIC SYSTEM (2026 TEST SPLIT)")
    print("="*70)

    n_stations = len(data['stations'])
    bin_start = data['bin_start']
    days = bin_start.astype('datetime64[D]')
    hours = (bin_start - days).astype('timedelta64[h]').astype(int)
    dow = ((days.astype('int64') + 3) % 7)
    is_weekday = (dow < 5)

    feb_mask = (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01')) & is_weekday & (hours == 6)
    mar_mask = (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01')) & is_weekday & (hours == 6)

    feb_indices = np.flatnonzero(feb_mask)
    mar_indices = np.flatnonzero(mar_mask)

    h_t = torch.as_tensor(hours, device=DEV)
    w_t = torch.as_tensor(dow, device=DEV)
    m_t = torch.as_tensor(bin_start.astype('datetime64[M]').astype('int64') % 12, device=DEV)
    act_t = torch.as_tensor(data['active'], device=DEV)

    # Train policy network on top of hybrid features
    from neural_allocation_policy import NeuralPolicyAllocator
    policy = NeuralPolicyAllocator(n_stations, hidden_dim=64).to(DEV)
    pol_ckpt = Path('data/neural_policy_model.pt')
    if pol_ckpt.exists():
        policy.load_state_dict(torch.load(pol_ckpt, map_location=DEV, weights_only=True))

    model.eval()
    policy.eval()

    results = {}

    for split_name, indices in [('Feb_2026', feb_indices), ('Mar_2026', mar_indices)]:
        base_unmet_1h = []
        hybrid_unmet_1h = []
        base_unmet_15m = []
        hybrid_unmet_15m = []
        coverage_list = []

        for idx_6am in indices:
            commute_hrs = [idx_6am + 1, idx_6am + 2, idx_6am + 3, idx_6am + 4]
            if commute_hrs[-1] >= len(data['rentals']):
                continue

            c_tensor = torch.as_tensor(commute_hrs, device=DEV)
            with torch.no_grad():
                pred = model(h_t[c_tensor], w_t[c_tensor], m_t[c_tensor], act_t[c_tensor])
                dem_4h = pred['lambda_effective'].sum(dim=0)
                p_zero = pred['p_zero'].mean(dim=0)

                base_inv_6am = torch.as_tensor(data['inv_start'][idx_6am], dtype=torch.float32, device=DEV)
                total_bikes = float(base_inv_6am.sum().item())
                capacities = torch.full((n_stations,), 60.0, device=DEV)
                is_active = act_t[idx_6am]

                alloc, floor = policy(dem_4h, p_zero, capacities, is_active, total_bikes=total_bikes)

            alloc_np = alloc.cpu().numpy()
            base_inv_np = data['inv_start'][idx_6am].astype(np.float32)
            actual_ret_4h = data['returns'][commute_hrs].astype(np.float32)
            cap_np = np.full(n_stations, 60.0, dtype=np.float32)
            active_np = cat_stats['is_active']
            latent_rent_4h = pred['lambda_effective'].cpu().numpy()

            # 1-Hour dynamic sim
            sim_base = simulate_dynamic_peak(base_inv_np, latent_rent_4h, actual_ret_4h, cap_np, active_np)
            sim_hybrid = simulate_dynamic_peak(alloc_np, latent_rent_4h, actual_ret_4h, cap_np, active_np)
            base_unmet_1h.append(sim_base['total_unmet'])
            hybrid_unmet_1h.append(sim_hybrid['total_unmet'])

            # 15-Min micro-timeline sim
            tot_latent = latent_rent_4h.sum(axis=0)
            sim_b15 = simulate_15min_micro_timeline(base_inv_np, tot_latent, od_matrix_norm)
            sim_h15 = simulate_15min_micro_timeline(alloc_np, tot_latent, od_matrix_norm)
            base_unmet_15m.append(sim_b15['total_unmet'])
            hybrid_unmet_15m.append(sim_h15['total_unmet'])

            eligible = active_np & (p_zero.cpu().numpy() < 0.6)
            coverage_list.append(float((alloc_np[eligible] >= 1.0).mean() * 100.0))

        m_b1 = float(np.mean(base_unmet_1h))
        m_h1 = float(np.mean(hybrid_unmet_1h))
        red_1h = (1.0 - m_h1 / m_b1) * 100.0

        m_b15 = float(np.mean(base_unmet_15m))
        m_h15 = float(np.mean(hybrid_unmet_15m))
        red_15m = (1.0 - m_h15 / m_b15) * 100.0

        mean_cov = float(np.mean(coverage_list))

        results[split_name] = {
            'baseline_unmet_1h': m_b1,
            'hybrid_unmet_1h': m_h1,
            'reduction_1h_pct': red_1h,
            'baseline_unmet_15m': m_b15,
            'hybrid_unmet_15m': m_h15,
            'reduction_15m_pct': red_15m,
            'active_coverage_pct': mean_cov
        }

        print(f"\n[{split_name} Hybrid ST-GCN-CFM Results]")
        print(f"  [1-Hour Sim] Baseline {m_b1:.1f} -> Hybrid ST-GCN-CFM {m_h1:.1f} ({red_1h:+.1f}%)")
        print(f"  [15-Min Micro Sim] Baseline {m_b15:.1f} -> Hybrid ST-GCN-CFM {m_h15:.1f} ({red_15m:+.1f}%)")
        print(f"  Active Station Exploration Coverage: {mean_cov:.1f}%")

    out_json = Path('data/hybrid_benchmark_report.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Exported benchmark report to {out_json}.")


def main():
    data = load_dataset()
    n_stations = len(data['stations'])

    od_path = Path('data/processed/od_train_before_2026-02-01.npy')
    if od_path.exists():
        od_matrix = np.load(od_path)
    else:
        od_matrix = data['od']

    # Load dist_km
    with np.load('data/processed/tashu_60min.npz') as f:
        dist_matrix = f['dist_km'] if 'dist_km' in f else None

    row_sums = od_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    od_matrix_norm = od_matrix / row_sums

    cat_stats = classify_all_stations('data/processed/tashu_60min.npz')

    hybrid_model = train_hybrid_model(data, od_matrix, dist_matrix, epochs=10)
    evaluate_hybrid(hybrid_model, data, cat_stats, od_matrix_norm)


if __name__ == '__main__':
    main()
