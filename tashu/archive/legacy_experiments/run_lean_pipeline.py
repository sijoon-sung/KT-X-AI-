"""
run_lean_pipeline.py
--------------------
End-to-End One-Click Runner for the Lean Parsimonious Architecture (lean_gcn_kinetic.py):
- 1-hop Spatial Neighbor GCN
- Exact Mass-Conserving Physical Flow (OD)
- Exploration Floor Guarantee (Zero Starvation)
- 15-minute Micro-Timeline Turnover Simulation
- Exports final definitive allocation table and benchmark reports.
"""

import time
import json
import csv
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim

from kinetic_flux_demand import load_dataset, DEV, nb_logpmf
from lean_gcn_kinetic import LeanGCNKineticModel, allocate_lean_robust
from rigorous_zero_classifier import classify_all_stations, simulate_15min_micro_timeline
from tashu_unified_kinetic_system import simulate_dynamic_peak

def main():
    start_time = time.time()
    print("="*70)
    print("      TASHU LEAN PARSIMONIOUS PIPELINE (적재적소 경량화 시스템)")
    print("="*70)
    
    # 1. Load dataset & graphs
    print("\n[Step 1] Loading 1,425-station dataset and spatial/OD graphs...")
    data = load_dataset()
    stations = data['stations']
    N = len(stations)
    
    od_path = Path('data/processed/od_train_before_2026-02-01.npy')
    if od_path.exists():
        od_matrix = np.load(od_path)
    else:
        od_matrix = data['od']
        
    with np.load('data/processed/tashu_60min.npz') as f:
        dist_matrix = f['dist_km'] if 'dist_km' in f else np.ones((N, N), dtype=np.float32)
        
    row_sums = od_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    od_matrix_norm = od_matrix / row_sums

    # 2. Classify stations (571-day empirical evidence)
    print("\n[Step 2] Classifying stations (Distinguishing structural zero vs censored)...")
    cat_stats = classify_all_stations('data/processed/tashu_60min.npz')
    is_active = cat_stats['is_active']
    
    # 3. Train Lean Model
    print("\n[Step 3] Training LeanGCNKineticModel (1-hop Spatial GCN + Mass Flow)...")
    model = LeanGCNKineticModel(N, od_matrix, dist_matrix, hidden_dim=32).to(DEV)
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    
    bin_start = data['bin_start']
    is_train = bin_start < np.datetime64('2026-02-01')
    valid_train = is_train & data['valid_hours']
    t_pool = np.flatnonzero(valid_train)
    
    h_all = torch.as_tensor(data['hour'], dtype=torch.long, device=DEV)
    w_all = torch.as_tensor(data['weekday'], dtype=torch.long, device=DEV)
    m_all = torch.as_tensor(data['month'], dtype=torch.long, device=DEV)
    rent_all = torch.as_tensor(data['rentals'], dtype=torch.float32, device=DEV)
    ret_all = torch.as_tensor(data['returns'], dtype=torch.float32, device=DEV)
    exp_all = torch.as_tensor(data['exposure'], dtype=torch.float32, device=DEV)
    act_all = torch.as_tensor(data['active'], dtype=torch.bool, device=DEV)
    
    epochs = 8
    batch_size = 64
    model.train()
    
    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(t_pool)
        n_batches = min(len(perm) // batch_size, 60)
        total_loss = 0.0
        
        for b in range(n_batches):
            b_idx = perm[b * batch_size:(b + 1) * batch_size]
            b_tensor = torch.as_tensor(b_idx, device=DEV)
            
            out = model(h_all[b_tensor], w_all[b_tensor], m_all[b_tensor], act_all[b_tensor])
            lam = out['lambda_rent']
            mu = out['mu_return']
            
            # Exposure-adjusted Negative-Binomial loss
            tau = torch.clamp(exp_all[b_tensor], min=0.01, max=1.0)
            rate_rent = torch.clamp(lam * tau, min=1e-5)
            mu_return = torch.clamp(mu, min=1e-5)
            
            y_rent = rent_all[b_tensor]
            y_ret = ret_all[b_tensor]
            mask = act_all[b_tensor].float()
            
            r_rent = torch.exp(model.log_r[0]).clamp(0.1, 50.0)
            r_ret = torch.exp(model.log_r[1]).clamp(0.1, 50.0)
            
            ll_rent = nb_logpmf(y_rent, rate_rent, r_rent)
            ll_ret = nb_logpmf(y_ret, mu_return, r_ret)
            
            loss_rent = -(mask * ll_rent).sum() / (mask.sum() + 1e-6)
            loss_ret = -(mask * ll_ret).sum() / (mask.sum() + 1e-6)
            loss = loss_rent + 0.3 * loss_ret
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            
        print(f"  Epoch {epoch}/{epochs} | Lean Loss: {total_loss / n_batches:.4f}")
        
    model.eval()
    
    # 4. Benchmark on Feb & Mar 2026 Test Weekdays
    print("\n[Step 4] Benchmarking on 2026 February & March Test Weekdays...")
    days = bin_start.astype('datetime64[D]')
    hours = (bin_start - days).astype('timedelta64[h]').astype(int)
    dow = ((days.astype('int64') + 3) % 7)
    is_weekday = (dow < 5)

    feb_mask = (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01')) & is_weekday & (hours == 6)
    mar_mask = (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01')) & is_weekday & (hours == 6)

    feb_indices = np.flatnonzero(feb_mask)
    mar_indices = np.flatnonzero(mar_mask)

    benchmark_results = {}
    
    for split_name, indices in [('Feb_2026', feb_indices), ('Mar_2026', mar_indices)]:
        b_1h_list, l_1h_list = [], []
        b_15m_list, l_15m_list = [], []
        cov_list = []
        
        for idx_6am in indices:
            commute_hrs = [idx_6am + 1, idx_6am + 2, idx_6am + 3, idx_6am + 4]
            if commute_hrs[-1] >= len(data['rentals']):
                continue
                
            c_tensor = torch.as_tensor(commute_hrs, device=DEV)
            with torch.no_grad():
                out = model(h_all[c_tensor], w_all[c_tensor], m_all[c_tensor], act_all[c_tensor])
                dem_4h = out['lambda_rent'].sum(dim=0).cpu().numpy()
                latent_rent_4h = out['lambda_rent'].cpu().numpy()
                
            base_inv_6am = data['inv_start'][idx_6am].astype(np.float32)
            total_bikes = int(base_inv_6am.sum())
            actual_ret_4h = data['returns'][commute_hrs].astype(np.float32)
            cap_np = np.full(N, 60.0, dtype=np.float32)
            
            # Lean Robust Allocation (Equi-marginal with Floor Guarantee)
            lean_alloc = allocate_lean_robust(dem_4h, cat_stats, total_bikes=total_bikes)
            
            # A. 1-Hour dynamic sim
            sim_base_1h = simulate_dynamic_peak(base_inv_6am, latent_rent_4h, actual_ret_4h, cap_np, is_active)
            sim_lean_1h = simulate_dynamic_peak(lean_alloc, latent_rent_4h, actual_ret_4h, cap_np, is_active)
            b_1h_list.append(sim_base_1h['total_unmet'])
            l_1h_list.append(sim_lean_1h['total_unmet'])
            
            # B. 15-Minute micro-timeline sim (Intra-hour turnover)
            tot_latent = latent_rent_4h.sum(axis=0)
            sim_base_15m = simulate_15min_micro_timeline(base_inv_6am, tot_latent, od_matrix_norm)
            sim_lean_15m = simulate_15min_micro_timeline(lean_alloc, tot_latent, od_matrix_norm)
            b_15m_list.append(sim_base_15m['total_unmet'])
            l_15m_list.append(sim_lean_15m['total_unmet'])
            
            # Coverage: active residential stations with >= 1 bike
            eligible = is_active & (cat_stats['category_labels'] >= 2)
            cov_list.append(float((lean_alloc[eligible] >= 1).mean() * 100.0))
            
        m_b1 = float(np.mean(b_1h_list))
        m_l1 = float(np.mean(l_1h_list))
        red_1h = (1.0 - m_l1 / m_b1) * 100.0
        
        m_b15 = float(np.mean(b_15m_list))
        m_l15 = float(np.mean(l_15m_list))
        red_15m = (1.0 - m_l15 / m_b15) * 100.0
        
        mean_cov = float(np.mean(cov_list))
        
        benchmark_results[split_name] = {
            'days': len(indices),
            'baseline_unmet_1h': m_b1,
            'lean_unmet_1h': m_l1,
            'reduction_1h_pct': red_1h,
            'baseline_unmet_15m': m_b15,
            'lean_unmet_15m': m_l15,
            'reduction_15m_pct': red_15m,
            'active_coverage_pct': mean_cov
        }
        
        print(f"\n[{split_name} Results]")
        print(f"  [1-Hour Sim] Baseline {m_b1:.1f} -> Lean System {m_l1:.1f} ({red_1h:+.1f}%)")
        print(f"  [15-Min Micro Sim] Baseline {m_b15:.1f} -> Lean System {m_l15:.1f} ({red_15m:+.1f}%)")
        print(f"  Active Station Exploration Coverage: {mean_cov:.1f}%")

    # 5. Export Final Golden Table
    print("\n[Step 5] Exporting Final Lean Golden Allocation Table CSV...")
    sample_commute = [mar_indices[0] + 1, mar_indices[0] + 2, mar_indices[0] + 3, mar_indices[0] + 4]
    c_tensor = torch.as_tensor(sample_commute, device=DEV)
    with torch.no_grad():
        out = model(h_all[c_tensor], w_all[c_tensor], m_all[c_tensor], act_all[c_tensor])
        dem_rep = out['lambda_rent'].sum(dim=0).cpu().numpy()
        
    final_alloc = allocate_lean_robust(dem_rep, cat_stats, total_bikes=3070)
    
    out_csv = Path('data/daejeon_lean_golden_allocation_table.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'station_id', 'category', 'lean_6am_allocation',
            'exploration_floor_guaranteed', 'predicted_4h_demand'
        ])
        cat_names = {0: 'Inactive', 1: 'Structural_Zero', 2: 'Moderate_Uncertain', 3: 'Starved_Censored', 4: 'Core_Commuter'}
        for i in range(N):
            cat_id = cat_stats['category_labels'][i]
            floor_g = 1 if (is_active[i] and cat_id >= 2) else 0
            writer.writerow([
                stations[i],
                cat_names[cat_id],
                int(final_alloc[i]),
                floor_g,
                f"{dem_rep[i]:.2f}"
            ])
            
    print(f"  Exported final allocation table to {out_csv} ({N} stations).")
    
    # Save checkpoint & JSON
    torch.save(model.state_dict(), 'data/lean_gcn_kinetic_model.pt')
    out_json = Path('data/tashu_lean_report.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"  Exported lean report to {out_json}.")
    
    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.1f} seconds.")
    print("="*70)

if __name__ == '__main__':
    main()
