"""
evaluate_rigorous_system.py
---------------------------
Evaluates:
1. Ground truth evidence separating Structural Zero vs Censored Starved Demand.
2. Actual Baseline vs Naive Allocation vs Safe Proactive Allocation (with exploration protection).
3. 15-minute micro-timeline intra-hour reuse simulation on Feb & Mar 2026 test weekdays.
"""

import json
import numpy as np
import datetime
from pathlib import Path
import torch

from kinetic_flux_demand import load_dataset, DEV
from neural_flux_operator import HighDimKineticModel, predict_neural_hour
from rigorous_zero_classifier import (
    classify_all_stations,
    allocate_safe_robust_proactive,
    simulate_15min_micro_timeline
)
from tashu_unified_kinetic_system import compute_multi_hour_flux, simulate_dynamic_peak

def run_evaluation():
    print("==================================================================")
    print("   TASHU RIGOROUS ZERO-DEMAND & MICRO-TIMELINE EVALUATION PIPELINE")
    print("==================================================================")
    
    # 1. Classify stations
    print("\n[Step 1] Running 571-day empirical station classification...")
    cat_stats = classify_all_stations('data/processed/tashu_60min.npz')
    stations = cat_stats['stations']
    N = len(stations)
    labels = cat_stats['category_labels']
    
    counts = {
        'inactive': int(np.sum(labels == 0)),
        'structural_zero': int(np.sum(labels == 1)),
        'moderate_uncertain': int(np.sum(labels == 2)),
        'starved_censored': int(np.sum(labels == 3)),
        'core_commuter': int(np.sum(labels == 4))
    }
    print(f"Total Stations: {N}")
    print(f" - Inactive/Decommissioned: {counts['inactive']} stations")
    print(f" - Verified Structural Zeroes (had bikes, nobody rode): {counts['structural_zero']} stations (ONLY {counts['structural_zero']/N*100:.1f}%!)")
    print(f" - Heavily Censored / Starved (need bikes, demand was choked): {counts['starved_censored']} stations ({counts['starved_censored']/N*100:.1f}%)")
    print(f" - Core Commuter Sinks: {counts['core_commuter']} stations ({counts['core_commuter']/N*100:.1f}%)")
    print(f" - Moderate / Exploration: {counts['moderate_uncertain']} stations ({counts['moderate_uncertain']/N*100:.1f}%)")

    # 2. Load Neural Flux Model
    print("\n[Step 2] Loading Neural Flux Operator...")
    data = load_dataset()

    # Load OD matrix
    od_path = Path('data/processed/od_train_before_2026-02-01.npy')
    if od_path.exists():
        od_matrix = np.load(od_path)
    else:
        od_matrix = np.eye(N) / N

    model = HighDimKineticModel(N, od_matrix, k=32, d_model=32).to(DEV)
    ckpt_path = Path('data/neural_flux_model.pt')
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEV, weights_only=False)
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
        print("  Loaded trained Neural Flux checkpoint successfully.")
    else:
        print("  Model checkpoint not found, will use trained weights if available.")
    model.eval()

    # Normalize OD rows
    row_sums = od_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    od_matrix_norm = od_matrix / row_sums

    # 3. Test on February 2026 and March 2026 weekdays
    bin_start = data['bin_start']
    days = bin_start.astype('datetime64[D]')
    hours = (bin_start - days).astype('timedelta64[h]').astype(int)
    dow = ((days.astype('int64') + 3) % 7) # 0=Mon ... 4=Fri
    is_weekday = (dow < 5)

    feb_mask = (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01')) & is_weekday & (hours == 6)
    mar_mask = (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01')) & is_weekday & (hours == 6)

    feb_6am_indices = np.flatnonzero(feb_mask)
    mar_6am_indices = np.flatnonzero(mar_mask)

    print(f"\n[Step 3] Evaluating on Test Weekdays: Feb ({len(feb_6am_indices)} days), Mar ({len(mar_6am_indices)} days)...")

    results = {}
    
    for split_name, six_am_days in [('Feb_2026', feb_6am_indices), ('Mar_2026', mar_6am_indices)]:
        base_unmet_1h = []
        safe_unmet_1h = []
        safe_relay_unmet_1h = []
        
        base_unmet_15m = []
        safe_unmet_15m = []

        total_bikes_list = []

        for idx_6am in six_am_days:
            # 4 hours of commute: 07:00, 08:00, 09:00, 10:00
            commute_hrs = [idx_6am + 1, idx_6am + 2, idx_6am + 3, idx_6am + 4]
            if commute_hrs[-1] >= len(data['rentals']):
                continue

            actual_rent_4h = data['rentals'][commute_hrs] # [4, N]
            actual_ret_4h = data['returns'][commute_hrs]  # [4, N]
            base_inv_6am = data['inv_start'][idx_6am]     # [N]
            total_bikes = float(base_inv_6am.sum())
            total_bikes_list.append(total_bikes)

            # Compute latent demand via Neural Flux Model
            flux = compute_multi_hour_flux(model, data, commute_hrs)
            latent_rent_4h = flux['hourly_lambda']
            dem_4h = flux['total_demand']

            # Allocations:
            # 1. Safe Proactive Allocation (with Exploration Floor for all starved & moderate stations)
            safe_alloc = allocate_safe_robust_proactive(
                dem_4h, cat_stats, total_budget=int(total_bikes), track='soft'
            )

            # Cap vector for 1h dynamic simulation
            cap_soft = np.full(N, 60.0, dtype=np.float32)

            # A. 1-Hour dynamic simulation
            sim_base = simulate_dynamic_peak(
                base_inv_6am, latent_rent_4h, actual_ret_4h, cap_soft,
                cat_stats['is_active'], enable_mid_peak_relay=False
            )
            sim_safe = simulate_dynamic_peak(
                safe_alloc, latent_rent_4h, actual_ret_4h, cap_soft,
                cat_stats['is_active'], enable_mid_peak_relay=False
            )
            sim_safe_relay = simulate_dynamic_peak(
                safe_alloc, latent_rent_4h, actual_ret_4h, cap_soft,
                cat_stats['is_active'], enable_mid_peak_relay=True, relay_bikes=250.0
            )

            base_unmet_1h.append(sim_base['total_unmet'])
            safe_unmet_1h.append(sim_safe['total_unmet'])
            safe_relay_unmet_1h.append(sim_safe_relay['total_unmet'])

            # B. 15-Minute Micro-Timeline simulation (Intra-hour turnover)
            total_latent_rent = latent_rent_4h.sum(axis=0)
            sim_base_15m = simulate_15min_micro_timeline(base_inv_6am, total_latent_rent, od_matrix_norm)
            sim_safe_15m = simulate_15min_micro_timeline(safe_alloc, total_latent_rent, od_matrix_norm)

            base_unmet_15m.append(sim_base_15m['total_unmet'])
            safe_unmet_15m.append(sim_safe_15m['total_unmet'])

        mean_base_1h = float(np.mean(base_unmet_1h))
        mean_safe_1h = float(np.mean(safe_unmet_1h))
        mean_safe_relay_1h = float(np.mean(safe_relay_unmet_1h))

        red_safe_1h = (1.0 - mean_safe_1h / mean_base_1h) * 100.0
        red_safe_relay_1h = (1.0 - mean_safe_relay_1h / mean_base_1h) * 100.0

        mean_base_15m = float(np.mean(base_unmet_15m))
        mean_safe_15m = float(np.mean(safe_unmet_15m))
        red_safe_15m = (1.0 - mean_safe_15m / mean_base_15m) * 100.0

        results[split_name] = {
            'days_evaluated': len(six_am_days),
            'avg_bikes_at_6am': float(np.mean(total_bikes_list)),
            '1h_simulation': {
                'baseline_unmet': mean_base_1h,
                'safe_proactive_unmet': mean_safe_1h,
                'safe_proactive_reduction_pct': red_safe_1h,
                'safe_relay_unmet': mean_safe_relay_1h,
                'safe_relay_reduction_pct': red_safe_relay_1h
            },
            '15m_micro_timeline': {
                'baseline_unmet': mean_base_15m,
                'safe_proactive_unmet': mean_safe_15m,
                'safe_proactive_reduction_pct': red_safe_15m
            }
        }

        print(f"\n[{split_name} Results]")
        print(f"  Avg Bikes in Circulation at 06:00: {np.mean(total_bikes_list):.0f}")
        print(f"  [1-Hour Sim] Baseline Unmet: {mean_base_1h:.1f} | Safe Proactive: {mean_safe_1h:.1f} ({red_safe_1h:+.1f}%) | With Relay: {mean_safe_relay_1h:.1f} ({red_safe_relay_1h:+.1f}%)")
        print(f"  [15-Min Micro Sim] Baseline Unmet: {mean_base_15m:.1f} | Safe Proactive: {mean_safe_15m:.1f} ({red_safe_15m:+.1f}%)")

    # 4. Generate Safe Golden Allocation CSV
    print("\n[Step 4] Generating Safe Robust Golden Allocation CSV...")
    # Average 6 AM total bikes across test
    avg_bikes = int(results['Mar_2026']['avg_bikes_at_6am'])
    # Representative demand vector
    sample_commute = [mar_6am_indices[0] + 1, mar_6am_indices[0] + 2, mar_6am_indices[0] + 3, mar_6am_indices[0] + 4]
    sample_flux = compute_multi_hour_flux(model, data, sample_commute)
    safe_golden_alloc = allocate_safe_robust_proactive(
        sample_flux['total_demand'], cat_stats, total_budget=avg_bikes, track='soft'
    )

    import csv
    out_csv = Path('data/daejeon_safe_golden_allocation_table.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'station_id', 'category', 'safe_6am_allocation',
            'exploration_floor_guaranteed', 'mean_rent_when_stocked', 'evening_return_hr'
        ])
        cat_names = {0: 'Inactive', 1: 'Structural_Zero', 2: 'Moderate_Uncertain', 3: 'Starved_Censored', 4: 'Core_Commuter'}
        for i in range(N):
            cat_id = labels[i]
            floor_guar = 1 if cat_id in [2, 3, 4] else 0
            writer.writerow([
                stations[i],
                cat_names[cat_id],
                int(safe_golden_alloc[i]),
                floor_guar,
                f"{cat_stats['mean_rent_inv_ge2'][i]:.2f}",
                f"{cat_stats['mean_eve_ret'][i]:.2f}"
            ])
    print(f"  Exported safe golden allocation to {out_csv} ({N} stations).")

    # 5. Save detailed report
    out_json = Path('data/rigorous_censoring_report.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({
            'station_classification': counts,
            'evaluation_results': results
        }, f, indent=2)
    print(f"  Exported full report to {out_json}.")
    print("\n==================================================================")
    print("                     EVALUATION COMPLETE                          ")
    print("==================================================================")

if __name__ == '__main__':
    run_evaluation()
