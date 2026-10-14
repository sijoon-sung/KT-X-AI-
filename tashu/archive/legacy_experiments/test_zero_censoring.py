import numpy as np
import datetime

d = np.load('data/processed/tashu_60min.npz')
bin_start = d['bin_start']
rentals = d['rentals']
returns = d['returns']
inv_start = d['inv_start']
stations = d['stations']
num_stations = len(stations)

dts = [datetime.datetime.fromisoformat(str(s)) for s in bin_start]
hours = np.array([dt.hour for dt in dts])
weekdays = np.array([dt.weekday() for dt in dts]) # 0=Mon ... 4=Fri
is_weekday = (weekdays < 5)
is_train = np.array([dt < datetime.datetime(2026, 2, 1) for dt in dts])

morn_mask = is_train & is_weekday & ((hours == 7) | (hours == 8))
eve_mask = is_train & is_weekday & ((hours == 18) | (hours == 19))

print(f"Total morning commute hours in train: {morn_mask.sum()}")
print(f"Total evening commute hours in train: {eve_mask.sum()}")

morn_inv = inv_start[morn_mask]     # (T_morn, 1425)
morn_rent = rentals[morn_mask]     # (T_morn, 1425)
eve_ret = returns[eve_mask]         # (T_eve, 1425)

mean_rent_inv0 = np.zeros(num_stations)
mean_rent_inv_ge2 = np.zeros(num_stations)
pct_days_inv0 = np.zeros(num_stations)
count_inv_ge2 = np.zeros(num_stations)

for i in range(num_stations):
    inv_col = morn_inv[:, i]
    rent_col = morn_rent[:, i]
    
    m0 = (inv_col == 0)
    m2 = (inv_col >= 2)
    
    pct_days_inv0[i] = m0.mean() * 100.0
    count_inv_ge2[i] = m2.sum()
    mean_rent_inv0[i] = rent_col[m0].mean() if m0.sum() > 0 else 0.0
    mean_rent_inv_ge2[i] = rent_col[m2].mean() if m2.sum() > 0 else 0.0

mean_eve_ret = eve_ret.mean(axis=0)
overall_morn_rent = morn_rent.mean(axis=0)

struct_zero = (mean_rent_inv_ge2 < 0.1) & (mean_eve_ret < 0.3)
starved_censored = (~struct_zero) & (pct_days_inv0 >= 30.0) & (mean_rent_inv_ge2 >= 0.5)
core_demand = (mean_rent_inv_ge2 >= 1.0) & (~starved_censored)
other_moderate = (~struct_zero) & (~starved_censored) & (~core_demand)

print("\n==================================================")
print("  RIGOROUS EMPIRICAL CENSORING ANALYSIS RESULTS")
print("==================================================")
print(f"Total Stations: {num_stations}")
print(f"1. Verified Structural Zeroes (even with >=2 bikes, rent < 0.1 & eve_ret < 0.3): {struct_zero.sum()} ({struct_zero.mean()*100:.1f}%)")
print(f"2. Heavily Censored / Starved (frequently 0 bikes, but rent jumps >= 0.5 when bikes exist): {starved_censored.sum()} ({starved_censored.mean()*100:.1f}%)")
print(f"3. Core Active Commuter Sinks: {core_demand.sum()} ({core_demand.mean()*100:.1f}%)")
print(f"4. Moderate / Exploration Sinks: {other_moderate.sum()} ({other_moderate.mean()*100:.1f}%)")

print("\n--- PROOF OF DEMAND CENSORING IN STARVED STATIONS (First 10) ---")
print(f"{'Station':<10} | {'0-Bike Days':<12} | {'Rent(Inv=0)':<12} | {'Rent(Inv>=2)':<12} | {'Demand Surge':<12} | {'Eve Ret/hr':<10}")
print("-" * 75)
sample_starved = np.where(starved_censored)[0][:10]
for idx in sample_starved:
    surge = mean_rent_inv_ge2[idx] / (mean_rent_inv0[idx] + 1e-3)
    print(f"{stations[idx]:<10} | {pct_days_inv0[idx]:>10.1f}% | {mean_rent_inv0[idx]:>12.2f} | {mean_rent_inv_ge2[idx]:>12.2f} | {surge:>10.1f}x | {mean_eve_ret[idx]:>10.2f}")

print("\n--- VERIFICATION OF STRUCTURAL ZERO STATIONS (First 10) ---")
print(f"{'Station':<10} | {'0-Bike Days':<12} | {'Days(Inv>=2)':<12} | {'Rent(Inv>=2)':<12} | {'Eve Ret/hr':<10}")
print("-" * 65)
sample_sz = np.where(struct_zero)[0][:10]
for idx in sample_sz:
    print(f"{stations[idx]:<10} | {pct_days_inv0[idx]:>10.1f}% | {int(count_inv_ge2[idx]):>12d} | {mean_rent_inv_ge2[idx]:>12.2f} | {mean_eve_ret[idx]:>10.2f}")
