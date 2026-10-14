"""Builds Unified Multi-City Scale-Invariant Foundation Dataset.

Integrates:
1. Seoul Ttareungi (8,760 hours + 2,586 stations)
2. Washington D.C. Capital Bikeshare (17,379 hours)
3. Daejeon Tashu (14,616 hours + 1,425 stations)

Total: >40,000 hours of multi-city bike sharing data.
Uses standard csv module and numpy (no external pandas dependency).
"""

import csv
import json
from pathlib import Path
import numpy as np


def load_seoul_data(csv_path='data/external/SeoulBikeData.csv'):
    print("Loading Seoul Ttareungi dataset...")
    hours = []
    rentals = []
    with open(csv_path, 'r', encoding='cp949') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 4:
                try:
                    rentals.append(float(row[1]))
                    hours.append(int(row[2]))
                except ValueError:
                    continue

    return {
        'city': 'Seoul',
        'rentals': np.array(rentals, dtype=np.float32),
        'hours': np.array(hours, dtype=np.int32),
        'n_hours': len(rentals)
    }


def load_capital_data(csv_path='data/external/capital_bikeshare_hourly.csv'):
    print("Loading Washington D.C. Capital Bikeshare dataset...")
    hours = []
    rentals = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                hours.append(int(row['hr']))
                rentals.append(float(row['cnt']))
            except (ValueError, KeyError):
                continue

    return {
        'city': 'WashingtonDC',
        'rentals': np.array(rentals, dtype=np.float32),
        'hours': np.array(hours, dtype=np.int32),
        'n_hours': len(rentals)
    }


def load_tashu_macro_data(npz_path='data/processed/tashu_60min.npz'):
    print("Loading Daejeon Tashu dataset...")
    npz = np.load(npz_path, allow_pickle=True)
    rentals = npz['rentals'].sum(axis=1).astype(np.float32)
    bin_start = npz['bin_start']
    ts = bin_start.astype('datetime64[h]')
    hours = (ts.astype('int64') % 24).astype(np.int32)

    return {
        'city': 'Daejeon',
        'rentals': rentals,
        'hours': hours,
        'n_hours': len(rentals)
    }


def build_scale_invariant_features():
    seoul = load_seoul_data()
    capital = load_capital_data()
    tashu = load_tashu_macro_data()

    total_hours = seoul['n_hours'] + capital['n_hours'] + tashu['n_hours']
    print(f"\nUnified Multi-City Dataset Statistics:")
    print(f" - Seoul Ttareungi: {seoul['n_hours']:,} hours (Mean Hourly: {seoul['rentals'].mean():.1f})")
    print(f" - Washington D.C. Capital: {capital['n_hours']:,} hours (Mean Hourly: {capital['rentals'].mean():.1f})")
    print(f" - Daejeon Tashu: {tashu['n_hours']:,} hours (Mean Hourly: {tashu['rentals'].mean():.1f})")
    print(f" Total Multi-City Volume: {total_hours:,} hours")

    # Compute Simplex Normalized Diurnal Profile per city (sum p = 1.0)
    city_profiles = {}
    for c in [seoul, capital, tashu]:
        c_name = c['city']
        hourly_means = np.zeros(24, dtype=np.float32)
        for h in range(24):
            mask = (c['hours'] == h)
            hourly_means[h] = c['rentals'][mask].mean()
        simplex_profile = hourly_means / np.maximum(hourly_means.sum(), 1e-6)
        city_profiles[c_name] = simplex_profile

    # Save to NPZ
    out_path = 'data/multicity_foundation_features.npz'
    np.savez_compressed(
        out_path,
        seoul_profile=city_profiles['Seoul'],
        capital_profile=city_profiles['WashingtonDC'],
        tashu_profile=city_profiles['Daejeon']
    )
    print(f"\nSaved Multi-City Foundation Features to: {out_path}")

    # Cross-city correlation
    r_seoul_cap = float(np.corrcoef(city_profiles['Seoul'], city_profiles['WashingtonDC'])[0, 1])
    r_seoul_tashu = float(np.corrcoef(city_profiles['Seoul'], city_profiles['Daejeon'])[0, 1])
    r_cap_tashu = float(np.corrcoef(city_profiles['WashingtonDC'], city_profiles['Daejeon'])[0, 1])

    report = {
        'total_multi_city_hours': total_hours,
        'simplex_pearson_correlations': {
            'Seoul_vs_WashingtonDC': round(r_seoul_cap, 4),
            'Seoul_vs_Daejeon': round(r_seoul_tashu, 4),
            'WashingtonDC_vs_Daejeon': round(r_cap_tashu, 4)
        }
    }
    Path('data/multicity_foundation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print("\nCross-City Simplex Diurnal Correlations:")
    print(f" - Seoul <-> Washington D.C.: Pearson r = {r_seoul_cap:.4f}")
    print(f" - Seoul <-> Daejeon Tashu:   Pearson r = {r_seoul_tashu:.4f}")
    print(f" - Washington D.C. <-> Tashu: Pearson r = {r_cap_tashu:.4f}")


if __name__ == '__main__':
    build_scale_invariant_features()
