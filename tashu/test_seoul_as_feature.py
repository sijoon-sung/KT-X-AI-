"""Ablation Experiment: Using Seoul Ttareungi as a Foundation Feature for Tashu.

Empirical verification of the mentor's hypothesis:
"Since Ttareungi has much larger data volume, process/engineer it as an exogenous
Foundation Prior Feature for Tashu to boost accuracy and regularize small-data overfitting."

Compares:
1. Baseline Model (Tashu-only features: local cyclic time, local weather, hour-of-week historical rate)
2. Enhanced Model (Baseline + Seoul Foundation Mobility Prior Feature)

Evaluates strictly on held-out test periods:
- Feb 2026 (Winter)
- Mar 2026 (Early Spring Transition)
"""

import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn


def train_seoul_foundation_network(bike_csv_path: str) -> nn.Module:
    print("[1/4] Pre-training Foundation Mobility Prior Network on 8,760h Seoul Data...")
    seoul_feats = []
    seoul_targets = []
    with open(bike_csv_path, 'r', encoding='unicode_escape') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            cnt = float(row[1])
            hr = int(row[2])
            temp = float(row[3])
            wind = float(row[5])
            rain = float(row[9])

            p = row[0].split('/')
            dt = np.datetime64(f'{int(p[2]):04d}-{int(p[1]):02d}-{int(p[0]):02d}')
            dow = (dt.astype('int64') + 3) % 7
            is_we = float(dow >= 5)

            rad = 2.0 * math.pi * hr / 24.0
            dow_rad = 2.0 * math.pi * dow / 7.0

            seoul_feats.append([
                math.sin(rad), math.cos(rad),
                math.sin(dow_rad), math.cos(dow_rad),
                is_we,
                (temp - 15.0) / 15.0,
                np.log1p(max(0.0, rain)),
                (wind - 2.0) / 2.0
            ])
            # Scale target to [0, 1] relative intensity
            seoul_targets.append(cnt / 3556.0)

    X_seoul = torch.tensor(seoul_feats, dtype=torch.float32)
    y_seoul = torch.tensor(seoul_targets, dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(42)
    seoul_net = nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
        nn.Sigmoid()
    )

    optimizer = torch.optim.Adam(seoul_net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(120):
        optimizer.zero_grad()
        loss = criterion(seoul_net(X_seoul), y_seoul)
        loss.backward()
        optimizer.step()

    print(f"  -> Seoul Pretraining Loss: {loss.item():.6f}")
    return seoul_net


def extract_features_and_evaluate(seoul_net: nn.Module) -> Dict[str, Any]:
    print("[2/4] Extracting Seoul Foundation Prior for Tashu Timeline...")
    npz = np.load('data/processed/tashu_60min.npz', allow_pickle=True)
    bin_start = npz['bin_start']
    valid = npz['valid'] & ~npz['closed']
    rentals = npz['rentals']
    sys_rent = rentals.sum(axis=1)

    # Weather
    with open('data/external/daejeon_hourly_openmeteo.json', 'r', encoding='utf-8') as f:
        w_raw = json.load(f)['hourly']
        w_dict = {}
        for i, ts in enumerate(w_raw['time']):
            w_dict[ts] = {
                'temp': float(w_raw['temperature_2m'][i]) if w_raw['temperature_2m'][i] is not None else 15.0,
                'rain': float(w_raw['precipitation'][i]) if w_raw['precipitation'][i] is not None else 0.0,
                'wind': float(w_raw['wind_speed_10m'][i]) if w_raw['wind_speed_10m'][i] is not None else 5.0
            }

    tashu_feats = []
    hours, dows = [], []
    for dt in bin_start:
        ts_str = str(dt)[:13] + ':00'
        w = w_dict.get(ts_str, {'temp': 15.0, 'rain': 0.0, 'wind': 5.0})
        ts_h = dt.astype('datetime64[h]')
        hr = int(ts_h.astype('int64') % 24)
        dow = int(((ts_h.astype('int64') // 24) + 3) % 7)
        hours.append(hr)
        dows.append(dow)
        is_we = float(dow >= 5)

        rad = 2.0 * math.pi * hr / 24.0
        dow_rad = 2.0 * math.pi * dow / 7.0

        tashu_feats.append([
            math.sin(rad), math.cos(rad),
            math.sin(dow_rad), math.cos(dow_rad),
            is_we,
            (w['temp'] - 15.0) / 15.0,
            np.log1p(max(0.0, w['rain'])),
            (w['wind'] - 2.0) / 2.0
        ])

    X_tashu = torch.tensor(tashu_feats, dtype=torch.float32)
    with torch.no_grad():
        seoul_prior_feature = seoul_net(X_tashu).numpy()  # [T, 1]

    print("[3/4] Building Experimental Datasets (Train < 2026-02-01, Test=Feb/Mar 2026)...")
    train_mask = valid & (bin_start < np.datetime64('2026-02-01'))
    feb_mask = valid & (bin_start >= np.datetime64('2026-02-01')) & (bin_start < np.datetime64('2026-03-01'))
    mar_mask = valid & (bin_start >= np.datetime64('2026-03-01')) & (bin_start < np.datetime64('2026-04-01'))

    # Historical rate baseline feature
    hist_table = np.zeros((7, 24), dtype=np.float32)
    for d in range(7):
        for h in range(24):
            m = train_mask & (np.array(dows) == d) & (np.array(hours) == h)
            hist_table[d, h] = sys_rent[m].mean() if m.any() else 0.0

    hist_feat = np.array([hist_table[d, h] for d, h in zip(dows, hours)], dtype=np.float32)[:, None]

    # Baseline Feature Matrix (9 features)
    base_X = np.column_stack([tashu_feats, hist_feat / 1000.0]).astype(np.float32)
    # Enhanced Feature Matrix (Baseline + Seoul Foundation Prior: 10 features)
    enhanced_X = np.column_stack([base_X, seoul_prior_feature]).astype(np.float32)

    y_all = sys_rent.astype(np.float32)

    class Predictor(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.net(x)

    def train_and_eval(X_mat, name):
        torch.manual_seed(42)
        model = Predictor(X_mat.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        x_train = torch.tensor(X_mat[train_mask])
        y_train = torch.tensor(y_all[train_mask]).unsqueeze(1)

        for _ in range(200):
            optimizer.zero_grad()
            loss = criterion(model(x_train), y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_feb = model(torch.tensor(X_mat[feb_mask])).numpy().flatten()
            pred_mar = model(torch.tensor(X_mat[mar_mask])).numpy().flatten()

        y_feb = y_all[feb_mask]
        y_mar = y_all[mar_mask]

        mae_feb = float(np.mean(np.abs(pred_feb - y_feb)))
        rmse_feb = float(np.sqrt(np.mean((pred_feb - y_feb)**2)))
        corr_feb = float(np.corrcoef(pred_feb, y_feb)[0, 1])

        mae_mar = float(np.mean(np.abs(pred_mar - y_mar)))
        rmse_mar = float(np.sqrt(np.mean((pred_mar - y_mar)**2)))
        corr_mar = float(np.corrcoef(pred_mar, y_mar)[0, 1])

        return {
            'feb': {'mae': round(mae_feb, 2), 'rmse': round(rmse_feb, 2), 'corr': round(corr_feb, 4)},
            'mar': {'mae': round(mae_mar, 2), 'rmse': round(rmse_mar, 2), 'corr': round(corr_mar, 4)}
        }

    print("[4/4] Training Baseline vs Enhanced Models...")
    eval_base = train_and_eval(base_X, "Baseline (Tashu Only)")
    eval_enh = train_and_eval(enhanced_X, "Enhanced (Tashu + Seoul Feature)")

    return {
        'baseline_tashu_only': eval_base,
        'enhanced_with_seoul_feature': eval_enh,
        'improvement_feb_mae': round(eval_base['feb']['mae'] - eval_enh['feb']['mae'], 2),
        'improvement_mar_mae': round(eval_base['mar']['mae'] - eval_enh['mar']['mae'], 2),
    }


def main():
    bike_csv = "data/external/SeoulBikeData.csv"
    seoul_net = train_seoul_foundation_network(bike_csv)
    results = extract_features_and_evaluate(seoul_net)

    print("\n================ EXPERIMENTAL RESULTS ================")
    print("1. February 2026 (Winter Test):")
    print(f"   - Baseline (Tashu Only)       : MAE = {results['baseline_tashu_only']['feb']['mae']} | Corr = {results['baseline_tashu_only']['feb']['corr']}")
    print(f"   - Enhanced (+ Seoul Feature)  : MAE = {results['enhanced_with_seoul_feature']['feb']['mae']} | Corr = {results['enhanced_with_seoul_feature']['feb']['corr']}")
    print(f"   - MAE Reduction               : {results['improvement_feb_mae']} trips/hr improvement!")

    print("\n2. March 2026 (Early Spring Transition Test):")
    print(f"   - Baseline (Tashu Only)       : MAE = {results['baseline_tashu_only']['mar']['mae']} | Corr = {results['baseline_tashu_only']['mar']['corr']}")
    print(f"   - Enhanced (+ Seoul Feature)  : MAE = {results['enhanced_with_seoul_feature']['mar']['mae']} | Corr = {results['enhanced_with_seoul_feature']['mar']['corr']}")
    print(f"   - MAE Reduction               : {results['improvement_mar_mae']} trips/hr improvement!")
    print("======================================================\n")

    out_file = "data/seoul_feature_transfer_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation results to: {out_file}")


if __name__ == '__main__':
    main()
