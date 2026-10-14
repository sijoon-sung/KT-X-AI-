"""학습에 쓰지 않은 API 재고 스냅샷으로 기존 모델의 외부 평가를 수행한다.

재고만으로 실제 대여/반납 건수는 알 수 없으므로 다음 1시간의 재고 0 발생과
순재고 변화만 평가한다. 입력 CSV는 절대 학습에 사용하지 않는다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from censored_flow_matching import CensoringAwareFlowMatching, DEV, sample_with_realtime_anchor
from direct_demand_forecast import DirectDemandModel
from stockout_head import StockoutModel


def auc(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    ranks = pd.Series(scores).rank(method='average').to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def nb_tail(mu, r, threshold):
    """음이항 대여량이 threshold를 초과할 확률 (점수 보정용, 학습 없음)."""
    threshold = np.floor(threshold).astype(int)
    p = r / (r + mu)
    q = mu / (r + mu)
    pmf = p ** r
    cdf = np.where(threshold >= 0, pmf, 0.0)
    for k in range(1, max(int(threshold.max()), 0) + 1):
        pmf = pmf * ((k - 1 + r) / k) * q
        cdf += np.where(threshold >= k, pmf, 0.0)
    return np.clip(1 - cdf, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True, help='tashu_stock_10min.csv.gz')
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--model', default='data/censored_flow_model.pt')
    ap.add_argument('--direct-model', default=None)
    ap.add_argument('--stockout-model', default=None)
    ap.add_argument('--out', default='data/api_stock_evaluation.json')
    args = ap.parse_args()

    raw = pd.read_csv(args.stock, usecols=['ts', 'station_id', 'stock'], parse_dates=['ts'])
    if raw.duplicated(['ts', 'station_id']).any():
        raise ValueError('Duplicate station snapshots at the same timestamp')
    panel = raw.pivot(index='ts', columns='station_id', values='stock').sort_index()
    if panel.isna().any().any():
        raise ValueError('Missing station snapshots in the API panel')
    stock = panel.to_numpy(dtype=np.float32)
    times = panel.index.to_numpy(dtype='datetime64[s]')
    with np.load(args.data) as f:
        model_stations = f['stations'].astype(str)
        od = f['od']
    api_index = {sid: i for i, sid in enumerate(panel.columns)}
    model_idx = np.array([i for i, sid in enumerate(model_stations) if sid in api_index])
    api_idx = np.array([api_index[model_stations[i]] for i in model_idx])
    model = CensoringAwareFlowMatching(len(model_stations), od).to(DEV)
    model.load_state_dict(torch.load(args.model, map_location=DEV, weights_only=True))
    direct_model = None
    if args.direct_model:
        checkpoint = torch.load(args.direct_model, map_location=DEV, weights_only=True)
        if checkpoint['stations'] != model_stations.tolist():
            raise ValueError('Direct model station order differs from evaluation data')
        direct_model = DirectDemandModel(len(model_stations)).to(DEV)
        direct_model.load_state_dict(checkpoint['state_dict'])
        direct_model.eval()
    stockout_model = None
    if args.stockout_model:
        checkpoint = torch.load(args.stockout_model, map_location=DEV, weights_only=True)
        if checkpoint['stations'] != model_stations.tolist():
            raise ValueError('Stockout model station order differs from evaluation data')
        stockout_model = StockoutModel(len(model_stations)).to(DEV)
        stockout_model.load_state_dict(checkpoint['state_dict'])
        stockout_model.eval()

    by_model = {name: {'score': [], 'event': [], 'abs_delta': [], 'pred_delta': []}
                for name in ('flow_10_step', 'flow_1_step', 'stock_only',
                             'stock_trend', 'direct', 'direct_nb_risk', 'stockout_head')}
    valid_starts = []
    for t in range(0, len(times) - 6, 6):
        if t < 6:
            continue
        hour = int(pd.Timestamp(times[t]).hour)
        gaps = np.diff(times[t:t + 7]).astype('timedelta64[s]').astype(int)
        total_seconds = int((times[t + 6] - times[t]).astype('timedelta64[s]').astype(int))
        past_seconds = int((times[t] - times[t - 6]).astype('timedelta64[s]').astype(int))
        past_gaps = np.diff(times[t - 6:t + 1]).astype('timedelta64[s]').astype(int)
        if not (7 <= hour <= 22 and 3300 <= total_seconds <= 3900 and
                3300 <= past_seconds <= 3900 and
                np.all((gaps >= 540) & (gaps <= 900)) and
                np.all((past_gaps >= 540) & (past_gaps <= 900))):
            continue
        current = stock[t, api_idx]
        in_stock = current > 0
        if not in_stock.any():
            continue
        event = (stock[t + 1:t + 7, api_idx].min(axis=0) == 0)
        observed_delta = stock[t + 6, api_idx] - current
        inventory = np.zeros(len(model_stations), dtype=np.float32)
        inventory[model_idx] = current
        predictions = {
            'flow_10_step': sample_with_realtime_anchor(model, inventory, 10),
            'flow_1_step': sample_with_realtime_anchor(model, inventory, 1),
        }
        if direct_model is not None:
            timestamp = pd.Timestamp(times[t])
            n = torch.arange(len(model_stations), device=DEV)
            inv = torch.as_tensor(inventory, device=DEV)
            with torch.no_grad():
                mu = direct_model(n, torch.full_like(n, timestamp.hour),
                                  torch.full_like(n, timestamp.weekday()),
                                  torch.full_like(n, timestamp.month - 1), inv).cpu().numpy()
            predictions['direct'] = {'pred_pin': mu[:, 1], 'pred_pout': mu[:, 0],
                                     'stockout_risk': mu[:, 0] - mu[:, 1] - inventory}
            # 1시간 반납이 균등하게 들어온다고 보고 평균 절반만 품절 전 가용량에 반영.
            r = float(torch.exp(direct_model.log_r[0]).item())
            risk_probability = nb_tail(mu[:, 0], r, inventory + 0.5 * mu[:, 1])
            predictions['direct_nb_risk'] = {
                'pred_pin': mu[:, 1], 'pred_pout': mu[:, 0],
                'stockout_risk': risk_probability}
        if stockout_model is not None:
            timestamp = pd.Timestamp(times[t])
            n = torch.arange(len(model_stations), device=DEV)
            inv = torch.as_tensor(inventory, device=DEV)
            with torch.no_grad():
                logit = stockout_model(n, torch.full_like(n, timestamp.hour),
                                       torch.full_like(n, timestamp.weekday()),
                                       torch.full_like(n, timestamp.month - 1), inv).cpu().numpy()
            by_model['stockout_head']['score'].append(logit[model_idx][in_stock])
            by_model['stockout_head']['event'].append(event[in_stock])
        for name, pred in predictions.items():
            score = pred['stockout_risk'][model_idx]
            pred_delta = (pred['pred_pin'] - pred['pred_pout'])[model_idx]
            by_model[name]['score'].append(score[in_stock])
            by_model[name]['event'].append(event[in_stock])
            by_model[name]['abs_delta'].append(np.abs(pred_delta - observed_delta))
            by_model[name]['pred_delta'].append(pred_delta)
        by_model['stock_only']['score'].append(-current[in_stock])
        by_model['stock_only']['event'].append(event[in_stock])
        by_model['stock_only']['abs_delta'].append(np.abs(observed_delta))
        by_model['stock_only']['pred_delta'].append(np.zeros_like(observed_delta))
        past_delta = current - stock[t - 6, api_idx]
        by_model['stock_trend']['score'].append(-(current + past_delta)[in_stock])
        by_model['stock_trend']['event'].append(event[in_stock])
        by_model['stock_trend']['abs_delta'].append(np.abs(past_delta - observed_delta))
        by_model['stock_trend']['pred_delta'].append(past_delta)
        valid_starts.append(str(times[t]))

    if not valid_starts:
        raise ValueError('No complete daytime one-hour API windows')
    metrics = {}
    for name, values in by_model.items():
        if not values['score']:
            continue
        scores = np.concatenate(values['score'])
        events = np.concatenate(values['event'])
        metrics[name] = {'stockout_auc': auc(scores, events),
                         'stockout_events': int(events.sum()),
                         'starting_in_stock_cells': int(len(events)),
                         'stockout_event_rate': float(events.mean())}
        if values['abs_delta']:
            errors = np.concatenate(values['abs_delta'])
            pred_delta = np.concatenate(values['pred_delta'])
            metrics[name]['net_stock_change_mae'] = float(errors.mean())
            metrics[name]['mean_predicted_net_change'] = float(pred_delta.mean())
    result = {'source': str(Path(args.stock)), 'source_start': str(times[0]),
              'source_end': str(times[-1]), 'snapshots': int(len(times)),
              'api_stations': int(stock.shape[1]), 'matched_stations': int(len(model_idx)),
              'evaluated_hour_windows': int(len(valid_starts)),
              'first_window': valid_starts[0], 'last_window': valid_starts[-1],
              'metrics': metrics,
              'limitations': 'Stock snapshots do not reveal gross rentals/returns; net change may include rebalancing. Historical stockout labels are reconstructed from trips; API evaluation spans only three days.'}
    for candidate in ('direct_nb_risk', 'stockout_head'):
      if by_model[candidate]['score']:
        paired = []
        for direct_scores, base_scores, labels in zip(
                by_model[candidate]['score'], by_model['stock_only']['score'],
                by_model['stock_only']['event']):
            direct_auc, base_auc = auc(direct_scores, labels), auc(base_scores, labels)
            if direct_auc is not None and base_auc is not None:
                paired.append(direct_auc - base_auc)
        differences = np.asarray(paired)
        if len(differences):
            rng = np.random.default_rng(0)
            bootstrap = rng.choice(differences, size=(5000, len(differences)), replace=True).mean(1)
            result[f'{candidate}_paired_hourly_auc_gain'] = {
                'hours': int(len(differences)), 'mean_gain': float(differences.mean()),
                'hours_improved': int((differences > 0).sum()),
                'bootstrap_95pct': np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
