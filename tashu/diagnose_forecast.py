"""시간 분할로 재고 조건부 flow 모델과 단순 수요 기준선을 비교한다.

완전 노출(exposure == 1)된 대여소·시간에서만 관측 대여를 정답으로 사용한다.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from censored_flow_matching import CensoringAwareFlowMatching, DEV, sample_with_realtime_anchor


NON_CUSTOMER = ('ST0001', 'ST1220', 'ST1198')


def calendar_features(bin_start):
    days = bin_start.astype('datetime64[D]')
    hour = (bin_start - days).astype('timedelta64[h]').astype(np.int64)
    weekend = ((days.astype(np.int64) + 3) % 7 >= 5).astype(np.int64)
    return hour * 2 + weekend


def fit_historical_rate(rentals, exposure, active, train_times, hw, station_ok,
                        alpha=20.0, full_only=False):
    """노출 시간으로 나눈 대여율; 대여소·시간대별 추정치를 완만히 평활한다."""
    n = rentals.shape[1]
    sum_y = np.zeros(n * 48, dtype=np.float64)
    sum_e = np.zeros(n * 48, dtype=np.float64)
    stations = np.arange(n, dtype=np.int64)
    for start in range(0, len(train_times), 128):
        t = train_times[start:start + 128]
        y = rentals[t]
        e = exposure[t]
        usable = active[t] & station_ok[None, :]
        if full_only:
            usable &= e == 1
        else:
            usable &= (e > 0) | (y > 0)
        category = stations[None, :] * 48 + hw[t, None]
        e_effective = np.ones_like(e) if full_only else np.where(y > 0, np.maximum(e, 1 / 60), e)
        sum_y += np.bincount(category[usable], weights=y[usable], minlength=n * 48)
        sum_e += np.bincount(category[usable], weights=e_effective[usable], minlength=n * 48)
    sum_y = sum_y.reshape(n, 48)
    sum_e = sum_e.reshape(n, 48)
    city_rate = sum_y.sum(0) / np.maximum(sum_e.sum(0), 1)
    station_rate = sum_y.sum(1) / np.maximum(sum_e.sum(1), 1)
    overall = sum_y.sum() / max(sum_e.sum(), 1)
    prior = station_rate[:, None] * city_rate[None, :] / max(overall, 1e-6)
    rate = (sum_y + alpha * prior) / np.maximum(sum_e + alpha, 1e-6)
    rate[~station_ok] = 0
    return rate


def score(prediction, observed, mask):
    pred = np.asarray(prediction)[mask]
    actual = np.asarray(observed)[mask]
    return dict(cells=int(mask.sum()), predicted=float(pred.sum()), actual=float(actual.sum()),
                abs_error=float(np.abs(pred - actual).sum()))


def combine(parts):
    totals = {k: sum(p[k] for p in parts) for k in ('cells', 'predicted', 'actual', 'abs_error')}
    totals['mae'] = totals['abs_error'] / max(totals['cells'], 1)
    totals['pred_actual_ratio'] = totals['predicted'] / max(totals['actual'], 1)
    return totals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--model', default='data/censored_flow_holdout.pt')
    ap.add_argument('--out', default='data/forecast_diagnosis_2026_02.json')
    ap.add_argument('--cutoff', default='2026-02-01')
    ap.add_argument('--end', default='2026-03-01')
    args = ap.parse_args()
    with np.load(args.data) as f:
        rentals = f['rentals'].astype(np.float32)
        exposure = f['exposure'].astype(np.float32)
        exposure_ub = f['exposure_ub'].astype(np.float32)
        active = f['active']
        valid = f['valid'] & ~f['closed']
        inventory = f['inv_start']
        bin_start = f['bin_start']
        stations = f['stations'].astype(str)
        od = f['od']
    hw = calendar_features(bin_start)
    station_ok = ~np.isin(stations, NON_CUSTOMER)
    train_times = np.flatnonzero(valid & (bin_start < np.datetime64(args.cutoff)))
    test_times = np.flatnonzero(valid & (bin_start >= np.datetime64(args.cutoff)) &
                                (bin_start < np.datetime64(args.end)))
    if not len(train_times) or not len(test_times):
        raise ValueError('Training or test period has no valid hours')
    exposure_lbx = np.where((rentals > 0) & (exposure < 0.1),
                            np.maximum(exposure, exposure_ub), exposure)
    rate = fit_historical_rate(rentals, exposure_lbx, active, train_times, hw, station_ok)
    full_only_rate = fit_historical_rate(rentals, exposure, active, train_times, hw,
                                         station_ok, full_only=True)

    historical_scores = []
    full_only_scores = []
    for t in test_times:
        mask = active[t] & station_ok & (exposure[t] == 1)
        historical_scores.append(score(rate[:, hw[t]], rentals[t], mask))
        full_only_scores.append(score(full_only_rate[:, hw[t]], rentals[t], mask))

    model = CensoringAwareFlowMatching(len(stations), od).to(DEV)
    model.load_state_dict(torch.load(args.model, map_location=DEV, weights_only=True))
    sample_times = np.unique(np.r_[test_times[::8],
                                    test_times[bin_start[test_times].astype('datetime64[D]') ==
                                               np.datetime64('2026-02-15')]])
    sampled_scores = {name: [] for name in ('historical', 'full_only', 'flow_1_step',
                                            'flow_10_step', 'zero')}
    day = []
    for t in sample_times:
        inv = inventory[t].astype(np.float32)
        p1 = sample_with_realtime_anchor(model, inv, 1)['pred_pout']
        p10 = sample_with_realtime_anchor(model, inv, 10)['pred_pout']
        mask = active[t] & station_ok & (exposure[t] == 1)
        predictions = {'historical': rate[:, hw[t]], 'full_only': full_only_rate[:, hw[t]],
                       'flow_1_step': p1,
                       'flow_10_step': p10, 'zero': np.zeros_like(p10)}
        for name, pred in predictions.items():
            sampled_scores[name].append(score(pred, rentals[t], mask))
        if bin_start[t].astype('datetime64[D]') == np.datetime64('2026-02-15'):
            day.append({'time': str(bin_start[t]), 'actual_total': int(rentals[t].sum()),
                        'historical_total': float(rate[:, hw[t]][active[t] & station_ok].sum()),
                        'full_only_total': float(full_only_rate[:, hw[t]][active[t] & station_ok].sum()),
                        'flow_1_step_total': float(p1[active[t] & station_ok].sum()),
                        'flow_10_step_total': float(p10[active[t] & station_ok].sum())})

    result = {'train_before': args.cutoff, 'test_start': args.cutoff, 'test_end': args.end,
              'train_hours': int(len(train_times)), 'test_hours': int(len(test_times)),
              'sampled_test_hours': int(len(sample_times)),
              'historical_all_full_exposure': combine(historical_scores),
              'full_only_all_full_exposure': combine(full_only_scores),
              'sampled_full_exposure': {name: combine(parts) for name, parts in sampled_scores.items()},
              'day_2026_02_15': day}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'day_2026_02_15'},
                     ensure_ascii=False, indent=2))
    print(f'Saved {args.out}')


if __name__ == '__main__':
    main()
