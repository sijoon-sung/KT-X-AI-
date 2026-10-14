"""Descriptive audit of trip-only history and separate recent API snapshots."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from direct_demand_forecast import calendar


def f(x):
    return float(x)


def pct(x):
    return [f(v) for v in np.quantile(x, [0, .1, .25, .5, .75, .9, 1])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/processed/tashu_60min.npz')
    ap.add_argument('--api', default='C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
    ap.add_argument('--out', default='data/data_structure_audit.json')
    args = ap.parse_args()
    with np.load(args.data) as z:
        times, stations = z['bin_start'], z['stations']
        rent, ret = z['rentals'], z['returns']
        active, valid, closed = z['active'], z['valid'], z['closed']
        exposure, exposure_ub = z['exposure'], z['exposure_ub']
        lat, lon = z['lat'], z['lon']
    valid = valid & ~closed
    hour, weekday, _ = calendar(times)
    customer = ~np.isin(stations, ['ST0001', 'ST1220', 'ST1198'])
    months = times.astype('datetime64[M]')
    report = {'trip_source': args.data, 'period': [str(times.min()), str(times.max())],
              'stations_all': len(stations), 'valid_hours': int(valid.sum()),
              'all_hours': len(times), 'months': {}, 'months_without_valid_hours': [],
              'station': {}, 'calendar': {},
              'censoring_proxy': {}, 'api': {}}
    for month in np.unique(months):
        m = valid & (months == month)
        if not m.any():
            report['months_without_valid_hours'].append(str(month))
            continue
        a = active[m] & customer
        rr, bb = rent[m], ret[m]
        report['months'][str(month)] = {
            'valid_hours': int(m.sum()), 'calendar_hours': int((months == month).sum()),
            'active_station_median': f(np.median(a.sum(1))),
            'rentals': int((rr * a).sum()), 'returns': int((bb * a).sum()),
            'rentals_per_valid_hour': f((rr * a).sum() / m.sum()),
            'returns_per_valid_hour': f((bb * a).sum() / m.sum())}
    train = valid & (times >= np.datetime64('2025-08-01')) & (times < np.datetime64('2026-02-01'))
    feb = valid & (times >= np.datetime64('2026-02-01')) & (times < np.datetime64('2026-03-01'))
    mar = valid & (times >= np.datetime64('2026-03-01')) & (times < np.datetime64('2026-04-01'))
    stable = (active[train].mean(0) >= .95) & customer
    report['station']['stable_pre_feb'] = int(stable.sum())
    report['station']['new_or_inactive_pre_feb'] = int((customer & ~stable).sum())
    sums = {}
    for name, mask in [('train', train), ('feb', feb), ('mar', mar)]:
        r = rent[mask][:, stable].sum(0)
        b = ret[mask][:, stable].sum(0)
        sums[name] = (r, b)
        report['station'][name] = {
            'rentals_per_station_quantiles': pct(r), 'returns_per_station_quantiles': pct(b),
            'top_10pct_station_share_rentals': f(np.sort(r)[-max(1, len(r)//10):].sum()/max(r.sum(),1)),
            'top_10pct_station_share_returns': f(np.sort(b)[-max(1, len(b)//10):].sum()/max(b.sum(),1)),
            'station_return_rental_corr': f(np.corrcoef(r,b)[0,1]),
            'zero_rental_station_hour_fraction': f(np.mean(rent[mask][:,stable] == 0)),
            'zero_return_station_hour_fraction': f(np.mean(ret[mask][:,stable] == 0))}
    for name in ['feb', 'mar']:
        tr, tb = sums['train']; er, eb = sums[name]
        report['station'][name]['rank_stability_return_corr_train'] = f(np.corrcoef(tb,eb)[0,1])
        report['station'][name]['rank_stability_rental_corr_train'] = f(np.corrcoef(tr,er)[0,1])
        report['station'][name]['mean_absolute_net_flow_per_hour'] = f(np.abs(eb-er).sum()/max(int((feb if name=='feb' else mar).sum()),1))
    for name, mask in [('train', train), ('feb', feb), ('mar', mar)]:
        rows = {}
        for kind, arr in [('rentals', rent), ('returns', ret)]:
            by_hour = []
            for h in range(24):
                s = mask & (hour == h)
                by_hour.append(f(arr[s][:,stable].sum()/max(s.sum(),1)))
            by_day = []
            for d in range(7):
                s = mask & (weekday == d)
                by_day.append(f(arr[s][:,stable].sum()/max(s.sum(),1)))
            rows[kind] = {'hourly_mean': by_hour, 'weekday_mean': by_day,
                          'peak_hour': int(np.argmax(by_hour)),
                          'peak_trough_ratio': f(max(by_hour)/max(min(by_hour),1e-6))}
        report['calendar'][name] = rows
    for name, mask in [('train', train), ('feb', feb), ('mar', mar)]:
        a = active[mask][:,stable]
        e = exposure[mask][:,stable]
        eu = exposure_ub[mask][:,stable]
        r = rent[mask][:,stable]
        report['censoring_proxy'][name] = {
            'active_cells': int(a.sum()),
            'lower_exposure_zero_fraction': f(np.mean(e[a] == 0)),
            'lower_exposure_full_fraction': f(np.mean(e[a] >= .999)),
            'upper_exposure_zero_fraction': f(np.mean(eu[a] == 0)),
            'upper_exposure_full_fraction': f(np.mean(eu[a] >= .999)),
            'rental_events_lower_exposure_zero_fraction': f(r[(a)&(e==0)].sum()/max(r[a].sum(),1)),
            'rental_events_lower_exposure_partial_fraction': f(r[(a)&(e<.999)].sum()/max(r[a].sum(),1)),
            'mean_rentals_full_lower_exposure': f(r[a & (e>=.999)].mean()) if np.any(a & (e>=.999)) else None,
            'mean_rentals_partial_lower_exposure': f(r[a & (e<.999)].mean()) if np.any(a & (e<.999)) else None}
    api = pd.read_csv(args.api, usecols=['ts','station_id','stock','capacity'], parse_dates=['ts'])
    api['station_id'] = api['station_id'].astype(str)
    stock = pd.to_numeric(api['stock'], errors='coerce')
    capacity = pd.to_numeric(api['capacity'], errors='coerce')
    count_by_ts = api.groupby('ts').size()
    report['api'] = {'source': args.api, 'rows': int(len(api)),
                     'first': str(api['ts'].min()), 'last': str(api['ts'].max()),
                     'timestamps': int(api['ts'].nunique()), 'stations': int(api['station_id'].nunique()),
                     'station_overlap_with_trips': int(np.isin(api['station_id'].unique(),stations).sum()),
                     'stations_per_snapshot_quantiles': pct(count_by_ts),
                     'empty_fraction': f((stock==0).mean()),
                     'stock_quantiles': pct(stock.dropna()),
                     'capacity_missing_fraction': f(capacity.isna().mean()),
                     'stock_above_capacity_fraction': f((stock>capacity).mean()),
                     'duplicate_timestamp_station': int(api.duplicated(['ts','station_id']).sum())}
    per_st = api.groupby('station_id')['stock'].agg(['size',lambda x:(x==0).mean()])
    per_st.columns = ['n','empty_frac']
    report['api']['station_empty_fraction_quantiles'] = pct(per_st['empty_frac'])
    report['api']['station_snapshot_count_quantiles'] = pct(per_st['n'])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': args.out, 'trip_months': len(report['months']),
                      'stable_stations': int(stable.sum()), 'api_rows': len(api)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
