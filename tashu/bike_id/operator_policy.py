"""Reconstruct the current operator rebalancing from bike-number chains.

A bike returned at A and next rented at B != A (both customer stations) was moved by the operator
somewhere inside [return at A, rental at B].
"""
import json
import numpy as np

z = np.load('trips.npz')
bike, rt, rs, tt, ts = z['bike'], z['rt'], z['rs'], z['tt'], z['ts']
stations, names = z['stations'], z['st_names']
ops = np.isin(stations, ['ST0001', 'ST1220', 'ST1198'])
o = np.lexsort((rt, bike))
b_, rt_, rs_, tt_, ts_ = bike[o], rt[o], rs[o], tt[o], ts[o]
k = np.flatnonzero(b_[1:] == b_[:-1]); k1 = k + 1
fr, to, s, e = ts_[k], rs_[k1], tt_[k], rt_[k1]
mv = (fr != to) & ~ops[fr] & ~ops[to] & (e > s)
fr, to, s, e = fr[mv], to[mv], s[mv], e[mv]
gap_h = (e - s) / 3600
R = {'moves': int(mv.sum())}
days = len(np.unique(rt // 86400))
R['moves_per_day'] = round(R['moves'] / days, 1)
R['gap_bins'] = {lab: float(((gap_h > a) & (gap_h <= b)).mean())
                 for lab, a, b in (('<=2h', 0, 2), ('2-12h', 2, 12), ('12-72h', 12, 72), ('>72h', 72, 1e9))}
short = gap_h <= 2
mid_hour = (((s + e) // 2) // 3600) % 24
R['short_move_hour_hist'] = {int(h): int(((mid_hour == h) & short).sum()) for h in range(24)}
wd = ((((s + e) // 2) // 86400) + 3) % 7
R['short_move_weekend_share'] = float((wd[short] >= 5).mean())
# batches: same from->to on the same arrival day
key = (fr.astype(np.int64) * 5000 + to) * 100000 + e // 86400
_, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
R['share_moves_in_batches_ge3'] = float((cnt[inv] >= 3).mean())
# where from / where to (customer rentals, returns for context)
cust = ~ops[rs] & ~ops[ts]
rent_n = np.bincount(rs[cust], minlength=len(stations))
ret_n = np.bincount(ts[cust], minlength=len(stations))
out_n = np.bincount(fr, minlength=len(stations))
in_n = np.bincount(to, minlength=len(stations))


def top(arr, n=10):
    idx = np.argsort(-arr)[:n]
    return [dict(id=str(stations[i]), name=str(names[i]), moves=int(arr[i]),
                 net_customer=int(ret_n[i] - rent_n[i])) for i in idx]


R['top_pickup_stations'] = top(out_n)
R['top_dropoff_stations'] = top(in_n)
srt = np.sort(out_n)[::-1]
R['top50_pickup_share'] = float(srt[:50].sum() / srt.sum())
srt = np.sort(in_n)[::-1]
R['top50_dropoff_share'] = float(srt[:50].sum() / srt.sum())
# are moves compensating customer imbalance? stations with net inflow vs outflow
net = ret_n - rent_n
R['pickups_from_net_inflow_stations'] = float(out_n[net > 0].sum() / out_n.sum())
R['dropoffs_to_net_outflow_stations'] = float(in_n[net < 0].sum() / in_n.sum())
R['moves_vs_customer_imbalance'] = dict(
    total_abs_net_customer=int(np.abs(net[~ops]).sum()),
    total_moves=int(len(fr)))
# distance of moves
lat = np.zeros(len(stations)); lon = np.zeros(len(stations))
P = np.load(r'C:/Users/DISLAB/Desktop/KT X AI/tashu/data/processed/tashu_60min.npz')
pidx = {s_: i for i, s_ in enumerate(P['stations'].astype(str))}
for i, s_ in enumerate(stations):
    j = pidx.get(str(s_))
    if j is not None:
        lat[i], lon[i] = P['lat'][j], P['lon'][j]
dkm = np.hypot((lat[fr] - lat[to]) * 111.0, (lon[fr] - lon[to]) * 89.7)
okd = (lat[fr] > 0) & (lat[to] > 0)
R['move_km_quantiles'] = {q: float(np.quantile(dkm[okd], q)) for q in (0.25, 0.5, 0.75, 0.9)}
R['short_move_km_median'] = float(np.median(dkm[okd & short]))
print(json.dumps(R, ensure_ascii=False, indent=1))
json.dump(R, open('operator_policy.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
