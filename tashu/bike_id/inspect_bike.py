"""What does the bike number tell us?  Precise checks on raw Tashu trips.

A. fleet: bike-number prefixes, active bikes per month, trips per bike
B. operator stations (정비대기/관제센터/물류센터): how they appear in trips
C. failed-rental signature: same-station return within 2 min
D. 1-hour free policy: durations around 60 min, re-rent (<3 min) pairs
E. unusable bikes: bike stays at a station while others are rented (Kaspi-style evidence)
F. group rides: co-departures that also co-arrive (batch arrivals)
G. Poisson check on full-exposure station-hours (dispersion, zeros)
"""
import json
import numpy as np

z = np.load('trips.npz')
bike, rt, rs, tt, ts = z['bike'], z['rt'], z['rs'], z['tt'], z['ts']
km = z['km']
bikes, stations, names = z['bikes'], z['stations'], z['st_names']
dur = tt - rt
R = {}
N = len(bike)
OPS_ID = ['ST0001', 'ST1220', 'ST1198']
ops = np.isin(np.arange(len(stations)), np.flatnonzero(np.isin(stations, OPS_ID)))
month = rt.astype('datetime64[s]').astype('datetime64[M]')
hour = (rt // 3600) % 24

# ---------------------------------------------------------------- A. fleet
prefix = np.array([b.split('-')[0] for b in bikes])
pf_of_trip = prefix[bike]
R['A_prefix_bikes'] = {p: int((prefix == p).sum()) for p in np.unique(prefix)}
R['A_prefix_trips'] = {p: int((pf_of_trip == p).sum()) for p in np.unique(prefix)}
months = np.unique(month)
first_m = np.full(len(bikes), np.datetime64('2100-01'), 'datetime64[M]')
np.minimum.at(first_m, bike, month)
fleet = {}
for m in months:
    sel = month == m
    days = len(np.unique(rt[sel] // 86400))
    act = np.unique(bike[sel])
    fleet[str(m)] = dict(
        trips=int(sel.sum()), days=days, active_bikes=int(len(act)),
        new_bikes=int((first_m[act] == m).sum()),
        trips_per_bike_day=round(float(sel.sum()) / max(len(act), 1) / max(days, 1), 3),
        by_prefix={p: int((prefix[act] == p).sum()) for p in np.unique(prefix)})
R['A_fleet_by_month'] = fleet

# ---------------------------------------------------------------- B. operator stations
R['B_ops'] = {sid: dict(rent_from=int((rs == np.flatnonzero(stations == sid)[0]).sum()),
                        return_to=int((ts == np.flatnonzero(stations == sid)[0]).sum()))
              for sid in OPS_ID}
opt = ops[rs] | ops[ts]
R['B_ops_trip_share'] = float(opt.mean())
R['B_ops_trip_minutes_median'] = float(np.median(dur[opt]) / 60)

# ---------------------------------------------------------------- per-bike chains
o = np.lexsort((rt, bike))
b_, rt_, rs_, tt_, ts_, dur_, km_ = bike[o], rt[o], rs[o], tt[o], ts[o], dur[o], km[o]
nxt = np.flatnonzero(b_[1:] == b_[:-1])          # pair (k, k+1) of the same bike
k, k1 = nxt, nxt + 1
gap = rt_[k1] - tt_[k]
same = ts_[k] == rs_[k1]
reloc = ~same & ~ops[ts_[k]] & ~ops[rs_[k1]]
to_ops = ~same & ops[rs_[k1]]                    # next appearance starts from an operator station
R['pairs'] = int(len(k))
R['pairs_same_station'] = float(same.mean())
R['pairs_relocated_regular'] = float(reloc.mean())
R['pairs_next_from_ops'] = float(to_ops.mean())

# ---------------------------------------------------------------- C. failed-rental signature
loop = rs_ == ts_
failed = loop & (dur_ <= 120) & ~ops[rs_]
R['C_same_station_trip_share'] = float((loop & ~ops[rs_]).mean())
R['C_failed_share_of_trips'] = float(failed.mean())
R['C_failed_km_median'] = float(np.nanmedian(km_[failed]))
f_k = failed[k]
base = ~failed[k] & ~ops[rs_[k]]


def after(mask):
    g = gap[mask]
    return dict(n=int(mask.sum()),
                next_rent_same_station=float(same[mask].mean()),
                re_rent_within_3min=float((same[mask] & (g <= 180)).mean()),
                idle_over_24h=float((g > 86400).mean()),
                relocated=float(reloc[mask].mean()),
                next_from_ops=float(to_ops[mask].mean()),
                next_trip_failed_again=float(failed[k1][mask].mean()))


R['C_after_failed'] = after(f_k)
R['C_after_normal'] = after(base)

# ---------------------------------------------------------------- D. 1-hour free
mins = dur // 60
R['D_duration_hist_45_75'] = {int(m): int((mins == m).sum()) for m in range(45, 76)}
rr = same & (gap <= 180) & (gap >= 0) & ~ops[ts_[k]]
first_leg = dur_[k][rr] / 60
R['D_rerent_pairs'] = int(rr.sum())
R['D_rerent_share_of_trips'] = float(rr.sum() / N)
R['D_rerent_first_leg'] = dict(le_2min=float((first_leg <= 2).mean()),
                               m50_60=float(((first_leg >= 50) & (first_leg < 61)).mean()),
                               other=float(((first_leg > 2) & (first_leg < 50)).mean()),
                               gt_60=float((first_leg >= 61).mean()))
R['D_all_trips_50_60_share'] = float(((dur / 60 >= 50) & (dur / 60 < 61)).mean())

# ---------------------------------------------------------------- E. unusable bikes (skip evidence)
# intervals where a bike provably stood at a station: same-station pairs
iv = same & (gap > 0) & ~ops[ts_[k]]
st_i, s_i, e_i = ts_[k][iv], tt_[k][iv], rt_[k1][iv]
SC = np.int64(10 ** 10)
skey = np.sort(st_i.astype(np.int64) * SC + s_i)
ekey = np.sort(st_i.astype(np.int64) * SC + e_i)
# all customer rentals, station-sorted
cust = ~ops[rs]
rkey = np.sort(rs[cust].astype(np.int64) * SC + rt[cust])
# lower-bound stock just before each rental (includes the rented bike)
n_at = np.searchsorted(skey, rkey, 'right') - np.searchsorted(ekey, rkey, 'left')
R['E_rentals_with_no_reconstructed_bike'] = float((n_at <= 0).mean())
n_at = np.maximum(n_at, 2)                     # a skipped bike was present too
v = np.log1p(-1.0 / n_at)                       # P(this bike not chosen | usable, uniform choice)
C = np.concatenate([[0.0], np.cumsum(v)])
lo = np.searchsorted(rkey, st_i.astype(np.int64) * SC + s_i, 'right')
hi = np.searchsorted(rkey, st_i.astype(np.int64) * SC + e_i, 'left')
skips = hi - lo
loglik = C[hi] - C[lo]                          # log P(skipped every time | usable)
strong = loglik <= np.log(0.01)                  # LR >= 100 in favour of "unusable"
hrs = (e_i - s_i) / 3600
R['E_intervals'] = int(iv.sum())
R['E_skips_quantiles'] = {q: float(np.quantile(skips, q)) for q in (0.5, 0.9, 0.99)}
R['E_strong_share_intervals'] = float(strong.mean())
R['E_strong_share_bike_hours'] = float(hrs[strong].sum() / hrs.sum())
day = (((s_i // 3600) % 24) >= 7) & (((s_i // 3600) % 24) <= 21)
R['E_strong_share_bike_hours_le72h'] = float(hrs[strong & (hrs <= 72)].sum() / hrs[hrs <= 72].sum())
# validation: what happens right after the interval ends (the bike's next trip)
nxt_trip = k1[iv]
nf = failed[nxt_trip]
R['E_next_trip_failed'] = dict(strong=float(nf[strong].mean()), not_strong=float(nf[~strong].mean()))
# pair after next trip: relocated / to ops
pos = np.searchsorted(k, nxt_trip)
okp = (pos < len(k)) & (k[np.minimum(pos, len(k) - 1)] == nxt_trip)
p2 = pos[okp]
st2 = strong[okp]
R['E_after_next_trip'] = dict(
    strong_relocated_or_ops=float((reloc[p2] | to_ops[p2])[st2].mean()),
    not_strong_relocated_or_ops=float((reloc[p2] | to_ops[p2])[~st2].mean()),
    strong_idle_over_24h=float((gap[p2] > 86400)[st2].mean()),
    not_strong_idle_over_24h=float((gap[p2] > 86400)[~st2].mean()))
# per-bike: how concentrated is strong evidence?
b_iv = b_[k][iv]
per_bike = np.bincount(b_iv, weights=strong.astype(float), minlength=len(bikes))
per_bike_n = np.bincount(b_iv, minlength=len(bikes))
okb = per_bike_n >= 20
R['E_bikes_strong_rate_quantiles'] = {q: float(np.quantile(per_bike[okb] / per_bike_n[okb], q))
                                      for q in (0.5, 0.9, 0.99)}
# does the prefix matter (generation)?
R['E_strong_rate_by_prefix'] = {p: float(strong[prefix[b_iv] == p].mean())
                                for p in np.unique(prefix) if (prefix[b_iv] == p).sum() > 1000}

# ---------------------------------------------------------------- F. group rides
g = np.lexsort((rt, rs))
rs_g, rt_g, ts_g, tt_g = rs[g], rt[g], ts[g], tt[g]
pair_grp = ((rs_g[1:] == rs_g[:-1]) & (rt_g[1:] - rt_g[:-1] <= 60) &
            (ts_g[1:] == ts_g[:-1]) & (np.abs(tt_g[1:] - tt_g[:-1]) <= 120) & ~ops[rs_g[1:]])
in_grp = np.zeros(N, bool)
in_grp[1:] |= pair_grp
in_grp[:-1] |= pair_grp
grp_start = np.concatenate([[True], ~pair_grp])   # first trip of each group/singleton
R['F_trips_in_groups'] = float(in_grp.mean())
R['F_groups'] = int((grp_start & np.concatenate([pair_grp, [False]])).sum())
R['F_group_share_by_daytype'] = {}
wd = ((rt_g // 86400) + 3) % 7
for name_, sel in (('weekday', wd < 5), ('weekend', wd >= 5)):
    R['F_group_share_by_daytype'][name_] = float(in_grp[sel].mean())

# ---------------------------------------------------------------- G. Poisson check
P = np.load(r'C:/Users/DISLAB/Desktop/KT X AI/tashu/data/processed/tashu_60min.npz')
bs = P['bin_start']
t0 = bs[0].astype('datetime64[s]').astype(np.int64)
pst = P['stations'].astype(str)
map_st = {s: i for i, s in enumerate(pst)}
col = np.array([map_st.get(s, -1) for s in stations])
T, NS = P['rentals'].shape
full = (P['exposure'] >= 0.999) & P['active'] & P['valid'][:, None] & ~P['closed'][:, None]
tb = (bs.astype('datetime64[h]').astype(np.int64))
hr = tb % 24
dow = ((tb // 24) + 3) % 7
mo = bs.astype('datetime64[M]').astype(np.int64)
train = (bs >= np.datetime64('2025-08-01')) & (bs < np.datetime64('2026-02-01'))
# counts from raw trips: all rentals vs first-of-group only
bi = (rt[g] - t0) // 3600
cc = col[rs_g]
ok = (bi >= 0) & (bi < T) & (cc >= 0)
raw = np.zeros((T, NS), np.int32)
np.add.at(raw, (bi[ok], cc[ok]), 1)
grp = np.zeros((T, NS), np.int32)
m2 = ok & grp_start
np.add.at(grp, (bi[m2], cc[m2]), 1)
# also drop quick re-rents (merged series used by the project)
merged = P['rentals']
cells_t, cells_n = np.nonzero(full & train[:, None] & (hr[:, None] >= 7) & (hr[:, None] <= 21))
key = ((cells_n.astype(np.int64) * 24 + hr[cells_t]) * 2 + (dow[cells_t] >= 5)) * 1000 + mo[cells_t]
uk, inv = np.unique(key, return_inverse=True)
cnt = np.bincount(inv)


def disp(Y):
    y = Y[cells_t, cells_n].astype(float)
    s1 = np.bincount(inv, y)
    s2 = np.bincount(inv, y * y)
    use = cnt >= 8
    mean = s1[use] / cnt[use]
    var = (s2[use] - cnt[use] * mean ** 2) / (cnt[use] - 1)
    z0 = np.bincount(inv, (y == 0).astype(float))[use] / cnt[use]
    w = cnt[use]
    busy = mean >= 1
    return dict(groups=int(use.sum()),
                pooled_var_over_mean=float((var * w).sum() / (mean * w).sum()),
                busy_groups_var_over_mean=float((var[busy] * w[busy]).sum() / (mean[busy] * w[busy]).sum()),
                share_groups_dispersion_gt_1_5=float((var > 1.5 * mean)[mean > 0.2].mean()),
                zero_obs=float((z0 * w).sum() / w.sum()),
                zero_poisson=float((np.exp(-mean) * w).sum() / w.sum()))


R['G_poisson_check'] = dict(raw_trips=disp(raw), merged_rerent=disp(merged),
                            groups_collapsed=disp(grp))

json.dump(R, open('bike_findings.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
