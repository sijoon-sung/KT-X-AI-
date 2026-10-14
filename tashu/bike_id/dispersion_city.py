"""Is station-hour overdispersion explained by a city-wide shared factor?

Pearson dispersion sum((y-m)^2)/sum(m) on full-exposure cells (07-21, 2025-08..2026-01, merged re-rents),
with m = station x hour x daytype x month mean, then m scaled by a city factor per day or per hour.
The city factor is computed from the same cells (oracle), so this is an upper bound of what a
perfect city-level forecast (weather, events) could explain.
"""
import json
import numpy as np

P = np.load(r'C:/Users/DISLAB/Desktop/KT X AI/tashu/data/processed/tashu_60min.npz')
bs = P['bin_start']
tb = bs.astype('datetime64[h]').astype(np.int64)
hr, dow, day = tb % 24, ((tb // 24) + 3) % 7, tb // 24
mo = bs.astype('datetime64[M]').astype(np.int64)
train = (bs >= np.datetime64('2025-08-01')) & (bs < np.datetime64('2026-02-01'))
full = (P['exposure'] >= 0.999) & P['active'] & P['valid'][:, None] & ~P['closed'][:, None]
ct, cn = np.nonzero(full & train[:, None] & (hr[:, None] >= 7) & (hr[:, None] <= 21))
y = P['rentals'][ct, cn].astype(float)
key = ((cn.astype(np.int64) * 24 + hr[ct]) * 2 + (dow[ct] >= 5)) * 1000 + mo[ct]
_, g = np.unique(key, return_inverse=True)
cnt = np.bincount(g)
use = cnt[g] >= 8
m = (np.bincount(g, y) / cnt)[g]


def pearson(mu, sel):
    return float(((y - mu) ** 2)[sel].sum() / mu[sel].sum())


def city_factor(unit):
    _, u = np.unique(unit, return_inverse=True)
    f = np.bincount(u, y) / np.bincount(u, m)
    return m * f[u]


R = {'cells': int(use.sum())}
R['station_hour_mean_only'] = pearson(m, use & (m > 0))
md = city_factor(day[ct])
R['x_city_day_factor'] = pearson(md, use & (md > 0))
mh = city_factor(tb[ct])
R['x_city_hour_factor'] = pearson(mh, use & (mh > 0))
busy = use & (m >= 1)
R['busy_station_hour_mean_only'] = pearson(m, busy)
R['busy_x_city_hour_factor'] = pearson(mh, busy & (mh > 0))
# zeros vs Poisson under the city-hour model
R['zero_obs'] = float((y[use] == 0).mean())
R['zero_poisson_city_hour'] = float(np.exp(-mh[use]).mean())
print(json.dumps(R, indent=1))
json.dump(R, open('dispersion_city.json', 'w'), indent=1)
