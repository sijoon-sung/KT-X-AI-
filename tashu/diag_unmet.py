"""놓친 수요 추정치가 어디서 나오는지 분해: 노출이 거의 0 인데 대여가 있는 셀, 정비 거점 등."""
import csv
import numpy as np

with np.load('data/processed/tashu_60min.npz') as f:
    z = {k: f[k] for k in ('rentals', 'exposure', 'exposure_ub', 'active', 'valid', 'stations')}
with np.load('data/demand/demand_lb.npz') as f:
    unmet = f['unmet']
m = z['active'] & z['valid'][:, None]
y, e, eu = z['rentals'], z['exposure'], np.maximum(z['exposure_ub'], z['exposure'])
u = np.nan_to_num(unmet)
tot = u[m].sum()
for name, sel in [('e_lb < 0.1 & y > 0', m & (e < 0.1) & (y > 0)),
                  ('e_lb < 0.1 & y = 0', m & (e < 0.1) & (y == 0)),
                  ('0.1 <= e_lb < 1', m & (e >= 0.1) & (e < 1)),
                  ('e_lb < 0.1 & y > 0 & e_ub >= 0.5', m & (e < 0.1) & (y > 0) & (eu >= 0.5))]:
    print(f'{name:34s} cells {sel.sum():>10,}  unmet share {u[sel].sum() / tot:.1%}  rentals {y[sel].sum():,}')
names = {r['station_id']: r['name'] for r in csv.DictReader(open('data/processed/stations.csv', encoding='utf-8-sig'))}
print('depot-like:', [(s, names.get(s)) for s in z['stations'] if any(k in names.get(s, '') for k in ('관제', '정비', '센터', '창고', '물류'))])
