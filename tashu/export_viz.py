"""설명 페이지용 수치를 data/viz.json 으로 내보낸다."""
import csv
import json
import numpy as np

with np.load('data/processed/tashu_60min.npz') as f:
    z = {k: f[k] for k in f.files}
with np.load('data/demand/demand_lbx.npz') as f:
    d = {k: f[k] for k in f.files}
summ = json.load(open('data/processed/summary_60min.json', encoding='utf-8'))
vs = json.load(open('data/demand/validation_sim.json', encoding='utf-8'))
vr = json.load(open('data/demand/validation_real.json', encoding='utf-8'))
fit = json.load(open('data/demand/summary_lbx.json', encoding='utf-8'))

bs = z['bin_start']; days = bs.astype('datetime64[D]')
hour = (bs - days).astype('timedelta64[h]').astype(int)
dow = (days.astype(np.int64) + 3) % 7
month = bs.astype('datetime64[M]')
A = z['active'] & z['valid'][:, None]
y = z['rentals'].astype(float)
ulo, uhi, mu = np.nan_to_num(d['unmet_low']), np.nan_to_num(d['unmet_high']), d['mu']
valid_t = z['valid']

out = {}
# 1. 월별 관측 대여량 (일평균) — 장애 달은 유효 일수로 나눔
mo = []
for m in np.unique(month):
    sel = month == m
    vd = len(np.unique(days[sel & valid_t]))
    tot = y[sel].sum()
    mo.append(dict(month=str(m), per_day=round(tot / vd) if vd else None, valid_days=int(vd),
                   unmet_lo=round(ulo[sel].sum() / vd) if vd else None, unmet_hi=round(uhi[sel].sum() / vd) if vd else None))
out['monthly'] = mo

# 2. 시간대별 (평일, 유효일) : 대여소당 평균 관측/놓친 수요, 재고 0 비율
wd = dow < 5
hr = []
for h in range(24):
    sel = (hour == h) & wd & valid_t
    if not sel.any():
        continue
    a = A[sel]
    ndays = len(np.unique(days[sel]))
    hr.append(dict(hour=h,
                   observed=round(y[sel][a].sum() / ndays, 1),
                   unmet_lo=round(ulo[sel][a].sum() / ndays, 1), unmet_hi=round(uhi[sel][a].sum() / ndays, 1),
                   empty_lb=round(float((z['exposure'][sel][a] == 0).mean()), 4),
                   empty_ub=round(float((np.maximum(z['exposure_ub'], z['exposure'])[sel][a] == 0).mean()), 4)))
out['hourly_weekday'] = hr

# 3. 대여소 지도
names = {r['station_id']: r for r in csv.DictReader(open('data/processed/stations.csv', encoding='utf-8-sig'))}
rows = {r['station_id']: r for r in csv.DictReader(open('data/demand/station_unmet_lbx.csv', encoding='utf-8-sig'))}
pts = []
for i, s in enumerate(z['stations']):
    r = rows.get(s)
    if r is None or not np.isfinite(z['lat'][i]):
        continue
    nm = r['name'] if '�' not in r['name'] else f'{s} (원본 이름 깨짐)'
    gu = r['gu'] if '�' not in r['gu'] else ''
    pts.append([round(float(z['lon'][i]), 5), round(float(z['lat'][i]), 5), int(r['observed_rentals']),
                round(float(r['unmet_low'])), round(float(r['unmet_high'])), round(float(r['stockout_hour_share']), 3),
                nm, gu, r['reliable'] == 'True', int(r['rank'])])
out['stations'] = pts

# 4. 한 대여소의 하루 예시: 충남대 중앙도서관, 2025-10 평일 중 놓친 수요가 중간쯤인 날
sid = [i for i, s in enumerate(z['stations']) if names.get(s, {}).get('name') == '충남대학교 중앙도서관'][0]
cand = []
for dday in np.unique(days[(month == np.datetime64('2025-10')) & wd & valid_t]):
    sel = days == dday
    cand.append((ulo[sel, sid].sum(), y[sel, sid].sum(), dday))
cand.sort()
_, _, dday = cand[len(cand) * 3 // 4]
sel = np.nonzero(days == dday)[0]
out['example'] = dict(station='충남대학교 중앙도서관', date=str(dday), hours=[dict(
    hour=int(hour[t]), inv=int(z['inv_start'][t, sid]), exposure=round(float(z['exposure'][t, sid]), 3),
    rentals=int(y[t, sid]), returns=int(z['returns'][t, sid]),
    mu=None if np.isnan(mu[t, sid]) else round(float(mu[t, sid]), 2),
    unmet=round(float(ulo[t, sid]), 2), unmet_hi=round(float(uhi[t, sid]), 2)) for t in sel])

# 5. 대여소별 낮 시간 재고 0 비율 분포 (하한) + 이용량 3분위
day_sel = valid_t & (hour >= 7) & (hour < 22)
ex = z['exposure'][day_sel]; act = A[day_sel]
zr = ((ex == 0) & act).sum(0) / np.maximum(act.sum(0), 1)
tot = y.sum(0)
ok = act.sum(0) > 500
q = np.quantile(tot[ok], [1 / 3, 2 / 3])
tier = np.digitize(tot, q)
bins = np.linspace(0, 1, 11)
out['empty_hist'] = [dict(tier=nm, counts=np.histogram(zr[ok & (tier == k)], bins)[0].tolist())
                     for k, nm in enumerate(('이용 적음', '중간', '이용 많음'))]

out['numbers'] = dict(trips_raw=9116473, trips=summ['trips'], merged=9116465 - summ['trips'],
                      stations=summ['N'], bikes=6776, relocations=summ['relocations'],
                      same_station=summ['same_station_pair_rate'],
                      censored_daytime=summ['censored_share_daytime'], zero_daytime=summ['zero_exposure_share_daytime'],
                      unmet_total=fit['unmet_total'], unmet_share=fit['unmet_share'], observed_total=fit['observed_total'],
                      unreliable=fit['unreliable_stations'])
out['validation_sim'] = dict(censored=vs['censored_share'], results=vs['results'])
out['validation_real'] = vr['results']
json.dump(out, open('data/viz.json', 'w', encoding='utf-8'), ensure_ascii=False, separators=(',', ':'),
          default=lambda o: o.item() if hasattr(o, 'item') else str(o))
print('ok', len(pts), out['example']['date'])
