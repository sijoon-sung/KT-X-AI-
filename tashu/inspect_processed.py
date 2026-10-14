"""전처리 결과 점검: 일별 건수, 재고 합계 대비 운용 대수, 대여소별 절단 비율 분포."""
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else 'data/processed/tashu_60min.npz'
with np.load(path) as f:
    z = {k: f[k] for k in f.files}
bs = z['bin_start']; T, N = z['rentals'].shape
days = bs.astype('datetime64[D]')
valid, active = z['valid'], z['active']

def daily(x):
    u, inv = np.unique(days, return_inverse=True)
    return u, np.bincount(inv, weights=x)

u, r = daily(z['rentals'].sum(1))
for lo, hi in (('2025-02-24', '2025-03-06'),):
    m = (u >= np.datetime64(lo)) & (u <= np.datetime64(hi))
    print('일별 대여:', {str(a): int(b) for a, b in zip(u[m], r[m])})

# 시간대별로 대여소에 서 있는 자전거 합계 (재고 하한의 총량)
hour = (bs - days).astype('timedelta64[h]').astype(int)
stock = z['inv_start'].sum(1).astype(float)
month = bs.astype('datetime64[M]')
print('\n월  | 평균 총재고(04시) | 평균 총재고(18시) | 낮 절단비율')
for m in np.unique(month):
    sel = (month == m) & valid
    if not sel.any():
        continue
    a4, a18 = stock[sel & (hour == 4)].mean(), stock[sel & (hour == 18)].mean()
    day_sel = sel & (hour >= 7) & (hour < 22)
    cen = z['censored'][day_sel][active[day_sel]].mean()
    print(f'{str(m)} | {a4:8.0f} | {a18:8.0f} | {cen:.1%}')

# 대여소별 낮 시간 노출 0 비율 분포
day_sel = valid & (hour >= 7) & (hour < 22)
ex = z['exposure'][day_sel]; act = active[day_sel]
zero_rate = ((ex == 0) & act).sum(0) / np.maximum(act.sum(0), 1)
tot_r = z['rentals'].sum(0)
q = np.quantile(zero_rate[act.sum(0) > 0], [.1, .25, .5, .75, .9])
print('\n대여소별 낮 재고0 비율 분위수 p10/25/50/75/90:', np.round(q, 2))
busy = tot_r >= np.quantile(tot_r, .9)
print('상위10% 대여 대여소의 재고0 비율 중앙값:', round(float(np.median(zero_rate[busy])), 2))
