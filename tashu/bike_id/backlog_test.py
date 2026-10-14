"""결품 중 수요는 손실(lost)인가 대기(backlog)인가.
자전거가 대여소에 도착(반납)한 시점의 재고 n (도착 직전, 자전거번호로 복원) 별로
'도착 → 그 대여소의 다음 대여' 까지의 시간 분포를 비교. 07~21시, 고객 대여소.
n=0 (빈 곳에 첫 자전거) 에서 간격이 재고 있는 경우보다 크게 짧으면 기다리던 손님이 있다는 뜻.
"""
import json
import numpy as np

z = np.load('trips.npz')
bike, rt, rs, tt, ts = z['bike'], z['rt'], z['rs'], z['tt'], z['ts']
stations = z['stations']; ops = np.isin(stations, ['ST0001', 'ST1220', 'ST1198'])
o = np.lexsort((rt, bike)); b_, rt_, rs_, tt_, ts_ = bike[o], rt[o], rs[o], tt[o], ts[o]
k = np.flatnonzero(b_[1:] == b_[:-1]); k1 = k + 1
same = (ts_[k] == rs_[k1]) & ~ops[ts_[k]] & (rt_[k1] > tt_[k])
iv_s, s0, e0 = ts_[k][same], tt_[k][same], rt_[k1][same]
SC = np.int64(10 ** 10)
skey = np.sort(iv_s.astype(np.int64) * SC + s0); ekey = np.sort(iv_s.astype(np.int64) * SC + e0)
# 도착 직전 재고 (복원 하한): 시작<=t 인 구간 수 − 종료<=t 인 구간 수, 자기 자신 제외
arrivals_key = iv_s.astype(np.int64) * SC + s0
n_before = (np.searchsorted(skey, arrivals_key, 'left') - np.searchsorted(ekey, arrivals_key, 'right'))
# 다음 대여 (아무 자전거) 시각
cust = ~ops[rs]
rkey = np.sort(rs[cust].astype(np.int64) * SC + rt[cust])
pos = np.searchsorted(rkey, arrivals_key, 'right')
has = pos < len(rkey)
nxt = np.where(has, rkey[np.minimum(pos, len(rkey) - 1)], -1)
same_st = has & ((nxt // SC) == iv_s)
gap = np.where(same_st, nxt % SC - s0, np.nan) / 60.0                     # 분
hour = (s0 // 3600) % 24
day = (hour >= 7) & (hour <= 21)
# 대여소 시간당 대여율(전체 기간 평균)로 정규화: 기대 간격 = 60/λ
st_rate = np.bincount(rs[cust], minlength=len(stations)) / ((rt.max() - rt.min()) / 3600)
lam = st_rate[iv_s]
busy = lam >= 0.5
R = {}
for lab, m in (('n=0 (빈 곳 첫 도착)', n_before == 0), ('n=1', n_before == 1), ('n=2-4', (n_before >= 2) & (n_before <= 4)), ('n>=5', n_before >= 5)):
    sel = m & day & busy & ~np.isnan(gap)
    g = gap[sel]
    R[lab] = {'arrivals': int(sel.sum()), 'gap_median_min': float(np.median(g)),
              'share_rented_within_2min': float((g <= 2).mean()), 'share_within_10min': float((g <= 10).mean()),
              'gap_over_expected_median': float(np.median(g * lam[sel] / 60))}
# 같은 대여소 안에서 비교 (대여소 고정효과 제거): 대여소별 n=0 중앙값 / n>=2 중앙값
ratios = []
for s in np.unique(iv_s[day & busy]):
    a = (iv_s == s) & day & ~np.isnan(gap)
    g0, g2 = gap[a & (n_before == 0)], gap[a & (n_before >= 2)]
    if len(g0) >= 30 and len(g2) >= 30:
        ratios.append(np.median(g0) / np.median(g2))
R['within_station_ratio_n0_over_n2plus'] = {'stations': len(ratios), 'median': float(np.median(ratios)), 'q25': float(np.quantile(ratios, .25)), 'q75': float(np.quantile(ratios, .75))}
# 빈 시간이 길었을수록(대기 누적) 첫 도착 후 더 빨리 나가는가: 직전 구간 종료(재고 0 된 시각) 부터 도착까지의 공백
prev_end_pos = np.searchsorted(ekey, arrivals_key, 'right') - 1
prev_end = np.where(prev_end_pos >= 0, ekey[np.maximum(prev_end_pos, 0)], -1)
empty_min = np.where((prev_end // SC) == iv_s, (s0 - prev_end % SC) / 60.0, np.nan)
sel0 = (n_before == 0) & day & busy & ~np.isnan(gap) & ~np.isnan(empty_min)
R['n0_by_empty_duration'] = {}
for lab, lo, hi in (('<5min', 0, 5), ('5-30', 5, 30), ('30-120', 30, 120), ('>120', 120, 1e9)):
    m = sel0 & (empty_min > lo) & (empty_min <= hi)
    R['n0_by_empty_duration'][lab] = {'n': int(m.sum()), 'gap_median_min': float(np.median(gap[m])), 'within_2min': float((gap[m] <= 2).mean())}
print(json.dumps(R, ensure_ascii=False, indent=1)); json.dump(R, open('backlog_test.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
