"""자전거번호로 할 수 있는 모델링 검사.
A. 선택 규칙: 대여 시점에 서 있던 자전거 중 어느 것을 고르는가 (체류 순위, 세대)
B. 자전거별 기피 성향의 지속성: 학습기간(~2026-01) 성향이 2026-02/03 의 skip·대여실패를 예측하는가
C. 유효 재고: '기피 자전거'를 뺀 재고가 결품(대여 실패) 신호를 더 잘 설명하는가
"""
import json
import numpy as np

z = np.load('trips.npz')
bike, rt, rs, tt, ts = z['bike'], z['rt'], z['rs'], z['tt'], z['ts']
bikes, stations = z['bikes'], z['stations']
ops = np.isin(stations, ['ST0001', 'ST1220', 'ST1198'])
prefix = np.array([b.split('-')[0] for b in bikes])
o = np.lexsort((rt, bike)); b_, rt_, rs_, tt_, ts_ = bike[o], rt[o], rs[o], tt[o], ts[o]
k = np.flatnonzero(b_[1:] == b_[:-1]); k1 = k + 1
same = (ts_[k] == rs_[k1]) & ~ops[ts_[k]] & (rt_[k1] > tt_[k])
# 체류 구간: 자전거 b 가 대여소 s 에 [s0, e0) 동안 서 있음 (e0 = 다음 대여 시각)
iv_b, iv_s, iv_s0, iv_e0 = b_[k][same], ts_[k][same], tt_[k][same], rt_[k1][same]
failed_next = ((rs_[k1] == ts_[k1]) & (tt_[k1] - rt_[k1] <= 120))[same]     # 그 체류 뒤 대여가 실패(2분 내 같은 곳 반납)
cut = np.datetime64('2026-02-01').astype('datetime64[s]').astype(np.int64)
R = {}

# ---------------------------------------------------------------- A. 선택 규칙
# 각 대여(체류 종료 e0) 시점에 같은 대여소에 서 있던 자전거 집합 = 구간 [s0,e0) 가 e0 를 포함하는 것들
SC = np.int64(10 ** 10)
order = np.argsort(iv_s.astype(np.int64) * SC + iv_s0)
S_, s0, e0, bb = iv_s[order], iv_s0[order], iv_e0[order], iv_b[order]
# 대여소별로 스윕: 각 종료 사건에 대해 현재 서 있는 집합에서 선택된 자전거의 체류 순위 계산 (표본: 대여소별 최대 3000 사건)
rng = np.random.default_rng(0)
rank_rel = []; gen_pick = []; n_present = []
starts = np.searchsorted(S_, np.arange(len(stations)), 'left'); ends = np.searchsorted(S_, np.arange(len(stations)), 'right')
for s in range(len(stations)):
    a, b2 = starts[s], ends[s]
    if b2 - a < 200 or ops[s]:
        continue
    s0s, e0s, bs = s0[a:b2], e0[a:b2], bb[a:b2]
    # 사건: 각 구간의 종료. 그 시점에 존재하는 다른 구간: s0 <= e0_i < e0_j
    idx = rng.choice(b2 - a, min(300, b2 - a), replace=False)
    for i in idx:
        t = e0s[i]
        present = (s0s <= t) & (e0s > t)            # 아직 안 떠난 것
        present[i] = True
        n = int(present.sum())
        if n < 2:
            continue
        arr = s0s[present]                          # 도착 시각
        # 선택된 자전거의 '오래됨' 순위: 0 = 가장 최근 도착, 1 = 가장 오래됨
        r = float((arr < s0s[i]).sum()) / (n - 1)
        rank_rel.append(r); n_present.append(n)
        g = prefix[bs[present]]
        if (g == 'DJ2').any() and (g == 'DJ3').any():
            gen_pick.append((prefix[bs[i]] == 'DJ3', float((g == 'DJ3').mean())))
rank_rel = np.array(rank_rel); n_present = np.array(n_present)
R['A_events'] = int(len(rank_rel))
R['A_oldest_rank_mean(0.5=random,1=FIFO,0=LIFO)'] = float(rank_rel.mean())
R['A_oldest_rank_by_n'] = {lab: float(rank_rel[m].mean()) for lab, m in (('2-3', n_present <= 3), ('4-9', (n_present > 3) & (n_present < 10)), ('10+', n_present >= 10))}
R['A_rank_hist'] = np.histogram(rank_rel, bins=5, range=(0, 1))[0].tolist()
gp = np.array(gen_pick)
R['A_DJ3_picked_share'] = float(gp[:, 0].mean()); R['A_DJ3_present_share'] = float(gp[:, 1].mean()); R['A_mixed_events'] = int(len(gp))

# ---------------------------------------------------------------- B. 기피 성향 지속성
# 자전거별 skip 지표: 체류 중 다른 자전거가 대여된 횟수 (이 자전거는 안 뽑힘) 대비 기대치
# 근사: 체류 시간 × 그 대여소의 시간당 대여율 (학습기간 평균) 를 기대 skip 으로, 실제 skip 은 체류 중 그 대여소 대여 수
cust = ~ops[rs]
rkey = np.sort(rs[cust].astype(np.int64) * SC + rt[cust])
lo = np.searchsorted(rkey, iv_s.astype(np.int64) * SC + iv_s0, 'right'); hi = np.searchsorted(rkey, iv_s.astype(np.int64) * SC + iv_e0, 'left')
skips = hi - lo
hours = (iv_e0 - iv_s0) / 3600
st_rate = np.bincount(rs[cust], minlength=len(stations)) / ((rt.max() - rt.min()) / 3600)
exp_skip = hours * st_rate[iv_s]
train = iv_e0 < cut; test = ~train
def propensity(mask):
    obs = np.bincount(iv_b[mask], weights=skips[mask], minlength=len(bikes))
    ex = np.bincount(iv_b[mask], weights=exp_skip[mask], minlength=len(bikes))
    n = np.bincount(iv_b[mask], minlength=len(bikes))
    return (obs + 5) / (ex + 5), n            # 축소된 skip 비 (1 = 평균)
p_tr, n_tr = propensity(train); p_te, n_te = propensity(test)
okb = (n_tr >= 30) & (n_te >= 10)
R['B_bikes'] = int(okb.sum())
R['B_corr_train_test_skip_ratio'] = float(np.corrcoef(np.log(p_tr[okb]), np.log(p_te[okb]))[0, 1])
q = np.quantile(p_tr[okb], [0.2, 0.8])
low, high = okb & (p_tr <= q[0]), okb & (p_tr >= q[1])
R['B_test_skip_ratio_by_train_quintile'] = {'bottom20%': float(p_te[low].mean()), 'top20%': float(p_te[high].mean())}
# 대여 실패율 예측: 학습 성향 상위 20% 자전거의 2~3월 다음 대여 실패율
fr = np.bincount(iv_b[test], weights=failed_next[test], minlength=len(bikes)) / np.maximum(np.bincount(iv_b[test], minlength=len(bikes)), 1)
R['B_test_failed_rate_by_train_quintile'] = {'bottom20%': float(fr[low].mean()), 'top20%': float(fr[high].mean())}
# 학습기간 대여실패 이력이 시험기간 실패를 예측하는가
ff_tr = np.bincount(iv_b[train], weights=failed_next[train], minlength=len(bikes)) / np.maximum(np.bincount(iv_b[train], minlength=len(bikes)), 1)
q2 = np.quantile(ff_tr[okb], [0.2, 0.8]); l2, h2 = okb & (ff_tr <= q2[0]), okb & (ff_tr >= q2[1])
R['B_test_failed_rate_by_train_failed_quintile'] = {'bottom20%': float(fr[l2].mean()), 'top20%': float(fr[h2].mean())}
R['B_prefix_skip_ratio'] = {p: float(p_tr[okb & (prefix == p)].mean()) for p in ('DJ2', 'DJ3')}

# ---------------------------------------------------------------- C. 유효 재고: 기피 자전거 비중이 큰 대여소에서 대여 실패가 더 잦은가 (시험기간)
sticky = p_tr >= q[1]
# 시험기간 각 대여소의 체류 자전거-시간 중 기피 자전거 비중
bh = np.bincount(iv_s[test], weights=hours[test], minlength=len(stations))
bh_st = np.bincount(iv_s[test], weights=hours[test] * sticky[iv_b[test]], minlength=len(stations))
share_sticky = bh_st / np.maximum(bh, 1e-9)
test_trip = (rt >= cut) & cust
fail_trip = test_trip & (rs == ts) & (tt - rt <= 120)
fail_rate_st = np.bincount(rs[fail_trip], minlength=len(stations)) / np.maximum(np.bincount(rs[test_trip], minlength=len(stations)), 1)
oks = (np.bincount(rs[test_trip], minlength=len(stations)) >= 200) & (bh > 100)
R['C_stations'] = int(oks.sum())
R['C_corr_sticky_share_vs_failed_rate'] = float(np.corrcoef(share_sticky[oks], fail_rate_st[oks])[0, 1])
qq = np.quantile(share_sticky[oks], [0.25, 0.75])
R['C_failed_rate_by_sticky_share'] = {'low25%': float(fail_rate_st[oks & (share_sticky <= qq[0])].mean()), 'high25%': float(fail_rate_st[oks & (share_sticky >= qq[1])].mean())}
R['C_sticky_share_quantiles'] = {p: float(np.quantile(share_sticky[oks], p)) for p in (0.1, 0.5, 0.9)}
print(json.dumps(R, ensure_ascii=False, indent=1))
json.dump(R, open('bike_choice.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
