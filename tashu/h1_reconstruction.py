"""H1: 재고 복원 규칙 비교.  (재배치 자전거 도착 시각) × (방치 절단 기준)
도착 시각:  rent   = 다음 대여 직전 1초 (현행 하한)
            mid    = 공백의 중간
            batch  = 같은 출발→도착·같은 날 무리의 가장 빠른 다음 대여 시각 (무리 아니면 mid)
            depart = 출발 대여소 반납 직후 (현행 상한의 도착 쪽)
방치 절단:  72h / 24h (같은 대여소에 이보다 오래 서 있으면 이후 제외)
기준: (i) 재고 0 인데 대여 발생 비율  (ii) 2025-09 복원 vs 2026-09 API 실측: 낮 시간 빈 대여소 비율, 대여소별 평균 재고 상관
"""
import json
import numpy as np

import baseline_udf as B

TR = np.load(r'C:/Users/DISLAB/AppData/Roaming/Claude/scratch-workspaces/b0df6897-a471-4883-884c-fdc5e8c4e400/fd9184c3-66bc-4d36-9f89-67a0abc45a3f/scratch-2026-09-24-53c25d/bike/trips.npz')
bike, rt, rs, tt, ts = TR['bike'], TR['rt'], TR['rs'], TR['tt'], TR['ts']
stations = TR['stations']; ops = np.isin(stations, ['ST0001', 'ST1220', 'ST1198'])
# 3분 내 재대여 합치기 (preprocess 와 동일)
o = np.lexsort((rt, bike)); bike, rs, ts, rt, tt = bike[o], rs[o], ts[o], rt[o], tt[o]
cont = np.concatenate([[False], (bike[1:] == bike[:-1]) & (rs[1:] == ts[:-1]) & (rt[1:] >= tt[:-1]) & (rt[1:] - tt[:-1] < 180)])
head = np.flatnonzero(~cont); tail = np.concatenate([head[1:] - 1, [len(bike) - 1]])
bike, rs, rt = bike[head], rs[head], rt[head]; ts, tt = ts[tail], tt[tail]
N = len(stations)
nxt = bike[1:] == bike[:-1]
a_t, a_s, c_t, c_s = tt[:-1][nxt], ts[:-1][nxt], rt[1:][nxt], rs[1:][nxt]
okp = c_t >= a_t
same = (a_s == c_s) & okp; reloc = (a_s != c_s) & okp
# 무리: 같은 (from,to, 도착일=다음 대여일) → 최소 다음 대여 시각
key = (a_s[reloc].astype(np.int64) * 5000 + c_s[reloc]) * 100000 + c_t[reloc] // 86400
uk, inv_k = np.unique(key, return_inverse=True)
kmin = np.full(len(uk), np.iinfo(np.int64).max); np.minimum.at(kmin, inv_k, c_t[reloc])
ksize = np.bincount(inv_k)
batch_arrival = np.where(ksize[inv_k] >= 2, kmin[inv_k], (a_t[reloc] + c_t[reloc]) // 2)
batch_arrival = np.minimum(batch_arrival, c_t[reloc] - 1)

# 2025-09 낮 시간 (07~21시) 1시간 격자
t0 = np.datetime64('2025-09-01').astype('datetime64[s]').astype(np.int64); t1 = np.datetime64('2025-10-01').astype('datetime64[s]').astype(np.int64)
T = (t1 - t0) // 3600
hour = np.arange(T) % 24; day = (hour >= 7) & (hour <= 21)
cust_rent = (~ops[rs]) & (rt >= t0) & (rt < t1) & (((rt // 3600) % 24 >= 7) & ((rt // 3600) % 24 <= 21))


def reconstruct(arrival, idle_h):
    cap = idle_h * 3600
    st = [a_s[same]]; s = [a_t[same]]; e = [np.minimum(c_t[same], a_t[same] + cap)]
    arr = {'rent': c_t[reloc] - 1, 'mid': (a_t[reloc] + c_t[reloc]) // 2, 'batch': batch_arrival, 'depart': a_t[reloc]}[arrival]
    st.append(c_s[reloc]); s.append(arr); e.append(c_t[reloc])
    st, s, e = map(np.concatenate, (st, s, e))
    k = (e > s) & (e > t0) & (s < t1); st, s, e = st[k], np.maximum(s[k], t0), np.minimum(e[k], t1)
    # 시간 격자 시작 재고
    D = np.zeros((T + 1, N), np.int32)
    np.add.at(D, (np.minimum(-((t0 - s) // 3600), T), st), 1); np.add.at(D, (np.minimum(-((t0 - e) // 3600), T), st), -1)
    inv = np.cumsum(D, 0)[:T]
    # 각 대여 직전 재고 (초 단위): 시작<=t 구간 수 − 종료<=t 구간 수
    SC = np.int64(10 ** 10)
    skey = np.sort(st.astype(np.int64) * SC + s); ekey = np.sort(st.astype(np.int64) * SC + e)
    rk = rs[cust_rent].astype(np.int64) * SC + rt[cust_rent]
    n_at = np.searchsorted(skey, rk, 'right') - np.searchsorted(ekey, rk, 'right')
    return inv, n_at


# API 실측 (2026-09, 07~21시)
ts_api, sids, stock = B.load_api(r'C:/Users/DISLAB/Downloads/tashu_stock_10min.csv.gz')
h_api = (ts_api.astype('datetime64[h]').astype(np.int64)) % 24
dsel = (h_api >= 7) & (h_api <= 21)
api_mean = np.nanmean(stock[dsel], 0); api_empty = float(np.nanmean(stock[dsel] == 0))
idx = {s: i for i, s in enumerate(stations.astype(str))}
m_idx = np.array([idx[s] for s in sids if s in idx]); a_idx = np.array([j for j, s in enumerate(sids) if s in idx])
active = np.zeros(N, bool); active[np.unique(np.concatenate([rs[(rt >= t0) & (rt < t1)], ts[(tt >= t0) & (tt < t1)]]))] = True
R = {'api_2026_09': {'empty_share_daytime': api_empty, 'mean_stock_per_station': float(np.nanmean(api_mean)), 'stations': int(len(a_idx))}}
for arrival in ('rent', 'mid', 'batch', 'depart'):
    for idle in (72, 24):
        inv, n_at = reconstruct(arrival, idle)
        sel = active & ~ops
        inv_d = inv[day][:, sel]
        rec_mean = inv[day].mean(0)
        both = sel[m_idx]
        R[f'{arrival}/{idle}h'] = {
            'rentals_at_zero_stock': round(float((n_at <= 0).mean()), 4),
            'empty_share_daytime': round(float((inv_d == 0).mean()), 4),
            'mean_stock': round(float(inv_d.mean()), 2),
            'corr_station_mean_vs_api': round(float(np.corrcoef(rec_mean[m_idx][both], np.nan_to_num(api_mean[a_idx][both]))[0, 1]), 3),
            'ratio_mean_stock_vs_api': round(float(rec_mean[m_idx][both].mean() / np.nanmean(api_mean[a_idx][both])), 3)}
        print(arrival, idle, R[f'{arrival}/{idle}h'], flush=True)
R['batch_share_of_relocations'] = float((ksize[inv_k] >= 2).mean())
json.dump(R, open('data/h1_reconstruction.json', 'w', encoding='utf-8'), indent=1)
print(json.dumps(R['api_2026_09'], indent=1), 'batch share', R['batch_share_of_relocations'])
