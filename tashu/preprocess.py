"""타슈 대여이력 → 대여소×시간 텐서 변환.

대여이력 ZIP(월별 CSV, cp949)만으로 다음을 만든다.

  rentals[T,N]    구간 내 대여 건수 (관측 수요 = 절단된 수요)
  returns[T,N]    구간 내 반납 건수
  inv_start[T,N]  구간 시작 시점 재고 (자전거번호 궤적으로 복원한 하한)
  exposure[T,N]   구간 중 재고 > 0 이었던 시간 비율 (0~1)
  censored[T,N]   exposure < 1  → 이 구간의 대여량은 수요의 하한일 뿐
  active[T,N]     대여소 운영 기간 (첫/마지막 이용 사이)
  valid[T]        데이터가 존재하는 구간 (시스템 장애일 제외)

재고 복원 원리: 같은 자전거의 연속 이용 (a, c) 에서 a 반납 대여소 == c 대여 대여소이면
자전거는 [a.반납, c.대여] 동안 그 대여소에 서 있었다. 다르면 트럭 재배치/정비로
보고 relocations.csv 에 기록하되, 재고에는 넣지 않는다 (시각을 모르므로).
따라서 재고는 하한이고, 절단 판정(exposure<1)은 보수적(절단을 더 많이 표시)이다.
절단이 아닌 구간을 절단으로 처리해도 censored likelihood 는 여전히 참인 부등식이라
편향이 생기지 않지만, 반대 방향 오류는 수요를 과소추정하게 만든다.
"""
import argparse
import csv
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np

COLS = dict(bike=0, rent_t=1, rent_st=2, rent_name=3, rent_lat=4, rent_lon=5, rent_gu=6,
            ret_t=9, ret_st=10, ret_name=11, ret_lat=12, ret_lon=13, ret_gu=14,
            minutes=17, km=18)


DT_RE = re.compile(r'(\d{4}-\d\d-\d\d)[ T](\d{1,2}):(\d\d)(?::(\d\d))?')


def norm_dt(v):
    """'2025-03-05 7:16' 같은 변형(25년 3월 파일)을 'YYYY-MM-DD HH:MM:SS' 로 정규화."""
    if len(v) == 19:
        return v
    m = DT_RE.match(v)
    return f'{m[1]} {int(m[2]):02d}:{m[3]}:{m[4] or "00"}' if m else None


def read_zip(zip_path):
    """모든 월별 CSV를 읽어 정수 코드 배열과 대여소/자전거 사전을 반환 (파일 단위로 변환해 메모리 절약).

    파일명의 'YY년MM월' 과 대여일시의 연-월이 다른 행은 버린다
    (실제로 '25년12월' 파일에는 25년 1월 데이터가 중복 수록되어 있음).
    """
    bike_id, st_id, st_meta = {}, {}, {}
    parts = {k: [] for k in ('bike', 'rent_t', 'rent_st', 'ret_t', 'ret_st')}
    bad = mismatch = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = sorted(n for n in zf.namelist() if n.lower().endswith('.csv'))
        for name in names:
            ym = re.search(r'(\d\d)년(\d\d)월', name)
            prefix = f'20{ym[1]}-{ym[2]}' if ym else None
            bk, rt, rs, tt, ts = [], [], [], [], []
            file_mis = 0
            with zf.open(name) as fh:
                r = csv.reader(io.TextIOWrapper(fh, encoding='cp949', errors='replace'))
                next(r)
                for x in r:
                    if len(x) < 19 or not x[2] or not x[10]:
                        bad += 1
                        continue
                    x[1], x[9] = norm_dt(x[1]), norm_dt(x[9])
                    if x[1] is None or x[9] is None:
                        bad += 1
                        continue
                    if prefix and not x[1].startswith(prefix):
                        file_mis += 1
                        continue
                    for sid, nm, la, lo, gu in ((x[2], x[3], x[4], x[5], x[6]), (x[10], x[11], x[12], x[13], x[14])):
                        if sid not in st_id:
                            st_id[sid] = len(st_id)
                        if sid not in st_meta or '�' in st_meta[sid][0]:   # 깨진 이름은 깨끗한 것으로 교체
                            try:
                                st_meta[sid] = (nm, float(la), float(lo), gu)
                            except ValueError:
                                pass
                    bk.append(bike_id.setdefault(x[0], len(bike_id)))
                    rt.append(x[1]); rs.append(st_id[x[2]]); tt.append(x[9]); ts.append(st_id[x[10]])
            parts['bike'].append(np.array(bk, np.int32))
            parts['rent_st'].append(np.array(rs, np.int32))
            parts['ret_st'].append(np.array(ts, np.int32))
            parts['rent_t'].append(np.array(rt, dtype='datetime64[s]').astype(np.int64))
            parts['ret_t'].append(np.array(tt, dtype='datetime64[s]').astype(np.int64))
            mismatch += file_mis
            print(f'  {name}: {len(bk):,} rows' + (f'  (파일명 월과 불일치 {file_mis:,}행 제외)' if file_mis else ''), flush=True)
    d = {k: np.concatenate(v) for k, v in parts.items()}
    return d, st_id, st_meta, bad, mismatch


def detect_outage_days(rent_t, day0, n_days, min_trips):
    per_day = np.bincount((rent_t - day0) // 86400, minlength=n_days)[:n_days]
    return per_day < min_trips, per_day


def add_interval_coverage(cov, st, s, e, t0, bin_s, T):
    """[s,e) 구간들이 각 (시간구간, 대여소)를 덮는 초를 cov 에 누적 (벡터화)."""
    sb = (s - t0) // bin_s
    eb = (e - t0) // bin_s
    same = sb == eb
    np.add.at(cov, (sb[same], st[same]), (e - s)[same])
    m = ~same
    sb, eb, st, s, e = sb[m], eb[m], st[m], s[m], e[m]
    np.add.at(cov, (sb, st), t0 + (sb + 1) * bin_s - s)
    in_range = eb < T
    np.add.at(cov, (eb[in_range], st[in_range]), (e - (t0 + eb * bin_s))[in_range])
    # 완전히 덮인 중간 구간들: 차분 배열
    full = np.zeros((T + 1, cov.shape[1]), dtype=np.int64)
    np.add.at(full, (sb + 1, st), 1)
    np.add.at(full, (np.minimum(eb, T), st), -1)
    cov += np.cumsum(full, axis=0)[:T] * bin_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--out', default='data/processed')
    ap.add_argument('--bin-min', type=int, default=60, help='시간 구간 길이(분)')
    ap.add_argument('--max-idle-h', type=float, default=72,
                    help='한 대여소에 이보다 오래 서 있으면 이후는 고장/미운영으로 보고 재고에서 제외')
    ap.add_argument('--min-trips-per-day', type=int, default=500, help='이보다 적은 날은 장애일로 마스킹')
    ap.add_argument('--gap-h', type=float, default=3, help='전체 대여가 이 시간 이상 끊기면 부분 장애로 마스킹')
    ap.add_argument('--merge-min', type=float, default=3,
                    help='같은 자전거·같은 대여소 재대여가 이 시간(분) 이내면 한 이용으로 합침 (0 = 끔)')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bin_s = args.bin_min * 60
    max_idle = int(args.max_idle_h * 3600)

    print('reading zip ...', flush=True)
    d, st_id, st_meta, bad, mismatch = read_zip(args.zip)
    ok = (d['ret_t'] >= d['rent_t']) & (d['ret_t'] - d['rent_t'] < 86400)
    print(f'rows {len(ok):,}  malformed {bad:,}  month-mismatch {mismatch:,}  dropped(bad duration) {(~ok).sum():,}')
    d = {k: v[ok] for k, v in d.items()}

    # 대여소 인덱스를 ID 정렬 순으로 재배열
    raw_ids = np.array(list(st_id))
    order_ids = np.argsort(raw_ids)
    stations = raw_ids[order_ids]
    remap = np.empty(len(raw_ids), np.int32); remap[order_ids] = np.arange(len(raw_ids))
    N = len(stations)
    rs, ts, bk = remap[d['rent_st']], remap[d['ret_st']], d['bike']
    rent_t, ret_t = d['rent_t'], d['ret_t']

    # 같은 자전거를 같은 대여소에 반납하고 merge_min 분 안에 다시 빌린 경우는 한 이용으로 합친다
    # (잠깐 세워두기·반납/대여 오류 — 새 손님이 아니며, 노출 몇 초에 대여 1건으로 수요를 부풀림)
    o = np.lexsort((rent_t, bk))
    bk, rs, ts, rent_t, ret_t = bk[o], rs[o], ts[o], rent_t[o], ret_t[o]
    cont = np.concatenate([[False], (bk[1:] == bk[:-1]) & (rs[1:] == ts[:-1])
                           & (rent_t[1:] >= ret_t[:-1]) & (rent_t[1:] - ret_t[:-1] < args.merge_min * 60)])
    head = np.nonzero(~cont)[0]
    tail = np.concatenate([head[1:] - 1, [len(bk) - 1]])
    print(f'merged quick re-rents (< {args.merge_min} min): {cont.sum():,} ({cont.mean():.1%})')
    bk, rs, rent_t = bk[head], rs[head], rent_t[head]
    ts, ret_t = ts[tail], ret_t[tail]

    day0 = (rent_t.min() // 86400) * 86400                        # 시각은 naive(KST) 값 그대로 사용
    t_end = ((max(ret_t.max(), rent_t.max()) // 86400) + 1) * 86400
    n_days = (t_end - day0) // 86400
    T = (t_end - day0) // bin_s
    t0 = day0
    print(f'stations {N}  bikes {bk.max() + 1}  bins {T} x {args.bin_min}min')

    outage_day, per_day = detect_outage_days(rent_t, day0, n_days, args.min_trips_per_day)
    bins_per_day = 86400 // bin_s
    valid = np.repeat(~outage_day, bins_per_day)[:T]
    print('outage days:', [str(np.datetime64(int(day0 + i * 86400), 's'))[:10] for i in np.where(outage_day)[0]])
    # 전체 대여 스트림이 gap_h 이상 끊긴 구간:
    #  - 야간 운영중단(밤 21시 이후 ~ 다음날 06:30 이전, 8시간 미만): 자전거는 제자리 → closed (학습만 제외)
    #  - 그 외: 부분 장애 → valid=False (경계 구간 포함)
    srt = np.sort(rent_t)
    g = np.where(np.diff(srt) >= args.gap_h * 3600)[0]
    ga, gb = srt[g], srt[g + 1]
    tod_a, tod_b = (ga - day0) % 86400, (gb - day0) % 86400
    night = (gb - ga < 8 * 3600) & (tod_b <= 6.5 * 3600) & ((tod_a >= 21 * 3600) | ((ga - day0) // 86400 == (gb - day0) // 86400))
    closed = np.zeros(T, bool)
    for a, b, is_night in zip(ga, gb, night):
        if is_night:
            closed[-((t0 - a) // bin_s): (b - t0) // bin_s] = True     # 완전히 포함된 구간만
        else:
            valid[(a - t0) // bin_s: (b - t0) // bin_s + 1] = False
    print(f'stream gaps >= {args.gap_h}h: {len(g)} (night closures {night.sum()}), partial outages:',
          [f'{np.datetime64(int(a), "s")}~{np.datetime64(int(b), "s")}' for a, b, n in zip(ga, gb, night)
           if not n and b - a < 86400 * 2])

    # 대여/반납 카운트
    rb = (rent_t - t0) // bin_s
    tb = np.minimum((ret_t - t0) // bin_s, T - 1)
    rentals = np.zeros((T, N), np.int32); np.add.at(rentals, (rb, rs), 1)
    returns = np.zeros((T, N), np.int32); np.add.at(returns, (tb, ts), 1)

    # 자전거별 궤적 → 정차 구간
    order = np.lexsort((rent_t, bk))
    b_o, rt_o, rs_o, tt_o, ts_o = bk[order], rent_t[order], rs[order], ret_t[order], ts[order]
    nxt = b_o[1:] == b_o[:-1]                                     # 같은 자전거의 연속 이용 쌍
    a_ret_t, a_ret_s = tt_o[:-1][nxt], ts_o[:-1][nxt]
    c_rent_t, c_rent_s = rt_o[1:][nxt], rs_o[1:][nxt]
    gap_ok = c_rent_t >= a_ret_t
    ba = np.minimum((a_ret_t - t0) // bin_s, T - 1)
    bc = np.minimum((c_rent_t - t0) // bin_s, T - 1)
    # 장애 구간을 가로지르는 쌍은 중간 이용이 빠졌을 수 있으므로 재배치 판정에서 제외
    cum_inv = np.concatenate([[0], np.cumsum(~valid)])
    spans_outage = (cum_inv[bc + 1] - cum_inv[ba] > 0) & gap_ok
    same = (a_ret_s == c_rent_s) & gap_ok & ~spans_outage
    reloc = (a_ret_s != c_rent_s) & gap_ok & ~spans_outage
    print(f'consecutive pairs {nxt.sum():,}  same-station {same.sum() / (same.sum() + reloc.sum()):.2%}  '
          f'relocated {reloc.sum():,}  spanning-outage {spans_outage.sum():,}')

    first = np.concatenate([[True], ~nxt]); last = np.concatenate([~nxt, [True]])
    f_t, f_s = rt_o[first], rs_o[first]
    l_t, l_s = tt_o[last], ts_o[last]
    last_bad = np.maximum.accumulate(np.where(~valid, np.arange(T), -1))
    resume = t0 + (last_bad[bc[spans_outage]] + 1) * bin_s

    def segments(upper):
        """대여소에 자전거가 서 있던 구간 [s, e).

        하한(upper=False): 확실한 정차만. 같은 대여소 정차는 max_idle 로 절단(방치/고장 가능성),
                           재배치 자전거는 도착 시각을 모르므로 대여 직전 1초만.
        상한(upper=True):  같은 대여소 정차는 절단 없이, 재배치 자전거는 공백 전체 동안
                           출발·도착 두 대여소 모두에 있었다고 봄.
        """
        cap = np.iinfo(np.int64).max // 4 if upper else max_idle
        st = [a_ret_s[same]]; s = [a_ret_t[same]]; e = [np.minimum(c_rent_t[same], a_ret_t[same] + cap)]
        if upper:
            st += [a_ret_s[reloc], c_rent_s[reloc]]; s += [a_ret_t[reloc]] * 2; e += [c_rent_t[reloc]] * 2
        else:
            st.append(c_rent_s[reloc]); s.append(c_rent_t[reloc] - 1); e.append(c_rent_t[reloc])
        # 장애를 가로지른 자전거: 장애 종료 시점부터 다음 대여 대여소에 있었다고 봄 (7일 이내만)
        cr, cs = c_rent_t[spans_outage], c_rent_s[spans_outage]
        near = (cr - resume < 7 * 86400) & (cr > resume)
        st.append(cs[near]); s.append(resume[near]); e.append(cr[near])
        # 첫 이용 이전(데이터 시작 7일 내 첫 이용 = 기존 운용 자전거) / 마지막 반납 이후
        early = f_t - t0 < 7 * 86400
        st.append(f_s[early]); s.append(np.full(early.sum(), t0)); e.append(f_t[early])
        st.append(l_s); s.append(l_t); e.append(np.minimum(l_t + (7 * 86400 if upper else max_idle), t_end))
        st, s, e = map(np.concatenate, (st, s, e))
        k = e > s
        return st[k], s[k], e[k]

    def inventory_exposure(seg_st, seg_s, seg_e):
        # 구간 시작 재고: 구간 b 시작 시각 edge_b 에 대해 s <= edge_b < e 인 정차 구간 수
        D = np.zeros((T + 1, N), np.int32)
        np.add.at(D, (np.minimum(-((t0 - seg_s) // bin_s), T), seg_st), 1)   # ceil((s-t0)/bin)
        np.add.at(D, (np.minimum(-((t0 - seg_e) // bin_s), T), seg_st), -1)  # ceil((e-t0)/bin)
        inv = np.cumsum(D, axis=0)[:T]
        # exposure: 대여소별로 재고>0 인 연속 구간을 합친 뒤 구간별 덮인 시간 계산
        ev_t = np.concatenate([seg_s, seg_e]); ev_st = np.concatenate([seg_st, seg_st])
        ev_d = np.concatenate([np.ones_like(seg_s), -np.ones_like(seg_e)])
        o = np.lexsort((-ev_d, ev_t, ev_st))                      # 같은 시각이면 +1 먼저 (틈 없이 이어붙임)
        ev_t, ev_d, ev_st = ev_t[o], ev_d[o], ev_st[o]
        level = np.cumsum(ev_d)                                    # 대여소마다 0 으로 끝나므로 전역 누적 가능
        prev_level = np.concatenate([[0], level[:-1]])
        starts = (prev_level == 0) & (ev_d == 1)
        ends = (level == 0) & (ev_d == -1)
        u_st, u_s, u_e = ev_st[starts], ev_t[starts], ev_t[ends]
        assert len(u_s) == len(u_e) and (ev_st[ends] == u_st).all()
        cov = np.zeros((T, N), np.int64)
        u_e = np.minimum(u_e, t_end); m = u_e > u_s
        add_interval_coverage(cov, u_st[m], u_s[m], u_e[m], t0, bin_s, T)
        return inv, np.clip(cov / bin_s, 0, 1).astype(np.float32)

    inv_start, exposure = inventory_exposure(*segments(upper=False))
    inv_ub, exposure_ub = inventory_exposure(*segments(upper=True))

    # 대여소 운영 기간
    first_seen = np.full(N, np.iinfo(np.int64).max); last_seen = np.zeros(N, np.int64)
    np.minimum.at(first_seen, rs, rent_t); np.minimum.at(first_seen, ts, ret_t)
    np.maximum.at(last_seen, rs, rent_t); np.maximum.at(last_seen, ts, ret_t)
    edges = t0 + np.arange(T) * bin_s
    active = (edges[:, None] + bin_s > first_seen[None]) & (edges[:, None] <= last_seen[None])
    valid_data = valid.copy()                                      # 데이터 존재 여부 (재고 연속성 판단용)
    valid = valid & ~closed                                        # 학습/평가에 쓸 구간 (운영중단 제외)
    censored = (exposure < 1) & active & valid[:, None]
    censored_ub = (exposure_ub < 1) & active & valid[:, None]

    # OD 그래프
    od = np.zeros((N, N), np.int32); np.add.at(od, (rs, ts), 1)
    lat = np.array([st_meta.get(s, ('', np.nan, np.nan, ''))[1] for s in stations])
    lon = np.array([st_meta.get(s, ('', np.nan, np.nan, ''))[2] for s in stations])
    la, lo = np.radians(lat), np.radians(lon)
    h = np.sin((la[:, None] - la[None]) / 2) ** 2 + np.cos(la[:, None]) * np.cos(la[None]) * np.sin((lo[:, None] - lo[None]) / 2) ** 2
    dist_km = (2 * 6371 * np.arcsin(np.sqrt(h))).astype(np.float32)

    tag = f'{args.bin_min}min'
    np.savez_compressed(out / f'tashu_{tag}.npz', rentals=rentals, returns=returns,
                        inv_start=inv_start.astype(np.int16), exposure=exposure, censored=censored,
                        inv_ub=inv_ub.astype(np.int16), exposure_ub=exposure_ub, censored_ub=censored_ub,
                        active=active, valid=valid, valid_data=valid_data, closed=closed, bin_start=edges.astype('datetime64[s]'),
                        stations=stations, lat=lat, lon=lon, od=od, dist_km=dist_km)
    with open(out / 'stations.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['idx', 'station_id', 'name', 'lat', 'lon', 'gu', 'first_seen', 'last_seen'])
        for i, s in enumerate(stations):
            nm, a, b, gu = st_meta.get(s, ('', '', '', ''))
            w.writerow([i, s, nm, a, b, gu, np.datetime64(int(first_seen[i]), 's'), np.datetime64(int(last_seen[i]), 's')])
    with open(out / 'relocations.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.writer(fh); w.writerow(['from_station', 'to_station', 'left_after', 'arrived_before', 'gap_h'])
        for fs, tss, a, c in zip(stations[a_ret_s[reloc]], stations[c_rent_s[reloc]], a_ret_t[reloc], c_rent_t[reloc]):
            w.writerow([fs, tss, np.datetime64(int(a), 's'), np.datetime64(int(c), 's'), round((c - a) / 3600, 2)])

    # 요약
    A = active & valid[:, None]
    hour = ((edges - t0) % 86400) // 3600
    day = (hour >= 7) & (hour < 22)
    Ad = A & day[:, None]
    summary = dict(
        bin_min=args.bin_min, T=int(T), N=int(N), trips=int(len(rent_t)), merge_min=args.merge_min,
        period=[str(np.datetime64(int(t0), 's')), str(np.datetime64(int(t_end), 's'))],
        missing_data_bins=int((~valid_data).sum()), night_closed_bins=int(closed.sum()),
        same_station_pair_rate=float(same.sum() / (same.sum() + reloc.sum())), relocations=int(reloc.sum()),
        # [하한 재고 기준, 상한 재고 기준] — 실제 값은 이 사이
        censored_share_daytime=[float(censored[Ad].mean()), float(censored_ub[Ad].mean())],
        zero_exposure_share_daytime=[float((exposure[Ad] == 0).mean()), float((exposure_ub[Ad] == 0).mean())],
        rentals_in_censored_bins_share=[float(rentals[censored].sum() / rentals[A].sum()),
                                        float(rentals[censored_ub].sum() / rentals[A].sum())],
        mean_inv_start_daytime=[float(inv_start[Ad].mean()), float(inv_ub[Ad].mean())],
        negative_inv_cells=int((inv_start < 0).sum() + (inv_ub < 0).sum()),
    )
    (out / f'summary_{tag}.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
