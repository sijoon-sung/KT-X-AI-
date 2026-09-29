"""전처리 v2: trips.parquet -> 대여소(노드) × 1시간 패널.

바뀐 점 (outputs/eda 점검 결과 반영)
  1. 관제센터(ST1220): 대여소 수요에서 뺌. 대전 전체 '위치 모름' 대여 건수로 따로 저장
     - 실제 대여소에서 빌려 관제센터로 반납된 기록의 '대여' 는 그 대여소 수요로 남긴다
  2. 취소 추정: 같은 대여소 번호로 1분 이하 반납 -> 대여·반납 모두 뺌. 24시간 초과도 뺌
  3. 대여소 정의: 번호 + 좌표. 같은 번호라도 좌표가 200m 넘게 옮겨지면 다른 노드
     (구·동 글자는 표기가 바뀌므로 쓰지 않음)
  4. 운영 기간: 노드별 첫 기록 ~ 마지막 기록, 그 사이 21일(유효일) 이상 무기록 구간은 운영 안 함
  5. 시간 가리기
     - 새벽 미운영일: 그날 0~4시 전체 대여가 50건 이하면 0~4시는 '운영 안 함'
     - 장애: 운영 시간인데 대전 전체 대여가 0건인 시간 + 앞뒤 1시간 (12월 전체, 2025-02-26~03-04 포함)
     - 기존 규칙이 잘못 버린 2024-08-01~06, 2025-03-04~06 은 살아남

산출: data/processed/v2/
  nodes.parquet    node, station_id, name, lat, lon, capacity, first_ts, last_ts, n_rent
  panel.npz        ts(T), rent(T,N), ret(T,N), node_active(T,N), hour_ok(T), night_ops(T), outage(T), ctrl_rent(T)
  od_month.parquet month, src, dst, n   (노드 간 이동 건수, 그래프용)
  report.json      걸러낸 건수 등
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = PROC / "v2"
CTRL = "ST1220"
MOVE_M = 200.0
GAP_DAYS = 21


def hav_m(a1, o1, a2, o2):
    a1, o1, a2, o2 = (np.radians(np.asarray(x, float)) for x in (a1, o1, a2, o2))
    return 2 * 6371000 * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))


def build_nodes(t: pd.DataFrame):
    """번호별로 좌표를 처음 나온 순서대로 훑으며, 기준 좌표에서 200m 넘게 벗어나면 새 노드."""
    ev = pd.concat(
        [
            t[["rent_st", "rent_lat", "rent_lon", "rent_ts", "rent_name"]].set_axis(["st", "lat", "lon", "ts", "name"], axis=1),
            t[["ret_st", "ret_lat", "ret_lon", "ret_ts", "ret_name"]].set_axis(["st", "lat", "lon", "ts", "name"], axis=1),
        ]
    )
    ev = ev[ev.st != CTRL]
    ev["lat5"], ev["lon5"] = ev.lat.round(5), ev.lon.round(5)
    co = (
        ev.groupby(["st", "lat5", "lon5"])
        .agg(first=("ts", "min"), last=("ts", "max"), n=("ts", "size"), name=("name", lambda s: s.mode().iat[0]))
        .reset_index()
        .sort_values(["st", "first"])
    )
    seg_rows, key2node = [], {}
    for st, g in co.groupby("st", sort=True):
        anchor, seg = None, -1
        for r in g.itertuples():
            if anchor is None or hav_m(anchor[0], anchor[1], r.lat5, r.lon5) > MOVE_M:
                seg += 1
                anchor = (r.lat5, r.lon5)
                seg_rows.append({"station_id": st, "seg": seg, "lat": r.lat5, "lon": r.lon5, "first_ts": r.first,
                                 "last_ts": r.last, "n_events": r.n, "name": r.name})
            else:
                s = seg_rows[-1]
                s["last_ts"] = max(s["last_ts"], r.last)
                s["first_ts"] = min(s["first_ts"], r.first)
                if r.n > s["n_events"]:  # 가장 많이 쓰인 좌표·이름을 대표로
                    s.update(lat=r.lat5, lon=r.lon5, name=r.name)
                s["n_events"] += r.n
            key2node[(st, r.lat5, r.lon5)] = (st, seg)
    nodes = pd.DataFrame(seg_rows).sort_values(["station_id", "seg"]).reset_index(drop=True)
    nodes["node"] = np.arange(len(nodes))
    nid = {(r.station_id, r.seg): r.node for r in nodes.itertuples()}
    key2id = {k: nid[v] for k, v in key2node.items()}
    return nodes, key2id


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.read_parquet(PROC / "trips.parquet")
    for c in ("rent_st", "ret_st"):
        t[c] = t[c].astype(str)
    rep = {"trips_in": len(t)}

    too_long = t.dur_min > 24 * 60
    cancel = (t.rent_st == t.ret_st) & (t.dur_min <= 1)
    rep["drop_over_24h"] = int(too_long.sum())
    rep["drop_cancel_same_station_le1min"] = int((cancel & ~too_long).sum())
    t = t[~too_long & ~cancel].reset_index(drop=True)

    end = pd.Timestamp("2026-04-01")
    ts = pd.date_range("2024-08-01", end, freq="h", inclusive="left")
    T = len(ts)

    nodes, key2id = build_nodes(t)
    N = len(nodes)
    rep["nodes"] = N
    rep["station_ids"] = int(nodes.station_id.nunique())
    rep["station_ids_split_by_move"] = int((nodes.groupby("station_id").size() > 1).sum())

    # 거치대 수 (Open API 스냅샷, 현재 기준이라 옮겨지기 전 노드에는 안 붙임)
    snap = json.loads((ROOT / "data/raw/stations_api_snapshot.json").read_text(encoding="utf-8"))
    import re

    cap = {}
    for x in snap["results"]:
        m = re.search(r"/\s*(\d+)\s*$", x.get("name_cn") or "")
        if m:
            cap[x["id"]] = (int(m.group(1)), float(x["x_pos"]), float(x["y_pos"]))
    def cap_of(r):
        if r.station_id not in cap:
            return np.nan
        c, la, lo = cap[r.station_id]
        return c if hav_m(la, lo, r.lat, r.lon) <= MOVE_M else np.nan
    nodes["capacity"] = [cap_of(r) for r in nodes.itertuples()]

    def node_of(st, la, lo):
        keys = list(zip(st, la.round(5), lo.round(5)))
        return np.array([key2id.get(k, -1) for k in keys], dtype=np.int32)

    t["rent_node"] = node_of(t.rent_st.values, t.rent_lat, t.rent_lon)
    t["ret_node"] = node_of(t.ret_st.values, t.ret_lat, t.ret_lon)
    rt = ((t.rent_ts - ts[0]) // pd.Timedelta(hours=1)).to_numpy()
    et = ((t.ret_ts - ts[0]) // pd.Timedelta(hours=1)).to_numpy()

    rent = np.zeros((T, N), np.float32)
    ret = np.zeros((T, N), np.float32)
    m = (t.rent_node.values >= 0) & (rt >= 0) & (rt < T)
    np.add.at(rent, (rt[m], t.rent_node.values[m]), 1)
    m = (t.ret_node.values >= 0) & (et >= 0) & (et < T)
    np.add.at(ret, (et[m], t.ret_node.values[m]), 1)
    ctrl = np.zeros(T, np.float32)
    m = (t.rent_st.values == CTRL) & (rt >= 0) & (rt < T)
    np.add.at(ctrl, rt[m], 1)
    rep["ctrl_center_rentals_kept_separately"] = int(m.sum())

    # ---- 시간 가리기 ----
    sys_rent = rent.sum(1) + ctrl
    hour = ts.hour.to_numpy()
    day = ts.normalize()
    night = pd.Series(sys_rent * (hour < 5)).groupby(day).transform("sum").to_numpy()
    night_ops = night > 50
    operating = night_ops | (hour >= 5)
    zero = operating & (sys_rent == 0)
    outage = zero.copy()
    outage[1:] |= zero[:-1]
    outage[:-1] |= zero[1:]
    hour_ok = operating & ~outage
    rep["hours_total"] = T
    rep["hours_not_operating_night"] = int((~operating).sum())
    rep["hours_outage"] = int((outage & operating).sum())
    out_days = pd.Series(outage & operating, index=ts).resample("D").sum()
    rep["outage_days_with_>=6h"] = [d.strftime("%Y-%m-%d") for d in out_days[out_days >= 6].index]
    nd = pd.Series(~night_ops, index=ts).resample("D").first()
    runs, cur = [], None
    for d, v in nd.items():
        if v and cur is None:
            cur = d
        if not v and cur is not None:
            runs.append(f"{cur:%Y-%m-%d}~{d - pd.Timedelta(days=1):%Y-%m-%d}")
            cur = None
    if cur is not None:
        runs.append(f"{cur:%Y-%m-%d}~{nd.index[-1]:%Y-%m-%d}")
    rep["no_night_service_periods"] = runs

    # ---- 노드 운영 기간 ----
    first_h = ((nodes.first_ts - ts[0]) // pd.Timedelta(hours=1)).clip(0, T - 1).to_numpy()
    last_h = ((nodes.last_ts - ts[0]) // pd.Timedelta(hours=1)).clip(0, T - 1).to_numpy()
    active = np.zeros((T, N), bool)
    for i in range(N):
        active[first_h[i] : last_h[i] + 1, i] = True
    # 긴 공백: 유효일(장애 아닌 날) 기준 21일 이상 대여·반납 0
    day_idx = (np.arange(T) // 24)
    D = day_idx.max() + 1
    ev_day = np.zeros((D, N), np.float32)
    np.add.at(ev_day, day_idx, rent + ret)
    day_valid = pd.Series(hour_ok).groupby(day_idx).sum().to_numpy() >= 12
    n_gap_cells = 0
    for i in range(N):
        vdays = np.nonzero(day_valid)[0]
        has = ev_day[vdays, i] > 0
        run_start = None
        for j, (d, h) in enumerate(zip(vdays, has)):
            if not h and run_start is None:
                run_start = j
            if (h or j == len(vdays) - 1) and run_start is not None:
                run_end = j - 1 if h else j
                if run_end - run_start + 1 >= GAP_DAYS:
                    a, b = vdays[run_start], vdays[run_end]
                    active[a * 24 : (b + 1) * 24, i] = False
                    n_gap_cells += (b - a + 1)
                run_start = None
    span = np.zeros((T, N), bool)
    for i in range(N):
        span[first_h[i] : last_h[i] + 1, i] = True
    rep["node_days_inactive_long_gap_within_span"] = int((span & ~active).sum() // 24)
    rep["nodes_with_long_gap"] = int(((span & ~active).sum(0) > 0).sum())
    rep["node_hours_active_share"] = float(active.mean())
    nodes["n_rent"] = rent.sum(0)

    np.savez_compressed(OUT / "panel.npz", ts=ts.values.astype("datetime64[h]").astype(np.int64), rent=rent, ret=ret,
                        node_active=active, hour_ok=hour_ok, night_ops=night_ops, outage=outage, ctrl_rent=ctrl)
    nodes.to_parquet(OUT / "nodes.parquet", index=False)

    od = t[(t.rent_node >= 0) & (t.ret_node >= 0)]
    od = od.groupby([od.rent_ts.dt.to_period("M").astype(str), "rent_node", "ret_node"]).size().rename("n").reset_index()
    od.columns = ["month", "src", "dst", "n"]
    od.to_parquet(OUT / "od_month.parquet", index=False)

    valid_cells = active & hour_ok[:, None]
    rep["valid_cells"] = int(valid_cells.sum())
    rep["valid_cells_share_of_full_panel"] = float(valid_cells.mean())
    rep["rent_in_valid_cells"] = int(rent[valid_cells].sum())
    rep["rent_outside_valid_cells"] = int(rent[~valid_cells].sum())
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
