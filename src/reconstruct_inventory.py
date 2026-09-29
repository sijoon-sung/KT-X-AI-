"""① 재고 복원: 자전거번호 연쇄로 대여소별 15분 재고를 되살린다 (과거 대여이력만 사용).

원리
  대여 = 그 대여소에서 자전거 한 대가 빠짐(-1), 반납 = 한 대가 들어옴(+1).
  자전거번호로 기록을 이어 붙이면 "직전 반납 대여소"와 "다음 대여 대여소"가 보통 같다.
  다르면 그 사이에 누군가 옮긴 것이므로, 옮김 사건(-1 저기, +1 여기)을 하나 더 만들어 준다.
  이렇게 하면 대여소별 재고 변화가 닫힌다. 남는 문제는 "시작 수량을 모른다"는 것뿐이다.

모르는 것과 처리 방법 (모두 옵션으로 바꿔 가며 비교 가능)
  옮긴 시각    --move-at {midpoint, next_rent, on_return}   기본 midpoint
  정비 반출    --maint-hours H   반납 후 H시간 넘게 안 쓰이면 그 사이에는 대여소에 없다고 본다 (기본 72, 0이면 끔)
  시작 수량    --anchor {min0, daily, none}   기본 min0 (관측 기간 최소 재고를 0으로 맞춤)
  위치 미상    관제센터(ST1220)는 대여소가 아니므로, 그쪽으로 간 자전거는 재고에서 빠진다

산출  data/processed/v3/inventory_15min.npz   slot_ts, node, stock(int16), net(int16), 마스크
      data/processed/v3/inventory_hourly.parquet  ts, node, stock_start/min/mean, empty_frac, rent, ret
      data/processed/v3/inventory_report.json     정합성 점검 숫자
실행  python src/reconstruct_inventory.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
V2 = PROC / "v2"
OUT = PROC / "v3"
CTRL = "ST1220"
SLOT_MIN = 15


def load_trips(nodes: pd.DataFrame) -> pd.DataFrame:
    """전처리 v2 와 같은 규칙으로 기록을 거르고, 대여소 번호+좌표로 노드를 붙인다."""
    t = pd.read_parquet(PROC / "trips.parquet",
                        columns=["bike_id", "rent_ts", "ret_ts", "rent_st", "ret_st", "rent_name", "ret_name",
                                 "rent_lat", "rent_lon", "ret_lat", "ret_lon", "dur_min"])
    for c in ("rent_st", "ret_st"):
        t[c] = t[c].astype(str)
    n0 = len(t)
    t = t[(t.dur_min <= 24 * 60) & ~((t.rent_st == t.ret_st) & (t.dur_min <= 1))]
    # 전처리 v2 와 똑같은 (번호, 좌표) -> 노드 대응표를 그대로 쓴다
    from preprocess_v2 import build_nodes
    _, key2id = build_nodes(t)
    def to_node(st, la, lo):
        keys = list(zip(st, pd.Series(la).round(5), pd.Series(lo).round(5)))
        return np.array([key2id.get(k, -1) for k in keys], np.int32)
    t["rent_node"] = to_node(t.rent_st.values, t.rent_lat, t.rent_lon)
    t["ret_node"] = to_node(t.ret_st.values, t.ret_lat, t.ret_lon)
    bad = t.rent_node < 0
    print(f"기록 {n0:,} -> 사용 {len(t):,} | 노드 못 붙인 대여 {int(bad.sum()):,} "
          f"(그중 관제센터 {int((t.rent_st[bad] == CTRL).sum()):,})", flush=True)
    return t.sort_values(["bike_id", "rent_ts"]).reset_index(drop=True)


def build_events(t: pd.DataFrame, args) -> tuple[pd.DataFrame, dict]:
    """이용 사건 + 옮김 사건 + 정비 반출/반입 사건을 한 표로 만든다."""
    rep = {}
    ev = [
        pd.DataFrame({"ts": t.rent_ts, "node": t.rent_node, "delta": -1, "kind": "rent"}),
        pd.DataFrame({"ts": t.ret_ts, "node": t.ret_node, "delta": 1, "kind": "ret"}),
    ]
    g = t.groupby("bike_id", sort=False)
    nxt_node = g.rent_node.shift(-1)
    nxt_ts = g.rent_ts.shift(-1)
    has_next = nxt_node.notna()
    prev_node = t.ret_node
    prev_ts = t.ret_ts
    gap_h = (nxt_ts - prev_ts).dt.total_seconds() / 3600
    moved = has_next & (nxt_node.fillna(-99).astype(np.int32) != prev_node)
    rep["chain_pairs"] = int(has_next.sum())
    rep["chain_same_station"] = float((~moved & has_next).sum() / max(1, has_next.sum()))
    rep["moves_total"] = int(moved.sum())
    rep["moves_per_day"] = round(float(moved.sum()) / max(1.0, (t.rent_ts.max() - t.rent_ts.min()).days), 1)
    rep["move_gap_hours_median"] = round(float(gap_h[moved].median()), 2)

    if args.move_at == "midpoint":
        t_move = prev_ts + (nxt_ts - prev_ts) / 2
    elif args.move_at == "next_rent":
        t_move = nxt_ts - pd.Timedelta(minutes=1)
    else:  # on_return: 반납 직후 옮겨졌다고 봄
        t_move = prev_ts + pd.Timedelta(minutes=1)
    mv = pd.DataFrame({"ts": t_move[moved], "from": prev_node[moved], "to": nxt_node[moved].astype(np.int32)})
    ev.append(pd.DataFrame({"ts": mv.ts, "node": mv["from"], "delta": -1, "kind": "move_out"}))
    ev.append(pd.DataFrame({"ts": mv.ts, "node": mv["to"], "delta": 1, "kind": "move_in"}))

    if args.maint_hours > 0:
        idle = has_next & ~moved & (gap_h > args.maint_hours)
        rep["maint_events"] = int(idle.sum())
        t_out = prev_ts[idle] + pd.Timedelta(hours=args.maint_hours)
        t_in = nxt_ts[idle] - pd.Timedelta(minutes=1)
        ev.append(pd.DataFrame({"ts": t_out, "node": prev_node[idle], "delta": -1, "kind": "maint_out"}))
        ev.append(pd.DataFrame({"ts": t_in, "node": prev_node[idle], "delta": 1, "kind": "maint_in"}))
    else:
        rep["maint_events"] = 0

    e = pd.concat(ev, ignore_index=True)
    e = e[(e.node >= 0) & e.ts.notna()]
    return e, rep


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_parquet(V2 / "nodes.parquet")
    panel = np.load(V2 / "panel.npz")
    hours = pd.to_datetime(panel["ts"].astype("datetime64[h]"))
    N = len(nodes)

    t = load_trips(nodes)
    ev, rep = build_events(t, args)

    # ---- 15분 격자에 사건을 얹어 누적 ----
    t0 = hours[0]
    end = hours[-1] + pd.Timedelta(hours=1)
    slots = pd.date_range(t0, end, freq=f"{SLOT_MIN}min", inclusive="left")
    S = len(slots)
    idx = ((ev.ts - t0) // pd.Timedelta(minutes=SLOT_MIN)).to_numpy()
    ok = (idx >= 0) & (idx < S)
    net = np.zeros((S, N), np.float32)
    np.add.at(net, (idx[ok], ev.node.to_numpy()[ok]), ev.delta.to_numpy()[ok])
    stock = np.cumsum(net, axis=0)

    # ---- 시작 수량 맞추기 ----
    if args.anchor == "min0":
        stock -= stock.min(axis=0, keepdims=True)
    elif args.anchor == "daily":
        day = np.repeat(np.arange(S // (96) + 1), 96)[:S]
        for d in np.unique(day):
            m = day == d
            stock[m] -= stock[m].min(axis=0, keepdims=True)
    drift = stock[-1] - stock[0]
    rep["anchor"] = args.anchor
    rep["move_at"] = args.move_at
    rep["maint_hours"] = args.maint_hours

    # ---- 정합성 점검 ----
    active = panel["node_active"]
    hour_ok = panel["hour_ok"]
    obs_h = active & hour_ok[:, None]
    obs_s = np.repeat(obs_h, 4, axis=0)[:S]
    st = stock[obs_s]
    rep["slots"] = int(S)
    rep["nodes"] = int(N)
    rep["stock_negative_share"] = round(float((st < -0.5).mean()), 4)
    rep["stock_zero_share"] = round(float((st < 0.5).mean()), 4)
    rep["stock_mean"] = round(float(st.mean()), 2)
    rep["city_stock_mean"] = round(float(stock.sum(1).mean()), 1)
    rep["city_stock_p05_p95"] = [round(float(np.percentile(stock.sum(1), 5)), 1), round(float(np.percentile(stock.sum(1), 95)), 1)]
    rep["drift_per_node_median"] = round(float(np.median(drift)), 2)
    cap = nodes.capacity.to_numpy(np.float32)
    okc = np.isfinite(cap)
    rep["over_capacity_share"] = round(float((stock[:, okc][obs_s[:, okc]] > cap[okc][None, :].repeat(S, 0)[obs_s[:, okc]]).mean()), 4)

    # 가장 중요한 점검: 재고가 0 이라고 복원된 시간에 실제 대여 기록이 있으면 모순
    rent_slot = np.zeros((S, N), np.float32)
    ri = ((t.rent_ts - t0) // pd.Timedelta(minutes=SLOT_MIN)).to_numpy()
    rok = (ri >= 0) & (ri < S) & (t.rent_node.to_numpy() >= 0)
    np.add.at(rent_slot, (ri[rok], t.rent_node.to_numpy()[rok]), 1)
    zero = (stock < 0.5) & obs_s
    rep["rentals_when_stock_zero_share"] = round(float(rent_slot[zero].sum() / max(1.0, rent_slot[obs_s].sum())), 4)
    rep["zero_slots_with_rental_share"] = round(float((rent_slot[zero] > 0).mean()), 4)

    # 복원 재고가 대여 발생 확률을 설명하는가 (재고 구간별 대여 확률)
    bins = [0, 1, 2, 3, 5, 10, 1e9]
    lab = ["0", "1", "2", "3~4", "5~9", "10+"]
    b = np.digitize(stock[obs_s], bins) - 1
    prob = [float((rent_slot[obs_s][b == i] > 0).mean()) if (b == i).any() else np.nan for i in range(len(lab))]
    rep["rental_prob_by_stock"] = {k: round(v, 4) for k, v in zip(lab, prob)}

    # ---- 저장 ----
    np.savez_compressed(OUT / "inventory_15min.npz", slots=slots.values.astype("datetime64[m]").astype(np.int64),
                        stock=np.clip(stock, -32000, 32000).astype(np.int16), net=net.astype(np.int16))
    hourly = pd.DataFrame({
        "ts": np.repeat(hours.values, N),
        "node": np.tile(np.arange(N), len(hours)),
        "stock_start": stock[::4][: len(hours)].ravel(),
        "stock_min": np.minimum.reduceat(stock, np.arange(0, S, 4))[: len(hours)].ravel(),
        "stock_mean": np.add.reduceat(stock, np.arange(0, S, 4))[: len(hours)].ravel() / 4,
        "empty_frac": np.add.reduceat((stock < 0.5).astype(np.float32), np.arange(0, S, 4))[: len(hours)].ravel() / 4,
        "rent": panel["rent"].ravel(),
        "ret": panel["ret"].ravel(),
        "obs": obs_h.ravel(),
    })
    hourly = hourly[hourly.obs].drop(columns="obs")
    hourly.to_parquet(OUT / "inventory_hourly.parquet", index=False)
    (OUT / "inventory_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))

    print("\n시각별: 복원 재고가 0 인 대여소 비율과 실제 대여 건수")
    h = hourly.assign(hour=hourly.ts.dt.hour).groupby("hour").agg(
        빈대여소비율=("empty_frac", "mean"), 평균재고=("stock_mean", "mean"), 시간당대여=("rent", "mean")).round(3)
    print(h.to_string())
    return rep


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--move-at", choices=["midpoint", "next_rent", "on_return"], default="midpoint")
    p.add_argument("--maint-hours", type=float, default=72.0)
    p.add_argument("--anchor", choices=["min0", "daily", "none"], default="min0")
    main(p.parse_args())
