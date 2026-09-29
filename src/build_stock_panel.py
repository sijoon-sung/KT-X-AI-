"""VM 이 모은 재고 스냅샷 -> 모델에 쓸 수 있는 표.

입력  data/raw/stock/YYYY-MM-DD.csv.gz   (ts_kst, station_id, parking_count)
      data/raw/stock/stations_log.csv     (대여소 이름·좌표·거치대 수가 바뀐 시점)

산출  data/processed/v3/stock_5min.parquet
        ts(5분 격자), station_id, stock, is_empty, is_full, capacity
      data/processed/v3/stock_hourly.parquet   ← 대여 패널과 바로 붙일 수 있는 형태
        ts(1시간), station_id, node(있으면), stock_start, stock_min, stock_mean,
        empty_frac(그 시간 중 빈 상태였던 비율), full_frac, n_snapshots
      data/processed/v3/stock_report.json

빈 대여소 시간을 알면 두 가지가 가능해진다.
  1. 재고 0 인 시간의 대여 0 건은 '수요 없음'이 아니라 '못 빌림'으로 구분
  2. 제안서가 목표한 결품을 예측 대상·정답으로 직접 사용

실행:  python src/build_stock_panel.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "stock"
OUT = ROOT / "data" / "processed" / "v3"


def load_snapshots() -> pd.DataFrame:
    files = sorted(RAW.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"재고 파일이 없습니다: {RAW}\n  scripts/sync_stock.sh 로 VM 에서 먼저 가져오세요.")
    df = pd.concat([pd.read_csv(f, dtype={"station_id": str}) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["ts_kst"])
    df["stock"] = pd.to_numeric(df["parking_count"], errors="coerce")
    return df.dropna(subset=["stock"]).drop(columns=["ts_kst", "parking_count"])


def capacity_table() -> pd.DataFrame:
    p = RAW / "stations_log.csv"
    if not p.exists():
        return pd.DataFrame(columns=["station_id", "capacity", "lat", "lon", "name"])
    log = pd.read_csv(p, dtype={"station_id": str})
    log["ts"] = pd.to_datetime(log["ts_kst"])
    last = log.sort_values("ts").groupby("station_id").last().reset_index()
    return last[["station_id", "capacity", "lat", "lon", "name"]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_snapshots()
    cap = capacity_table()

    # 한 번 찍을 때 1,374곳이 같은 시각을 갖는다. 5분 격자로 맞춰 중복은 마지막 값 사용
    df["ts5"] = df["ts"].dt.floor("5min")
    five = (
        df.sort_values("ts")
        .groupby(["ts5", "station_id"], as_index=False)
        .agg(stock=("stock", "last"), ts_raw=("ts", "last"))
        .rename(columns={"ts5": "ts"})
    )
    five = five.merge(cap[["station_id", "capacity"]], on="station_id", how="left")
    five["is_empty"] = five["stock"] == 0
    five["is_full"] = five["capacity"].notna() & (five["stock"] >= five["capacity"])
    five.to_parquet(OUT / "stock_5min.parquet", index=False)

    # 1시간 단위 요약: 대여 패널(v2) 과 같은 시간축에 붙여 쓰기 위한 형태
    five["hour"] = five["ts"].dt.floor("h")
    g = five.sort_values("ts").groupby(["hour", "station_id"])
    hourly = g.agg(
        stock_start=("stock", "first"),
        stock_min=("stock", "min"),
        stock_mean=("stock", "mean"),
        empty_frac=("is_empty", "mean"),
        full_frac=("is_full", "mean"),
        n_snapshots=("stock", "size"),
    ).reset_index().rename(columns={"hour": "ts"})

    # v2 전처리의 대여소(노드)와 연결. 번호가 같아도 위치가 옮겨졌으면 가까운 쪽에 붙인다
    nodes_p = ROOT / "data" / "processed" / "v2" / "nodes.parquet"
    if nodes_p.exists() and not cap.empty:
        nodes = pd.read_parquet(nodes_p)
        cap2 = cap.dropna(subset=["lat", "lon"]).copy()
        cap2[["lat", "lon"]] = cap2[["lat", "lon"]].astype(float)
        m = nodes.merge(cap2, on="station_id", how="inner", suffixes=("_node", "_api"))
        d = np.hypot((m.lat_node - m.lat_api) * 111000, (m.lon_node - m.lon_api) * 90000)
        m = m.assign(dist_m=d).sort_values("dist_m").groupby("station_id", as_index=False).first()
        link = m.loc[m.dist_m <= 200, ["station_id", "node", "dist_m"]]
        hourly = hourly.merge(link[["station_id", "node"]], on="station_id", how="left")
    hourly.to_parquet(OUT / "stock_hourly.parquet", index=False)

    span = (five["ts"].max() - five["ts"].min()) if len(five) else pd.Timedelta(0)
    slots = five["ts"].nunique()
    expected = int(span / pd.Timedelta(minutes=5)) + 1 if len(five) else 0
    rep = {
        "files": len(list(RAW.glob("*.csv.gz"))),
        "first_ts": str(five["ts"].min()),
        "last_ts": str(five["ts"].max()),
        "hours_covered": round(span / pd.Timedelta(hours=1), 1),
        "slots_collected": int(slots),
        "slots_expected": expected,
        "slot_coverage": round(slots / expected, 3) if expected else None,
        "stations": int(five["station_id"].nunique()),
        "rows_5min": int(len(five)),
        "linked_to_nodes": int(hourly["node"].notna().sum()) if "node" in hourly else 0,
        "empty_share": round(float(five["is_empty"].mean()), 3),
        "bikes_mean_total": round(float(five.groupby("ts")["stock"].sum().mean()), 1) if len(five) else None,
    }
    (OUT / "stock_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("\n대여소별 빈 시간 비율 상위 10곳")
    top = five.groupby("station_id").is_empty.mean().sort_values(ascending=False).head(10)
    names = cap.set_index("station_id")["name"] if not cap.empty else pd.Series(dtype=str)
    print(pd.DataFrame({"empty_frac": top.round(3), "name": names.reindex(top.index)}).to_string())


if __name__ == "__main__":
    main()
