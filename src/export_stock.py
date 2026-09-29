"""VM2 가 모으고 있는 재고 스냅숏을 바로 쓸 수 있는 파일로 만든다.

입력  data/raw/stock_vm2/data/stock_YYYYMMDD.csv.gz   (ts, id, parking_count)  10분 간격
      data/raw/stock_vm2/meta/stations_YYYYMMDD.json.gz (그날 첫 호출의 원본 전체)
      data/raw/seoul_vm2/stock_YYYYMMDD.csv.gz          (서울 따릉이, 같은 형식)

산출  exports/tashu_stock_10min.csv.gz    ts, station_id, stock, capacity, name, lat, lon
      exports/tashu_station_profile.csv   대여소별 분포: 평균·중앙·최소·최대 재고, 빈 비율, 가득 비율, 회전 추정
      exports/tashu_station_hour_profile.csv  대여소 × 시각(0~23) 평균 재고와 빈 비율
      exports/tashu_stock_hourly.parquet  ts(1시간), station_id, node, stock_start/min/mean, empty_frac …
      exports/seoul_station_profile.csv   서울 따릉이 대여소별 같은 요약 (있을 때만)
      exports/stock_export_report.json

실행:  python src/export_stock.py
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASHU = ROOT / "data" / "raw" / "stock_vm2"
SEOUL = ROOT / "data" / "raw" / "seoul_vm2"
EXP = ROOT / "exports"


def load_stock(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("stock_*.csv.gz"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f, dtype={"id": str}) for f in files], ignore_index=True)
    # 타슈: ts,id,parking_count / 서울: ts,sid,parking,rack
    df = df.rename(columns={"id": "station_id", "sid": "station_id", "parking_count": "stock", "parking": "stock", "rack": "capacity"})
    df["ts"] = pd.to_datetime(df["ts"])
    df["stock"] = pd.to_numeric(df["stock"], errors="coerce")
    return df.dropna(subset=["stock"])


def station_master() -> pd.DataFrame:
    """meta 의 원본 응답에서 이름·좌표·거치대 수를 뽑는다 (가장 최근 파일 기준)."""
    metas = sorted((TASHU / "meta").glob("stations_*.json.gz"))
    if not metas:
        return pd.DataFrame()
    import re

    with gzip.open(metas[-1], "rt", encoding="utf-8") as f:
        js = json.load(f)
    rows = []
    for x in js["results"]:
        m = re.search(r"/\s*(\d+)\s*$", x.get("name_cn") or "")
        rows.append({"station_id": x["id"], "name": x.get("name"), "lat": float(x["x_pos"]), "lon": float(x["y_pos"]),
                     "capacity": int(m.group(1)) if m else np.nan, "address": x.get("address")})
    return pd.DataFrame(rows)


def profile(df: pd.DataFrame, cap: pd.DataFrame | None = None) -> pd.DataFrame:
    g = df.sort_values("ts").groupby("station_id")["stock"]
    prof = pd.DataFrame({
        "n_snapshots": g.size(),
        "stock_mean": g.mean().round(2),
        "stock_median": g.median(),
        "stock_min": g.min(),
        "stock_max": g.max(),
        "stock_sd": g.std().round(2),
        "empty_frac": g.apply(lambda s: (s == 0).mean()).round(3),
    })
    # 재고가 바뀐 총량 = 그 대여소의 대여·반납 활동 크기 (두 스냅숏 사이 변화의 절댓값 합)
    ch = df.sort_values("ts").groupby("station_id")["stock"].apply(lambda s: s.diff().abs().sum())
    hours = (df["ts"].max() - df["ts"].min()) / pd.Timedelta(hours=1)
    prof["turnover_per_hour"] = (ch / max(hours, 1e-9)).round(2)
    if cap is not None and not cap.empty:
        prof = prof.join(cap.set_index("station_id")[["name", "lat", "lon", "capacity"]])
        prof["full_frac"] = np.nan
        ok = prof["capacity"].notna()
        base = df.drop(columns=["capacity"]) if "capacity" in df else df
        full = base.merge(cap[["station_id", "capacity"]], on="station_id", how="left")
        full = full.dropna(subset=["capacity"]).groupby("station_id")[["stock", "capacity"]].apply(lambda q: (q.stock >= q.capacity).mean())
        prof.loc[ok, "full_frac"] = full.reindex(prof.index[ok]).round(3)
    return prof.reset_index()


def main():
    EXP.mkdir(parents=True, exist_ok=True)
    rep = {}
    df = load_stock(TASHU / "data")
    if df.empty:
        raise SystemExit(f"재고 파일이 없습니다: {TASHU/'data'}")
    cap = station_master()

    out = df.merge(cap, on="station_id", how="left")
    out = out[["ts", "station_id", "stock", "capacity", "name", "lat", "lon"]].sort_values(["ts", "station_id"])
    out.to_csv(EXP / "tashu_stock_10min.csv.gz", index=False, compression="gzip")

    prof = profile(df, cap)
    prof.sort_values("stock_mean", ascending=False).to_csv(EXP / "tashu_station_profile.csv", index=False, encoding="utf-8-sig")

    hp = df.assign(hour=df.ts.dt.hour).groupby(["station_id", "hour"])["stock"].agg(
        stock_mean="mean", empty_frac=lambda s: (s == 0).mean(), n="size").reset_index()
    hp["stock_mean"] = hp["stock_mean"].round(2)
    hp["empty_frac"] = hp["empty_frac"].round(3)
    hp = hp.merge(cap[["station_id", "name"]], on="station_id", how="left")
    hp.to_csv(EXP / "tashu_station_hour_profile.csv", index=False, encoding="utf-8-sig")

    hourly = df.assign(ts=df.ts.dt.floor("h")).sort_values("ts").groupby(["ts", "station_id"]).agg(
        stock_start=("stock", "first"), stock_min=("stock", "min"), stock_mean=("stock", "mean"),
        empty_frac=("stock", lambda s: (s == 0).mean()), n_snapshots=("stock", "size")).reset_index()
    nodes_p = ROOT / "data" / "processed" / "v2" / "nodes.parquet"
    if nodes_p.exists() and not cap.empty:
        nodes = pd.read_parquet(nodes_p)
        m = nodes.merge(cap[["station_id", "lat", "lon"]], on="station_id", how="inner", suffixes=("_node", "_api"))
        d = np.hypot((m.lat_node - m.lat_api) * 111000, (m.lon_node - m.lon_api) * 90000)
        link = m.assign(dist_m=d).sort_values("dist_m").groupby("station_id", as_index=False).first()
        link = link.loc[link.dist_m <= 200, ["station_id", "node"]]
        hourly = hourly.merge(link, on="station_id", how="left")
        rep["linked_to_v2_nodes"] = int(hourly["node"].notna().groupby(hourly.station_id).first().sum())
    hourly.to_parquet(EXP / "tashu_stock_hourly.parquet", index=False)

    rep.update({
        "tashu_first_ts": str(df.ts.min()), "tashu_last_ts": str(df.ts.max()),
        "tashu_snapshots": int(df.ts.nunique()), "tashu_stations": int(df.station_id.nunique()),
        "tashu_rows": int(len(df)),
        "tashu_interval_min_median": float(pd.Series(sorted(df.ts.unique())).diff().median() / pd.Timedelta(minutes=1)),
        "tashu_empty_share": round(float((df.stock == 0).mean()), 3),
        "tashu_bikes_mean": round(float(df.groupby("ts").stock.sum().mean()), 1),
        "tashu_bikes_min": int(df.groupby("ts").stock.sum().min()),
        "tashu_bikes_max": int(df.groupby("ts").stock.sum().max()),
    })

    s = load_stock(SEOUL)
    if not s.empty:
        scap = s.groupby("station_id").capacity.last().reset_index() if "capacity" in s else None
        profile(s, scap.assign(name=None, lat=np.nan, lon=np.nan) if scap is not None else None).sort_values("stock_mean", ascending=False).to_csv(EXP / "seoul_station_profile.csv", index=False, encoding="utf-8-sig")
        rep.update({"seoul_first_ts": str(s.ts.min()), "seoul_last_ts": str(s.ts.max()),
                    "seoul_snapshots": int(s.ts.nunique()), "seoul_stations": int(s.station_id.nunique()),
                    "seoul_rows": int(len(s)), "seoul_empty_share": round(float((s.stock == 0).mean()), 3),
                    "seoul_bikes_mean": round(float(s.groupby("ts").stock.sum().mean()), 1)})

    (EXP / "stock_export_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("\n한 번 찍을 때의 전체 자전거 수와 빈 대여소 수 (타슈, 최근 12시간)")
    snap = df.groupby("ts").agg(bikes=("stock", "sum"), empty=("stock", lambda s: int((s == 0).sum())))
    snap["hour"] = snap.index.floor("h")
    print(snap.groupby("hour")[["bikes", "empty"]].mean().round(0).tail(12).to_string())


if __name__ == "__main__":
    main()
