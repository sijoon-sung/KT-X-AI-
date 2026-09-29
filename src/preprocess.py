"""1단계: 원본 대여이력 CSV(월별, cp949/utf-8 혼재) -> 정리된 parquet.

산출물 (data/processed/)
  trips.parquet         건별 기록 (필요 열만, 시각 파싱, 이상치 플래그)
  stations.parquet      대여소 마스터 (좌표·구·동은 기록에서, 거치대 용량은 Open API 스냅샷에서)
  counts_15min.parquet  대여소×15분 단위 대여(rent)/반납(ret) 건수 (0인 칸은 저장 안 함)
  missing_days.json     시스템 장애 등으로 기록이 비정상적으로 적은 날짜 목록

실행:  python src/preprocess.py
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

COLS = {
    "자전거번호": "bike_id",
    "대여일시": "rent_ts",
    "대여_대여소ID": "rent_st",
    "대여_대여소명": "rent_name",
    "대여_X좌표": "rent_lat",
    "대여_Y좌표": "rent_lon",
    "대여_구": "rent_gu",
    "대여_동": "rent_dong",
    "반납일시": "ret_ts",
    "반납_대여소ID": "ret_st",
    "반납_대여소명": "ret_name",
    "반납_X좌표": "ret_lat",
    "반납_Y좌표": "ret_lon",
    "반납_구": "ret_gu",
    "반납_동": "ret_dong",
    "이용시간(분)": "dur_min",
    "이용거리(km)": "dist_km",
}


def detect_encoding(path: Path) -> str:
    """월별 파일이 utf-8(BOM) 과 cp949 가 섞여 있다."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp949"


def load_month(path: Path) -> pd.DataFrame:
    enc = detect_encoding(path)
    header = pd.read_csv(path, encoding=enc, nrows=0).columns
    missing = [c for c in COLS if c not in header]
    if missing:
        raise ValueError(f"{path.name}: 열 이름 다름 {missing} / 실제 {list(header)}")
    df = pd.read_csv(path, encoding=enc, usecols=list(COLS), dtype=str)
    df = df.rename(columns=COLS)
    df["rent_ts"] = pd.to_datetime(df["rent_ts"], errors="coerce")
    df["ret_ts"] = pd.to_datetime(df["ret_ts"], errors="coerce")
    for c in ("rent_lat", "rent_lon", "ret_lat", "ret_lon", "dist_km"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["dur_min"] = pd.to_numeric(df["dur_min"], errors="coerce").astype("float32")
    df["src_file"] = path.stem
    return df


def build_station_table(trips: pd.DataFrame) -> pd.DataFrame:
    """대여/반납 양쪽 기록을 합쳐 대여소별 최빈 이름·좌표·구·동을 뽑는다."""
    a = trips[["rent_st", "rent_name", "rent_lat", "rent_lon", "rent_gu", "rent_dong"]]
    a.columns = ["station_id", "name", "lat", "lon", "gu", "dong"]
    b = trips[["ret_st", "ret_name", "ret_lat", "ret_lon", "ret_gu", "ret_dong"]]
    b.columns = a.columns
    both = pd.concat([a, b], ignore_index=True)
    st = (
        both.groupby("station_id")
        .agg(
            name=("name", lambda s: s.mode().iat[0]),
            lat=("lat", "median"),
            lon=("lon", "median"),
            gu=("gu", lambda s: s.mode().iat[0]),
            dong=("dong", lambda s: s.mode().iat[0]),
            n_events=("name", "size"),
        )
        .reset_index()
    )

    # Open API 스냅샷: name_cn 끝의 "/ 10" 이 거치대 수, parking_count 는 조회 시점 잔여 자전거
    snap = RAW / "stations_api_snapshot.json"
    if snap.exists():
        js = json.loads(snap.read_text(encoding="utf-8"))
        rows = []
        for x in js["results"]:
            m = re.search(r"/\s*(\d+)\s*$", x.get("name_cn") or "")
            rows.append(
                {
                    "station_id": x["id"],
                    "capacity": int(m.group(1)) if m else np.nan,
                    "api_name": x["name"],
                    "api_parking_now": x.get("parking_count"),
                    "in_api": True,
                }
            )
        api = pd.DataFrame(rows)
        st = st.merge(api, on="station_id", how="outer")
        st["in_api"] = st["in_api"].fillna(False).astype(bool)
        st["name"] = st["name"].fillna(st["api_name"])
        st["snapshot_at"] = js.get("fetched_at")
    return st.sort_values("station_id").reset_index(drop=True)


def find_missing_days(trips: pd.DataFrame) -> list[str]:
    """기록이 '실제로 비는' 날만 결측일로 본다.
      - 하루 기록이 0 건 (2025-02-27~03-03 시스템 장애, 2025-12 원본 파일 오류)
      - 기록이 있는 시간대 수가 그 달 중앙값보다 3시간 이상 적음 (장애 시작/복구일, 첫날 05:00 시작 등)
    비 오는 날처럼 이용이 적기만 한 날은 정상으로 둔다 (겨울철 05~24시 운영도 정상)."""
    day = trips["rent_ts"].dt.floor("D")
    daily = trips.groupby(day).size()
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full, fill_value=0)
    hours = trips.groupby(day)["rent_ts"].apply(lambda s: s.dt.hour.nunique()).reindex(full, fill_value=0)
    month_hours = hours[hours > 0].groupby(hours[hours > 0].index.to_period("M")).median()
    ref = hours.index.to_period("M").map(month_hours).to_numpy()
    bad = (daily == 0) | (hours < ref - 3)
    return [d.strftime("%Y-%m-%d") for d in daily.index[bad]]


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW.glob("2*.csv"))
    if not files:
        raise SystemExit(f"raw csv 없음: {RAW}")
    parts = []
    for f in files:
        df = load_month(f)
        print(f"{f.name}: {len(df):,} rows, {df['rent_ts'].min()} ~ {df['rent_ts'].max()}")
        # 파일 이름의 연월과 내용의 연월이 다르면 (예: 202512.csv 안에 2025-01 기록) 버린다
        ym = df["rent_ts"].dt.strftime("%Y%m").mode().iat[0]
        if ym != f.stem:
            print(f"  !! {f.name} 내용은 {ym} 기록 → 제외 (원본 배포 오류)")
            continue
        parts.append(df)
    trips = pd.concat(parts, ignore_index=True)
    del parts

    n0 = len(trips)
    # 기본 정합성 플래그 (버리지 않고 표시만; 집계 시엔 제외)
    bad_ts = trips["rent_ts"].isna() | trips["ret_ts"].isna() | (trips["ret_ts"] < trips["rent_ts"])
    bad_id = ~trips["rent_st"].str.match(r"^ST\d+$", na=False) | ~trips["ret_st"].str.match(r"^ST\d+$", na=False)
    too_long = trips["dur_min"] > 24 * 60
    trips["is_valid"] = ~(bad_ts | bad_id | too_long)
    print(f"total {n0:,} | bad_ts {bad_ts.sum():,} | bad_id {bad_id.sum():,} | >24h {too_long.sum():,}")

    # 중복 제거 (같은 자전거·같은 대여시각)
    dup = trips.duplicated(subset=["bike_id", "rent_ts", "rent_st", "ret_ts"], keep="first")
    print(f"duplicates {dup.sum():,}")
    trips = trips[~dup].reset_index(drop=True)

    stations = build_station_table(trips)
    stations.to_parquet(OUT / "stations.parquet", index=False)
    print(f"stations: {len(stations):,} (capacity known: {stations['capacity'].notna().sum():,})")

    missing = find_missing_days(trips)
    (OUT / "missing_days.json").write_text(json.dumps(missing, indent=1), encoding="utf-8")
    print("missing days:", missing)

    for c in ("rent_st", "ret_st", "rent_gu", "ret_gu", "rent_dong", "ret_dong", "src_file"):
        trips[c] = trips[c].astype("category")
    trips.to_parquet(OUT / "trips.parquet", index=False)
    print("trips.parquet written")

    # 15분 집계 (대여/반납 따로 → outer join)
    v = trips[trips["is_valid"]]
    rent = (
        v.assign(ts=v["rent_ts"].dt.floor("15min"))
        .groupby(["rent_st", "ts"], observed=True)
        .size()
        .rename("rent")
    )
    rent.index = rent.index.set_names(["station_id", "ts"])
    ret = (
        v.assign(ts=v["ret_ts"].dt.floor("15min"))
        .groupby(["ret_st", "ts"], observed=True)
        .size()
        .rename("ret")
    )
    ret.index = ret.index.set_names(["station_id", "ts"])
    counts = pd.concat([rent, ret], axis=1).fillna(0).astype("int16").reset_index()
    counts["station_id"] = counts["station_id"].astype(str)
    counts = counts.sort_values(["ts", "station_id"]).reset_index(drop=True)
    counts.to_parquet(OUT / "counts_15min.parquet", index=False)
    print(f"counts_15min: {len(counts):,} rows, {counts['ts'].min()} ~ {counts['ts'].max()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(p.parse_args())
