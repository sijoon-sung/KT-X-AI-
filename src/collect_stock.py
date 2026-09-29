"""타슈 Open API 를 주기적으로 찍어 대여소 재고 이력을 만든다.

대여이력 원자료에는 "그 시각 대여소에 몇 대가 있었는지"가 없다. 그래서 빈 대여소 때문에
못 빌린 수요(잘린 수요)와 실제 결품 시각을 알 수 없다. 이 스크립트가 그 빈칸을 채운다.

동작
  - https://bikeapp.tashu.or.kr:50041/v1/openapi/station 를 한 번 호출 (헤더 api-token)
  - 대여소별 잔여 자전거 수를 하루 단위 CSV(gzip) 에 한 줄씩 덧붙임
      data/stock/YYYY-MM-DD.csv.gz : ts_kst, station_id, parking_count
  - 대여소 이름·좌표·거치대 수가 처음 보이거나 바뀌면 data/stations_log.csv 에 기록
  - 실패해도 조용히 넘어가고 log/collect.log 에 남긴다 (cron 이 다음 주기에 다시 시도)

실행:  python3 collect_stock.py            (한 번 수집)
       python3 collect_stock.py --check    (수집 상태 요약만 출력)
cron:  */5 * * * * cd ~/tashu_collect && python3 collect_stock.py >> log/cron.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

URL = "https://bikeapp.tashu.or.kr:50041/v1/openapi/station"
TOKEN = os.environ.get("TASHU_API_TOKEN", "5kapc2kii77b1q3c")
KST = timezone(timedelta(hours=9))
BASE = Path(__file__).resolve().parent
STOCK = BASE / "data" / "stock"
STATIONS_LOG = BASE / "data" / "stations_log.csv"
LOG = BASE / "log" / "collect.log"
RETRIES = 3


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(KST):%Y-%m-%d %H:%M:%S} {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def fetch() -> list[dict]:
    last = ""
    for i in range(RETRIES):
        try:
            r = requests.get(URL, headers={"api-token": TOKEN}, timeout=30)
            if r.status_code == 200:
                js = r.json()
                items = js["results"] if isinstance(js, dict) else js
                if not items:
                    raise ValueError("빈 응답")
                return items
            last = f"HTTP {r.status_code}"
            if r.status_code in (401, 403, 429):  # 인증·호출한도 문제는 재시도해도 소용없음
                break
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"수집 실패: {last}")


def capacity_of(item: dict):
    m = re.search(r"/\s*(\d+)\s*$", item.get("name_cn") or "")
    return int(m.group(1)) if m else ""


def append_stock(ts: datetime, items: list[dict]) -> Path:
    STOCK.mkdir(parents=True, exist_ok=True)
    path = STOCK / f"{ts:%Y-%m-%d}.csv.gz"
    new = not path.exists()
    with gzip.open(path, "at", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ts_kst", "station_id", "parking_count"])
        stamp = f"{ts:%Y-%m-%d %H:%M:%S}"
        for x in items:
            w.writerow([stamp, x["id"], x.get("parking_count", "")])
    return path


def update_station_log(ts: datetime, items: list[dict]) -> int:
    """이름·좌표·거치대 수가 바뀔 때만 한 줄 남긴다 (대여소 신설·이전 추적용)."""
    seen: dict[str, tuple] = {}
    if STATIONS_LOG.exists():
        with open(STATIONS_LOG, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                seen[row["station_id"]] = (row["name"], row["lat"], row["lon"], row["capacity"])
    rows = []
    for x in items:
        key = (x.get("name", ""), x.get("x_pos", ""), x.get("y_pos", ""), str(capacity_of(x)))
        if seen.get(x["id"]) != key:
            rows.append([f"{ts:%Y-%m-%d %H:%M:%S}", x["id"], *key, x.get("address", "")])
            seen[x["id"]] = key
    if rows:
        STATIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
        new = not STATIONS_LOG.exists()
        with open(STATIONS_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ts_kst", "station_id", "name", "lat", "lon", "capacity", "address"])
            w.writerows(rows)
    return len(rows)


def check() -> None:
    files = sorted(STOCK.glob("*.csv.gz"))
    if not files:
        print("아직 모은 파일이 없습니다.")
        return
    total = 0
    print(f"{'날짜':<12}{'행':>10}{'찍은 횟수':>10}{'크기(MB)':>10}")
    for p in files:
        stamps = set()
        n = 0
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for i, row in enumerate(csv.reader(f)):
                if i == 0:
                    continue
                n += 1
                stamps.add(row[0])
        total += n
        print(f"{p.stem:<12}{n:>10,}{len(stamps):>10,}{p.stat().st_size/1e6:>10.1f}")
    print(f"합계 {total:,} 행, 파일 {len(files)}개")
    if STATIONS_LOG.exists():
        with open(STATIONS_LOG, encoding="utf-8") as f:
            print(f"대여소 변경 기록 {sum(1 for _ in f) - 1}줄")
    if LOG.exists():
        tail = LOG.read_text(encoding="utf-8").strip().splitlines()[-3:]
        print("최근 기록:", *tail, sep="\n  ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true", help="모은 상태만 확인")
    args = p.parse_args()
    if args.check:
        check()
        return 0
    ts = datetime.now(KST)
    try:
        items = fetch()
    except Exception as e:  # noqa: BLE001
        log(f"ERROR {e}")
        return 1
    path = append_stock(ts, items)
    changed = update_station_log(ts, items)
    bikes = sum(int(x.get("parking_count") or 0) for x in items)
    empty = sum(1 for x in items if int(x.get("parking_count") or 0) == 0)
    log(f"OK 대여소 {len(items)} 자전거 {bikes} 빈곳 {empty} -> {path.name}" + (f" 대여소변경 {changed}" if changed else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
