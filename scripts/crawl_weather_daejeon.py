"""대전 기상 데이터(2024-08-01 ~ 2026-09-28 현재) 자동 크롤링 및 파팃 저장 스크립트."""
import sys
import json
import urllib.request
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "raw" / "weather_openmeteo_daejeon.parquet"
BACKUP_PATH = ROOT / "data" / "raw" / "weather_openmeteo_daejeon_backup.parquet"

LAT = 36.3504  # 대전시청 위도
LON = 127.3845 # 대전시청 경도

def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def main():
    print(f"=== [1] 대전 기상 데이터 크롤링 시작 (Open-Meteo API) ===")
    print(f"위경도: lat={LAT}, lon={LON} (대전 중심부)")

    # 1. Archive API (2024-08-01 ~ 2026-09-20)
    print("\n--- [1단계] 과거 기상 아카이브 수집 (2024-08-01 ~ 2026-09-20) ---")
    archive_url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={LAT}&longitude={LON}&"
        f"start_date=2024-08-01&end_date=2026-09-20&"
        f"hourly=precipitation,temperature_2m,wind_speed_10m,snowfall&"
        f"timezone=Asia%2FSeoul"
    )
    print(f"Request: {archive_url[:80]}...")
    archive_data = fetch_json(archive_url)
    h_arch = archive_data.get("hourly", {})
    df_arch = pd.DataFrame(h_arch)
    print(f"아카이브 수집 완료: {len(df_arch):,} 시간대 ({df_arch['time'].min()} ~ {df_arch['time'].max()})")

    # 2. Forecast API (최근 및 현재 2026-09-21 ~ 2026-09-28)
    print("\n--- [2단계] 최근 및 오늘 기상 수집 (2026-09-21 ~ 2026-09-28) ---")
    forecast_url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LAT}&longitude={LON}&"
        f"past_days=7&forecast_days=2&"
        f"hourly=precipitation,temperature_2m,wind_speed_10m,snowfall&"
        f"timezone=Asia%2FSeoul"
    )
    print(f"Request: {forecast_url[:80]}...")
    forecast_data = fetch_json(forecast_url)
    h_fore = forecast_data.get("hourly", {})
    df_fore = pd.DataFrame(h_fore)
    print(f"최근 데이터 수집 완료: {len(df_fore):,} 시간대 ({df_fore['time'].min()} ~ {df_fore['time'].max()})")

    # 3. 병합 및 중복 제거
    print("\n--- [3단계] 데이터 결합, 정렬 및 결측치 검증 ---")
    combined = pd.concat([df_arch, df_fore], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    # 2026-09-28 23:00 까지만 자르기
    combined = combined[combined["time"] <= "2026-09-28T23:00"].reset_index(drop=True)
    
    # 파생 변수
    combined["is_rain"] = (combined["precipitation"] > 0.0).astype(int)
    combined["is_heavy_rain"] = (combined["precipitation"] >= 5.0).astype(int)

    print(f"최종 병합 레코드: 총 {len(combined):,} 시간 ({combined['time'].min()} ~ {combined['time'].max()})")
    print(f"결측치 확인: {combined.isnull().sum().to_dict()}")

    # 4. 백업 및 저장
    if OUTPUT_PATH.exists():
        shutil.copy2(OUTPUT_PATH, BACKUP_PATH)
        print(f"기존 파일 백업 완료: {BACKUP_PATH.name}")

    combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"저장 성공: {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size:,} bytes)")

    # 5. 수집된 날씨 데이터 통계 요약
    print("\n" + "="*70)
    print(" [수집된 대전 날씨 데이터 통계 요약]")
    print("="*70)
    print(f"  - 기간: {combined['time'].min()[:10]} ~ {combined['time'].max()[:10]} (총 {len(combined):,} 시간)")
    print(f"  - 평균 기온: {combined['temperature_2m'].mean():.1f} °C (최저 {combined['temperature_2m'].min():.1f} °C ~ 최고 {combined['temperature_2m'].max():.1f} °C)")
    print(f"  - 강수 발생 시간대: {combined['is_rain'].sum():,} 시간 ({combined['is_rain'].mean()*100:.1f}%)")
    print(f"  - 최대 시간당 강수량: {combined['precipitation'].max():.1f} mm")
    print(f"  - 평균 풍속: {combined['wind_speed_10m'].mean():.1f} km/h")
    print(f"  - 적설 발생 시간대: {(combined['snowfall'] > 0).sum():,} 시간")

    # 최근 5시간 날씨 출력
    print("\n[오늘(2026-09-28) 최근 시간대 날씨]")
    today_records = combined[combined['time'].str.startswith("2026-09-28")].tail(6)
    for _, r in today_records.iterrows():
        print(f"  {r['time']} | 기온 {r['temperature_2m']:4.1f}°C | 강수 {r['precipitation']:3.1f}mm | 풍속 {r['wind_speed_10m']:4.1f}km/h")

if __name__ == "__main__":
    main()
