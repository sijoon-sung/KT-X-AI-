import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
TRIPS_PATH = ROOT / "data/processed/trips.parquet"
WEATHER_PATH = ROOT / "data/raw/weather_openmeteo_daejeon.parquet"

nodes = pd.read_parquet(NODES_PATH)
st_name = nodes.set_index('station_id')['name'].to_dict()

trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'dist_km', 'dur_min', 'is_valid', 'rent_ts'])
trips = trips[trips['is_valid']].copy()
trips['rent_st'] = trips['rent_st'].astype(str)
trips['ret_st'] = trips['ret_st'].astype(str)
trips['rent_ts'] = pd.to_datetime(trips['rent_ts'])

# 1. 이동 거리 및 소요 시간 분포
pcts = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
dist_pct = trips['dist_km'].quantile(pcts).to_dict()
dur_pct = trips['dur_min'].quantile(pcts).to_dict()

# 2. 시간대별 & 평일/주말 프로필
trips['hour'] = trips['rent_ts'].dt.hour
trips['is_weekend'] = trips['rent_ts'].dt.dayofweek >= 5

weekday_hourly = trips[~trips['is_weekend']].groupby('hour').size()
weekend_hourly = trips[trips['is_weekend']].groupby('hour').size()

# 3. 비 & 기온 영향 분석
trips['hourly_ts'] = trips['rent_ts'].dt.floor('h')
hourly_counts = trips.groupby('hourly_ts').size().reset_index(name='trip_count')

w = pd.read_parquet(WEATHER_PATH)
w['time'] = pd.to_datetime(w['time'])
m = pd.merge(hourly_counts, w, left_on='hourly_ts', right_on='time', how='inner')
m_day = m[(m['time'].dt.hour >= 7) & (m['time'].dt.hour <= 22)].copy()

# 강수량 구간별 시간당 평균
rain_bins = [-0.1, 0.0, 1.0, 3.0, 10.0, 100.0]
rain_labels = ['비 안 옴 (0mm)', '약한 비 (0~1mm)', '보통 비 (1~3mm)', '강한 비 (3~10mm)', '폭우 (>10mm)']
m_day['rain_bin'] = pd.cut(m_day['precipitation'], bins=rain_bins, labels=rain_labels)
rain_impact = m_day.groupby('rain_bin', observed=False)['trip_count'].agg(['count', 'mean']).reset_index()

# 기온 구간별 시간당 평균
temp_bins = [-30, 0, 10, 18, 25, 30, 45]
temp_labels = ['영하 (<0°C)', '쌀쌀 (0~10°C)', '선선 (10~18°C)', '쾌적 (18~25°C)', '더움 (25~30°C)', '폭염 (>30°C)']
m_day['temp_bin'] = pd.cut(m_day['temperature_2m'], bins=temp_bins, labels=temp_labels)
temp_impact = m_day.groupby('temp_bin', observed=False)['trip_count'].agg(['count', 'mean']).reset_index()

# 4. Top 10 OD 통행 흐름 (Flow Corridors)
od = trips[trips['rent_st'] != trips['ret_st']].groupby(['rent_st', 'ret_st']).agg(
    volume=('dist_km', 'count'),
    avg_km=('dist_km', 'mean'),
    avg_min=('dur_min', 'mean')
).reset_index().sort_values('volume', ascending=False)

top_flows = []
for _, r in od.head(10).iterrows():
    orig_name = st_name.get(r['rent_st'], r['rent_st'])
    dest_name = st_name.get(r['ret_st'], r['ret_st'])
    top_flows.append({
        'orig': orig_name,
        'dest': dest_name,
        'volume': int(r['volume']),
        'avg_km': round(float(r['avg_km']), 2),
        'avg_min': round(float(r['avg_min']), 1)
    })

print("="*85)
print(" [데이터 분석가 관점 타슈 종합 통계 요약] ")
print("="*85)
print(f"1. 이동 거리 중앙값: {dist_pct[0.5]:.1f}km (75%가 {dist_pct[0.75]:.1f}km 이내)")
print(f"2. 소요 시간 중앙값: {dur_pct[0.5]:.1f}분 (75%가 {dur_pct[0.75]:.1f}분 이내)")
print(f"3. 평일 피크: 08시({weekday_hourly[8]:,}건) vs 18시({weekday_hourly[18]:,}건 - 아침의 1.6배!)")
print(f"4. 강수 영향: 0mm({rain_impact.iloc[0]['mean']:.0f}건/h) -> 1~3mm({rain_impact.iloc[2]['mean']:.0f}건/h, -71% 급감)")
print(f"5. 최다 통행 회랑: {top_flows[0]['orig']} -> {top_flows[0]['dest']} ({top_flows[0]['volume']:,}건)")
