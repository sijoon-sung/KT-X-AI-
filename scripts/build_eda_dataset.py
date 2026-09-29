import sys
import json
from pathlib import Path
import ctypes.wintypes
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

# 1. 거리 및 시간 통계
dist_q = trips['dist_km'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()
dur_q = trips['dur_min'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_dict()

# 거리 구간별 비율
d_bins = [-0.1, 1.0, 2.0, 4.0, 100.0]
d_labels = ['1km 이내 (초단거리)', '1~2km (생활권 표준)', '2~4km (중거리 환승)', '4km 초과 (장거리/레저)']
dist_shares = pd.cut(trips['dist_km'], bins=d_bins, labels=d_labels).value_counts(normalize=True).to_dict()

# 2. 시간대별 통행량 (평일 vs 주말)
trips['hour'] = trips['rent_ts'].dt.hour
trips['is_weekend'] = trips['rent_ts'].dt.dayofweek >= 5

wk_hourly = trips[~trips['is_weekend']].groupby('hour').size().to_dict()
we_hourly = trips[trips['is_weekend']].groupby('hour').size().to_dict()

# 3. 날씨 영향
trips['hourly_ts'] = trips['rent_ts'].dt.floor('h')
hourly_counts = trips.groupby('hourly_ts').size().reset_index(name='trip_count')

w = pd.read_parquet(WEATHER_PATH)
w['time'] = pd.to_datetime(w['time'])
m = pd.merge(hourly_counts, w, left_on='hourly_ts', right_on='time', how='inner')
m_day = m[(m['time'].dt.hour >= 7) & (m['time'].dt.hour <= 22)].copy()

# 강수 구간
m_day['rain_bin'] = pd.cut(m_day['precipitation'], 
                           bins=[-0.1, 0.0, 1.0, 3.0, 10.0, 100.0], 
                           labels=['비 안 옴 (0mm)', '약한 비 (0~1mm)', '보통 비 (1~3mm)', '강한 비 (3~10mm)', '폭우 (>10mm)'])
rain_stat = m_day.groupby('rain_bin', observed=False)['trip_count'].agg(['count', 'mean']).to_dict('index')

# 기온 구간
m_day['temp_bin'] = pd.cut(m_day['temperature_2m'], 
                           bins=[-30, 0, 10, 18, 25, 30, 45], 
                           labels=['영하 (<0°C)', '쌀쌀 (0~10°C)', '선선 (10~18°C)', '쾌적 (18~25°C)', '더움 (25~30°C)', '폭염 (>30°C)'])
temp_stat = m_day.groupby('temp_bin', observed=False)['trip_count'].agg(['count', 'mean']).to_dict('index')

# 4. Top 10 OD 통행 흐름
od = trips[trips['rent_st'] != trips['ret_st']].groupby(['rent_st', 'ret_st']).agg(
    volume=('dist_km', 'count'),
    avg_km=('dist_km', 'mean'),
    avg_min=('dur_min', 'mean')
).reset_index().sort_values('volume', ascending=False)

top_flows = []
for _, r in od.head(10).iterrows():
    top_flows.append({
        'orig': st_name.get(r['rent_st'], r['rent_st']),
        'dest': st_name.get(r['ret_st'], r['ret_st']),
        'volume': int(r['volume']),
        'avg_km': round(float(r['avg_km']), 2),
        'avg_min': round(float(r['avg_min']), 1)
    })

# JSON 저장
eda_summary = {
    'distance_quantiles_km': {f"p{int(k*100)}": round(v, 2) for k, v in dist_q.items()},
    'duration_quantiles_min': {f"p{int(k*100)}": round(v, 1) for k, v in dur_q.items()},
    'distance_shares': {k: round(v*100, 1) for k, v in dist_shares.items()},
    'weekday_hourly': wk_hourly,
    'weekend_hourly': we_hourly,
    'rain_impact': {k: round(v['mean'], 1) for k, v in rain_stat.items()},
    'temp_impact': {k: round(v['mean'], 1) for k, v in temp_stat.items()},
    'top_10_flows': top_flows
}

with open(ROOT / "outputs/eda_analysis_summary.json", 'w', encoding='utf-8') as f:
    json.dump(eda_summary, f, ensure_ascii=False, indent=2)

print("EDA json generated successfully.")
