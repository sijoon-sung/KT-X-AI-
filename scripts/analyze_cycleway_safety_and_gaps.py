import sys
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
TRIPS_PATH = ROOT / "data/processed/trips.parquet"
CYCLEWAY_PATH = ROOT / "data/raw/osm_cycleway.json"

nodes = pd.read_parquet(NODES_PATH)
st_name = nodes.set_index('station_id')['name'].to_dict()
st_lat = nodes.set_index('station_id')['lat'].to_dict()
st_lon = nodes.set_index('station_id')['lon'].to_dict()

# 1. OSM 자전거 도로 로드
with open(CYCLEWAY_PATH, 'r', encoding='utf-8') as f:
    osm_cw = json.load(f)

cw_points = []
for el in osm_cw.get('elements', []):
    for pt in el.get('geometry', []):
        cw_points.append((pt['lat'], pt['lon']))

cw_arr = np.array(cw_points)

def min_dist_to_cycleway(lat, lon):
    # 유클리드 거리 약산 (1도 ≈ 111km)
    d_lat = (cw_arr[:, 0] - lat) * 111.0
    d_lon = (cw_arr[:, 1] - lon) * 88.0
    d_km = np.sqrt(d_lat**2 + d_lon**2)
    return np.min(d_km)

# 2. 통행 이력 상위 50개 통행 회랑(Corridor) 추출
trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'dist_km', 'dur_min', 'is_valid'])
trips['rent_st'] = trips['rent_st'].astype(str)
trips['ret_st'] = trips['ret_st'].astype(str)
trips = trips[trips['is_valid'] & (trips['rent_st'] != trips['ret_st'])].copy()

od = trips.groupby(['rent_st', 'ret_st']).agg(
    trips=('dist_km', 'count'),
    avg_dist=('dist_km', 'mean'),
    avg_dur=('dur_min', 'mean')
).reset_index().sort_values('trips', ascending=False)

# 상위 회랑 중 자전거 도로와의 최소 이격 거리 및 단절 위험도 평가
top_corridors = od.head(60).copy()
records = []

for _, r in top_corridors.iterrows():
    s_orig = r['rent_st']
    s_dest = r['ret_st']
    
    lat1, lon1 = st_lat.get(s_orig, 0), st_lon.get(s_orig, 0)
    lat2, lon2 = st_lat.get(s_dest, 0), st_lon.get(s_dest, 0)
    if lat1 == 0 or lat2 == 0:
        continue
        
    mid_lat = (lat1 + lat2) / 2.0
    mid_lon = (lon1 + lon2) / 2.0
    
    # 출발지, 중간 경로, 도착지의 자전거 도로 이격 거리
    d_orig = min_dist_to_cycleway(lat1, lon1) * 1000 # m
    d_mid = min_dist_to_cycleway(mid_lat, mid_lon) * 1000 # m
    d_dest = min_dist_to_cycleway(lat2, lon2) * 1000 # m
    
    # 단절 위험 지수: 통행량 * (중간지점 자전거도로 이격거리 / 100)
    # 중간지점(차도 구간)이 자전거 도로에서 150m 이상 떨어져 있으면 단절/차도 혼재 구간!
    is_gap = (d_mid >= 120) or (d_orig >= 150) or (d_dest >= 150)
    risk_score = r['trips'] * (d_mid / 100.0)
    
    records.append({
        'orig_st': s_orig,
        'dest_st': s_dest,
        'orig_name': st_name.get(s_orig, s_orig),
        'dest_name': st_name.get(s_dest, s_dest),
        'trips': r['trips'],
        'avg_km': round(r['avg_dist'], 2),
        'mid_cycleway_dist_m': round(d_mid, 1),
        'is_gap': is_gap,
        'risk_score': round(risk_score, 1)
    })

df_eval = pd.DataFrame(records)
# 자전거 전용도로 단절 위험 구간 Top 10
df_risk_top = df_eval[df_eval['is_gap'] == True].sort_values('trips', ascending=False).head(10)

print("="*85)
print(" [통행량 폭발 vs 자전거도로 단절/부족 '최고 위험 구간' Top 10 (우선 투자 입지)] ")
print("="*85)
print(f"| 순위 | 출발 대여소 | 도착 대여소 | 20개월 통행량 | 주행거리 | 자전거도로 이격 | 위험 판정 및 개선 필요성 |")
print(f"| :---: | :--- | :--- | :---: | :---: | :---: | :--- |")

for idx, (_, r) in enumerate(df_risk_top.iterrows(), 1):
    print(f"| {idx:2d}위 | {r['orig_name'][:14]:14s} | {r['dest_name'][:14]:14s} | **{r['trips']:,} 건** | {r['avg_km']} km | **{r['mid_cycleway_dist_m']:.0f} m 단절** | 차도 혼재 주행 (전용도로 신설 1순위) |")
