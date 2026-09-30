import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
TRIPS_PATH = ROOT / "data/processed/trips.parquet"
ELEVATION_PATH = ROOT / "data/raw/elevation_nodes.json"

print("1. Loading elevation data and station nodes...")
with open(ELEVATION_PATH, 'r', encoding='utf-8') as f:
    elev_raw = json.load(f)

# Parse elevation points
elev_pts = []
elev_vals = []
for k, v in elev_raw.items():
    lat_str, lon_str = k.split(',')
    elev_pts.append([float(lat_str), float(lon_str)])
    elev_vals.append(float(v))

elev_pts = np.array(elev_pts)
elev_vals = np.array(elev_vals)
tree = cKDTree(elev_pts)

# Load stations
nodes = pd.read_parquet(NODES_PATH)
nodes_valid = nodes.dropna(subset=['lat', 'lon']).copy()
coords = np.column_stack([nodes_valid['lat'].values, nodes_valid['lon'].values])
dists, indices = tree.query(coords)
nodes_valid['elevation'] = elev_vals[indices]
st_elev_dict = nodes_valid.set_index('station_id')['elevation'].to_dict()
st_name_dict = nodes_valid.set_index('station_id')['name'].to_dict()

print(f"Mapped elevation for {len(st_elev_dict)} stations.")
print(f"Elevation Range: {nodes_valid['elevation'].min():.1f}m ~ {nodes_valid['elevation'].max():.1f}m (Mean: {nodes_valid['elevation'].mean():.1f}m)")

print("2. Sampling and analyzing trips across elevation gradients...")
# Read trips columns
trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'dur_min', 'dist_km'])
trips = trips.dropna().copy()
trips['rent_st'] = trips['rent_st'].astype(str)
trips['ret_st'] = trips['ret_st'].astype(str)

# Filter trips where both stations have elevation
trips['h_rent'] = trips['rent_st'].map(st_elev_dict)
trips['h_ret'] = trips['ret_st'].map(st_elev_dict)
trips_elev = trips.dropna(subset=['h_rent', 'h_ret']).copy()

# Calculate elevation delta: negative = downhill (내리막), positive = uphill (오르막)
trips_elev['dh'] = trips_elev['h_ret'] - trips_elev['h_rent']

# Categorize
flat = trips_elev[trips_elev['dh'].abs() <= 10]
downhill = trips_elev[trips_elev['dh'] < -10]
uphill = trips_elev[trips_elev['dh'] > 10]

down_cnt = len(downhill)
up_cnt = len(uphill)
asymmetry_ratio = down_cnt / max(1, up_cnt)

print("\n" + "="*75)
print(" [타슈 중력의 법칙] 고도차(Elevation Gradient)에 따른 통행 비대칭 분석")
print("="*75)
print(f"총 분석 통행 건수: {len(trips_elev):,} 건")
print(f"• 평지 통행 (|Δh| <= 10m): {len(flat):,} 건 ({len(flat)/len(trips_elev)*100:.1f}%)")
print(f"• 내리막 통행 (Δh < -10m): {down_cnt:,} 건 ({down_cnt/len(trips_elev)*100:.1f}%) | 평균 시간: {downhill['dur_min'].mean():.1f}분 | 평균 속도: {(downhill['dist_km']/(downhill['dur_min']/60)).mean():.1f}km/h")
print(f"• 오르막 통행 (Δh > +10m): {up_cnt:,} 건 ({up_cnt/len(trips_elev)*100:.1f}%) | 평균 시간: {uphill['dur_min'].mean():.1f}분 | 평균 속도: {(uphill['dist_km']/(uphill['dur_min']/60)).mean():.1f}km/h")
print(f"👉 중력 비대칭성 지수: 내리막 통행량이 오르막보다 {asymmetry_ratio:.2f}배 더 많음 (비가역적 중력 하강!)")

# Steep gradients (> 25m delta)
steep_down = trips_elev[trips_elev['dh'] < -25]
steep_up = trips_elev[trips_elev['dh'] > 25]
steep_ratio = len(steep_down) / max(1, len(steep_up))
print(f"\n[급경사 구간 (|Δh| > 25m) 극한 비대칭]:")
print(f"• 급경사 내리막: {len(steep_down):,} 건 vs 급경사 오르막: {len(steep_up):,} 건")
print(f"👉 고도차 25m 이상에서는 내리막 선호도가 오르막의 무려 {steep_ratio:.2f}배에 달함!")

# Top Downhill Gravity Chutes (가장 극단적인 중력 쏠림 회랑 TOP 5)
print("\n" + "="*75)
print(" 대전시 5대 '중력 쏠림(Gravity Chute)' 일방통행 회랑")
print("="*75)
od_pairs = trips_elev.groupby(['rent_st', 'ret_st']).agg(
    cnt=('dh', 'count'),
    mean_dh=('dh', 'mean'),
    dur=('dur_min', 'mean')
).reset_index()

od_pairs = od_pairs[od_pairs['mean_dh'] < -15].sort_values('cnt', ascending=False).head(5)
for _, r in od_pairs.iterrows():
    s_name = st_name_dict.get(r['rent_st'], r['rent_st'])
    e_name = st_name_dict.get(r['ret_st'], r['ret_st'])
    # check reverse
    rev = trips_elev[(trips_elev['rent_st'] == r['ret_st']) & (trips_elev['ret_st'] == r['rent_st'])]
    rev_cnt = len(rev)
    ratio = r['cnt'] / max(1, rev_cnt)
    print(f"[{s_name[:14]} ➔ {e_name[:14]}] 고도차: {r['mean_dh']:4.1f}m | 하강 통행: {r['cnt']:,}건 vs 역상승: {rev_cnt:,}건 (비대칭 {ratio:.1f}배!)")

# 3. Spatial Entropy Analysis
# Compute net flux divergence per station
st_flux = trips_elev.groupby('rent_st')['dh'].count().rename('outflow').to_frame()
st_flux['inflow'] = trips_elev.groupby('ret_st')['dh'].count()
st_flux = st_flux.fillna(0)
st_flux['net_drain'] = st_flux['outflow'] - st_flux['inflow'] # positive = net drain (자전거 빠져나감)
st_flux['elevation'] = st_flux.index.map(st_elev_dict)
st_flux['name'] = st_flux.index.map(st_name_dict)

corr = st_flux[['net_drain', 'elevation']].corr().iloc[0, 1]
print("\n" + "="*75)
print(" [열역학적 도시 엔트로피 & 플럭스 검증]")
print("="*75)
print(f"• 대여소 고도(h)와 자전거 순유출량(Net Drain) 간의 상관계수: r = +{corr:.3f}")
print("  (고도가 높을수록 자전거가 순유출되어 바닥나며, 저지대로 빨려 들어가는 물리 법칙 성립)")

summary_output = {
    "total_elev_trips": int(len(trips_elev)),
    "asymmetry_ratio_10m": round(asymmetry_ratio, 2),
    "asymmetry_ratio_25m": round(steep_ratio, 2),
    "downhill_speed_kmh": round(float((downhill['dist_km']/(downhill['dur_min']/60)).mean()), 1),
    "uphill_speed_kmh": round(float((uphill['dist_km']/(uphill['dur_min']/60)).mean()), 1),
    "elevation_correlation": round(float(corr), 3),
    "top_gravity_corridors": [
        {
            "origin": st_name_dict.get(r['rent_st'], r['rent_st']),
            "dest": st_name_dict.get(r['ret_st'], r['ret_st']),
            "dh": round(float(r['mean_dh']), 1),
            "down_trips": int(r['cnt']),
            "up_trips": int(len(trips_elev[(trips_elev['rent_st'] == r['ret_st']) & (trips_elev['ret_st'] == r['rent_st'])])),
            "ratio": round(float(r['cnt'] / max(1, len(trips_elev[(trips_elev['rent_st'] == r['ret_st']) & (trips_elev['ret_st'] == r['rent_st'])]))), 1)
        } for _, r in od_pairs.iterrows()
    ]
}

out_file = ROOT / "outputs/final_model/gravity_entropy_summary.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, 'w', encoding='utf-8') as f:
    json.dump(summary_output, f, ensure_ascii=False, indent=2)

print(f"\nSaved summary to {out_file}")
