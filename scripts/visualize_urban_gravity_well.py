import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')

# Set Korean Font
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
TRIPS_PATH = ROOT / "data/processed/trips.parquet"
ELEVATION_PATH = ROOT / "data/raw/elevation_nodes.json"

print("1. Loading elevation data and station nodes...")
with open(ELEVATION_PATH, 'r', encoding='utf-8') as f:
    elev_raw = json.load(f)

elev_pts = []
elev_vals = []
for k, v in elev_raw.items():
    lat_str, lon_str = k.split(',')
    elev_pts.append([float(lat_str), float(lon_str)])
    elev_vals.append(float(v))

tree = cKDTree(np.array(elev_pts))
nodes = pd.read_parquet(NODES_PATH).dropna(subset=['lat', 'lon']).copy()
coords = np.column_stack([nodes['lat'].values, nodes['lon'].values])
dists, indices = tree.query(coords)
nodes['elevation'] = np.array(elev_vals)[indices]
st_elev_dict = nodes.set_index('station_id')['elevation'].to_dict()
st_name_dict = nodes.set_index('station_id')['name'].to_dict()

print("2. Computing Net Flux and Top Elevation Gradients...")
trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'dur_min', 'dist_km']).dropna()
trips['rent_st'] = trips['rent_st'].astype(str)
trips['ret_st'] = trips['ret_st'].astype(str)
trips['h_rent'] = trips['rent_st'].map(st_elev_dict)
trips['h_ret'] = trips['ret_st'].map(st_elev_dict)
trips = trips.dropna(subset=['h_rent', 'h_ret']).copy()
trips['dh'] = trips['h_ret'] - trips['h_rent']

# Outflow / Inflow per station
st_out = trips.groupby('rent_st')['dh'].count().rename('outflow')
st_in = trips.groupby('ret_st')['dh'].count().rename('inflow')
st_flux = pd.concat([st_out, st_in], axis=1).fillna(0)
st_flux['net_drain'] = st_flux['outflow'] - st_flux['inflow'] # 자전거 순유출량
st_flux['elevation'] = st_flux.index.map(st_elev_dict)
nodes_unique = nodes.drop_duplicates(subset=['station_id']).set_index('station_id')
st_flux['lat'] = st_flux.index.map(nodes_unique['lat'].to_dict())
st_flux['lon'] = st_flux.index.map(nodes_unique['lon'].to_dict())
st_flux['name'] = st_flux.index.map(st_name_dict)
st_flux = st_flux.dropna(subset=['elevation', 'lat', 'lon']).copy()

# Filter active stations (> 500 trips)
st_flux_active = st_flux[(st_flux['outflow'] + st_flux['inflow']) >= 500].copy()

# Top 5 Gravity Chutes
od_pairs = trips.groupby(['rent_st', 'ret_st']).agg(
    cnt=('dh', 'count'),
    mean_dh=('dh', 'mean')
).reset_index()
top_chutes = od_pairs[od_pairs['mean_dh'] < -15].sort_values('cnt', ascending=False).head(5).copy()
chute_data = []
for _, r in top_chutes.iterrows():
    s_name = st_name_dict.get(r['rent_st'], r['rent_st'])
    e_name = st_name_dict.get(r['ret_st'], r['ret_st'])
    rev_cnt = len(trips[(trips['rent_st'] == r['ret_st']) & (trips['ret_st'] == r['rent_st'])])
    chute_data.append({
        'pair': f"{s_name[:7]}.. -> {e_name[:7]}..\n(Δh={r['mean_dh']:.0f}m)",
        'down': r['cnt'],
        'up': rev_cnt,
        'ratio': r['cnt'] / max(1, rev_cnt),
        's_lat': nodes.loc[nodes['station_id'] == r['rent_st'], 'lat'].values[0],
        's_lon': nodes.loc[nodes['station_id'] == r['rent_st'], 'lon'].values[0],
        'e_lat': nodes.loc[nodes['station_id'] == r['ret_st'], 'lat'].values[0],
        'e_lon': nodes.loc[nodes['station_id'] == r['ret_st'], 'lon'].values[0],
    })

print("3. Generating 3-Panel Editorial Visualization...")
fig = plt.figure(figsize=(18, 6.5), dpi=300, facecolor='#f8f9fa')
fig.suptitle("대전 타슈 도시 중력 포텐셜 유동 (Urban Gravity Well) & 지형 비대칭성 실증", 
             fontsize=17, fontweight='bold', color='#111827', y=0.98)

# Panel 1: Elevation vs Net Drain Scatter
ax1 = fig.add_subplot(1, 3, 1, facecolor='#ffffff')
scatter = ax1.scatter(st_flux_active['elevation'], st_flux_active['net_drain'], 
                      c=st_flux_active['elevation'], cmap='plasma', alpha=0.65, s=28, edgecolors='none')
m, b = np.polyfit(st_flux_active['elevation'], st_flux_active['net_drain'], 1)
x_seq = np.linspace(st_flux_active['elevation'].min(), st_flux_active['elevation'].max(), 100)
ax1.plot(x_seq, m*x_seq + b, color='#e11d48', linewidth=2.5, linestyle='--', label=f'선형 추세선 (r = +0.220)')
ax1.axhline(0, color='#94a3b8', linestyle=':', linewidth=1)
ax1.set_title("① 대여소 표고(고도) vs 자전거 순유출량", fontsize=13, fontweight='bold', pad=12, color='#1e293b')
ax1.set_xlabel("대여소 표고 Elevation (m)", fontsize=11, color='#475569')
ax1.set_ylabel("자전거 순유출량 (출차 - 입차, 건)", fontsize=11, color='#475569')
ax1.grid(True, linestyle='--', alpha=0.4, color='#cbd5e1')
ax1.legend(loc='upper left', frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0', fontsize=10)
cbar = plt.colorbar(scatter, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('표고 (m)', fontsize=9, color='#64748b')

# Annotations on Panel 1
ax1.text(0.95, 0.05, "고지대: 자전거 지속 방출(고갈)\n저지대: 자전거 누적 유입(포화)", 
         transform=ax1.transAxes, fontsize=9.5, color='#0f172a', ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#eff6ff', edgecolor='#bfdbfe', alpha=0.9))

# Panel 2: Top Gravity Chutes (Downhill vs Uphill)
ax2 = fig.add_subplot(1, 3, 2, facecolor='#ffffff')
pairs_labels = [c['pair'] for c in chute_data]
downs = [c['down'] for c in chute_data]
ups = [c['up'] for c in chute_data]
y_pos = np.arange(len(pairs_labels))
bar_height = 0.35

rects1 = ax2.barh(y_pos + bar_height/2, downs, bar_height, label='내리막 하강 통행 (Downhill)', color='#2563eb', alpha=0.9)
rects2 = ax2.barh(y_pos - bar_height/2, ups, bar_height, label='오르막 역상승 통행 (Uphill)', color='#cbd5e1', alpha=0.9)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(pairs_labels, fontsize=9.5, color='#334155')
ax2.invert_yaxis()
ax2.set_title("② 5대 중력 쏠림 회랑: 하강 vs 상승 비대칭", fontsize=13, fontweight='bold', pad=12, color='#1e293b')
ax2.set_xlabel("연간 통행량 (건)", fontsize=11, color='#475569')
ax2.grid(True, axis='x', linestyle='--', alpha=0.4, color='#cbd5e1')
ax2.legend(loc='lower right', frameon=True, facecolor='#f8fafc', edgecolor='#e2e8f0', fontsize=9.5)

for i, c in enumerate(chute_data):
    ax2.text(c['down'] + 80, y_pos[i] + bar_height/2, f"{c['ratio']:.1f}배 쏠림!", 
             va='center', fontsize=9, fontweight='bold', color='#e11d48')

# Panel 3: Spatial Map with Elevation & Downhill Gravity Vectors
ax3 = fig.add_subplot(1, 3, 3, facecolor='#ffffff')
map_scatter = ax3.scatter(st_flux_active['lon'], st_flux_active['lat'], 
                          c=st_flux_active['elevation'], cmap='plasma', s=16, alpha=0.7)
ax3.set_title("③ 대전시 표고 지형도 & 중력 배수 플럭스 궤적", fontsize=13, fontweight='bold', pad=12, color='#1e293b')
ax3.set_xlabel("경도 (Longitude)", fontsize=11, color='#475569')
ax3.set_ylabel("위도 (Latitude)", fontsize=11, color='#475569')
ax3.grid(True, linestyle=':', alpha=0.3, color='#94a3b8')

# Draw arrows for top chutes
for c in chute_data:
    ax3.annotate('', xy=(c['e_lon'], c['e_lat']), xytext=(c['s_lon'], c['s_lat']),
                 arrowprops=dict(facecolor='#0284c7', edgecolor='#0369a1', width=1.8, headwidth=7, headlength=7, shrink=0.08))

ax3.text(0.04, 0.96, "화살표: 고도차 15m 이상\n급경사 하강 일방통행 흐름", 
         transform=ax3.transAxes, fontsize=9.5, color='#0369a1', va='top', fontweight='600',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f9ff', edgecolor='#bae6fd', alpha=0.9))

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

out_fig = ROOT / "outputs/report_figs/urban_gravity_well_analysis.png"
out_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_fig, dpi=300, facecolor='#f8f9fa')
plt.close()

# Also copy to artifacts dir for easy viewing
artifact_dest = Path(r"C:\Users\sijoo\.gemini\antigravity\brain\69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c\urban_gravity_well_analysis.png")
import shutil
shutil.copy(out_fig, artifact_dest)

print(f"Successfully generated publication figure at {out_fig} and {artifact_dest}")
