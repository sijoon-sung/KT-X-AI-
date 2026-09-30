import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
TRIPS_PATH = ROOT / "data/processed/trips.parquet"
ELEVATION_PATH = ROOT / "data/raw/elevation_nodes.json"

print("1. Loading spatial nodes and trip trajectories...")
nodes = pd.read_parquet(NODES_PATH).dropna(subset=['lat', 'lon']).copy()
nodes_u = nodes.drop_duplicates(subset=['station_id']).set_index('station_id')
st_coords = {st: (row['lon'], row['lat']) for st, row in nodes_u.iterrows()}
st_names = nodes_u['name'].to_dict()

# Elevation
with open(ELEVATION_PATH, 'r', encoding='utf-8') as f:
    elev_raw = json.load(f)
elev_pts = [list(map(float, k.split(','))) for k in elev_raw.keys()]
elev_vals = list(elev_raw.values())
tree_elev = cKDTree(np.array(elev_pts))
node_coords = np.column_stack([nodes_u['lat'].values, nodes_u['lon'].values])
_, idxs = tree_elev.query(node_coords)
nodes_u['elev'] = np.array(elev_vals)[idxs]
st_elev = nodes_u['elev'].to_dict()

# Trips aggregation into OD flows
trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'dur_min'])
trips['rent_st'] = trips['rent_st'].astype(str)
trips['ret_st'] = trips['ret_st'].astype(str)
trips = trips[(trips['rent_st'].isin(st_coords)) & (trips['ret_st'].isin(st_coords))].copy()

# Filter out self-loops for flow vector field
trips_flow = trips[trips['rent_st'] != trips['ret_st']].copy()
od_agg = trips_flow.groupby(['rent_st', 'ret_st']).agg(
    count=('dur_min', 'count'),
    mean_dur=('dur_min', 'mean')
).reset_index()

print(f"Total non-loop OD pairs: {len(od_agg):,} pairs, representing {od_agg['count'].sum():,} trips.")

# Compute flow vectors and midpoints
od_data = []
for _, r in od_agg.iterrows():
    s_lon, s_lat = st_coords[r['rent_st']]
    e_lon, e_lat = st_coords[r['ret_st']]
    cnt = r['count']
    dx = e_lon - s_lon
    dy = e_lat - s_lat
    dist = np.hypot(dx, dy)
    if dist < 0.0005 or dist > 0.15: # Ignore micro or extreme outliers
        continue
    # Midpoint
    mx = (s_lon + e_lon) / 2.0
    my = (s_lat + e_lat) / 2.0
    # Velocity unit vector * volume
    od_data.append([mx, my, dx * cnt, dy * cnt, cnt])

od_arr = np.array(od_data)
print(f"Valid flow segments: {len(od_arr):,}")

print("2. Constructing continuous 2D Navier-Stokes grid (Velocity & Flux Field)...")
# Regular Grid covering core Daejeon (Yuseong, Dunsan, Jung-gu, Dong-gu)
nx, ny = 90, 80
x_min, x_max = 127.315, 127.455
y_min, y_max = 36.300, 36.415

grid_x = np.linspace(x_min, x_max, nx)
grid_y = np.linspace(y_min, y_max, ny)
X, Y = np.meshgrid(grid_x, grid_y)

# Kernel density interpolation of velocity field U, V
tree_od = cKDTree(od_arr[:, :2])
bandwidth = 0.015 # ~1.5 km influence radius
U = np.zeros_like(X)
V = np.zeros_like(Y)
DENS = np.zeros_like(X)

for j in range(ny):
    for i in range(nx):
        pt = [X[j, i], Y[j, i]]
        idx_neighbors = tree_od.query_ball_point(pt, r=bandwidth)
        if not idx_neighbors:
            continue
        pts_near = od_arr[idx_neighbors]
        dists = np.hypot(pts_near[:, 0] - pt[0], pts_near[:, 1] - pt[1])
        weights = np.exp(-0.5 * (dists / (bandwidth / 2.5)) ** 2)
        total_w = np.sum(weights)
        if total_w > 0:
            U[j, i] = np.sum(pts_near[:, 2] * weights) / total_w
            V[j, i] = np.sum(pts_near[:, 3] * weights) / total_w
            DENS[j, i] = np.sum(pts_near[:, 4] * weights)

# Compute Speed & Divergence (Divergence = dU/dx + dV/dy)
SPEED = np.hypot(U, V)
# Normalize for smooth Navier-Stokes streamline tracing
mask = SPEED < 0.05
U_norm = np.where(mask, 0, U / (SPEED + 1e-6))
V_norm = np.where(mask, 0, V / (SPEED + 1e-6))

# Numerical Divergence (Sources vs Sinks)
dx_step = (x_max - x_min) / nx
dy_step = (y_max - y_min) / ny
dUdX = np.gradient(U, dx_step, axis=1)
dVdY = np.gradient(V, dy_step, axis=0)
DIV = dUdX + dVdY

print("3. Rendering Navier-Stokes Streamline & Potential Flux Visualization...")
fig = plt.figure(figsize=(16, 12), dpi=300, facecolor='#0b0f19')
ax = fig.add_subplot(1, 1, 1, facecolor='#0b0f19')

# Background: Subtle Divergence potential field (Sinks in Cyan/Blue, Sources in Rose/Red)
div_limit = np.percentile(np.abs(DIV[SPEED > 0.1]), 95)
norm_div = mcolors.TwoSlopeNorm(vmin=-div_limit, vcenter=0, vmax=div_limit)
cf = ax.contourf(X, Y, DIV, levels=50, cmap='RdBu_r', norm=norm_div, alpha=0.35, extend='both')

# Beautiful Fluid Streamlines (colored by speed/momentum)
stream = ax.streamplot(grid_x, grid_y, U, V, 
                      density=2.2, 
                      color=SPEED, 
                      cmap='plasma', 
                      linewidth=1.2 + 2.5 * (SPEED / np.max(SPEED)), 
                      arrowsize=1.4, 
                      arrowstyle='->',
                      broken_streamlines=True)

# Overlay Key Stations as glowing nodes
core_stations = [
    ("ST0373", "카이스트 서문 (고지대 분출구)"),
    ("ST1016", "충남대 학생회관"),
    ("ST1085", "유성온천역 (거대 유체 흡입 싱크)"),
    ("ST0012", "시청역 (도심 복합 흡입구)"),
    ("ST0400", "대덕대 (고지대 중력 폭포)"),
    ("ST0181", "송림마을 5단지 (과적 정체 싱크)"),
    ("ST0095", "갈마역 (퇴근 유체 병목)"),
    ("ST0511", "서대전네거리 (동남부 수렴 싱크)"),
]

for st_id, label in core_stations:
    if st_id in st_coords:
        lon, lat = st_coords[st_id]
        elev = st_elev.get(st_id, 50)
        # Pulse node
        ax.scatter(lon, lat, color='#38bdf8', s=110, edgecolors='#ffffff', linewidth=1.5, zorder=10)
        ax.text(lon + 0.002, lat + 0.0015, f"{label}\n({elev:.0f}m)", 
                fontsize=9.5, fontweight='bold', color='#ffffff', zorder=11,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', edgecolor='#38bdf8', alpha=0.85))

# Title & Watermark block
ax.set_title("대전시 타슈 863만 건 나비에-스토크스 유체 유선장 (Navier-Stokes Streamline & Divergence Field)", 
             fontsize=16, fontweight='bold', color='#f8fafc', pad=18)
ax.set_xlabel("경도 Longitude (°E)", fontsize=11, color='#94a3b8')
ax.set_ylabel("위도 Latitude (°N)", fontsize=11, color='#94a3b8')

ax.tick_params(colors='#94a3b8')
for spine in ax.spines.values():
    spine.set_color('#1e293b')

# Colorbar for Streamline Momentum
cbar = plt.colorbar(stream.lines, ax=ax, fraction=0.03, pad=0.03)
cbar.set_label('유동 플럭스 모멘텀 (Flow Momentum Flux: ||J|| = ρ||v||)', fontsize=10, color='#cbd5e1')
cbar.ax.yaxis.set_tick_params(color='#cbd5e1')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#cbd5e1')

# Legend annotation box
legend_html = (
    "■ 나비에-스토크스 유체 해석 안내:\n"
    "• 유선 (Streamlines): 863만 건 자전거 이동의 순간 접선 방향 (유동 벡터장)\n"
    "• 유선 색상/두께: 유체 모멘텀 플럭스 (노란색/밝을수록 초고밀도 통행)\n"
    "• 배경 음영: 유체 발산도 (Divergence ∇·J)\n"
    "   - 붉은색 (∇·J > 0): 고지대 자전거 배출구 (Sources)\n"
    "   - 푸른색 (∇·J < 0): 저지대/역세권 자전거 흡입 싱크 (Sinks / Blackholes)"
)
ax.text(0.02, 0.04, legend_html, transform=ax.transAxes, fontsize=10, color='#e2e8f0', 
        va='bottom', ha='left', zorder=20,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#020617', edgecolor='#475569', alpha=0.92))

plt.tight_layout()

out_cfd = ROOT / "outputs/report_figs/tashu_navier_stokes_streamlines.png"
out_cfd.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_cfd, dpi=300, facecolor='#0b0f19')
plt.close()

# Copy to root and docs and artifact
shutil_path = ROOT / "tashu_navier_stokes_streamlines.png"
import shutil
shutil.copy(out_cfd, shutil_path)
shutil.copy(out_cfd, ROOT / "docs/tashu_navier_stokes_streamlines.png")
artifact_dest = Path(r"C:\Users\sijoo\.gemini\antigravity\brain\69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c\tashu_navier_stokes_streamlines.png")
shutil.copy(out_cfd, artifact_dest)

print(f"CFD Streamline map saved successfully to:\n- {out_cfd}\n- {shutil_path}\n- {artifact_dest}")

# 4. Also export vector field grid to JSON for Web Particle Animation!
particles_data = {
    "x_min": x_min, "x_max": x_max, "nx": nx,
    "y_min": y_min, "y_max": y_max, "ny": ny,
    "U": np.round(U, 4).tolist(),
    "V": np.round(V, 4).tolist(),
    "SPEED": np.round(SPEED, 3).tolist()
}
json_out = ROOT / "outputs/final_model/fluid_vector_field.json"
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(particles_data, f)
print(f"Exported fluid vector grid JSON to {json_out}")
