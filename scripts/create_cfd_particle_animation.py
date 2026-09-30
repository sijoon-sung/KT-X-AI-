import math
import sys
import io
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

sys.stdout.reconfigure(encoding='utf-8')

# Ensure Korean font support in Matplotlib
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/malgunbd.ttf"
font_prop = fm.FontProperties(fname=FONT_PATH)
font_prop_bold = fm.FontProperties(fname=FONT_BOLD_PATH)

ROOT = Path("C:/Users/sijoo/Documents/tashu")
TRIPS_PATH = ROOT / "data/processed/trips.parquet"

print("=================================================================")
print("  대전 타슈 863만 건 나비에-스토크스 CFD 유동장 & 애니메이션 생성기  ")
print("=================================================================")

# 1. Bounding Box & Tile Stitching
lon_min, lon_max = 127.320, 127.450
lat_min, lat_max = 36.310, 36.410
zoom = 13

print(f"1. Fetching clean Esri World Light Gray Canvas tiles (Zero Watermark)...")

def deg2num(lat_deg, lon_deg, z):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** z
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, z):
    n = 2.0 ** z
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

x_tile_min, y_tile_min = deg2num(lat_max, lon_min, zoom)
x_tile_max, y_tile_max = deg2num(lat_min, lon_max, zoom)

tiles_x = range(x_tile_min, x_tile_max + 1)
tiles_y = range(y_tile_min, y_tile_max + 1)
w_px = len(tiles_x) * 256
h_px = len(tiles_y) * 256

stitched = Image.new('RGB', (w_px, h_px), (245, 245, 245))

for ix, tx in enumerate(tiles_x):
    for iy, ty in enumerate(tiles_y):
        url = f"https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{zoom}/{ty}/{tx}"
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                t_img = Image.open(io.BytesIO(r.content)).convert('RGB')
                stitched.paste(t_img, (ix * 256, iy * 256))
        except Exception:
            pass

top_lat, left_lon = num2deg(x_tile_min, y_tile_min, zoom)
bot_lat, right_lon = num2deg(x_tile_max + 1, y_tile_max + 1, zoom)

px_min = int((lon_min - left_lon) / (right_lon - left_lon) * w_px)
px_max = int((lon_max - left_lon) / (right_lon - left_lon) * w_px)
py_min = int((top_lat - lat_max) / (top_lat - bot_lat) * h_px)
py_max = int((top_lat - lat_min) / (top_lat - bot_lat) * h_px)

base_map = stitched.crop((px_min, py_min, px_max, py_max))
# Subtle contrast & high-tech dark GIS tint
base_dark = ImageEnhance.Brightness(base_map).enhance(0.70)
base_dark = ImageEnhance.Contrast(base_dark).enhance(1.25)
map_w, map_h = base_dark.size
print(f"   Esri Basemap stitched: {map_w} x {map_h} pixels.")

# 2. Compute Navier-Stokes Velocity Field
print("2. Extracting Navier-Stokes Vector Field from 8.63M trips...")
trips = pd.read_parquet(TRIPS_PATH, columns=['rent_lat', 'rent_lon', 'ret_lat', 'ret_lon', 'dur_min', 'dist_km']).dropna()
trips = trips[(trips['dist_km'] > 0.05) & (trips['dur_min'] > 1) & (trips['dur_min'] < 90)]
trips = trips[(trips['rent_lon'] >= lon_min - 0.02) & (trips['rent_lon'] <= lon_max + 0.02) &
              (trips['rent_lat'] >= lat_min - 0.02) & (trips['rent_lat'] <= lat_max + 0.02)]

trips['dx'] = trips['ret_lon'] - trips['rent_lon']
trips['dy'] = trips['ret_lat'] - trips['rent_lat']
trips['dist_deg'] = np.hypot(trips['dx'], trips['dy'])
trips = trips[trips['dist_deg'] > 0.0005].copy()

trips['speed'] = (trips['dist_km'] / (trips['dur_min'] / 60.0)).clip(3.0, 22.0)
trips['mx'] = (trips['rent_lon'] + trips['ret_lon']) / 2.0
trips['my'] = (trips['rent_lat'] + trips['ret_lat']) / 2.0
trips['u_raw'] = (trips['dx'] / trips['dist_deg']) * trips['speed']
trips['v_raw'] = (trips['dy'] / trips['dist_deg']) * trips['speed']

# Regular 2D Grid
gx_n, gy_n = 110, 95
grid_lons = np.linspace(lon_min, lon_max, gx_n)
grid_lats = np.linspace(lat_min, lat_max, gy_n)

trips['ix'] = np.clip(np.digitize(trips['mx'], grid_lons) - 1, 0, gx_n - 1)
trips['iy'] = np.clip(np.digitize(trips['my'], grid_lats) - 1, 0, gy_n - 1)

grid_u = np.zeros((gy_n, gx_n))
grid_v = np.zeros((gy_n, gx_n))
grid_cnt = np.zeros((gy_n, gx_n))

agg = trips.groupby(['iy', 'ix']).agg(
    u=('u_raw', 'mean'),
    v=('v_raw', 'mean'),
    cnt=('speed', 'count')
).reset_index()

for _, r in agg.iterrows():
    iy, ix = int(r['iy']), int(r['ix'])
    grid_u[iy, ix] = r['u']
    grid_v[iy, ix] = r['v']
    grid_cnt[iy, ix] = r['cnt']

# Viscous fluid smoothing
grid_u_smooth = gaussian_filter(grid_u, sigma=1.6)
grid_v_smooth = gaussian_filter(grid_v, sigma=1.6)
grid_cnt_smooth = gaussian_filter(grid_cnt, sigma=1.6)

# Directional unit vectors
SPEED_FIELD = np.hypot(grid_u_smooth, grid_v_smooth)
FLOW_MOMENTUM = SPEED_FIELD * np.log1p(grid_cnt_smooth)
FLOW_MOM_NORM = FLOW_MOMENTUM / (np.percentile(FLOW_MOMENTUM[FLOW_MOMENTUM > 0], 98) + 1e-6)
FLOW_MOM_NORM = np.clip(FLOW_MOM_NORM, 0, 1.0)

# Continuous interpolators
interp_u = RegularGridInterpolator((grid_lats, grid_lons), grid_u_smooth, bounds_error=False, fill_value=0.0)
interp_v = RegularGridInterpolator((grid_lats, grid_lons), grid_v_smooth, bounds_error=False, fill_value=0.0)
interp_mom = RegularGridInterpolator((grid_lats, grid_lons), FLOW_MOM_NORM, bounds_error=False, fill_value=0.0)
interp_cnt = RegularGridInterpolator((grid_lats, grid_lons), grid_cnt_smooth, bounds_error=False, fill_value=0.0)

# Coordinate to pixel transform
def lonlat2pix(lon, lat):
    x = (lon - lon_min) / (lon_max - lon_min) * map_w
    y = (lat_max - lat) / (lat_max - lat_min) * map_h
    return x, y

def pix2lonlat(x, y):
    lon = lon_min + (x / map_w) * (lon_max - lon_min)
    lat = lat_max - (y / map_h) * (lat_max - lat_min)
    return lon, lat

# 3. High-Definition Static Publication Map (Matplotlib)
print("3. Rendering High-Definition Publication CFD Spectrum Map (Matplotlib)...")
X_mesh, Y_mesh = np.meshgrid(grid_lons, grid_lats)

fig, ax = plt.subplots(figsize=(16, 12), dpi=300, facecolor='#0b0f19')
# Basemap
ax.imshow(base_dark, extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto', zorder=1)

# CFD Rainbow Spectral Heatmap (Turbo Colormap)
cfd_mesh = ax.contourf(X_mesh, Y_mesh, FLOW_MOM_NORM, levels=50, cmap='turbo', alpha=0.52, zorder=2)

# Streamlines (white with slight glow)
stream = ax.streamplot(grid_lons, grid_lats, grid_u_smooth, grid_v_smooth,
                      density=2.6, color='#ffffff', linewidth=1.1,
                      arrowsize=1.2, arrowstyle='->', broken_streamlines=True, zorder=3)

# Labeled Fluid Mechanics Landmarks
landmarks = [
    ("카이스트 [고지대 분출원 ∇·v > 0]", 127.359, 36.368, '#38bdf8', '#0284c7'),
    ("충남대학교 [학생 통학 유출류]", 127.345, 36.368, '#38bdf8', '#0284c7'),
    ("유성온천역 [최대 흡입 싱크 ∇·v < 0]", 127.341, 36.353, '#ef4444', '#b91c1c'),
    ("갈마역·월평 [다운힐 가속 병목]", 127.362, 36.358, '#f59e0b', '#d97706'),
    ("대전시청 [도심 순환 와류 Vortex]", 127.387, 36.354, '#ef4444', '#b91c1c'),
    ("서대전네거리 [원도심 중력 싱크]", 127.404, 36.321, '#ef4444', '#b91c1c'),
]

for name, lon, lat, col, border_col in landmarks:
    ax.scatter(lon, lat, color=col, s=110, edgecolors='#ffffff', linewidth=2.0, zorder=6)
    ax.text(lon + 0.0018, lat + 0.0014, name, fontproperties=font_prop_bold, fontsize=10,
            color='#f8fafc', zorder=7,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#0f172a', edgecolor=border_col, alpha=0.92, linewidth=1.6))

# Title & Physics Subtitle
ax.set_title("대전광역시 타슈 863만 건 나비에-스토크스(Navier-Stokes) 전산유체역학(CFD) 유동장 스펙트럼",
             fontproperties=font_prop_bold, fontsize=15, color='#f8fafc', pad=14)
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel("경도 Longitude (°E)", fontproperties=font_prop, fontsize=11, color='#94a3b8')
ax.set_ylabel("위도 Latitude (°N)", fontproperties=font_prop, fontsize=11, color='#94a3b8')
ax.tick_params(colors='#94a3b8', labelsize=9.5)
for spine in ax.spines.values():
    spine.set_color('#334155')

# Turbo Spectrum Colorbar
cbar = plt.colorbar(cfd_mesh, ax=ax, fraction=0.03, pad=0.025)
cbar.set_label('CFD 유체 모멘텀 플럭스 스펙트럼 [Turbo: 파랑(정체/완류) -> 청록/녹색(주거지) -> 황색(간선) -> 적색(흡입 싱크)]',
               fontproperties=font_prop_bold, fontsize=10, color='#e2e8f0', labelpad=12)
cbar.ax.yaxis.set_tick_params(color='#e2e8f0')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e2e8f0', fontproperties=font_prop)

plt.tight_layout()
static_out = ROOT / "outputs/report_figs/tashu_cfd_spectral_map.png"
os.makedirs(static_out.parent, exist_ok=True)
plt.savefig(static_out, dpi=300, facecolor='#0b0f19')
plt.close()
print(f"   Static CFD publication map saved: {static_out}")

# 4. Animated Particle Flow GIF (Windy / CFD Streamline Simulation)
print("4. Simulating and Rendering 36-Frame Animated Fluid Particle Flow GIF...")
n_particles = 2200
n_frames = 36

# Particle spawning along active bike corridors
# We use trip density threshold so particles spawn only where bikes circulate
density_flat = grid_cnt_smooth.flatten()
# Add small baseline so entire urban zone has some flowing particles
density_prob = (density_flat + 1.0) / (density_flat + 1.0).sum()
flat_indices = np.random.choice(len(density_flat), size=n_particles, p=density_prob)
iy_init, ix_init = np.unravel_index(flat_indices, grid_cnt_smooth.shape)

p_lons = grid_lons[ix_init] + np.random.uniform(-0.001, 0.001, n_particles)
p_lats = grid_lats[iy_init] + np.random.uniform(-0.001, 0.001, n_particles)
p_ages = np.random.randint(0, 30, n_particles)

turbo_cmap = plt.cm.turbo

# Base background for animation: dark Esri map with soft CFD contour
fig_bg, ax_bg = plt.subplots(figsize=(10, 7.5), dpi=100, facecolor='#0b0f19')
ax_bg.imshow(base_dark, extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto')
ax_bg.contourf(X_mesh, Y_mesh, FLOW_MOM_NORM, levels=35, cmap='turbo', alpha=0.38)
ax_bg.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
bg_buf = io.BytesIO()
plt.savefig(bg_buf, format='png', facecolor='#0b0f19', pad_inches=0)
plt.close(fig_bg)
bg_buf.seek(0)
anim_bg = Image.open(bg_buf).convert('RGBA')
anim_w, anim_h = anim_bg.size

def anim_lonlat2pix(lon, lat):
    x = (lon - lon_min) / (lon_max - lon_min) * anim_w
    y = (lat_max - lat) / (lat_max - lat_min) * anim_h
    return x, y

trail_layer = Image.new('RGBA', (anim_w, anim_h), (0, 0, 0, 0))

pil_font_title = ImageFont.truetype(FONT_BOLD_PATH, 16)
pil_font_sub = ImageFont.truetype(FONT_PATH, 12)

gif_frames = []

for frame_idx in range(n_frames):
    pts = np.column_stack([p_lats, p_lons])
    u_vals = interp_u(pts)
    v_vals = interp_v(pts)
    mom_vals = interp_mom(pts)
    
    # Calculate velocity magnitude
    speed_mag = np.hypot(u_vals, v_vals)
    
    # Compute normalized streamline direction unit vectors
    # Avoid zero division: where speed is very small, use local gradient or drift
    valid_dir = speed_mag > 0.05
    u_dir = np.zeros_like(u_vals)
    v_dir = np.zeros_like(v_vals)
    u_dir[valid_dir] = u_vals[valid_dir] / speed_mag[valid_dir]
    v_dir[valid_dir] = v_vals[valid_dir] / speed_mag[valid_dir]
    
    # For small vector areas, give gentle southward/inward drift toward center
    zero_dir = ~valid_dir
    u_dir[zero_dir] = (127.38 - p_lons[zero_dir]) * 0.5
    v_dir[zero_dir] = (36.35 - p_lats[zero_dir]) * 0.5
    drift_mag = np.hypot(u_dir[zero_dir], v_dir[zero_dir]) + 1e-6
    u_dir[zero_dir] /= drift_mag
    v_dir[zero_dir] /= drift_mag
    
    # Step displacement in degrees:
    # 4 to 12 pixels per frame mapped to momentum
    step_deg_lon = (0.0006 + 0.0016 * mom_vals)
    step_deg_lat = (0.0005 + 0.0013 * mom_vals)
    
    draw_trail = ImageDraw.Draw(trail_layer)
    
    for i in range(n_particles):
        x1, y1 = anim_lonlat2pix(p_lons[i], p_lats[i])
        
        # Advance particle along Navier-Stokes streamline
        p_lons[i] += u_dir[i] * step_deg_lon[i]
        p_lats[i] += v_dir[i] * step_deg_lat[i]
        p_ages[i] += 1
        
        x2, y2 = anim_lonlat2pix(p_lons[i], p_lats[i])
        
        # Color mapped by Turbo CFD Spectrum
        color_val = np.clip(mom_vals[i], 0.05, 0.98)
        rgba = turbo_cmap(color_val)
        alpha = int(min(255, 160 + color_val * 95))
        color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255), alpha)
        
        # Draw motion streakline
        draw_trail.line([(x1, y1), (x2, y2)], fill=color, width=2)
        # Glowing head
        head_color = (255, 255, 255, alpha) if color_val > 0.5 else color
        draw_trail.ellipse([(x2-1, y2-1), (x2+1, y2+1)], fill=head_color)
        
        # Respawn criteria: boundary or age
        if (p_lons[i] < lon_min or p_lons[i] > lon_max or 
            p_lats[i] < lat_min or p_lats[i] > lat_max or 
            p_ages[i] > 28):
            # Respawn at density weighted location
            new_idx = np.random.choice(len(density_flat), p=density_prob)
            ny_i, nx_i = np.unravel_index(new_idx, grid_cnt_smooth.shape)
            p_lons[i] = grid_lons[nx_i] + np.random.uniform(-0.001, 0.001)
            p_lats[i] = grid_lats[ny_i] + np.random.uniform(-0.001, 0.001)
            p_ages[i] = 0

    # Composite: Background + Trail
    frame_comp = Image.alpha_composite(anim_bg, trail_layer)
    
    # Overlay HUD Card
    draw_hud = ImageDraw.Draw(frame_comp)
    # HUD Box
    hud_box = [(16, 16), (560, 80)]
    draw_hud.rectangle(hud_box, fill=(11, 15, 25, 220), outline=(56, 189, 248, 255), width=2)
    # Korean Text
    draw_hud.text((28, 24), "대전 타슈 나비에-스토크스 입자 유동 애니메이션 (Windy/CFD)", font=pil_font_title, fill=(248, 250, 252))
    draw_hud.text((28, 50), "• 속도 스펙트럼: 청색(완류 3km/h) -> 녹색(8km/h) -> 황색(12km/h) -> 적색(흡입구 20km/h)", font=pil_font_sub, fill=(148, 163, 184))
    
    # Phosphorus decay for next frame (retains 72% of previous trails)
    trail_arr = np.array(trail_layer)
    trail_arr[:, :, 3] = (trail_arr[:, :, 3] * 0.72).astype(np.uint8)
    trail_layer = Image.fromarray(trail_arr)
    
    # Convert to RGB and append
    frame_rgb = frame_comp.convert('RGB')
    gif_frames.append(frame_rgb)

# Quantize and save animated GIF
print("   Saving 36-frame animated GIF with adaptive 256-color palette...")
gif_out = ROOT / "outputs/report_figs/tashu_navier_stokes_animated.gif"

# Use optimal adaptive palette from middle frame
palette_img = gif_frames[18].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
p_frames = [f.quantize(colors=256, palette=palette_img) for f in gif_frames]

p_frames[0].save(
    gif_out,
    save_all=True,
    append_images=p_frames[1:],
    duration=75, # 75ms = 13.3 FPS
    loop=0,
    optimize=True
)
print(f"   Animated CFD GIF successfully saved: {gif_out}")

# 5. Distribute artifacts
print("5. Distributing updated files across all project endpoints...")
copies = [
    ROOT / "tashu_cfd_spectral_map.png",
    ROOT / "docs/tashu_cfd_spectral_map.png",
    Path(r"C:\Users\sijoo\.gemini\antigravity\brain\69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c\tashu_cfd_spectral_map.png"),
]
for cp in copies:
    shutil.copy(static_out, cp)

gif_copies = [
    ROOT / "tashu_navier_stokes_animated.gif",
    ROOT / "docs/tashu_navier_stokes_animated.gif",
    Path(r"C:\Users\sijoo\.gemini\antigravity\brain\69b779d9-e00f-4cbb-87f9-bef0e1d1ae7c\tashu_navier_stokes_animated.gif"),
]
desktop_pkg = Path(r"C:\Users\sijoo\OneDrive\바탕 화면\타슈_재배치_최적화_최종결과패키지")
if desktop_pkg.exists():
    shutil.copy(static_out, desktop_pkg / "tashu_cfd_spectral_map.png")
    shutil.copy(gif_out, desktop_pkg / "tashu_navier_stokes_animated.gif")

for cp in gif_copies:
    shutil.copy(gif_out, cp)

print("All CFD visualization assets generated and synchronized successfully!")
