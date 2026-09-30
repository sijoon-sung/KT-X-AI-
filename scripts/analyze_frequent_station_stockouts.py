import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
INVENTORY_PATH = ROOT / "data/processed/v3/inventory_hourly.parquet"

print("1. Loading nodes and inventory data...")
nodes = pd.read_parquet(NODES_PATH)
node_dict = nodes.set_index('node')[['station_id', 'name', 'lat', 'lon', 'n_rent']].to_dict('index')

# We can focus on 2025 peak season (e.g., 2025-09-01 to 2025-09-30, or entire 2025)
# Let's inspect September 2025 first (720 hours) and full 2025
# Let's filter for frequently used stations: n_rent >= 5000 (top active stations)
active_nodes = [node for node, d in node_dict.items() if d['n_rent'] >= 5000]
print(f"Total active nodes with n_rent >= 5,000: {len(active_nodes)}")

# Load inventory for September 2025
print("2. Reading September 2025 inventory data...")
cols = ['ts', 'node', 'stock_min', 'stock_mean', 'empty_frac', 'rent', 'ret']
t_start = pd.Timestamp('2025-09-01 00:00:00')
t_end = pd.Timestamp('2025-09-30 23:59:59')
df = pd.read_parquet(INVENTORY_PATH, columns=cols, filters=[
    ('node', 'in', active_nodes),
    ('ts', '>=', t_start),
    ('ts', '<=', t_end)
])
df = df.sort_values(['node', 'ts']).reset_index(drop=True)
print(f"Loaded {len(df):,} records for September 2025.")

# Define 0-bike condition:
# stock_min == 0 means during that hour, bikes ran out to 0 at least once
# empty_frac >= 0.5 means for more than 30 minutes in that hour, stock was 0
# stock_mean == 0 means completely empty throughout the hour
df['is_zero'] = (df['stock_min'] == 0).astype(int)
df['is_severe_zero'] = (df['empty_frac'] >= 0.5).astype(int)

# Calculate consecutive 0-stock episodes (Run Length Encoding per node)
results = []
all_episodes = []

for node, group in df.groupby('node'):
    st_info = node_dict[node]
    st_id = st_info['station_id']
    st_name = st_info['name']
    tot_rent = group['rent'].sum()
    
    # Identify continuous blocks where is_zero == 1
    # group is sorted by ts
    group = group.copy()
    group['zero_block'] = (group['is_zero'] != group['is_zero'].shift(1)).cumsum()
    
    # Filter to only zero blocks
    zero_blocks = group[group['is_zero'] == 1]
    
    if len(zero_blocks) == 0:
        continue
        
    episode_stats = zero_blocks.groupby('zero_block').agg(
        start_ts=('ts', 'min'),
        end_ts=('ts', 'max'),
        duration_hours=('ts', 'count'),
        avg_empty_frac=('empty_frac', 'mean'),
        lost_rent=('rent', 'sum')
    ).reset_index()
    
    episode_stats['node'] = node
    episode_stats['station_id'] = st_id
    episode_stats['name'] = st_name
    all_episodes.append(episode_stats)
    
    tot_zero_hours = group['is_zero'].sum()
    tot_severe_hours = group['is_severe_zero'].sum()
    tot_hours = len(group)
    max_duration = episode_stats['duration_hours'].max()
    avg_duration = episode_stats['duration_hours'].mean()
    num_episodes = len(episode_stats)
    
    results.append({
        'node': node,
        'station_id': st_id,
        'name': st_name,
        'total_rent': tot_rent,
        'n_rent_overall': st_info['n_rent'],
        'tot_zero_hours': tot_zero_hours,
        'zero_hour_pct': tot_zero_hours / tot_hours * 100,
        'tot_severe_hours': tot_severe_hours,
        'severe_hour_pct': tot_severe_hours / tot_hours * 100,
        'max_duration': max_duration,
        'avg_duration': avg_duration,
        'num_episodes': num_episodes
    })

res_df = pd.DataFrame(results)
ep_df = pd.concat(all_episodes, ignore_index=True) if all_episodes else pd.DataFrame()

print("\n" + "="*85)
print(" 2025년 9월(성수기): 자주 이용되는 고수요 대여소 중 '0대 결품' 발생 상위 20개 대여소")
print("="*85)
# Filter: total_rent in Sept >= 1000 or top rent, sorted by tot_zero_hours
high_demand_zeros = res_df[res_df['total_rent'] >= 1000].sort_values('tot_zero_hours', ascending=False)

header = f"{'대여소ID':<8} | {'대여소명':<22} | {'9월대여':<7} | {'0대총시간(비율)':<16} | {'최장연속(h)':<10} | {'평균지속(h)':<10} | {'결품발생횟수':<10}"
print(header)
print("-" * len(header))

for _, r in high_demand_zeros.head(20).iterrows():
    print(f"{r['station_id']:<8} | {r['name'][:20]:<22} | {r['total_rent']:<7.0f} | {r['tot_zero_hours']:3d}시간 ({r['zero_hour_pct']:4.1f}%) | {r['max_duration']:4d}시간     | {r['avg_duration']:4.1f}시간     | {r['num_episodes']:3d}회")

print("\n" + "="*85)
print(" 실제 최장 연속 0대 방치(결품) 에피소드 사례 TOP 10")
print("="*85)
longest_episodes = ep_df.sort_values('duration_hours', ascending=False).head(10)
for _, ep in longest_episodes.iterrows():
    print(f"[{ep['station_id']}] {ep['name'][:20]} | 지속시간: {ep['duration_hours']}시간 연속 결품! ({ep['start_ts']} ~ {ep['end_ts']})")

# Hour of day distribution of 0-stock
print("\n" + "="*85)
print(" 시간대별 0대 결품 발생 빈도 (어느 시간에 가장 많이 0대가 되는가?)")
print("="*85)
df['hour'] = pd.to_datetime(df['ts']).dt.hour
hourly_zero = df.groupby('hour')['is_zero'].agg(['sum', 'mean']).reset_index()
for _, h in hourly_zero.iterrows():
    bar = "█" * int(h['mean'] * 150)
    print(f"{int(h['hour']):02d}시 | 결품율: {h['mean']*100:4.1f}% | {bar}")
