import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
INVENTORY_PATH = ROOT / "data/processed/v3/inventory_hourly.parquet"

nodes = pd.read_parquet(NODES_PATH)
node_to_name = nodes.set_index('node')['name'].to_dict()
node_to_st = nodes.set_index('node')['station_id'].to_dict()

# Load September 2025 inventory
df = pd.read_parquet(INVENTORY_PATH,
                     columns=['ts', 'node', 'stock_min', 'stock_mean', 'stock_start', 'rent', 'ret'],
                     filters=[('ts', '>=', pd.Timestamp('2025-09-01')), 
                              ('ts', '<=', pd.Timestamp('2025-09-30 23:59:59'))])

df = df.sort_values(['node', 'ts']).reset_index(drop=True)
df['name'] = df['node'].map(node_to_name)
df['st_id'] = df['node'].map(node_to_st)

# 1. Stagnant episodes: stock >= 3 and rent == 0 and ret == 0 (no activity, bikes just sitting there)
# Or stock variation is 0 for consecutive hours
df['is_stagnant'] = ((df['stock_min'] >= 3) & (df['rent'] == 0) & (df['ret'] == 0)).astype(int)

# Group by node to analyze:
# A) Overflowing Sinks (자전거가 계속 쌓여서 수십~수백 대씩 방치되는 곳)
st_agg = df.groupby(['node', 'st_id', 'name']).agg(
    avg_stock=('stock_mean', 'mean'),
    max_stock=('stock_mean', 'max'),
    min_stock=('stock_min', 'min'),
    std_stock=('stock_mean', 'std'),
    total_rent=('rent', 'sum'),
    total_ret=('ret', 'sum'),
    stagnant_hours=('is_stagnant', 'sum'),
    tot_hours=('rent', 'count')
).reset_index()

st_agg['net_diff'] = st_agg['total_ret'] - st_agg['total_rent'] # positive means influx sink
st_agg['stagnant_pct'] = st_agg['stagnant_hours'] / st_agg['tot_hours'] * 100

print("="*90)
print(" [타입 1] '거대 잉여 고갈지(Sink)': 자전거가 끊임없이 쌓여 수십~100대 이상 방치되는 거점 TOP 15")
print("="*90)
sinks = st_agg.sort_values('avg_stock', ascending=False).head(15)
header1 = f"{'대여소ID':<8} | {'대여소명':<24} | {'평균재고':<7} | {'최대재고':<7} | {'최소재고':<7} | {'월반납-월대여(순유입)':<15}"
print(header1)
print("-" * len(header1))
for _, r in sinks.iterrows():
    print(f"{r['st_id']:<8} | {r['name'][:22]:<24} | {r['avg_stock']:5.1f}대 | {r['max_stock']:5.0f}대 | {r['min_stock']:5.0f}대 | +{r['net_diff']:4.0f}대 (반납:{r['total_ret']:.0f}/대여:{r['total_rent']:.0f})")

# B) Ghost / Dead Stations (자전거가 항상 5~15대 이상 있는데 한 달 내내 대여/반납이 거의 없는 유령 대여소)
print("\n" + "="*90)
print(" [타입 2] '고인 물 유휴 거점': 자전거는 항상 5대 이상 있는데 아무도 안 타는 대여소 TOP 15")
print("="*90)
dead = st_agg[(st_agg['min_stock'] >= 3) & (st_agg['tot_hours'] >= 600)].sort_values('total_rent', ascending=True).head(15)
header2 = f"{'대여소ID':<8} | {'대여소명':<24} | {'평균재고':<7} | {'9월총대여':<7} | {'무변동방치시간':<15} | {'방치비율':<8}"
print(header2)
print("-" * len(header2))
for _, r in dead.iterrows():
    print(f"{r['st_id']:<8} | {r['name'][:22]:<24} | {r['avg_stock']:5.1f}대 | {r['total_rent']:4.0f}건 | {r['stagnant_hours']:3.0f}시간 / 720h | {r['stagnant_pct']:4.1f}%")

# C) Longest continuous frozen episodes (자전거 수가 단 1대도 변하지 않고 연속으로 멈춰있는 최장 기간)
print("\n" + "="*90)
print(" [타입 3] '최장 연속 무변동(동결) 방치 사례': 대여·반납 0건으로 자전거가 완전히 멈춰있던 최장 시간")
print("="*90)
# Find continuous runs of is_stagnant == 1
df['stag_block'] = (df['is_stagnant'] != df.groupby('node')['is_stagnant'].shift(1)).cumsum()
stag_episodes = df[df['is_stagnant'] == 1].groupby(['st_id', 'name', 'stag_block']).agg(
    start_ts=('ts', 'min'),
    end_ts=('ts', 'max'),
    duration=('ts', 'count'),
    stock=('stock_min', 'mean')
).reset_index()

longest_stag = stag_episodes.sort_values('duration', ascending=False).head(15)
for _, ep in longest_stag.iterrows():
    days = ep['duration'] / 24
    print(f"[{ep['st_id']}] {ep['name'][:20]:20s} | {ep['duration']:3d}시간 ({days:.1f}일간) 연속 무변동! (재고 {ep['stock']:.0f}대 고정, {ep['start_ts']} ~ {ep['end_ts']})")
