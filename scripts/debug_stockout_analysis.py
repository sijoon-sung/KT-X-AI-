import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
MAPPING_PATH = ROOT / "data/processed/cluster_mapping_300m.parquet"
INVENTORY_PATH = ROOT / "data/processed/v3/inventory_hourly.parquet"

nodes = pd.read_parquet(NODES_PATH)
df_map = pd.read_parquet(MAPPING_PATH)
node_to_st = nodes.set_index('node')['station_id'].to_dict()
st_to_name = nodes.set_index('station_id')['name'].to_dict()
st_to_c = df_map.set_index('station_id')['cluster_id'].to_dict()

cols = ['ts', 'node', 'stock_min', 'stock_mean', 'empty_frac', 'rent']
inv = pd.read_parquet(INVENTORY_PATH, columns=cols)
inv = inv[(inv['ts'] >= '2025-09-01') & (inv['ts'] <= '2025-09-30 23:00')].copy()
inv['st_id'] = inv['node'].map(node_to_st)
inv['name'] = inv['st_id'].map(st_to_name)
inv['cluster'] = inv['st_id'].map(st_to_c)

# 1. 개별 대여소 결품 현황 (Top 15 대여소)
top_st = inv.groupby('st_id').agg(
    total_rent=('rent', 'sum'),
    stockout_hours=('empty_frac', lambda x: (x >= 0.5).sum()),
    zero_stock_hours=('stock_min', lambda x: (x == 0).sum()),
    total_hours=('empty_frac', 'count'),
    mean_stock=('stock_mean', 'mean')
).reset_index()
top_st['name'] = top_st['st_id'].map(st_to_name)
top_st['stockout_pct'] = top_st['stockout_hours'] / top_st['total_hours'] * 100
top_st['zero_min_pct'] = top_st['zero_stock_hours'] / top_st['total_hours'] * 100

print("=== [1] 2025년 9월 성수기 고수요 대여소 TOP 15의 실제 결품(0대) 발생률 ===")
top_st_sorted = top_st.sort_values('total_rent', ascending=False).head(15)
for _, r in top_st_sorted.iterrows():
    print(f"[{r['st_id']}] {r['name'][:18]:18s} | 월대여: {r['total_rent']:4.0f}건 | 평균재고: {r['mean_stock']:4.1f}대 | 품절비율: {r['stockout_pct']:4.1f}% | 0대경험비율: {r['zero_min_pct']:4.1f}%")

# 2. 군집으로 묶었을 때의 결품 방어율 비교
print("\n=== [2] 300m 군집(Super-Station)으로 묶었을 때의 동시 결품률 ===")
c_inv = inv.groupby(['ts', 'cluster']).agg(
    stock_sum=('stock_mean', 'sum'),
    cluster_empty=('empty_frac', 'min'),
    rent_sum=('rent', 'sum')
).reset_index()

top_c = c_inv.groupby('cluster').agg(
    total_rent=('rent_sum', 'sum'),
    cluster_empty_hours=('cluster_empty', lambda x: (x >= 0.5).sum()),
    zero_stock_hours=('stock_sum', lambda x: (x <= 1.0).sum()),
    total_hours=('rent_sum', 'count'),
    mean_cluster_stock=('stock_sum', 'mean')
).reset_index()
top_c['cluster_empty_pct'] = top_c['cluster_empty_hours'] / top_c['total_hours'] * 100
top_c['zero_stock_pct'] = top_c['zero_stock_hours'] / top_c['total_hours'] * 100

top_c_sorted = top_c.sort_values('total_rent', ascending=False).head(15)
for _, r in top_c_sorted.iterrows():
    c_name = df_map[df_map['cluster_id'] == r['cluster']]['station_id'].iloc[0]
    st_name = st_to_name.get(c_name, '')
    print(f"[{r['cluster']}] {st_name[:18]:18s} | 월대여: {r['total_rent']:4.0f}건 | 군집평균재고: {r['mean_cluster_stock']:4.1f}대 | 군집품절비율: {r['cluster_empty_pct']:4.1f}% | 군집완전고갈: {r['zero_stock_pct']:4.1f}%")
