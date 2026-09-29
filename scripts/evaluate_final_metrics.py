import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("C:/Users/sijoo/Documents/tashu")
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
MAPPING_PATH = ROOT / "data/processed/cluster_mapping_300m.parquet"
INVENTORY_PATH = ROOT / "data/processed/v3/inventory_hourly.parquet"

nodes = pd.read_parquet(NODES_PATH)
df_map = pd.read_parquet(MAPPING_PATH)
inv = pd.read_parquet(INVENTORY_PATH, columns=['ts', 'node', 'stock_mean', 'rent', 'ret'])
inv['ts'] = pd.to_datetime(inv['ts'])

node_to_st = nodes.set_index('node')['station_id'].to_dict()
st_to_cluster = df_map.set_index('station_id')['cluster_id'].to_dict()

# 2025년 2월, 3월, 9월 평일 퇴근길(17:00 -> 18:00) 10일치 표본 추출하여 정밀 지표 산출
test_dates = [
    # 9월 가을 피크
    ('2025-09-15 17:00:00', '2025-09-15 18:00:00', '9월', 12, 1.45, 2.5),
    ('2025-09-16 17:00:00', '2025-09-16 18:00:00', '9월', 12, 1.45, 2.5),
    ('2025-09-17 17:00:00', '2025-09-17 18:00:00', '9월', 12, 1.45, 2.5),
    # 3월 봄 개강
    ('2025-03-17 17:00:00', '2025-03-17 18:00:00', '3월', 8, 1.30, 1.8),
    ('2025-03-18 17:00:00', '2025-03-18 18:00:00', '3월', 8, 1.30, 1.8),
    ('2025-03-19 17:00:00', '2025-03-19 18:00:00', '3월', 8, 1.30, 1.8),
    # 2월 겨울
    ('2025-02-17 17:00:00', '2025-02-17 18:00:00', '2월', 5, 1.15, 1.0),
    ('2025-02-18 17:00:00', '2025-02-18 18:00:00', '2월', 5, 1.15, 1.0),
    ('2025-02-19 17:00:00', '2025-02-19 18:00:00', '2월', 5, 1.15, 1.0)
]

results = []

for t_curr, t_next, m_label, trucks, mult, buf in test_dates:
    curr = inv[inv['ts'] == t_curr].copy()
    nxt = inv[inv['ts'] == t_next].copy()
    curr['st_id'] = curr['node'].map(node_to_st)
    nxt['st_id'] = nxt['node'].map(node_to_st)
    curr['cluster'] = curr['st_id'].map(st_to_cluster)
    
    m = pd.merge(curr, nxt[['st_id', 'rent']], on='st_id', suffixes=('', '_next'))
    m['rent_next'] = m['rent_next'].fillna(0.0)
    
    c_agg = m.groupby('cluster').agg(
        stock=('stock_mean', 'sum'),
        rent_curr=('rent', 'sum'),
        rent_next=('rent_next', 'sum'),
    ).reset_index()
    
    c_agg['q85_pred'] = np.maximum(2.0, c_agg['rent_curr'] * mult + buf)
    c_agg['deficit'] = np.maximum(0.0, c_agg['q85_pred'] - c_agg['stock'])
    
    # 1. 모델 Pinball Loss (Q85 예측 오차)
    # L_q(y, p) = max(q*(y - p), (q-1)*(y - p))
    diff = c_agg['rent_next'] - c_agg['q85_pred']
    pinball = np.maximum(0.85 * diff, (0.85 - 1.0) * diff).mean()
    coverage = (c_agg['rent_next'] <= c_agg['q85_pred']).mean()
    
    # 2. Q85 배차 실행
    TOTAL_DISPATCH = trucks * 18
    dropoffs = c_agg.sort_values('deficit', ascending=False)
    
    d_allocated = 0
    cluster_boost = {}
    for _, r in dropoffs.iterrows():
        amt = min(r['deficit'], TOTAL_DISPATCH - d_allocated)
        if amt >= 4:
            d_allocated += amt
            cluster_boost[r['cluster']] = amt
        if d_allocated >= TOTAL_DISPATCH or len(cluster_boost) >= 10:
            break
            
    # 배차 적용
    m['boost'] = 0.0
    for cid, b_amt in cluster_boost.items():
        mask = m['cluster'] == cid
        c_weight = m.loc[mask, 'rent'] + 0.5
        c_norm = c_weight / c_weight.sum()
        m.loc[mask, 'boost'] = np.round(b_amt * c_norm)
        
    m['stock_after'] = m['stock_mean'] + m['boost']
    
    # 지표 1: 결품 수요 (Unmet Demand)
    unmet_before = np.maximum(0.0, m['rent_next'] - m['stock_mean']).sum()
    unmet_after = np.maximum(0.0, m['rent_next'] - m['stock_after']).sum()
    saved_citizens = unmet_before - unmet_after
    service_gain = (saved_citizens / max(1.0, unmet_before)) * 100.0
    
    # 지표 2: 투입 자전거의 실제 회전율 (Turnover Rate)
    # 투입된 boost 대수 중 실제로 대여로 소진된 대수
    effective_consumed = 0.0
    total_boost = m['boost'].sum()
    for _, row in m[m['boost'] > 0].iterrows():
        # 재배치 자전거 중 실제 추가로 대여된 분량 = min(boost, max(0, rent_next - stock_mean))
        needed = max(0.0, row['rent_next'] - row['stock_mean'])
        used = min(row['boost'], needed)
        effective_consumed += used
    turnover_rate = (effective_consumed / max(1.0, total_boost)) * 100.0
    
    # 지표 3: 경제적 총비용 (Newsvendor Loss: 결품 5점 + 잉여 1점 + 배차 0.2점)
    cost_before = 5.0 * unmet_before + 1.0 * np.maximum(0.0, m['stock_mean'] - m['rent_next']).sum()
    cost_after = 5.0 * unmet_after + 1.0 * np.maximum(0.0, m['stock_after'] - m['rent_next']).sum() + 0.2 * total_boost
    cost_reduction = ((cost_before - cost_after) / cost_before) * 100.0
    
    results.append({
        'season': m_label,
        'ts': t_curr,
        'pinball': pinball,
        'coverage': coverage * 100.0,
        'unmet_before': unmet_before,
        'unmet_after': unmet_after,
        'saved': saved_citizens,
        'service_gain': service_gain,
        'dispatched': total_boost,
        'turnover_rate': turnover_rate,
        'cost_reduction': cost_reduction
    })

df_res = pd.DataFrame(results)
print("=== 종합 실전 평가 지표 (평균치) ===")
summary = df_res.groupby('season').agg({
    'coverage': 'mean',
    'pinball': 'mean',
    'unmet_before': 'mean',
    'unmet_after': 'mean',
    'saved': 'mean',
    'turnover_rate': 'mean',
    'cost_reduction': 'mean'
}).reindex(['9월', '3월', '2월'])

print(summary.to_string())
