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

st_to_name = nodes.set_index('station_id')['name'].to_dict()
node_to_st = nodes.set_index('node')['station_id'].to_dict()
st_to_cluster = df_map.set_index('station_id')['cluster_id'].to_dict()
anc = df_map[df_map['is_anchor']==True].set_index('cluster_id')['station_id'].to_dict()

def evaluate_dispatch(curr_time, next_time, month_name, multiplier, buffer_val, truck_count, truck_cap):
    curr = inv[inv['ts'] == curr_time].copy()
    nxt = inv[inv['ts'] == next_time].copy()
    curr['st_id'] = curr['node'].map(node_to_st)
    nxt['st_id'] = nxt['node'].map(node_to_st)
    curr['cluster'] = curr['st_id'].map(st_to_cluster)
    
    m = pd.merge(curr, nxt[['st_id', 'rent']], on='st_id', suffixes=('', '_next'))
    m['rent_next'] = m['rent_next'].fillna(0.0)
    
    c_agg = m.groupby('cluster').agg(
        stock=('stock_mean', 'sum'),
        rent_curr=('rent', 'sum'),
        ret_curr=('ret', 'sum'),
        rent_next=('rent_next', 'sum'),
        zero_stations=('stock_mean', lambda s: (s == 0).sum())
    ).reset_index()
    
    # Q85 모델 예측 수요
    c_agg['q85_pred'] = np.maximum(2.0, c_agg['rent_curr'] * multiplier + buffer_val)
    
    # 수급 차이: 부족(Deficit)과 잉여(Surplus)
    c_agg['deficit'] = np.maximum(0.0, c_agg['q85_pred'] - c_agg['stock'])
    c_agg['surplus'] = np.maximum(0.0, c_agg['stock'] - (c_agg['q85_pred'] + 3.0))
    
    TOTAL_DISPATCH = truck_count * truck_cap
    
    # 1. Pickup: 잉여 재고가 가장 많은 군집
    pickups = c_agg.sort_values('surplus', ascending=False)
    p_list = []
    p_collected = 0
    for _, r in pickups.iterrows():
        amt = min(r['surplus'], TOTAL_DISPATCH - p_collected)
        if amt >= 4:
            st = anc.get(r['cluster'], '')
            p_list.append({
                'cluster': r['cluster'],
                'name': st_to_name.get(st, st),
                'stock': int(r['stock']),
                'amt': int(amt)
            })
            p_collected += amt
        if p_collected >= TOTAL_DISPATCH or len(p_list) >= 5:
            break
            
    # 2. Dropoff: Q85 결품 위험(Deficit)이 가장 큰 군집
    dropoffs = c_agg.sort_values('deficit', ascending=False)
    d_list = []
    d_allocated = 0
    
    # 군집별 보충 수량 기록
    cluster_boost = {}
    for _, r in dropoffs.iterrows():
        amt = min(r['deficit'], TOTAL_DISPATCH - d_allocated)
        if amt >= 4:
            st = anc.get(r['cluster'], '')
            d_list.append({
                'cluster': r['cluster'],
                'name': st_to_name.get(st, st),
                'stock': int(r['stock']),
                'pred_rent': int(r['q85_pred']),
                'amt': int(amt),
                'zero_st': int(r['zero_stations'])
            })
            d_allocated += amt
            cluster_boost[r['cluster']] = amt
        if d_allocated >= TOTAL_DISPATCH or len(d_list) >= 5:
            break
            
    # 효과 비교: 배차 전 vs 후
    # 대여소별로 배차 수량 분배
    m['boost'] = 0.0
    for cid, b_amt in cluster_boost.items():
        mask = m['cluster'] == cid
        c_weight = m.loc[mask, 'rent'] + 0.5
        c_norm = c_weight / c_weight.sum()
        m.loc[mask, 'boost'] = np.round(b_amt * c_norm)
        
    m['stock_after'] = m['stock_mean'] + m['boost']
    
    m['unmet_before'] = np.maximum(0.0, m['rent_next'] - m['stock_mean'])
    m['unmet_after'] = np.maximum(0.0, m['rent_next'] - m['stock_after'])
    
    total_unmet_before = m['unmet_before'].sum()
    total_unmet_after = m['unmet_after'].sum()
    saved = total_unmet_before - total_unmet_after
    
    zero_before = (m['stock_mean'] == 0.0).sum()
    zero_after = (m['stock_after'] == 0.0).sum()
    
    # 상위 5대 공급 거점만의 집중 효과
    top_cids = list(cluster_boost.keys())
    m_top = m[m['cluster'].isin(top_cids)]
    top_unmet_before = m_top['unmet_before'].sum()
    top_unmet_after = m_top['unmet_after'].sum()
    
    print("="*85)
    print(f" ■ {month_name} ({curr_time} -> {next_time}) | 트럭: {truck_count}대 (총 배차: {int(d_allocated)}대)")
    print("="*85)
    print(" [1. Top 회수 거점 (어디서 얼만큼 싣는가? - Pickup)]")
    for idx, p in enumerate(p_list, 1):
        print(f"  {idx}위: [{p['cluster']}] {p['name'][:24]} | 현재고 {p['stock']}대 -> -{p['amt']}대 회수")
        
    print("\n [2. Top 공급 거점 (어디에 얼만큼 내리는가? - Drop-off)]")
    for idx, d in enumerate(d_list, 1):
        print(f"  {idx}위: [{d['cluster']}] {d['name'][:24]} | 현재고 {d['stock']}대 (예측수요 {d['pred_rent']}대) -> +{d['amt']}대 투하 (0대소: {d['zero_st']}개)")
        
    print("\n [3. 실제 효과 (Before vs After)]")
    print(f"  - 0대 완전 품절 대여소: {zero_before}개소 -> {zero_after}개소 ({zero_before - zero_after}개소 즉시 0대 탈출)")
    print(f"  - 결품으로 발길 돌린 시민: {int(total_unmet_before)}명 -> {int(total_unmet_after)}명 ({int(saved)}명 즉시 구제!)")
    print(f"  - 집중 공급 5대 거점 결품 피해 감소율: {top_unmet_before:.0f}명 -> {top_unmet_after:.0f}명 ({((top_unmet_before - top_unmet_after)/max(1, top_unmet_before))*100:.1f}% 결품 방어 성공!)")
    print()

evaluate_dispatch('2025-09-15 17:00:00', '2025-09-15 18:00:00', '9월 가을 성수기', 1.45, 2.5, 12, 18)
evaluate_dispatch('2025-03-17 17:00:00', '2025-03-17 18:00:00', '3월 봄 개강기', 1.30, 1.8, 8, 18)
evaluate_dispatch('2025-02-17 17:00:00', '2025-02-17 18:00:00', '2월 겨울 극비수기', 1.15, 1.0, 5, 18)
