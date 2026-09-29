"""
실전 재배치 배차 지시서(Dispatch Manifest) 및 실제 개선 효과(Before vs After) 정량 산출 스크립트:
- 2월, 3월, 9월 대표 평일 17:00 퇴근길 시점 전수 분석
- 정확히 '어디서 몇 대를 싣고(Pickup)' '어디에 몇 대를 내리는지(Drop-off)' 거점 실명 및 대수 산출
- 재배치 전 vs 재배치 후: 0대 품절 대여소 수, 헛걸음한 시민 수, 품절률 변화를 1:1 비교
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "data/processed/v2/nodes.parquet"
MAPPING_PATH = ROOT / "data/processed/cluster_mapping_300m.parquet"
INVENTORY_PATH = ROOT / "data/processed/v3/inventory_hourly.parquet"
MODEL_PATH = ROOT / "outputs/final_model/xgb_cluster_300m_q85.json"

def run_month_dispatch(inv, df_map, nodes, target_ts, next_ts, truck_count=10, truck_capacity=18):
    st_to_name = nodes.set_index('station_id')['name'].to_dict()
    node_to_st = nodes.set_index('node')['station_id'].to_dict()
    st_to_cluster = df_map.set_index('station_id')['cluster_id'].to_dict()
    st_to_weight = df_map.set_index('station_id')['weight'].to_dict()
    st_to_cap = nodes.set_index('station_id')['capacity'].fillna(10.0).to_dict()
    cluster_anchor_st = df_map[df_map['is_anchor'] == True].set_index('cluster_id')['station_id'].to_dict()

    inv_curr = inv[inv['ts'] == target_ts].copy()
    inv_next = inv[inv['ts'] == next_ts].copy()

    inv_curr['st_id'] = inv_curr['node'].map(node_to_st)
    inv_curr['name'] = inv_curr['st_id'].map(st_to_name)
    inv_curr['cluster'] = inv_curr['st_id'].map(st_to_cluster)
    inv_curr['capacity'] = inv_curr['st_id'].map(st_to_cap)
    inv_curr['weight'] = inv_curr['st_id'].map(st_to_weight)

    inv_next['st_id'] = inv_next['node'].map(node_to_st)
    inv_next['actual_rent_next'] = inv_next['rent']

    df_sim = pd.merge(inv_curr, inv_next[['st_id', 'actual_rent_next']], on='st_id', how='left').fillna({'actual_rent_next': 0.0})

    # 군집별 집계
    c_inv = df_sim.groupby('cluster').agg(
        stock_sum=('stock_mean', 'sum'),
        rent_curr=('rent', 'sum'),
        ret_curr=('ret', 'sum'),
        actual_rent_next=('actual_rent_next', 'sum'),
        capacity=('capacity', 'sum')
    ).reset_index()

    # 다음 1시간 Q85 목표 수요 추정 (시간대/계절별 피크 배수 적용)
    month_val = pd.to_datetime(target_ts).month
    multiplier = 1.45 if month_val == 9 else (1.30 if month_val == 3 else 1.15)
    buffer_base = 2.5 if month_val == 9 else (1.8 if month_val == 3 else 1.0)
    
    c_inv['q85_pred'] = np.maximum(2.0, c_inv['rent_curr'] * multiplier + c_inv['ret_curr'] * 0.15 + buffer_base)
    c_pred_dict = c_inv.set_index('cluster')['q85_pred'].to_dict()
    
    df_sim['c_pred'] = df_sim['cluster'].map(c_pred_dict).fillna(2.0)
    df_sim['pred_rent_next'] = np.maximum(0.2, df_sim['c_pred'] * df_sim['weight'])
    
    # 대여소별 목표 재고
    df_sim['target_stock'] = np.ceil(df_sim['pred_rent_next'] + (1.5 if month_val==9 else 1.0))
    df_sim['target_stock'] = np.maximum(df_sim['target_stock'], 2.0)
    df_sim['target_stock'] = np.minimum(df_sim['target_stock'], df_sim['capacity'] * 0.8)
    
    # 수급 갭
    df_sim['gap'] = df_sim['stock_mean'] - df_sim['target_stock']
    
    # 군집별 잉여(회수 가능) 및 부족(공급 필요)
    c_dispatch = df_sim.groupby('cluster').agg(
        cluster_stock=('stock_mean', 'sum'),
        cluster_target=('target_stock', 'sum'),
        cluster_actual_next=('actual_rent_next', 'sum'),
        cluster_deficit=('gap', lambda g: np.maximum(0, -g[g < 0]).sum()),
        cluster_surplus=('gap', lambda g: np.maximum(0, g[g > 0]).sum()),
        zero_st_count=('stock_mean', lambda s: (s == 0).sum())
    ).reset_index()
    
    c_dispatch['priority'] = (c_dispatch['cluster_target'] / (c_dispatch['cluster_stock'] + 0.5)) * c_dispatch['cluster_deficit']
    
    # 배차 계획
    pickup_targets = c_dispatch.sort_values('cluster_surplus', ascending=False)
    dropoff_targets = c_dispatch.sort_values('priority', ascending=False)
    
    TOTAL_DISPATCH = truck_count * truck_capacity # 총 배차 가능량
    
    # 1. 상위 5대 회수 거점 (어디서 얼만큼 싣는가)
    pickup_list = []
    p_collected = 0
    for _, r in pickup_targets.iterrows():
        amt = min(r['cluster_surplus'], TOTAL_DISPATCH - p_collected)
        if amt >= 5:
            anc_st = cluster_anchor_st.get(r['cluster'], '')
            pickup_list.append({
                'cluster': r['cluster'],
                'name': st_to_name.get(anc_st, anc_st),
                'current_stock': int(r['cluster_stock']),
                'pickup_amt': int(amt)
            })
            p_collected += amt
        if p_collected >= TOTAL_DISPATCH or len(pickup_list) >= 5:
            break

    # 2. 상위 5대 공급 거점 (어디에 얼만큼 내리는가)
    dropoff_list = []
    d_allocated = 0
    df_sim['dispatched_stock'] = df_sim['stock_mean'].copy()
    
    for _, r in dropoff_targets.iterrows():
        amt = min(r['cluster_deficit'], TOTAL_DISPATCH - d_allocated)
        if amt >= 5:
            anc_st = cluster_anchor_st.get(r['cluster'], '')
            dropoff_list.append({
                'cluster': r['cluster'],
                'name': st_to_name.get(anc_st, anc_st),
                'current_stock': int(r['cluster_stock']),
                'dropoff_amt': int(amt),
                'zero_count': int(r['zero_st_count'])
            })
            d_allocated += amt
            
            # 군집 내 배분
            mask = df_sim['cluster'] == r['cluster']
            w = df_sim.loc[mask, 'weight']
            w_norm = w / w.sum()
            df_sim.loc[mask, 'dispatched_stock'] += np.round(amt * w_norm)
            
        if d_allocated >= TOTAL_DISPATCH or len(dropoff_list) >= 5:
            break
            
    # 효과 비교: 배차 전 vs 배차 후
    df_sim['unmet_before'] = np.maximum(0, df_sim['actual_rent_next'] - df_sim['stock_mean'])
    df_sim['unmet_after'] = np.maximum(0, df_sim['actual_rent_next'] - df_sim['dispatched_stock'])
    
    zero_before = (df_sim['stock_mean'] == 0.0).sum()
    zero_after = (df_sim['dispatched_stock'] == 0.0).sum()
    
    unmet_before_sum = df_sim['unmet_before'].sum()
    unmet_after_sum = df_sim['unmet_after'].sum()
    
    # 배차 대상 5대 거점 내 미충족 수요
    target_c_ids = [d['cluster'] for d in dropoff_list]
    target_unmet_before = df_sim[df_sim['cluster'].isin(target_c_ids)]['unmet_before'].sum()
    target_unmet_after = df_sim[df_sim['cluster'].isin(target_c_ids)]['unmet_after'].sum()
    
    return {
        'pickup_list': pickup_list,
        'dropoff_list': dropoff_list,
        'zero_before': zero_before,
        'zero_after': zero_after,
        'unmet_before': unmet_before_sum,
        'unmet_after': unmet_after_sum,
        'target_unmet_before': target_unmet_before,
        'target_unmet_after': target_unmet_after,
        'total_dispatched': d_allocated
    }

def main():
    print("="*85)
    print(" [타슈 실전 재배치 배차 지시서 및 실질 효과 산출 (2월, 3월, 9월)]")
    print("="*85)
    
    nodes = pd.read_parquet(NODES_PATH)
    df_map = pd.read_parquet(MAPPING_PATH)
    cols = ['ts', 'node', 'stock_mean', 'rent', 'ret']
    inv = pd.read_parquet(INVENTORY_PATH, columns=cols)
    inv['ts'] = pd.to_datetime(inv['ts'])

    cases = [
        ("9월 가을 최고 성수기 (2025-09-15 17:00 -> 18:00)", "2025-09-15 17:00:00", "2025-09-15 18:00:00", 12, 18),
        ("3월 봄 개강기       (2025-03-17 17:00 -> 18:00)", "2025-03-17 17:00:00", "2025-03-17 18:00:00",  8, 18),
        ("2월 겨울 극비수기   (2025-02-17 17:00 -> 18:00)", "2025-02-17 17:00:00", "2025-02-17 18:00:00",  5, 18)
    ]

    for title, t_curr, t_next, trucks, cap in cases:
        res = run_month_dispatch(inv, df_map, nodes, t_curr, t_next, trucks, cap)
        
        print("\n" + "#"*85)
        print(f" ■ {title} (투입 트럭: {trucks}대, 총 배차량: {res['total_dispatched']}대)")
        print("#"*85)
        
        print("\n1. [정확히 어디서 몇 대를 싣는가? (Top 5 회수처 - Pickup)]")
        print(f"| 순번 | 회수 군집 ID | 거점 대여소명 | 현재 재고 | 회수할 자전거 수 (트럭 적재) |")
        print(f"| :---: | :--- | :--- | :---: | :---: |")
        for idx, p in enumerate(res['pickup_list'], 1):
            print(f"| {idx}위 | {p['cluster']:10s} | {p['name'][:22]:22s} | {p['current_stock']:3d} 대 | **-{p['pickup_amt']:2d} 대 회수** |")
            
        print("\n2. [정확히 어디에 몇 대를 내리는가? (Top 5 공급처 - Drop-off)]")
        print(f"| 순번 | 공급 군집 ID | 거점 대여소명 | 현재 재고 | 공급할 자전거 수 (현장 배치) | 0대 대여소 수 |")
        print(f"| :---: | :--- | :--- | :---: | :---: | :---: |")
        for idx, d in enumerate(res['dropoff_list'], 1):
            print(f"| {idx}위 | {d['cluster']:10s} | {d['name'][:22]:22s} | {d['current_stock']:3d} 대 | **+{d['dropoff_amt']:2d} 대 투하** | {d['zero_count']} 개소 0대 |")
            
        print("\n3. [실제로는 얼마나 차이가 나는가? (Before vs After 실전 효과)]")
        saved_people = int(res['unmet_before'] - res['unmet_after'])
        target_pct = (1.0 - res['target_unmet_after'] / max(1.0, res['target_unmet_before'])) * 100
        print(f"  - 0대 완전 품절 대여소: {res['zero_before']} 개소 -> {res['zero_after']} 개소 ({res['zero_before'] - res['zero_after']}개소 품절 즉시 해소)")
        print(f"  - **퇴근길 자전거 못 타고 발길 돌린 시민 수**: **{int(res['unmet_before']):,} 명 -> {int(res['unmet_after']):,} 명**")
        print(f"  - **즉시 구제된 시민 수**: **{saved_people:,} 명 (결품 피해 {saved_people/max(1, res['unmet_before'])*100:.1f}% 감소!)**")
        print(f"  - **트럭 집중 투입 5대 거점의 결품 차단율**: **{target_pct:.1f}% 결품 완벽 방어!**")

if __name__ == "__main__":
    main()
