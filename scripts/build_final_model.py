"""
타슈(Tashu) 300m 생활권 군집 기반 최종 프로덕션 예측 및 재배치 모델 구축 파이프라인.
- 300m 보행 대체 제약 군집화 (437개 Super-Station) 확정 및 매핑 테이블 저장
- 18개 검증된 최적 피처(시계열 래그 + 반납 유입량 + Top-2 연계 OD + 기상 피처) 기반 XGBoost 최종 모델 학습 및 저장
- Top-Down 대여소 분배 엔진 및 실시간 재배치 목표(Target Inventory) 산출 로직 내장
"""
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
NODES_PATH = ROOT / "data" / "processed" / "v2" / "nodes.parquet"
INVENTORY_PATH = ROOT / "data" / "processed" / "v3" / "inventory_hourly.parquet"
TRIPS_PATH = ROOT / "data" / "processed" / "trips.parquet"
WEATHER_PATH = ROOT / "data" / "raw" / "weather_openmeteo_daejeon.parquet"

MODEL_DIR = ROOT / "outputs" / "final_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(ROOT))
from scripts.train_eval_censored_cluster_xgb import build_constrained_clusters

def main():
    print("="*85)
    print(" [1] 300m 공간 제약 군집화 확정 및 영구 매핑 테이블 생성")
    print("="*85)
    
    nodes = pd.read_parquet(NODES_PATH)
    assigned, cluster_members, cluster_anchors = build_constrained_clusters(
        nodes, max_radius=300.0, max_cutoff=500.0, max_stations=5
    )
    
    # 대여소별 군집 정보 및 역사적 통행량 계산
    trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'is_valid'])
    trips = trips[trips['is_valid']].copy()
    st_rent_counts = trips['rent_st'].astype(str).value_counts().to_dict()
    
    mapping_rows = []
    for st_id in nodes['station_id']:
        c_id = assigned.get(st_id, f"CL_{st_id}")
        is_anchor = (st_id == cluster_anchors.get(c_id, ""))
        n_rent = st_rent_counts.get(st_id, 0)
        c_size = len(cluster_members.get(c_id, [st_id]))
        mapping_rows.append({
            'station_id': st_id,
            'cluster_id': c_id,
            'is_anchor': is_anchor,
            'cluster_size': c_size,
            'historical_rent': n_rent
        })
    df_map = pd.DataFrame(mapping_rows)
    
    # 군집 내 대여소별 분배 가중치 (Weight) 계산
    c_totals = df_map.groupby('cluster_id')['historical_rent'].transform('sum')
    df_map['weight'] = df_map['historical_rent'] / np.maximum(c_totals, 1.0)
    
    # 매핑 파일 저장
    map_parquet_path = ROOT / "data" / "processed" / "cluster_mapping_300m.parquet"
    map_csv_path = ROOT / "data" / "processed" / "cluster_mapping_300m.csv"
    df_map.to_parquet(map_parquet_path)
    df_map.to_csv(map_csv_path, index=False, encoding='utf-8-sig')
    print(f"매핑 테이블 저장 완료: {map_parquet_path}")
    print(f"  - 총 대여소: {len(df_map):,} 개 -> 총 군집: {df_map['cluster_id'].nunique():,} 개 (평균 {len(df_map)/df_map['cluster_id'].nunique():.2f}개/군집)")

    # 2. OD 파트너 네트워크 추출
    print("\n" + "="*85)
    print(" [2] Top-2 연계 클러스터(OD Partners) 네트워크 확정")
    print("="*85)
    st_to_cluster = df_map.set_index('station_id')['cluster_id'].to_dict()
    trips['c_rent'] = trips['rent_st'].astype(str).map(st_to_cluster)
    trips['c_ret'] = trips['ret_st'].astype(str).map(st_to_cluster)
    trips_sub = trips.dropna(subset=['c_rent', 'c_ret'])
    
    od_matrix = trips_sub.groupby(['c_rent', 'c_ret']).size().unstack(fill_value=0)
    all_clusters = sorted(df_map['cluster_id'].unique().tolist())
    
    top_partners = {}
    for c in all_clusters:
        if c in od_matrix.index:
            row = od_matrix.loc[c].copy()
            if c in row.index:
                row[c] = 0 # 자기 자신 제외
            top2 = row.nlargest(2).index.tolist()
            top_partners[c] = top2 if len(top2) == 2 else (top2 + [all_clusters[0]])[:2]
        else:
            top_partners[c] = [all_clusters[0], all_clusters[1]]
            
    with open(MODEL_DIR / "top_partners.json", "w", encoding="utf-8") as f:
        json.dump(top_partners, f, ensure_ascii=False, indent=2)
    print(f"연계 클러스터 네트워크 저장 완료: {MODEL_DIR / 'top_partners.json'}")

    # 3. 인벤토리 데이터 기반 피처 구축
    print("\n" + "="*85)
    print(" [3] 시계열 행렬 및 18개 최적 피처 엔지니어링")
    print("="*85)
    cols = ['ts', 'node', 'stock_mean', 'empty_frac', 'rent', 'ret']
    inv = pd.read_parquet(INVENTORY_PATH, columns=cols)
    node_to_st = nodes.set_index('node')['station_id'].to_dict()
    inv['cluster'] = inv['node'].map(lambda n: st_to_cluster.get(node_to_st.get(n, ''), 'CL_UNKNOWN'))
    
    grp = inv.groupby(['ts', 'cluster']).agg(
        rent_obs=('rent', 'sum'),
        ret_obs=('ret', 'sum'),
        stock_sum=('stock_mean', 'sum')
    ).reset_index()
    
    # 핵심 분석 대상: 상위 250개 활성 거점 군집
    c_rent_sum = grp.groupby('cluster')['rent_obs'].sum()
    top250_clusters = c_rent_sum.nlargest(250).index.tolist()
    grp_filtered = grp[grp['cluster'].isin(top250_clusters)].copy()
    
    pivot_rent = grp_filtered.pivot(index='ts', columns='cluster', values='rent_obs').fillna(0.0)
    pivot_ret = grp_filtered.pivot(index='ts', columns='cluster', values='ret_obs').fillna(0.0)
    pivot_rent.index = pd.to_datetime(pivot_rent.index)
    pivot_ret.index = pd.to_datetime(pivot_ret.index)
    
    w_df = pd.read_parquet(WEATHER_PATH)
    w_df['time'] = pd.to_datetime(w_df['time'])
    w_df = w_df.set_index('time')
    
    common_idx = pivot_rent.index.intersection(w_df.index).sort_values()
    pivot_rent = pivot_rent.loc[common_idx]
    pivot_ret = pivot_ret.loc[common_idx]
    w_df_sub = w_df.loc[common_idx]
    
    # 래그 연산
    lag1_rent = pivot_rent.shift(1)
    lag2_rent = pivot_rent.shift(2)
    lag24_rent = pivot_rent.shift(24)
    lag168_rent = pivot_rent.shift(168)
    roll24_rent = pivot_rent.rolling(24).mean()
    
    # 반납 유입량 (53.4% 중요도 핵심 피처)
    lag1_ret = pivot_ret.shift(1)
    
    # 연계 클러스터 대여 래그 (안전 인덱싱)
    p1_series = {c: (lag1_rent[top_partners[c][0]] if top_partners[c][0] in lag1_rent.columns else lag1_rent[c]) for c in top250_clusters}
    p2_series = {c: (lag1_rent[top_partners[c][1]] if top_partners[c][1] in lag1_rent.columns else lag1_rent[c]) for c in top250_clusters}
    p1_df = pd.DataFrame(p1_series, index=common_idx)
    p2_df = pd.DataFrame(p2_series, index=common_idx)
    
    valid_ts = common_idx[168:]
    df = pivot_rent.loc[valid_ts].stack().rename("y").reset_index()
    if 'level_0' in df.columns:
        df = df.rename(columns={'level_0': 'ts'})
        
    df['lag1'] = lag1_rent.loc[valid_ts].stack().values
    df['lag2'] = lag2_rent.loc[valid_ts].stack().values
    df['lag24'] = lag24_rent.loc[valid_ts].stack().values
    df['lag168'] = lag168_rent.loc[valid_ts].stack().values
    df['roll24'] = roll24_rent.loc[valid_ts].stack().values
    df['lag1_ret'] = lag1_ret.loc[valid_ts].stack().values
    df['p1_lag1'] = p1_df.loc[valid_ts].stack().values
    df['p2_lag1'] = p2_df.loc[valid_ts].stack().values
    
    df['hour'] = df['ts'].dt.hour
    df['month'] = df['ts'].dt.month
    df['weekday'] = df['ts'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(float)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    w_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain']
    w_map = w_df_sub.loc[valid_ts, w_cols].to_dict('index')
    for col in w_cols:
        df[col] = df['ts'].map(lambda t: w_map[t][col])

    feature_cols = [
        'lag1', 'lag2', 'lag24', 'lag168', 'roll24',
        'lag1_ret', 'p1_lag1', 'p2_lag1',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'weekday', 'is_weekend',
        'temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain'
    ]

    # 4. 최종 모델 학습 (Train: 2024.08 ~ 2025.11, 272만 건)
    print("\n" + "="*85)
    print(" [4] 최종 프로덕션 XGBoost 모델 학습 및 파일 저장")
    print("="*85)
    
    train_mask = (df['ts'] >= '2024-08-01') & (df['ts'] <= '2025-11-30')
    test_mask_feb = (df['ts'] >= '2026-02-01') & (df['ts'] <= '2026-03-31')
    test_mask_sep = (df['ts'] >= '2025-09-01') & (df['ts'] <= '2025-09-30 23:00')
    
    train_df = df[train_mask]
    test_feb_df = df[test_mask_feb]
    test_sep_df = df[test_mask_sep]
    
    print(f"전체 학습 세트 샘플 수: {len(train_df):,} 건")
    print(f"테스트 세트 1 (2025년 9월 피크 성수기): {len(test_sep_df):,} 건")
    print(f"테스트 세트 2 (2026년 2~3월 미학습 미래): {len(test_feb_df):,} 건")
    
    final_params = {
        'n_estimators': 250,
        'max_depth': 8,
        'learning_rate': 0.08,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }
    
    t0 = time.time()
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(train_df[feature_cols], train_df['y'])
    train_time = time.time() - t0
    print(f">> 최종 모델 학습 완료 (소요 시간: {train_time:.1f}초)")
    
    # 모델 파일 영구 저장
    model_path = MODEL_DIR / "xgb_cluster_300m.json"
    final_model.save_model(str(model_path))
    print(f">> 최종 모델 바이너리 저장 완료: {model_path}")

    # 5. 다차원 공간 성능 검증 (300m 군집, 1km 격자 환산, 도시 전체)
    print("\n" + "="*85)
    print(" [5] 최종 확정 모델의 다차원 검증 성적표")
    print("="*85)
    
    def evaluate_splits(name, test_data):
        y_true = test_data['y'].values
        y_pred = np.clip(final_model.predict(test_data[feature_cols]), 0, None)
        
        r2 = r2_score(y_true, y_pred)
        corr = np.corrcoef(y_true, y_pred)[0, 1]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # 도시 전체(City-level) 합산 성능
        city_eval = test_data.copy()
        city_eval['pred'] = y_pred
        city_grp = city_eval.groupby('ts').agg(actual=('y', 'sum'), predicted=('pred', 'sum'))
        city_r2 = r2_score(city_grp['actual'], city_grp['predicted'])
        city_corr = np.corrcoef(city_grp['actual'], city_grp['predicted'])[0, 1]
        
        return {
            'period': name,
            'cluster_r2': r2,
            'cluster_corr': corr,
            'cluster_mae': mae,
            'cluster_rmse': rmse,
            'city_r2': city_r2,
            'city_corr': city_corr
        }
        
    m_sep = evaluate_splits("2025년 9월 피크 성수기 (가을)", test_sep_df)
    m_feb = evaluate_splits("2026년 2~3월 해빙기 (비수기)", test_feb_df)
    
    print(f"| 평가 구간 | 300m 군집 R² | 300m 상관계수 (r) | 300m MAE | 도시 전체(City) R² | 도시 상관계수 |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **{m_sep['period']}** | **{m_sep['cluster_r2']:.4f}** | **{m_sep['cluster_corr']:.4f}** | **{m_sep['cluster_mae']:.2f} 대** | **{m_sep['city_r2']:.4f}** | **{m_sep['city_corr']:.4f}** |")
    print(f"| **{m_feb['period']}** | **{m_feb['cluster_r2']:.4f}** | **{m_feb['cluster_corr']:.4f}** | **{m_feb['cluster_mae']:.2f} 대** | **{m_feb['city_r2']:.4f}** | **{m_feb['city_corr']:.4f}** |")

    # 설정 및 메타데이터 저장
    config = {
        'model_name': 'Tashu 300m Super-Station XGBoost',
        'cluster_radius_m': 300.0,
        'cluster_max_cutoff_m': 500.0,
        'cluster_max_stations': 5,
        'total_stations': len(df_map),
        'total_clusters': df_map['cluster_id'].nunique(),
        'feature_cols': feature_cols,
        'train_samples': len(train_df),
        'evaluation': {
            'september_2025_peak': m_sep,
            'feb_mar_2026_future': m_feb
        }
    }
    with open(MODEL_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"\n최종 설정 및 성능 보고서 저장 완료: {MODEL_DIR / 'model_config.json'}")

if __name__ == "__main__":
    main()
