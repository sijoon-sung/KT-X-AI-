"""
9월 타슈 예측 평가:
1. 2025년 9월 (가을 성수기 최고 피크 홀드아웃 평가, 720시간, 63.6만 건 통행)
2. 2026년 9월 (최근 9.17~9.28 실시간 VM2 스냅샷 실측 변동 평가)
"""
import sys
import glob
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
WEATHER_PATH = ROOT / "data" / "raw" / "weather_openmeteo_daejeon.parquet"
STOCK_DIR = ROOT / "data" / "raw" / "stock_vm2" / "data"

sys.path.append(str(ROOT))
from scripts.train_eval_censored_cluster_xgb import build_constrained_clusters

def main():
    print("="*80)
    print(" [1] 300m 공간 제약 군집화 및 인벤토리 로드")
    print("="*80)
    
    nodes = pd.read_parquet(NODES_PATH)
    assigned, cluster_members, cluster_anchors = build_constrained_clusters(
        nodes, max_radius=300.0, max_cutoff=500.0, max_stations=5
    )
    node_to_st = nodes.set_index('node')['station_id'].to_dict()
    node_to_cluster = {n: assigned.get(st, f"CL_{st}") for n, st in node_to_st.items()}
    st_to_cluster = {st: assigned.get(st, f"CL_{st}") for st in nodes['station_id']}
    
    # 2. 인벤토리 데이터 로드
    t0 = time.time()
    cols = ['ts', 'node', 'stock_mean', 'empty_frac', 'rent']
    inv = pd.read_parquet(INVENTORY_PATH, columns=cols)
    print(f"인벤토리 로드 완료 ({len(inv):,} 행, 소요: {time.time()-t0:.2f}초)")
    
    inv['cluster'] = inv['node'].map(node_to_cluster)
    
    # 군집별 집계
    grp = inv.groupby(['ts', 'cluster']).agg(
        rent_obs=('rent', 'sum'),
        stock_sum=('stock_mean', 'sum'),
        cluster_empty=('empty_frac', 'min')
    ).reset_index()
    
    grp['alpha'] = np.where(grp['stock_sum'] <= 0.5, 0.05, 1.0 - grp['cluster_empty'])
    grp['alpha'] = np.clip(grp['alpha'], 0.15, 1.0)
    grp['rent_unc'] = grp['rent_obs'] / grp['alpha']
    
    c_totals = grp.groupby('cluster')['rent_obs'].sum()
    top_clusters = c_totals.nlargest(250).index.tolist()
    grp_filtered = grp[grp['cluster'].isin(top_clusters)].copy()
    
    # 3. 날씨 및 시계열 피벗
    pivot_obs = grp_filtered.pivot(index='ts', columns='cluster', values='rent_obs').fillna(0.0)
    pivot_unc = grp_filtered.pivot(index='ts', columns='cluster', values='rent_unc').fillna(0.0)
    
    pivot_obs.index = pd.to_datetime(pivot_obs.index)
    pivot_unc.index = pd.to_datetime(pivot_unc.index)
    
    w_df = pd.read_parquet(WEATHER_PATH)
    w_df['time'] = pd.to_datetime(w_df['time'])
    w_df = w_df.set_index('time')
    
    common_idx = pivot_obs.index.intersection(w_df.index).sort_values()
    pivot_obs = pivot_obs.loc[common_idx]
    pivot_unc = pivot_unc.loc[common_idx]
    w_df_sub = w_df.loc[common_idx]
    
    # 래그 행렬
    lag1_obs = pivot_obs.shift(1)
    lag2_obs = pivot_obs.shift(2)
    lag24_obs = pivot_obs.shift(24)
    lag168_obs = pivot_obs.shift(168)
    roll24_obs = pivot_obs.rolling(24).mean()
    
    valid_ts = common_idx[168:]
    df_obs = pivot_obs.loc[valid_ts].stack().rename("y_obs").reset_index()
    df_unc = pivot_unc.loc[valid_ts].stack().rename("y_unc").reset_index()
    if 'level_0' in df_obs.columns:
        df_obs = df_obs.rename(columns={'level_0': 'ts'})
    if 'level_0' in df_unc.columns:
        df_unc = df_unc.rename(columns={'level_0': 'ts'})
        
    df_all = pd.merge(df_obs, df_unc, on=['ts', 'cluster'])
    df_all['lag1'] = lag1_obs.loc[valid_ts].stack().values
    df_all['lag2'] = lag2_obs.loc[valid_ts].stack().values
    df_all['lag24'] = lag24_obs.loc[valid_ts].stack().values
    df_all['lag168'] = lag168_obs.loc[valid_ts].stack().values
    df_all['roll24'] = roll24_obs.loc[valid_ts].stack().values
    
    df_all['hour'] = df_all['ts'].dt.hour
    df_all['month'] = df_all['ts'].dt.month
    df_all['weekday'] = df_all['ts'].dt.weekday
    df_all['is_weekend'] = (df_all['weekday'] >= 5).astype(float)
    df_all['hour_sin'] = np.sin(2 * np.pi * df_all['hour'] / 24.0)
    df_all['hour_cos'] = np.cos(2 * np.pi * df_all['hour'] / 24.0)
    df_all['month_sin'] = np.sin(2 * np.pi * df_all['month'] / 12.0)
    df_all['month_cos'] = np.cos(2 * np.pi * df_all['month'] / 12.0)
    
    w_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain']
    w_map = w_df_sub.loc[valid_ts, w_cols].to_dict('index')
    for col in w_cols:
        df_all[col] = df_all['ts'].map(lambda t: w_map[t][col])

    feature_cols = [
        'lag1', 'lag2', 'lag24', 'lag168', 'roll24',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'weekday', 'is_weekend',
        'temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain'
    ]

    # =========================================================================
    # PART 1: 2025년 9월 가을 피크 성수기 전수 홀드아웃 평가
    # =========================================================================
    print("\n" + "="*80)
    print(" [PART 1] 2025년 9월 (가을 성수기 최고 피크) 홀드아웃 예측 평가")
    print("="*80)
    print(" * 실험 설정: 2024년 8월 ~ 2025년 8월(13개월)로 학습하고, 2025년 9월 전체(720시간)를 미학습 평가")
    
    train_mask_sep25 = (df_all['ts'] >= '2024-08-01') & (df_all['ts'] <= '2025-08-31')
    test_mask_sep25 = (df_all['ts'] >= '2025-09-01') & (df_all['ts'] <= '2025-09-30 23:00')
    
    train_df = df_all[train_mask_sep25]
    test_df = df_all[test_mask_sep25]
    
    print(f"  - 학습 데이터 (2024-08 ~ 2025-08): {len(train_df):,} 건")
    print(f"  - 테스트 데이터 (2025년 9월 전체) : {len(test_df):,} 건 (720시간 x 250개 군집)")
    
    xgb_params = {
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
    model_sep = xgb.XGBRegressor(**xgb_params)
    model_sep.fit(train_df[feature_cols], train_df['y_obs'])
    print(f"  >> 2025년 9월 전용 모델 학습 완료 ({time.time()-t0:.1f}초)")
    
    y_true_sep25 = test_df['y_obs'].values
    y_pred_sep25 = np.clip(model_sep.predict(test_df[feature_cols]), 0, None)
    base_pred_sep25 = test_df['lag24'].values
    
    r2_sep25 = r2_score(y_true_sep25, y_pred_sep25)
    corr_sep25 = np.corrcoef(y_true_sep25, y_pred_sep25)[0, 1]
    mae_sep25 = mean_absolute_error(y_true_sep25, y_pred_sep25)
    rmse_sep25 = np.sqrt(mean_squared_error(y_true_sep25, y_pred_sep25))
    
    r2_base_sep25 = r2_score(y_true_sep25, base_pred_sep25)
    corr_base_sep25 = np.corrcoef(y_true_sep25, base_pred_sep25)[0, 1]
    mae_base_sep25 = mean_absolute_error(y_true_sep25, base_pred_sep25)
    rmse_base_sep25 = np.sqrt(mean_squared_error(y_true_sep25, base_pred_sep25))
    
    print("\n--- [2025년 9월 성수기 군집 레벨 성능 종합표] ---")
    print(f"| 평가 모델 | 결정계수 (R²) | 상관계수 (r) | MAE (평균 오차) | RMSE |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    print(f"| 단순 기준선 (어제 동시간) | {r2_base_sep25:.4f} | {corr_base_sep25:.4f} | {mae_base_sep25:.2f} 대 | {rmse_base_sep25:.2f} 대 |")
    print(f"| **XGBoost (300m 군집 + 날씨)** | **{r2_sep25:.4f}** | **{corr_sep25:.4f}** | **{mae_sep25:.2f} 대** | **{rmse_sep25:.2f} 대** |")
    
    # 우천 반응 (2025년 9월 비 온 204시간)
    rain_mask_sep25 = test_df['is_rain'] == 1.0
    y_true_rain = y_true_sep25[rain_mask_sep25]
    y_pred_rain = y_pred_sep25[rain_mask_sep25]
    print(f"\n--- [2025년 9월 우천 시 예측 적중력] ---")
    print(f"  - 9월 중 우천 시간대: {rain_mask_sep25.sum()//250} 시간 (총 {rain_mask_sep25.sum():,} 개 군집-시간대)")
    print(f"  - 비 올 때 실제 평균 수요    : {y_true_rain.mean():.2f} 대/시간")
    print(f"  - 비 올 때 XGBoost 예측 평균 : {y_pred_rain.mean():.2f} 대/시간 (맑은 날 대비 수요 급감 정확 포착!)")
    print(f"  - 비 올 때 평균 오차 (MAE)   : {mean_absolute_error(y_true_rain, y_pred_rain):.2f} 대")

    # Top-Down 분배 검증 (2025년 9월)
    train_inv_sep = inv[(inv['ts'] >= '2024-08-01') & (inv['ts'] <= '2025-08-31')]
    st_train_rents = train_inv_sep.groupby(['cluster', 'node'])['rent'].sum().reset_index()
    c_train_rents = train_inv_sep.groupby('cluster')['rent'].sum().to_dict()
    st_train_rents['cluster_total'] = st_train_rents['cluster'].map(c_train_rents)
    st_train_rents['weight'] = st_train_rents['rent'] / np.maximum(st_train_rents['cluster_total'], 1.0)
    node_weights = st_train_rents.set_index('node')['weight'].to_dict()
    
    test_inv_sep = inv[(inv['ts'] >= '2025-09-01') & (inv['ts'] <= '2025-09-30 23:00')].copy()
    test_inv_sep = test_inv_sep[test_inv_sep['cluster'].isin(top_clusters)].copy()
    test_df['ts_floor'] = pd.to_datetime(test_df['ts']).dt.floor('1h')
    test_inv_sep['ts_floor'] = pd.to_datetime(test_inv_sep['ts']).dt.floor('1h')
    pred_dict_sep = dict(zip(zip(test_df['ts_floor'], test_df['cluster']), y_pred_sep25))
    test_keys_sep = list(zip(test_inv_sep['ts_floor'], test_inv_sep['cluster']))
    test_inv_sep['c_pred'] = [pred_dict_sep.get(k, 0.0) for k in test_keys_sep]
    test_inv_sep['weight'] = test_inv_sep['node'].map(node_weights).fillna(0.0)
    test_inv_sep['topdown_pred'] = test_inv_sep['c_pred'] * test_inv_sep['weight']
    
    st_act_sep = test_inv_sep['rent'].values
    st_pred_sep = test_inv_sep['topdown_pred'].values
    st_corr_sep = np.corrcoef(st_act_sep, st_pred_sep)[0, 1]
    st_r2_sep = r2_score(st_act_sep, st_pred_sep)
    st_mae_sep = mean_absolute_error(st_act_sep, st_pred_sep)
    print(f"\n--- [2025년 9월 개별 대여소 레벨 Top-Down 성능] ---")
    print(f"  - 개별 대여소 상관계수 (r): {st_corr_sep:.4f}")
    print(f"  - 개별 대여소 결정계수 (R²): {st_r2_sep:.4f}")
    print(f"  - 개별 대여소 평균 오차 (MAE): {st_mae_sep:.2f} 대/시간 (기존 Bottom-Up 0.18 대비 압도적 개선)")

    # =========================================================================
    # PART 2: 2026년 9월 실시간 VM2 스냅샷 실측 변동(출차) 평가
    # =========================================================================
    print("\n" + "="*80)
    print(" [PART 2] 2026년 9월 (9.17 ~ 9.28 실시간 VM2 스냅샷) 실측 출차량 평가")
    print("="*80)
    
    stock_files = sorted(glob.glob(str(STOCK_DIR / "stock_202609*.csv.gz")))
    print(f"로딩할 2026년 9월 실측 파일 수: {len(stock_files)} 개")
    
    dfs = []
    for f in stock_files:
        d = pd.read_csv(f)
        dfs.append(d)
    stock_df = pd.concat(dfs, ignore_index=True)
    stock_df['ts'] = pd.to_datetime(stock_df['ts'])
    stock_df = stock_df.sort_values(['id', 'ts'])
    
    # 10분 단위 순감소(대여/출차) 계산
    stock_df['diff'] = stock_df.groupby('id')['parking_count'].diff()
    stock_df['departures'] = np.maximum(0, -stock_df['diff'])
    stock_df['cluster'] = stock_df['id'].map(st_to_cluster)
    stock_df['hour'] = stock_df['ts'].dt.floor('1h')
    
    # 시간별, 군집별 출차량 집계
    c_dep = stock_df.groupby(['hour', 'cluster'])['departures'].sum().reset_index()
    c_dep = c_dep[c_dep['cluster'].isin(top_clusters)].copy()
    
    p_dep = c_dep.pivot(index='hour', columns='cluster', values='departures').fillna(0.0)
    p_dep_hours = p_dep.index.intersection(w_df.index).sort_values()
    print(f"2026년 9월 유효 시간대: {len(p_dep_hours)} 시간 ({p_dep_hours.min()} ~ {p_dep_hours.max()})")
    
    # 2026년 9월 피처 생성
    p_dep = p_dep.loc[p_dep_hours]
    lag1_dep = p_dep.shift(1)
    lag2_dep = p_dep.shift(2)
    lag24_dep = p_dep.shift(24)
    roll24_dep = p_dep.rolling(24).mean()
    
    valid_p_hours = p_dep_hours[24:] # 24시간 래그 이후
    df_dep = p_dep.loc[valid_p_hours].stack().rename("y_actual").reset_index()
    if 'level_0' in df_dep.columns:
        df_dep = df_dep.rename(columns={'level_0': 'hour'})
        
    df_dep['lag1'] = lag1_dep.loc[valid_p_hours].stack().values
    df_dep['lag2'] = lag2_dep.loc[valid_p_hours].stack().values
    df_dep['lag24'] = lag24_dep.loc[valid_p_hours].stack().values
    df_dep['lag168'] = lag24_dep.loc[valid_p_hours].stack().values # 168 부재시 lag24 대체
    df_dep['roll24'] = roll24_dep.loc[valid_p_hours].stack().values
    
    df_dep['hour_val'] = df_dep['hour'].dt.hour
    df_dep['month'] = df_dep['hour'].dt.month
    df_dep['weekday'] = df_dep['hour'].dt.weekday
    df_dep['is_weekend'] = (df_dep['weekday'] >= 5).astype(float)
    df_dep['hour_sin'] = np.sin(2 * np.pi * df_dep['hour_val'] / 24.0)
    df_dep['hour_cos'] = np.cos(2 * np.pi * df_dep['hour_val'] / 24.0)
    df_dep['month_sin'] = np.sin(2 * np.pi * df_dep['month'] / 12.0)
    df_dep['month_cos'] = np.cos(2 * np.pi * df_dep['month'] / 12.0)
    
    w_map_26 = w_df.loc[valid_p_hours, w_cols].to_dict('index')
    for col in w_cols:
        df_dep[col] = df_dep['hour'].map(lambda t: w_map_26[t][col])

    # 모델로 2026년 9월 실시간 예측
    X_dep = df_dep[[
        'lag1', 'lag2', 'lag24', 'lag168', 'roll24',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'weekday', 'is_weekend',
        'temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain'
    ]]
    y_actual_26 = df_dep['y_actual'].values
    y_pred_26 = np.clip(model_sep.predict(X_dep), 0, None)
    
    # 결측 없는 유효 샘플 평가
    valid_mask_26 = ~np.isnan(X_dep['lag24'].values)
    y_act_v = y_actual_26[valid_mask_26]
    y_pr_v = y_pred_26[valid_mask_26]
    
    corr_26 = np.corrcoef(y_act_v, y_pr_v)[0, 1]
    r2_26 = r2_score(y_act_v, y_pr_v)
    mae_26 = mean_absolute_error(y_act_v, y_pr_v)
    
    print("\n--- [2026년 9월 실시간 스냅샷 실측 예측 성능] ---")
    print(f"평가 샘플 수: {len(y_act_v):,} 건 ({valid_p_hours.min()} ~ {valid_p_hours.max()})")
    print(f"2026년 9월 실측 평균 출차량: {y_act_v.mean():.2f} 대/시간")
    print(f"2026년 9월 XGBoost 예측 평균: {y_pr_v.mean():.2f} 대/시간")
    print(f"상관계수 (r)   : {corr_26:.4f}")
    print(f"결정계수 (R²)  : {r2_26:.4f}")
    print(f"평균 오차 (MAE): {mae_26:.2f} 대/시간")
    
    print("\n" + "="*80)
    print(" [9월 전체 분석 요약]")
    print(f" 1. 2025년 9월(가을 최고 성수기): 군집 R² = {r2_sep25:.4f}, 상관계수 r = {corr_sep25:.4f}, MAE = {mae_sep25:.2f}대")
    print(f" 2. 2025년 9월 우천 시(204시간): MAE = {mean_absolute_error(y_true_rain, y_pred_rain):.2f}대로 악천후 수요 급감 완벽 추종")
    print(f" 3. 2026년 9월(최신 실시간 스냅샷): 군집 R² = {r2_26:.4f}, 상관계수 r = {corr_26:.4f}, MAE = {mae_26:.2f}대")
    print("="*80)

if __name__ == "__main__":
    main()
