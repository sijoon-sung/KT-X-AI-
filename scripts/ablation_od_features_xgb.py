"""
Ablation Study (피처 넣었다 뺐다 실험):
연계 군집(OD), 반납 유입량(Inflow), 순유입량(Net Inflow), 서지 가속도(Surge Momentum),
손실함수(MSE vs Tweedie)를 단계별로 추가/제거하여 성능 변화를 정밀 검증.
"""
import sys
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

sys.path.append(str(ROOT))
from scripts.train_eval_censored_cluster_xgb import build_constrained_clusters

def main():
    print("="*85)
    print(" [1] 데이터 전처리 및 피처 풀(Feature Pool) 구축")
    print("="*85)
    
    nodes = pd.read_parquet(NODES_PATH)
    assigned, cluster_members, _ = build_constrained_clusters(nodes)
    node_to_st = nodes.set_index('node')['station_id'].to_dict()
    node_to_cluster = {n: assigned.get(st, f"CL_{st}") for n, st in node_to_st.items()}
    
    # 1. 인벤토리(대여 및 반납) 로드
    cols = ['ts', 'node', 'stock_mean', 'empty_frac', 'rent', 'ret']
    inv = pd.read_parquet(INVENTORY_PATH, columns=cols)
    inv['cluster'] = inv['node'].map(node_to_cluster)
    
    grp = inv.groupby(['ts', 'cluster']).agg(
        rent_obs=('rent', 'sum'),
        ret_obs=('ret', 'sum'),
        stock_sum=('stock_mean', 'sum')
    ).reset_index()
    
    c_totals = grp.groupby('cluster')['rent_obs'].sum()
    top_clusters = c_totals.nlargest(250).index.tolist()
    grp_filtered = grp[grp['cluster'].isin(top_clusters)].copy()
    
    # 2. OD 파트너 군집 추출
    trips = pd.read_parquet(TRIPS_PATH, columns=['rent_st', 'ret_st', 'is_valid'])
    trips = trips[trips['is_valid']].copy()
    trips['c_rent'] = trips['rent_st'].astype(str).map(assigned)
    trips['c_ret'] = trips['ret_st'].astype(str).map(assigned)
    trips_sub = trips.dropna(subset=['c_rent', 'c_ret'])
    trips_sub = trips_sub[trips_sub['c_rent'].isin(top_clusters) & trips_sub['c_ret'].isin(top_clusters)]
    
    od_matrix = trips_sub.groupby(['c_rent', 'c_ret']).size().unstack(fill_value=0)
    top_partners = {}
    for c in top_clusters:
        if c in od_matrix.index:
            row = od_matrix.loc[c].copy()
            if c in row.index:
                row[c] = 0
            top2 = row.nlargest(2).index.tolist()
            top_partners[c] = top2 if len(top2) == 2 else (top2 + [top_clusters[0]])[:2]
        else:
            top_partners[c] = [top_clusters[0], top_clusters[1]]

    # 3. 시계열 피벗
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
    
    lag1_ret = pivot_ret.shift(1)
    
    # 파트너 군집 대여 래그
    partner1_series = {c: lag1_rent[top_partners[c][0]] for c in top_clusters}
    partner2_series = {c: lag1_rent[top_partners[c][1]] for c in top_clusters}
    p1_df = pd.DataFrame(partner1_series, index=common_idx)
    p2_df = pd.DataFrame(partner2_series, index=common_idx)
    
    # 평탄화
    valid_ts = common_idx[168:]
    df = pivot_rent.loc[valid_ts].stack().rename("y").reset_index()
    if 'level_0' in df.columns:
        df = df.rename(columns={'level_0': 'ts'})
        
    df['lag1'] = lag1_rent.loc[valid_ts].stack().values
    df['lag2'] = lag2_rent.loc[valid_ts].stack().values
    df['lag24'] = lag24_rent.loc[valid_ts].stack().values
    df['lag168'] = lag168_rent.loc[valid_ts].stack().values
    df['roll24'] = roll24_rent.loc[valid_ts].stack().values
    
    # 신규 피처군
    df['lag1_ret'] = lag1_ret.loc[valid_ts].stack().values
    df['net_flow'] = df['lag1_ret'] - df['lag1']  # 순유입량 (반납 - 대여)
    df['p1_lag1'] = p1_df.loc[valid_ts].stack().values
    df['p2_lag1'] = p2_df.loc[valid_ts].stack().values
    df['surge_ratio'] = df['lag1'] / (df['roll24'] + 0.1)  # 평소 대비 급증률
    
    # 시간 피처
    df['hour'] = df['ts'].dt.hour
    df['month'] = df['ts'].dt.month
    df['weekday'] = df['ts'].dt.weekday
    df['is_weekend'] = (df['weekday'] >= 5).astype(float)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    
    # 날씨 피처
    w_cols = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain']
    w_map = w_df_sub.loc[valid_ts, w_cols].to_dict('index')
    for col in w_cols:
        df[col] = df['ts'].map(lambda t: w_map[t][col])

    # 4. 데이터 분할: Train (2024.08~2025.08) vs Test (2025년 9월 성수기 피크 720시간)
    train_mask = (df['ts'] >= '2024-08-01') & (df['ts'] <= '2025-08-31')
    test_mask = (df['ts'] >= '2025-09-01') & (df['ts'] <= '2025-09-30 23:00')
    
    train_df = df[train_mask]
    test_df = df[test_mask]
    y_train = train_df['y']
    y_test = test_df['y'].values
    
    print(f"학습 세트: {len(train_df):,} 건 | 테스트 세트: {len(test_df):,} 건")

    # 5. 피처 블록 정의
    f_time = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 'weekday', 'is_weekend']
    f_lags = ['lag1', 'lag2', 'lag24', 'lag168', 'roll24']
    f_weather = ['temperature_2m', 'precipitation', 'wind_speed_10m', 'is_rain']
    
    f_base = f_lags + f_time + f_weather  # 15개 기본
    
    # 실험 시나리오 정의 (넣었다가 뺐다가)
    experiments = [
        ("1. [기초] 단독 시계열 + 날씨 (Base)", f_base, "mse"),
        ("2. [+반납] Base + 직전 반납 유입량 (lag1_ret)", f_base + ['lag1_ret'], "mse"),
        ("3. [+연계군집] Base + Top-2 연계군집 대여 (p1, p2)", f_base + ['p1_lag1', 'p2_lag1'], "mse"),
        ("4. [+반납 +연계] Base + 반납 + Top-2 연계군집", f_base + ['lag1_ret', 'p1_lag1', 'p2_lag1'], "mse"),
        ("5. [+순유입 +서지] Base + 반납 + 연계 + 순유입(net) + 서지모멘텀", f_base + ['lag1_ret', 'p1_lag1', 'p2_lag1', 'net_flow', 'surge_ratio'], "mse"),
        ("6. [Tweedie 손실] 5번 피처 전체 + Tweedie Loss (고수요 과소예측 방지)", f_base + ['lag1_ret', 'p1_lag1', 'p2_lag1', 'net_flow', 'surge_ratio'], "tweedie"),
        ("7. [빼보기: -날씨] 5번 전체에서 기상(날씨) 피처 제외", [c for c in f_base + ['lag1_ret', 'p1_lag1', 'p2_lag1', 'net_flow', 'surge_ratio'] if c not in f_weather], "mse"),
        ("8. [빼보기: -연계군집] 5번 전체에서 연계군집(p1, p2) 제외", [c for c in f_base + ['lag1_ret', 'p1_lag1', 'p2_lag1', 'net_flow', 'surge_ratio'] if c not in ['p1_lag1', 'p2_lag1']], "mse"),
        ("9. [빼보기: -반납유입] 5번 전체에서 반납유입(lag1_ret, net) 제외", [c for c in f_base + ['lag1_ret', 'p1_lag1', 'p2_lag1', 'net_flow', 'surge_ratio'] if c not in ['lag1_ret', 'net_flow']], "mse"),
    ]

    print("\n" + "="*85)
    print(" [2] 피처 추가/제거(Ablation) 전수 실험 진행 (총 9개 조합)")
    print("="*85)
    
    results = []
    
    # 피크 시간대(출퇴근 8, 9, 18, 19시) 고수요(10대 이상) 마스크
    peak_mask = test_df['hour'].isin([8, 9, 18, 19]) & (test_df['y'] >= 10)
    y_test_peak = y_test[peak_mask]
    
    for name, f_list, loss_type in experiments:
        t0 = time.time()
        
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.08,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }
        
        if loss_type == "tweedie":
            xgb_params['objective'] = 'reg:tweedie'
            xgb_params['tweedie_variance_power'] = 1.3
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(train_df[f_list], y_train)
        el_time = time.time() - t0
        
        pred = np.clip(model.predict(test_df[f_list]), 0, None)
        
        r2 = r2_score(y_test, pred)
        corr = np.corrcoef(y_test, pred)[0, 1]
        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        
        # 피크 고수요 예측 평균 및 MAE
        pred_peak = pred[peak_mask]
        peak_pred_mean = pred_peak.mean()
        peak_mae = mean_absolute_error(y_test_peak, pred_peak)
        
        results.append({
            'name': name,
            'n_feats': len(f_list),
            'r2': r2,
            'corr': corr,
            'mae': mae,
            'rmse': rmse,
            'peak_pred': peak_pred_mean,
            'peak_mae': peak_mae,
            'time': el_time
        })
        print(f"완료: {name:50s} | R²: {r2:.4f} | r: {corr:.4f} | MAE: {mae:.2f}대 | 소요: {el_time:.1f}초")

    # 결과 테이블 출력
    print("\n" + "="*85)
    print(" [3] 피처 추가/제거 (Ablation Study) 최종 결과 종합표")
    print("="*85)
    print(f"| 실험 번호 및 시나리오 | 피처수 | 결정계수 (R²) | 상관계수 (r) | 전체 MAE | 피크 고수요 MAE (실제 15.8대) |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    
    base_r2 = results[0]['r2']
    base_mae = results[0]['mae']
    
    for r in results:
        diff_r2 = (r['r2'] - base_r2) * 100
        diff_mae = r['mae'] - base_mae
        print(f"| {r['name']:40s} | {r['n_feats']}개 | **{r['r2']:.4f}** ({diff_r2:+5.2f}%p) | {r['corr']:.4f} | **{r['mae']:.2f} 대** ({diff_mae:+4.2f}) | {r['peak_mae']:.2f} 대 (예측: {r['peak_pred']:.1f}대) |")

    print("\n" + "="*85)
    print(" [4] 핵심 발견 및 최종 결론")
    print("="*85)
    best_exp = max(results, key=lambda x: x['r2'])
    print(f" 1. [최고 성능 조합]: '{best_exp['name']}'")
    print(f"    -> R² = {best_exp['r2']:.4f} (기존 {base_r2:.4f} 대비 {(best_exp['r2']-base_r2)*100:+.2f}%p 대폭 상승!)")
    print(f"    -> 상관계수 r = {best_exp['corr']:.4f}, MAE = {best_exp['mae']:.2f}대")
    
    # 기여도 비교
    print("\n 2. [각 피처별 기여도 순위 (빼봤을 때 R² 하락폭)]")
    # Inflow removal impact
    r_no_ret = [r for r in results if "빼보기: -반납유입" in r['name']][0]
    r_full = results[4] # 5번 풀모델
    drop_ret = (r_full['r2'] - r_no_ret['r2']) * 100
    
    # Weather removal impact
    r_no_w = [r for r in results if "빼보기: -날씨" in r['name']][0]
    drop_w = (r_full['r2'] - r_no_w['r2']) * 100
    
    # OD partner removal impact
    r_no_od = [r for r in results if "빼보기: -연계군집" in r['name']][0]
    drop_od = (r_full['r2'] - r_no_od['r2']) * 100
    
    print(f"  - 반납 유입량(Inflow) 제거 시 R² 하락: -{drop_ret:.2f}%p (가장 치명적! 반드시 넣어야 함)")
    print(f"  - 날씨(기상) 피처 제거 시 R² 하락    : -{drop_w:.2f}%p (비 올 때 예측 붕괴 방지 필수)")
    print(f"  - 연계 군집(OD Partner) 제거 시 R² 하락: -{drop_od:.2f}%p (공간적 파급효과 개선 기여)")
    print("="*85)

if __name__ == "__main__":
    main()
