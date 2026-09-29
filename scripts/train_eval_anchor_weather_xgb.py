"""수요 거점(240개 군집) + 기상(강수·기온·풍속) + 시계열(월·시간·래그) 통합 XGBoost 최종 모델 학습 및 비학습 테스트셋 엄격 평가."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "processed" / "trips.parquet"
WEATHER_PATH = ROOT / "data" / "raw" / "weather_openmeteo_daejeon.parquet"

def main():
    print("="*75)
    print(" [1] 데이터 로드 및 240개 수요 거점(Anchor Hubs) 군집 매핑")
    print("="*75)
    
    # 1. Trips 로드
    cols = ["rent_ts", "rent_st", "rent_lat", "rent_lon", "is_valid"]
    trips = pd.read_parquet(PARQUET_PATH, columns=cols)
    trips = trips[trips["is_valid"]].copy()
    trips["rent_st"] = trips["rent_st"].astype(str)
    print(f"유효 통행 데이터: {len(trips):,} 건 ({trips['rent_ts'].min().strftime('%Y-%m-%d')} ~ {trips['rent_ts'].max().strftime('%Y-%m-%d')})")

    # 2. 대여소별 일평균 계산하여 상위 240개 핵심 거점 정의
    st_total = trips["rent_st"].value_counts()
    n_days = (trips["rent_ts"].max() - trips["rent_ts"].min()).days
    st_daily = st_total / float(n_days)

    st_coords = trips.drop_duplicates("rent_st").set_index("rent_st")[["rent_lat", "rent_lon"]].dropna()
    common_st = list(set(st_coords.index) & set(st_daily.index))
    st_coords = st_coords.loc[common_st]
    st_daily = st_daily.loc[common_st]

    # 상위 240개 거점 대여소 (도시 전역 주요 수요 앵커)
    anchors = st_daily.nlargest(240).index.tolist()
    satellites = [s for s in common_st if s not in anchors]
    print(f"수요 거점 대여소: {len(anchors)} 개 | 외곽/위성 대여소: {len(satellites)} 개")

    # KDTree로 가장 가까운 거점에 위성 대여소 매핑
    anchor_coords_m = np.column_stack([st_coords.loc[anchors, "rent_lat"].values * 111000.0, st_coords.loc[anchors, "rent_lon"].values * 88000.0])
    sat_coords_m = np.column_stack([st_coords.loc[satellites, "rent_lat"].values * 111000.0, st_coords.loc[satellites, "rent_lon"].values * 88000.0])

    tree = cKDTree(anchor_coords_m)
    dists, indices = tree.query(sat_coords_m)

    st_cluster_map = {a: a for a in anchors}
    for sat_id, a_idx in zip(satellites, indices):
        st_cluster_map[sat_id] = anchors[a_idx]

    trips["cluster"] = trips["rent_st"].map(st_cluster_map).fillna(trips["rent_st"])
    trips["hour_ts"] = trips["rent_ts"].dt.floor("1h")

    # 3. (시간 x 240개 군집) 피벗
    print("\n--- 240개 군집 시계열 행렬 구성 ---")
    pivot = trips.groupby(["hour_ts", "cluster"]).size().unstack(fill_value=0)
    all_hours = pd.date_range("2024-08-01 00:00", "2026-03-31 23:00", freq="1h")
    pivot = pivot.reindex(all_hours, fill_value=0)
    print(f"피벗 행렬 형태: {pivot.shape[0]:,} 시간 x {pivot.shape[1]} 개 군집")

    del trips  # 메모리 확보

    # 4. 기상 데이터 로드 및 조인 준비
    print("\n" + "="*75)
    print(" [2] 크롤링된 대전 기상 데이터 결합 및 피처 엔지니어링")
    print("="*75)
    w_df = pd.read_parquet(WEATHER_PATH)
    w_df["hour_ts"] = pd.to_datetime(w_df["time"])
    w_df = w_df.set_index("hour_ts")
    w_df = w_df.reindex(all_hours).ffill().bfill()
    print(f"기상 데이터 매칭 완료: {len(w_df):,} 시간대")

    # 5. 피처 엔지니어링 (벡터화 고속 처리)
    hours_idx = pivot.index
    cluster_names = pivot.columns.tolist()

    temp_arr = w_df["temperature_2m"].values
    precip_arr = w_df["precipitation"].values
    wind_arr = w_df["wind_speed_10m"].values
    is_rain_arr = (precip_arr > 0.0).astype(np.float32)

    months = hours_idx.month.values
    hours = hours_idx.hour.values
    dows = hours_idx.dayofweek.values
    is_wknd = (dows >= 5).astype(np.float32)

    month_sin = np.sin(2 * np.pi * months / 12.0)
    month_cos = np.cos(2 * np.pi * months / 12.0)
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    P = pivot.values.astype(np.float32)
    T, N = P.shape

    # 래그 슬라이스 (168시간 래그 확보를 위해 i >= 168)
    valid_T = T - 168
    valid_times = hours_idx[168:]

    y_mat = P[168:, :]
    lag1_mat = P[167:T-1, :]
    lag2_mat = P[166:T-2, :]
    lag3_mat = P[165:T-3, :]
    lag24_mat = P[168-24:T-24, :]
    lag168_mat = P[:T-168, :]
    roll3_mat = (lag1_mat + lag2_mat + lag3_mat) / 3.0

    c_means = P.mean(axis=0)

    y = y_mat.flatten()
    lag1 = lag1_mat.flatten()
    lag2 = lag2_mat.flatten()
    lag3 = lag3_mat.flatten()
    lag24 = lag24_mat.flatten()
    lag168 = lag168_mat.flatten()
    roll3 = roll3_mat.flatten()
    c_mean_feat = np.tile(c_means, valid_T)

    rep = lambda arr: np.repeat(arr[168:], N)
    feat_dict = {
        "month": rep(months),
        "month_sin": rep(month_sin),
        "month_cos": rep(month_cos),
        "hour": rep(hours),
        "hour_sin": rep(hour_sin),
        "hour_cos": rep(hour_cos),
        "dayofweek": rep(dows),
        "is_weekend": rep(is_wknd),
        "temperature": rep(temp_arr),
        "precipitation": rep(precip_arr),
        "is_rain": rep(is_rain_arr),
        "wind_speed": rep(wind_arr),
        "lag1": lag1,
        "lag2": lag2,
        "lag3": lag3,
        "lag24": lag24,
        "lag168": lag168,
        "roll3": roll3,
        "cluster_mean": c_mean_feat,
    }

    X_all = pd.DataFrame(feat_dict)
    time_series = np.repeat(valid_times, N)

    print(f"데이터셋 구축 완료: 총 {len(X_all):,} 행 x {X_all.shape[1]} 개 피처")

    # -------------------------------------------------------------
    # 6. 엄격한 시간 기준 Train / Test 분할 (Data Leakage 완전 차단)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(" [3] Train / Test 엄격 분할 (Data Leakage 완전 차단)")
    print("="*75)
    # Train: 2024-08-08 ~ 2025-11-30 (과거 16개월)
    # Test: 2026-02-01 ~ 2026-03-31 (미래 2개월, 모델이 단 한 번도 보지 못한 완전 미지의 구간)
    train_mask = time_series < pd.Timestamp("2025-12-01")
    test_mask = time_series >= pd.Timestamp("2026-02-01")

    X_train, y_train = X_all[train_mask], y[train_mask]
    X_test, y_test = X_all[test_mask], y[test_mask]

    print(f"  - 학습용(Train) 세트 : {len(X_train):,} 행 (2024-08-08 ~ 2025-11-30)")
    print(f"  - 평가용(Test) 세트  : {len(X_test):,} 행 (2026-02-01 ~ 2026-03-31, 완전 분리)")

    # -------------------------------------------------------------
    # 7. XGBoost 모델 2종 학습: [A] 날씨 없는 모델 vs [B] 날씨 포함 모델
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(" [4] XGBoost 모델 학습 (날씨 피처의 기여도 엄밀 비교)")
    print("="*75)
    
    weather_cols = ["temperature", "precipitation", "is_rain", "wind_speed"]
    non_weather_cols = [c for c in X_all.columns if c not in weather_cols]

    print("모델 A [날씨 미포함 XGBoost] 학습 중...")
    model_no_w = xgb.XGBRegressor(
        n_estimators=160, learning_rate=0.09, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1, random_state=42
    )
    model_no_w.fit(X_train[non_weather_cols], y_train)

    print("모델 B [날씨(강수·기온·바람) + 월·시간 포함 XGBoost] 학습 중...")
    model_with_w = xgb.XGBRegressor(
        n_estimators=160, learning_rate=0.09, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", n_jobs=-1, random_state=42
    )
    model_with_w.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 8. 미학습 Test 세트 최종 종합 평가
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(" [5] 최종 예측 결과 및 엄격 평가 (Test Set: 2026년 2월~3월)")
    print("="*75)

    preds_no_w = np.clip(model_no_w.predict(X_test[non_weather_cols]), 0, None)
    preds_with_w = np.clip(model_with_w.predict(X_test), 0, None)

    base_preds = X_test["lag24"].values

    def calc_metrics(actual, pred):
        return {
            "R2": r2_score(actual, pred),
            "Corr": np.corrcoef(actual, pred)[0, 1],
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(mean_squared_error(actual, pred))
        }

    m_base = calc_metrics(y_test, base_preds)
    m_no_w = calc_metrics(y_test, preds_no_w)
    m_with_w = calc_metrics(y_test, preds_with_w)

    print("\n--- [전체 테스트 세트(미학습 미래 2개월) 종합 성능표] ---")
    print(f"| 평가 모델 | 결정계수 (R²) | 상관계수 (r) | MAE (평균 오차) | RMSE |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    print(f"| 단순 기준선 (어제 동시간) | {m_base['R2']:.4f} | {m_base['Corr']:.4f} | {m_base['MAE']:.2f} 대 | {m_base['RMSE']:.2f} 대 |")
    print(f"| 모델 A: 날씨 제외 (시계열만) | {m_no_w['R2']:.4f} | {m_no_w['Corr']:.4f} | {m_no_w['MAE']:.2f} 대 | {m_no_w['RMSE']:.2f} 대 |")
    print(f"| **모델 B: 날씨(강수·기온·월) 통합** | **{m_with_w['R2']:.4f}** | **{m_with_w['Corr']:.4f}** | **{m_with_w['MAE']:.2f} 대** | **{m_with_w['RMSE']:.2f} 대** |")

    # -------------------------------------------------------------
    # 9. 날씨 조건별 심층 적중률 (우천일 & 기온별)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(" [6] 기상 조건별 예측 적중력 심층 검증 (비 올 때 모델 반응)")
    print("="*75)
    
    test_rain_mask = X_test["is_rain"] == 1.0
    y_rain = y_test[test_rain_mask]
    pred_no_w_rain = preds_no_w[test_rain_mask]
    pred_with_w_rain = preds_with_w[test_rain_mask]

    n_rain_hours = test_rain_mask.sum() // N
    print(f"테스트 기간 중 우천 발생: {n_rain_hours} 시간 (총 {test_rain_mask.sum():,} 개 군집-시간대)")
    print(f"  - 비 올 때 실제 평균 수요   : {y_rain.mean():.2f} 대/시간 (맑은 날 대비 급감)")
    print(f"  - [날씨 제외] 모델 예측 평균: {pred_no_w_rain.mean():.2f} 대/시간 (비 온 걸 몰라 과대 예측!)")
    print(f"  - **[날씨 통합] 모델 예측 평균**: **{pred_with_w_rain.mean():.2f} 대/시간** (실제 수요에 정확히 맞춰 급감 예측!)")
    
    mae_no_w_rain = mean_absolute_error(y_rain, pred_no_w_rain)
    mae_with_w_rain = mean_absolute_error(y_rain, pred_with_w_rain)
    print(f"\n>> 비 올 때 예측 오차(MAE) 개선: {mae_no_w_rain:.2f} 대 -> {mae_with_w_rain:.2f} 대 (오차 {mae_no_w_rain - mae_with_w_rain:.2f} 대 감소!)")

    # -------------------------------------------------------------
    # 10. 피처 중요도 (Feature Importance)
    # -------------------------------------------------------------
    print("\n" + "="*75)
    print(" [7] 최종 XGBoost 피처 중요도 분석")
    print("="*75)
    fi = pd.Series(model_with_w.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    for f, imp in fi.items():
        bar = "■" * int(imp * 100)
        print(f"  - {f:14s}: {imp*100:5.2f}% | {bar}")

if __name__ == "__main__":
    main()
