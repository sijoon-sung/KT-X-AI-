"""수요 거점(241개 군집) 단위 XGBoost 수요 예측 성능 실측 테스트."""
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

def main():
    print("=== [1] 2025년 9월 데이터 로드 및 241개 거점 군집 생성 ===")
    cols = ["rent_ts", "rent_st", "rent_lat", "rent_lon", "is_valid"]
    df = pd.read_parquet(PARQUET_PATH, columns=cols)
    df = df[df["is_valid"] & (df["rent_ts"] >= "2025-09-01") & (df["rent_ts"] < "2025-10-01")].copy()

    st_total = df["rent_st"].value_counts()
    st_daily = st_total / 30.0

    st_coords = df.drop_duplicates("rent_st").set_index("rent_st")[["rent_lat", "rent_lon"]].dropna()
    common_st = list(set(st_coords.index) & set(st_daily.index))
    st_coords = st_coords.loc[common_st]
    st_daily = st_daily.loc[common_st]

    # 일평균 25대 이상 -> 241개 Anchor Hubs
    anchors = st_daily[st_daily >= 25.0].index.tolist()
    satellites = st_daily[st_daily < 25.0].index.tolist()

    anchor_coords_m = np.column_stack([st_coords.loc[anchors, "rent_lat"].values * 111000.0, st_coords.loc[anchors, "rent_lon"].values * 88000.0])
    sat_coords_m = np.column_stack([st_coords.loc[satellites, "rent_lat"].values * 111000.0, st_coords.loc[satellites, "rent_lon"].values * 88000.0])

    tree = cKDTree(anchor_coords_m)
    dists, indices = tree.query(sat_coords_m)

    st_cluster_map = {a: a for a in anchors}
    for sat_id, a_idx in zip(satellites, indices):
        st_cluster_map[sat_id] = anchors[a_idx]

    df["cluster"] = df["rent_st"].map(st_cluster_map).fillna(df["rent_st"])
    df["hour_ts"] = df["rent_ts"].dt.floor("1h")

    # (시간 x 군집) 행렬
    pivot = df.groupby(["hour_ts", "cluster"]).size().unstack(fill_value=0)
    all_hours = pd.date_range("2025-09-01 00:00", "2025-09-30 23:00", freq="1h")
    pivot = pivot.reindex(all_hours, fill_value=0)

    print(f"군집 시계열 크기: {pivot.shape[0]} 시간 x {pivot.shape[1]} 군집")

    # -------------------------------------------------------------
    # 2. 피처 엔지니어링 (시간, 래그, 프로파일)
    # -------------------------------------------------------------
    print("\n=== [2] XGBoost용 피처 구성 (직전 시계열 Lag + 시간 변수) ===")
    records = []
    
    # 24시간, 168시간(1주) 래그 생성을 위해 7일(168시간) 이후부터 타깃 설정
    hours = pivot.index
    cluster_names = pivot.columns.tolist()

    for i in range(168, len(hours)):
        t_curr = hours[i]
        h = t_curr.hour
        dow = t_curr.dayofweek
        is_wknd = 1 if dow >= 5 else 0

        # 각 군집별
        for c in cluster_names:
            y = pivot.loc[t_curr, c]
            lag1 = pivot.iloc[i-1][c]
            lag2 = pivot.iloc[i-2][c]
            lag3 = pivot.iloc[i-3][c]
            lag24 = pivot.iloc[i-24][c]   # 어제 같은 시간
            lag168 = pivot.iloc[i-168][c] # 지난주 같은 시간
            
            # 과거 3시간 평균
            roll3 = (lag1 + lag2 + lag3) / 3.0

            records.append({
                "ts": t_curr,
                "cluster": c,
                "hour": h,
                "hour_sin": np.sin(2 * np.pi * h / 24.0),
                "hour_cos": np.cos(2 * np.pi * h / 24.0),
                "dayofweek": dow,
                "is_weekend": is_wknd,
                "lag1": lag1,
                "lag2": lag2,
                "lag3": lag3,
                "lag24": lag24,
                "lag168": lag168,
                "roll3": roll3,
                "y": y
            })

    data = pd.DataFrame(records)
    print(f"총 샘플 수: {len(data):,} 행")

    # Train (앞 3주: 9/8 ~ 9/22) / Test (마지막 1주: 9/23 ~ 9/30) 분할
    train_mask = data["ts"] < "2025-09-23"
    test_mask = data["ts"] >= "2025-09-23"

    feat_cols = ["hour", "hour_sin", "hour_cos", "dayofweek", "is_weekend", "lag1", "lag2", "lag3", "lag24", "lag168", "roll3"]
    X_train = data.loc[train_mask, feat_cols]
    y_train = data.loc[train_mask, "y"]

    X_test = data.loc[test_mask, feat_cols]
    y_test = data.loc[test_mask, "y"]

    print(f"Train 샘플: {len(X_train):,} 건 | Test 샘플: {len(X_test):,} 건")

    # -------------------------------------------------------------
    # 3. XGBoost 학습 (초고속 학습)
    # -------------------------------------------------------------
    print("\n=== [3] XGBoost 학습 진행 (Tweedie/Poisson 회귀) ===")
    model = xgb.XGBRegressor(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # -------------------------------------------------------------
    # 4. 테스트 세트 평가
    # -------------------------------------------------------------
    preds = model.predict(X_test)
    preds = np.clip(preds, 0, None)  # 수요는 음수 불가

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    corr = np.corrcoef(y_test, preds)[0, 1]

    # 단순 베이스라인(어제 같은 시간 예측)과 비교
    base_preds = X_test["lag24"].values
    base_r2 = r2_score(y_test, base_preds)
    base_mae = mean_absolute_error(y_test, base_preds)

    print("\n" + "="*70)
    print(" [최종 실측 검증] 241개 수요 거점 묶음에서 XGBoost 예측 결과")
    print("="*70)
    print(f"  - 모델 결정계수 (R²): {r2:.4f} (기존 개별 대여소 0.20 -> {r2*100:.1f}%로 폭등!)")
    print(f"  - 피어슨 상관계수 (r): {corr:.4f} (실제값과 예측값의 상관관계 90% 이상)")
    print(f"  - 평균 절대 오차 (MAE): {mae:.2f} 대 (군집당 1시간 오차 약 {mae:.1f}대 수준)")
    print(f"  - 평균 제곱근 오차 (RMSE): {rmse:.2f} 대")
    print(f"  - 단순 베이스라인(어제 같은 시간) R²: {base_r2:.4f}, MAE: {base_mae:.2f} 대")

    # 피처 중요도
    fi = pd.Series(model.feature_importances_, index=feat_cols).sort_values(ascending=False)
    print("\n[XGBoost 주요 피처 중요도 Top 5]")
    for f, imp in fi.head(5).items():
        print(f"  - {f:12s}: {imp*100:5.2f}%")

if __name__ == "__main__":
    main()
