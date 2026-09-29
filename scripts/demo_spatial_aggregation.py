"""대여소 공간 군집화(Aggregation)에 따른 푸아송 노이즈 감쇄 및 예측 신호 회복 실증 분석."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]
PARQUET_PATH = ROOT / "data" / "processed" / "trips.parquet"

def main():
    print("=== [1] 공간 군집화(Micro -> Meso -> Macro) 비교 분석 데이터 로드 ===")
    cols = ["rent_ts", "rent_st", "rent_dong", "rent_lat", "rent_lon", "is_valid"]
    # 빠른 분석을 위해 2025년 9월(가장 전형적인 성수기 1개월) 데이터로 정밀 비교
    df = pd.read_parquet(PARQUET_PATH, columns=cols)
    df = df[df["is_valid"] & (df["rent_ts"] >= "2025-09-01") & (df["rent_ts"] < "2025-10-01")].copy()
    print(f"2025년 9월 유효 대여: {len(df):,} 건")

    df["rent_st"] = df["rent_st"].astype(str)
    df["rent_dong"] = df["rent_dong"].astype(str)
    df["hour_ts"] = df["rent_ts"].dt.floor("1h")

    # 대여소별 좌표
    st_coords = df.drop_duplicates("rent_st").set_index("rent_st")[["rent_lat", "rent_lon"]].dropna()

    # -------------------------------------------------------------
    # 3단계 공간 집계 정의
    # -------------------------------------------------------------
    # Level 1: 개별 대여소 (1,300여 개)
    df["lvl1_id"] = df["rent_st"]

    # Level 2: 반경 400m 클러스터 (Super-station, 도보 5분 생활권)
    # 위도 1도 ≈ 111km, 경도 1도 ≈ 88km -> 미터 좌표 변환
    coords_m = np.column_stack([
        st_coords["rent_lat"].values * 111000.0,
        st_coords["rent_lon"].values * 88000.0
    ])
    tree = cKDTree(coords_m)
    
    # 400m 이내 이웃 클러스터링 (Greedy grouping)
    visited = set()
    st_to_cluster = {}
    cluster_idx = 0
    st_list = st_coords.index.tolist()

    for i, sid in enumerate(st_list):
        if sid in visited:
            continue
        neighbors = tree.query_ball_point(coords_m[i], r=400.0)
        c_name = f"Cluster_{cluster_idx:03d}"
        for n_idx in neighbors:
            nid = st_list[n_idx]
            if nid not in visited:
                st_to_cluster[nid] = c_name
                visited.add(nid)
        cluster_idx += 1

    df["lvl2_id"] = df["rent_st"].map(st_to_cluster).fillna(df["rent_st"])

    # Level 3: 행정동 (동 단위, ~70개)
    df["lvl3_id"] = df["rent_dong"]

    # -------------------------------------------------------------
    # 레벨별 시계열 집계 및 신호 대 잡음비(SNR), 푸아송 특성 측정
    # -------------------------------------------------------------
    levels = [
        ("Level 1: 개별 대여소 (Micro, 1,300+개)", "lvl1_id"),
        ("Level 2: 반경 400m 클러스터 (Meso, ~400개 슈퍼대여소)", "lvl2_id"),
        ("Level 3: 행정동 단위 (Macro, ~70개 동)", "lvl3_id"),
    ]

    all_hours = pd.date_range("2025-09-01 00:00", "2025-09-30 23:00", freq="1h")

    print("\n" + "="*80)
    print(" [실험 결과] 공간을 묶었을 때 노이즈(우연)가 어떻게 사라지고 신호가 살아나는가?")
    print("="*80)

    for label, col in levels:
        n_units = df[col].nunique()
        # (시간 x 단위) 행렬
        pivot = df.groupby(["hour_ts", col]).size().unstack(fill_value=0)
        pivot = pivot.reindex(all_hours, fill_value=0)

        vals = pivot.values.flatten()
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        zero_pct = np.mean(vals == 0) * 100
        small_pct = np.mean(vals <= 2) * 100
        snr = mean_val / (std_val + 1e-6)

        # 각 단위별 1시간 Lag-1 자기상관
        autocorrs = []
        for c in pivot.columns:
            s = pivot[c]
            if s.std() > 0:
                ac = s.autocorr(1)
                if np.isfinite(ac):
                    autocorrs.append(ac)
        avg_ac = np.mean(autocorrs)

        # 푸아송 이론상 상대 오차 (Relative Standard Error = 1 / sqrt(lambda))
        rel_noise = (1.0 / np.sqrt(mean_val)) * 100 if mean_val > 0 else 999.0

        print(f"\n▶ {label}")
        print(f"  - 공간 단위 개수         : {n_units:,} 개")
        print(f"  - 시간당 평균 대여량(λ)  : {mean_val:.2f} 대/시간")
        print(f"  - 대여 0건(공실) 비율    : {zero_pct:5.2f}%")
        print(f"  - 2대 이하 극소수 비율   : {small_pct:5.2f}% (한 자리 소수 영역)")
        print(f"  - 신호 대 잡음비 (Mean/Std): {snr:.3f} (높을수록 예측 가능)")
        print(f"  - 시계열 자기상관 (Lag-1): r = {avg_ac:+.4f} (추세의 연속성)")
        print(f"  - 푸아송 내재 노이즈 비율: ±{rel_noise:.1f}% (평균 대비 오차 범위)")

if __name__ == "__main__":
    main()
