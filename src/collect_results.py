"""흩어진 실험 결과(로컬 + HPC 에서 받은 것)를 한 표로 모은다.

산출  outputs/results_all.csv   실험 하나 = 한 줄
        experiment : 실험 묶음 이름 (zone_c1000_h1, st_ws, abl_xgb_weather …)
        unit       : 예측 단위 (station_1h / zone1km_1h / zone1km_2h)
        weather, spatial : 그 실험이 날씨·공간 특징을 썼는지
        fold, model, R2, R2_ceiling, RMSE, MAE, acc_1mWMAPE, sec …
      outputs/results_summary.csv  단위·모델별 네 계절 평균

실행:  python src/collect_results.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# 실험 묶음 -> (예측 단위, 날씨 사용, 공간 사용, 실행 위치)
SPEC = {
    "st": ("station_1h", False, False, "local"),
    "st_summer": ("station_1h", False, False, "hpc"),
    "st_autumn": ("station_1h", False, False, "hpc"),
    "st_ws": ("station_1h", True, True, "hpc"),
    "abl_xgb_base": ("station_1h", False, False, "hpc"),
    "abl_xgb_weather": ("station_1h", True, False, "hpc"),
    "abl_xgb_spatial": ("station_1h", False, True, "hpc"),
    "abl_xgb_both": ("station_1h", True, True, "hpc"),
    "zone_c1000_h1": ("zone1km_1h", True, True, "hpc"),
    "zone_c1000_h2": ("zone1km_2h", True, True, "hpc"),
}


def main():
    rows = []
    for name, (unit, weather, spatial, where) in SPEC.items():
        for base in (OUT / name, OUT / "hpc" / name):
            p = base / "metrics.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df.insert(0, "experiment", name)
            df.insert(1, "unit", unit)
            df.insert(2, "weather", weather)
            df.insert(3, "spatial", spatial)
            df.insert(4, "run_on", where)
            rows.append(df)
            break
    if not rows:
        raise SystemExit("metrics.csv 를 찾지 못했습니다.")
    all_df = pd.concat(rows, ignore_index=True)
    # 대여소 단위 실험은 구역·도시 단위 점수를 따로 저장해 둔 파일에서 붙인다
    agg_rows = []
    for name in SPEC:
        for base in (OUT / name, OUT / "hpc" / name):
            p = base / "agg_metrics.csv"
            if p.exists():
                a = pd.read_csv(p)
                a.insert(0, "experiment", name)
                agg_rows.append(a)
                break
    if agg_rows:
        agg = pd.concat(agg_rows, ignore_index=True)
        wide = agg.pivot_table(index=["experiment", "fold", "model"], columns="level", values="R2").reset_index()
        wide.columns = [c if c in ("experiment", "fold", "model") else f"R2_{c}" for c in wide.columns]
        all_df = all_df.merge(wide, on=["experiment", "fold", "model"], how="left")
    all_df = all_df.sort_values(["unit", "experiment", "model", "fold"])
    all_df.to_csv(OUT / "results_all.csv", index=False)

    keep = [c for c in ["R2", "R2_ceiling", "RMSE", "MAE", "acc_1mWMAPE", "R2_grid1km_x_1h", "R2_city_x_1h", "city_R2", "sec"] if c in all_df]
    summ = all_df.groupby(["unit", "experiment", "weather", "spatial", "model"])[keep].mean().round(3).reset_index()
    summ.to_csv(OUT / "results_summary.csv", index=False)
    print(f"results_all.csv  {len(all_df)} 줄")
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
