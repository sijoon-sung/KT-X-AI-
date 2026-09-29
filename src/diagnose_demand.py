"""수요(대여 건수) 예측이 왜 R² 0.65 에서 멈추는지 진단.

1) 데이터 양: 학습 행을 5% / 20% / 50% / 100% 로 늘려 성능이 오르는지
2) 우연에 의한 흔들림: 기대 대여량을 정확히 알아도 실제 건수는 푸아송처럼 흔들린다.
   - 이론 상한  R²max = 1 - mean(y)/var(y)   (순수 푸아송 가정)
   - 모의 실험  모델 예측값을 참 기대값으로 두고 푸아송으로 뽑은 가짜 정답에 대한 R²
   - 실제 흔들림이 푸아송보다 큰지 (예측값 구간별 잔차 분산 / 평균)
3) 큰 값 몇 개가 오차를 좌우하는지: 한 시간 10건·20건 이상 칸이 제곱오차에서 차지하는 비율
4) 묶는 단위를 키우면: 같은 1시간 예측을 대여소×3시간, 대여소×하루, 동×1시간, 구×1시간, 대전 전체×1시간으로 합쳐 R²
5) 붐비는 대여소만 보면

평가: compare_ml.py 의 winter 구간 (2026-01~03, 과거만으로 학습 = 운영 조건)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from common import OUTPUTS, PROC

DS = PROC / "dataset_f60_h1"
OUT = OUTPUTS / "diagnose"
OUT.mkdir(parents=True, exist_ok=True)


def r2(y, p):
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def train_xgb(X, y, Xva, yva):
    params = dict(objective="reg:squarederror", tree_method="hist", device="cuda", max_depth=8, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=0)
    dtr = xgb.QuantileDMatrix(X, y, max_bin=256)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    b = xgb.train(params, dtr, 1000, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
    return b


def main():
    meta = json.loads((DS / "meta.json").read_text(encoding="utf-8"))
    feats = meta["features"]
    df = pd.read_parquet(DS / "all.parquet", columns=["ts_idx", "st_idx", "y_rent"] + feats)
    ts = pd.read_parquet(DS / "ts_index.parquet")["ts"].to_numpy()
    stations = pd.read_parquet(DS / "stations.parquet")
    t = ts[df["ts_idx"].to_numpy()]
    X = df[feats].to_numpy(np.float32)
    y = df["y_rent"].to_numpy(np.float32)

    lo, hi = np.datetime64("2026-01-01"), np.datetime64("2026-04-01")
    te = (t >= lo) & (t < hi)
    trall = t < lo - np.timedelta64(1, "D")
    inner = (t.astype("datetime64[D]").astype(np.int64) % 10) == 3
    tr, va = trall & ~inner, trall & inner
    rep = {}

    # ---- 1) 데이터 양 ----
    rng = np.random.default_rng(0)
    tr_idx = np.nonzero(tr)[0]
    lc = []
    booster = None
    for frac in [0.05, 0.2, 0.5, 1.0]:
        idx = tr_idx if frac == 1.0 else np.sort(rng.choice(tr_idx, int(len(tr_idx) * frac), replace=False))
        t0 = time.time()
        b = train_xgb(X[idx], y[idx], X[va], y[va])
        p = b.predict(xgb.DMatrix(X[te]), iteration_range=(0, b.best_iteration + 1))
        lc.append({"train_frac": frac, "train_rows": len(idx), "test_R2": r2(y[te], p), "sec": time.time() - t0})
        print(lc[-1], flush=True)
        if frac == 1.0:
            booster, p_full = b, np.clip(p, 0, None)
    rep["learning_curve"] = lc

    yt = y[te].astype(np.float64)
    pt = p_full.astype(np.float64)

    # ---- 2) 우연에 의한 흔들림 ----
    rep["R2_model"] = r2(yt, pt)
    rep["R2_ceiling_pure_poisson"] = float(1 - yt.mean() / yt.var())
    sims = [r2(rng.poisson(pt), pt) for _ in range(5)]
    rep["R2_if_model_were_true_rate_and_poisson"] = float(np.mean(sims))
    bins = [0, 0.1, 0.3, 0.6, 1, 2, 4, 8, 1e9]
    b = pd.cut(pt, bins, right=False)
    disp = pd.DataFrame({"bin": b, "y": yt, "p": pt, "res2": (yt - pt) ** 2}).groupby("bin", observed=True).agg(
        n=("y", "size"), mean_pred=("p", "mean"), mean_y=("y", "mean"), resid_var=("res2", "mean"))
    disp["resid_var_over_mean"] = disp["resid_var"] / disp["mean_pred"]
    rep["dispersion_by_pred_bin"] = disp.round(3).reset_index().astype({"bin": str}).to_dict("records")

    # ---- 3) 큰 값의 영향 ----
    se = (yt - pt) ** 2
    rep["share_of_rows_y>=10"] = float((yt >= 10).mean())
    rep["share_of_sq_error_from_y>=10"] = float(se[yt >= 10].sum() / se.sum())
    rep["share_of_sq_error_from_y>=20"] = float(se[yt >= 20].sum() / se.sum())
    rep["R2_excluding_y>=20"] = r2(yt[yt < 20], pt[yt < 20])
    top = pd.DataFrame({"ts": t[te], "st_idx": df["st_idx"].to_numpy()[te], "y": yt, "pred": pt}).nlargest(15, "y")
    top = top.merge(stations[["st_idx", "station_id", "name"]], on="st_idx")
    rep["largest_actual_values"] = top.astype({"ts": str}).round(2).to_dict("records")

    # ---- 4) 묶는 단위 ----
    d = pd.DataFrame({"ts": t[te], "st_idx": df["st_idx"].to_numpy()[te], "y": yt, "p": pt})
    d = d.merge(stations[["st_idx", "dong", "gu"]], on="st_idx", how="left")
    d["day"] = d["ts"].dt.floor("D")
    d["h3"] = d["ts"].dt.floor("3h")
    agg = {}
    for name, keys in {
        "station x 1h": ["st_idx", "ts"],
        "station x 3h": ["st_idx", "h3"],
        "station x day": ["st_idx", "day"],
        "dong x 1h": ["dong", "ts"],
        "gu x 1h": ["gu", "ts"],
        "city x 1h": ["ts"],
    }.items():
        g = d.groupby(keys, observed=True)[["y", "p"]].sum()
        agg[name] = {"R2": r2(g["y"], g["p"]), "mean_y": float(g["y"].mean()),
                     "R2_ceiling_pure_poisson": float(1 - g["y"].mean() / g["y"].var())}
        print(name, agg[name], flush=True)
    rep["aggregation"] = agg

    # ---- 5) 붐비는 대여소 ----
    st_mean = d.groupby("st_idx")["y"].mean()
    busy = st_mean[st_mean >= st_mean.quantile(0.9)].index
    m = d["st_idx"].isin(busy).to_numpy()
    rep["busy_top10pct_stations"] = {"R2": r2(yt[m], pt[m]), "mean_y": float(yt[m].mean()),
                                     "R2_ceiling_pure_poisson": float(1 - yt[m].mean() / yt[m].var())}

    (OUT / "demand_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k not in ("largest_actual_values",)}, ensure_ascii=False, indent=1, default=str))
    print(pd.DataFrame(rep["largest_actual_values"]).to_string())


if __name__ == "__main__":
    main()
