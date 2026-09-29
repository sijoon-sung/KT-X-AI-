"""일반적인 머신러닝 모델을 같은 조건에서 비교.

평가 방식: 계절별로 한 구간씩 시험용으로 빼고, 나머지 전체 기간으로 학습 (계절 교차검증)
  spring 2025-04~05 / summer 2025-07~08 / autumn 2025-10~11 / winter 2026-01~03
  - 시험 구간 직전 1일, 직후 28일은 학습에서 뺀다 (피처가 과거 28일을 보므로 정답이 섞이지 않게)
  - winter 는 뒤에 데이터가 없어 "과거만으로 미래를 맞히는" 실제 운영 조건과 같다
  - 나머지 계절은 미래 데이터도 학습에 쓰므로 운영보다 약간 유리한 조건이다

모델: 0 / 전주 같은 시간 / 지난 4주 같은 시간 평균 / 선형회귀(릿지) / 랜덤포레스트 / LightGBM / XGBoost / MLP
조기 종료가 필요한 모델은 학습 기간 안에서 10일에 하루씩 뺀 날로 판단한다 (시험 구간은 절대 안 봄).

실행:  python src/compare_ml.py --target rent
산출:  outputs/compare/{target}_folds.csv, {target}_summary.csv, {target}_mlp_history.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import OUTPUTS, PROC, event_metrics

FOLDS = {
    "spring": ("2025-04-01", "2025-06-01"),
    "summer": ("2025-07-01", "2025-09-01"),
    "autumn": ("2025-10-01", "2025-12-01"),
    "winter": ("2026-01-01", "2026-04-01"),
}


def metrics(y, p, target):
    err = p - y
    sst = ((y - y.mean()) ** 2).sum()
    r = {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "R2": float(1 - (err**2).sum() / sst),
    }
    ev = event_metrics(y, p, target, 3.0)
    r.update({"ev3_recall": ev["recall"], "ev3_precision": ev["precision"], "ev3_f1": ev["f1"]})
    return r


# ---------------------------------------------------------------- models
def fit_ridge(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = ((Xtr - mu) / sd).astype(np.float64)
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ ytr.astype(np.float64))
    del Z
    Zt = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1))])
    return (Zt @ w).astype(np.float32)


def fit_rf(Xtr, ytr, Xte, n_rows, seed):
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xtr), size=min(n_rows, len(Xtr)), replace=False)
    rf = RandomForestRegressor(n_estimators=60, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=seed)
    rf.fit(Xtr[idx], ytr[idx])
    return rf.predict(Xte).astype(np.float32)


def fit_lgbm(Xtr, ytr, Xva, yva, Xte):
    import lightgbm as lgb

    dtr = lgb.Dataset(Xtr, ytr, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr)
    params = dict(objective="regression", learning_rate=0.1, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=16)
    m = lgb.train(params, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m.predict(Xte, num_iteration=m.best_iteration).astype(np.float32), m.best_iteration


def fit_xgb(Xtr, ytr, Xva, yva, Xte):
    import xgboost as xgb

    params = dict(objective="reg:squarederror", tree_method="hist", device="cuda", max_depth=8, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=0)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    b = xgb.train(params, dtr, 1000, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
    p = b.predict(xgb.DMatrix(Xte), iteration_range=(0, b.best_iteration + 1)).astype(np.float32)
    return p, b.best_iteration


def fit_mlp(Xtr, str_, ytr, Xva, sva, yva, Xte, ste, n_station, epochs, lr, hist_rows, fold):
    import torch
    import torch.nn as nn

    from train_nn import MLP

    dev = torch.device("cuda")
    torch.manual_seed(0)
    mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0) + 1e-6
    norm = lambda X: torch.from_numpy(np.nan_to_num((X - mu) / sd).astype(np.float32)).to(dev)  # noqa: E731
    ys = float(ytr.std() + 1e-6)
    Xt, St, Yt = norm(Xtr), torch.from_numpy(str_.astype(np.int64)).to(dev), torch.from_numpy(ytr / ys).to(dev)
    Xv, Sv = norm(Xva), torch.from_numpy(sva.astype(np.int64)).to(dev)
    Xe, Se = norm(Xte), torch.from_numpy(ste.astype(np.int64)).to(dev)
    model = MLP(Xt.shape[1], n_station).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    bs = 8192
    steps = epochs * ((len(Xt) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    loss_fn = nn.HuberLoss(delta=1.0)

    @torch.no_grad()
    def pred(X, S):
        model.eval()
        return torch.cat([model(X[i : i + 65536], S[i : i + 65536]) for i in range(0, len(X), 65536)]).cpu().numpy() * ys

    best, best_state = np.inf, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        tot = 0.0
        for i in range(0, len(Xt), bs):
            j = perm[i : i + bs]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(Xt[j], St[j]), Yt[j])
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(j)
        pv = pred(Xv, Sv)
        rmse = float(np.sqrt(((pv - yva) ** 2).mean()))
        mae = float(np.abs(pv - yva).mean())
        hist_rows.append({"fold": fold, "epoch": ep, "train_loss": tot / len(Xt), "inner_val_MAE": mae, "inner_val_RMSE": rmse,
                          "lr": sched.get_last_lr()[0]})
        print(f"    mlp ep{ep} loss {tot/len(Xt):.4f} inner-val MAE {mae:.4f} RMSE {rmse:.4f}", flush=True)
        if rmse < best:
            best, best_state = rmse, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    out = pred(Xe, Se).astype(np.float32)
    del Xt, St, Yt, Xv, Sv, Xe, Se
    torch.cuda.empty_cache()
    return out


# ---------------------------------------------------------------- main
def main(args):
    ds = Path(args.dataset)
    out = OUTPUTS / "compare"
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ds / "meta.json").read_text(encoding="utf-8"))
    feats = meta["features"]
    cols = ["ts_idx", "st_idx", f"y_{args.target}"] + feats
    df = pd.read_parquet(ds / "all.parquet", columns=list(dict.fromkeys(cols)))
    ts = pd.read_parquet(ds / "ts_index.parquet")["ts"].to_numpy()
    t = ts[df["ts_idx"].to_numpy()]
    X = df[feats].to_numpy(np.float32)
    y = df[f"y_{args.target}"].to_numpy(np.float32)
    st = df["st_idx"].to_numpy()
    base = {
        "zero": np.zeros(len(df), np.float32),
        "lastweek_same_hour": df[f"{args.target}_next_lweek"].to_numpy(np.float32),
        "recent4w_same_hour": df[f"prof_{args.target}_4w"].to_numpy(np.float32),
    }
    del df
    # 선형회귀·랜덤포레스트용: NaN 은 학습 구간 중앙값으로
    day = t.astype("datetime64[D]")
    inner_val_day = (day.astype(np.int64) % 10) == 3

    rows, hist = [], []
    folds = [f for f in FOLDS if f in args.folds]
    for fold in folds:
        lo, hi = (np.datetime64(x) for x in FOLDS[fold])
        te = (t >= lo) & (t < hi)
        buf = (t >= lo - np.timedelta64(1, "D")) & (t < hi + np.timedelta64(28, "D"))
        tr_all = ~buf
        tr = tr_all & ~inner_val_day
        va = tr_all & inner_val_day
        print(f"\n=== {fold}: train {tr.sum():,} / inner-val {va.sum():,} / test {te.sum():,}", flush=True)
        yte = y[te]

        def record(name, p, sec, extra=None):
            r = {"fold": fold, "model": name, "sec": round(sec, 1), **metrics(yte, p, args.target)}
            if extra:
                r.update(extra)
            rows.append(r)
            print(f"  {name:20s} MAE {r['MAE']:.3f} RMSE {r['RMSE']:.3f} R2 {r['R2']:.3f} "
                  f"ev3 R/P {r['ev3_recall']:.2f}/{r['ev3_precision']:.2f} ({sec:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(out / f"{args.target}_folds.csv", index=False)

        for name, b in base.items():
            record(name, b[te], 0.0)

        Xtr, ytr, Xte = X[tr], y[tr], X[te]
        med = np.nanmedian(Xtr[:: 50], axis=0)
        fill = lambda A: np.where(np.isnan(A), med, A)  # noqa: E731

        if "ridge" in args.models:
            t0 = time.time()
            record("linear_ridge", fit_ridge(fill(Xtr), ytr, fill(Xte)), time.time() - t0)
        if "rf" in args.models:
            t0 = time.time()
            record("random_forest", fit_rf(fill(Xtr), ytr, fill(Xte), args.rf_rows, 0), time.time() - t0,
                   {"note": f"train sample {args.rf_rows:,} rows"})
        if "lgbm" in args.models:
            t0 = time.time()
            p, it = fit_lgbm(Xtr, ytr, X[va], y[va], Xte)
            record("lightgbm", p, time.time() - t0, {"note": f"best_iter {it}"})
        if "xgb" in args.models:
            t0 = time.time()
            p, it = fit_xgb(Xtr, ytr, X[va], y[va], Xte)
            record("xgboost", p, time.time() - t0, {"note": f"best_iter {it}"})
        if "mlp" in args.models:
            t0 = time.time()
            p = fit_mlp(Xtr, st[tr], ytr, X[va], st[va], y[va], Xte, st[te], meta["n_stations"], args.mlp_epochs,
                        args.mlp_lr, hist, fold)
            record("mlp", p, time.time() - t0)
            pd.DataFrame(hist).to_csv(out / f"{args.target}_mlp_history.csv", index=False)
        del Xtr, ytr, Xte

    res = pd.DataFrame(rows)
    order = list(dict.fromkeys(res["model"]))
    summ = res.groupby("model")[["MAE", "RMSE", "R2", "ev3_recall", "ev3_precision", "ev3_f1", "sec"]].mean().loc[order]
    summ.round(4).to_csv(out / f"{args.target}_summary.csv")
    print("\n=== 네 계절 평균 ===")
    print(summ.round(3).to_string())
    for m in ["RMSE", "R2", "ev3_f1"]:
        print(f"\n--- {m} by fold ---")
        print(res.pivot(index="model", columns="fold", values=m).loc[order, folds].round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default=str(PROC / "dataset_f60_h1"))
    p.add_argument("--target", choices=["net", "rent", "ret"], default="rent")
    p.add_argument("--folds", nargs="+", default=list(FOLDS))
    p.add_argument("--models", nargs="+", default=["ridge", "rf", "lgbm", "xgb", "mlp"])
    p.add_argument("--rf-rows", type=int, default=2_000_000)
    p.add_argument("--mlp-epochs", type=int, default=8)
    p.add_argument("--mlp-lr", type=float, default=1e-3)
    main(p.parse_args())
