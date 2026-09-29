"""3단계-A: XGBoost 회귀.

실행:  python src/train_xgb.py --dataset data/processed/dataset_f60_h1 --target net
산출:  outputs/xgb_{target}/model.json, metrics.csv, by_hour.csv, importance.png, test_pred.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from common import OUTPUTS, baselines, evaluate_all, load_split, metrics_by_hour


def main(args):
    ds = Path(args.dataset)
    out = OUTPUTS / f"xgb_{args.target}"
    out.mkdir(parents=True, exist_ok=True)

    df_tr, X_tr, y_tr, meta = load_split(ds, "train", args.target, args.sample_frac)
    df_va, X_va, y_va, _ = load_split(ds, "val", args.target)
    df_te, X_te, y_te, _ = load_split(ds, "test", args.target)
    feats = meta["features"]
    print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}")

    device = "cuda" if args.gpu else "cpu"
    params = {
        "objective": "reg:squarederror" if args.target == "net" else "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
        "tree_method": "hist",
        "device": device,
        "max_depth": args.max_depth,
        "learning_rate": args.lr,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "seed": 0,
    }
    t0 = time.time()
    dtr = xgb.QuantileDMatrix(X_tr, y_tr, feature_names=feats, max_bin=256)
    dva = xgb.QuantileDMatrix(X_va, y_va, feature_names=feats, ref=dtr)
    booster = xgb.train(
        params,
        dtr,
        num_boost_round=args.rounds,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    print(f"trained in {time.time()-t0:.0f}s, best_iter={booster.best_iteration}")
    booster.save_model(out / "model.json")

    dte = xgb.DMatrix(X_te, feature_names=feats)
    p_te = booster.predict(dte, iteration_range=(0, booster.best_iteration + 1)).astype(np.float32)
    preds = {**baselines(df_te, args.target), "xgboost": p_te}
    m = evaluate_all(df_te, y_te, preds, args.target, thr=args.event_thr)
    print("\n=== test metrics ===")
    print(m.round(3).to_string())
    m.to_csv(out / "metrics.csv")
    metrics_by_hour(df_te, y_te, p_te).round(3).to_csv(out / "by_hour.csv")
    df_te[["ts_idx", "st_idx", f"y_{args.target}"]].assign(pred=p_te).to_parquet(out / "test_pred.parquet", index=False)

    imp = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False)
    imp.to_csv(out / "importance.csv")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 8))
        imp.head(25)[::-1].plot.barh(ax=ax)
        ax.set_title(f"XGBoost gain importance (target={args.target})")
        fig.tight_layout()
        fig.savefig(out / "importance.png", dpi=120)
    except Exception as e:  # noqa: BLE001
        print("plot skipped:", e)
    json.dump({"params": params, "best_iteration": booster.best_iteration, "args": vars(args)}, open(out / "run.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/processed/dataset_f60_h1")
    p.add_argument("--target", choices=["net", "rent", "ret"], default="net")
    p.add_argument("--rounds", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--sample-frac", type=float, default=1.0, help="train 행 샘플 비율 (메모리 절약)")
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--event-thr", type=float, default=3.0, help="결품 위험 사건 임계값(대)")
    main(p.parse_args())
