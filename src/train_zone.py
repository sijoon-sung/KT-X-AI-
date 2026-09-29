"""구역 단위 수요 예측: 1km 격자 × 앞으로 H시간 대여 합.

희소한 대여소 × 1시간 대신, 대여소를 1km 격자로 묶어 직접 예측한다.
  - 대상: 시각 h 에서 h .. h+H-1 동안 그 구역의 대여 합 (예측 시점은 h 직전, 관측은 h-1 까지)
  - 입력·모델은 train_st.py 와 같다 (XGBoost / GRU / GCN+GRU, 포아송 손실)
  - 구역 특징: 운영 중 대여소 수(시간마다 바뀜), 거치대 합, 공간 특징 평균, 중심 좌표
  - 날씨: 예측 구간(H시간)의 강수 합·비 여부·기온/바람 평균·적설 합 + 직전 3시간 강수
  - 그래프: 구역 중심 1.5km 안 이웃(거리), 학습 기간 구역 간 이동 건수 상위 8곳(이동)
  - XGBoost 는 제곱오차와 Tweedie(희소 수요용) 둘 다 비교 가능

실행:  python src/train_zone.py --cell 1000 --H 1 --models xgb xgb_tweedie gru gcn_gru --weather --spatial
산출:  outputs/zone_c{cell}_h{H}/metrics.csv, by_hour.csv, history.csv, {fold}_{model}_pred.npz
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train_st import FOLDS, L, STNet, hav_m, r2

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "processed" / "v2"


def shift(a, k):
    out = np.zeros_like(a)
    if k > 0:
        out[k:] = a[:-k]
    return out


class ZoneData:
    def __init__(self, cell_m=1000, H=1, weather=False, spatial=False):
        z = np.load(V2 / "panel.npz")
        self.ts = pd.to_datetime(z["ts"].astype("datetime64[h]"))
        nodes = pd.read_parquet(V2 / "nodes.parquet")
        hour_ok, night_ops = z["hour_ok"], z["night_ops"]
        T = len(self.ts)
        dlat, dlon = 0.009 * cell_m / 1000, 0.0112 * cell_m / 1000
        key = np.floor(nodes.lat / dlat).astype(int) * 100000 + np.floor(nodes.lon / dlon).astype(int)
        zones, zidx = np.unique(key, return_inverse=True)
        U = len(zones)
        self.T, self.N, self.H = T, U, H
        M = np.zeros((U, len(nodes)), np.float32)
        M[zidx, np.arange(len(nodes))] = 1
        act = z["node_active"].astype(np.float32)
        self.rent = z["rent"] @ M.T  # (T,U)
        self.ret = z["ret"] @ M.T
        self.n_active = act @ M.T
        self.active = self.n_active > 0
        self.hour_ok = hour_ok
        self.obs = self.active & hour_ok[:, None]
        hour = self.ts.hour.to_numpy()
        self.operating = night_ops | (hour >= 5)

        # ---- 대상: 앞으로 H시간 합 ----
        Y = np.zeros((T, U), np.float32)
        tm = np.ones((T, U), bool)
        for k in range(H):
            Y[: T - k] += self.rent[k:]
            tm[: T - k] &= self.obs[k:]
        tm[T - H + 1 :] = False
        self.Y, self.tmask = Y, tm

        # ---- 과거 값 (모두 h 보다 과거) ----
        self.lag24, self.f24 = shift(Y, 24) * shift(tm, 24), shift(tm, 24)
        self.lag168, self.f168 = shift(Y, 168) * shift(tm, 168), shift(tm, 168)
        acc = np.zeros((T, U), np.float32)
        cnt = np.zeros((T, U), np.float32)
        for k in range(1, 29):
            f = shift(tm, 24 * k)
            acc += shift(Y, 24 * k) * f
            cnt += f
        self.m4w, self.f4w = acc / np.maximum(cnt, 1), cnt >= 3

        # ---- 시간 특징 (예측 구간 시작 h 기준) ----
        kr = holidays.KR(years=range(2024, 2027))
        hol = np.array([d in kr for d in self.ts.date])
        dow = self.ts.dayofweek.to_numpy()
        doy = self.ts.dayofyear.to_numpy()
        tf = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
              ((dow >= 5) | hol).astype(float), hol.astype(float), self.operating.astype(float),
              np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]
        self.time_names = ["h_sin", "h_cos", "dow_sin", "dow_cos", "offday", "holiday", "operating", "doy_sin", "doy_cos"]
        if weather:
            w = pd.read_parquet(ROOT / "data/raw/weather_openmeteo_daejeon.parquet")
            pr = w.precipitation.to_numpy(np.float32)
            win = lambda a, fn: np.array([fn(a[i : i + H]) for i in range(T)], np.float32)  # noqa: E731
            prev3 = np.concatenate([[0, 0, 0], np.convolve(pr, np.ones(3), "valid")[:-1]]).astype(np.float32)
            tf += [np.log1p(win(pr, np.sum)), (win(pr, np.max) >= 0.1).astype(np.float32), np.log1p(prev3),
                   (win(w.temperature_2m.to_numpy(np.float32), np.mean) - 13) / 11, win(w.wind_speed_10m.to_numpy(np.float32), np.mean) / 10,
                   np.log1p(win(w.snowfall.to_numpy(np.float32), np.sum) * 10)]
            self.time_names += ["rain_log", "rain_flag", "rain_prev3h_log", "temp", "wind", "snow_log"]
        self.time_feat = np.stack(tf, 1).astype(np.float32)

        # ---- 구역 정적 특징 ----
        cnt_nodes = M.sum(1)
        lat = (M @ nodes.lat.to_numpy()) / cnt_nodes
        lon = (M @ nodes.lon.to_numpy()) / cnt_nodes
        self.zone_lat, self.zone_lon = lat, lon
        cap = nodes.capacity.to_numpy(np.float32)
        st = [(lat - lat.mean()) / lat.std(), (lon - lon.mean()) / lon.std(), np.log1p(M @ np.nan_to_num(cap)),
              (M @ np.isfinite(cap).astype(np.float32)) / cnt_nodes, np.log1p(cnt_nodes)]
        self.static_names = ["lat", "lon", "cap_sum_log", "cap_known_share", "n_nodes_log"]
        if spatial:
            sp = pd.read_parquet(V2 / "node_spatial.parquet").set_index("node").loc[nodes.node]
            for c in sp.columns:
                v = sp[c].to_numpy(np.float64)
                if c.startswith("dist_") or c.startswith("n_") or c.endswith("_km_500m"):
                    v = np.log1p(np.clip(v, 0, None))
                v = (M @ v) / cnt_nodes
                st.append((v - v.mean()) / (v.std() + 1e-9))
                self.static_names.append(c)
        self.static = np.stack(st, 1).astype(np.float32)

        # ---- 신경망 입력 (T,U,C): 스텝 s 관측 + s+1 에 시작하는 예측 구간 정보 ----
        nxt = lambda a: np.concatenate([a[1:], a[-1:]], 0)  # noqa: E731
        C = 10 + self.time_feat.shape[1]
        X = np.zeros((T, U, C), np.float16)
        X[..., 0] = np.log1p(self.rent * self.obs)
        X[..., 1] = np.log1p(self.ret * self.obs)
        X[..., 2] = self.obs
        X[..., 3] = np.log1p(nxt(self.lag24))
        X[..., 4] = nxt(self.f24)
        X[..., 5] = np.log1p(nxt(self.lag168))
        X[..., 6] = nxt(self.f168)
        X[..., 7] = np.log1p(nxt(self.m4w))
        X[..., 8] = nxt(self.f4w)
        X[..., 9] = np.log1p(nxt(self.n_active))
        X[..., 10:] = nxt(self.time_feat)[:, None, :]
        self.X, self.C = X, C
        self.zidx, self.nodes = zidx, nodes

    def fold_masks(self, fold):
        lo, hi = (pd.Timestamp(x) for x in FOLDS[fold])
        ts = self.ts
        end_ts = ts + pd.Timedelta(hours=self.H - 1)
        test_h = (ts >= lo) & (end_ts < hi)
        buf = (end_ts >= lo - pd.Timedelta(days=1)) & (ts < hi + pd.Timedelta(days=28))
        inner = (ts.normalize().to_numpy().astype("datetime64[D]").astype(np.int64) % 10) == 3
        ok_h = np.arange(self.T) >= 24 * 29
        has = self.tmask.any(1)
        f = lambda m: np.nonzero(m & ok_h & has)[0]  # noqa: E731
        return f(~buf & ~inner), f(~buf & inner), f(test_h), (lo, hi)

    def graphs(self, lo, hi, k=8):
        U = self.N
        D = hav_m(self.zone_lat[:, None], self.zone_lon[:, None], self.zone_lat[None, :], self.zone_lon[None, :])
        np.fill_diagonal(D, np.inf)
        rows, cols, vals = [], [], []
        for i in range(U):
            j = np.argsort(D[i])[:k]
            j = j[D[i, j] <= 1500]
            rows += [i] * len(j)
            cols += list(j)
            vals += list(np.exp(-((D[i, j] / 1000.0) ** 2)))
        A_dist = self._norm(rows, cols, vals, U)
        od = pd.read_parquet(V2 / "od_month.parquet")
        m = pd.PeriodIndex(od.month, freq="M")
        od = od[(m < (lo - pd.Timedelta(days=1)).to_period("M")) | (m > (hi + pd.Timedelta(days=28)).to_period("M"))]
        od = od.assign(src=self.zidx[od.src], dst=self.zidx[od.dst])
        od = od[od.src != od.dst]
        f = od.groupby(["src", "dst"]).n.sum().reset_index()
        f2 = pd.concat([f, f.rename(columns={"src": "dst", "dst": "src"})]).groupby(["src", "dst"]).n.sum().reset_index()
        f2 = f2.sort_values(["src", "n"], ascending=[True, False]).groupby("src").head(k)
        A_flow = self._norm(f2.src.to_numpy(), f2.dst.to_numpy(), np.log1p(f2.n.to_numpy()), U)
        return A_dist, A_flow

    @staticmethod
    def _norm(rows, cols, vals, N):
        rows, cols, vals = np.asarray(rows, np.int64), np.asarray(cols, np.int64), np.asarray(vals, np.float32)
        deg = np.bincount(rows, weights=vals, minlength=N)
        vals = (vals / np.maximum(deg[rows], 1e-6)).astype(np.float32)
        return torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(vals), (N, N)).coalesce()


# ======================================================================== XGBoost
def tab_features(D: ZoneData, hs):
    tt, uu = np.nonzero(D.tmask[hs])
    h = hs[tt]
    obs = D.obs
    cols = {}
    for k in (1, 2, 3):
        f = obs[h - k, uu]
        cols[f"rent_l{k}"] = np.where(f, D.rent[h - k, uu], np.nan)
        cols[f"ret_l{k}"] = np.where(f, D.ret[h - k, uu], np.nan)
    cols["rent_6h"] = sum(D.rent[h - k, uu] * obs[h - k, uu] for k in range(1, 7))
    cols["ret_6h"] = sum(D.ret[h - k, uu] * obs[h - k, uu] for k in range(1, 7))
    cols["obs_6h"] = sum(obs[h - k, uu].astype(np.float32) for k in range(1, 7))
    cols["y_d1"] = np.where(D.f24[h, uu], D.lag24[h, uu], np.nan)
    cols["y_w1"] = np.where(D.f168[h, uu], D.lag168[h, uu], np.nan)
    cols["y_4w"] = np.where(D.f4w[h, uu], D.m4w[h, uu], np.nan)
    cols["n_active"] = D.n_active[h, uu]
    for i, n in enumerate(D.time_names):
        cols[n] = D.time_feat[h, i]
    for i, n in enumerate(D.static_names):
        cols[n] = D.static[uu, i]
    cols["zone_id"] = uu.astype(np.float32)
    return np.stack(list(cols.values()), 1).astype(np.float32), D.Y[h, uu], h, uu, list(cols)


def run_xgb(D, tr, va, te, tweedie=False):
    import xgboost as xgb

    Xtr, ytr, *_ = tab_features(D, tr)
    Xva, yva, *_ = tab_features(D, va)
    Xte, yte, h, u, names = tab_features(D, te)
    params = dict(tree_method="hist", device="cuda", max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=10, seed=0)
    if tweedie:
        params.update(objective="reg:tweedie", tweedie_variance_power=1.2, eval_metric="rmse")
    else:
        params.update(objective="reg:squarederror")
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    b = xgb.train(params, dtr, 3000, evals=[(dva, "va")], early_stopping_rounds=50, verbose_eval=False)
    p = np.clip(b.predict(xgb.DMatrix(Xte), iteration_range=(0, b.best_iteration + 1)), 0, None).astype(np.float32)
    imp = b.get_score(importance_type="gain")
    top = sorted(((names[int(k[1:])], v) for k, v in imp.items()), key=lambda x: -x[1])[:10]
    print("    top gain:", ", ".join(f"{n}:{v:.0f}" for n, v in top), flush=True)
    del dtr, dva, b
    return p, yte, h, u, {"best_iter": None}


# ======================================================================== neural
def run_nn(D, tr, va, te, graph, args, fold, hist):
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    X = torch.from_numpy(D.X).to(dev)
    Y = torch.from_numpy(D.Y).to(dev)
    Mk = torch.from_numpy(D.tmask).to(dev)
    S = torch.from_numpy(D.static).to(dev)
    A_dist = A_flow = None
    if graph:
        a, f = D.graphs(*(pd.Timestamp(x) for x in FOLDS[fold]))
        A_dist, A_flow = a.to(dev), f.to(dev)
    model = STNet(D.C, D.N, S.shape[1], graph=graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps = math.ceil(len(tr) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps, pct_start=0.1)
    offs = torch.arange(-L, 0, device=dev)

    @torch.no_grad()
    def predict(hs_np):
        model.eval()
        out = []
        for i in range(0, len(hs_np), 48):
            hs = torch.from_numpy(hs_np[i : i + 48]).to(dev)
            out.append(model(X[hs[:, None] + offs[None]].float(), S, A_dist, A_flow).cpu())
        return torch.cat(out).numpy()

    best, best_state, bad = np.inf, None, 0
    name = "gcn_gru" if graph else "gru"
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(tr)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            hs = torch.from_numpy(perm[i : i + args.bs]).to(dev)
            lam = model(X[hs[:, None] + offs[None]].float(), S, A_dist, A_flow)
            m = Mk[hs]
            loss = ((lam - Y[hs] * torch.log(lam)) * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        lam = predict(va)
        mv = D.tmask[va]
        yv, pv = D.Y[va][mv], lam[mv]
        v_nll = float((pv - yv * np.log(pv)).mean())
        v_r2 = r2(yv, pv)
        hist.append({"fold": fold, "model": name, "epoch": ep, "train": tot / steps, "val_poisson": v_nll, "val_R2": v_r2})
        print(f"    {name} ep{ep} train {tot/steps:.4f} | val poisson {v_nll:.4f} R2 {v_r2:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if v_nll < best - 1e-4:
            best, bad = v_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    lam = predict(te)
    hi_, ui = np.nonzero(D.tmask[te])
    del X, Y, Mk, model, opt
    torch.cuda.empty_cache()
    return lam[hi_, ui].astype(np.float32), D.Y[te][hi_, ui], te[hi_], ui, {"best_val_poisson": best}


# ======================================================================== eval
def evaluate(D, y, p, h):
    y64, p64 = y.astype(np.float64), p.astype(np.float64)
    res = {"mean_y": float(y64.mean()), "MAE": float(np.abs(p64 - y64).mean()), "RMSE": float(np.sqrt(((p64 - y64) ** 2).mean())),
           "R2": r2(y64, p64), "R2_ceiling": float(1 - y64.mean() / y64.var()), "acc_1mWMAPE": float(1 - np.abs(p64 - y64).sum() / y64.sum())}
    hour = D.ts[h].hour.to_numpy()
    for name, m in {"peak7_9": (hour >= 7) & (hour <= 9), "peak17_19": (hour >= 17) & (hour <= 19), "day6_23": (hour >= 6)}.items():
        res[f"R2_{name}"] = r2(y64[m], p64[m])
        res[f"acc_{name}"] = float(1 - np.abs(p64[m] - y64[m]).sum() / y64[m].sum())
    city = pd.DataFrame({"h": h, "y": y64, "p": p64}).groupby("h").sum()
    res["city_R2"] = r2(city.y, city.p)
    bh = pd.DataFrame({"hour": hour, "y": y64, "p": p64}).groupby("hour").apply(
        lambda q: pd.Series({"mean_y": q.y.mean(), "R2": r2(q.y, q.p), "acc": 1 - np.abs(q.p - q.y).sum() / max(q.y.sum(), 1e-9)}))
    return res, bh


def main(args):
    out = Path(args.out) if args.out else ROOT / "outputs" / f"zone_c{args.cell}_h{args.H}"
    out.mkdir(parents=True, exist_ok=True)
    D = ZoneData(args.cell, args.H, args.weather, args.spatial)
    print(f"zones={D.N} H={args.H} C={D.C} static={D.static.shape[1]} valid target cells={D.tmask.sum():,} "
          f"mean target={D.Y[D.tmask].mean():.2f} zero share={np.mean(D.Y[D.tmask] == 0):.3f}", flush=True)
    rows, hist, bhs = [], [], []
    for fold in args.folds:
        tr, va, te, _ = D.fold_masks(fold)
        print(f"\n=== {fold}: train {len(tr):,} / val {len(va):,} / test {len(te):,} hours", flush=True)
        hh, uu = np.nonzero(D.tmask[te])
        base = {"lastweek": np.where(D.f168[te][hh, uu], D.lag168[te][hh, uu], 0), "recent4w_mean": np.where(D.f4w[te][hh, uu], D.m4w[te][hh, uu], 0)}
        for name, p in base.items():
            res, _ = evaluate(D, D.Y[te][hh, uu], p.astype(np.float32), te[hh])
            rows.append({"fold": fold, "model": name, "sec": 0, **res})
        for model in args.models:
            t0 = time.time()
            if model.startswith("xgb"):
                p, y, h, u, info = run_xgb(D, tr, va, te, tweedie=(model == "xgb_tweedie"))
            else:
                p, y, h, u, info = run_nn(D, tr, va, te, model == "gcn_gru", args, fold, hist)
            res, bh = evaluate(D, y, p, h)
            sec = time.time() - t0
            rows.append({"fold": fold, "model": model, "sec": round(sec), **res})
            bhs.append(bh.assign(fold=fold, model=model).reset_index())
            print(f"  {model:12s} R2 {res['R2']:.3f} (ceiling {res['R2_ceiling']:.3f}) acc {res['acc_1mWMAPE']:.3f} | "
                  f"06-23시 R2 {res['R2_day6_23']:.3f} | 출근 R2 {res['R2_peak7_9']:.3f} 퇴근 R2 {res['R2_peak17_19']:.3f} | city R2 {res['city_R2']:.3f} ({sec:.0f}s)", flush=True)
            np.savez_compressed(out / f"{fold}_{model}_pred.npz", h=h, u=u, y=y, p=p)
            pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
            pd.concat(bhs).to_csv(out / "by_hour.csv", index=False)
            if hist:
                pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    res = pd.DataFrame(rows)
    cols = ["R2", "R2_ceiling", "acc_1mWMAPE", "R2_day6_23", "R2_peak7_9", "R2_peak17_19", "city_R2", "sec"]
    print("\n=== 네 계절 평균 ===")
    print(res.groupby("model", sort=False)[cols].mean().round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cell", type=int, default=1000)
    p.add_argument("--H", type=int, default=1)
    p.add_argument("--folds", nargs="+", default=list(FOLDS))
    p.add_argument("--models", nargs="+", default=["xgb", "xgb_tweedie", "gru", "gcn_gru"])
    p.add_argument("--weather", action="store_true")
    p.add_argument("--spatial", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    main(p.parse_args())
