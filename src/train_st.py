"""대여소 × 다음 1시간 대여 건수(수요) 예측: XGBoost vs GRU vs GCN+GRU.

입력은 세 모델이 같은 정보를 보게 맞춘다.
  - 직전 24시간 대여·반납 (XGBoost 는 직전 3시간 + 6시간 합)
  - 예측할 시간의 하루 전·일주일 전 대여, 지난 4주 같은 시간 평균
  - 예측할 시간의 시각·요일·휴일·새벽 운영 여부
  - 대여소 좌표·거치대 수 (신경망은 대여소 임베딩도)
  - 관측이 없던 칸(장애·미운영·대여소 없던 기간)은 0 + '없음' 표시
그래프 (GCN+GRU):
  - 거리 그래프: 1km 안 가까운 10곳, 가중치 exp(-(d/500m)^2)
  - 이동 그래프: 학습 기간 대여 기록에서 실제로 오간 건수 상위 10곳
손실: 신경망은 포아송 (건수 데이터), XGBoost 는 제곱오차 (앞선 비교와 동일)

평가: 계절 교차검증 (compare_ml.py 와 같은 구간), 유효 칸만 (대여소 운영 중 + 운영 시간 + 장애 아님)
실행:  python src/train_st.py --folds winter spring summer autumn --models xgb gru gcn_gru
산출:  outputs/st/metrics.csv, agg_metrics.csv, history.csv, {fold}_{model}_pred.npz
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "data" / "processed" / "v2"
OUT = ROOT / "outputs" / "st"
FOLDS = {
    "winter": ("2026-01-01", "2026-04-01"),
    "spring": ("2025-04-01", "2025-06-01"),
    "summer": ("2025-07-01", "2025-09-01"),
    "autumn": ("2025-10-01", "2025-12-01"),
}
L = 24


def hav_m(a1, o1, a2, o2):
    a1, o1, a2, o2 = (np.radians(np.asarray(x, float)) for x in (a1, o1, a2, o2))
    return 2 * 6371000 * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))


# ======================================================================== data
class Data:
    def __init__(self, weather=False, spatial=False):
        self.use_weather, self.use_spatial = weather, spatial
        z = np.load(V2 / "panel.npz")
        self.ts = pd.to_datetime(z["ts"].astype("datetime64[h]"))
        self.rent, self.ret = z["rent"], z["ret"]
        self.active, self.hour_ok, self.night_ops = z["node_active"], z["hour_ok"], z["night_ops"]
        self.nodes = pd.read_parquet(V2 / "nodes.parquet")
        self.T, self.N = self.rent.shape
        self.obs = self.active & self.hour_ok[:, None]  # 관측이 믿을 만한 칸 = 평가 대상 칸
        hour = self.ts.hour.to_numpy()
        self.operating = self.night_ops | (hour >= 5)
        self._build_arrays()

    def lag(self, a, k, ok):
        out = np.zeros_like(a)
        flag = np.zeros(a.shape, bool)
        out[k:] = a[:-k]
        flag[k:] = ok[:-k]
        return out * flag, flag

    def _build_arrays(self):
        T, N = self.T, self.N
        obs = self.obs
        r = self.rent * obs
        e = self.ret * obs
        # 예측 대상 시간 h 기준 값들 (모두 h 보다 과거만 사용)
        self.lag24, self.f24 = self.lag(self.rent, 24, obs)
        self.lag168, self.f168 = self.lag(self.rent, 168, obs)
        acc = np.zeros((T, N), np.float32)
        cnt = np.zeros((T, N), np.float32)
        for k in range(1, 29):
            v, f = self.lag(self.rent, 24 * k, obs)
            acc += v
            cnt += f
        self.m4w = acc / np.maximum(cnt, 1)
        self.f4w = cnt >= 3
        kr = holidays.KR(years=range(2024, 2027))
        hol = np.array([d in kr for d in self.ts.date])
        dow = self.ts.dayofweek.to_numpy()
        hour = self.ts.hour.to_numpy()
        self.time_feat = np.stack(
            [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
             ((dow >= 5) | hol).astype(float), hol.astype(float), self.operating.astype(float),
             np.sin(2 * np.pi * self.ts.dayofyear.to_numpy() / 365.25), np.cos(2 * np.pi * self.ts.dayofyear.to_numpy() / 365.25)],
            1,
        ).astype(np.float32)  # (T, 9)
        self.time_names = ["h_sin", "h_cos", "dow_sin", "dow_cos", "offday", "holiday", "operating", "doy_sin", "doy_cos"]
        if self.use_weather:
            # 예측할 시간의 날씨(운영 시엔 기상청 단기예보로 대체) + 직전 3시간 실제 강수
            w = pd.read_parquet(ROOT / "data/raw/weather_openmeteo_daejeon.parquet")
            assert len(w) == T
            pr = w.precipitation.to_numpy(np.float32)
            prev3 = np.concatenate([[0, 0, 0], np.convolve(pr, np.ones(3), "valid")[:-1]]).astype(np.float32)
            wf = np.stack([np.log1p(pr), (pr >= 0.1).astype(np.float32), np.log1p(prev3),
                           (w.temperature_2m.to_numpy(np.float32) - 13) / 11, w.wind_speed_10m.to_numpy(np.float32) / 10,
                           np.log1p(w.snowfall.to_numpy(np.float32) * 10)], 1)
            self.time_feat = np.concatenate([self.time_feat, wf], 1)
            self.time_names += ["rain_log", "rain_flag", "rain_prev3h_log", "temp", "wind", "snow_log"]
        nd = self.nodes
        lat = ((nd.lat - nd.lat.mean()) / nd.lat.std()).to_numpy(np.float32)
        lon = ((nd.lon - nd.lon.mean()) / nd.lon.std()).to_numpy(np.float32)
        cap = nd.capacity.to_numpy(np.float32)
        self.static = np.stack([lat, lon, np.nan_to_num(cap / 10.0), np.isfinite(cap).astype(np.float32)], 1)  # (N, 4)
        self.static_names = ["lat", "lon", "cap10", "cap_known"]
        if self.use_spatial:
            sp = pd.read_parquet(V2 / "node_spatial.parquet").set_index("node").loc[nd.node]
            cols = []
            for c in sp.columns:
                v = sp[c].to_numpy(np.float64)
                if c.startswith("dist_") or c.startswith("n_") or c.endswith("_km_500m"):
                    v = np.log1p(np.clip(v, 0, None))
                cols.append(((v - v.mean()) / (v.std() + 1e-9)).astype(np.float32))
                self.static_names.append(c)
            self.static = np.concatenate([self.static, np.stack(cols, 1)], 1)

        # 신경망 입력: 스텝 s 의 채널은 "s 시점 관측 + s+1 시간에 대한 사전 정보"
        #   c0 log1p rent_s, c1 log1p ret_s, c2 obs_s,
        #   c3 log1p rent[s+1-24], c4 flag, c5 log1p rent[s+1-168], c6 flag, c7 log1p m4w[s+1], c8 flag, c9 node active[s+1]
        #   + time_feat[s+1] (9)
        nxt = lambda a: np.concatenate([a[1:], a[-1:]], 0)  # noqa: E731
        C = 10 + self.time_feat.shape[1]
        X = np.zeros((T, N, C), np.float16)
        X[..., 0] = np.log1p(r)
        X[..., 1] = np.log1p(e)
        X[..., 2] = obs
        X[..., 3] = np.log1p(nxt(self.lag24))
        X[..., 4] = nxt(self.f24)
        X[..., 5] = np.log1p(nxt(self.lag168))
        X[..., 6] = nxt(self.f168)
        X[..., 7] = np.log1p(nxt(self.m4w))
        X[..., 8] = nxt(self.f4w)
        X[..., 9] = nxt(self.active)
        X[..., 10:] = nxt(self.time_feat)[:, None, :]
        self.X = X
        self.C = C

    def fold_masks(self, fold):
        lo, hi = (pd.Timestamp(x) for x in FOLDS[fold])
        ts = self.ts
        test_h = (ts >= lo) & (ts < hi)
        buf = (ts >= lo - pd.Timedelta(days=1)) & (ts < hi + pd.Timedelta(days=28))
        inner = (ts.normalize().to_numpy().astype("datetime64[D]").astype(np.int64) % 10) == 3
        ok_h = np.arange(self.T) >= 24 * 29  # 4주 평균·일주일 전 값이 만들어지는 시점부터
        train_h = ok_h & ~buf & ~inner
        val_h = ok_h & ~buf & inner
        test_h = test_h & ok_h
        has = self.obs.any(1)
        return np.nonzero(train_h & has)[0], np.nonzero(val_h & has)[0], np.nonzero(test_h & has)[0], (lo, hi)

    def graphs(self, lo, hi, k=10):
        nd = self.nodes
        N = self.N
        la, lo_ = nd.lat.to_numpy(), nd.lon.to_numpy()
        D = hav_m(la[:, None], lo_[:, None], la[None, :], lo_[None, :])
        np.fill_diagonal(D, np.inf)
        rows, cols, vals = [], [], []
        for i in range(N):
            j = np.argsort(D[i])[:k]
            j = j[D[i, j] <= 1000]
            rows += [i] * len(j)
            cols += list(j)
            vals += list(np.exp(-((D[i, j] / 500.0) ** 2)))
        A_dist = self._norm(rows, cols, vals, N)
        od = pd.read_parquet(V2 / "od_month.parquet")
        m = pd.PeriodIndex(od.month, freq="M")
        bad_lo = (lo - pd.Timedelta(days=1)).to_period("M")
        bad_hi = (hi + pd.Timedelta(days=28)).to_period("M")
        od = od[((m < bad_lo) | (m > bad_hi)) & (od.src != od.dst)]
        f = od.groupby(["src", "dst"]).n.sum().reset_index()
        f2 = pd.concat([f, f.rename(columns={"src": "dst", "dst": "src"})]).groupby(["src", "dst"]).n.sum().reset_index()
        f2 = f2.sort_values(["src", "n"], ascending=[True, False]).groupby("src").head(k)
        A_flow = self._norm(f2.src.tolist(), f2.dst.tolist(), np.log1p(f2.n).tolist(), N)
        return A_dist, A_flow

    @staticmethod
    def _norm(rows, cols, vals, N):
        rows, cols, vals = np.asarray(rows), np.asarray(cols), np.asarray(vals, np.float32)
        deg = np.bincount(rows, weights=vals, minlength=N)
        vals = (vals / np.maximum(deg[rows], 1e-6)).astype(np.float32)
        return torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols]).astype(np.int64)), torch.from_numpy(vals), (N, N)).coalesce()


# ======================================================================== models
class SpatialMix(nn.Module):
    """h + ReLU(W_d A_dist h + W_f A_flow h)  — 두 그래프를 따로 섞는 1층 GCN."""

    def __init__(self, d):
        super().__init__()
        self.wd, self.wf = nn.Linear(d, d), nn.Linear(d, d)

    @staticmethod
    def spmm(A, h):  # h (..., N, d)
        shp = h.shape
        hN = h.reshape(-1, shp[-2], shp[-1]).transpose(0, 1).reshape(shp[-2], -1)  # (N, B*d)
        out = torch.sparse.mm(A, hN.float()).reshape(shp[-2], -1, shp[-1]).transpose(0, 1)
        return out.reshape(shp)

    def forward(self, h, A_dist, A_flow):
        return h + torch.relu(self.wd(self.spmm(A_dist, h)) + self.wf(self.spmm(A_flow, h)))


class STNet(nn.Module):
    def __init__(self, C, n_nodes, n_static, d=64, hidden=128, emb=16, graph=False):
        super().__init__()
        self.graph = graph
        self.emb = nn.Embedding(n_nodes, emb)
        self.inp = nn.Sequential(nn.Linear(C + n_static, d), nn.ReLU())
        if graph:
            self.mix_in = SpatialMix(d)
            self.mix_out = SpatialMix(hidden)
        self.gru = nn.GRU(d + emb, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + emb + C + n_static, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, x, static, A_dist=None, A_flow=None):
        B, Lw, N, C = x.shape
        s = static[None, None].expand(B, Lw, N, -1)
        h = self.inp(torch.cat([x, s], -1))  # (B,L,N,d)
        if self.graph:
            h = self.mix_in(h, A_dist, A_flow)
        e = self.emb.weight[None, None].expand(B, Lw, N, -1)
        h = torch.cat([h, e], -1).permute(0, 2, 1, 3).reshape(B * N, Lw, -1)
        _, hn = self.gru(h)
        hn = hn[-1].reshape(B, N, -1)
        if self.graph:
            hn = self.mix_out(hn, A_dist, A_flow)
        z = torch.cat([hn, self.emb.weight[None].expand(B, N, -1), x[:, -1], static[None].expand(B, N, -1)], -1)
        return nn.functional.softplus(self.head(z).squeeze(-1)) + 1e-4  # 기대 대여 건수 λ


# ======================================================================== xgboost features
def tab_features(D: Data, hs):
    """시간 인덱스 hs 에서 유효 칸만 뽑아 표 형태 피처 (h 보다 과거만)."""
    tt, nn_ = np.nonzero(D.obs[hs])
    h = hs[tt]
    obs = D.obs
    cols = {}
    for k in (1, 2, 3):
        f = obs[h - k, nn_]
        cols[f"rent_l{k}"] = np.where(f, D.rent[h - k, nn_], np.nan)
        cols[f"ret_l{k}"] = np.where(f, D.ret[h - k, nn_], np.nan)
    r6 = sum(D.rent[h - k, nn_] * obs[h - k, nn_] for k in range(1, 7))
    e6 = sum(D.ret[h - k, nn_] * obs[h - k, nn_] for k in range(1, 7))
    n6 = sum(obs[h - k, nn_].astype(np.float32) for k in range(1, 7))
    cols["rent_6h"] = np.where(n6 > 0, r6, np.nan)
    cols["ret_6h"] = np.where(n6 > 0, e6, np.nan)
    cols["obs_6h"] = n6
    cols["rent_d1"] = np.where(D.f24[h, nn_], D.lag24[h, nn_], np.nan)
    cols["rent_w1"] = np.where(D.f168[h, nn_], D.lag168[h, nn_], np.nan)
    cols["rent_4w"] = np.where(D.f4w[h, nn_], D.m4w[h, nn_], np.nan)
    for i, name in enumerate(D.time_names):
        cols[name] = D.time_feat[h, i]
    for i, name in enumerate(D.static_names):
        cols[name] = D.static[nn_, i]
    X = np.stack(list(cols.values()), 1).astype(np.float32)
    tab_features.last_names = list(cols)
    y = D.rent[h, nn_]
    return X, y, h, nn_, list(cols)


THREADS = 16


def run_xgb(D, tr, va, te, device="cuda"):
    import xgboost as xgb

    Xtr, ytr, *_ = tab_features(D, tr)
    Xva, yva, *_ = tab_features(D, va)
    Xte, yte, h, n, _ = tab_features(D, te)
    params = dict(objective="reg:squarederror", tree_method="hist", device=device, nthread=THREADS, max_depth=8, learning_rate=0.1,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, seed=0)
    dtr = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256)
    dva = xgb.QuantileDMatrix(Xva, yva, ref=dtr)
    b = xgb.train(params, dtr, 1000, evals=[(dva, "va")], early_stopping_rounds=30, verbose_eval=False)
    p = np.clip(b.predict(xgb.DMatrix(Xte), iteration_range=(0, b.best_iteration + 1)), 0, None)
    imp = b.get_score(importance_type="gain")
    names = tab_features.last_names
    top = sorted(((names[int(k[1:])], v) for k, v in imp.items()), key=lambda x: -x[1])[:12]
    print("    xgb top gain:", ", ".join(f"{n}:{v:.0f}" for n, v in top), flush=True)
    return p.astype(np.float32), yte, h, n, {"best_iter": b.best_iteration}


# ======================================================================== neural training
def run_nn(D, tr, va, te, graph, args, fold, hist):
    dev = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    X = torch.from_numpy(D.X).to(dev)  # float16 (T,N,C)
    Y = torch.from_numpy(D.rent).to(dev)
    M = torch.from_numpy(D.obs).to(dev)
    S = torch.from_numpy(D.static).to(dev)
    A_dist = A_flow = None
    if graph:
        lo, hi = FOLDS[fold]
        a, f = D.graphs(pd.Timestamp(lo), pd.Timestamp(hi))
        A_dist, A_flow = a.to(dev), f.to(dev)
    model = STNet(D.C, D.N, S.shape[1], graph=graph).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_ep = min(len(tr), args.hours_per_epoch)
    steps_per_ep = math.ceil(n_ep / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_ep, pct_start=0.1)
    offs = torch.arange(-L, 0, device=dev)

    def batch(hs):
        idx = hs[:, None] + offs[None]  # (B,L)  스텝 h-24 .. h-1
        return X[idx].float(), Y[hs], M[hs]

    @torch.no_grad()
    def predict(hs_np):
        model.eval()
        out = []
        for i in range(0, len(hs_np), args.bs * 2):
            hs = torch.from_numpy(hs_np[i : i + args.bs * 2]).to(dev)
            x, _, _ = batch(hs)
            lam = model(x, S, A_dist, A_flow)
            out.append(lam.float().cpu())
        return torch.cat(out).numpy()

    def score(hs_np):
        lam = predict(hs_np)
        m = D.obs[hs_np]
        y = D.rent[hs_np][m]
        p = lam[m]
        return float(np.sqrt(((p - y) ** 2).mean())), float((p - y * np.log(p)).mean())

    best, best_state, bad = np.inf, None, 0
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = np.random.permutation(tr)[:n_ep]
        tot, cnt = 0.0, 0
        for i in range(0, len(perm), args.bs):
            hs = torch.from_numpy(perm[i : i + args.bs]).to(dev)
            x, y, m = batch(hs)
            lam = model(x, S, A_dist, A_flow)
            loss = ((lam - y * torch.log(lam)) * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            cnt += 1
        v_rmse, v_nll = score(va)
        name = "gcn_gru" if graph else "gru"
        hist.append({"fold": fold, "model": name, "epoch": ep, "train_poisson": tot / cnt, "val_rmse": v_rmse, "val_poisson": v_nll,
                     "sec": time.time() - t0})
        print(f"    {name} ep{ep} train {tot/cnt:.4f} | val RMSE {v_rmse:.4f} poisson {v_nll:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if v_nll < best - 1e-4:
            best, bad = v_nll, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    lam = predict(te)
    h_idx, n_idx = np.nonzero(D.obs[te])
    p = lam[h_idx, n_idx]
    y = D.rent[te][h_idx, n_idx]
    del X, Y, M
    torch.cuda.empty_cache()
    return p.astype(np.float32), y, te[h_idx], n_idx, {"best_val_poisson": best}


# ======================================================================== evaluation
def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def evaluate(D, y, p, h, n):
    err = p - y
    ev_t, ev_p = y >= 3, p >= 3
    tp = (ev_t & ev_p).sum()
    res = {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err**2).mean())), "R2": r2(y, p),
           "R2_ceiling": float(1 - y.mean() / y.var()),
           "ev3_recall": float(tp / max(1, ev_t.sum())), "ev3_precision": float(tp / max(1, ev_p.sum())),
           "ev3_f1": float(2 * tp / max(1, ev_t.sum() + ev_p.sum()))}
    nd = D.nodes
    grid = (np.floor(nd.lat / 0.009).astype(int) * 1000 + np.floor(nd.lon / 0.0112).astype(int)).to_numpy()
    df = pd.DataFrame({"h": h, "n": n, "y": y, "p": p})
    agg = {}
    for name, keys in {"node_x_3h": [df.n, df.h // 3], "grid1km_x_1h": [grid[df.n], df.h], "city_x_1h": [df.h],
                       "node_x_day": [df.n, df.h // 24]}.items():
        g = df.groupby(keys)[["y", "p"]].sum()
        agg[name] = {"R2": r2(g.y, g.p), "R2_ceiling": float(1 - g.y.mean() / g.y.var()), "mean_y": float(g.y.mean())}
    return res, agg


def main(args):
    global OUT
    if getattr(args, "out", None):
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    global THREADS
    THREADS = getattr(args, "threads", 16)
    torch.set_num_threads(THREADS)
    D = Data(weather=args.weather, spatial=args.spatial)
    print(f"panel T={D.T} N={D.N} C={D.C}, valid cells={D.obs.sum():,}", flush=True)
    rows, agg_rows, hist = [], [], []
    mpath = OUT / "metrics.csv"
    if mpath.exists() and args.resume:
        rows = pd.read_csv(mpath).to_dict("records")
        agg_rows = pd.read_csv(OUT / "agg_metrics.csv").to_dict("records")
        hist = pd.read_csv(OUT / "history.csv").to_dict("records") if (OUT / "history.csv").exists() else []
    done = {(r["fold"], r["model"]) for r in rows}
    for fold in args.folds:
        tr, va, te, _ = D.fold_masks(fold)
        print(f"\n=== {fold}: train hours {len(tr):,} / inner-val {len(va):,} / test {len(te):,}", flush=True)
        # 기준선: 지난 4주 같은 시간 평균
        if (fold, "recent4w_mean") not in done:
            hh, nn_ = np.nonzero(D.obs[te])
            p = np.where(D.f4w[te][hh, nn_], D.m4w[te][hh, nn_], 0).astype(np.float32)
            y = D.rent[te][hh, nn_]
            res, agg = evaluate(D, y, p, te[hh], nn_)
            rows.append({"fold": fold, "model": "recent4w_mean", "sec": 0, **res})
            agg_rows += [{"fold": fold, "model": "recent4w_mean", "level": k, **v} for k, v in agg.items()]
        for model in args.models:
            if (fold, model) in done:
                continue
            t0 = time.time()
            if model == "xgb":
                p, y, h, n, info = run_xgb(D, tr, va, te, args.xgb_device)
            else:
                p, y, h, n, info = run_nn(D, tr, va, te, model == "gcn_gru", args, fold, hist)
            res, agg = evaluate(D, y, p, h, n)
            sec = time.time() - t0
            rows.append({"fold": fold, "model": model, "sec": round(sec), **res, **info})
            agg_rows += [{"fold": fold, "model": model, "level": k, **v} for k, v in agg.items()]
            print(f"  {model:8s} RMSE {res['RMSE']:.3f} R2 {res['R2']:.3f} (ceiling {res['R2_ceiling']:.3f}) "
                  f"ev3 F1 {res['ev3_f1']:.3f} | node×3h R2 {agg['node_x_3h']['R2']:.3f} grid×1h {agg['grid1km_x_1h']['R2']:.3f} "
                  f"city×1h {agg['city_x_1h']['R2']:.3f} ({sec:.0f}s)", flush=True)
            np.savez_compressed(OUT / f"{fold}_{model}_pred.npz", h=h, n=n, y=y, p=p)
            pd.DataFrame(rows).to_csv(mpath, index=False)
            pd.DataFrame(agg_rows).to_csv(OUT / "agg_metrics.csv", index=False)
            if hist:
                pd.DataFrame(hist).to_csv(OUT / "history.csv", index=False)
    res = pd.DataFrame(rows)
    print("\n=== 계절 평균 ===")
    print(res.groupby("model")[["RMSE", "R2", "R2_ceiling", "ev3_f1", "sec"]].mean().round(3).to_string())
    ag = pd.DataFrame(agg_rows)
    print(ag.groupby(["level", "model"])[["R2", "R2_ceiling"]].mean().round(3).unstack(0).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--folds", nargs="+", default=list(FOLDS))
    p.add_argument("--models", nargs="+", default=["xgb", "gru", "gcn_gru"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--hours-per-epoch", type=int, default=3000, help="바퀴마다 무작위로 뽑는 학습 시각 수")
    p.add_argument("--bs", type=int, default=8, help="한 번에 쓰는 예측 시각 수 (각 시각마다 대여소 전체)")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--out", default=None, help="결과 폴더 (기본 outputs/st)")
    p.add_argument("--weather", action="store_true", help="날씨 특징 추가")
    p.add_argument("--spatial", action="store_true", help="대여소 주변 공간 특징 추가")
    p.add_argument("--xgb-device", default="cuda")
    p.add_argument("--threads", type=int, default=16)
    main(p.parse_args())
