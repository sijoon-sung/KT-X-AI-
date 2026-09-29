"""3단계-B: 기본 신경망 (MLP + 대여소 임베딩).

수치 피처는 train 통계로 표준화하고, 대여소 ID 는 임베딩으로 넣는다.
손실은 Huber (대여량 분포가 꼬리가 길어 MSE 보다 안정적).

실행:  python src/train_nn.py --dataset data/processed/dataset_f60_h1 --target net
산출:  outputs/nn_{target}/model.pt, metrics.csv, by_hour.csv, history.csv, test_pred.parquet
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from common import OUTPUTS, baselines, evaluate_all, load_split, metrics_by_hour


class MLP(nn.Module):
    def __init__(self, n_num: int, n_station: int, emb_dim: int = 16, hidden=(256, 128, 64), dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(n_station, emb_dim)
        layers, d = [], n_num + emb_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, st_idx):
        z = torch.cat([x_num, self.emb(st_idx)], dim=1)
        return self.net(z).squeeze(1)


def to_tensors(X, st, y, device):
    return (
        torch.from_numpy(X).to(device),
        torch.from_numpy(st.astype(np.int64)).to(device),
        torch.from_numpy(y).to(device),
    )


@torch.no_grad()
def predict(model, X, st, bs=65536):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(X[i : i + bs], st[i : i + bs]).float().cpu())
    return torch.cat(out).numpy()


def main(args):
    ds = Path(args.dataset)
    out = OUTPUTS / f"nn_{args.target}"
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    df_tr, X_tr, y_tr, meta = load_split(ds, "train", args.target, args.sample_frac)
    df_va, X_va, y_va, _ = load_split(ds, "val", args.target)
    df_te, X_te, y_te, _ = load_split(ds, "test", args.target)
    print(f"train {X_tr.shape}, val {X_va.shape}, test {X_te.shape}, device={device}")

    # 표준화 (NaN 은 0 으로: 용량 미상 대여소 등)
    mu = np.nanmean(X_tr, axis=0)
    sd = np.nanstd(X_tr, axis=0) + 1e-6

    def norm(X):
        Z = (X - mu) / sd
        return np.nan_to_num(Z, nan=0.0).astype(np.float32)

    X_tr, X_va, X_te = norm(X_tr), norm(X_va), norm(X_te)
    st_tr, st_va, st_te = (d["st_idx"].to_numpy() for d in (df_tr, df_va, df_te))
    y_scale = float(np.std(y_tr) + 1e-6)

    # 데이터가 GPU 메모리에 다 들어가면 올려두고 인덱싱만 한다 (8GB 기준 2천만 행 x 40 피처 = 3.2GB)
    data_dev = device if X_tr.nbytes < 4e9 else torch.device("cpu")
    Xt, St, Yt = to_tensors(X_tr, st_tr, y_tr / y_scale, data_dev)
    Xv, Sv, _ = to_tensors(X_va, st_va, y_va, device)
    Xe, Se, _ = to_tensors(X_te, st_te, y_te, device)

    model = MLP(X_tr.shape[1], meta["n_stations"], emb_dim=args.emb_dim, hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * ((len(Xt) + args.bs - 1) // args.bs))
    loss_fn = nn.HuberLoss(delta=1.0)

    best, best_state, bad, hist = np.inf, None, 0, []
    n = len(Xt)
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n, device=data_dev)
        tot = 0.0
        for i in range(0, n, args.bs):
            idx = perm[i : i + args.bs]
            xb, sb, yb = Xt[idx].to(device, non_blocking=True), St[idx].to(device), Yt[idx].to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb, sb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        p_va = predict(model, Xv, Sv) * y_scale
        mae = float(np.abs(p_va - y_va).mean())
        hist.append({"epoch": ep, "train_loss": tot / n, "val_MAE": mae, "sec": time.time() - t0})
        print(f"ep {ep:3d} loss {tot/n:.4f} val_MAE {mae:.4f} ({time.time()-t0:.0f}s)")
        if mae < best - 1e-4:
            best, bad = mae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "mu": mu, "sd": sd, "y_scale": y_scale, "features": meta["features"], "args": vars(args)}, out / "model.pt")
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

    p_te = (predict(model, Xe, Se) * y_scale).astype(np.float32)
    preds = {**baselines(df_te, args.target), "mlp": p_te}
    m = evaluate_all(df_te, y_te, preds, args.target, thr=args.event_thr)
    print("\n=== test metrics ===")
    print(m.round(3).to_string())
    m.to_csv(out / "metrics.csv")
    metrics_by_hour(df_te, y_te, p_te).round(3).to_csv(out / "by_hour.csv")
    df_te[["ts_idx", "st_idx", f"y_{args.target}"]].assign(pred=p_te).to_parquet(out / "test_pred.parquet", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/processed/dataset_f60_h1")
    p.add_argument("--target", choices=["net", "rent", "ret"], default="net")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=8192)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--event-thr", type=float, default=3.0)
    main(p.parse_args())
