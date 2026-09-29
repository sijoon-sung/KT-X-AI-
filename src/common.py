"""데이터 로딩·평가 공통 코드."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"


def load_split(dataset_dir: Path, name: str, target: str, sample_frac: float = 1.0, seed: int = 0):
    df = pd.read_parquet(dataset_dir / f"{name}.parquet")
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
    meta = json.loads((dataset_dir / "meta.json").read_text(encoding="utf-8"))
    X = df[meta["features"]].to_numpy(np.float32)
    y = df[f"y_{target}"].to_numpy(np.float32)
    return df, X, y, meta


def baselines(df: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    """비교용 단순 예측 (제안서의 '단순 평균' 단계)."""
    return {
        "zero": np.zeros(len(df), np.float32),
        "yesterday_same_hour": df[f"{target}_next_yday"].to_numpy(np.float32),
        "lastweek_same_hour": df[f"{target}_next_lweek"].to_numpy(np.float32),
        "recent4w_same_hour": df[f"prof_{target}_4w"].to_numpy(np.float32),
    }


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
    }


def event_metrics(y: np.ndarray, p: np.ndarray, target: str, thr: float, capacity: np.ndarray | None = None) -> dict:
    """'결품 위험' 사건 적중률.
    net 타깃: 다음 구간 순유출(rent-ret) >= thr 이면 위험 (net <= -thr)
    rent 타깃: 다음 구간 대여 >= thr 이면 수요 집중
    capacity 가 있으면 thr 대신 용량의 절반 이상 빠지는지로 본다."""
    if capacity is not None:
        t = np.maximum(2.0, capacity * 0.5)
    else:
        t = thr
    if target == "net":
        yt, pt = y <= -t, p <= -t
    else:
        yt, pt = y >= t, p >= t
    tp = float((yt & pt).sum())
    fp = float((~yt & pt).sum())
    fn = float((yt & ~pt).sum())
    return {
        "threshold": "0.5*capacity" if capacity is not None else thr,
        "n_events": int(yt.sum()),
        "recall": tp / max(1.0, tp + fn),
        "precision": tp / max(1.0, tp + fp),
        "f1": 2 * tp / max(1.0, 2 * tp + fp + fn),
    }


def evaluate_all(df: pd.DataFrame, y: np.ndarray, preds: dict[str, np.ndarray], target: str, thr: float = 3.0) -> pd.DataFrame:
    rows = []
    cap = df["capacity"].to_numpy(np.float32) if "capacity" in df else None
    for name, p in preds.items():
        r = {"model": name, **regression_metrics(y, p)}
        ev = event_metrics(y, p, target, thr)
        r.update({f"ev{int(thr)}_{k}": v for k, v in ev.items() if k != "threshold"})
        if cap is not None and np.isfinite(cap).any():
            ok = np.isfinite(cap)
            evc = event_metrics(y[ok], p[ok], target, thr, cap[ok])
            r.update({f"evcap_{k}": v for k, v in evc.items() if k != "threshold"})
        rows.append(r)
    return pd.DataFrame(rows).set_index("model")


def metrics_by_hour(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    h = df["hour"].to_numpy().astype(int)
    out = pd.DataFrame({"hour": h, "ae": np.abs(p - y), "y": y})
    return out.groupby("hour").agg(MAE=("ae", "mean"), mean_y=("y", "mean"), n=("ae", "size"))
