"""Identical masked count and measured-stock evaluation for all models."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .operations import simulate


def count_metrics(samples, observed, mask):
    # [S,B,H,N,2], actual and mask [B,H,N,2]
    pred = samples.mean(0)
    error = (pred - observed)[mask]
    actual = observed[mask]
    if not len(error):
        raise ValueError("No observed evaluation cells.")
    values = samples[:, mask]
    # Empirical CRPS computed in O(S log S), without S*S*all_cells memory.
    ordered = np.sort(values, axis=0)
    s = len(samples)
    weight = (2 * np.arange(1, s + 1) - s - 1)[:, None]
    crps = np.abs(values - actual).mean(0) - (ordered * weight).sum(0) / (s * s)
    lo, hi = np.quantile(values, [.05, .95], axis=0)
    peaks = actual >= 3
    zeros = actual == 0
    return {"cells": int(mask.sum()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "wape": float(np.abs(error).sum() / actual.sum()) if actual.sum() else None,
            "crps": float(crps.mean()), "coverage_90": float(((actual >= lo) & (actual <= hi)).mean()),
            "interval_width_90": float((hi - lo).mean()),
            "peak_mae": float(np.abs(error[peaks]).mean()) if peaks.any() else None,
            "zero_mae": float(np.abs(error[zeros]).mean()) if zeros.any() else None}


def stock_evidence(panel, origin, samples):
    """Evaluate only complete measured stock trajectories, never pseudo-labels."""
    horizon = samples.shape[1]
    if origin + horizon >= len(panel.times):
        return None
    initial = panel.stock[origin]
    actual = panel.stock[origin + 1:origin + horizon + 1]
    observed = np.isfinite(initial) & np.isfinite(actual).all(0)
    # A partially observed city is valid; only independent known station columns are simulated.
    idx = np.flatnonzero(observed)
    if not len(idx):
        return None
    cap = panel.capacity[idx]
    path = simulate(initial[idx], samples[:, :, idx], cap)["stock"]
    probability = (path == 0).any(1).mean(0)
    event = (actual[:, idx] == 0).any(0)
    return {"probability": probability, "event": event, "in_stock": initial[idx] > 0,
            "abs_error": np.abs(path.mean(0) - actual[:, idx]).reshape(-1),
            "current_empty": initial[idx] == 0,
            "recovery_probability": (path > 0).any(1).mean(0),
            "recovered": (actual[:, idx] > 0).any(0)}


def stock_metrics(evidence):
    evidence = [r for r in evidence if r is not None]
    if not evidence:
        return {"status": "not_evaluated_no_measured_stock", "cells": 0}
    flat = {k: np.concatenate([r[k] for r in evidence]) for k in evidence[0]}
    mask = flat["in_stock"]
    p, y = flat["probability"][mask], flat["event"][mask]
    empty = flat["current_empty"]
    result = {"status": "evaluated_hourly_snapshots", "cells": int(mask.sum()),
              "net_inventory_mae": float(flat["abs_error"].mean()),
              "brier": float(np.square(p - y).mean()) if len(y) else None,
              "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
              "average_precision": float(average_precision_score(y, p)) if len(y) and y.any() else None,
              "event_rate": float(y.mean()) if len(y) else None,
              "currently_empty_cells": int(empty.sum()),
              "recovery_brier": float(np.square(flat["recovery_probability"][empty] - flat["recovered"][empty]).mean())
              if empty.any() else None}
    if len(y):
        top = np.argsort(-p, kind="stable")[:max(1, int(np.ceil(.1 * len(y))))]
        result["top10pct_precision"] = float(y[top].mean())
    return result
