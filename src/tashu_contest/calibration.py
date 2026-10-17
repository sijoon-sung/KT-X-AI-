"""Validation-only integer scenario calibration; test labels never choose weights."""
import numpy as np

from .evaluation import stock_evidence


def predict_calibrated(models, x, graph, horizon, samples, steps, seed, opening,
                       calibration, sample_chunk=None, amp=False):
    from .models import predict_scenarios
    gcn = predict_scenarios("gcn", models, x, graph, horizon, samples, steps, seed, sample_chunk, amp)
    uses_flow = any(calibration[g]["flow_weight"] > 0 for g in ("in_stock", "empty"))
    flow = predict_scenarios("gcn_flow", models, x, graph, horizon, samples, steps, seed, sample_chunk, amp) if uses_flow else gcn
    return apply_calibration(gcn, flow, opening, calibration)


def apply_calibration(gcn, flow, opening, calibration):
    """Mixture members, not fractional bicycle counts. Shape [S,B,H,N,2]."""
    factor = np.asarray(calibration["count_factor"], dtype=np.float32)[None, None, :, None, :]
    # Deterministic stratified rounding: reproducible without a shared RNG stream.
    u = np.mod((np.arange(len(flow), dtype=np.float32) + .5) * .61803398875, 1.)
    corrected = np.floor(flow * factor + u[:, None, None, None, None])
    positive = calibration["in_stock"]
    empty = calibration["empty"]
    use_factor = np.where(opening == 0, empty["correct_counts"], positive["correct_counts"])
    corrected = np.where(use_factor[None, :, None, :, None], corrected, flow)
    weight = np.where(opening == 0, empty["flow_weight"], positive["flow_weight"])
    # Missing opening stocks use the positive-stock route for count forecasts only.
    members = (np.arange(len(flow)) + .5) / len(flow)
    choose_flow = members[:, None, None, None, None] < weight[None, :, None, :, None]
    return np.where(choose_flow, corrected, gcn).astype(np.float32)


def group_scores(panel, origins, scenarios, actual, mask, empty):
    opening = panel.stock[origins]
    group = opening == 0 if empty else opening != 0
    observed = mask & group[:, None, :, None]
    count = float(np.abs(scenarios.mean(0) - actual)[observed].mean()) if observed.any() else None
    stock_error, event_errors = [], []
    for i, origin in enumerate(origins):
        item = stock_evidence(panel, origin, scenarios[:, i])
        if item is None:
            continue
        selected = item["current_empty"] if empty else item["in_stock"]
        n = len(selected)
        errors = item["abs_error"].reshape(scenarios.shape[2], n)
        stock_error.extend(errors[:, selected].reshape(-1).tolist())
        probability = item["recovery_probability"] if empty else item["probability"]
        event = item["recovered"] if empty else item["event"]
        event_errors.extend(np.square(probability[selected] - event[selected]).tolist())
    return {"count_mae": count,
            "stock_mae": float(np.mean(stock_error)) if stock_error else None,
            "event_brier": float(np.mean(event_errors)) if event_errors else None}


def fit_calibration(panel, origins, gcn, flow, actual, mask):
    """Choose event Brier subject to count/stock MAE <= 1.01 * GCN on validation.

    Separate routing for initially empty stations targets natural recovery rather
    than classifying every already-empty station as a future stockout.
    """
    denominator = mask.sum(axis=(0, 2))
    truth = np.where(mask, actual, 0).sum(axis=(0, 2)) / np.maximum(denominator, 1)
    prediction = np.where(mask, flow.mean(0), 0).sum(axis=(0, 2)) / np.maximum(denominator, 1)
    factor = np.where(denominator > 0, np.clip(truth / np.maximum(prediction, 1e-6), .5, 2.), 1.)
    calibration = {"schema_version": 1, "fitted_on": "validation_only",
                   "count_factor": factor.tolist(), "mae_tolerance": .01,
                   "objective": "group_event_brier_with_count_and_stock_mae_guardrails",
                   "in_stock": {"flow_weight": 0., "correct_counts": False},
                   "empty": {"flow_weight": 0., "correct_counts": False}}
    opening = panel.stock[origins]
    for group, empty in (("in_stock", False), ("empty", True)):
        baseline = group_scores(panel, origins, gcn, actual, mask, empty)
        best_key, chosen, trials = None, None, []
        for correct in (False, True):
            for weight in (0., .25, .5, .75, 1.):
                if correct and weight == 0:
                    continue
                candidate = dict(calibration)
                candidate[group] = {"flow_weight": weight, "correct_counts": correct}
                values = apply_calibration(gcn, flow, opening, candidate)
                scores = group_scores(panel, origins, values, actual, mask, empty)
                admissible = all(baseline[k] is None or (scores[k] is not None and scores[k] <= baseline[k] * 1.01 + 1e-8)
                                 for k in ("count_mae", "stock_mae"))
                # Without measured stock, tune counts only. With no group labels,
                # weight zero is the deterministic, conservative fallback.
                primary = scores["event_brier"] if scores["event_brier"] is not None else scores["count_mae"]
                key = (primary if primary is not None else float("inf"), weight, correct)
                trials.append({"flow_weight": weight, "correct_counts": correct, "admissible": admissible, **scores})
                if admissible and (best_key is None or key < best_key):
                    best_key, chosen = key, candidate[group]
        calibration[group] = chosen or {"flow_weight": 0., "correct_counts": False}
        calibration[group + "_validation"] = {"gcn_baseline": baseline, "candidates": trials}
    return calibration
