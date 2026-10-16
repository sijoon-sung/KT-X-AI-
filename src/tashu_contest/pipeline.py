"""Train, persist, reload, compare, and make concrete bike-count recommendations."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import build_graph, make_features, read_panel, split_origins, targets
from .evaluation import count_metrics, stock_evidence, stock_metrics
from .models import (CountGCN, GraphFlow, load_models, predict_scenarios, save_models,
                     seed_everything, train_neural, train_xgboost)
from .operations import plan_transfers, station_summary

MODELS = ("xgboost", "gcn", "gcn_flow")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def device_for(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu.")
    return torch.device(name)


def limit_evenly(indices, maximum):
    if maximum < 1:
        raise ValueError("Sampling limits must be positive.")
    return indices if len(indices) <= maximum else indices[np.linspace(0, len(indices) - 1, maximum, dtype=int)]


def fit(args):
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder is not empty. Choose a new run folder to preserve results.")
    if args.history < 1 or args.horizon < 1 or args.epochs < 1 or args.hidden < 4 or args.samples < 2 or args.ode_steps < 1 or args.batch_size < 1 or args.trees < 1 or args.neighbors < 1:
        raise ValueError("Invalid history/horizon/epochs/hidden/samples/steps/batch/trees/neighbors.")
    if args.max_training_rows < 1 or args.max_eval_origins < 1:
        raise ValueError("Sampling limits must be positive.")
    seed_everything(args.seed)
    panel = read_panel(args.panel)
    parts = split_origins(panel, args.history, args.horizon, args.train_end, args.valid_end)
    original_sizes = {k: len(v) for k, v in parts.items()}
    parts["train"] = limit_evenly(parts["train"], max(1, args.max_training_rows // len(panel.stations)))
    for part in ("validation", "test"):
        parts[part] = limit_evenly(parts[part], args.max_eval_origins)
    origins = np.concatenate(list(parts.values()))
    boundaries = np.cumsum([0] + [len(v) for v in parts.values()])
    slices = {k: np.arange(boundaries[i], boundaries[i + 1]) for i, k in enumerate(parts)}
    x = make_features(panel, args.history, origins)
    y, mask = targets(panel, origins, args.horizon)
    graph = build_graph(panel, args.train_end, args.neighbors, args.graph)
    device = device_for(args.device)
    print(f"stations={len(panel.stations)} partitions={ {k: len(v) for k, v in parts.items()} } features={x.shape[-1]}", flush=True)
    gcn = CountGCN(x.shape[-1], args.hidden, args.horizon * 2).to(device)
    gcn_log = train_neural(gcn, x, y, mask, slices["train"], slices["validation"], graph,
                           args.epochs, args.batch_size, args.lr, args.seed)
    flow = GraphFlow(gcn, args.hidden, args.horizon * 2).to(device)
    flow_log = train_neural(flow, x, y, mask, slices["train"], slices["validation"], graph,
                            args.epochs, args.batch_size, args.lr, args.seed, flow=True)
    xgb = train_xgboost(x, y, mask, slices["train"], slices["validation"], args.seed, args.trees)
    out.mkdir(parents=True, exist_ok=True)
    save_models(out / "models", xgb, gcn, flow, {"gcn": gcn_log, "gcn_flow": flow_log})
    np.save(out / "graph.npy", graph)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit, dirty = None, None
    metadata = {"schema_version": 1, "source": args.source, "panel_sha256": file_hash(args.panel),
                "code_commit": commit, "code_dirty": dirty, "seed": args.seed,
                "history": args.history, "horizon": args.horizon, "hidden": args.hidden,
                "feature_count": x.shape[-1], "stations": panel.stations,
                "coords": panel.coords.tolist(), "train_end": str(pd.Timestamp(args.train_end)),
                "valid_end": str(pd.Timestamp(args.valid_end)), "graph": args.graph,
                "neighbors": args.neighbors, "samples": args.samples, "ode_steps": args.ode_steps,
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "trees": args.trees, "device": str(device), "python": platform.python_version(),
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "torch", "xgboost", "scikit-learn")},
                "available_origins": original_sizes, "evaluated_origins": {k: [str(panel.times[t]) for t in v] for k, v in parts.items()},
                "limitations": ["Observed transactions, not identified latent demand.",
                                "Hourly endpoint stockout labels; intra-hour events are unobserved.",
                                "Rentals occur before returns in the simulator.",
                                "GCN/XGBoost probabilities use independent Poisson scenarios.",
                                "Direct transfer batches; no road routing, dispatch timing or causal policy guarantee."]}
    write_json(out / "metadata.json", metadata)
    # Evaluate serialized artifacts, rather than a possibly different last epoch.
    models = load_models(out / "models", metadata, device)
    results = {}
    for name in MODELS:
        results[name] = {}
        for part in ("validation", "test"):
            idx = slices[part]
            count_chunks, actual_chunks, mask_chunks, evidence = [], [], [], []
            for offset in range(0, len(idx), 8):
                selected = idx[offset:offset + 8]
                scenarios = predict_scenarios(name, models, x[selected], graph, args.horizon,
                                             args.samples, args.ode_steps, args.seed + 2000 + offset)
                count_chunks.append(scenarios)
                actual_chunks.append(y[selected].reshape(len(selected), len(panel.stations), args.horizon, 2).transpose(0, 2, 1, 3))
                mask_chunks.append(mask[selected].reshape(len(selected), len(panel.stations), args.horizon, 2).transpose(0, 2, 1, 3))
                for j, at in enumerate(selected):
                    evidence.append(stock_evidence(panel, origins[at], scenarios[:, j]))
            results[name][part] = {"counts": count_metrics(np.concatenate(count_chunks, 1), np.concatenate(actual_chunks), np.concatenate(mask_chunks)),
                                   "stock": stock_metrics(evidence)}
        print(f"{name}: test MAE={results[name]['test']['counts']['mae']:.4f}", flush=True)
    # All three candidates must be comparable on the same selection metric.
    has_stock = all(results[n]["validation"]["stock"].get("brier") is not None for n in MODELS)
    metric = "validation_stock_brier" if has_stock else "validation_count_crps"
    best = min(MODELS, key=lambda n: results[n]["validation"]["stock"]["brier"] if has_stock else results[n]["validation"]["counts"]["crps"])
    report = {"source": args.source, "selection_metric": metric, "selected_model": best,
              "models": results, "limitations": metadata["limitations"]}
    write_json(out / "evaluation.json", report)
    return out


def recommend(args):
    run = Path(args.run)
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    evaluation = json.loads((run / "evaluation.json").read_text(encoding="utf-8"))
    panel = read_panel(args.panel)
    if panel.stations != metadata["stations"] or not np.allclose(panel.coords, metadata["coords"], atol=1e-6, rtol=0):
        raise ValueError("Station IDs/order/coordinates differ from training. Retrain for a new station set.")
    at = len(panel.times) - 1 if args.at is None else panel.times.get_indexer([pd.Timestamp(args.at)])[0]
    if at < metadata["history"]:
        raise ValueError("Forecast timestamp missing or insufficient past history.")
    if args.snapshot:
        snap = pd.read_csv(args.snapshot, dtype={"station_id": str}, encoding="utf-8-sig")
        if not {"station_id", "stock"} <= set(snap) or snap.station_id.duplicated().any() or set(snap.station_id) != set(panel.stations):
            raise ValueError("Snapshot must contain exactly one station_id/stock row for every model station.")
        # Explicit --at is required so a snapshot cannot silently inherit a historical date.
        if args.at is None:
            raise ValueError("Use --at with --snapshot to identify the measured opening-stock time.")
        panel.stock[at] = pd.to_numeric(snap.set_index("station_id").loc[panel.stations, "stock"], errors="raise").to_numpy()
    initial = panel.stock[at]
    if not np.isfinite(initial).all():
        raise ValueError("Recommendations require measured opening stock for every station; blanks are not zero. Supply --snapshot.")
    if args.samples < 2 or args.ode_steps < 1:
        raise ValueError("Require samples >= 2 and ode-steps >= 1.")
    x = make_features(panel, metadata["history"], [at])
    graph = np.load(run / "graph.npy", allow_pickle=False)
    seed_everything(metadata["seed"])
    models = load_models(run / "models", metadata, device_for(args.device))
    name = evaluation["selected_model"] if args.model == "auto" else args.model
    scenarios = predict_scenarios(name, models, x, graph, metadata["horizon"], args.samples,
                                 args.ode_steps, metadata["seed"] + 3000)[:, 0]
    plan = plan_transfers(initial, scenarios, panel.coords, panel.capacity, args.budget,
                          args.truck_capacity, args.service_level, args.max_distance_km)
    transfers = [{**r, "from_station": panel.stations[r["from_index"]], "to_station": panel.stations[r["to_index"]]} for r in plan["transfers"]]
    result = {"source": metadata["source"], "timestamp": str(panel.times[at]), "model": name,
              "horizon_hours": metadata["horizon"], "samples": args.samples, "ode_steps": args.ode_steps,
              "forecast_only_no_dispatch": True, "budget_bikes": args.budget, "truck_capacity": args.truck_capacity,
              "service_level": args.service_level, "max_distance_km": args.max_distance_km,
              "snapshot_file": str(args.snapshot) if args.snapshot else None,
              "initial_fleet_bikes": int(initial.sum()), "final_fleet_bikes": int(plan["after_transfer"].sum()),
              **{k: plan[k] for k in ("moved_bikes", "requested_topup_bikes", "unfilled_topup_bikes", "service_failures_before", "service_failures_after", "expected_failures_reduction")},
              "transfers": transfers, "stations": station_summary(panel, initial, scenarios, plan),
              "limitations": metadata["limitations"]}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "recommendation.json", result)
    pd.DataFrame(result["stations"]).drop(columns=["forecast_stock_p10", "forecast_stock_p50", "forecast_stock_p90", "forecast_rentals_mean", "forecast_returns_mean"]).to_csv(out / "stations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(transfers, columns=["batch", "from_station", "to_station", "bikes", "direct_distance_km", "from_index", "to_index"]).to_csv(out / "transfers.csv", index=False, encoding="utf-8-sig")
    from .report import render_report
    render_report(out / "report.html", evaluation, result)
    print(f"model={name}; moved={plan['moved_bikes']} bikes; report={out / 'report.html'}", flush=True)
    return out
